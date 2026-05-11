# Training loop, checkpoint saving
"""
Training script cho Baseline Seq2Seq + Attention model
Cải tiến: Coverage Loss + Label Smoothing + BPE support
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
import math
from tqdm import tqdm
from datetime import datetime

from config_baseline import BASELINE_CONFIG
from seq2seq_model import build_model, count_parameters
from data_utils import (
    build_vocabularies, build_bpe_vocabularies,
    create_dataloaders, Vocabulary,
    load_bpe_vocabularies
)


# ============================================================
# Label Smoothing Loss
# ============================================================

class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss
    Thay vì target = [0, 0, 1, 0, 0] (one-hot)
    Dùng target = [0.025, 0.025, 0.9, 0.025, 0.025] (smoothed)
    
    Giúp model không quá tự tin → generalize tốt hơn
    """
    def __init__(self, vocab_size, padding_idx=0, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
        # KL Divergence loss
        self.criterion = nn.KLDivLoss(reduction='sum')
    
    def forward(self, pred, target):
        """
        Args:
            pred: [batch*seq_len, vocab_size] - log probabilities (raw logits)
            target: [batch*seq_len] - target indices
        Returns:
            loss: scalar
        """
        # Convert logits to log-probabilities
        log_probs = F.log_softmax(pred, dim=1)
        
        # Tạo smoothed distribution
        with torch.no_grad():
            smooth_target = torch.zeros_like(log_probs)
            smooth_target.fill_(self.smoothing / (self.vocab_size - 2))  # -2 cho pad và target
            smooth_target.scatter_(1, target.unsqueeze(1), self.confidence)
            smooth_target[:, self.padding_idx] = 0  # Không smooth vào padding
            
            # Mask padding positions
            mask = target != self.padding_idx
            smooth_target = smooth_target * mask.unsqueeze(1)
        
        loss = self.criterion(log_probs, smooth_target)
        
        # Normalize bằng số token thực (không padding)
        num_tokens = mask.sum().item()
        if num_tokens > 0:
            loss = loss / num_tokens
        
        return loss


import torch.nn.functional as F


def train_epoch(model, dataloader, optimizer, criterion, clip, device, 
                teacher_forcing_ratio, scaler=None, coverage_lambda=0.0):
    """
    Train một epoch (hỗ trợ FP16 Mixed Precision + Coverage Loss)
    """
    model.train()
    epoch_loss = 0
    epoch_cov_loss = 0
    use_amp = scaler is not None
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch in progress_bar:
        src = batch['src'].to(device)
        trg = batch['trg'].to(device)
        src_lengths = batch['src_lengths'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass với FP16 autocast
        with torch.cuda.amp.autocast(enabled=use_amp):
            output, coverage_loss = model(src, src_lengths, trg, teacher_forcing_ratio)
            
            # Reshape for loss calculation (skip first token - SOS)
            output = output[:, 1:, :].contiguous().view(-1, output.size(-1))
            trg_flat = trg[:, 1:].contiguous().view(-1)
            
            # NLL loss + coverage loss
            nll_loss = criterion(output, trg_flat)
            loss = nll_loss + coverage_lambda * coverage_loss
        
        # Backward pass với GradScaler
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        
        epoch_loss += nll_loss.item()
        epoch_cov_loss += coverage_loss.item()
        progress_bar.set_postfix({
            'loss': f'{nll_loss.item():.4f}',
            'cov': f'{coverage_loss.item():.4f}'
        })
    
    avg_loss = epoch_loss / len(dataloader)
    avg_cov = epoch_cov_loss / len(dataloader)
    return avg_loss, avg_cov


def evaluate(model, dataloader, criterion, device, coverage_lambda=0.0):
    """
    Evaluate model trên validation set
    """
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            src = batch['src'].to(device)
            trg = batch['trg'].to(device)
            src_lengths = batch['src_lengths'].to(device)
            
            # Forward pass với autocast (tự detect GPU)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                output, coverage_loss = model(src, src_lengths, trg, teacher_forcing_ratio=0)
                
                output = output[:, 1:, :].contiguous().view(-1, output.size(-1))
                trg_flat = trg[:, 1:].contiguous().view(-1)
                
                nll_loss = criterion(output, trg_flat)
                loss = nll_loss + coverage_lambda * coverage_loss
            
            epoch_loss += nll_loss.item()
    
    return epoch_loss / len(dataloader)


def epoch_time(start_time, end_time):
    """Calculate elapsed time"""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def main():
    """Main training function"""
    config = BASELINE_CONFIG
    device = torch.device(config["device"])
    print(f"Using device: {device}")
    
    use_bpe = config.get("use_bpe", False)
    use_coverage = config.get("use_coverage", False)
    coverage_lambda = config.get("coverage_lambda", 0.0)
    label_smoothing = config.get("label_smoothing", 0.0)
    
    print(f"BPE: {'ON' if use_bpe else 'OFF'}")
    print(f"Coverage: {'ON' if use_coverage else 'OFF'} (λ={coverage_lambda})")
    print(f"Label Smoothing: {label_smoothing}")
    
    # Paths
    data_dir = "../split_dataset"
    vocab_dir = "./vocab"
    checkpoint_dir = config["checkpoint_dir"]
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(vocab_dir, exist_ok=True)
    
    # ============================================================
    # Build or load vocabularies
    # ============================================================
    if use_bpe:
        # BPE Vocabulary
        bpe_model_path = os.path.join(vocab_dir, "bpe_en.json")
        
        if os.path.exists(bpe_model_path):
            print("Loading existing BPE vocabularies...")
            src_vocab, trg_vocab = load_bpe_vocabularies(vocab_dir)
        else:
            print("Training BPE vocabularies...")
            bpe_vocab_size = config.get("bpe_vocab_size", 16000)
            src_vocab, trg_vocab = build_bpe_vocabularies(
                data_dir, vocab_size=bpe_vocab_size, save_dir=vocab_dir
            )
    else:
        # Word-level Vocabulary
        src_vocab_path = os.path.join(vocab_dir, "src_vocab.pkl")
        trg_vocab_path = os.path.join(vocab_dir, "trg_vocab.pkl")
        
        if os.path.exists(src_vocab_path) and os.path.exists(trg_vocab_path):
            print("Loading existing vocabularies...")
            src_vocab = Vocabulary.load(src_vocab_path)
            trg_vocab = Vocabulary.load(trg_vocab_path)
        else:
            print("Building vocabularies...")
            src_vocab, trg_vocab = build_vocabularies(data_dir, min_freq=2, save_dir=vocab_dir)
    
    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(trg_vocab)}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir, src_vocab, trg_vocab, config
    )
    
    # Build model
    print("\nBuilding model...")
    model = build_model(len(src_vocab), len(trg_vocab), config)
    print(f"Model has {count_parameters(model):,} trainable parameters")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)
    
    # LR Scheduler: giảm LR × 0.5 nếu val_loss không giảm sau 2 epoch
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # ============================================================
    # Loss function: Label Smoothing hoặc CrossEntropy
    # ============================================================
    if label_smoothing > 0:
        criterion = LabelSmoothingLoss(
            vocab_size=len(trg_vocab),
            padding_idx=src_vocab.pad_idx,
            smoothing=label_smoothing
        )
        print(f"✓ Label Smoothing Loss (smoothing={label_smoothing})")
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=src_vocab.pad_idx)
        print("✓ CrossEntropy Loss")
    
    # TensorBoard writer
    log_dir = f"./runs/baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs: {log_dir}")
    
    # FP16 Mixed Precision
    scaler = None
    if torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        print("✓ FP16 Mixed Precision: ENABLED")
    else:
        print("⚠ FP16 Mixed Precision: DISABLED (CPU mode)")
    
    # Training loop
    best_valid_loss = float('inf')
    patience = 5           # Early Stopping: dừng sau N epoch không cải thiện
    patience_counter = 0
    
    print("\n" + "="*50)
    print("Starting training...")
    print(f"Early Stopping: patience={patience}")
    print("="*50 + "\n")
    
    for epoch in range(config["epochs"]):
        start_time = time.time()
        
        # Train
        train_loss, train_cov = train_epoch(
            model, train_loader, optimizer, criterion,
            config["grad_clip"], device, config["teacher_forcing_ratio"],
            scaler=scaler, coverage_lambda=coverage_lambda
        )
        
        # Evaluate
        valid_loss = evaluate(model, val_loader, criterion, device, coverage_lambda)
        
        # Cập nhật LR Scheduler
        scheduler.step(valid_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0  # Reset counter
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'src_vocab_size': len(src_vocab),
                'trg_vocab_size': len(trg_vocab),
                'config': config,
                'use_bpe': use_bpe,
            }, os.path.join(checkpoint_dir, 'best_model.pt'))
            print(f"  ★ New best model saved!")
        else:
            patience_counter += 1
            print(f"  ⚠ No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\n Early Stopping! No improvement for {patience} epochs.")
                break
        
        # Save checkpoint every N epochs
        if (epoch + 1) % config["save_every"] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
            }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        writer.add_scalar('PPL/train', math.exp(min(train_loss, 100)), epoch)
        writer.add_scalar('PPL/valid', math.exp(min(valid_loss, 100)), epoch)
        if use_coverage:
            writer.add_scalar('Loss/coverage', train_cov, epoch)
        
        # Print progress
        print(f"Epoch: {epoch+1:02}/{config['epochs']} | Time: {epoch_mins}m {epoch_secs}s | LR: {current_lr:.2e}")
        print(f"  Train Loss: {train_loss:.4f} | Train PPL: {math.exp(min(train_loss, 100)):7.3f}")
        if use_coverage:
            print(f"  Coverage Loss: {train_cov:.4f}")
        print(f"  Valid Loss: {valid_loss:.4f} | Valid PPL: {math.exp(min(valid_loss, 100)):7.3f}")
        print()
    
    writer.close()
    
    # Final evaluation on test set
    print("="*50)
    print("Evaluating on test set...")
    
    # Load best model
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss = evaluate(model, test_loader, criterion, device, coverage_lambda)
    print(f"Test Loss: {test_loss:.4f} | Test PPL: {math.exp(min(test_loss, 100)):7.3f}")
    print("="*50)
    
    print("\n Training completed!")
    print(f"Best model saved to: {os.path.join(checkpoint_dir, 'best_model.pt')}")
    print(f"TensorBoard logs: tensorboard --logdir={log_dir}")


if __name__ == "__main__":
    main()
