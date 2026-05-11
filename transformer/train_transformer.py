"""
Training script cho Transformer Seq2Seq Model (EN → VI)
Features:
  - Noam LR Scheduler (warmup theo paper gốc)
  - Label Smoothing Loss
  - Mixed Precision (FP16) nếu có GPU
  - Early Stopping
  - TensorBoard logging
  - Sử dụng BPE Vocab đã train sẵn
"""

import sys, os
# Thêm thư mục gốc vào path để import baseline utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import math
from datetime import datetime

from transformer.config_transformer import TRANSFORMER_CONFIG
from transformer.transformer_model import build_transformer, count_parameters
from transformer.shared_bpe_vocab import SharedBPEVocabulary, build_shared_bpe, load_shared_bpe
from baseline.data_utils import create_dataloaders


# ============================================================
# Noam Learning Rate Scheduler
# Theo "Attention Is All You Need" - lrate = d^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
# ============================================================

class NoamScheduler:
    """
    Warmup + inverse sqrt decay scheduler theo Vaswani et al. 2017
    """
    def __init__(self, optimizer, d_model: int, warmup_steps: int):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step = 0

    def step(self):
        self._step += 1
        lr = self._compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def _compute_lr(self):
        step = self._step
        return (self.d_model ** -0.5) * min(
            step ** -0.5,
            step * (self.warmup_steps ** -1.5)
        )

    def get_last_lr(self):
        return [self._compute_lr()]


# ============================================================
# Label Smoothing Loss
# ============================================================

class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Cross Entropy Loss
    Thay one-hot target bằng distribution mềm hơn để tránh overconfident
    """
    def __init__(self, vocab_size: int, padding_idx: int = 0, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.criterion = nn.KLDivLoss(reduction='sum')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   [N, vocab_size] - raw logits
            target: [N]             - target token indices
        Returns:
            loss: scalar
        """
        log_probs = F.log_softmax(pred, dim=-1)

        with torch.no_grad():
            smooth_target = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
            smooth_target.scatter_(1, target.unsqueeze(1), self.confidence)
            smooth_target[:, self.padding_idx] = 0.0

            # Mask padding positions
            pad_mask = (target == self.padding_idx)
            smooth_target[pad_mask] = 0.0

        loss = self.criterion(log_probs, smooth_target)

        num_tokens = (~pad_mask).sum().item()
        if num_tokens > 0:
            loss = loss / num_tokens

        return loss


# ============================================================
# Training helpers
# ============================================================

def train_epoch(model, dataloader, optimizer, criterion, scheduler, clip, device, scaler=None):
    """Train một epoch"""
    model.train()
    epoch_loss = 0
    use_amp = scaler is not None

    progress_bar = tqdm(dataloader, desc="  Training", leave=False)

    for batch in progress_bar:
        src = batch['src'].to(device)    # [batch, src_len]
        trg = batch['trg'].to(device)    # [batch, trg_len]

        optimizer.zero_grad()

        # Teacher forcing: dùng <sos>...<eos-1> làm input, <sos+1>...<eos> làm target
        tgt_input = trg[:, :-1]    # [batch, trg_len-1]
        tgt_output = trg[:, 1:]    # [batch, trg_len-1]

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(src, tgt_input)  # [batch, trg_len-1, vocab]

            # Flatten cho loss
            batch_size, trg_len, vocab_size = logits.shape
            logits_flat = logits.contiguous().view(-1, vocab_size)   # [batch*trg_len, vocab]
            tgt_flat = tgt_output.contiguous().view(-1)              # [batch*trg_len]

            loss = criterion(logits_flat, tgt_flat)

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

        # Cập nhật Noam scheduler mỗi bước
        current_lr = scheduler.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.2e}'})

    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate trên validation set"""
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Evaluating", leave=False):
            src = batch['src'].to(device)
            trg = batch['trg'].to(device)

            tgt_input = trg[:, :-1]
            tgt_output = trg[:, 1:]

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(src, tgt_input)

                logits_flat = logits.contiguous().view(-1, logits.size(-1))
                tgt_flat = tgt_output.contiguous().view(-1)

                loss = criterion(logits_flat, tgt_flat)

            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def epoch_time(start, end):
    elapsed = end - start
    return int(elapsed / 60), int(elapsed % 60)


# ============================================================
# Main Training Function
# ============================================================

def main():
    config = TRANSFORMER_CONFIG
    device = torch.device(config["device"])

    print("=" * 60)
    print("  TRANSFORMER MT TRAINING (EN → VI)")
    print("=" * 60)
    print(f"  Device     : {device}")
    print(f"  d_model    : {config['d_model']}")
    print(f"  num_heads  : {config['num_heads']}")
    print(f"  enc_layers : {config['num_encoder_layers']}")
    print(f"  dec_layers : {config['num_decoder_layers']}")
    print(f"  d_ff       : {config['d_ff']}")
    print(f"  dropout    : {config['dropout']}")
    print(f"  warmup     : {config['warmup_steps']}")
    print(f"  batch_size : {config['batch_size']}")
    print(f"  epochs     : {config['epochs']}")
    print("=" * 60)

    # ─── Paths ────────────────────────────────────────────
    data_dir = config["data_dir"]
    vocab_dir = config["vocab_dir"]
    checkpoint_dir = config["checkpoint_dir"]

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(vocab_dir, exist_ok=True)

    # ─── Load SHARED BPE vocabulary (Paper Section 5.1) ────
    # Paper: "byte-pair encoding, which has a SHARED source-target
    # vocabulary of about 37000 tokens"
    shared_bpe_model = os.path.join(vocab_dir, "shared_bpe.model")

    if os.path.exists(shared_bpe_model):
        print("\n[1/4] Loading existing Shared BPE vocabulary...")
        shared_vocab = load_shared_bpe(vocab_dir)
    else:
        print("\n[1/4] Training Shared BPE vocabulary (lần đầu tiên)...")
        shared_vocab = build_shared_bpe(
            data_dir, vocab_size=config["bpe_vocab_size"], save_dir=vocab_dir
        )

    # Paper: shared vocab → dùng cùng 1 vocab cho cả src và trg
    src_vocab = shared_vocab
    trg_vocab = shared_vocab

    print(f"  Shared vocab size: {len(shared_vocab):,} tokens")

    # ─── DataLoaders ──────────────────────────────────────
    print("\n[2/4] Creating DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir, src_vocab, trg_vocab, config
    )

    # ─── Build Model ──────────────────────────────────────
    # Paper: shared vocab → src_vocab_size == trg_vocab_size
    print("\n[3/4] Building Transformer model...")
    vocab_size = len(shared_vocab)
    model = build_transformer(vocab_size, vocab_size, config)
    n_params = count_parameters(model)
    print(f"  Số tham số: {n_params:,} ({n_params/1e6:.2f}M)")

    # ─── Optimizer + Scheduler ────────────────────────────
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0,  # LR sẽ được Noam scheduler kiểm soát
        betas=config["adam_betas"],
        eps=config["adam_eps"],
    )

    scheduler = NoamScheduler(
        optimizer,
        d_model=config["d_model"],
        warmup_steps=config["warmup_steps"],
    )

    # ─── Loss Function ────────────────────────────────────
    smoothing = config.get("label_smoothing", 0.1)
    criterion = LabelSmoothingLoss(
        vocab_size=len(trg_vocab),
        padding_idx=config["pad_idx"],
        smoothing=smoothing,
    )
    print(f"  Loss: Label Smoothing (ε={smoothing})")

    # ─── FP16 Mixed Precision ─────────────────────────────
    scaler = None
    if torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        print("  FP16 Mixed Precision: ENABLED ✓")
    else:
        print("  FP16 Mixed Precision: DISABLED (CPU)")

    # ─── TensorBoard ──────────────────────────────────────
    log_dir = f"./runs/transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)
    print(f"  TensorBoard: {log_dir}")

    # ─── Training Loop ────────────────────────────────────
    print("\n[4/4] Starting training...")
    print("=" * 60 + "\n")

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(config["epochs"]):
        start_time = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            scheduler, config["grad_clip"], device, scaler
        )

        # Validate
        val_loss = evaluate(model, val_loader, criterion, device)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        current_lr = scheduler.get_last_lr()[0]

        # TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('PPL/train', math.exp(min(train_loss, 100)), epoch)
        writer.add_scalar('PPL/val', math.exp(min(val_loss, 100)), epoch)
        writer.add_scalar('LR', current_lr, epoch)

        # Print
        print(f"Epoch {epoch+1:02}/{config['epochs']} | {epoch_mins}m {epoch_secs}s | LR: {current_lr:.2e}")
        print(f"  Train Loss: {train_loss:.4f} | Train PPL: {math.exp(min(train_loss, 100)):8.3f}")
        print(f"  Val   Loss: {val_loss:.4f}   | Val   PPL: {math.exp(min(val_loss, 100)):8.3f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_step': scheduler._step,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
                'src_vocab_size': len(src_vocab),
                'trg_vocab_size': len(trg_vocab),
            }, os.path.join(checkpoint_dir, 'best_model.pt'))
            print("  ★ New best model saved!")
        else:
            patience_counter += 1
            print(f"  ⚠ No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\n  Early Stopping after {epoch+1} epochs.")
                break

        # Save periodic checkpoint
        if (epoch + 1) % config["save_every"] == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))

        print()

    writer.close()

    # ─── Final test evaluation ─────────────────────────────
    print("=" * 60)
    print("Final Evaluation on Test Set")

    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"  Test Loss: {test_loss:.4f} | Test PPL: {math.exp(min(test_loss, 100)):8.3f}")
    print("=" * 60)
    print("\n✓ Training hoàn tất!")
    print(f"  Best model: {os.path.join(checkpoint_dir, 'best_model.pt')}")
    print(f"  TensorBoard: tensorboard --logdir={log_dir}")


if __name__ == "__main__":
    main()
