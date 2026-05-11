# Transformer cho Machine Translation EN -> VI
# Training Notebook (Kaggle-ready)
# Author: Khiem
# Cach chay: Copy noi dung file nay vao cac cells tren Kaggle notebook

# ── Cell 1: Setup & Install ─────────────────────────────────
import subprocess, sys

packages = ["sentencepiece", "sacrebleu", "nltk", "datasets"]
for pkg in packages:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

print("Packages installed")

# ── Cell 2: Imports ─────────────────────────────────────────
import os, sys, torch, math, time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Cell 3: Copy code tu Kaggle Dataset vao working dir ─────
# *** Upload 1 Kaggle Dataset chua: transformer/ va baseline/ ***
# Dataset baseline da co san: /kaggle/input/datasets/khimvng/baseline20/baseline
#
# Ban can tao 1 Kaggle Dataset moi chua thu muc transformer/
# Roi add ca 2 dataset vao notebook

import shutil

WORK_DIR = "/kaggle/working"
os.chdir(WORK_DIR)
sys.path.insert(0, WORK_DIR)

KAGGLE_INPUT = "/kaggle/input"

# === Tu dong tim baseline/ va transformer/ trong tat ca datasets ===
BASELINE_SRC = None
TRANSFORMER_SRC = None

# Duyet tat ca datasets da add vao notebook
def find_in_kaggle_input(target_name):
    """Tim thu muc target_name trong /kaggle/input (nhieu cap)"""
    for root, dirs, files in os.walk(KAGGLE_INPUT):
        if target_name in dirs:
            found = os.path.join(root, target_name)
            # Kiem tra co file .py ben trong khong (tranh thu muc rong)
            py_files = [f for f in os.listdir(found) if f.endswith('.py')]
            if py_files:
                return found
    return None

BASELINE_SRC = find_in_kaggle_input("baseline")
TRANSFORMER_SRC = find_in_kaggle_input("transformer")

print(f"Baseline found:    {BASELINE_SRC}")
print(f"Transformer found: {TRANSFORMER_SRC}")

# Copy baseline/ vao working dir
dst_baseline = os.path.join(WORK_DIR, "baseline")
if not os.path.exists(dst_baseline):
    if BASELINE_SRC:
        shutil.copytree(BASELINE_SRC, dst_baseline)
        print("Copied baseline/ to working dir")
    else:
        print("WARNING: baseline/ not found in any Kaggle dataset!")
        print("  Hay kiem tra datasets da add vao notebook")

# Copy transformer/ vao working dir
dst_transformer = os.path.join(WORK_DIR, "transformer")
if not os.path.exists(dst_transformer):
    if TRANSFORMER_SRC:
        shutil.copytree(TRANSFORMER_SRC, dst_transformer)
        print("Copied transformer/ to working dir")
    else:
        print("WARNING: transformer/ not found in any Kaggle dataset!")
        print("  Hay kiem tra datasets da add vao notebook")

# Tao __init__.py cho cac package (neu chua co)
for pkg_dir in [dst_baseline, dst_transformer]:
    if os.path.isdir(pkg_dir):
        init_file = os.path.join(pkg_dir, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write("")
            print(f"  Created {init_file}")

# === Copy checkpoint va vocab tu session truoc (neu co) ===
# Khi ban "Save & Run All" tren Kaggle, output duoc luu vao /kaggle/working/
# Tao 1 Kaggle Dataset tu output do, roi add vao session moi de resume

def find_dir_in_kaggle_input(target_name):
    """Tim thu muc target_name trong /kaggle/input (nhieu cap)"""
    for root, dirs, files in os.walk(KAGGLE_INPUT):
        if target_name in dirs:
            found = os.path.join(root, target_name)
            # Kiem tra co file ben trong khong (tranh thu muc rong)
            if os.listdir(found):
                return found
    return None

# Copy transformer_checkpoints/ (chua best_model.pt va last_checkpoint.pt)
dst_ckpt = os.path.join(WORK_DIR, "transformer_checkpoints")
if not os.path.exists(dst_ckpt):
    ckpt_src = find_dir_in_kaggle_input("transformer_checkpoints")
    if ckpt_src:
        shutil.copytree(ckpt_src, dst_ckpt)
        ckpt_files = os.listdir(dst_ckpt)
        print(f"Copied transformer_checkpoints/ from previous session: {ckpt_files}")
    else:
        print("No previous checkpoints found (fresh training)")

# Copy vocab/ (chua shared_bpe.model va shared_bpe_meta.pkl)
dst_vocab = os.path.join(WORK_DIR, "vocab")
if not os.path.exists(dst_vocab):
    vocab_src = find_dir_in_kaggle_input("vocab")
    if vocab_src:
        shutil.copytree(vocab_src, dst_vocab)
        vocab_files = os.listdir(dst_vocab)
        print(f"Copied vocab/ from previous session: {vocab_files}")
    else:
        print("No previous vocab found (will train new BPE)")

print(f"\nWorking dir contents: {os.listdir(WORK_DIR)}")

# ── Cell 4: Import modules ──────────────────────────────────
from transformer.config_transformer import TRANSFORMER_CONFIG
from transformer.transformer_model import build_transformer, count_parameters
from transformer.shared_bpe_vocab import SharedBPEVocabulary, build_shared_bpe, load_shared_bpe
from baseline.data_utils import BPEVocabulary, TranslationDataset, collate_fn

# tqdm: notebook version
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

print("All modules imported OK")

# ── Cell 5: Load dataset tu HuggingFace ─────────────────────
# Giong nhu notebook cu: load truc tiep tu HuggingFace, lay 500K train
from datasets import load_dataset

full_dataset = load_dataset("ncduy/mt-en-vi")
dataset = {
    "train": full_dataset["train"].select(range(2000000)),
    "validation": full_dataset["validation"],
    "test": full_dataset["test"]
}
print(f"\nDataset info:")
print(f"  Train:      {len(dataset['train']):,} samples")
print(f"  Validation: {len(dataset['validation']):,} samples")
print(f"  Test:       {len(dataset['test']):,} samples")

# ── Cell 6: Config ──────────────────────────────────────────
config = TRANSFORMER_CONFIG.copy()
config["device"] = device
config["checkpoint_dir"] = "/kaggle/working/transformer_checkpoints"
config["vocab_dir"] = "/kaggle/working/vocab"

os.makedirs(config["checkpoint_dir"], exist_ok=True)
os.makedirs(config["vocab_dir"], exist_ok=True)

print(f"\nConfig:")
print(f"  d_model    = {config['d_model']}")
print(f"  num_heads  = {config['num_heads']}")
print(f"  enc_layers = {config['num_encoder_layers']}")
print(f"  dec_layers = {config['num_decoder_layers']}")
print(f"  d_ff       = {config['d_ff']}")
print(f"  dropout    = {config['dropout']}")
print(f"  batch_size = {config['batch_size']}")
print(f"  epochs     = {config['epochs']}")

# ── Cell 7: Build Shared BPE Vocabulary ─────────────────────
VOCAB_DIR = config["vocab_dir"]
shared_bpe_model = os.path.join(VOCAB_DIR, "shared_bpe.model")

if os.path.exists(shared_bpe_model):
    print("\nLoading existing Shared BPE vocabulary...")
    shared_vocab = load_shared_bpe(VOCAB_DIR)
else:
    print("\nTraining Shared BPE vocabulary (lan dau tien)...")
    # Tron EN + VI tu dataset
    sentences = []
    for item in tqdm(dataset["train"], desc="Collecting sentences"):
        sentences.append(item["en"].lower())
        sentences.append(item["vi"])
    print(f"Total: {len(sentences):,} sentences (EN + VI)")

    shared_vocab = SharedBPEVocabulary(vocab_size=config["bpe_vocab_size"])
    model_prefix = os.path.join(VOCAB_DIR, "shared_bpe")
    shared_vocab.train(sentences, model_prefix)
    shared_vocab.save_metadata(os.path.join(VOCAB_DIR, "shared_bpe_meta.pkl"))

src_vocab = shared_vocab
trg_vocab = shared_vocab
print(f"Shared vocab size: {len(shared_vocab):,} tokens")

# ── Cell 8: Create DataLoaders ──────────────────────────────
# Tao Dataset va DataLoader truc tiep (giong notebook cu)
train_dataset = TranslationDataset(dataset["train"], src_vocab, trg_vocab, config["max_length"])
val_dataset = TranslationDataset(dataset["validation"], src_vocab, trg_vocab, config["max_length"])
test_dataset = TranslationDataset(dataset["test"], src_vocab, trg_vocab, config["max_length"])

train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, src_vocab.pad_idx),
    num_workers=0,
    pin_memory=True if device.type == 'cuda' else False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, src_vocab.pad_idx),
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, src_vocab.pad_idx),
    num_workers=0
)

print(f"\nDataLoaders created!")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val   batches: {len(val_loader)}")
print(f"  Test  batches: {len(test_loader)}")

# ── Cell 9: Build Model ────────────────────────────────────
model = build_transformer(len(shared_vocab), len(shared_vocab), config)
n_params = count_parameters(model)
print(f"\nModel: Transformer")
print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")

# ── Cell 10: Training Setup ─────────────────────────────────

# Noam Scheduler (Paper Section 5.3)
class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step = 0

    def step(self):
        self._step += 1
        lr = (self.d_model ** -0.5) * min(
            self._step ** -0.5,
            self._step * (self.warmup_steps ** -1.5)
        )
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

# Label Smoothing Loss (Paper Section 5.4)
class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, padding_idx=0, smoothing=0.1):
        super().__init__()
        self.vocab_size  = vocab_size
        self.padding_idx = padding_idx
        self.smoothing   = smoothing
        self.confidence  = 1.0 - smoothing
        self.criterion   = nn.KLDivLoss(reduction='sum')

    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            smooth_target = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
            smooth_target.scatter_(1, target.unsqueeze(1), self.confidence)
            smooth_target[:, self.padding_idx] = 0.0
            pad_mask = (target == self.padding_idx)
            smooth_target[pad_mask] = 0.0
        loss = self.criterion(log_probs, smooth_target)
        n = (~pad_mask).sum().item()
        return loss / n if n > 0 else loss

optimizer = optim.Adam(
    model.parameters(), lr=0.0,
    betas=config["adam_betas"], eps=config["adam_eps"]
)
scheduler = NoamScheduler(optimizer, config["d_model"], config["warmup_steps"])
criterion = LabelSmoothingLoss(len(trg_vocab), config["pad_idx"], config["label_smoothing"])
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

print("Optimizer: Adam (Noam LR schedule)")
print(f"Label Smoothing: {config['label_smoothing']}")
print(f"FP16: {'ENABLED' if scaler else 'DISABLED'}")

# ── Cell 11: Resume from checkpoint ─────────────────────────
CHECKPOINT_DIR = config["checkpoint_dir"]
resume_path = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pt")
best_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")

start_epoch = 0
best_val_loss = float('inf')
history = {'train_loss': [], 'val_loss': [], 'train_ppl': [], 'val_ppl': []}

if os.path.exists(resume_path):
    print(f"\nResuming from {resume_path}...")
    ckpt = torch.load(resume_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler._step = ckpt.get('scheduler_step', 0)
    start_epoch = ckpt['epoch']
    best_val_loss = ckpt.get('best_val_loss', float('inf'))
    history = ckpt.get('history', {'train_loss': [], 'val_loss': [], 'train_ppl': [], 'val_ppl': []})
    print(f"  Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
    # In lại history
    for i, (tl, vl) in enumerate(zip(history['train_loss'], history['val_loss'])):
        print(f"    Epoch {i+1}: Train Loss={tl:.4f} PPL={math.exp(min(tl,100)):.2f} | Val Loss={vl:.4f} PPL={math.exp(min(vl,100)):.2f}")
else:
    print("\nStarting fresh training...")

# ── Cell 12: Training Loop ──────────────────────────────────
patience = 7
patience_counter = 0
accumulate_grad = config.get("accumulate_grad", 1)

print(f"\n{'='*60}")
print(f"  TRANSFORMER TRAINING: epoch {start_epoch+1} -> {config['epochs']}")
print(f"  Effective batch size: {config['batch_size']} x {accumulate_grad} = {config['batch_size'] * accumulate_grad}")
print(f"{'='*60}\n")

for epoch in range(start_epoch, config["epochs"]):
    epoch_start = time.time()

    # ── Train ──
    model.train()
    epoch_loss = 0
    optimizer.zero_grad()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} Train")
    for step_i, batch in enumerate(pbar):
        src = batch['src'].to(device)
        trg = batch['trg'].to(device)
        tgt_in  = trg[:, :-1]
        tgt_out = trg[:, 1:]

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(src, tgt_in)
            loss = criterion(
                logits.contiguous().view(-1, logits.size(-1)),
                tgt_out.contiguous().view(-1)
            )
            loss = loss / accumulate_grad  # Scale loss for accumulation

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update weights every accumulate_grad steps
        if (step_i + 1) % accumulate_grad == 0 or (step_i + 1) == len(train_loader):
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                optimizer.step()
            lr = scheduler.step()
            optimizer.zero_grad()

        epoch_loss += loss.item() * accumulate_grad  # Unscale for logging
        pbar.set_postfix(loss=f"{loss.item() * accumulate_grad:.4f}", lr=f"{lr:.2e}" if 'lr' in dir() else "warmup")

    train_loss = epoch_loss / len(train_loader)

    # ── Validate ──
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Val", leave=False):
            src = batch['src'].to(device)
            trg = batch['trg'].to(device)
            tgt_in  = trg[:, :-1]
            tgt_out = trg[:, 1:]
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                logits = model(src, tgt_in)
                loss = criterion(
                    logits.contiguous().view(-1, logits.size(-1)),
                    tgt_out.contiguous().view(-1)
                )
            val_loss += loss.item()
    val_loss /= len(val_loader)

    # Save history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_ppl'].append(math.exp(min(train_loss, 100)))
    history['val_ppl'].append(math.exp(min(val_loss, 100)))

    elapsed = time.time() - epoch_start
    mins, secs = int(elapsed // 60), int(elapsed % 60)

    print(f"\nEpoch {epoch+1}/{config['epochs']} | {mins}m {secs}s")
    print(f"  Train Loss: {train_loss:.4f} | PPL: {math.exp(min(train_loss,100)):.2f}")
    print(f"  Val   Loss: {val_loss:.4f} | PPL: {math.exp(min(val_loss,100)):.2f}")

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
            'best_val_loss': best_val_loss,
            'history': history,
            'config': config,
        }, best_path)
        print("  >>> Best model saved!")
    else:
        patience_counter += 1
        print(f"  No improvement ({patience_counter}/{patience})")
        if patience_counter >= patience:
            print("  Early stopping!")
            break

    # Save resume checkpoint moi epoch (de resume khi Kaggle het 12h)
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_step': scheduler._step,
        'best_val_loss': best_val_loss,
        'history': history,
    }, resume_path)

print(f"\nTraining done! Best val loss: {best_val_loss:.4f}")

# ── Cell 13: Plot Training Curves ───────────────────────────
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train', marker='o', markersize=4)
plt.plot(history['val_loss'],   label='Val',   marker='s', markersize=4)
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Loss Curves - Transformer EN->VI"); plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['train_ppl'], label='Train PPL', marker='o', markersize=4)
plt.plot(history['val_ppl'],   label='Val PPL',   marker='s', markersize=4)
plt.xlabel("Epoch"); plt.ylabel("Perplexity")
plt.title("Perplexity - Transformer EN->VI"); plt.legend(); plt.grid(True)

plt.tight_layout()
plt.savefig("/kaggle/working/transformer_training_curves.png", dpi=150)
plt.show()

# ── Cell 14: Quick BLEU Evaluation ──────────────────────────
from sacrebleu.metrics import BLEU as SBLEU

# Load best model
ckpt = torch.load(best_path, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

preds, refs = [], []
MAX_EVAL = 500

print(f"\nEvaluating BLEU on {MAX_EVAL} test samples...")
with torch.no_grad():
    count = 0
    for batch in tqdm(test_loader, desc="Evaluating"):
        src = batch['src'].to(device)
        trg = batch['trg']

        for j in range(src.size(0)):
            if count >= MAX_EVAL:
                break
            src_j = src[j].unsqueeze(0)
            out = model.beam_search_decode(
                src_j, sos_idx=trg_vocab.sos_idx,
                eos_idx=trg_vocab.eos_idx, pad_idx=trg_vocab.pad_idx,
                beam_size=4, length_penalty=1.0
            )
            preds.append(trg_vocab.decode(out))
            refs.append(trg_vocab.decode(trg[j].tolist()))
            count += 1

        if count >= MAX_EVAL:
            break

bleu = SBLEU()
bleu_score = bleu.corpus_score(preds, [refs])
print(f"\n{'='*50}")
print(f"  BLEU (first {MAX_EVAL} samples): {bleu_score.score:.2f}")
print(f"  Precisions: {[f'{p:.1f}' for p in bleu_score.precisions]}")
print(f"  BP: {bleu_score.bp:.4f}")
print(f"{'='*50}")

# Debug: print 5 sample predictions vs references
print(f"\n--- Debug: First 5 predictions vs references ---")
for i in range(min(5, len(preds))):
    print(f"\n[{i+1}] PRED (len={len(preds[i].split())}): {preds[i][:150]}")
    print(f"     REF  (len={len(refs[i].split())}): {refs[i][:150]}")
print("-"*50)

# ── Cell 15: METEOR + TER ───────────────────────────────────
from sacrebleu.metrics import TER
from nltk.translate.meteor_score import meteor_score
import nltk
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

meteor_scores = []
for pred, ref in zip(preds, refs):
    score = meteor_score([ref.split()], pred.split())
    meteor_scores.append(score)
avg_meteor = sum(meteor_scores) / len(meteor_scores) * 100

ter = TER()
ter_score = ter.corpus_score(preds, [refs])

print(f"\n{'='*60}")
print(f"  EVALUATION RESULTS - Transformer")
print(f"{'='*60}")
print(f"  BLEU   : {bleu_score.score:.2f}")
print(f"  METEOR : {avg_meteor:.2f}")
print(f"  TER    : {ter_score.score:.2f}")
print(f"{'='*60}")

# ── Cell 16: Sample Translations ────────────────────────────
def translate_sentence_simple(model, sentence, src_vocab, trg_vocab, device, beam_size=4):
    model.eval()
    with torch.no_grad():
        src_indices = src_vocab.encode(sentence)
        src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
        out = model.beam_search_decode(
            src_tensor, sos_idx=trg_vocab.sos_idx,
            eos_idx=trg_vocab.eos_idx, pad_idx=trg_vocab.pad_idx,
            beam_size=beam_size, length_penalty=1.0
        )
        return trg_vocab.decode(out)

test_sentences = [
    "The weather is beautiful today.",
    "I love learning new languages.",
    "Technology is changing the world.",
    "The doctor prescribed antibiotics for the infection.",
    "She has been studying medicine for six years.",
]

print("\n" + "="*60)
print("  SAMPLE TRANSLATIONS")
print("="*60)

for sent in test_sentences:
    vi = translate_sentence_simple(model, sent, src_vocab, trg_vocab, device)
    print(f"\nEN: {sent}")
    print(f"VI: {vi}")
    print("-"*60)

# Test set samples
print("\n" + "="*60)
print("  TEST SET SAMPLES")
print("="*60)

for i in range(5):
    sample = test_dataset[i]
    src_text = sample['src_text']
    ref_text = sample['trg_text']
    pred_text = translate_sentence_simple(model, src_text, src_vocab, trg_vocab, device)
    print(f"\n[{i+1}]")
    print(f"  Source (EN):    {src_text[:150]}")
    print(f"  Reference (VI): {ref_text[:150]}")
    print(f"  Predicted (VI): {pred_text[:150]}")
    print("-"*60)

# ── Cell 17: Save Results vào checkpoint + JSON ─────────────
import json

results = {
    'model': 'Transformer (Attention Is All You Need)',
    'params': count_parameters(model),
    'd_model': config['d_model'],
    'num_heads': config['num_heads'],
    'num_encoder_layers': config['num_encoder_layers'],
    'num_decoder_layers': config['num_decoder_layers'],
    'epochs_trained': len(history['train_loss']),
    'best_val_loss': best_val_loss,
    'best_val_ppl': math.exp(min(best_val_loss, 100)),
    'bleu_score': bleu_score.score,
    'bleu_precisions': list(bleu_score.precisions),
    'bleu_bp': bleu_score.bp,
    'meteor_score': avg_meteor,
    'ter_score': ter_score.score,
    'history': history,
}

# Lưu vào JSON
with open('/kaggle/working/transformer_results.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Lưu kết quả vào best_model checkpoint
print("\nSaving evaluation results into checkpoint...")
ckpt = torch.load(best_path, map_location=device)
ckpt['eval_results'] = {
    'bleu_score': bleu_score.score,
    'bleu_precisions': list(bleu_score.precisions),
    'bleu_bp': bleu_score.bp,
    'meteor_score': avg_meteor,
    'ter_score': ter_score.score,
}
ckpt['history'] = history
torch.save(ckpt, best_path)
print(f"  >>> Checkpoint updated: {best_path}")

print("\n" + "="*60)
print("  TRAINING SUMMARY")
print("="*60)
print(f"  Model: Transformer (d_model={config['d_model']}, {config['num_encoder_layers']}L)")
print(f"  Parameters: {count_parameters(model):,}")
print(f"  Best Val Loss: {best_val_loss:.4f}")
print(f"  Best Val PPL:  {math.exp(min(best_val_loss, 100)):.2f}")
print(f"  BLEU:   {bleu_score.score:.2f} (BP={bleu_score.bp:.4f})")
print(f"  METEOR: {avg_meteor:.2f}")
print(f"  TER:    {ter_score.score:.2f}")
print("="*60)
print(f"\nFiles saved:")
print(f"  - {best_path}")
print(f"  - /kaggle/working/transformer_results.json")
print("Done!")
