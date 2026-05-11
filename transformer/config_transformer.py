"""
Configuration cho Transformer Seq2Seq (EN → VI)
Theo paper: "Attention Is All You Need" (Vaswani et al., 2017)

2 profiles:
  - TRANSFORMER_BASE : Đúng paper Table 3 (~44M params, cần ≥12GB VRAM)
  - TRANSFORMER_SMALL: Cho dataset nhỏ / Kaggle free T4 (~15M params)
"""

import torch


# ============================================================
# TRANSFORMER_BASE — Đúng paper Table 3
# ============================================================

TRANSFORMER_BASE = {
    # ===== Model architecture (Paper §3) =====
    "d_model": 512,                 # Embedding & hidden dimension
    "num_heads": 8,                 # Số attention heads (§3.2.2)
    "num_encoder_layers": 6,        # Encoder layers (§3.1)
    "num_decoder_layers": 6,        # Decoder layers (§3.1)
    "d_ff": 2048,                   # Feed-forward inner dim (§3.3)
    "dropout": 0.1,                 # Dropout rate (§5.4)
    "max_seq_len": 512,             # Max positional encoding length

    # ===== Optimizer (Paper §5.3) =====
    "adam_betas": (0.9, 0.98),      # Adam β1, β2
    "adam_eps": 1e-9,               # Adam ε
    "warmup_steps": 8000,           # Noam warmup steps
    "grad_clip": 1.0,              # Gradient clipping

    # ===== Training =====
    "batch_size": 32,
    "accumulate_grad": 4,           # Gradient accumulation → effective batch = 128
    "epochs": 30,                   # Tăng epochs để model converge tốt hơn
    "label_smoothing": 0.1,         # Label smoothing ε (§5.4)

    # ===== Data =====
    "max_length": 200,              # Max token length per sentence
    "min_length": 5,

    # ===== Vocabulary / BPE =====
    "bpe_vocab_size": 32000,        # Shared BPE vocab size

    # ===== Special tokens =====
    "pad_idx": 0,
    "sos_idx": 1,
    "eos_idx": 2,
    "unk_idx": 3,

    # ===== Paths =====
    "data_dir": "./split_dataset",
    "checkpoint_dir": "./transformer/checkpoints",
    "vocab_dir": "./vocab",
    "save_every": 1,

    # ===== Device =====
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# ============================================================
# TRANSFORMER_SMALL — Cho Kaggle free T4 / dataset nhỏ
# ============================================================

TRANSFORMER_SMALL = {
    # ===== Model architecture =====
    "d_model": 256,                 # Nhỏ hơn → ít tham số
    "num_heads": 8,                 # Vẫn giữ 8 heads (d_k = 256/8 = 32)
    "num_encoder_layers": 4,        # Ít layers hơn
    "num_decoder_layers": 4,
    "d_ff": 1024,                   # FFN nhỏ hơn
    "dropout": 0.3,                 # Dropout cao hơn (dataset nhỏ → overfit)
    "max_seq_len": 512,

    # ===== Optimizer (vẫn theo paper) =====
    "adam_betas": (0.9, 0.98),
    "adam_eps": 1e-9,
    "warmup_steps": 4000,
    "grad_clip": 1.0,

    # ===== Training =====
    "batch_size": 64,               # Batch lớn hơn vì model nhỏ
    "epochs": 20,
    "label_smoothing": 0.1,

    # ===== Data =====
    "max_length": 600,
    "min_length": 5,

    # ===== Vocabulary / BPE =====
    "bpe_vocab_size": 16000,

    # ===== Special tokens =====
    "pad_idx": 0,
    "sos_idx": 1,
    "eos_idx": 2,
    "unk_idx": 3,

    # ===== Paths =====
    "data_dir": "./split_dataset",
    "checkpoint_dir": "./transformer/checkpoints",
    "vocab_dir": "./vocab",
    "save_every": 1,

    # ===== Device =====
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# ============================================================
# Chọn profile mặc định — đổi tại đây
# ============================================================

TRANSFORMER_CONFIG = TRANSFORMER_BASE  # 500K samples → dùng BASE (T4 16GB đủ chạy)
