# Config cho GRU + Attention (GRU encoder + Bahdanau Attention)
"""
GRU Encoder (1 chiều) + Bahdanau Attention + LSTM Decoder
Dùng để so sánh với LSTM+Attention và Bi-LSTM+Attention
"""

import torch

GRU_ATT_CONFIG = {
    # ===== Kiến trúc Model =====
    "embedding_dim": 256,
    "hidden_dim": 512,
    "num_layers": 2,
    "dropout": 0.3,
    "bidirectional": False,
    
    # ===== Huấn luyện =====
    "learning_rate": 3e-4,
    "batch_size": 32,
    "epochs": 20,
    "teacher_forcing_ratio": 0.5,
    "grad_clip": 1.0,
    
    # ===== Dữ liệu =====
    "max_length": 200,
    "min_length": 5,
    "min_freq": 1,
    
    # ===== Vocabulary tokens =====
    "pad_token": "<pad>",
    "sos_token": "<sos>",
    "eos_token": "<eos>",
    "unk_token": "<unk>",
    
    # ===== Chỉ số đặc biệt =====
    "pad_idx": 0,
    "sos_idx": 1,
    "eos_idx": 2,
    "unk_idx": 3,
    
    # ===== Đường dẫn =====
    "data_dir": "./split_dataset",
    "checkpoint_dir": "./gru_att/checkpoints",
    "vocab_dir": "./baseline/vocab",
    "save_every": 1,
    
    # ===== BPE Tokenization =====
    "use_bpe": True,
    "bpe_vocab_size": 12000,
    
    # ===== Attention & Coverage =====
    "use_coverage": True,
    
    # ===== Label Smoothing =====
    "label_smoothing": 0.1,
    
    # ===== Thiết bị =====
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# Derived configs
GRU_ATT_CONFIG["encoder_hidden_dim"] = (
    GRU_ATT_CONFIG["hidden_dim"] * 2 
    if GRU_ATT_CONFIG["bidirectional"] 
    else GRU_ATT_CONFIG["hidden_dim"]
)
