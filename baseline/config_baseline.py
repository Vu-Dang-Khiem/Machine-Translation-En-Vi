# Hyperparameters (learning_rate, batch_size, hidden_dim...)
"""
Configuration cho Baseline Seq2Seq + Attention model
"""

import torch

BASELINE_CONFIG = {
    # ===== Kiến trúc Model =====
    "embedding_dim": 256,           # Kích thước vector embedding (biểu diễn từ),256
    "hidden_dim": 512,              # Kích thước hidden state LSTM
    "num_layers": 2,                # Số layer LSTM xếp chồng,2
    "dropout": 0.3,                 # Tỷ lệ dropout (chống overfitting)
    "bidirectional": True,          # Bi-LSTM cho encoder (đọc 2 chiều)
    
    # ===== Huấn luyện =====
    "learning_rate": 3e-4,          # Tốc độ học (learning rate)
    "batch_size": 32,
    "epochs": 20,                    # Số epoch huấn luyện
    "teacher_forcing_ratio": 0.5,   # Tỷ lệ dùng ground truth khi decode
    "grad_clip": 1.0,               # Gradient clipping (tránh exploding gradient)
    
    # ===== Dữ liệu =====
    "max_length": 200,              # Độ dài tối đa câu (token)
    "min_length": 5,                # Độ dài tối thiểu câu (token)
    "min_freq": 1,                  # Tần suất tối thiểu để giữ token trong vocab
    
    # ===== Vocabulary tokens =====
    "pad_token": "<pad>",           # Token padding (đệm câu ngắn)
    "sos_token": "<sos>",           # Start of sentence (bắt đầu câu)
    "eos_token": "<eos>",           # End of sentence (kết thúc câu)
    "unk_token": "<unk>",           # Unknown token (từ không biết)
    
    # ===== Chỉ số đặc biệt =====
    "pad_idx": 0,                   # Index của <pad>
    "sos_idx": 1,                   # Index của <sos>
    "eos_idx": 2,                   # Index của <eos>
    "unk_idx": 3,                   # Index của <unk>
    
    # ===== Đường dẫn =====
    "data_dir": "./split_dataset",              # Thư mục chứa dataset
    "checkpoint_dir": "./baseline/checkpoints",  # Thư mục lưu checkpoint
    "vocab_dir": "./baseline/vocab",             # Thư mục lưu vocabulary
    "save_every": 1,                # Lưu checkpoint mỗi N epoch
    
    # ===== BPE Tokenization =====
    "use_bpe": True,                # Dùng BPE subword thay word-level
    "bpe_vocab_size": 12000,        # Kích thước vocab BPE mỗi ngôn ngữ
    
    # ===== Coverage Mechanism =====
    "use_coverage": True,           # Bật coverage attention
    "coverage_lambda": 0.3,         # Trọng số coverage loss
    
    # ===== Label Smoothing =====
    "label_smoothing": 0.1,         # Hệ số label smoothing (0 = tắt)
    
    # ===== Thiết bị =====
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # GPU hoặc CPU
}

# Derived configs
BASELINE_CONFIG["encoder_hidden_dim"] = (
    BASELINE_CONFIG["hidden_dim"] * 2 
    if BASELINE_CONFIG["bidirectional"] 
    else BASELINE_CONFIG["hidden_dim"]
)
