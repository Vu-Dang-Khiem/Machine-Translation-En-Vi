# Baseline Seq2Seq + Attention Model

Mô hình baseline cho Machine Translation Anh-Việt sử dụng kiến trúc Seq2Seq với Bahdanau Attention.

## Kiến trúc

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Encoder      │────▶│    Attention    │────▶│    Decoder      │
│  (Bi-LSTM)      │     │   (Bahdanau)    │     │    (LSTM)       │
│  2 layers       │     │                 │     │   2 layers      │
│  hidden=512     │     │                 │     │   hidden=512    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Cấu trúc files

```
baseline/
├── config_baseline.py      # Hyperparameters
├── seq2seq_model.py        # Model architecture
├── data_utils.py           # Vocabulary & DataLoader
├── train_baseline.py       # Training script
├── evaluate_baseline.py    # Evaluation & BLEU
├── vocab/                  # Saved vocabularies
└── checkpoints/            # Model checkpoints
```

## Hướng dẫn sử dụng

### 1. Chuẩn bị dữ liệu

Chạy script chia dataset (ở thư mục gốc):

```bash
cd "Machine Translation"
python split_data.py
```

### 2. Training

```bash
cd baseline
python train_baseline.py
```

Training sẽ:
- Build vocabulary từ training data
- Train model với teacher forcing
- Lưu best model theo validation loss
- Log metrics vào TensorBoard

### 3. Theo dõi training

```bash
tensorboard --logdir=./runs
```

### 4. Evaluation

```bash
python evaluate_baseline.py --checkpoint ./checkpoints/best_model.pt
```

Options:
- `--num_samples`: Số mẫu để tính BLEU (default: 1000)
- `--show_samples`: Số mẫu dịch hiển thị (default: 5)

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding dim | 256 |
| Hidden dim | 512 |
| Num layers | 2 |
| Dropout | 0.3 |
| Learning rate | 1e-3 |
| Batch size | 64 |
| Epochs | 20 |
| Teacher forcing ratio | 0.5 |
| Max length | 100 |

## Expected Results

- **BLEU Score**: 10-20 (baseline)
- **Training time**: 2-4 giờ (tùy GPU)
- **Parameters**: ~15M

## Notes

⚠️ **Hạn chế của Seq2Seq LSTM:**
- Khó xử lý câu dài (>100 tokens)
- Vanishing gradient với long sequences
- Chậm hơn Transformer do sequential processing

→ Đây là baseline để so sánh với Transformer.
