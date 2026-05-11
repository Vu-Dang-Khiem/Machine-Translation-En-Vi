# 🌐 Machine Translation EN ↔ VI

> **Transformer-based Neural Machine Translation** for English–Vietnamese bidirectional translation.  
> Implements the full "Attention Is All You Need" (Vaswani et al., 2017) architecture from scratch in PyTorch.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📊 Results

### Transformer (EN → VI)

| Metric | Score |
|--------|-------|
| **BLEU** | **45.71** |
| **METEOR** | **66.56** |
| **TER** ↓ | **38.60** |
| Val Loss | 1.172 |
| Val PPL | 3.23 |
| Parameters | 76.9M |

### Baseline Bi-LSTM + Attention (EN → VI)

| Metric | Score |
|--------|-------|
| BLEU | 23.61 |
| Parameters | 44.7M |

> The Transformer model achieves **+22 BLEU** improvement over the Bi-LSTM baseline.

---

## 🏗️ Architecture

### Transformer (Main Model)

```
Encoder (6 layers)                    Decoder (6 layers)
┌──────────────────┐                  ┌──────────────────┐
│ Multi-Head Attn  │                  │ Masked Self-Attn │
│ (8 heads)        │                  │ (8 heads)        │
│ + LayerNorm      │                  │ + LayerNorm      │
│ + Residual       │                  │                  │
├──────────────────┤                  ├──────────────────┤
│ Feed-Forward     │                  │ Cross-Attention  │
│ (d_ff=2048)      │──── memory ────▶│ + LayerNorm      │
│ + LayerNorm      │                  ├──────────────────┤
│ + Residual       │                  │ Feed-Forward     │
└──────────────────┘                  │ + LayerNorm      │
                                      └──────────────────┘
                                              │
                                      ┌──────────────────┐
                                      │ Output Projection│
                                      │ (Weight Tying)   │
                                      └──────────────────┘
```

### Paper Compliance Checklist

| Paper Section | Feature | Status |
|---|---|---|
| §3.1 | Post-LayerNorm residual connections | ✅ |
| §3.2.1 | Scaled Dot-Product Attention | ✅ |
| §3.2.2 | Multi-Head Attention (h=8) | ✅ |
| §3.2.3 | Causal mask for decoder | ✅ |
| §3.3 | FFN: ReLU(xW₁+b₁)W₂+b₂ | ✅ |
| §3.4 | Embedding scale × √d_model | ✅ |
| §3.4 | Weight tying (trg_embed ↔ output_proj) | ✅ |
| §3.5 | Sinusoidal Positional Encoding | ✅ |
| §5.1 | Shared BPE vocabulary (32K subwords) | ✅ |
| §5.3 | Adam (β₁=0.9, β₂=0.98) + Noam LR | ✅ |
| §5.4 | Label Smoothing (ε=0.1) | ✅ |
| §6.1 | Beam Search (beam=4, α=0.6) | ✅ |

### Baseline (Bi-LSTM + Bahdanau Attention)

- Bidirectional LSTM Encoder (2 layers, hidden=512)
- Bahdanau Additive Attention + Coverage Mechanism
- LSTM Decoder with Teacher Forcing
- Beam Search with Repetition Penalty + N-gram Blocking

---

## 📁 Project Structure

```
Machine Translation/
│
├── transformer/                        # 🔥 Transformer package
│   ├── config_transformer.py           #    Config (BASE + SMALL profiles)
│   ├── transformer_model.py            #    Full Transformer model
│   ├── shared_bpe_vocab.py             #    Shared BPE vocabulary (HF Tokenizers)
│   ├── train_transformer.py            #    Training script (local)
│   ├── evaluate_transformer.py         #    Evaluation (BLEU/METEOR/TER)
│   └── README.md                       #    Transformer documentation
│
├── baseline/                           # 📋 Baseline Seq2Seq package
│   ├── config_baseline.py              #    Hyperparameters
│   ├── seq2seq_model.py                #    Bi-LSTM + Attention model
│   ├── gru_attention_model.py          #    GRU + Attention variant
│   ├── data_utils.py                   #    Dataset & DataLoader utilities
│   ├── train_baseline.py               #    Training script
│   ├── evaluate_baseline.py            #    Evaluation script
│   └── README.md                       #    Baseline documentation
│
├── training_Transformer.py             # 📓 Kaggle training script (EN→VI)
├── training_Transformer_vi2en.py       # 📓 Kaggle training script (VI→EN)
├── finetune_vi2en_from_envi.py         # 📓 Fine-tune VI→EN from EN→VI weights
│
├── demo_transformer.py                 # 🎨 Gradio web demo (Transformer)
├── demo_gradio.py                      # 🎨 Gradio web demo (Baseline)
├── run_demo.bat                        #    Windows launcher script
│
├── EDA_data.ipynb                      # 📊 Exploratory Data Analysis
├── Training_MT_Seq2seq.ipynb           # 📓 Seq2Seq training notebook
├── baseline_training_BiLSTM.ipynb      # 📓 BiLSTM training notebook
├── training_model_LSTM.ipynb           # 📓 LSTM training notebook
│
├── results.json                        # 📈 Baseline evaluation results
├── transformer_results.json            # 📈 Transformer evaluation results
├── transformer_training_curves.png     # 📉 Training loss/PPL curves
│
├── requirements.txt                    # 📦 Python dependencies
├── .gitignore                          #    Git ignore rules
└── README.md                           #    This file
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/Vu-Dang-Khiem/Machine-Translation-En-Vi.git
cd Machine-Translation-En-Vi

pip install -r requirements.txt
```

### 2. Download Pretrained Models

Model checkpoints are **not included** in the repository due to their size (~900MB each).  
You can train them yourself on Kaggle (see below) or download from the releases.

Place checkpoints in:
```
transformer_checkpoints/best_model.pt       # EN→VI model
transformer_checkpoints_vi2en/best_model.pt # VI→EN model
checkpoints/best_model.pt                   # Baseline LSTM model
```

### 3. Run Demo

```bash
python demo_transformer.py
```

This launches a Gradio web interface with:
- Bidirectional EN ↔ VI translation
- Adjustable beam size and length penalty
- Document translation (TXT, DOCX, PDF)
- Model performance metrics display

### 4. Train from Scratch (Kaggle)

1. Upload `transformer/` and `baseline/` as a Kaggle Dataset
2. Create a new Kaggle notebook with GPU (T4)
3. Copy content from `training_Transformer.py` into notebook cells
4. Run all cells (~12 hours per session, supports resume)

---

## 📦 Dataset

**HuggingFace**: [`ncduy/mt-en-vi`](https://huggingface.co/datasets/ncduy/mt-en-vi)

| Split | Samples |
|-------|---------|
| Train | 2,000,000 |
| Validation | ~10,000 |
| Test | ~10,000 |

The dataset contains English–Vietnamese parallel sentence pairs from various domains.

---

## 🔧 Configuration

Two config profiles are available in `transformer/config_transformer.py`:

| Profile | d_model | Heads | Layers | d_ff | Params | GPU |
|---------|---------|-------|--------|------|--------|-----|
| **BASE** | 512 | 8 | 6+6 | 2048 | ~77M | ≥12GB |
| SMALL | 256 | 8 | 4+4 | 1024 | ~15M | T4 free |

Switch profiles:
```python
TRANSFORMER_CONFIG = TRANSFORMER_BASE   # or TRANSFORMER_SMALL
```

---

## 📝 Training Details

| Setting | Value |
|---------|-------|
| Optimizer | Adam (β₁=0.9, β₂=0.98, ε=10⁻⁹) |
| LR Schedule | Noam warmup (8000 steps) |
| Batch Size | 32 × 4 (gradient accumulation) |
| Label Smoothing | ε = 0.1 |
| Mixed Precision | FP16 (AMP) |
| Early Stopping | Patience = 5–7 |
| Tokenization | Shared BPE (32K subwords, HF Tokenizers) |
| Decoding | Beam Search (beam=4, α=0.6) |

---

## 🤝 Acknowledgments

- **Paper**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- **Dataset**: [ncduy/mt-en-vi](https://huggingface.co/datasets/ncduy/mt-en-vi)
- **Framework**: PyTorch, Hugging Face Tokenizers, Gradio

---

## 📄 License

This project is for educational and research purposes.

---

**Author**: Khiem  
**Built with** ❤️ **using PyTorch**
