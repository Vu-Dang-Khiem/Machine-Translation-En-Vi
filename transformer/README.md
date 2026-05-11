# Transformer MT (EN → VI) — Theo Paper "Attention Is All You Need"

## 📄 Paper Reference

**Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.**
- arXiv: https://arxiv.org/abs/1706.03762

---

## ✅ Checklist tuân thủ Paper

| Paper Section | Yêu cầu | Code | Trạng thái |
|---|---|---|---|
| **3.1** Encoder/Decoder | Post-LayerNorm: `LayerNorm(x + Sublayer(x))` | `EncoderLayer`, `DecoderLayer` | ✅ |
| **3.1** Residual | Residual connection quanh mỗi sub-layer | Tất cả layers | ✅ |
| **3.2.1** Attention | Scaled Dot-Product: `softmax(QK^T/√d_k)V` | `nn.MultiheadAttention` | ✅ |
| **3.2.2** Multi-Head | h=8 heads, d_k=d_v=d_model/h | Config: `num_heads=8` | ✅ |
| **3.2.3** Masked Attn | Causal mask trong decoder self-attention | `make_causal_mask()` | ✅ |
| **3.3** FFN | `ReLU(xW1+b1)W2+b2`, d_ff=2048 (base) | `nn.Sequential(Linear, ReLU, Linear)` | ✅ |
| **3.4** Embedding | Scale × √d_model | `TokenEmbedding` | ✅ |
| **3.4** Weight Tying | Share trg_embed ↔ output_projection | `output_projection.weight = trg_embed.weight` | ✅ |
| **3.5** Pos. Encoding | Sin/cos positional encoding | `PositionalEncoding` | ✅ |
| **5.1** Tokenization | **Shared BPE** vocab cho cả src+trg | `SharedBPEVocabulary` (1 model chung) | ✅ |
| **5.3** Optimizer | Adam(β1=0.9, β2=0.98, ε=10⁻⁹) | Config | ✅ |
| **5.3** LR Schedule | Noam warmup 4000 steps | `NoamScheduler` | ✅ |
| **5.4** Dropout | P_drop=0.1 trên sub-layer output + embedding | Config + code | ✅ |
| **5.4** Label Smooth | ε_ls=0.1 | `LabelSmoothingLoss` | ✅ |
| **6.1** Beam Search | beam=4, α=0.6 | `beam_search_decode()` | ✅ |

---

## 📁 Cấu trúc files

```
Machine Translation/
├── transformer/
│   ├── config_transformer.py      ← Config (2 profiles: BASE + SMALL)
│   ├── transformer_model.py       ← Model (Post-Norm, Weight Tying)
│   ├── shared_bpe_vocab.py        ← Shared BPE vocab (Paper §5.1)
│   ├── train_transformer.py       ← Training script
│   ├── evaluate_transformer.py    ← Evaluation (BLEU/METEOR/TER)
│   └── README.md                  ← File này
├── training_Transformer.py        ← Kaggle notebook
└── vocab/
    └── shared_bpe.model           ← 1 BPE model chung (EN+VI)
```
---

## ⚙️ 2 Config Profiles

### `TRANSFORMER_BASE` — Đúng paper (Table 3)
| Param | Value |
|---|---|
| d_model | 512 |
| num_heads | 8 |
| layers | 6 enc + 6 dec |
| d_ff | 2048 |
| dropout | 0.1 |
| ~params | ~44M |
| GPU cần | ≥ 12GB VRAM |

### `TRANSFORMER_SMALL` — Cho dataset nhỏ / Kaggle free
| Param | Value |
|---|---|
| d_model | 256 |
| num_heads | 8 |
| layers | 4 enc + 4 dec |
| d_ff | 1024 |
| dropout | 0.3 |
| ~params | ~15M |
| GPU cần | T4 (Kaggle free) |

**Chọn profile** trong `config_transformer.py`:
```python
TRANSFORMER_CONFIG = TRANSFORMER_SMALL  # hoặc TRANSFORMER_BASE
```

---

## 🚀 Cách chạy

### Local
```bash
cd "Machine Translation"

# Train
python -m transformer.train_transformer

# Evaluate
python -m transformer.evaluate_transformer \
    --checkpoint ./transformer/checkpoints/best_model.pt \
    --beam_size 4
```

### Kaggle
Upload `transformer/`, `baseline/`, `vocab/`, `split_dataset/` → chạy `training_Transformer.py`

---

## 🔑 Điểm khác biệt quan trọng so với baseline BiLSTM

| | BiLSTM + Attention | Transformer (paper) |
|---|---|---|
| Vocabulary | 2 vocab riêng (EN + VI, unigram) | **1 shared BPE vocab** |
| Normalization | Không LayerNorm | **Post-LayerNorm** |
| Attention | Bahdanau additive | **Scaled dot-product × 8 heads** |
| Position info | RNN thứ tự tự nhiên | **Sinusoidal Positional Encoding** |
| Weight sharing | Không | **trg_embed ↔ output_proj** |
| LR schedule | ReduceOnPlateau | **Noam warmup** |
| Parallelization | Sequential (chậm) | **Fully parallel (nhanh)** |
| Teacher forcing | Ratio 0.5 | **100% (ground truth khi train)** |
