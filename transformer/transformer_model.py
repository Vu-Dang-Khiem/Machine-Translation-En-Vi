# -*- coding: utf-8 -*-
"""
Transformer Seq2Seq Model cho Machine Translation (EN → VI)
Theo paper: "Attention Is All You Need" (Vaswani et al., 2017)

Checklist paper:
  ✅ §3.1  Post-LayerNorm: LayerNorm(x + Sublayer(x))
  ✅ §3.2.1 Scaled Dot-Product Attention
  ✅ §3.2.2 Multi-Head Attention (h=8)
  ✅ §3.2.3 Masked self-attention (causal mask) trong decoder
  ✅ §3.3  FFN: ReLU(xW1+b1)W2+b2
  ✅ §3.4  Embedding scale × √d_model
  ✅ §3.4  Weight tying: trg_embed ↔ output_projection
  ✅ §3.5  Sinusoidal Positional Encoding
  ✅ §6.1  Beam Search decoding (beam=4, α=0.6)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# §3.5 — Sinusoidal Positional Encoding
# PE(pos,2i)   = sin(pos / 10000^(2i/d_model))
# PE(pos,2i+1) = cos(pos / 10000^(2i/d_model))
# ============================================================

class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding — không cần train.
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)                  # [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1).float()  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )  # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)        # even indices
        pe[:, 1::2] = torch.cos(position * div_term)        # odd indices
        pe = pe.unsqueeze(0)                                 # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================
# §3.4 — Token Embedding (scale × √d_model)
# ============================================================

class TokenEmbedding(nn.Module):
    """
    Token embedding nhân thêm √d_model theo paper §3.4.
    """
    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)


# ============================================================
# §3.1 — Encoder Layer (Post-LayerNorm)
# LayerNorm(x + MultiHeadAttention(x))
# LayerNorm(x + FFN(x))
# ============================================================

class EncoderLayer(nn.Module):
    """
    Một encoder layer: Self-Attention → FFN, cả hai có residual + LayerNorm.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor = None):
        """
        Args:
            src: [batch, src_len, d_model]
            src_key_padding_mask: [batch, src_len]  (True = masked/padding)
        """
        # Self-attention + residual + norm (Post-LN)
        attn_out, _ = self.self_attn(
            src, src, src, key_padding_mask=src_key_padding_mask
        )
        src = self.norm1(src + self.dropout1(attn_out))

        # FFN + residual + norm
        ffn_out = self.ffn(src)
        src = self.norm2(src + self.dropout2(ffn_out))

        return src


# ============================================================
# §3.1 — Decoder Layer (Post-LayerNorm)
# LayerNorm(x + MaskedSelfAttention(x))
# LayerNorm(x + CrossAttention(x, encoder_output))
# LayerNorm(x + FFN(x))
# ============================================================

class DecoderLayer(nn.Module):
    """
    Một decoder layer: Masked Self-Attention → Cross-Attention → FFN.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
    ):
        """
        Args:
            tgt:    [batch, tgt_len, d_model]
            memory: [batch, src_len, d_model]  (encoder output)
            tgt_mask: [tgt_len, tgt_len]  causal mask
            tgt_key_padding_mask: [batch, tgt_len]
            memory_key_padding_mask: [batch, src_len]
        """
        # 1) Masked self-attention
        attn_out, _ = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        tgt = self.norm1(tgt + self.dropout1(attn_out))

        # 2) Cross-attention (query=decoder, key/value=encoder)
        attn_out, _ = self.cross_attn(
            tgt, memory, memory,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = self.norm2(tgt + self.dropout2(attn_out))

        # 3) FFN
        ffn_out = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout3(ffn_out))

        return tgt


# ============================================================
# Full Transformer Encoder
# ============================================================

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int,
                 d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src, src_key_padding_mask=None):
        for layer in self.layers:
            src = layer(src, src_key_padding_mask)
        return src


# ============================================================
# Full Transformer Decoder
# ============================================================

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int,
                 d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, tgt, memory, tgt_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask,
                        tgt_key_padding_mask, memory_key_padding_mask)
        return tgt


# ============================================================
# §3.2.3 — Causal mask helper
# ============================================================

def make_causal_mask(sz: int, device: torch.device) -> torch.Tensor:
    """
    Tạo causal mask (upper-triangular = True) cho decoder self-attention.
    Shape: [sz, sz], dtype=bool
    True = vị trí bị masked (tương lai, không được nhìn).
    """
    return torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)


# ============================================================
# Full Transformer Seq2Seq Model
# ============================================================

class TransformerSeq2Seq(nn.Module):
    """
    Complete Transformer Seq2Seq model.
    Paper: "Attention Is All You Need" (Vaswani et al., 2017)
    """

    def __init__(
        self,
        src_vocab_size: int,
        trg_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.pad_idx = pad_idx
        self.d_model = d_model

        # §3.4 — Embeddings (scale × √d_model)
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model, pad_idx)
        self.trg_embedding = TokenEmbedding(trg_vocab_size, d_model, pad_idx)

        # §3.5 — Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # §3.1 — Encoder & Decoder stacks
        self.encoder = TransformerEncoder(
            num_encoder_layers, d_model, num_heads, d_ff, dropout
        )
        self.decoder = TransformerDecoder(
            num_decoder_layers, d_model, num_heads, d_ff, dropout
        )

        # Output projection
        self.output_projection = nn.Linear(d_model, trg_vocab_size)

        # §3.4 — Weight tying: share trg_embedding ↔ output_projection
        self.output_projection.weight = self.trg_embedding.embedding.weight

        # Initialize parameters (Xavier uniform, paper-aligned)
        self._init_parameters()

    def _init_parameters(self):
        """Xavier uniform initialization cho tất cả Linear và Embedding layers."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _make_src_key_padding_mask(self, src: torch.Tensor) -> torch.Tensor:
        """[batch, src_len] → True tại vị trí padding."""
        return (src == self.pad_idx)

    def _make_tgt_key_padding_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """[batch, tgt_len] → True tại vị trí padding."""
        return (tgt == self.pad_idx)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """
        Encode source sentence.
        Args:
            src: [batch, src_len]
        Returns:
            memory: [batch, src_len, d_model]
        """
        src_pad_mask = self._make_src_key_padding_mask(src)
        src_emb = self.positional_encoding(self.src_embedding(src))
        memory = self.encoder(src_emb, src_key_padding_mask=src_pad_mask)
        return memory

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode target tokens given encoder memory.
        Args:
            tgt:    [batch, tgt_len]
            memory: [batch, src_len, d_model]
            src:    [batch, src_len]   (for creating src padding mask)
        Returns:
            logits: [batch, tgt_len, trg_vocab_size]
        """
        tgt_len = tgt.size(1)
        tgt_mask = make_causal_mask(tgt_len, tgt.device)
        tgt_pad_mask = self._make_tgt_key_padding_mask(tgt)
        src_pad_mask = self._make_src_key_padding_mask(src)

        tgt_emb = self.positional_encoding(self.trg_embedding(tgt))

        decoder_out = self.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )

        logits = self.output_projection(decoder_out)
        return logits

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass (training).
        Args:
            src: [batch, src_len]
            tgt: [batch, tgt_len]    (teacher forcing input: <sos>...token_{T-1})
        Returns:
            logits: [batch, tgt_len, trg_vocab_size]
        """
        memory = self.encode(src)
        logits = self.decode(tgt, memory, src)
        return logits

    # ============================================================
    # Greedy Decoding (inference)
    # ============================================================

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int = 100,
    ) -> torch.Tensor:
        """
        Greedy decoding: chọn token có xác suất cao nhất mỗi bước.
        Args:
            src: [1, src_len]
        Returns:
            output: [1, decoded_len]  (không gồm <sos>)
        """
        self.eval()
        memory = self.encode(src)

        # Bắt đầu bằng <sos>
        ys = torch.LongTensor([[sos_idx]]).to(src.device)

        for _ in range(max_len):
            logits = self.decode(ys, memory, src)          # [1, cur_len, vocab]
            next_token = logits[:, -1, :].argmax(dim=-1)   # [1]
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)

            if next_token.item() == eos_idx:
                break

        # Bỏ <sos>, giữ nguyên <eos> (decode() sẽ xử lý)
        return ys[:, 1:]

    # ============================================================
    # §6.1 — Beam Search Decoding
    # score = log_prob / (len^α)     (length penalty)
    # ============================================================

    @torch.no_grad()
    def beam_search_decode(
        self,
        src: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        pad_idx: int,
        max_len: int = 200,
        beam_size: int = 4,
        length_penalty: float = 0.6,
    ) -> list:
        """
        Beam search decoding theo paper §6.1.
        Args:
            src: [1, src_len]
        Returns:
            best_sequence: list of token indices (không gồm <sos>/<eos>)
        """
        self.eval()
        device = src.device
        memory = self.encode(src)  # [1, src_len, d_model]

        # Beam: (score, token_list)
        beams = [(0.0, [sos_idx])]
        completed = []

        for step in range(max_len):
            candidates = []

            for score, tokens in beams:
                # Nếu beam đã kết thúc
                if tokens[-1] == eos_idx:
                    completed.append((score, tokens))
                    continue

                # Decode
                ys = torch.LongTensor([tokens]).to(device)  # [1, cur_len]
                logits = self.decode(ys, memory, src)       # [1, cur_len, vocab]
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)  # [vocab]

                # Top-k tokens
                topk_log_probs, topk_ids = log_probs.topk(beam_size)

                for i in range(beam_size):
                    token_id = topk_ids[i].item()
                    new_score = score + topk_log_probs[i].item()
                    candidates.append((new_score, tokens + [token_id]))

            if not candidates:
                break

            # Length-normalized score, chọn top beam_size
            alpha = length_penalty
            candidates.sort(
                key=lambda x: x[0] / (len(x[1]) ** alpha),
                reverse=True,
            )
            beams = candidates[:beam_size]

        # Thêm beams chưa hoàn thành
        for score, tokens in beams:
            completed.append((score, tokens))

        # Chọn best
        alpha = length_penalty
        completed.sort(
            key=lambda x: x[0] / (len(x[1]) ** alpha),
            reverse=True,
        )
        best_tokens = completed[0][1]

        # Loại bỏ <sos> và <eos>
        if best_tokens and best_tokens[0] == sos_idx:
            best_tokens = best_tokens[1:]
        if best_tokens and best_tokens[-1] == eos_idx:
            best_tokens = best_tokens[:-1]

        return best_tokens


# ============================================================
# Factory functions
# ============================================================

def build_transformer(src_vocab_size: int, trg_vocab_size: int, config: dict) -> TransformerSeq2Seq:
    """Build TransformerSeq2Seq model từ config dict."""
    device = config["device"]
    if isinstance(device, str):
        device = torch.device(device)

    model = TransformerSeq2Seq(
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        max_seq_len=config.get("max_seq_len", 512),
        pad_idx=config["pad_idx"],
    )

    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    """Đếm số trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# Quick test
# ============================================================

if __name__ == "__main__":
    from config_transformer import TRANSFORMER_CONFIG

    config = TRANSFORMER_CONFIG
    vocab_size = 16000

    model = build_transformer(vocab_size, vocab_size, config)
    n_params = count_parameters(model)
    print(f"Model: TransformerSeq2Seq")
    print(f"  d_model      = {config['d_model']}")
    print(f"  num_heads    = {config['num_heads']}")
    print(f"  enc_layers   = {config['num_encoder_layers']}")
    print(f"  dec_layers   = {config['num_decoder_layers']}")
    print(f"  d_ff         = {config['d_ff']}")
    print(f"  Parameters   = {n_params:,} ({n_params/1e6:.2f}M)")

    # Test forward
    batch_size, src_len, tgt_len = 2, 20, 15
    src = torch.randint(4, vocab_size, (batch_size, src_len))
    tgt = torch.randint(4, vocab_size, (batch_size, tgt_len))

    logits = model(src, tgt)
    print(f"\n  Input  src: {src.shape}")
    print(f"  Input  tgt: {tgt.shape}")
    print(f"  Output logits: {logits.shape}")  # [batch, tgt_len, vocab]

    # Test greedy decode
    src_single = src[:1]
    output = model.greedy_decode(src_single, sos_idx=1, eos_idx=2, max_len=30)
    print(f"\n  Greedy decode: {output.shape}")

    # Test beam search
    result = model.beam_search_decode(
        src_single, sos_idx=1, eos_idx=2, pad_idx=0,
        max_len=30, beam_size=4, length_penalty=0.6
    )
    print(f"  Beam search result: {result[:10]}...")
    print("\nAll tests passed!")
