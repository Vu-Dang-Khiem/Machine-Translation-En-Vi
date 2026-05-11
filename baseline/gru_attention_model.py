# -*- coding: utf-8 -*-
"""
Seq2Seq + GRU + Bahdanau Attention
GRU Encoder (1 chiều) + Bahdanau Attention + LSTM Decoder

Khác baseline: Encoder dùng GRU thay Bi-LSTM
Giữ nguyên: Attention, Coverage, Decoder (LSTM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class GRUAttEncoder(nn.Module):
    """
    GRU Encoder (1 chiều) - trả về outputs + hidden cho Attention
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(GRUAttEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        self.gru = nn.GRU(
            embedding_dim, hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
            batch_first=True
        )
    
    def forward(self, src, src_lengths):
        """
        Returns:
            outputs: [batch, src_len, hidden_dim] - cho Attention
            hidden: [num_layers, batch, hidden_dim]
            cell: [num_layers, batch, hidden_dim] - fake cell (zeros) cho Decoder LSTM
        """
        embedded = self.dropout(self.embedding(src))
        
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        packed_outputs, hidden = self.gru(packed)
        
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        
        # GRU không có cell state → tạo zeros cho Decoder LSTM
        cell = torch.zeros_like(hidden)
        
        return outputs, hidden, cell


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention với Coverage
    """
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, use_coverage=False):
        super(BahdanauAttention, self).__init__()
        
        self.use_coverage = use_coverage
        
        self.W_encoder = nn.Linear(encoder_hidden_dim, decoder_hidden_dim, bias=False)
        self.W_decoder = nn.Linear(decoder_hidden_dim, decoder_hidden_dim, bias=False)
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)
        
        if use_coverage:
            self.W_coverage = nn.Linear(1, decoder_hidden_dim, bias=False)
    
    def forward(self, decoder_hidden, encoder_outputs, mask=None, coverage=None):
        src_len = encoder_outputs.size(1)
        
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        energy = torch.tanh(
            self.W_encoder(encoder_outputs) + self.W_decoder(decoder_hidden_expanded)
        )
        
        if self.use_coverage and coverage is not None:
            coverage_input = coverage.unsqueeze(2)
            energy = energy + self.W_coverage(coverage_input)
            energy = torch.tanh(energy)
        
        attention_scores = self.v(energy).squeeze(2)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(attention_scores, dim=1)
        
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)
        
        return attention_weights, context


class AttDecoder(nn.Module):
    """
    LSTM Decoder với Attention (giống baseline)
    """
    def __init__(self, vocab_size, embedding_dim, encoder_hidden_dim,
                 decoder_hidden_dim, num_layers, dropout, use_coverage=False):
        super(AttDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.decoder_hidden_dim = decoder_hidden_dim
        self.num_layers = num_layers
        self.use_coverage = use_coverage
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        self.attention = BahdanauAttention(encoder_hidden_dim, decoder_hidden_dim, use_coverage)
        
        self.lstm = nn.LSTM(
            embedding_dim + encoder_hidden_dim,
            decoder_hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(
            decoder_hidden_dim + encoder_hidden_dim + embedding_dim,
            vocab_size
        )
    
    def forward(self, input_token, hidden, cell, encoder_outputs, mask=None, coverage=None):
        embedded = self.dropout(self.embedding(input_token))
        embedded = embedded.unsqueeze(1)
        
        attn_weights, context = self.attention(hidden[-1], encoder_outputs, mask, coverage)
        context = context.unsqueeze(1)
        
        lstm_input = torch.cat([embedded, context], dim=2)
        lstm_output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        lstm_output = lstm_output.squeeze(1)
        context = context.squeeze(1)
        embedded = embedded.squeeze(1)
        
        prediction = self.fc_out(torch.cat([lstm_output, context, embedded], dim=1))
        
        return prediction, hidden, cell, attn_weights


class GRUAttSeq2Seq(nn.Module):
    """
    Seq2Seq: GRU Encoder + Attention + LSTM Decoder
    """
    def __init__(self, encoder, decoder, device, sos_idx, eos_idx, pad_idx, use_coverage=False):
        super(GRUAttSeq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.use_coverage = use_coverage
    
    def create_mask(self, src):
        return (src != self.pad_idx)
    
    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        vocab_size = self.decoder.vocab_size
        
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(self.device)
        
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        mask = self.create_mask(src)
        
        input_token = trg[:, 0]
        
        coverage = torch.zeros(batch_size, encoder_outputs.size(1)).to(self.device) if self.use_coverage else None
        coverage_loss = torch.tensor(0.0).to(self.device)
        
        for t in range(1, trg_len):
            prediction, hidden, cell, attn_weights = self.decoder(
                input_token, hidden, cell, encoder_outputs, mask, coverage
            )
            
            outputs[:, t, :] = prediction
            
            if self.use_coverage:
                coverage_loss += torch.sum(torch.min(attn_weights, coverage))
                coverage = coverage + attn_weights
            
            if random.random() < teacher_forcing_ratio:
                input_token = trg[:, t]
            else:
                input_token = prediction.argmax(1)
        
        if self.use_coverage:
            coverage_loss = coverage_loss / batch_size
        
        return outputs, coverage_loss
    
    def beam_search_translate(self, src, src_lengths, max_len=100, beam_width=5,
                               repetition_penalty=1.2, no_repeat_ngram_size=3):
        """Beam Search Decoding"""
        self.eval()
        
        with torch.no_grad():
            encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
            mask = self.create_mask(src)
            
            beams = [(0.0, [self.sos_idx], hidden, cell,
                      torch.zeros(1, encoder_outputs.size(1)).to(self.device))]
            completed = []
            
            for step in range(max_len):
                candidates = []
                
                for score, seq, h, c, cov in beams:
                    if seq[-1] == self.eos_idx:
                        completed.append((score, seq, h, c, cov))
                        continue
                    
                    input_token = torch.LongTensor([seq[-1]]).to(self.device)
                    prediction, new_h, new_c, attn_w = self.decoder(
                        input_token, h, c, encoder_outputs, mask, cov
                    )
                    
                    new_cov = cov + attn_w if self.use_coverage else cov
                    log_probs = F.log_softmax(prediction, dim=-1).squeeze(0)
                    
                    for prev_token in set(seq):
                        log_probs[prev_token] /= repetition_penalty
                    
                    if len(seq) >= no_repeat_ngram_size:
                        ngram = tuple(seq[-(no_repeat_ngram_size-1):])
                        for i in range(len(seq) - no_repeat_ngram_size + 1):
                            prev_ngram = tuple(seq[i:i + no_repeat_ngram_size - 1])
                            if prev_ngram == ngram:
                                blocked_token = seq[i + no_repeat_ngram_size - 1]
                                log_probs[blocked_token] = float('-inf')
                    
                    topk_probs, topk_ids = log_probs.topk(beam_width)
                    
                    for prob, idx in zip(topk_probs.tolist(), topk_ids.tolist()):
                        candidates.append((score + prob, seq + [idx], new_h, new_c, new_cov))
                
                if not candidates:
                    break
                
                candidates.sort(key=lambda x: x[0] / len(x[1]), reverse=True)
                beams = candidates[:beam_width]
            
            completed.extend(beams)
            
            if not completed:
                return torch.LongTensor([[self.eos_idx]]).to(self.device), None
            
            best = max(completed, key=lambda x: x[0] / len(x[1]))
            best_seq = best[1]
            
            if best_seq[0] == self.sos_idx:
                best_seq = best_seq[1:]
            
            return torch.LongTensor([best_seq]).to(self.device), None


def build_gru_att_model(src_vocab_size, trg_vocab_size, config):
    """Factory function để build GRU + Attention model"""
    device = config["device"]
    encoder_hidden = config["hidden_dim"]  # GRU 1 chiều
    
    encoder = GRUAttEncoder(
        vocab_size=src_vocab_size,
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )
    
    decoder = AttDecoder(
        vocab_size=trg_vocab_size,
        embedding_dim=config["embedding_dim"],
        encoder_hidden_dim=encoder_hidden,
        decoder_hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        use_coverage=config.get("use_coverage", False),
    )
    
    model = GRUAttSeq2Seq(
        encoder=encoder,
        decoder=decoder,
        device=device,
        sos_idx=config["sos_idx"],
        eos_idx=config["eos_idx"],
        pad_idx=config["pad_idx"],
        use_coverage=config.get("use_coverage", False),
    )
    
    return model.to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
