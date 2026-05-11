# -*- coding: utf-8 -*-
# Kiến trúc model
"""
Baseline Seq2Seq + Bahdanau Attention Model cho Machine Translation
Kiến trúc: Bi-LSTM Encoder + Bahdanau Attention + LSTM Decoder
Cải tiến: Coverage Mechanism + Beam Search with Repetition Penalty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Encoder(nn.Module):
    """
    Bi-LSTM Encoder
    Encode câu tiếng Anh thành hidden representations
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, bidirectional=True):
        super(Encoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Project bidirectional hidden to decoder hidden
        if bidirectional:
            self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
            self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, src, src_lengths):
        """
        Args:
            src: [batch_size, src_len] - source sequence
            src_lengths: [batch_size] - lengths of each source sequence
        Returns:
            outputs: [batch_size, src_len, hidden_dim * num_directions]
            hidden: [num_layers, batch_size, hidden_dim]
            cell: [num_layers, batch_size, hidden_dim]
        """
        # Embedding
        embedded = self.dropout(self.embedding(src))  # [batch, src_len, emb_dim]
        
        # Pack padded sequence for efficient LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM
        packed_outputs, (hidden, cell) = self.lstm(packed)
        
        # Unpack
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        # outputs: [batch, src_len, hidden * num_directions]
        
        # Combine bidirectional hidden states
        if self.bidirectional:
            # hidden: [num_layers * 2, batch, hidden]
            # Reshape to [num_layers, 2, batch, hidden] then combine
            hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
            hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
            hidden = torch.tanh(self.fc_hidden(hidden))
            
            cell = cell.view(self.num_layers, 2, -1, self.hidden_dim)
            cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)
            cell = torch.tanh(self.fc_cell(cell))
        
        return outputs, hidden, cell


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention với Coverage
    score(h_t, h_s, c) = v^T * tanh(W_h * h_t + W_s * h_s + W_c * c)
    """
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, use_coverage=False):
        super(BahdanauAttention, self).__init__()
        
        self.use_coverage = use_coverage
        
        self.W_encoder = nn.Linear(encoder_hidden_dim, decoder_hidden_dim, bias=False)
        self.W_decoder = nn.Linear(decoder_hidden_dim, decoder_hidden_dim, bias=False)
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)
        
        # Coverage projection
        if use_coverage:
            self.W_coverage = nn.Linear(1, decoder_hidden_dim, bias=False)
    
    def forward(self, decoder_hidden, encoder_outputs, mask=None, coverage=None):
        """
        Args:
            decoder_hidden: [batch_size, decoder_hidden_dim]
            encoder_outputs: [batch_size, src_len, encoder_hidden_dim]
            mask: [batch_size, src_len] - padding mask
            coverage: [batch_size, src_len] - coverage vector (tổng attention trước đó)
        Returns:
            attention_weights: [batch_size, src_len]
            context: [batch_size, encoder_hidden_dim]
        """
        batch_size, src_len, _ = encoder_outputs.shape
        
        # [batch, src_len, dec_hidden]
        encoder_proj = self.W_encoder(encoder_outputs)
        
        # [batch, 1, dec_hidden] -> [batch, src_len, dec_hidden]
        decoder_proj = self.W_decoder(decoder_hidden).unsqueeze(1).expand(-1, src_len, -1)
        
        # Tính score
        energy = encoder_proj + decoder_proj
        
        # Thêm coverage vào score nếu bật
        if self.use_coverage and coverage is not None:
            # coverage: [batch, src_len] -> [batch, src_len, 1]
            coverage_proj = self.W_coverage(coverage.unsqueeze(2))
            energy = energy + coverage_proj
        
        # [batch, src_len, dec_hidden] -> [batch, src_len, 1] -> [batch, src_len]
        scores = self.v(torch.tanh(energy)).squeeze(2)
        
        # Apply mask (set padding positions to -inf)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax over source sequence
        attention_weights = F.softmax(scores, dim=1)  # [batch, src_len]
        
        # Context vector: weighted sum of encoder outputs
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)  # [batch, encoder_hidden_dim]
        
        return attention_weights, context


class Decoder(nn.Module):
    """
    LSTM Decoder với Attention + Coverage
    """
    def __init__(self, vocab_size, embedding_dim, encoder_hidden_dim, 
                 decoder_hidden_dim, num_layers, dropout, use_coverage=False):
        super(Decoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.decoder_hidden_dim = decoder_hidden_dim
        self.num_layers = num_layers
        self.use_coverage = use_coverage
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        self.attention = BahdanauAttention(encoder_hidden_dim, decoder_hidden_dim, use_coverage)
        
        # Input: embedding + context vector
        self.lstm = nn.LSTM(
            embedding_dim + encoder_hidden_dim,
            decoder_hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection
        self.fc_out = nn.Linear(
            decoder_hidden_dim + encoder_hidden_dim + embedding_dim,
            vocab_size
        )
    
    def forward(self, input_token, hidden, cell, encoder_outputs, mask=None, coverage=None):
        """
        Decode một bước (single time step)
        Args:
            input_token: [batch_size] - current input token
            hidden: [num_layers, batch_size, decoder_hidden_dim]
            cell: [num_layers, batch_size, decoder_hidden_dim]
            encoder_outputs: [batch_size, src_len, encoder_hidden_dim]
            mask: [batch_size, src_len]
            coverage: [batch_size, src_len] - coverage vector
        Returns:
            prediction: [batch_size, vocab_size]
            hidden: [num_layers, batch_size, decoder_hidden_dim]
            cell: [num_layers, batch_size, decoder_hidden_dim]
            attention_weights: [batch_size, src_len]
        """
        # Embedding
        embedded = self.dropout(self.embedding(input_token))  # [batch, emb_dim]
        embedded = embedded.unsqueeze(1)  # [batch, 1, emb_dim]
        
        # Attention using top layer hidden state (truyền coverage)
        attn_weights, context = self.attention(hidden[-1], encoder_outputs, mask, coverage)
        context = context.unsqueeze(1)  # [batch, 1, enc_hidden]
        
        # Concatenate embedding and context
        lstm_input = torch.cat([embedded, context], dim=2)  # [batch, 1, emb + enc_hidden]
        
        # LSTM step
        lstm_output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        # Output prediction
        lstm_output = lstm_output.squeeze(1)  # [batch, dec_hidden]
        context = context.squeeze(1)  # [batch, enc_hidden]
        embedded = embedded.squeeze(1)  # [batch, emb_dim]
        
        # Combine all for prediction
        prediction = self.fc_out(torch.cat([lstm_output, context, embedded], dim=1))
        
        return prediction, hidden, cell, attn_weights


class Seq2Seq(nn.Module):
    """
    Full Seq2Seq model với Attention + Coverage
    """
    def __init__(self, encoder, decoder, device, sos_idx, eos_idx, pad_idx, use_coverage=False):
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.use_coverage = use_coverage
    
    def create_mask(self, src):
        """Create mask for source padding"""
        mask = (src != self.pad_idx)
        return mask
    
    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        """
        Args:
            src: [batch_size, src_len]
            src_lengths: [batch_size]
            trg: [batch_size, trg_len]
            teacher_forcing_ratio: probability of using ground truth token
        Returns:
            outputs: [batch_size, trg_len, vocab_size]
            coverage_loss: scalar (0.0 nếu không dùng coverage)
        """
        batch_size = src.size(0)
        trg_len = trg.size(1)
        vocab_size = self.decoder.vocab_size
        
        # Store outputs
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(self.device)
        
        # Encode
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        
        # Create mask
        mask = self.create_mask(src)
        src_len = encoder_outputs.size(1)
        
        # Khởi tạo coverage vector
        coverage = torch.zeros(batch_size, src_len).to(self.device)
        coverage_loss = torch.tensor(0.0).to(self.device)
        
        # First input is SOS token
        input_token = trg[:, 0]  # [batch]
        
        for t in range(1, trg_len):
            # Decode one step (truyền coverage)
            prediction, hidden, cell, attn_weights = self.decoder(
                input_token, hidden, cell, encoder_outputs, mask,
                coverage if self.use_coverage else None
            )
            
            # Tính coverage loss: sum(min(attn, coverage))
            if self.use_coverage:
                # Coverage loss phạt khi attention lặp vào vị trí đã chú ý
                step_coverage_loss = torch.sum(torch.min(attn_weights, coverage), dim=1)
                coverage_loss = coverage_loss + step_coverage_loss.mean()
                
                # Cập nhật coverage
                coverage = coverage + attn_weights
            
            # Store prediction
            outputs[:, t, :] = prediction
            
            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            
            input_token = trg[:, t] if teacher_force else top1
        
        return outputs, coverage_loss
    
    def translate(self, src, src_lengths, max_len=100):
        """
        Translate without teacher forcing (for inference) - Greedy decoding
        """
        self.eval()
        
        with torch.no_grad():
            batch_size = src.size(0)
            
            # Encode
            encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
            mask = self.create_mask(src)
            src_len = encoder_outputs.size(1)
            
            # Khởi tạo coverage
            coverage = torch.zeros(batch_size, src_len).to(self.device)
            
            # Start with SOS
            input_token = torch.LongTensor([self.sos_idx] * batch_size).to(self.device)
            
            translations = []
            attentions = []
            
            for _ in range(max_len):
                prediction, hidden, cell, attn = self.decoder(
                    input_token, hidden, cell, encoder_outputs, mask,
                    coverage if self.use_coverage else None
                )
                
                # Cập nhật coverage
                if self.use_coverage:
                    coverage = coverage + attn
                
                top1 = prediction.argmax(1)
                translations.append(top1)
                attentions.append(attn)
                
                # Stop if all sequences have EOS
                if (top1 == self.eos_idx).all():
                    break
                
                input_token = top1
        
        translations = torch.stack(translations, dim=1)  # [batch, seq_len]
        attentions = torch.stack(attentions, dim=1)  # [batch, seq_len, src_len]
        
        return translations, attentions

    def beam_search_translate(self, src, src_lengths, max_len=100, beam_width=5,
                               repetition_penalty=1.2, no_repeat_ngram_size=3):
        """
        Beam Search Decoding với Repetition Penalty, N-gram Blocking và Coverage
        
        Args:
            src: [1, src_len] - chỉ hỗ trợ batch_size=1
            src_lengths: [1]
            max_len: số bước decode tối đa
            beam_width: số beam (ứng viên) giữ lại mỗi bước
            repetition_penalty: hệ số phạt từ đã xuất hiện (>1.0 để phạt lặp)
            no_repeat_ngram_size: chặn n-gram lặp lại (0 = tắt)
        Returns:
            best_sequence: [1, seq_len] - câu dịch tốt nhất
            attention_weights: [1, seq_len, src_len]
        """
        self.eval()

        with torch.no_grad():
            # Encode source
            encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
            mask = self.create_mask(src)
            src_len = encoder_outputs.size(1)

            # Khởi tạo coverage
            init_coverage = torch.zeros(1, src_len).to(self.device)

            # Beam list: [(score, tokens, hidden, cell, attentions, coverage)]
            beams = [(0.0, [self.sos_idx], hidden, cell, [], init_coverage)]
            completed_beams = []

            for step in range(max_len):
                new_beams = []

                for score, tokens, h, c, attns, cov in beams:
                    # Nếu beam đã kết thúc (token cuối là EOS), chuyển sang completed
                    if tokens[-1] == self.eos_idx:
                        completed_beams.append((score, tokens, attns))
                        continue

                    # Decode một bước (truyền coverage)
                    input_token = torch.LongTensor([tokens[-1]]).to(self.device)
                    prediction, new_h, new_c, attn = self.decoder(
                        input_token, h, c, encoder_outputs, mask,
                        cov if self.use_coverage else None
                    )
                    
                    # Cập nhật coverage
                    new_cov = cov + attn if self.use_coverage else cov

                    # Tính log probability
                    log_probs = F.log_softmax(prediction, dim=1).squeeze(0)  # [vocab_size]

                    # === REPETITION PENALTY ===
                    if repetition_penalty > 1.0:
                        for prev_token in set(tokens):
                            if log_probs[prev_token] < 0:
                                log_probs[prev_token] = log_probs[prev_token] * repetition_penalty
                            else:
                                log_probs[prev_token] = log_probs[prev_token] / repetition_penalty

                    # === N-GRAM BLOCKING ===
                    if no_repeat_ngram_size > 0 and len(tokens) >= no_repeat_ngram_size:
                        ngram_prefix = tokens[-(no_repeat_ngram_size - 1):]
                        banned_tokens = set()
                        for i in range(len(tokens) - no_repeat_ngram_size + 1):
                            if tokens[i:i + no_repeat_ngram_size - 1] == ngram_prefix:
                                banned_tokens.add(tokens[i + no_repeat_ngram_size - 1])
                        for banned in banned_tokens:
                            log_probs[banned] = float('-inf')

                    # Lấy top beam_width token
                    topk_log_probs, topk_indices = log_probs.topk(beam_width)

                    for i in range(beam_width):
                        token_id = topk_indices[i].item()
                        token_score = topk_log_probs[i].item()
                        
                        if token_score == float('-inf'):
                            continue
                        
                        new_score = score + token_score

                        new_beams.append((
                            new_score,
                            tokens + [token_id],
                            new_h.clone(),
                            new_c.clone(),
                            attns + [attn],
                            new_cov.clone()
                        ))

                # Sắp xếp theo score (có length normalization)
                alpha = 0.8
                new_beams.sort(
                    key=lambda x: x[0] / (len(x[1]) ** alpha),
                    reverse=True
                )
                beams = new_beams[:beam_width]

                if len(beams) == 0:
                    break

            # Thêm các beam chưa hoàn thành
            for score, tokens, h, c, attns, cov in beams:
                completed_beams.append((score, tokens, attns))

            # Chọn beam tốt nhất
            alpha = 0.8
            completed_beams.sort(
                key=lambda x: x[0] / (len(x[1]) ** alpha),
                reverse=True
            )
            best_score, best_tokens, best_attns = completed_beams[0]

            # Loại bỏ SOS và EOS
            best_tokens = best_tokens[1:]
            if best_tokens and best_tokens[-1] == self.eos_idx:
                best_tokens = best_tokens[:-1]

            # Chuyển thành tensor
            result = torch.LongTensor([best_tokens]).to(self.device)
            
            if best_attns:
                attn_tensor = torch.stack(best_attns, dim=1)
            else:
                attn_tensor = None

        return result, attn_tensor


def build_model(src_vocab_size, trg_vocab_size, config):
    """
    Factory function để build model từ config
    """
    use_coverage = config.get("use_coverage", False)
    
    encoder = Encoder(
        vocab_size=src_vocab_size,
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        bidirectional=config["bidirectional"]
    )
    
    decoder = Decoder(
        vocab_size=trg_vocab_size,
        embedding_dim=config["embedding_dim"],
        encoder_hidden_dim=config["encoder_hidden_dim"],
        decoder_hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        use_coverage=use_coverage
    )
    
    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        device=config["device"],
        sos_idx=config["sos_idx"],
        eos_idx=config["eos_idx"],
        pad_idx=config["pad_idx"],
        use_coverage=use_coverage
    )
    
    return model.to(config["device"])


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    from config_baseline import BASELINE_CONFIG
    
    # Dummy vocab sizes
    src_vocab = 10000
    trg_vocab = 10000
    
    model = build_model(src_vocab, trg_vocab, BASELINE_CONFIG)
    print(f"Model có {count_parameters(model):,} trainable parameters")
    
    # Test forward pass
    batch_size = 4
    src_len = 20
    trg_len = 25
    
    src = torch.randint(1, src_vocab, (batch_size, src_len))
    src_lengths = torch.LongTensor([src_len] * batch_size)
    trg = torch.randint(1, trg_vocab, (batch_size, trg_len))
    
    outputs, cov_loss = model(src, src_lengths, trg)
    print(f"Output shape: {outputs.shape}")  # [batch, trg_len, vocab]
    print(f"Coverage loss: {cov_loss.item():.4f}")
