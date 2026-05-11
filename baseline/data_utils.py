# Xử lý dữ liệu, tokenization
"""
Data utilities cho baseline Seq2Seq model
- Build vocabulary từ dataset (Word-level hoặc BPE)
- Custom Dataset và DataLoader
- Text ↔ Index conversion
- Hỗ trợ SentencePiece tokenization (Unigram/BPE)
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
try:
    from datasets import load_from_disk
except ImportError:
    load_from_disk = None  # Không cần cho demo/inference
import pickle
import os
from tqdm import tqdm


# Import SentencePiece
try:
    import sentencepiece as spm
    HAS_SENTENCEPIECE = True
except ImportError:
    HAS_SENTENCEPIECE = False
    print("⚠ sentencepiece chưa cài đặt - chỉ dùng word-level vocab")


def tokenize_english(text):
    """Tokenizer cho tiếng Anh - lowercase và split"""
    return text.lower().split()


def tokenize_vietnamese(text):
    """
    Tokenizer cho tiếng Việt
    - Dùng split() đơn giản (SentencePiece sẽ xử lý subword)
    - Giữ nguyên case (không lowercase)
    """
    return text.split()


# ============================================================
# Word-level Vocabulary (giữ nguyên cho backward compatibility)
# ============================================================

class Vocabulary:
    """
    Vocabulary class cho text ↔ index conversion
    Hỗ trợ tokenizer riêng cho English/Vietnamese
    """
    def __init__(self, min_freq=2, special_tokens=None, lang="en"):
        if special_tokens is None:
            special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
        
        self.min_freq = min_freq
        self.special_tokens = special_tokens
        self.lang = lang  # "en" hoặc "vi"
        
        self.token2idx = {}
        self.idx2token = {}
        self.token_freq = Counter()
        
        # Add special tokens
        for idx, token in enumerate(special_tokens):
            self.token2idx[token] = idx
            self.idx2token[idx] = token
        
        self.pad_idx = self.token2idx.get("<pad>", 0)
        self.sos_idx = self.token2idx.get("<sos>", 1)
        self.eos_idx = self.token2idx.get("<eos>", 2)
        self.unk_idx = self.token2idx.get("<unk>", 3)
    
    def _tokenize(self, sentence):
        """Chọn tokenizer phù hợp theo ngôn ngữ"""
        if self.lang == "vi":
            return tokenize_vietnamese(sentence)
        else:
            return tokenize_english(sentence)
    
    def build_vocab(self, sentences):
        """Build vocabulary từ list of sentences với progress bar"""
        print(f"Đang build vocabulary ({self.lang})...")
        
        # Count frequencies với progress bar
        for sentence in tqdm(sentences, desc=f"Tokenizing {self.lang.upper()}"):
            tokens = self._tokenize(sentence)
            self.token_freq.update(tokens)
        
        # Add tokens that meet min_freq threshold
        idx = len(self.special_tokens)
        for token, freq in self.token_freq.items():
            if freq >= self.min_freq:
                self.token2idx[token] = idx
                self.idx2token[idx] = token
                idx += 1
        
        print(f"✓ Vocabulary size ({self.lang.upper()}): {len(self.token2idx):,} tokens")
        print(f"  - Unique tokens: {len(self.token_freq):,}")
        print(f"  - Filtered (min_freq={self.min_freq}): {len(self.token2idx) - len(self.special_tokens):,}")
        return self
    
    def __len__(self):
        return len(self.token2idx)
    
    def encode(self, sentence, add_special=True):
        """Convert sentence to list of indices"""
        tokens = self._tokenize(sentence)
        indices = [self.token2idx.get(t, self.unk_idx) for t in tokens]
        
        if add_special:
            indices = [self.sos_idx] + indices + [self.eos_idx]
        
        return indices
    
    def decode(self, indices, remove_special=True):
        """Convert list of indices to sentence"""
        tokens = []
        for idx in indices:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            
            token = self.idx2token.get(idx, "<unk>")
            
            # Skip special tokens if requested
            if remove_special and token in self.special_tokens:
                if token == "<eos>":
                    break
                continue
            
            tokens.append(token)
        
        # Join tokens
        return " ".join(tokens)
    
    def save(self, path):
        """Save vocabulary to file"""
        with open(path, 'wb') as f:
            pickle.dump({
                'token2idx': self.token2idx,
                'idx2token': self.idx2token,
                'token_freq': self.token_freq,
                'min_freq': self.min_freq,
                'special_tokens': self.special_tokens,
                'lang': self.lang
            }, f)
        print(f"✓ Saved vocabulary to {path}")
    
    @classmethod
    def load(cls, path):
        """Load vocabulary from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        lang = data.get('lang', 'en')  # Backward compatibility
        vocab = cls(min_freq=data['min_freq'], special_tokens=data['special_tokens'], lang=lang)
        vocab.token2idx = data['token2idx']
        vocab.idx2token = data['idx2token']
        vocab.token_freq = data['token_freq']
        print(f"✓ Loaded vocabulary from {path} ({len(vocab):,} tokens)")
        return vocab


# ============================================================
# BPE Vocabulary (SentencePiece)
# ============================================================

class BPEVocabulary:
    """
    BPE Vocabulary dùng SentencePiece
    Chia từ thành subword units để xử lý từ hiếm tốt hơn
    
    VD: "kháng sinh" -> "▁kháng" + "▁sinh"
        "fluoroquinolone" -> "▁fluoro" + "quinol" + "one"
    """
    def __init__(self, vocab_size=16000, lang="en"):
        self.vocab_size = vocab_size
        self.lang = lang
        self.sp = None  # SentencePiece model
        
        # Special token indices (SentencePiece mặc định)
        # <unk>=0, <s>=1 (sos), </s>=2 (eos)
        # Ta sẽ thêm <pad>=0, shift các special token index
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3
    
    def train(self, sentences, model_prefix):
        """
        Train SentencePiece BPE model từ list of sentences
        
        Args:
            sentences: list of strings
            model_prefix: đường dẫn prefix để save model (VD: "./vocab/bpe_en")
        """
        if not HAS_SENTENCEPIECE:
            raise RuntimeError("Cần cài sentencepiece: pip install sentencepiece")
        
        # Lọc câu quá dài (công thức hóa học, chemical formulas)
        # Những câu này làm SentencePiece rất chậm
        max_char_len = 2000
        original_len = len(sentences)
        sentences = [s for s in sentences if len(s) <= max_char_len]
        filtered = original_len - len(sentences)
        if filtered > 0:
            print(f"Lọc bỏ {filtered:,} câu dài >{max_char_len} ký tự")
        
        # Giới hạn số câu để train nhanh hơn (20k là đủ cho BPE)
        max_sentences = 20000
        if len(sentences) > max_sentences:
            import random
            random.seed(42)
            sentences = random.sample(sentences, max_sentences)
            print(f"Giới hạn {max_sentences:,} câu cho BPE training")
        
        # Export sentences ra file text tạm
        tmp_file = model_prefix + "_train_text.txt"
        os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
        
        print(f"Exporting {len(sentences):,} sentences cho BPE training ({self.lang})...")
        with open(tmp_file, 'w', encoding='utf-8') as f:
            for sent in tqdm(sentences, desc=f"Export {self.lang.upper()}"):
                f.write(sent.strip() + '\n')
        
        # Train SentencePiece model
        print(f"Training BPE model ({self.lang}, vocab_size={self.vocab_size})... (có thể mất 2-5 phút)")
        spm.SentencePieceTrainer.train(
            input=tmp_file,
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            model_type='unigram',      # Unigram nhanh hơn BPE (~2-5 phút vs 75+ phút)
            pad_id=0,           # <pad> = 0
            bos_id=1,           # <sos>/<s> = 1
            eos_id=2,           # <eos>/</s> = 2
            unk_id=3,           # <unk> = 3
            pad_piece='<pad>',
            bos_piece='<sos>',
            eos_piece='<eos>',
            unk_piece='<unk>',
            character_coverage=0.995,    # Bỏ qua ký tự cực hiếm (hóa học)
            num_threads=4,
            input_sentence_size=20000,   # Giới hạn xử lý trong SentencePiece
            shuffle_input_sentence=True,
            max_sentence_length=4096,    # Giới hạn độ dài câu
            train_extremely_large_corpus=False,
        )
        
        # Load model vừa train
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_prefix + '.model')
        
        # Cleanup temp file
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        
        print(f"✓ BPE Vocabulary ({self.lang.upper()}): {self.sp.get_piece_size():,} subwords")
    
    def __len__(self):
        if self.sp is None:
            return 0
        return self.sp.get_piece_size()
    
    def encode(self, sentence, add_special=True):
        """Convert sentence to list of subword indices"""
        if self.sp is None:
            raise RuntimeError("BPE model chưa được load!")
        
        indices = self.sp.encode(sentence, out_type=int)
        
        if add_special:
            indices = [self.sos_idx] + indices + [self.eos_idx]
        
        return indices
    
    def decode(self, indices, remove_special=True):
        """Convert list of subword indices to sentence"""
        if self.sp is None:
            raise RuntimeError("BPE model chưa được load!")
        
        # Convert tensor to list
        clean_indices = []
        for idx in indices:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            
            # Skip special tokens
            if remove_special and idx in [self.pad_idx, self.sos_idx, self.eos_idx]:
                if idx == self.eos_idx:
                    break
                continue
            
            clean_indices.append(idx)
        
        # SentencePiece decode ghép subword lại thành text
        text = self.sp.decode(clean_indices)
        
        return text
    
    def save(self, path):
        """Save BPE vocab info"""
        with open(path, 'wb') as f:
            pickle.dump({
                'vocab_size': self.vocab_size,
                'lang': self.lang,
                'type': 'bpe'
            }, f)
        print(f"✓ Saved BPE vocab info to {path}")
    
    @classmethod
    def load(cls, path, model_path):
        """
        Load BPE vocabulary
        
        Args:
            path: đường dẫn file .pkl chứa metadata
            model_path: đường dẫn file .model của SentencePiece
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        vocab = cls(vocab_size=data['vocab_size'], lang=data['lang'])
        vocab.sp = spm.SentencePieceProcessor()
        vocab.sp.load(model_path)
        
        print(f"✓ Loaded BPE vocab from {model_path} ({len(vocab):,} subwords)")
        return vocab


class TranslationDataset(Dataset):
    """
    PyTorch Dataset cho machine translation
    """
    def __init__(self, data, src_vocab, trg_vocab, max_length=100, reverse=False):
        self.data = data
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_length = max_length
        self.reverse = reverse  # True: VI->EN (dao chieu)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        if self.reverse:
            # VI -> EN: source la VI, target la EN
            src_text = item["vi"]
            trg_text = item["en"]
        else:
            # EN -> VI (mac dinh): source la EN, target la VI
            src_text = item["en"]
            trg_text = item["vi"]
        
        # Encode
        src_indices = self.src_vocab.encode(src_text)
        trg_indices = self.trg_vocab.encode(trg_text)
        
        # Truncate if too long
        src_indices = src_indices[:self.max_length]
        trg_indices = trg_indices[:self.max_length]
        
        return {
            'src': torch.LongTensor(src_indices),
            'trg': torch.LongTensor(trg_indices),
            'src_text': src_text,
            'trg_text': trg_text
        }


def collate_fn(batch, pad_idx=0):
    """
    Custom collate function để pad sequences trong batch
    """
    src_batch = [item['src'] for item in batch]
    trg_batch = [item['trg'] for item in batch]
    
    # Get lengths before padding
    src_lengths = torch.LongTensor([len(s) for s in src_batch])
    
    # Pad sequences
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    trg_padded = pad_sequence(trg_batch, batch_first=True, padding_value=pad_idx)
    
    return {
        'src': src_padded,
        'trg': trg_padded,
        'src_lengths': src_lengths
    }


def create_dataloaders(data_dir, src_vocab, trg_vocab, config):
    """
    Tạo DataLoader cho train/val/test sets
    """
    print(f"Loading dataset từ {data_dir}...")
    dataset = load_from_disk(data_dir)
    
    # Filter theo max_length (word level)
    def length_filter(example):
        en_len = len(example["en"].split())
        vi_len = len(example["vi"].split())
        return en_len <= config["max_length"] - 2 and vi_len <= config["max_length"] - 2
    
    print("Filtering theo max_length...")
    train_data = dataset["train"].filter(length_filter, desc="Filter train")
    val_data = dataset["validation"].filter(length_filter, desc="Filter val")
    test_data = dataset["test"].filter(length_filter, desc="Filter test")
    
    print(f"✓ Train: {len(train_data):,}, Val: {len(val_data):,}, Test: {len(test_data):,}")
    
    # Create datasets
    train_dataset = TranslationDataset(train_data, src_vocab, trg_vocab, config["max_length"])
    val_dataset = TranslationDataset(val_data, src_vocab, trg_vocab, config["max_length"])
    test_dataset = TranslationDataset(test_data, src_vocab, trg_vocab, config["max_length"])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, src_vocab.pad_idx),
        num_workers=0,
        pin_memory=True
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
    
    print(f"✓ DataLoaders created (batch_size={config['batch_size']})")
    return train_loader, val_loader, test_loader


def build_vocabularies(data_dir, min_freq=2, save_dir=None):
    """
    Build word-level vocabularies từ training data
    """
    print("="*50)
    print("Building Word-level Vocabularies")
    print("="*50)
    
    dataset = load_from_disk(data_dir)
    train_data = dataset["train"]
    
    # Build English vocabulary
    src_vocab = Vocabulary(min_freq=min_freq, lang="en")
    src_vocab.build_vocab([item["en"] for item in train_data])
    
    print()  # Empty line
    
    # Build Vietnamese vocabulary
    trg_vocab = Vocabulary(min_freq=min_freq, lang="vi")
    trg_vocab.build_vocab([item["vi"] for item in train_data])
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        src_vocab.save(os.path.join(save_dir, "src_vocab.pkl"))
        trg_vocab.save(os.path.join(save_dir, "trg_vocab.pkl"))
    
    print("="*50)
    return src_vocab, trg_vocab


def build_bpe_vocabularies(data_or_dir, vocab_size=16000, save_dir=None):
    """
    Build BPE vocabularies dùng SentencePiece
    
    Args:
        data_or_dir: dataset object (dict/DatasetDict) hoặc đường dẫn dataset
        vocab_size: kích thước vocab BPE
        save_dir: thư mục lưu model
    Returns:
        src_vocab: BPEVocabulary cho English
        trg_vocab: BPEVocabulary cho Vietnamese
    """
    if not HAS_SENTENCEPIECE:
        raise RuntimeError("Cần cài sentencepiece: pip install sentencepiece")
    
    print("="*50)
    print("Building BPE Vocabularies (SentencePiece)")
    print("="*50)
    
    if save_dir is None:
        save_dir = "./vocab"
    os.makedirs(save_dir, exist_ok=True)
    
    # Hỗ trợ cả dataset object và đường dẫn
    if isinstance(data_or_dir, str):
        dataset = load_from_disk(data_or_dir)
        train_data = dataset["train"]
    elif isinstance(data_or_dir, dict):
        train_data = data_or_dir["train"]
    else:
        train_data = data_or_dir["train"]
    
    # Build English BPE
    src_vocab = BPEVocabulary(vocab_size=vocab_size, lang="en")
    src_sentences = [item["en"].lower() for item in train_data]  # lowercase cho EN
    src_vocab.train(src_sentences, model_prefix=os.path.join(save_dir, "bpe_en"))
    src_vocab.save(os.path.join(save_dir, "src_bpe_vocab.pkl"))
    
    print()
    
    # Build Vietnamese BPE
    trg_vocab = BPEVocabulary(vocab_size=vocab_size, lang="vi")
    trg_sentences = [item["vi"] for item in train_data]
    trg_vocab.train(trg_sentences, model_prefix=os.path.join(save_dir, "bpe_vi"))
    trg_vocab.save(os.path.join(save_dir, "trg_bpe_vocab.pkl"))
    
    print("="*50)
    return src_vocab, trg_vocab


def load_bpe_vocabularies(vocab_dir):
    """
    Load BPE vocabularies đã train sẵn
    """
    src_vocab = BPEVocabulary.load(
        os.path.join(vocab_dir, "src_bpe_vocab.pkl"),
        os.path.join(vocab_dir, "bpe_en.model")
    )
    trg_vocab = BPEVocabulary.load(
        os.path.join(vocab_dir, "trg_bpe_vocab.pkl"),
        os.path.join(vocab_dir, "bpe_vi.model")
    )
    return src_vocab, trg_vocab


if __name__ == "__main__":
    # Test vocabulary building
    from config_baseline import BASELINE_CONFIG
    
    data_dir = "../split_dataset"
    
    if os.path.exists(data_dir):
        # Test word-level vocab
        src_vocab, trg_vocab = build_vocabularies(
            data_dir, 
            min_freq=2, 
            save_dir="./vocab"
        )
        
        # Test encoding/decoding
        test_en = "This is a test sentence."
        test_vi = "Đây là một câu thử nghiệm."
        
        print(f"\n--- Test Word-level Encoding/Decoding ---")
        print(f"English: {test_en}")
        print(f"Tokens:  {tokenize_english(test_en)}")
        print(f"Encoded: {src_vocab.encode(test_en)}")
        print(f"Decoded: {src_vocab.decode(src_vocab.encode(test_en))}")
        
        print(f"\nVietnamese: {test_vi}")
        print(f"Tokens:  {tokenize_vietnamese(test_vi)}")
        print(f"Encoded: {trg_vocab.encode(test_vi)}")
        print(f"Decoded: {trg_vocab.decode(trg_vocab.encode(test_vi))}")
        
        # Test BPE vocab nếu có sentencepiece
        if HAS_SENTENCEPIECE:
            print(f"\n--- Test BPE Encoding/Decoding ---")
            src_bpe, trg_bpe = build_bpe_vocabularies(data_dir, vocab_size=16000, save_dir="./vocab")
            
            print(f"\nEnglish BPE: {test_en}")
            print(f"Encoded: {src_bpe.encode(test_en)}")
            print(f"Decoded: {src_bpe.decode(src_bpe.encode(test_en))}")
            
            print(f"\nVietnamese BPE: {test_vi}")
            print(f"Encoded: {trg_bpe.encode(test_vi)}")
            print(f"Decoded: {trg_bpe.decode(trg_bpe.encode(test_vi))}")
    else:
        print(f"Data directory not found: {data_dir}")
        print("Please run split_data.py first!")
