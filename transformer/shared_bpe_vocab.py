# -*- coding: utf-8 -*-
"""
Shared BPE Vocabulary cho Transformer (EN+VI)
Sử dụng Hugging Face Tokenizers (Rust-based, rất nhanh)

Theo paper §5.1: "byte-pair encoding, which has a SHARED source-target vocabulary"

Thay vì 2 vocab riêng (baseline), Transformer dùng 1 BPE model chung
train trên cả tiếng Anh + tiếng Việt. Điều này giúp:
  - Weight tying dễ hơn (src_embed & trg_embed cùng kích thước)
  - Model học được subword patterns chung giữa 2 ngôn ngữ
"""

import os
import torch
import json

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    HAS_TOKENIZERS = True
except ImportError:
    HAS_TOKENIZERS = False
    print("tokenizers chua cai dat - pip install tokenizers")

try:
    from datasets import load_from_disk
except ImportError:
    load_from_disk = None

from tqdm import tqdm


class SharedBPEVocabulary:
    """
    Shared BPE Vocabulary dùng Hugging Face Tokenizers cho cả EN và VI.
    Nhanh hơn SentencePiece rất nhiều (Rust backend).

    Special tokens:
        0: <pad>
        1: <sos>
        2: <eos>
        3: <unk>
    """

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.tokenizer = None  # HF Tokenizer

        # Special token indices
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

        # Special token strings
        self._special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]

    def train(self, sentences: list, model_prefix: str):
        """
        Train BPE tokenizer từ list of sentences (EN + VI trộn lẫn).

        Args:
            sentences: list of strings (cả EN và VI)
            model_prefix: path prefix để save (VD: "./vocab/shared_bpe")
        """
        if not HAS_TOKENIZERS:
            raise RuntimeError("Can cai tokenizers: pip install tokenizers")

        os.makedirs(os.path.dirname(model_prefix), exist_ok=True)

        # Lọc câu quá dài
        max_char_len = 2000
        original_len = len(sentences)
        sentences = [s for s in sentences if len(s) <= max_char_len]
        filtered = original_len - len(sentences)
        if filtered > 0:
            print(f"  Loc bo {filtered:,} cau dai >{max_char_len} ky tu")

        # Giới hạn số câu cho tốc độ training
        # 500K câu để BPE coverage đủ tốt cho cả EN+VI
        max_sentences = 2000000
        if len(sentences) > max_sentences:
            import random
            random.seed(42)
            sentences = random.sample(sentences, max_sentences)
            print(f"  Gioi han {max_sentences:,} cau cho BPE training")

        print(f"  Training BPE with HF Tokenizers ({len(sentences):,} sentences, vocab_size={self.vocab_size})...")

        # Khởi tạo BPE tokenizer
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()

        # Trainer với special tokens ở đúng vị trí (index 0,1,2,3)
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self._special_tokens,  # ["<pad>", "<sos>", "<eos>", "<unk>"]
            show_progress=True,
            min_frequency=2,
        )

        # Train từ iterator (không cần export file tạm!)
        tokenizer.train_from_iterator(sentences, trainer=trainer)

        self.tokenizer = tokenizer

        # Verify special token indices
        assert self.tokenizer.token_to_id("<pad>") == 0, "pad_idx phải = 0"
        assert self.tokenizer.token_to_id("<sos>") == 1, "sos_idx phải = 1"
        assert self.tokenizer.token_to_id("<eos>") == 2, "eos_idx phải = 2"
        assert self.tokenizer.token_to_id("<unk>") == 3, "unk_idx phải = 3"

        # Save tokenizer (1 file JSON duy nhất)
        save_path = model_prefix + ".json"
        self.tokenizer.save(save_path)

        print(f"  Shared BPE Vocab: {self.tokenizer.get_vocab_size():,} subwords")
        print(f"  Saved to {save_path}")

    def __len__(self):
        if self.tokenizer is None:
            return 0
        return self.tokenizer.get_vocab_size()

    def encode(self, sentence: str, add_special: bool = True) -> list:
        """
        Encode câu thành list of subword indices.
        Args:
            sentence: string
            add_special: thêm <sos> và <eos>
        Returns:
            list of int
        """
        if self.tokenizer is None:
            raise RuntimeError("Shared BPE model chua duoc load!")

        encoding = self.tokenizer.encode(sentence)
        indices = encoding.ids

        if add_special:
            indices = [self.sos_idx] + indices + [self.eos_idx]

        return indices

    def decode(self, indices, remove_special: bool = True) -> str:
        """
        Decode list of subword indices thành câu.
        Args:
            indices: list of int hoặc torch.Tensor
            remove_special: bỏ <pad>, <sos>, <eos>
        Returns:
            string
        """
        if self.tokenizer is None:
            raise RuntimeError("Shared BPE model chua duoc load!")

        clean_indices = []
        for idx in indices:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()

            if remove_special and idx in (self.pad_idx, self.sos_idx, self.eos_idx):
                if idx == self.eos_idx:
                    break
                continue

            clean_indices.append(idx)

        text = self.tokenizer.decode(clean_indices)
        return text

    def save_metadata(self, path: str):
        """Save vocab metadata (cho backward compatibility)."""
        meta = {
            'vocab_size': self.vocab_size,
            'type': 'shared_bpe_hf',
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        print(f"  Saved metadata to {path}")

    @classmethod
    def load_from_files(cls, metadata_path: str, tokenizer_path: str) -> 'SharedBPEVocabulary':
        """
        Load SharedBPEVocabulary từ metadata + tokenizer JSON.
        Args:
            metadata_path: đường dẫn .json metadata
            tokenizer_path: đường dẫn .json (HF Tokenizer)
        """
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        vocab = cls(vocab_size=data['vocab_size'])
        vocab.tokenizer = Tokenizer.from_file(tokenizer_path)

        print(f"  Loaded Shared BPE from {tokenizer_path} ({len(vocab):,} subwords)")
        return vocab


# ============================================================
# High-level API functions
# ============================================================

def build_shared_bpe(
    data_dir: str,
    vocab_size: int = 32000,
    save_dir: str = "./vocab",
) -> SharedBPEVocabulary:
    """
    Train shared BPE vocabulary từ dataset (EN + VI trộn lẫn).

    Args:
        data_dir: đường dẫn thư mục dataset (HuggingFace format)
        vocab_size: kích thước vocab
        save_dir: thư mục lưu model
    Returns:
        SharedBPEVocabulary
    """
    if load_from_disk is None:
        raise RuntimeError("Can cai datasets: pip install datasets")

    print("=" * 50)
    print("Building SHARED BPE Vocabulary (EN + VI)")
    print("  Using: Hugging Face Tokenizers (Rust)")
    print("=" * 50)

    os.makedirs(save_dir, exist_ok=True)

    # Load training data
    dataset = load_from_disk(data_dir)
    train_data = dataset["train"]

    # Trộn EN + VI sentences (paper §5.1: shared source-target vocab)
    print("  Tron cau EN + VI cho shared BPE...")
    sentences = []
    for item in tqdm(train_data, desc="  Collecting"):
        sentences.append(item["en"].lower())    # EN lowercase
        sentences.append(item["vi"])           # VI giữ nguyên

    print(f"  Tong: {len(sentences):,} cau (EN + VI)")

    # Train
    shared_vocab = SharedBPEVocabulary(vocab_size=vocab_size)
    model_prefix = os.path.join(save_dir, "shared_bpe")
    shared_vocab.train(sentences, model_prefix)

    # Save metadata
    shared_vocab.save_metadata(os.path.join(save_dir, "shared_bpe_meta.json"))

    print("=" * 50)
    return shared_vocab


def load_shared_bpe(vocab_dir: str) -> SharedBPEVocabulary:
    """
    Load shared BPE vocabulary đã train sẵn.

    Args:
        vocab_dir: thư mục chứa shared_bpe.json
    Returns:
        SharedBPEVocabulary
    """
    tokenizer_path = os.path.join(vocab_dir, "shared_bpe.json")
    meta_path = os.path.join(vocab_dir, "shared_bpe_meta.json")

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Khong tim thay shared BPE tokenizer: {tokenizer_path}\n"
            f"Chay build_shared_bpe() truoc!"
        )

    # Nếu chưa có metadata file, tạo mặc định
    if not os.path.exists(meta_path):
        print(f"  Metadata {meta_path} not found, loading tokenizer directly...")
        vocab = SharedBPEVocabulary()
        vocab.tokenizer = Tokenizer.from_file(tokenizer_path)
        vocab.vocab_size = vocab.tokenizer.get_vocab_size()
        print(f"  Loaded Shared BPE from {tokenizer_path} ({len(vocab):,} subwords)")
        return vocab

    return SharedBPEVocabulary.load_from_files(meta_path, tokenizer_path)


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    vocab_dir = os.path.join(os.path.dirname(__file__), '..', 'vocab')
    tokenizer_path = os.path.join(vocab_dir, "shared_bpe.json")

    if os.path.exists(tokenizer_path):
        print("Loading existing shared BPE vocab...")
        vocab = load_shared_bpe(vocab_dir)

        test_en = "The weather is beautiful today."
        test_vi = "Hom nay thoi tiet rat dep."

        print(f"\nEN: {test_en}")
        enc = vocab.encode(test_en)
        print(f"  Encoded: {enc}")
        print(f"  Decoded: {vocab.decode(enc)}")

        print(f"\nVI: {test_vi}")
        enc = vocab.encode(test_vi)
        print(f"  Encoded: {enc}")
        print(f"  Decoded: {vocab.decode(enc)}")
    else:
        print(f"Shared BPE tokenizer not found at {tokenizer_path}")
        print("Run build_shared_bpe() or training script first!")
