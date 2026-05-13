# -*- coding: utf-8 -*-
"""
Human Evaluation — Bước 1: Lấy mẫu từ test set và dịch bằng model
Author: Khiem
Run:  python human_evaluation/sample_for_eval.py

Script này sẽ:
1. Random sample 100 câu từ test set (stratified theo độ dài)
2. Dịch bằng Transformer model (EN→VI và VI→EN)
3. Lưu thành eval_samples.json cho annotator chấm điểm
"""

import sys, os
import json
import random
import torch

# === Setup path ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from transformer.config_transformer import TRANSFORMER_CONFIG
from transformer.transformer_model import build_transformer, count_parameters
from transformer.shared_bpe_vocab import load_shared_bpe

# ============================================================
# CONFIG
# ============================================================
NUM_SAMPLES = 100           # Tổng số câu cần sample
BEAM_SIZE = 4               # Beam size cho translation
LENGTH_PENALTY = 0.6        # Length penalty
RANDOM_SEED = 42            # Seed cho reproducibility

# Paths
VOCAB_DIR = os.path.join(BASE_DIR, "vocab")
CKPT_ENVI = os.path.join(BASE_DIR, "transformer_checkpoints", "best_model.pt")
CKPT_VIEN = os.path.join(BASE_DIR, "transformer_checkpoints_vi2en", "best_model.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "human_evaluation")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "eval_samples.json")


# ============================================================
# Translation function
# ============================================================
def translate_sentence(model, sentence, vocab, device, beam_size=4, length_penalty=0.6):
    """Dịch một câu dùng beam search."""
    model.eval()
    with torch.no_grad():
        src_indices = vocab.encode(sentence)
        src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)

        output_indices = model.beam_search_decode(
            src_tensor,
            sos_idx=vocab.sos_idx,
            eos_idx=vocab.eos_idx,
            pad_idx=vocab.pad_idx,
            max_len=200,
            beam_size=beam_size,
            length_penalty=length_penalty,
        )
        return vocab.decode(output_indices)


# ============================================================
# Main
# ============================================================
def main():
    random.seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "annotations"), exist_ok=True)

    print("=" * 60)
    print("  HUMAN EVALUATION — SAMPLE GENERATOR")
    print("=" * 60)

    # --- Load vocab ---
    print("\n[1/5] Loading Shared BPE vocabulary...")
    shared_vocab = load_shared_bpe(VOCAB_DIR)
    print(f"       Vocab size: {len(shared_vocab):,}")

    # --- Load dataset ---
    print("\n[2/5] Loading test dataset...")
    try:
        from datasets import load_dataset
        full_dataset = load_dataset("ncduy/mt-en-vi")
        test_data = list(full_dataset["test"])
        print(f"       Test set: {len(test_data):,} samples")
    except Exception as e:
        print(f"  ERROR: Không load được dataset: {e}")
        print("  Thử load từ local split_dataset/...")
        try:
            from datasets import load_from_disk
            dataset = load_from_disk(os.path.join(BASE_DIR, "split_dataset"))
            test_data = list(dataset["test"])
            print(f"       Test set: {len(test_data):,} samples")
        except Exception as e2:
            print(f"  ERROR: {e2}")
            print("  Không tìm được test data. Thoát.")
            return

    # --- Stratified sampling theo độ dài ---
    print(f"\n[3/5] Stratified sampling {NUM_SAMPLES} câu...")

    # Phân loại theo độ dài source (English)
    short = []   # 5-15 words
    medium = []  # 16-30 words
    long = []    # 31+ words

    for item in test_data:
        en_words = len(item["en"].split())
        if 5 <= en_words <= 15:
            short.append(item)
        elif 16 <= en_words <= 30:
            medium.append(item)
        elif en_words > 30:
            long.append(item)

    print(f"       Short  (5-15 words):  {len(short):,}")
    print(f"       Medium (16-30 words): {len(medium):,}")
    print(f"       Long   (31+ words):   {len(long):,}")

    # Sample theo tỷ lệ: 40 short, 40 medium, 20 long
    n_short = min(40, len(short))
    n_medium = min(40, len(medium))
    n_long = min(NUM_SAMPLES - n_short - n_medium, len(long))

    sampled = []
    sampled += random.sample(short, n_short)
    sampled += random.sample(medium, n_medium)
    sampled += random.sample(long, n_long)

    # Shuffle để annotator không bị bias theo thứ tự
    random.shuffle(sampled)

    print(f"       Sampled: {len(sampled)} câu ({n_short} short + {n_medium} medium + {n_long} long)")

    # --- Load models ---
    config = TRANSFORMER_CONFIG.copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = device

    print(f"\n[4/5] Loading Transformer models... (device: {device})")

    # EN→VI model
    model_envi = None
    if os.path.exists(CKPT_ENVI):
        model_envi = build_transformer(len(shared_vocab), len(shared_vocab), config)
        ckpt = torch.load(CKPT_ENVI, map_location=device, weights_only=False)
        model_envi.load_state_dict(ckpt["model_state_dict"])
        model_envi.to(device)
        model_envi.eval()
        print(f"       [OK] EN→VI model loaded (Epoch {ckpt.get('epoch', '?')})")
    else:
        print(f"       [SKIP] EN→VI checkpoint not found: {CKPT_ENVI}")

    # VI→EN model
    model_vien = None
    if os.path.exists(CKPT_VIEN):
        model_vien = build_transformer(len(shared_vocab), len(shared_vocab), config)
        ckpt = torch.load(CKPT_VIEN, map_location=device, weights_only=False)
        model_vien.load_state_dict(ckpt["model_state_dict"])
        model_vien.to(device)
        model_vien.eval()
        print(f"       [OK] VI→EN model loaded (Epoch {ckpt.get('epoch', '?')})")
    else:
        print(f"       [SKIP] VI→EN checkpoint not found: {CKPT_VIEN}")

    if model_envi is None and model_vien is None:
        print("\n  ERROR: Không có model nào được load. Thoát.")
        return

    # --- Translate ---
    print(f"\n[5/5] Translating {len(sampled)} câu...")

    eval_samples = []
    for i, item in enumerate(sampled):
        en_text = item["en"]
        vi_text = item["vi"]
        en_word_count = len(en_text.split())

        sample = {
            "id": i + 1,
            "source_en": en_text,
            "reference_vi": vi_text,
            "source_word_count": en_word_count,
            "length_group": "short" if en_word_count <= 15 else ("medium" if en_word_count <= 30 else "long"),
        }

        # EN→VI translation
        if model_envi is not None:
            try:
                pred_vi = translate_sentence(model_envi, en_text, shared_vocab, device, BEAM_SIZE, LENGTH_PENALTY)
                sample["prediction_envi"] = pred_vi
            except Exception as e:
                sample["prediction_envi"] = f"[ERROR: {e}]"
                print(f"  Warning: EN→VI failed for sample {i+1}: {e}")

        # VI→EN translation
        if model_vien is not None:
            try:
                pred_en = translate_sentence(model_vien, vi_text, shared_vocab, device, BEAM_SIZE, LENGTH_PENALTY)
                sample["prediction_vien"] = pred_en
            except Exception as e:
                sample["prediction_vien"] = f"[ERROR: {e}]"
                print(f"  Warning: VI→EN failed for sample {i+1}: {e}")

        eval_samples.append(sample)

        if (i + 1) % 10 == 0:
            print(f"       Translated {i+1}/{len(sampled)} câu...")

    # --- Save ---
    output_data = {
        "metadata": {
            "total_samples": len(eval_samples),
            "beam_size": BEAM_SIZE,
            "length_penalty": LENGTH_PENALTY,
            "random_seed": RANDOM_SEED,
            "has_envi": model_envi is not None,
            "has_vien": model_vien is not None,
            "length_distribution": {
                "short": n_short,
                "medium": n_medium,
                "long": n_long,
            },
            "scoring_guide": {
                "adequacy": {
                    "5": "All meaning — Toàn bộ nghĩa được dịch đầy đủ",
                    "4": "Most meaning — Hầu hết nghĩa được dịch, thiếu chi tiết nhỏ",
                    "3": "Much meaning — Nhiều nghĩa được dịch nhưng thiếu vài phần quan trọng",
                    "2": "Little meaning — Chỉ dịch đúng một phần nhỏ",
                    "1": "None — Hoàn toàn sai nghĩa hoặc vô nghĩa",
                },
                "fluency": {
                    "5": "Flawless — Hoàn hảo, như người bản ngữ viết",
                    "4": "Good — Tốt, có thể có lỗi rất nhỏ không ảnh hưởng",
                    "3": "Non-native — Hiểu được nhưng nghe không tự nhiên",
                    "2": "Disfluent — Khó đọc, nhiều lỗi ngữ pháp",
                    "1": "Incomprehensible — Không đọc được",
                },
            },
        },
        "samples": eval_samples,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  DONE!")
    print(f"  Saved {len(eval_samples)} samples to:")
    print(f"    {OUTPUT_FILE}")
    print(f"\n  Bước tiếp theo:")
    print(f"    python human_evaluation/human_eval_interface.py")
    print(f"{'=' * 60}")

    # Show first 3 samples
    print("\n--- Preview (3 câu đầu) ---")
    for s in eval_samples[:3]:
        print(f"\n  [{s['id']}] ({s['length_group']})")
        print(f"    EN:     {s['source_en'][:120]}")
        print(f"    Ref VI: {s['reference_vi'][:120]}")
        if "prediction_envi" in s:
            print(f"    Pred:   {s['prediction_envi'][:120]}")


if __name__ == "__main__":
    main()
