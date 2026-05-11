"""
Evaluation script cho Transformer MT Model (EN → VI)
- BLEU, METEOR, TER metrics
- Beam Search translation
- Sample translations
- Interactive mode
Tương thích với cấu trúc evaluate_baseline.py
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from sacrebleu.metrics import BLEU, TER
from nltk.translate.meteor_score import meteor_score
import nltk
from tqdm import tqdm
import argparse

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

from transformer.config_transformer import TRANSFORMER_CONFIG
from transformer.transformer_model import build_transformer
from transformer.shared_bpe_vocab import load_shared_bpe
from baseline.data_utils import create_dataloaders


# ============================================================
# Translate một câu đơn
# ============================================================

def translate_sentence(
    model,
    sentence: str,
    src_vocab,
    trg_vocab,
    device,
    max_len: int = 100,
    beam_size: int = 4,
    length_penalty: float = 0.6,
) -> str:
    """
    Dịch một câu tiếng Anh sang tiếng Việt dùng Beam Search

    Args:
        model: TransformerSeq2Seq
        sentence: câu tiếng Anh
        src_vocab: BPEVocabulary (EN)
        trg_vocab: BPEVocabulary (VI)
        device: torch.device
    Returns:
        translated: câu tiếng Việt đã dịch
    """
    model.eval()

    with torch.no_grad():
        src_indices = src_vocab.encode(sentence)
        src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)  # [1, src_len]

        if beam_size <= 1:
            # Greedy decode
            output = model.greedy_decode(
                src_tensor,
                sos_idx=trg_vocab.sos_idx,
                eos_idx=trg_vocab.eos_idx,
                max_len=max_len,
            )
            output_indices = output[0].tolist()
        else:
            # Beam search decode
            output_indices = model.beam_search_decode(
                src_tensor,
                sos_idx=trg_vocab.sos_idx,
                eos_idx=trg_vocab.eos_idx,
                pad_idx=trg_vocab.pad_idx,
                max_len=max_len,
                beam_size=beam_size,
                length_penalty=length_penalty,
            )

        # Decode token indices về text
        translated = trg_vocab.decode(output_indices)

    return translated


# ============================================================
# Calculate BLEU / METEOR / TER
# ============================================================

def calculate_metrics(
    model,
    dataloader,
    src_vocab,
    trg_vocab,
    device,
    num_samples=None,
    beam_size: int = 4,
    length_penalty: float = 0.6,
):
    """
    Tính BLEU, METEOR, TER trên toàn bộ test set
    """
    model.eval()
    predictions = []
    references = []
    count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Translating"):
            src = batch['src'].to(device)
            trg = batch['trg']

            for j in range(src.size(0)):
                if num_samples and count >= num_samples:
                    break

                src_j = src[j].unsqueeze(0)  # [1, src_len]

                # Beam search
                output_indices = model.beam_search_decode(
                    src_j,
                    sos_idx=trg_vocab.sos_idx,
                    eos_idx=trg_vocab.eos_idx,
                    pad_idx=trg_vocab.pad_idx,
                    max_len=100,
                    beam_size=beam_size,
                    length_penalty=length_penalty,
                )

                ref_indices = trg[j].tolist()

                pred_text = trg_vocab.decode(output_indices)
                ref_text  = trg_vocab.decode(ref_indices)

                predictions.append(pred_text)
                references.append(ref_text)
                count += 1

            if num_samples and count >= num_samples:
                break

    # BLEU
    bleu_metric = BLEU()
    bleu_score = bleu_metric.corpus_score(predictions, [references])

    # METEOR
    meteor_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens  = ref.split()
        if pred_tokens and ref_tokens:
            meteor_scores.append(meteor_score([ref_tokens], pred_tokens))
        else:
            meteor_scores.append(0.0)
    avg_meteor = sum(meteor_scores) / len(meteor_scores) * 100

    # TER
    ter_metric = TER()
    ter_score = ter_metric.corpus_score(predictions, [references])

    return bleu_score, avg_meteor, ter_score, predictions, references


# ============================================================
# Show sample translations
# ============================================================

def show_samples(model, dataloader, src_vocab, trg_vocab, device, n=5, beam_size=4):
    model.eval()
    dataset = dataloader.dataset

    print("\n" + "=" * 80)
    print("SAMPLE TRANSLATIONS  (Transformer EN→VI)")
    print("=" * 80)

    for i in range(min(n, len(dataset))):
        item = dataset[i]
        src_text = item['src_text']
        ref_text  = item['trg_text']

        pred_text = translate_sentence(
            model, src_text, src_vocab, trg_vocab, device, beam_size=beam_size
        )

        print(f"\n[Sample {i+1}]")
        print(f"  Source  (EN): {src_text[:200]}")
        print(f"  Reference   : {ref_text[:200]}")
        print(f"  Prediction  : {pred_text[:200]}")
        print("-" * 80)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Transformer MT model")
    parser.add_argument("--checkpoint", type=str,
                        default="./transformer/checkpoints/best_model.pt")
    parser.add_argument("--data_dir",   type=str, default="./split_dataset")
    parser.add_argument("--vocab_dir",  type=str, default="./vocab")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Số câu evaluate (0 = all)")
    parser.add_argument("--show_samples", type=int, default=5)
    parser.add_argument("--beam_size",  type=int, default=4)
    parser.add_argument("--length_penalty", type=float, default=0.6)
    args = parser.parse_args()

    config = TRANSFORMER_CONFIG
    device = torch.device(config["device"])

    print("=" * 60)
    print("  TRANSFORMER EVALUATION (EN → VI)")
    print("=" * 60)
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Beam size  : {args.beam_size}")
    print()

    # Load shared BPE vocab (Paper Section 5.1)
    shared_vocab = load_shared_bpe(args.vocab_dir)
    src_vocab = shared_vocab
    trg_vocab = shared_vocab
    print(f"  Shared vocab: {len(shared_vocab):,} tokens")

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    vocab_size = len(shared_vocab)
    model = build_transformer(vocab_size, vocab_size, config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"  Loaded từ epoch {checkpoint.get('epoch', '?')} | Val loss: {checkpoint.get('val_loss', '?'):.4f}")

    # DataLoaders
    _, _, test_loader = create_dataloaders(args.data_dir, src_vocab, trg_vocab, config)

    # Metrics
    print(f"\nTính metrics trên {args.num_samples or 'tất cả'} câu...")
    num_samples = args.num_samples if args.num_samples > 0 else None

    bleu_score, meteor_avg, ter_score, preds, refs = calculate_metrics(
        model, test_loader, src_vocab, trg_vocab, device,
        num_samples=num_samples,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
    )

    print("\n" + "=" * 60)
    print("  RESULTS  –  Transformer EN→VI")
    print("=" * 60)
    print(f"  BLEU   : {bleu_score.score:.2f}")
    print(f"    Precisions: {[f'{p:.1f}' for p in bleu_score.precisions]}")
    print(f"    BP        : {bleu_score.bp:.4f}")
    print(f"  METEOR : {meteor_avg:.2f}")
    print(f"  TER    : {ter_score.score:.2f}")
    print("=" * 60)

    # Sample translations
    if args.show_samples > 0:
        show_samples(model, test_loader, src_vocab, trg_vocab, device,
                     n=args.show_samples, beam_size=args.beam_size)

    # Interactive
    print("\n" + "=" * 50)
    print("INTERACTIVE MODE – Nhập câu Anh để dịch (gõ 'quit' để thoát)")
    print("=" * 50)

    while True:
        try:
            sentence = input("\nEnglish: ").strip()
            if sentence.lower() in ('quit', 'exit', 'q'):
                break
            if not sentence:
                continue

            translation = translate_sentence(
                model, sentence, src_vocab, trg_vocab, device,
                beam_size=args.beam_size,
                length_penalty=args.length_penalty,
            )
            print(f"Vietnamese: {translation}")

        except KeyboardInterrupt:
            break

    print("\nGoodbye!")
