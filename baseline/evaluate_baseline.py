# BLEU, METEOR, TER score + sample translations
"""
Evaluation script cho baseline model
- Tính BLEU score
- Tính METEOR score
- Tính TER score
- Sample translations
- Hỗ trợ BPE vocabulary
"""

import torch
from sacrebleu.metrics import BLEU, TER
from nltk.translate.meteor_score import meteor_score
import nltk
from tqdm import tqdm
import os
import argparse

# Download wordnet cho METEOR (chỉ cần lần đầu)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

from config_baseline import BASELINE_CONFIG
from seq2seq_model import build_model
from data_utils import Vocabulary, create_dataloaders, load_bpe_vocabularies


def translate_sentence(model, sentence, src_vocab, trg_vocab, device, max_len=100):
    """
    Dịch một câu tiếng Anh sang tiếng Việt
    """
    model.eval()
    
    with torch.no_grad():
        # Encode input
        src_indices = src_vocab.encode(sentence)
        src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
        src_lengths = torch.LongTensor([len(src_indices)]).to(device)
        
        # Translate (Beam Search)
        translations, attention_weights = model.beam_search_translate(
            src_tensor, src_lengths, max_len, beam_width=5
        )
        
        # Decode output
        output_indices = translations[0].cpu().tolist()
        translated = trg_vocab.decode(output_indices)
        
    return translated, None


def calculate_metrics(model, dataloader, src_vocab, trg_vocab, device, num_samples=None):
    """
    Tính BLEU, METEOR, TER score trên dataset
    """
    model.eval()
    
    predictions = []
    references = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Calculating Metrics")):
            if num_samples and i * dataloader.batch_size >= num_samples:
                break
                
            src = batch['src'].to(device)
            src_lengths = batch['src_lengths'].to(device)
            trg = batch['trg']
            
            # Beam search chỉ hỗ trợ batch_size=1, nên dịch từng câu
            for j in range(src.size(0)):
                actual_len = src_lengths[j].item()
                src_j = src[j, :actual_len].unsqueeze(0)
                src_len_j = src_lengths[j].unsqueeze(0)
                
                translations, _ = model.beam_search_translate(src_j, src_len_j, beam_width=5)
                
                pred_indices = translations[0].cpu().tolist()
                ref_indices = trg[j].tolist()
                
                pred_text = trg_vocab.decode(pred_indices)
                ref_text = trg_vocab.decode(ref_indices)
                
                predictions.append(pred_text)
                references.append(ref_text)
    
    # === BLEU ===
    bleu = BLEU()
    bleu_score = bleu.corpus_score(predictions, [references])
    
    # === METEOR ===
    meteor_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            meteor_scores.append(0.0)
        else:
            score = meteor_score([ref_tokens], pred_tokens)
            meteor_scores.append(score)
    avg_meteor = sum(meteor_scores) / len(meteor_scores) * 100  # Nhân 100 cho dễ đọc
    
    # === TER ===
    ter = TER()
    ter_score = ter.corpus_score(predictions, [references])
    
    return bleu_score, avg_meteor, ter_score, predictions, references


def show_samples(model, dataloader, src_vocab, trg_vocab, device, n=5):
    """
    Hiển thị một số mẫu dịch
    """
    model.eval()
    
    print("\n" + "="*80)
    print("SAMPLE TRANSLATIONS")
    print("="*80)
    
    dataset = dataloader.dataset
    
    for i in range(min(n, len(dataset))):
        item = dataset[i]
        
        src_text = item['src_text']
        ref_text = item['trg_text']
        
        pred_text, _ = translate_sentence(model, src_text, src_vocab, trg_vocab, device)
        
        print(f"\n[Sample {i+1}]")
        print(f"  Source (EN): {src_text[:200]}...")
        print(f"  Reference (VI): {ref_text[:200]}...")
        print(f"  Prediction (VI): {pred_text[:200]}...")
        print("-"*80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline Seq2Seq model")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="../split_dataset",
                        help="Path to dataset")
    parser.add_argument("--vocab_dir", type=str, default="./vocab",
                        help="Path to vocabularies")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples to evaluate (0 for all)")
    parser.add_argument("--show_samples", type=int, default=5,
                        help="Number of sample translations to show")
    
    args = parser.parse_args()
    config = BASELINE_CONFIG
    device = torch.device(config["device"])
    use_bpe = config.get("use_bpe", False)
    
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Tokenization: {'BPE' if use_bpe else 'Word-level'}")
    
    # Load vocabularies
    if use_bpe:
        src_vocab, trg_vocab = load_bpe_vocabularies(args.vocab_dir)
    else:
        src_vocab = Vocabulary.load(os.path.join(args.vocab_dir, "src_vocab.pkl"))
        trg_vocab = Vocabulary.load(os.path.join(args.vocab_dir, "trg_vocab.pkl"))
    
    print(f"Source vocab: {len(src_vocab)}, Target vocab: {len(trg_vocab)}")
    
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(len(src_vocab), len(trg_vocab), config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    valid_loss = checkpoint.get('valid_loss', None)
    if valid_loss is not None:
        print(f"Valid loss: {valid_loss:.4f}")
    else:
        print("Valid loss: unknown")
    
    # Create test dataloader
    _, _, test_loader = create_dataloaders(args.data_dir, src_vocab, trg_vocab, config)
    
    # Calculate all metrics
    print("\nCalculating BLEU, METEOR, TER scores...")
    num_samples = args.num_samples if args.num_samples > 0 else None
    bleu_score, meteor_avg, ter_score, preds, refs = calculate_metrics(
        model, test_loader, src_vocab, trg_vocab, device, num_samples
    )
    
    print("\n" + "="*60)
    print(" EVALUATION RESULTS")
    print("="*60)
    print(f"\n BLEU Score: {bleu_score.score:.2f}")
    print(f"   Precisions: {[f'{p:.1f}' for p in bleu_score.precisions]}")
    print(f"   BP: {bleu_score.bp:.4f}")
    print(f"   Ratio: {bleu_score.sys_len / bleu_score.ref_len:.4f}")
    print(f"\n METEOR Score: {meteor_avg:.2f}")
    print(f"\n TER Score: {ter_score.score:.2f}")
    print("="*60)
    
    # Show sample translations
    if args.show_samples > 0:
        show_samples(model, test_loader, src_vocab, trg_vocab, device, args.show_samples)
    
    # Interactive mode
    print("\n" + "="*50)
    print("INTERACTIVE TRANSLATION")
    print("Enter English sentences to translate (type 'quit' to exit)")
    print("="*50)
    
    while True:
        try:
            sentence = input("\nEnglish: ").strip()
            if sentence.lower() == 'quit':
                break
            if not sentence:
                continue
                
            translation, _ = translate_sentence(model, sentence, src_vocab, trg_vocab, device)
            print(f"Vietnamese: {translation}")
            
        except KeyboardInterrupt:
            break
    
    print("\nGoodbye!")


if __name__ == "__main__":
    main()
