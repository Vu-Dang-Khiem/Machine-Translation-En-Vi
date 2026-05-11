"""
Gradio Demo - English to Vietnamese Translator
Seq2Seq + Bi-LSTM + Bahdanau Attention (BLEU 23.61)
"""

import torch
import os
import sys
import gradio as gr

# Thêm baseline vào path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "baseline"))

from seq2seq_model import build_model
from data_utils import load_bpe_vocabularies
from config_baseline import BASELINE_CONFIG

# ============================================================
# Cấu hình đường dẫn
# ============================================================
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pt")
VOCAB_DIR = os.path.join(os.path.dirname(__file__), "vocab")

# ============================================================
# Load model và vocab
# ============================================================
def load_model():
    """Load trained model và vocabulary"""
    print("Loading vocabularies...")
    src_vocab, trg_vocab = load_bpe_vocabularies(VOCAB_DIR)
    
    config = BASELINE_CONFIG.copy()
    device = torch.device("cpu")  # Demo chạy trên CPU
    config["device"] = "cpu"
    
    print("Loading checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    
    # Cập nhật config từ checkpoint nếu có
    if "config" in checkpoint:
        saved_config = checkpoint["config"]
        config["hidden_dim"] = saved_config.get("hidden_dim", config["hidden_dim"])
        config["num_layers"] = saved_config.get("num_layers", config["num_layers"])
        config["dropout"] = saved_config.get("dropout", config["dropout"])
        config["embedding_dim"] = saved_config.get("embedding_dim", config["embedding_dim"])
        config["bidirectional"] = saved_config.get("bidirectional", config["bidirectional"])
        config["use_coverage"] = saved_config.get("use_coverage", config.get("use_coverage", False))
    
    # Tính lại encoder_hidden_dim dựa trên bidirectional từ checkpoint
    config["encoder_hidden_dim"] = (
        config["hidden_dim"] * 2 if config["bidirectional"] else config["hidden_dim"]
    )
    
    print(f"  bidirectional={config['bidirectional']}, encoder_hidden_dim={config['encoder_hidden_dim']}")
    print("Building model...")
    model = build_model(
        src_vocab_size=len(src_vocab),
        trg_vocab_size=len(trg_vocab),
        config=config,
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    
    epoch = checkpoint.get("epoch", "?")
    valid_loss = checkpoint.get("valid_loss", "?")
    print(f"Model loaded! (Epoch: {epoch}, Valid Loss: {valid_loss:.4f})")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, src_vocab, trg_vocab, device

# ============================================================
# Hàm dịch
# ============================================================
def translate(text, beam_width=5):
    """Dịch câu tiếng Anh sang tiếng Việt"""
    if not text.strip():
        return ""
    
    try:
        with torch.no_grad():
            # Encode input
            src_indices = src_vocab.encode(text.lower())
            src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
            src_lengths = torch.LongTensor([len(src_indices)]).to(device)
            
            # Beam search translate
            translations, _ = model.beam_search_translate(
                src_tensor, src_lengths, max_len=128, beam_width=beam_width
            )
            
            # Decode output
            output_indices = translations[0].cpu().tolist()
            translated = trg_vocab.decode(output_indices)
            
        return translated
    except Exception as e:
        return f"Lỗi: {str(e)}"

# ============================================================
# Load model khi khởi động
# ============================================================
print("=" * 50)
print("Initializing EN→VI Translator...")
print("=" * 50)
model, src_vocab, trg_vocab, device = load_model()

# ============================================================
# Giao diện Gradio
# ============================================================
examples = [
    ["The weather is beautiful today."],
    ["I love learning new languages."],
    ["Technology is changing the world."],
    ["What is the meaning of life?"],
    ["The man that's giving the test has serious doubts."],
]

demo = gr.Interface(
    fn=translate,
    inputs=gr.Textbox(
        label="English",
        placeholder="Nhập câu tiếng Anh cần dịch...",
        lines=3,
    ),
    outputs=gr.Textbox(
        label="Vietnamese (Tiếng Việt)",
        lines=3,
    ),
    title="🌐 English → Vietnamese Translator",
    description="""
    **Seq2Seq + Bi-LSTM + Bahdanau Attention**  
    Trained on 500K sentence pairs from TED Talks | BLEU Score: **23.61**
    """,
    examples=examples,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(share=True)  # share=True tạo link public
