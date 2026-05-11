# -*- coding: utf-8 -*-
"""
Demo Transformer EN <-> VI Translation (Google Translate style)
Author: Khiem
Run:  python demo_transformer.py
"""

# === Fix Windows console encoding ===
import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import time
import math
import tempfile
import gradio as gr

# ============================================================
# Setup paths
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from transformer.config_transformer import TRANSFORMER_CONFIG
from transformer.transformer_model import build_transformer, count_parameters
from transformer.shared_bpe_vocab import load_shared_bpe

# ============================================================
# Paths config
# ============================================================
TRANSFORMER_CKPT_ENVI = os.path.join(BASE_DIR, "transformer_checkpoints", "best_model.pt")
TRANSFORMER_CKPT_VIEN = os.path.join(BASE_DIR, "transformer_checkpoints_vi2en", "best_model.pt")
VOCAB_DIR = os.path.join(BASE_DIR, "vocab")

# ============================================================
# Global state
# ============================================================
_model_envi = None
_model_vien = None
_shared_vocab = None
_device = None
_config = None
_checkpoint_envi = None
_checkpoint_vien = None
_n_params = 0

def load_model():
    """Load Transformer models + Shared BPE vocab."""
    global _model_envi, _model_vien, _shared_vocab, _device, _config
    global _checkpoint_envi, _checkpoint_vien, _n_params

    print("=" * 60)
    print("  LOADING TRANSFORMER MODELS")
    print("=" * 60)

    # 1. Shared BPE vocab
    print("\n[1/4] Loading Shared BPE vocabulary...")
    _shared_vocab = load_shared_bpe(VOCAB_DIR)
    print(f"       Vocab size: {len(_shared_vocab):,} subwords")

    # 2. Config
    _config = TRANSFORMER_CONFIG.copy()
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _config["device"] = _device

    # 3. Load EN->VI model
    print("[2/4] Loading EN->VI model...")
    if not os.path.exists(TRANSFORMER_CKPT_ENVI):
        raise FileNotFoundError(
            f"Checkpoint not found: {TRANSFORMER_CKPT_ENVI}\n"
            "Copy best_model.pt into transformer_checkpoints/"
        )
    _model_envi = build_transformer(len(_shared_vocab), len(_shared_vocab), _config)
    _n_params = count_parameters(_model_envi)
    _checkpoint_envi = torch.load(TRANSFORMER_CKPT_ENVI, map_location=_device, weights_only=False)
    _model_envi.load_state_dict(_checkpoint_envi["model_state_dict"])
    _model_envi.eval()
    print(f"       [OK] EN->VI loaded (Epoch {_checkpoint_envi.get('epoch', '?')})")

    # 4. Load VI->EN model (optional)
    print("[3/4] Loading VI->EN model...")
    if os.path.exists(TRANSFORMER_CKPT_VIEN):
        _model_vien = build_transformer(len(_shared_vocab), len(_shared_vocab), _config)
        _checkpoint_vien = torch.load(TRANSFORMER_CKPT_VIEN, map_location=_device, weights_only=False)
        _model_vien.load_state_dict(_checkpoint_vien["model_state_dict"])
        _model_vien.eval()
        print(f"       [OK] VI->EN loaded (Epoch {_checkpoint_vien.get('epoch', '?')})")
    else:
        print(f"       [SKIP] VI->EN checkpoint not found")
        print(f"       Train VI->EN model first, then copy to transformer_checkpoints_vi2en/")

    # Summary
    eval_res = _checkpoint_envi.get("eval_results", {})
    print(f"\n  Parameters: {_n_params:,} ({_n_params/1e6:.1f}M)")
    if eval_res:
        print(f"  EN->VI BLEU:   {eval_res.get('bleu_score', '-')}")
    print(f"  VI->EN:  {'Available' if _model_vien else 'Not available'}")
    print(f"  Device: {_device}")
    print("=" * 60)


# ============================================================
# Translation core
# ============================================================
import re

def _split_sentences(text):
    """Tach van ban thanh cac cau rieng le."""
    # Split theo dau cham, cham hoi, cham than (giu nguyen dau)
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _translate_single(text, model, beam_size=4, length_penalty=0.6):
    """Dich 1 cau ngan (< 200 tokens)."""
    with torch.no_grad():
        src_indices = _shared_vocab.encode(text)
        src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(_device)

        output_indices = model.beam_search_decode(
            src_tensor,
            sos_idx=_shared_vocab.sos_idx,
            eos_idx=_shared_vocab.eos_idx,
            pad_idx=_shared_vocab.pad_idx,
            beam_size=int(beam_size),
            length_penalty=float(length_penalty),
            max_len=200,
        )
        return _shared_vocab.decode(output_indices)


def _translate_text(text, model, beam_size=4, length_penalty=0.6):
    """Translate text using given model. Tu dong tach cau dai."""
    if not text or not text.strip():
        return ""

    text = text.strip()

    # Kiem tra do dai token
    tokens = _shared_vocab.encode(text, add_special=False)
    if len(tokens) <= 180:
        return _translate_single(text, model, beam_size, length_penalty)

    # Cau dai: tach thanh nhieu cau nho roi dich tung cau
    sentences = _split_sentences(text)
    results = []
    for sent in sentences:
        translated = _translate_single(sent, model, beam_size, length_penalty)
        results.append(translated)
    return " ".join(results)


def translate_en_to_vi(text, beam_size=4, length_penalty=0.6):
    """Translate English -> Vietnamese."""
    return _translate_text(text, _model_envi, beam_size, length_penalty)


def translate_vi_to_en(text, beam_size=4, length_penalty=0.6):
    """Translate Vietnamese -> English."""
    if _model_vien is None:
        return None
    return _translate_text(text, _model_vien, beam_size, length_penalty)


# ============================================================
# Gradio handler
# ============================================================
def do_translate(text, direction, beam_size, length_penalty):
    """Main translation handler called by Gradio."""
    if not text or not text.strip():
        return "", ""

    start = time.time()

    if direction == "EN -> VI":
        result = translate_en_to_vi(text, beam_size, length_penalty)
    else:
        result = translate_vi_to_en(text, beam_size, length_penalty)

    elapsed = time.time() - start

    if result is None:
        return (
            "[Model VI->EN chua duoc train]\n"
            "Ban can train them 1 Transformer voi VI la source, EN la target.\n"
            "Sau do dat checkpoint vao thu muc tuong ung."
        ), "Model VI->EN not available"

    n_src = len(text.split())
    n_trg = len(result.split())
    info = f"{elapsed:.2f}s  |  {n_src} words -> {n_trg} words  |  beam={int(beam_size)}"

    return result, info


def swap_languages(src_text, trg_text, direction):
    """Swap source <-> target (like Google Translate)."""
    new_dir = "VI -> EN" if direction == "EN -> VI" else "EN -> VI"
    if new_dir == "EN -> VI":
        return trg_text, "", new_dir, gr.update(value="English"), gr.update(value="Vietnamese")
    else:
        return trg_text, "", new_dir, gr.update(value="Vietnamese"), gr.update(value="English")


# ============================================================
# Document translation handler
# ============================================================
def _extract_text_from_file(file_path):
    """Extract text from uploaded file (.txt, .docx, .pdf)."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    elif ext == ".docx":
        try:
            from docx import Document
            doc = Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n".join(paragraphs)
        except ImportError:
            return "[ERROR] Can cai python-docx: pip install python-docx"

    elif ext == ".pdf":
        try:
            import PyPDF2
            text_parts = []
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            return "\n".join(text_parts)
        except ImportError:
            return "[ERROR] Can cai PyPDF2: pip install PyPDF2"

    else:
        return f"[ERROR] Khong ho tro dinh dang {ext}. Chi ho tro: .txt, .docx, .pdf"


def do_translate_document(file, direction, beam_size, length_penalty):
    """Translate an uploaded document."""
    if file is None:
        return "Vui long upload file truoc.", "", None

    start = time.time()

    # Extract text
    file_path = file.name if hasattr(file, "name") else file
    original_name = os.path.basename(file_path)
    text = _extract_text_from_file(file_path)

    if text.startswith("[ERROR]"):
        return text, "", None

    if not text.strip():
        return "File khong co noi dung text.", "", None

    # Translate paragraph by paragraph
    paragraphs = text.split("\n")
    translated_parts = []
    total_words = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            translated_parts.append("")
            continue

        total_words += len(para.split())
        if direction == "EN -> VI":
            result = translate_en_to_vi(para, beam_size, length_penalty)
        else:
            result = translate_vi_to_en(para, beam_size, length_penalty)

        if result is None:
            translated_parts.append("[Model not available]")
        else:
            translated_parts.append(result)

    translated_text = "\n".join(translated_parts)
    elapsed = time.time() - start

    # Save result to file for download
    name_no_ext = os.path.splitext(original_name)[0]
    suffix = "_vi" if direction == "EN -> VI" else "_en"
    output_filename = f"{name_no_ext}{suffix}_translated.txt"
    output_path = os.path.join(tempfile.gettempdir(), output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(translated_text)

    n_paras = len([p for p in paragraphs if p.strip()])
    info = (f"{elapsed:.1f}s  |  {total_words:,} words  |  "
            f"{n_paras} paragraphs  |  beam={int(beam_size)}")

    return translated_text, info, output_path


# ============================================================
# Initialize model at startup
# ============================================================
print("\nInitializing Transformer Translator...\n")
load_model()

eval_results = _checkpoint_envi.get("eval_results", {})
bleu = eval_results.get("bleu_score", 45.40)
meteor = eval_results.get("meteor_score", 66.09)
ter = eval_results.get("ter_score", 39.24)
best_val_loss = _checkpoint_envi.get("best_val_loss", 1.1726)
val_ppl = math.exp(best_val_loss) if isinstance(best_val_loss, (int, float)) else 3.23
brevity_penalty = eval_results.get("brevity_penalty", 0.9904)

# VI->EN metrics (neu co)
if _checkpoint_vien:
    vien_eval = _checkpoint_vien.get("eval_results", {})
    vien_bleu = vien_eval.get("bleu_score", "N/A")
else:
    vien_bleu = "Not trained"

# ============================================================
# CSS
# ============================================================
CUSTOM_CSS = """
/* ---- Global ---- */
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
    font-family: 'Segoe UI', 'Inter', sans-serif !important;
}

/* ---- Header ---- */
.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #a855f7 100%);
    border-radius: 18px;
    padding: 32px 28px 24px;
    color: #fff;
    text-align: center;
    box-shadow: 0 10px 40px rgba(79, 70, 229, 0.35);
    margin-bottom: 6px;
}
.hero h1 { font-size: 2.1em; margin: 0 0 4px; font-weight: 800; letter-spacing: -0.5px; }
.hero .sub { opacity: .88; font-size: 1.05em; margin-top: 2px; }

/* ---- Metric pills ---- */
.pills {
    display: flex; justify-content: center; gap: 14px;
    margin-top: 18px; flex-wrap: wrap;
}
.pill {
    background: rgba(255,255,255,.18);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(255,255,255,.22);
    border-radius: 14px;
    padding: 8px 22px;
    text-align: center;
    min-width: 80px;
}
.pill .val { font-size: 1.45em; font-weight: 700; display: block; }
.pill .lbl { font-size: .72em; text-transform: uppercase; letter-spacing: 1.2px; opacity: .8; }

/* ---- Language buttons (Google Translate style) ---- */
.lang-btn {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 24px !important;
    font-weight: 700 !important;
    font-size: 1.05em !important;
    letter-spacing: 0.3px !important;
    padding: 10px 28px !important;
    cursor: default !important;
    opacity: 1 !important;
    pointer-events: none;
}

/* ---- Swap button ---- */
.swap-btn {
    min-width: 48px !important;
    max-width: 48px !important;
    min-height: 48px !important;
    max-height: 48px !important;
    border-radius: 50% !important;
    font-size: 1.4em !important;
    padding: 0 !important;
    box-shadow: 0 4px 12px rgba(0,0,0,.15) !important;
    margin: 0 18px !important;
    transition: transform 0.3s ease, box-shadow 0.3s ease !important;
}
.swap-btn:hover {
    transform: rotate(180deg) !important;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4) !important;
}

/* ---- Textareas ---- */
textarea { font-size: 1.08em !important; line-height: 1.65 !important; }

/* ---- Footer ---- */
.foot { text-align: center; padding: 14px; color: #999; font-size: .82em; }

/* ---- Document tab ---- */
.doc-upload {
    border: 2px dashed rgba(79, 70, 229, 0.3) !important;
    border-radius: 16px !important;
    transition: border-color 0.3s ease !important;
}
.doc-upload:hover {
    border-color: rgba(79, 70, 229, 0.6) !important;
}
.doc-result textarea {
    font-size: 1.02em !important;
    line-height: 1.7 !important;
}
"""

# ============================================================
# Build Gradio UI
# ============================================================
with gr.Blocks(title="Transformer Translator EN-VI") as demo:

    # ---- Header ----
    gr.HTML(f"""
    <div class="hero">
        <h1>Transformer Translator</h1>
        <p class="sub">
            English &harr; Vietnamese &nbsp;|&nbsp;
            Attention Is All You Need (Vaswani et al., 2017)
        </p>
        <p style="font-size:.88em; opacity:.75; margin-top:2px;">
            d_model={_config['d_model']} &middot; {_config['num_encoder_layers']}L &middot;
            {_config['num_heads']}H &middot; {_n_params/1e6:.1f}M params
        </p>
        <div class="pills">
            <div class="pill"><span class="val">{bleu:.1f}</span><span class="lbl">BLEU</span></div>
            <div class="pill"><span class="val">{meteor:.1f}</span><span class="lbl">METEOR</span></div>
            <div class="pill"><span class="val">{ter:.1f}</span><span class="lbl">TER &darr;</span></div>
            <div class="pill"><span class="val">{val_ppl:.2f}</span><span class="lbl">PPL</span></div>
            <div class="pill"><span class="val">{brevity_penalty:.4f}</span><span class="lbl">BP</span></div>
        </div>
    </div>
    """)

    # ---- Direction state (hidden) ----
    direction = gr.Radio(
        choices=["EN -> VI", "VI -> EN"],
        value="EN -> VI",
        visible=False,
    )

    # ---- Language bar: [English] [⇄] [Vietnamese] ----
    with gr.Row(equal_height=True):
        src_lang_btn = gr.Button("English", elem_classes=["lang-btn"], size="sm", scale=3)
        swap_btn = gr.Button("⇄", elem_classes=["swap-btn"], size="sm", scale=1, min_width=60)
        trg_lang_btn = gr.Button("Vietnamese", elem_classes=["lang-btn"], size="sm", scale=3)

    # ---- Tabs: Text Translation | Document Translation ----
    with gr.Tabs():

        # ========== TAB 1: Text Translation ==========
        with gr.TabItem("✏️ Text Translation"):
            # ---- Translation area (side by side) ----
            with gr.Row(equal_height=True):
                src_box = gr.Textbox(
                    label="English (Source)",
                    placeholder="Type or paste text here...",
                    lines=7, max_lines=14,
                    scale=1,
                )
                trg_box = gr.Textbox(
                    label="Vietnamese (Translation)",
                    lines=7, max_lines=14,
                    interactive=False,
                    scale=1,
                )

            # ---- Info bar ----
            info_bar = gr.Textbox(show_label=False, interactive=False, max_lines=1)

            # ---- Controls + Translate button ----
            with gr.Row():
                beam_slider = gr.Slider(1, 10, value=4, step=1, label="Beam Size")
                lp_slider = gr.Slider(0.0, 2.0, value=0.6, step=0.1, label="Length Penalty")
                translate_btn = gr.Button("Translate", variant="primary", size="lg", scale=2)

            # ---- Examples ----
            gr.Examples(
                examples=[
                    ["The weather is beautiful today.", "EN -> VI"],
                    ["I love learning new languages.", "EN -> VI"],
                    ["Artificial intelligence will transform how we live and work.", "EN -> VI"],
                    ["The doctor prescribed antibiotics for the infection.", "EN -> VI"],
                    ["Climate change is one of the biggest challenges facing humanity.", "EN -> VI"],
                    ["She has been studying medicine for six years.", "EN -> VI"],
                    ["The students are preparing for their final exams next week.", "EN -> VI"],
                    ["Technology is changing the world rapidly.", "EN -> VI"],
                ],
                inputs=[src_box, direction],
                label="Example sentences",
            )

        # ========== TAB 2: Document Translation ==========
        with gr.TabItem("📄 Document Translation"):
            gr.HTML("""
            <div style="padding:12px 16px; background:linear-gradient(135deg,#ede9fe,#e0e7ff);
                        border-radius:12px; margin-bottom:12px;">
                <p style="margin:0; color:#4338ca; font-weight:600;">
                    📁 Upload a document to translate the entire content
                </p>
                <p style="margin:4px 0 0; color:#6366f1; font-size:.9em;">
                    Supported formats: .txt, .docx, .pdf
                </p>
            </div>
            """)

            with gr.Row():
                doc_upload = gr.File(
                    label="Upload Document",
                    file_types=[".txt", ".docx", ".pdf"],
                    elem_classes=["doc-upload"],
                    scale=1,
                )
                with gr.Column(scale=1):
                    doc_direction = gr.Radio(
                        choices=["EN -> VI", "VI -> EN"],
                        value="EN -> VI",
                        label="Translation Direction",
                    )
                    with gr.Row():
                        doc_beam = gr.Slider(1, 10, value=4, step=1, label="Beam Size")
                        doc_lp = gr.Slider(0.0, 2.0, value=0.6, step=0.1, label="Length Penalty")
                    doc_translate_btn = gr.Button(
                        "🔄 Translate Document", variant="primary", size="lg"
                    )

            doc_info = gr.Textbox(show_label=False, interactive=False, max_lines=1)

            doc_output = gr.Textbox(
                label="Translation Result",
                lines=12, max_lines=25,
                interactive=False,
                elem_classes=["doc-result"],
            )

            doc_download = gr.File(label="📥 Download translated file", interactive=False)

    # ---- Model Info (Accordion) ----
    with gr.Accordion("Model Information", open=False):
        # Support both old format (history dict) and new format (top-level keys)
        history = _checkpoint_envi.get("history", {})
        train_losses = _checkpoint_envi.get("train_losses", history.get("train_loss", []))
        val_losses = _checkpoint_envi.get("val_losses", history.get("val_loss", []))
        train_ppl_data = history.get("train_ppl", [])
        val_ppl_data = history.get("val_ppl", [])
        best_val = _checkpoint_envi.get("best_val_loss", "N/A")

        gr.HTML(f"""
        <div style="padding:16px;">
        <h3 style="color:#4f46e5;">Architecture</h3>
        <table style="width:100%;border-collapse:collapse;margin:10px 0;">
            <tr style="background:#f8f9fa;"><td style="padding:8px 14px;border:1px solid #e5e7eb;font-weight:600;">Component</td>
                <td style="padding:8px 14px;border:1px solid #e5e7eb;font-weight:600;">Value</td></tr>
            <tr><td style="padding:8px 14px;border:1px solid #e5e7eb;">Model</td>
                <td style="padding:8px 14px;border:1px solid #e5e7eb;">Transformer (Vaswani et al., 2017)</td></tr>
            <tr style="background:#f8f9fa;"><td style="padding:8px 14px;border:1px solid #e5e7eb;">d_model</td>
                <td style="padding:8px 14px;border:1px solid #e5e7eb;">{_config['d_model']}</td></tr>
            <tr><td style="padding:8px 14px;border:1px solid #e5e7eb;">Heads</td>
                <td style="padding:8px 14px;border:1px solid #e5e7eb;">{_config['num_heads']}</td></tr>
            <tr style="background:#f8f9fa;"><td style="padding:8px 14px;border:1px solid #e5e7eb;">Encoder / Decoder Layers</td>
                <td style="padding:8px 14px;border:1px solid #e5e7eb;">{_config['num_encoder_layers']} / {_config['num_decoder_layers']}</td></tr>
            <tr><td style="padding:8px 14px;border:1px solid #e5e7eb;">FFN dim (d_ff)</td>
                <td style="padding:8px 14px;border:1px solid #e5e7eb;">{_config['d_ff']}</td></tr>
            <tr style="background:#f8f9fa;"><td style="padding:8px 14px;border:1px solid #e5e7eb;">Parameters</td>
                <td style="padding:8px 14px;border:1px solid #e5e7eb;">{_n_params:,} ({_n_params/1e6:.1f}M)</td></tr>
            <tr><td style="padding:8px 14px;border:1px solid #e5e7eb;">Vocabulary</td>
                <td style="padding:8px 14px;border:1px solid #e5e7eb;">Shared BPE ({len(_shared_vocab):,} subwords)</td></tr>
            <tr style="background:#f8f9fa;"><td style="padding:8px 14px;border:1px solid #e5e7eb;">Training Data</td>
                <td style="padding:8px 14px;border:1px solid #e5e7eb;">2M EN-VI pairs</td></tr>
            <tr><td style="padding:8px 14px;border:1px solid #e5e7eb;">Best Val Loss</td>
                <td style="padding:8px 14px;border:1px solid #e5e7eb;">{best_val}</td></tr>
            <tr style="background:#f8f9fa;"><td style="padding:8px 14px;border:1px solid #e5e7eb;">Best Val PPL</td>
                <td style="padding:8px 14px;border:1px solid #e5e7eb;">{val_ppl:.2f}</td></tr>
            <tr><td style="padding:8px 14px;border:1px solid #e5e7eb;">Brevity Penalty</td>
                <td style="padding:8px 14px;border:1px solid #e5e7eb;">{brevity_penalty:.4f}</td></tr>
        </table>

        <h3 style="color:#4f46e5;margin-top:20px;">Evaluation (Test set, 500 samples)</h3>
        <table style="width:100%;border-collapse:collapse;margin:10px 0;">
            <tr style="background:#f8f9fa;">
                <td style="padding:8px 14px;border:1px solid #e5e7eb;font-weight:600;">Metric</td>
                <td style="padding:8px 14px;border:1px solid #e5e7eb;font-weight:600;">Transformer</td>
                <td style="padding:8px 14px;border:1px solid #e5e7eb;font-weight:600;">LSTM Baseline</td>
            </tr>
            <tr>
                <td style="padding:8px 14px;border:1px solid #e5e7eb;">BLEU</td>
                <td style="padding:8px 14px;border:1px solid #e5e7eb;color:#16a34a;font-weight:700;">{bleu:.2f}</td>
                <td style="padding:8px 14px;border:1px solid #e5e7eb;">23.61</td>
            </tr>
            <tr style="background:#f8f9fa;">
                <td style="padding:8px 14px;border:1px solid #e5e7eb;">METEOR</td>
                <td style="padding:8px 14px;border:1px solid #e5e7eb;color:#16a34a;font-weight:700;">{meteor:.2f}</td>
                <td style="padding:8px 14px;border:1px solid #e5e7eb;">-</td>
            </tr>
            <tr>
                <td style="padding:8px 14px;border:1px solid #e5e7eb;">TER (lower=better)</td>
                <td style="padding:8px 14px;border:1px solid #e5e7eb;color:#16a34a;font-weight:700;">{ter:.2f}</td>
                <td style="padding:8px 14px;border:1px solid #e5e7eb;">-</td>
            </tr>
        </table>

        <h3 style="color:#4f46e5;margin-top:20px;">Training History ({len(train_losses)} epochs)</h3>
        </div>
        """)

        # Training curves image
        if train_losses and val_losses:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
            epochs_x = range(1, len(train_losses) + 1)

            axes[0].plot(epochs_x, train_losses, "o-", color="#4f46e5", label="Train", ms=3, lw=2)
            axes[0].plot(epochs_x, val_losses, "s-", color="#f97316", label="Val", ms=3, lw=2)
            axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
            axes[0].set_title("Loss Curves"); axes[0].legend(); axes[0].grid(True, alpha=.3)

            # Compute PPL from losses if not stored
            if not train_ppl_data and train_losses:
                import math as _m
                train_ppl_data = [_m.exp(min(l, 100)) for l in train_losses]
                val_ppl_data = [_m.exp(min(l, 100)) for l in val_losses]
            if train_ppl_data and val_ppl_data:
                axes[1].plot(epochs_x, train_ppl_data, "o-", color="#4f46e5", label="Train PPL", ms=3, lw=2)
                axes[1].plot(epochs_x, val_ppl_data, "s-", color="#f97316", label="Val PPL", ms=3, lw=2)
                axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Perplexity")
                axes[1].set_title("Perplexity"); axes[1].legend(); axes[1].grid(True, alpha=.3)

            plt.tight_layout()
            curves_path = os.path.join(BASE_DIR, "_demo_curves.png")
            fig.savefig(curves_path, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            gr.Image(value=curves_path, show_label=False)

    # ---- Footer ----
    gr.HTML("""
    <div class="foot">
        Built by <b>Khiem</b> &nbsp;|&nbsp;
        Transformer &mdash; Attention Is All You Need &nbsp;|&nbsp;
        2M EN-VI pairs &nbsp;|&nbsp;
        BLEU 45.40 &bull; METEOR 66.09 &bull; TER 39.24
    </div>
    """)

    # ============================================================
    # Event bindings
    # ============================================================

    # Dynamic labels
    def update_labels(direction):
        if direction == "EN -> VI":
            return (
                gr.update(label="English (Source)"),
                gr.update(label="Vietnamese (Translation)"),
                gr.update(value="English"),
                gr.update(value="Vietnamese"),
            )
        else:
            return (
                gr.update(label="Vietnamese (Source)"),
                gr.update(label="English (Translation)"),
                gr.update(value="Vietnamese"),
                gr.update(value="English"),
            )

    direction.change(
        fn=update_labels,
        inputs=[direction],
        outputs=[src_box, trg_box, src_lang_btn, trg_lang_btn],
    )

    # Translate button
    translate_btn.click(
        fn=do_translate,
        inputs=[src_box, direction, beam_slider, lp_slider],
        outputs=[trg_box, info_bar],
    )

    # Enter key
    src_box.submit(
        fn=do_translate,
        inputs=[src_box, direction, beam_slider, lp_slider],
        outputs=[trg_box, info_bar],
    )

    # Swap button
    swap_btn.click(
        fn=swap_languages,
        inputs=[src_box, trg_box, direction],
        outputs=[src_box, trg_box, direction, src_lang_btn, trg_lang_btn],
    )

    # Document translate button
    doc_translate_btn.click(
        fn=do_translate_document,
        inputs=[doc_upload, doc_direction, doc_beam, doc_lp],
        outputs=[doc_output, doc_info, doc_download],
    )


# ============================================================
# Launch
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60, flush=True)
    print("  STARTING GRADIO SERVER...", flush=True)
    print("=" * 60, flush=True)

    app, local_url, share_url = demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=None,
        show_error=True,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(),
    )

    print("\n" + "=" * 60, flush=True)
    print(f"  LOCAL:  {local_url}", flush=True)
    if share_url:
        print(f"  PUBLIC: {share_url}", flush=True)
    else:
        print("  PUBLIC: (khong tao duoc share link)", flush=True)
    print("=" * 60, flush=True)
