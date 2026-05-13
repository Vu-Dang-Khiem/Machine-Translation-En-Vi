# -*- coding: utf-8 -*-
"""
Human Evaluation — Bước 2: Giao diện Streamlit cho Annotator chấm điểm
Author: Khiem
Run:  streamlit run human_evaluation/human_eval_interface.py

Giao diện web để annotator:
- Xem source, reference, prediction
- Cho điểm Adequacy (1-5) và Fluency (1-5)
- Lưu tự động, có thể dừng và tiếp tục
- Theo dõi tiến độ chấm
"""

import sys, os
import json
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import streamlit as st

# ============================================================
# CONFIG
# ============================================================
EVAL_DIR = os.path.join(BASE_DIR, "human_evaluation")
SAMPLES_FILE = os.path.join(EVAL_DIR, "eval_samples.json")
ANNOTATIONS_DIR = os.path.join(EVAL_DIR, "annotations")


# ============================================================
# Load / Save functions
# ============================================================
@st.cache_data
def load_samples():
    """Load eval samples JSON (cached)."""
    if not os.path.exists(SAMPLES_FILE):
        st.error(f"❌ File không tồn tại: {SAMPLES_FILE}")
        st.info("👉 Chạy `python human_evaluation/sample_for_eval.py` trước!")
        st.stop()

    with open(SAMPLES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["metadata"], data["samples"]


def get_annotation_path(annotator, direction):
    """Path file annotation."""
    return os.path.join(ANNOTATIONS_DIR, f"{annotator}_{direction}.json")


def load_annotations(annotator, direction):
    """Load annotations đã lưu (nếu có)."""
    path = get_annotation_path(annotator, direction)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {int(k): v for k, v in data.get("annotations", {}).items()}
    return {}


def save_annotations(annotator, direction, annotations, total_samples):
    """Lưu annotations ra file."""
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    path = get_annotation_path(annotator, direction)

    data = {
        "annotator": annotator,
        "direction": direction,
        "total_samples": total_samples,
        "completed": len(annotations),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "annotations": {str(k): v for k, v in annotations.items()},
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# Streamlit Page Config
# ============================================================
st.set_page_config(
    page_title="Human Evaluation — MT EN↔VI",
    page_icon="🧑‍🔬",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #059669 0%, #0d9488 50%, #0891b2 100%);
        border-radius: 16px;
        padding: 24px 20px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 8px 30px rgba(5, 150, 105, 0.3);
    }
    .main-header h1 { margin: 0; font-size: 2em; }
    .main-header p { margin: 4px 0 0; opacity: 0.85; }

    .score-guide-box {
        background: linear-gradient(135deg, #f0fdf4, #ecfdf5);
        border: 1px solid #bbf7d0;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 16px;
        font-size: 0.9em;
        line-height: 1.7;
    }
    .score-guide-box h4 { margin: 0 0 6px; color: #059669; }

    .text-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
        font-size: 1.05em;
        line-height: 1.6;
    }
    .text-box.prediction {
        background: #fef3c7;
        border-color: #fbbf24;
    }

    .text-label {
        font-weight: 700;
        font-size: 0.85em;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
        color: #64748b;
    }
    .text-label.pred-label { color: #d97706; }

    .status-saved {
        background: #d1fae5;
        border: 1px solid #6ee7b7;
        border-radius: 8px;
        padding: 8px 14px;
        color: #065f46;
        font-weight: 600;
        text-align: center;
    }

    div[data-testid="stProgress"] > div {
        height: 8px !important;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Header
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🧑‍🔬 Human Evaluation</h1>
    <p>Machine Translation EN ↔ VI &nbsp;|&nbsp; Adequacy + Fluency (1-5)</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# Sidebar — Settings
# ============================================================
with st.sidebar:
    st.header("⚙️ Cài đặt")

    annotator = st.text_input(
        "👤 Tên Annotator",
        value="annotator_1",
        help="VD: khiem, friend1, annotator_2",
    )

    direction = st.selectbox(
        "🔄 Chiều dịch đánh giá",
        options=["envi", "vien"],
        format_func=lambda x: "EN → VI" if x == "envi" else "VI → EN",
    )

    st.divider()

    # Scoring guide
    st.markdown("""
    ### 📋 Hướng dẫn chấm

    **Adequacy (Đầy đủ nghĩa):**
    | Điểm | Ý nghĩa |
    |------|---------|
    | 5 | Toàn bộ nghĩa |
    | 4 | Hầu hết nghĩa |
    | 3 | Nhiều nhưng thiếu |
    | 2 | Chỉ một phần nhỏ |
    | 1 | Sai hoàn toàn |

    **Fluency (Trôi chảy):**
    | Điểm | Ý nghĩa |
    |------|---------|
    | 5 | Hoàn hảo |
    | 4 | Tốt |
    | 3 | Hiểu được |
    | 2 | Khó đọc |
    | 1 | Không đọc được |
    """)


# ============================================================
# Load data
# ============================================================
metadata, samples = load_samples()

# Load existing annotations
if "annotations" not in st.session_state:
    st.session_state.annotations = load_annotations(annotator, direction)

if "current_idx" not in st.session_state:
    # Tìm câu chưa chấm đầu tiên
    st.session_state.current_idx = 0
    for i, s in enumerate(samples):
        if s["id"] not in st.session_state.annotations:
            st.session_state.current_idx = i
            break

# Reload annotations nếu đổi annotator/direction
annot_key = f"{annotator}_{direction}"
if st.session_state.get("_annot_key") != annot_key:
    st.session_state.annotations = load_annotations(annotator, direction)
    st.session_state._annot_key = annot_key
    # Tìm câu chưa chấm
    st.session_state.current_idx = 0
    for i, s in enumerate(samples):
        if s["id"] not in st.session_state.annotations:
            st.session_state.current_idx = i
            break


# ============================================================
# Progress bar
# ============================================================
total = len(samples)
done = len(st.session_state.annotations)
pct = done / total if total > 0 else 0

col_prog1, col_prog2 = st.columns([3, 1])
with col_prog1:
    st.progress(pct, text=f"✅ Đã chấm: {done}/{total} ({pct*100:.0f}%)")
with col_prog2:
    st.metric("Còn lại", f"{total - done} câu")


# ============================================================
# Navigation
# ============================================================
idx = st.session_state.current_idx

col_nav1, col_nav2, col_nav3, col_nav4 = st.columns([1, 1, 2, 1])

with col_nav1:
    if st.button("⬅️ Trước", use_container_width=True, disabled=(idx <= 0)):
        st.session_state.current_idx = max(0, idx - 1)
        st.rerun()

with col_nav2:
    if st.button("➡️ Sau", use_container_width=True, disabled=(idx >= total - 1)):
        st.session_state.current_idx = min(total - 1, idx + 1)
        st.rerun()

with col_nav3:
    new_idx = st.number_input(
        "Nhảy tới câu:",
        min_value=1, max_value=total,
        value=idx + 1,
        step=1,
        label_visibility="collapsed",
    )
    if new_idx - 1 != idx:
        st.session_state.current_idx = new_idx - 1
        st.rerun()

with col_nav4:
    if st.button("⏭️ Câu chưa chấm", use_container_width=True):
        for i in range(idx + 1, total):
            if samples[i]["id"] not in st.session_state.annotations:
                st.session_state.current_idx = i
                st.rerun()
                break
        else:
            # Tìm từ đầu
            for i in range(0, idx):
                if samples[i]["id"] not in st.session_state.annotations:
                    st.session_state.current_idx = i
                    st.rerun()
                    break
            else:
                st.toast("🎉 Đã chấm hết tất cả!")

st.divider()


# ============================================================
# Display current sample
# ============================================================
idx = st.session_state.current_idx
sample = samples[idx]
sample_id = sample["id"]

# Determine what to show based on direction
if direction == "envi":
    source_text = sample["source_en"]
    ref_text = sample["reference_vi"]
    pred_text = sample.get("prediction_envi", "[Không có bản dịch EN→VI]")
    source_label = "📝 SOURCE (English)"
    ref_label = "📖 REFERENCE (Vietnamese — từ dataset)"
    dir_label = "EN → VI"
else:
    source_text = sample["reference_vi"]
    ref_text = sample["source_en"]
    pred_text = sample.get("prediction_vien", "[Không có bản dịch VI→EN]")
    source_label = "📝 SOURCE (Vietnamese)"
    ref_label = "📖 REFERENCE (English — từ dataset)"
    dir_label = "VI → EN"

# Sample info
is_scored = sample_id in st.session_state.annotations
status_emoji = "✅" if is_scored else "⏳"
st.markdown(
    f"### {status_emoji} Câu {idx + 1}/{total}  —  ID #{sample_id}  |  "
    f"**{sample['length_group'].upper()}**  |  {sample['source_word_count']} words  |  {dir_label}"
)

# Text display
st.markdown(f'<div class="text-label">{source_label}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="text-box">{source_text}</div>', unsafe_allow_html=True)

col_ref, col_pred = st.columns(2)

with col_ref:
    st.markdown(f'<div class="text-label">{ref_label}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="text-box">{ref_text}</div>', unsafe_allow_html=True)

with col_pred:
    st.markdown('<div class="text-label pred-label">🤖 MODEL PREDICTION — CẦN ĐÁNH GIÁ</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="text-box prediction">{pred_text}</div>', unsafe_allow_html=True)

st.divider()


# ============================================================
# Scoring
# ============================================================
prev = st.session_state.annotations.get(sample_id, {})

col_score1, col_score2 = st.columns(2)

with col_score1:
    adequacy = st.slider(
        "⭐ Adequacy — Đầy đủ nghĩa (1-5)",
        min_value=1, max_value=5,
        value=prev.get("adequacy", 4),
        step=1,
        help="5 = Toàn bộ nghĩa | 4 = Hầu hết | 3 = Thiếu vài phần | 2 = Một phần | 1 = Sai hết",
    )

with col_score2:
    fluency = st.slider(
        "✨ Fluency — Trôi chảy (1-5)",
        min_value=1, max_value=5,
        value=prev.get("fluency", 4),
        step=1,
        help="5 = Hoàn hảo | 4 = Tốt | 3 = Hiểu được | 2 = Khó đọc | 1 = Vô nghĩa",
    )

note = st.text_input(
    "📝 Ghi chú (tùy chọn)",
    value=prev.get("note", ""),
    placeholder="VD: Sai tên riêng, thiếu giới từ, thừa từ...",
)


# ============================================================
# Submit button
# ============================================================
col_btn1, col_btn2 = st.columns([3, 1])

with col_btn1:
    if st.button(
        "✅ Chấm điểm & Tiếp theo",
        type="primary",
        use_container_width=True,
    ):
        # Save score
        st.session_state.annotations[sample_id] = {
            "adequacy": adequacy,
            "fluency": fluency,
            "note": note.strip(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Save to file
        save_annotations(
            annotator, direction,
            st.session_state.annotations, total,
        )

        st.toast(f"✅ Đã lưu câu #{sample_id}: Adequacy={adequacy}, Fluency={fluency}")

        # Go to next unscored
        if idx < total - 1:
            # Find next unscored
            next_idx = idx + 1
            for i in range(idx + 1, total):
                if samples[i]["id"] not in st.session_state.annotations:
                    next_idx = i
                    break
            st.session_state.current_idx = next_idx
        st.rerun()

with col_btn2:
    if st.button("💾 Lưu (không chuyển)", use_container_width=True):
        st.session_state.annotations[sample_id] = {
            "adequacy": adequacy,
            "fluency": fluency,
            "note": note.strip(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_annotations(
            annotator, direction,
            st.session_state.annotations, total,
        )
        st.toast(f"💾 Đã lưu câu #{sample_id}")
        st.rerun()


# ============================================================
# Summary stats (sidebar bottom)
# ============================================================
with st.sidebar:
    st.divider()
    st.header("📊 Thống kê")

    if st.session_state.annotations:
        all_adeq = [v["adequacy"] for v in st.session_state.annotations.values()]
        all_flu = [v["fluency"] for v in st.session_state.annotations.values()]

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Adequacy TB", f"{sum(all_adeq)/len(all_adeq):.2f}")
        with col_s2:
            st.metric("Fluency TB", f"{sum(all_flu)/len(all_flu):.2f}")

        # Score distribution
        st.markdown("**Phân bố điểm Adequacy:**")
        adeq_dist = {i: all_adeq.count(i) for i in range(1, 6)}
        st.bar_chart(adeq_dist)

        st.markdown("**Phân bố điểm Fluency:**")
        flu_dist = {i: all_flu.count(i) for i in range(1, 6)}
        st.bar_chart(flu_dist)
    else:
        st.info("Chưa có điểm nào")

    st.divider()
    st.caption("Built by **Khiem** | Transformer MT EN↔VI")
