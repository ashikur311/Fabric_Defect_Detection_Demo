import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from model_utils import DEFAULT_MODEL_PATH, load_model, predict  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"

# ---------- Class information shown to the user ----------
CLASS_INFO = {
    "good": {
        "label": "Good – No Defect",
        "icon": "✅",
        "color": "#16a34a",
        "about": "The fabric surface looks clean with a regular weave.",
        "action": "Pass – fabric can go to cutting.",
    },
    "hole": {
        "label": "Hole",
        "icon": "🕳️",
        "color": "#dc2626",
        "about": "An opening or tear in the fabric, usually from broken needles, sharp machine parts or snagging.",
        "action": "Reject / cut out this area. Check loom and needles for damage.",
    },
    "stain": {
        "label": "Stain",
        "icon": "💧",
        "color": "#ea580c",
        "about": "Oil, dirt or colour spot on the surface, usually from machine oil or handling.",
        "action": "Send for cleaning / spot removal. Check machine lubrication points.",
    },
    "missing_yarn": {
        "label": "Missing Yarn",
        "icon": "🧵",
        "color": "#d97706",
        "about": "One or more threads are missing, leaving a thin line across the fabric.",
        "action": "Mark the line and grade with the 4-point system. Check yarn breakage on the loom.",
    },
    "needle_mark": {
        "label": "Needle Mark",
        "icon": "📍",
        "color": "#9333ea",
        "about": "Small punctures or marks, usually in a line, caused by damaged or blunt needles.",
        "action": "Replace damaged needles. Grade fabric depending on how visible the marks are.",
    },
}
LOW_CONFIDENCE = 0.60   # below this we tell the user the model is unsure

st.set_page_config(page_title="Fabric Defect Detection", page_icon="🧵", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.result-card {border-radius: 14px; padding: 18px 22px; color: white; margin-bottom: 14px;}
.result-card h2 {margin: 0; font-size: 1.9rem; color: white;}
.result-card p {margin: 4px 0 0 0; font-size: 1.05rem; opacity: .95;}
.bar-row {display: flex; align-items: center; margin: 6px 0; font-size: .95rem;}
.bar-name {width: 120px;}
.bar-bg {flex: 1; background: rgba(128,128,128,.18); border-radius: 6px; height: 18px; margin: 0 10px;}
.bar-fill {height: 18px; border-radius: 6px;}
.bar-val {width: 52px; text-align: right; font-variant-numeric: tabular-nums;}
.info-box {border-left: 4px solid; padding: 8px 14px; border-radius: 4px;
           background: rgba(128,128,128,.08); margin-bottom: 8px;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    return load_model()


def decode_upload(file) -> np.ndarray:
    data = np.frombuffer(file.getvalue(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_REDUCED_COLOR_2)   # half size: faster for phone photos
    if img is None:
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def confidence_bars(preds):
    rows = ""
    for cls, prob in preds:
        color = CLASS_INFO[cls]["color"]
        rows += (f'<div class="bar-row"><span class="bar-name">{CLASS_INFO[cls]["icon"]} {cls}</span>'
                 f'<div class="bar-bg"><div class="bar-fill" style="width:{prob*100:.1f}%;background:{color}"></div></div>'
                 f'<span class="bar-val">{prob:.1%}</span></div>')
    st.markdown(rows, unsafe_allow_html=True)


def show_result(file, img, preds):
    top_cls, top_prob = preds[0]
    info = CLASS_INFO[top_cls]
    col_img, col_res = st.columns([1, 1.2], gap="large")

    with col_img:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=file.name, width="stretch")

    with col_res:
        status = "PASS" if top_cls == "good" else "DEFECT FOUND"
        st.markdown(
            f'<div class="result-card" style="background:{info["color"]}">'
            f'<p>{status}</p><h2>{info["icon"]} {info["label"]}</h2>'
            f'<p>Confidence: <b>{top_prob:.1%}</b></p></div>',
            unsafe_allow_html=True)

        if top_prob < LOW_CONFIDENCE:
            st.warning(f"The model is not very sure (below {LOW_CONFIDENCE:.0%}). "
                       f"Second guess: **{preds[1][0]}** ({preds[1][1]:.0%}). Please check manually.")

        st.markdown("**Confidence for each class**")
        confidence_bars(preds)

        st.markdown(f'<div class="info-box" style="border-color:{info["color"]}">'
                    f'<b>What is it?</b> {info["about"]}</div>'
                    f'<div class="info-box" style="border-color:{info["color"]}">'
                    f'<b>Suggested action:</b> {info["action"]}</div>',
                    unsafe_allow_html=True)


# ---------------- Page ----------------
st.title("🧵 Fabric Defect Detection")
st.caption("Upload a fabric photo – the AI model (MobileNetV2) classifies it as good or one of 4 defect types.")

if not DEFAULT_MODEL_PATH.exists():
    st.error("Model not found. Train it first with:  `python src/train.py`")
    st.stop()

model, ckpt = get_model()

with st.sidebar:
    st.header("About the model")
    st.metric("Test accuracy", "81.1%", help="On 359 images the model never saw during training")
    st.write(f"**Model:** MobileNetV2 (transfer learning)")
    st.write(f"**Input size:** {ckpt['img_size']}×{ckpt['img_size']} px")
    st.write("**Classes:**")
    for c in ckpt["classes"]:
        st.write(f"{CLASS_INFO[c]['icon']} {CLASS_INFO[c]['label']}")
    st.divider()


tab_check, tab_model = st.tabs(["🔍 Check fabric", "📊 Model performance"])

with tab_check:
    files = st.file_uploader("Choose fabric image(s) from your computer",
                             type=["jpg", "jpeg", "png", "bmp", "webp"],
                             accept_multiple_files=True)
    if not files:
        st.info("👆 Click **Browse files** and select one or more photos (Desktop, Downloads, any folder).")
    else:
        results = []
        for f in files:
            img = decode_upload(f)
            if img is None:
                st.error(f"Could not read {f.name}")
                continue
            preds = predict(model, ckpt, img)
            results.append((f, img, preds))

        # Summary when several images are uploaded
        if len(results) > 1:
            n_good = sum(1 for _, _, p in results if p[0][0] == "good")
            c1, c2, c3 = st.columns(3)
            c1.metric("Images checked", len(results))
            c2.metric("✅ Good", n_good)
            c3.metric("❌ Defective", len(results) - n_good)
            st.dataframe(
                [{"Image": f.name,
                  "Result": CLASS_INFO[p[0][0]]["label"],
                  "Confidence": f"{p[0][1]:.1%}",
                  "Status": "PASS" if p[0][0] == "good" else "DEFECT"} for f, _, p in results],
                width="stretch", hide_index=True)
            st.divider()

        for i, (f, img, preds) in enumerate(results):
            show_result(f, img, preds)
            if i < len(results) - 1:
                st.divider()

with tab_model:
    st.subheader("How well does the model work?")
    report = RESULTS_DIR / "test_report.txt"
    c1, c2 = st.columns(2)
    if (RESULTS_DIR / "confusion_matrix.png").exists():
        c1.image(str(RESULTS_DIR / "confusion_matrix.png"),
                 caption="Confusion matrix – rows: actual, columns: predicted", width="stretch")
    if (RESULTS_DIR / "training_curves.png").exists():
        c2.image(str(RESULTS_DIR / "training_curves.png"),
                 caption="Training progress (accuracy & loss per epoch)", width="stretch")
    if report.exists():
        with st.expander("Detailed test report (precision / recall per class)"):
            st.code(report.read_text(encoding="utf-8"))
    st.markdown("""
**Reading the results**
- Strong classes: **good**, **needle mark**, **hole**, **stain** (82–88% correct).
- Weakest class: **missing yarn** (66%) – thin missing threads become very small at 224×224 pixels.
""")
