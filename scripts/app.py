"""
Road Damage Detector — Streamlit Web App
========================================
Upload a road photo -> YOLOv8n detects damage -> red boxes + circles drawn
-> results table printed on screen.

Run:
    streamlit run scripts/app.py
"""
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

MODEL_PATH = Path(r"E:\RoadDamageDetector-master\runs\pretrained\rdd2022-yolov8n-best.pt")
LEGACY_MODEL_PATH = Path(r"E:\RoadDamageDetector-master\runs\rdd2022_india_yolov8n\weights\best.pt")
FINETUNED_MODEL_PATH = Path(r"E:\RoadDamageDetector-master\runs\finetune_india_416\weights\best.pt")
MODEL_PATHS = {
    "finetuned": FINETUNED_MODEL_PATH,
    "pretrained": MODEL_PATH,
    "legacy": LEGACY_MODEL_PATH,
}
IMGSZ_MAP = {"finetuned": 640, "pretrained": 640, "legacy": 320}

# Human-friendly names for the 4 damage classes (handles both models' naming)
CLASS_INFO = {
    "D00_longitudinal_crack": ("D00 Longitudinal Crack", (255, 0, 0)),
    "D10_transverse_crack": ("D10 Transverse Crack", (255, 140, 0)),
    "D20_alligator_crack": ("D20 Alligator Crack", (220, 0, 60)),
    "D40_pothole": ("D40 Pothole", (255, 0, 160)),
    "longitudinal_crack": ("Longitudinal Crack", (255, 0, 0)),
    "transverse_crack": ("Transverse Crack", (255, 140, 0)),
    "alligator_crack": ("Alligator Crack", (220, 0, 60)),
    "pothole": ("Pothole", (255, 0, 160)),
}
RED = (255, 0, 0)

st.set_page_config(
    page_title="Road Damage Detector",
    page_icon="🛣️",
    layout="wide",
)


@st.cache_resource
def load_model(model_key: str):
    """Load the selected model once (cached across reruns)."""
    return YOLO(str(MODEL_PATHS[model_key]))


def get_font(size=16):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_detections(img: Image.Image, boxes, names, conf_thres: float):
    """Draw RED rectangle + RED circle on each detection. Returns (image, rows)."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    font = get_font(16)
    rows = []
    n = 0
    for b in boxes:
        conf = float(b.conf[0])
        if conf < conf_thres:
            continue
        n += 1
        cls_id = int(b.cls[0])
        raw_name = names[cls_id]
        label, _ = CLASS_INFO.get(raw_name, (raw_name, RED))
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]

        # 1) red rectangle around the damage
        draw.rectangle([x1, y1, x2, y2], outline=RED, width=4)

        # 2) red circle (marker) at the damage center
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        r = 10
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=RED, width=3)

        # 3) label banner
        text = f"{label}  {conf:.2f}"
        tw = draw.textlength(text, font=font)
        draw.rectangle([x1, max(0, y1 - 24), x1 + tw + 8, max(24, y1)], fill=RED)
        draw.text((x1 + 4, max(4, y1 - 22)), text, fill=(255, 255, 255), font=font)

        rows.append({
            "#": n,
            "Damage Type": label,
            "Confidence": round(conf, 2),
            "Box (x1,y1,x2,y2)": f"({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})",
        })
    return img, rows


# ───────────────────────── UI ─────────────────────────
st.title("🛣️ Road Damage Detector")
st.markdown(
    "Upload a road photo — the AI (**YOLOv8n**, trained by you on RDD2022 India) "
    "finds cracks & potholes and **draws red boxes + circles** around them."
)

with st.sidebar:
    st.header("⚙️ Settings")
    model_key = st.radio(
        "🧠 Model",
        options=["finetuned", "pretrained", "legacy"],
        format_func=lambda k: {
            "finetuned": "🚀 Fine-tuned India model (Best — mAP 44%)",
            "pretrained": "⭐ Pretrained RDD2022 (mAP 38%)",
            "legacy": "🧠 Your first CPU model (mAP 7%)",
        }[k],
        help="Fine-tuned: pretrained + 2hr training on your India data. "
             "Best accuracy on Indian roads.",
    )
    IMGSZ = IMGSZ_MAP[model_key]
    conf_thres = st.slider("Confidence threshold", 0.02, 0.90, 0.15, 0.01,
                           help="Lower = more detections (but more false alarms). "
                                "Real damage often scores 0.10-0.35, so if you see "
                                "nothing, slide down to 0.10.")
    st.divider()
    st.caption(f"Input size: **{IMGSZ}px** (auto-set per model)")
    st.caption("Classes: longitudinal · transverse · alligator crack · pothole")

uploaded = st.file_uploader(
    "📤 Upload road image(s)", type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded and st.button("🔍 Analyze Damage", type="primary", width='stretch'):
    try:
        model = load_model(model_key)
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()

    for up in uploaded:
        st.subheader(f"📷 {up.name}")
        pil = Image.open(up).convert("RGB")

        with st.spinner("Analyzing..."):
            result = model.predict(source=pil, imgsz=IMGSZ, conf=conf_thres,
                                   device="cpu", verbose=False)[0]

        annotated, rows = draw_detections(pil, result.boxes, model.names, conf_thres)

        col_img, col_res = st.columns([3, 2])
        with col_img:
            st.image(annotated, caption="Detection result (red = damage)", width='stretch')
        with col_res:
            if rows:
                total = len(rows)
                st.success(f"🔴 **{total} damage instance(s) detected!**")
                counts = {}
                for r in rows:
                    counts[r["Damage Type"]] = counts.get(r["Damage Type"], 0) + 1
                for name, cnt in counts.items():
                    st.metric(label=name, value=cnt)
                st.markdown("**Detailed results:**")
                st.dataframe(rows, width='stretch', hide_index=True)
            else:
                st.info("✅ No damage detected above the confidence threshold.\n\n"
                        "Try lowering the slider in the sidebar, or the road may be fine!")
        st.divider()

    if rows:
        st.balloons()

elif not uploaded:
    st.info("👆 Upload one or more road photos, then click **Analyze Damage**.")
