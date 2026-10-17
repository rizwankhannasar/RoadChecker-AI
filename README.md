# 🛣️ RoadChecker-AI

**AI-powered road damage detection** — upload a road photo and get instant red-box detections of cracks and potholes.

Built with **YOLOv8 (Ultralytics)** and a **Streamlit** web app. Fine-tuned on the **RDD2022 India** dataset.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-orange) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)

---

## ✨ Features

- 📷 **Upload photos** (or multiple) and detect damage instantly
- 🟥 **Red boxes + circles** drawn around every detected damage
- 📊 Per-class damage counts and a detailed results table
- 🎚️ Adjustable **confidence slider** (0.02 – 0.90)
- 🧠 **3 selectable models**, including a model fine-tuned on Indian roads

## 🎯 Damage Classes

| Code | Class | Description |
|------|-------|-------------|
| D00 | Longitudinal Crack | Cracks running along the road direction |
| D10 | Transverse Crack | Cracks running across the road |
| D20 | Alligator Crack | Interconnected cracks (fatigue failure) |
| D40 | Pothole | Holes / cavities in the surface |

## 📈 Model Performance

Validated on 300 held-out Indian road images:

| Model | mAP50 | Recall | Precision |
|-------|-------|--------|-----------|
| First CPU model (scratch, 3 epochs) | 0.074 | 0.086 | 0.418 |
| Pretrained RDD2022 (dronefreak) | 0.380 | 0.340 | — |
| **🚀 Fine-tuned on India (this repo)** | **0.442** | **0.463** | **0.481** |

Per-class (fine-tuned):

| Class | mAP50 | Recall |
|-------|-------|--------|
| D00 Longitudinal | 0.303 | 0.358 |
| D10 Transverse | 0.555 | 0.333 |
| D20 Alligator | 0.538 | 0.692 |
| D40 Pothole | 0.372 | 0.470 |

## 🚀 Quick Start

### 1. Install

```bash
conda create -n rdd python=3.12 -y
conda activate rdd
pip install ultralytics streamlit opencv-python pillow pi-heif
```

### 2. Run the web app

```bash
streamlit run scripts/app.py
```

Or on Windows, just double-click **`start_app.bat`**.

Then open **http://localhost:8501**, upload a road photo, and click **Analyze Damage**.

### 3. CLI inference

```bash
python scripts/predict.py path/to/road.jpg
```

## 🏋️ Retraining (optional)

Download the RDD2022 India dataset (YOLO format) from HuggingFace:

```
https://huggingface.co/datasets/aiswarya9302/RDD_2022India/resolve/main/RDD2022_India.zip
```

Extract to `dataset/extracted/`, point `dataset/extracted/data.yaml` at it, then:

```bash
# CPU fine-tune (~2 hrs) from the pretrained RDD2022 weights
python scripts/train_finetune.py 114   # 114 = minute budget
```

Best weights are written to `runs/finetune_india_416/weights/best.pt`.

## 📁 Project Structure

```
RoadChecker-AI/
├── scripts/
│   ├── app.py              # Streamlit web app (main deliverable)
│   ├── predict.py          # CLI inference
│   ├── train.py            # Scratch training
│   ├── train_finetune.py   # Fine-tune from pretrained (recommended)
│   └── test_photo.py       # Ad-hoc testing at multiple confidences
├── runs/
│   ├── finetune_india_416/ # ✅ Best model (fine-tuned on India)
│   └── pretrained/         # RDD2022 pretrained checkpoint
├── dataset/                # (git-ignored) RDD2022 India
├── images/                 # Sample road photos
└── start_app.bat           # One-click launcher (Windows)
```

## ⚙️ Tips for better detections

- Real damage often scores **0.10 – 0.35** confidence — if you see no boxes, lower the slider to **0.10**.
- The fine-tuned model **infers best at 640px** (even though it was trained at 416px).
- Well-lit, close-up photos of the road surface work far better than distant/angled shots.

## 📄 Dataset & Credits

- Dataset: [RDD2022](https://github.com/sekilab/RoadDamageDetector) (Arya et al.) — India subset
- Pretrained weights: [dronefreak/rdd2022-yolov8n](https://huggingface.co/dronefreak/rdd2022-yolov8n)
- Detection framework: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

Dataset is **not** included in this repo (≈900 MB) — download it via the link above.

## 📜 License

See [LICENSE](LICENSE).
