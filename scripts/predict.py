"""
Run inference with the trained road-damage model.

- By default runs on a few test images (unlabeled, from RDD2022 India test)
- Or pass your own image(s):
    python predict.py path/to/photo.jpg
"""
import sys
from pathlib import Path

from ultralytics import YOLO

BEST = Path(r"E:\RoadDamageDetector-master\runs\rdd2022_india_yolov8n\weights\best.pt")
TEST_DIR = Path(r"E:\RoadDamageDetector-master\dataset\extracted\test\images")
OUT_DIR = Path(r"E:\RoadDamageDetector-master\runs\predictions")
N_SAMPLE = 6


def main():
    if not BEST.exists():
        sys.exit(f"Model not found: {BEST}\nTrain first with: python train.py")

    model = YOLO(str(BEST))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) > 1:
        sources = [p for p in sys.argv[1:] if Path(p).exists()]
    else:
        if not TEST_DIR.exists():
            sys.exit(f"No test images at {TEST_DIR}; pass image paths instead.")
        sources = sorted(TEST_DIR.glob("*.jpg"))[:N_SAMPLE]

    if not sources:
        sys.exit("No input images found.")

    print(f"Running detection on {len(sources)} image(s)...")
    results = model.predict(
        source=[str(s) for s in sources],
        imgsz=320,
        conf=0.15,
        device="cpu",
        save=True,
        project=str(OUT_DIR),
        name="detect",
        exist_ok=True,
    )

    names = model.names  # {0: 'D00_longitudinal', ...}
    for r in results:
        path = Path(r.path)
        counts = {}
        for b in r.boxes:
            cls = names[int(b.cls[0])]
            counts[cls] = counts.get(cls, 0) + 1
        summary = ", ".join(f"{v}x {k}" for k, v in counts.items()) or "no damage found"
        print(f"  {path.name}: {summary}")

    print(f"\nAnnotated results saved to: {OUT_DIR / 'detect'}")


if __name__ == "__main__":
    main()
