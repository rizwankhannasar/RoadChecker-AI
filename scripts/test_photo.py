"""Test the fine-tuned model on a user photo at multiple confidence levels."""
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

PHOTO = Path(r"E:\RoadDamageDetector-master\images\road.jpg")
FINETUNED = Path(r"E:\RoadDamageDetector-master\runs\finetune_india_416\weights\best.pt")
PRETRAINED = Path(r"E:\RoadDamageDetector-master\runs\pretrained\rdd2022-yolov8n-best.pt")
OUT_DIR = Path(r"E:\RoadDamageDetector-master\runs\test_photo")

if len(sys.argv) > 1:
    PHOTO = Path(sys.argv[1])

assert PHOTO.exists(), f"Photo not found: {PHOTO}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

for name, weights in [("finetuned", FINETUNED), ("pretrained", PRETRAINED)]:
    model = YOLO(str(weights))
    imgsz = 416 if name == "finetuned" else 640
    for conf in (0.05, 0.10, 0.25):
        results = model.predict(str(PHOTO), imgsz=imgsz, conf=conf, verbose=False)
        r = results[0]
        boxes = r.boxes
        print(f"\n=== {name} @ imgsz={imgsz} conf={conf} ===")
        print(f"Detections: {len(boxes)}")
        if len(boxes):
            for b in boxes:
                cls_id = int(b.cls[0])
                cls_name = model.names[cls_id]
                c = float(b.conf[0])
                xyxy = [round(float(v), 1) for v in b.xyxy[0]]
                print(f"  - {cls_name}: {c:.2f} at {xyxy}")
            out = OUT_DIR / f"{PHOTO.stem}_{name}_conf{conf}.jpg"
            # Draw red boxes + circles like the app
            im = r.plot()  # ultralytics annotated image
            cv2.imwrite(str(out), im)
            print(f"Saved: {out}")
        else:
            print("  (no detections)")

print("\nDone.")
