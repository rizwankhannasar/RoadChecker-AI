"""Try bigger imgsz + TTA to raise confidence on the user photo."""
from ultralytics import YOLO

PHOTO = r"E:\RoadDamageDetector-master\images\road.jpg"
FINETUNED = r"E:\RoadDamageDetector-master\runs\finetune_india_416\weights\best.pt"
PRETRAINED = r"E:\RoadDamageDetector-master\runs\pretrained\rdd2022-yolov8n-best.pt"

def show(tag, results, names):
    boxes = results[0].boxes
    print(f"\n=== {tag} === Detections: {len(boxes)}")
    for b in boxes:
        print(f"  - {names[int(b.cls[0])]}: {float(b.conf[0]):.2f}")

for label, path, imgsz in [("finetuned", FINETUNED, 416), ("finetuned", FINETUNED, 640),
                           ("pretrained", PRETRAINED, 640)]:
    m = YOLO(path)
    r = m.predict(PHOTO, imgsz=imgsz, conf=0.05, verbose=False)
    show(f"{label} imgsz={imgsz} plain", r, m.names)
    r = m.predict(PHOTO, imgsz=imgsz, conf=0.05, augment=True, verbose=False)
    show(f"{label} imgsz={imgsz} TTA", r, m.names)

print("\nDone.")
