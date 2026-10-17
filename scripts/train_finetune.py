"""
Fine-tune the strong pretrained RDD2022 model on India-specific data.

Strategy:
- Start from dronefreak/rdd2022-yolov8n (50 epochs, 47k images, mAP 38% on our val)
- Low learning rate so we ADAPT to India roads without destroying learned features
- Full India train set, 416px (balance of CPU speed + detail for thin cracks)
- Time-budgeted (default 114 min) so it always finishes and saves best.pt

Usage:
    python train_finetune.py            # 114-minute budget
    python train_finetune.py 60         # 60-minute budget
"""
import sys
from pathlib import Path

from ultralytics import YOLO

PRETRAINED = Path(r"E:\RoadDamageDetector-master\runs\pretrained\rdd2022-yolov8n-best.pt")
DATA = Path(r"E:\RoadDamageDetector-master\dataset\extracted\data.yaml")
RUNS = Path(r"E:\RoadDamageDetector-master\runs")
TIME_HOURS = (float(sys.argv[1]) if len(sys.argv) > 1 else 114) / 60.0


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    print(f"Init    : {PRETRAINED.name} (pretrained RDD2022)")
    print(f"Data    : {DATA} (India full train)")
    print(f"Budget  : {TIME_HOURS * 60:.0f} minutes\n")

    model = YOLO(str(PRETRAINED))  # warm start — not from scratch

    model.train(
        data=str(DATA),
        imgsz=416,            # middle ground: more detail than 320, 2.4x faster than 640
        batch=16,
        epochs=60,            # high cap; time budget stops us
        time=TIME_HOURS,      # hours
        fraction=1.0,         # FULL India train set this time
        optimizer="SGD",
        lr0=0.002,            # low LR = fine-tune, don't wipe pretrained knowledge
        lrf=0.01,
        warmup_epochs=2,
        mosaic=0.0,           # CPU-friendly (pretrained already learned augmentations)
        amp=False,
        device="cpu",
        workers=2,
        patience=12,
        project=str(RUNS),
        name="finetune_india_416",
        exist_ok=True,
        verbose=True,
    )

    best = RUNS / "finetune_india_416" / "weights" / "best.pt"
    print(f"\nBest weights: {best} (exists={best.exists()})")
    if best.exists():
        m = YOLO(str(best))
        metrics = m.val(data=str(DATA), imgsz=416, device="cpu")
        print(f"\n=== FINE-TUNED MODEL on India val ===")
        print(f"mAP50    : {metrics.box.map50:.4f}  (pretrained scored 0.3801)")
        print(f"mAP50-95 : {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall   : {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
