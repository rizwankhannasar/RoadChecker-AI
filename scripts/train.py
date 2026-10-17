"""
Train YOLOv8n on the RDD2022 India subset (CPU-friendly settings).

- Transfer learning from pretrained yolov8n.pt
- Time-capped training (default 45 min) so it always finishes
- Keeps best checkpoint based on validation mAP

Usage:
    python train.py            # 45-minute budget (default)
    python train.py 60         # 60-minute budget
"""
import sys
from pathlib import Path

from ultralytics import YOLO

DATA = Path(r"E:\RoadDamageDetector-master\dataset\extracted\data.yaml")
RUNS = Path(r"E:\RoadDamageDetector-master\runs")
TIME_HOURS = float(sys.argv[1]) / 60.0 if len(sys.argv) > 1 else 0.75  # 45 min


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    print(f"Data   : {DATA}")
    print(f"Budget : {TIME_HOURS * 60:.0f} minutes\n")

    # nano model = right size for CPU
    model = YOLO("yolov8n.pt")

    model.train(
        data=str(DATA),
        imgsz=320,          # 4x fewer pixels than 640 -> ~4x faster on CPU
        batch=16,           # bigger batches utilize the CPU better
        epochs=50,          # cap; the time limit stops us earlier
        time=TIME_HOURS,    # ultralytics time budget (hours)
        fraction=0.5,       # use 50% of train images -> shorter epochs
        mosaic=0.0,         # mosaic augmentation is expensive on CPU -> off
        amp=False,          # AMP gives little/no speedup on this CPU
        device="cpu",
        workers=2,
        patience=10,        # early stop if val stalls
        project=str(RUNS),
        name="rdd2022_india_yolov8n",
        exist_ok=True,
        verbose=True,
    )

    # validate best checkpoint explicitly
    best = RUNS / "rdd2022_india_yolov8n" / "weights" / "best.pt"
    print(f"\nBest weights: {best}  (exists={best.exists()})")
    if best.exists():
        m = YOLO(str(best))
        metrics = m.val(data=str(DATA), imgsz=320, device="cpu", fraction=0.5)
        print(f"mAP50-95 : {metrics.box.map:.4f}")
        print(f"mAP50    : {metrics.box.map50:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall   : {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
