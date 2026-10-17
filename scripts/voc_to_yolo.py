"""
Convert RDD2022 (India) Pascal VOC annotations to YOLO format.

Classes (CRDDC'2022 4-class taxonomy):
    D00 -> 0 : Longitudinal Crack
    D10 -> 1 : Transverse Crack
    D20 -> 2 : Alligator Crack
    D40 -> 3 : Pothole

Non-4-class labels (D01/D11 partial cracks, D43/D44 blur) are dropped but
counted and reported so the conversion stays honest.

Output layout (Ultralytics YOLO):
    dataset_yolo/
        images/train  images/val
        labels/train  labels/val
        data.yaml
"""
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

# --- Config -----------------------------------------------------------------
BASE = Path(r"E:\RoadDamageDetector-master\dataset")
SRC = BASE / "RDD2022" / "India" / "train"
OUT = BASE / "dataset_yolo"
VAL_FRACTION = 0.1
SEED = 42

CLASS_MAP = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}
CLASS_NAMES = ["D00_longitudinal", "D10_transverse", "D20_alligator", "D40_pothole"]
# -----------------------------------------------------------------------------


def find_source_dirs():
    """Locate images/ and annotations/xmls/ under the extracted zip (handles
    an extra wrapper folder level if present)."""
    candidates = [
        (SRC / "images", SRC / "annotations" / "xmls"),
        (SRC / "images", SRC / "annotations"),
    ]
    if not (SRC / "images").exists():
        # search one level down
        for child in BASE.glob("**/India/train"):
            cand = (child / "images", child / "annotations" / "xmls")
            if cand[0].exists():
                candidates.insert(0, cand)
    for img_dir, xml_dir in candidates:
        if img_dir.exists() and xml_dir.exists() and any(img_dir.iterdir()):
            return img_dir, xml_dir
    return None, None


def voc_to_yolo(xml_path, img_w, img_h, skipped):
    """Return list of (cls_id, cx, cy, w, h) normalized; None if malformed."""
    boxes = []
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        skipped["bad_xml"] += 1
        return None
    for obj in root.iter("object"):
        name = obj.findtext("name", "").strip().upper()
        if name not in CLASS_MAP:
            skipped[name or "<empty>"] += 1
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            skipped["no_bndbox"] += 1
            continue
        xmin = float(bnd.findtext("xmin", "0"))
        ymin = float(bnd.findtext("ymin", "0"))
        xmax = float(bnd.findtext("xmax", "0"))
        ymax = float(bnd.findtext("ymax", "0"))
        xmin, xmax = max(0.0, xmin), min(img_w, xmax)
        ymin, ymax = max(0.0, ymin), min(img_h, ymax)
        bw, bh = xmax - xmin, ymax - ymin
        if bw <= 1 or bh <= 1:
            skipped["degenerate"] += 1
            continue
        cx, cy = (xmin + xmax) / 2.0 / img_w, (ymin + ymax) / 2.0 / img_h
        boxes.append((CLASS_MAP[name], cx, cy, bw / img_w, bh / img_h))
    return boxes


def get_image_size(img_path):
    """Read image size without heavy deps (JPEG SOF parse via OpenCV if
    available, else PIL, else metadata from XML)."""
    try:
        import cv2
        im = cv2.imread(str(img_path))
        if im is not None:
            h, w = im.shape[:2]
            return w, h
    except ImportError:
        pass
    from PIL import Image
    with Image.open(img_path) as im:
        return im.size


def main():
    img_dir, xml_dir = find_source_dirs()
    if img_dir is None:
        sys.exit(f"Could not find images+annotations under {SRC}. "
                 f"Extract the zip first, e.g. dataset/RDD2022/India/train/images")
    print(f"Images : {img_dir}")
    print(f"XMLs   : {xml_dir}")

    images = sorted(p for p in img_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    print(f"Total images found: {len(images)}")
    if not images:
        sys.exit("No images found.")

    random.seed(SEED)
    random.shuffle(images)
    n_val = max(1, int(len(images) * VAL_FRACTION))
    val_set = set(images[:n_val])

    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    skipped = Counter()
    cls_counter = Counter()
    n_written = 0

    for img in images:
        xml_path = (xml_dir / (img.stem + ".xml"))
        if not xml_path.exists():
            skipped["missing_xml"] += 1
            continue
        # size: from XML if present, else from image file
        try:
            root = ET.parse(xml_path).getroot()
            size = root.find("size")
            if size is not None:
                w = float(size.findtext("width", "0"))
                h = float(size.findtext("height", "0"))
            else:
                w = h = 0
        except ET.ParseError:
            skipped["bad_xml"] += 1
            continue
        if w <= 0 or h <= 0:
            w, h = get_image_size(img)
        boxes = voc_to_yolo(xml_path, w, h, skipped)
        if boxes is None:
            continue
        for c, *_ in boxes:
            cls_counter[c] += 1

        split = "val" if img in val_set else "train"
        # copy/symlink image (copy: safer on Windows without dev-mode symlinks)
        dst_img = OUT / "images" / split / img.name
        if not dst_img.exists():
            shutil.copy2(img, dst_img)
        with open(OUT / "labels" / split / (img.stem + ".txt"), "w") as f:
            for c, cx, cy, bw, bh in boxes:
                f.write(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        n_written += 1

    # data.yaml
    yaml_path = OUT / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {OUT}\n")
        f.write("train: images/train\n")
        f.write("val: val_small/images\n")
        f.write(f"names: {CLASS_NAMES}\n")
        f.write("nc: 4\n")

    print(f"\nImages processed : {n_written}")
    print(f"Train / Val      : {n_written - len(val_set)} / approx {len(val_set)}")
    print("Class box counts :")
    for c in range(4):
        print(f"  {c} {CLASS_NAMES[c]}: {cls_counter[c]}")
    if skipped:
        print("Skipped (not in 4-class taxonomy or invalid):")
        for k, v in skipped.most_common():
            print(f"  {k}: {v}")
    print(f"\nYOLO dataset at  : {OUT}")
    print(f"Label map        : {yaml_path}")


if __name__ == "__main__":
    main()
