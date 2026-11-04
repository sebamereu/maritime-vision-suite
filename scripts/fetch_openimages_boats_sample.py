#!/usr/bin/env python3
"""Download a reproducible sample of Open Images V7's "Boat" class
(generic boats: far more varied than SeaShips' commercial-shipping-only
domain) and convert it to the same YOLO layout fetch_seaships_sample.py
produces, so it's a drop-in alternative or can be merged with it.

Open Images annotations are CC BY 4.0 (Google); the underlying photos
carry per-image Creative Commons licenses (mostly from Flickr) that
explicitly permit this kind of dataset use -- unlike scraping broadcast
footage, this is exactly what Open Images is built and licensed for.

Only the validation split is used (25MB of annotations, ~41k images) --
the full train split's annotation file alone is 2.2GB, overkill for a
"broader than SeaShips, not a from-scratch dataset" supplement.

Usage:
    python scripts/fetch_openimages_boats_sample.py --n 500
"""
import argparse
import csv
import random
import urllib.error
import urllib.request
from pathlib import Path

CLASS_DESCRIPTIONS_URL = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
VAL_BBOX_URL = "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"
IMAGE_URL_TEMPLATE = "https://s3.amazonaws.com/open-images-dataset/validation/{image_id}.jpg"

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "data" / "_cache" / "openimages"
OUT_DIR = REPO_ROOT / "data" / "datasets" / "openimages_boats_sample"


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def find_boat_class_id(csv_path: Path) -> str:
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if len(row) == 2 and row[1] == "Boat":
                return row[0]
    raise RuntimeError("'Boat' class not found in class-descriptions CSV")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="sample size (of 567 available)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    class_csv = CACHE_DIR / "class-descriptions.csv"
    bbox_csv = CACHE_DIR / "validation-annotations-bbox.csv"
    download(CLASS_DESCRIPTIONS_URL, class_csv)
    download(VAL_BBOX_URL, bbox_csv)

    boat_id = find_boat_class_id(class_csv)
    print(f"Boat class ID: {boat_id}")

    # image_id -> list of (cx, cy, w, h) YOLO boxes. Open Images XMin/XMax/
    # YMin/YMax are already normalized 0-1, so no pixel dimensions needed.
    # IsGroupOf boxes (one box covering several boats, e.g. a marina) are
    # dropped -- they're not single-object ground truth and would teach
    # the model to draw a box around a whole cluster of boats.
    boxes_by_image = {}
    with open(bbox_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["LabelName"] != boat_id or row["Confidence"] != "1":
                continue
            if row["IsGroupOf"] == "1":
                continue
            xmin, xmax = float(row["XMin"]), float(row["XMax"])
            ymin, ymax = float(row["YMin"]), float(row["YMax"])
            cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
            w, h = xmax - xmin, ymax - ymin
            boxes_by_image.setdefault(row["ImageID"], []).append((cx, cy, w, h))

    print(f"Images with at least one individual (non-grouped) boat box: {len(boxes_by_image)}")

    rng = random.Random(args.seed)
    image_ids = sorted(boxes_by_image.keys())
    rng.shuffle(image_ids)
    chosen = image_ids[: min(args.n, len(image_ids))]

    n_val = max(1, int(len(chosen) * args.val_fraction))
    val_ids, train_ids = chosen[:n_val], chosen[n_val:]

    for split, ids in [("train", train_ids), ("val", val_ids)]:
        img_dir = OUT_DIR / "images" / split
        lbl_dir = OUT_DIR / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for image_id in ids:
            img_path = img_dir / f"{image_id}.jpg"
            if not img_path.exists():
                try:
                    urllib.request.urlretrieve(
                        IMAGE_URL_TEMPLATE.format(image_id=image_id), img_path
                    )
                except (urllib.error.URLError, OSError) as e:
                    print(f"  skip {image_id}: {e}")
                    continue

            lines = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in boxes_by_image[image_id]]
            (lbl_dir / f"{image_id}.txt").write_text("\n".join(lines) + "\n")

    n_boxes = sum(len(v) for k, v in boxes_by_image.items() if k in chosen)
    data_yaml = OUT_DIR / "data.yaml"
    data_yaml.write_text(
        f"path: {OUT_DIR}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: boat\n"
    )

    print(f"\nSample ready: {len(chosen)} images ({len(train_ids)} train / {len(val_ids)} val), "
          f"{n_boxes} boat boxes, in {OUT_DIR}")


if __name__ == "__main__":
    main()
