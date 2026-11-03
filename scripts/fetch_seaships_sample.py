#!/usr/bin/env python3
"""Download a reproducible sample of the SeaShips(7000) ship-detection
dataset (Shao et al., IEEE TMM 2018) and convert it to YOLO labels.

The dataset ships Pascal VOC XML annotations across six ship-type
classes; this project collapses them all to one class, "boat" (see
boat_detection/src/dataset.py). The full zip (~1.2GB, 7000 images, via
a public Hugging Face mirror -- the original paper has no clearly
published redistribution license of its own, see README) is cached
under data/_cache/ and not committed (data/ is gitignored); a
fixed-seed sample is converted into data/datasets/seaships_sample/
along with a data.yaml ultralytics can train/val against directly.

Usage:
    python scripts/fetch_seaships_sample.py [--n 150] [--seed 42] [--val-fraction 0.2]
"""
import argparse
import random
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "boat_detection" / "src"))
from dataset import write_yolo_label
from model import COCO_BOAT_CLASS_ID

DATASET_URL = "https://huggingface.co/datasets/datnguyentien204/Seaship7000/resolve/main/SeaShips(7000).zip"

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "_cache" / "seaships_full"
SAMPLE_DIR = DATA_DIR / "datasets" / "seaships_sample"
ZEROSHOT_DIR = DATA_DIR / "datasets" / "seaships_sample_zeroshot"


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def extract(zip_path: Path, dest_dir: Path) -> None:
    if dest_dir.exists() and any(dest_dir.iterdir()):
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def find_dir(root: Path, name: str) -> Path:
    matches = [p for p in root.rglob(name) if p.is_dir()]
    if not matches:
        raise FileNotFoundError(f"no '{name}' directory found under {root}")
    return matches[0]


def write_data_yaml(dest_path: Path, sample_dir: Path) -> None:
    dest_path.write_text(
        f"path: {sample_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: boat\n"
    )


def write_zeroshot_data_yaml(dest_path: Path, zeroshot_dir: Path) -> None:
    """A second data.yaml, in its own directory (ultralytics resolves
    labels by substring-replacing "/images/" with "/labels/" in the
    image path, so this needs its own images/ + labels/ tree, not a
    same-named sibling to the main sample). Labels here are written at
    COCO_BOAT_CLASS_ID instead of 0. ultralytics' val() also requires
    the dataset's class ids to be contiguous from 0, so classes 0-7 are
    filled in with their real COCO names even though this dataset has
    no instances of them -- that's what lets the stock COCO-pretrained
    model (whose own output class ids are COCO's) be scored by
    ultralytics' own standard metric, not a hand-rolled one, before any
    fine-tuning happens.
    """
    dest_path.write_text(
        f"path: {zeroshot_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: person\n"
        "  1: bicycle\n"
        "  2: car\n"
        "  3: motorcycle\n"
        "  4: airplane\n"
        "  5: bus\n"
        "  6: train\n"
        "  7: truck\n"
        f"  {COCO_BOAT_CLASS_ID}: boat\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=150, help="sample size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    zip_path = CACHE_DIR / "SeaShips7000.zip"
    download(DATASET_URL, zip_path)

    extracted_dir = CACHE_DIR / "extracted"
    extract(zip_path, extracted_dir)

    images_dir = find_dir(extracted_dir, "JPEGImages")
    annotations_dir = find_dir(extracted_dir, "Annotations")

    random.seed(args.seed)
    all_images = sorted(images_dir.glob("*.jpg"))
    chosen = random.sample(all_images, min(args.n, len(all_images)))

    n_val = max(1, int(len(chosen) * args.val_fraction))
    splits = {"val": chosen[:n_val], "train": chosen[n_val:]}

    copied, missing = 0, []
    for split, images in splits.items():
        img_out = SAMPLE_DIR / "images" / split
        label_out = SAMPLE_DIR / "labels" / split
        img_zs_out = ZEROSHOT_DIR / "images" / split
        label_zs_out = ZEROSHOT_DIR / "labels" / split
        for d in (img_out, label_out, img_zs_out, label_zs_out):
            d.mkdir(parents=True, exist_ok=True)

        for img in images:
            xml_path = annotations_dir / f"{img.stem}.xml"
            if not xml_path.exists():
                missing.append(img.stem)
                continue
            shutil.copy(img, img_out / img.name)
            write_yolo_label(xml_path, label_out / f"{img.stem}.txt")

            zs_link = img_zs_out / img.name
            if not zs_link.exists():
                zs_link.symlink_to(img.resolve())
            write_yolo_label(xml_path, label_zs_out / f"{img.stem}.txt", class_id=COCO_BOAT_CLASS_ID)
            copied += 1

    write_data_yaml(SAMPLE_DIR / "data.yaml", SAMPLE_DIR)
    write_zeroshot_data_yaml(ZEROSHOT_DIR / "data_zeroshot.yaml", ZEROSHOT_DIR)

    print(f"Sample ready: {copied} image/label pairs "
          f"({len(splits['train'])} train / {len(splits['val'])} val) in {SAMPLE_DIR}")
    if missing:
        print(f"Warning: no annotation found for {len(missing)} images: {missing}")


if __name__ == "__main__":
    main()
