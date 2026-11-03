#!/usr/bin/env python3
"""Download a reproducible sample of the MaSTr1325 sea/sky/obstacle
segmentation dataset (CC BY 4.0, Bovcon et al., IROS 2019).

The full dataset (1325 images + per-pixel masks at 512x384, ~24MB total)
is downloaded, then a fixed-seed random sample of N image/mask pairs is
copied into data/datasets/mastr1325_sample/ for quick local experiments.
The full download is kept under a temp dir and not committed anywhere
(data/ is gitignored).

Usage:
    python scripts/fetch_mastr1325_sample.py [--n 50] [--seed 42]
"""
import argparse
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

IMAGES_URL = "https://data.vicos.si/borja/mastr1325_dataset/MaSTr1325_images_512x384.zip"
MASKS_URL = "https://data.vicos.si/borja/mastr1325_dataset/MaSTr1325_masks_512x384.zip"

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "_cache" / "mastr1325_full"
SAMPLE_DIR = DATA_DIR / "datasets" / "mastr1325_sample"


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="sample size")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    images_zip = CACHE_DIR / "images_512x384.zip"
    masks_zip = CACHE_DIR / "masks_512x384.zip"
    download(IMAGES_URL, images_zip)
    download(MASKS_URL, masks_zip)

    images_dir = CACHE_DIR / "images"
    masks_dir = CACHE_DIR / "masks"
    extract(images_zip, images_dir)
    extract(masks_zip, masks_dir)

    sample_images = SAMPLE_DIR / "images"
    sample_masks = SAMPLE_DIR / "masks"
    sample_images.mkdir(parents=True, exist_ok=True)
    sample_masks.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    all_images = sorted(images_dir.glob("*.jpg"))
    chosen = random.sample(all_images, min(args.n, len(all_images)))

    copied, missing = 0, []
    for img in chosen:
        mask = masks_dir / f"{img.stem}m.png"
        if not mask.exists():
            missing.append(img.stem)
            continue
        shutil.copy(img, sample_images / img.name)
        shutil.copy(mask, sample_masks / mask.name)
        copied += 1

    print(f"Sample ready: {copied} image/mask pairs in {SAMPLE_DIR}")
    if missing:
        print(f"Warning: no mask found for {len(missing)} images: {missing}")


if __name__ == "__main__":
    main()
