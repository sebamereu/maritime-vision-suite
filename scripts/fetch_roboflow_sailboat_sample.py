#!/usr/bin/env python3
"""Download the "Sailboat detector" dataset from Roboflow Universe
(learning-299jh/sailboat-detector, CC BY 4.0, 8,888 images, 11 classes
including a dedicated "sailboat" class) in YOLOv8 format.

Unlike Open Images' generic "Boat" class, this dataset was built and
labeled specifically for sail/motor/cargo/cruise/military vessel
*type* distinction -- the closest thing found so far to a real
sailing-specific detection dataset (see README for what else was
checked and ruled out).

Requires a free Roboflow account and API key (the platform requires an
account to download, regardless of the dataset's own license) --
never hardcode it here, set it as an environment variable:

    export ROBOFLOW_API_KEY="..."
    python scripts/fetch_roboflow_sailboat_sample.py

Collapses the 11 original classes into this project's single "boat"
class by default (consistent with SeaShips/Open Images), unless
--keep-classes is passed to keep the original per-type labels.

Subsamples to --n-train/--n-val by default (the full 7113/1337 split
makes an epoch over a mixed dataset ~5x slower per image than
SeaShips/Open Images alone, and lets one source dominate the mix
disproportionately) -- pass --n-train 0 to keep everything.
"""
import argparse
import os
import random
import shutil
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "datasets" / "roboflow_sailboat_sample"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "_cache" / "roboflow_sailboat_raw"


def collapse_to_single_class(dataset_dir: Path) -> None:
    """Rewrite every YOLO label file's class index to 0 ("boat"), and
    rewrite data.yaml accordingly -- matches SeaShips/Open Images so
    all three can be combined under one data.yaml."""
    for split_dir in (dataset_dir / "train" / "labels",
                       dataset_dir / "valid" / "labels",
                       dataset_dir / "test" / "labels"):
        if not split_dir.exists():
            continue
        for label_file in split_dir.glob("*.txt"):
            lines = label_file.read_text().splitlines()
            rewritten = []
            for line in lines:
                parts = line.split()
                if not parts:
                    continue
                rewritten.append(" ".join(["0", *parts[1:]]))
            label_file.write_text("\n".join(rewritten) + ("\n" if rewritten else ""))

    (dataset_dir / "data.yaml").write_text(
        f"path: {dataset_dir}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "names:\n"
        "  0: boat\n"
    )


def subsample_split(dataset_dir: Path, split: str, n: int, seed: int) -> int:
    img_dir = dataset_dir / split / "images"
    lbl_dir = dataset_dir / split / "labels"
    images = sorted(img_dir.glob("*"))
    if n <= 0 or n >= len(images):
        return len(images)

    rng = random.Random(seed)
    keep = set(rng.sample(images, n))
    for img in images:
        if img not in keep:
            img.unlink()
            lbl = lbl_dir / f"{img.stem}.txt"
            if lbl.exists():
                lbl.unlink()
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-classes", action="store_true",
                         help="keep the original 11 vessel-type classes instead of "
                              "collapsing to a single 'boat' class")
    parser.add_argument("--n-train", type=int, default=1200,
                         help="subsample to this many train images (0 = keep all 7113)")
    parser.add_argument("--n-val", type=int, default=300,
                         help="subsample to this many val images (0 = keep all 1337)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set ROBOFLOW_API_KEY first (export ROBOFLOW_API_KEY=... -- "
            "free account at roboflow.com, API key under Settings)"
        )

    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("learning-299jh").project("sailboat-detector")
    version = project.version(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    dataset = version.download("yolov8", location=str(RAW_DIR), overwrite=True)
    shutil.copytree(dataset.location, OUT_DIR)

    if not args.keep_classes:
        collapse_to_single_class(OUT_DIR)

    n_train = subsample_split(OUT_DIR, "train", args.n_train, args.seed)
    n_val = subsample_split(OUT_DIR, "valid", args.n_val, args.seed)
    print(f"\nDataset ready in {OUT_DIR}: {n_train} train, {n_val} val images")


if __name__ == "__main__":
    main()
