#!/usr/bin/env python3
"""Evaluate the classical horizon detector against MaSTr1325 ground-truth
masks and report mean/median absolute pixel error, following the same
"water-edge accuracy in pixels" methodology used in the MODD2 paper
(Bovcon et al.) so the number is comparable to published baselines.

Usage:
    python scripts/evaluate_horizon.py \
        --images data/datasets/mastr1325_sample/images \
        --masks data/datasets/mastr1325_sample/masks
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "segmentation" / "src"))
from ground_truth import horizon_from_mask
from horizon import detect_horizon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data/datasets/mastr1325_sample/images")
    parser.add_argument("--masks", default="data/datasets/mastr1325_sample/masks")
    args = parser.parse_args()

    images_dir = Path(args.images)
    masks_dir = Path(args.masks)

    all_errors = []
    per_image_mae = []
    skipped = []
    valid_column_fractions = []

    for img_path in sorted(images_dir.glob("*.jpg")):
        mask_path = masks_dir / f"{img_path.stem}m.png"
        if not mask_path.exists():
            skipped.append(img_path.name)
            continue

        frame = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

        predicted = detect_horizon(frame)
        valid_cols, gt_rows = horizon_from_mask(mask)

        if len(valid_cols) == 0:
            skipped.append(img_path.name)
            continue

        valid_column_fractions.append(len(valid_cols) / mask.shape[1])
        errors = np.abs(predicted[valid_cols] - gt_rows)
        all_errors.extend(errors.tolist())
        per_image_mae.append(float(np.mean(errors)))

    if not all_errors:
        print("No valid images evaluated.")
        return

    all_errors = np.array(all_errors)
    print(f"Images evaluated: {len(per_image_mae)} (skipped: {len(skipped)})")
    print(f"Mean fraction of columns with clean sky/water ground truth: "
          f"{np.mean(valid_column_fractions):.1%}")
    print("Pixel error over all valid columns, all images:")
    print(f"  MAE:    {np.mean(all_errors):.1f} px")
    print(f"  Median: {np.median(all_errors):.1f} px")
    print(f"  Std:    {np.std(all_errors):.1f} px")
    print(f"Per-image MAE: mean {np.mean(per_image_mae):.1f} px, "
          f"worst {np.max(per_image_mae):.1f} px, "
          f"best {np.min(per_image_mae):.1f} px")


if __name__ == "__main__":
    main()
