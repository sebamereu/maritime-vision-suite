#!/usr/bin/env python3
"""Draw the predicted horizon line against ground truth on sample images,
so the numbers in evaluate_horizon.py can actually be looked at.

Each output image is two panels side by side:

LEFT  — the scoring view:
  Red   = predicted horizon (segmentation/src/horizon.py), full width.
  Green = ground-truth horizon, drawn ONLY on columns with a clean
          sky/water transition (see ground_truth.py). Gaps are columns
          the metric excludes (coastline, obstacles) — on a coastline
          -heavy image this panel can look almost empty, which is
          correct, not a bug: see the right panel for the full picture.

RIGHT — the actual ground truth, nothing hidden:
  The full MaSTr1325 semantic mask blended over the frame — orange=sky,
  blue=water, red=obstacle/environment, gray=ignore region.

Usage:
    python scripts/visualize_horizon.py
    python scripts/visualize_horizon.py --output data/outputs/horizon_overlays
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "segmentation" / "src"))
from ground_truth import IGNORE, OBSTACLE, SKY, WATER, horizon_from_mask
from horizon import detect_horizon

MASK_COLORS_BGR = {
    SKY: (60, 170, 255),       # orange
    WATER: (255, 120, 40),     # blue
    OBSTACLE: (0, 0, 255),     # red
    IGNORE: (128, 128, 128),   # gray
}


def draw_scoring_panel(frame: np.ndarray, predicted: np.ndarray,
                        valid_columns: np.ndarray, gt_rows: np.ndarray) -> np.ndarray:
    panel = frame.copy()
    width = frame.shape[1]
    rows = np.round(predicted).astype(int)

    for x in range(width - 1):
        cv2.line(panel, (x, rows[x]), (x + 1, rows[x + 1]), (0, 0, 255), 1)  # BGR red

    for x, y in zip(valid_columns, gt_rows):
        cv2.circle(panel, (int(x), int(y)), 1, (0, 200, 0), -1)  # BGR green

    cv2.putText(panel, "predicted (red) vs scored GT (green)", (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def draw_mask_panel(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    colored = np.zeros_like(frame)
    for label, color in MASK_COLORS_BGR.items():
        colored[mask == label] = color

    panel = cv2.addWeighted(frame, 0.55, colored, 0.45, 0)
    cv2.putText(panel, "full ground truth: sky/water/obstacle/ignore", (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def draw_overlay(frame: np.ndarray, mask: np.ndarray, predicted: np.ndarray,
                  valid_columns: np.ndarray, gt_rows: np.ndarray) -> np.ndarray:
    scoring = draw_scoring_panel(frame, predicted, valid_columns, gt_rows)
    full_gt = draw_mask_panel(frame, mask)
    return cv2.hconcat([scoring, full_gt])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data/datasets/mastr1325_sample/images")
    parser.add_argument("--masks", default="data/datasets/mastr1325_sample/masks")
    parser.add_argument("--output", default="data/outputs/horizon_overlays")
    parser.add_argument("--examples-dir", default=None,
                         help="if set, copy the best/median/worst overlays here")
    args = parser.parse_args()

    images_dir = Path(args.images)
    masks_dir = Path(args.masks)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []  # (mae_or_None, image_name, overlay)

    for img_path in sorted(images_dir.glob("*.jpg")):
        mask_path = masks_dir / f"{img_path.stem}m.png"
        if not mask_path.exists():
            continue

        frame = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

        predicted = detect_horizon(frame)
        valid_cols, gt_rows = horizon_from_mask(mask)

        overlay = draw_overlay(frame, mask, predicted, valid_cols, gt_rows)
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), overlay)

        mae = None
        if len(valid_cols) > 0:
            mae = float(np.mean(np.abs(predicted[valid_cols] - gt_rows)))
        results.append((mae, img_path.name, out_path))

    print(f"Wrote {len(results)} overlays to {output_dir}")

    scored = sorted([r for r in results if r[0] is not None], key=lambda r: r[0])
    if scored:
        best = scored[0]
        worst = scored[-1]
        median = scored[len(scored) // 2]
        print(f"Best:   {best[1]} (MAE {best[0]:.1f}px)")
        print(f"Median: {median[1]} (MAE {median[0]:.1f}px)")
        print(f"Worst:  {worst[1]} (MAE {worst[0]:.1f}px)")

        if args.examples_dir:
            examples_dir = Path(args.examples_dir)
            examples_dir.mkdir(parents=True, exist_ok=True)
            for label, (mae, name, path) in zip(
                ["best", "median", "worst"], [best, median, worst]
            ):
                dest = examples_dir / f"horizon_{label}.jpg"
                cv2.imwrite(str(dest), cv2.imread(str(path)))
                print(f"Copied example '{label}' -> {dest}")


if __name__ == "__main__":
    main()
