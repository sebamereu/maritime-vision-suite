#!/usr/bin/env python3
"""Run the trained TinyUNet on sample images and draw predicted vs
ground-truth masks side by side — the CNN equivalent of
visualize_horizon.py, for a direct, honest comparison against the
classical baseline.

Usage:
    python scripts/visualize_segmentation.py
    python scripts/visualize_segmentation.py --checkpoint data/checkpoints/tiny_unet.pt
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "segmentation" / "src"))
from dataset import IGNORE_INDEX, NUM_CLASSES, remap_mask
from metrics import mean_iou
from model import TinyUNet

MASK_COLORS_BGR = {
    2: (60, 170, 255),   # sky, orange
    1: (255, 120, 40),   # water, blue
    0: (0, 0, 255),      # obstacle, red
}


def colorize(mask: np.ndarray, frame_shape) -> np.ndarray:
    colored = np.zeros(frame_shape, dtype=np.uint8)
    for label, color in MASK_COLORS_BGR.items():
        colored[mask == label] = color
    return colored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data/datasets/mastr1325_sample/images")
    parser.add_argument("--masks", default="data/datasets/mastr1325_sample/masks")
    parser.add_argument("--checkpoint", default="data/checkpoints/tiny_unet.pt")
    parser.add_argument("--size", type=int, nargs=2, default=[192, 256],
                         metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--output", default="data/outputs/segmentation_overlays")
    parser.add_argument("--examples-dir", default=None)
    args = parser.parse_args()

    height, width = args.size
    device = torch.device("cpu")
    model = TinyUNet(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    images_dir = Path(args.images)
    masks_dir = Path(args.masks)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for img_path in sorted(images_dir.glob("*.jpg")):
        mask_path = masks_dir / f"{img_path.stem}m.png"
        if not mask_path.exists():
            continue

        frame = cv2.imread(str(img_path))
        raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        gt_mask = remap_mask(raw_mask)

        small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)
            pred_small = logits.argmax(dim=1).squeeze(0).numpy().astype(np.uint8)

        pred = cv2.resize(pred_small, (frame.shape[1], frame.shape[0]),
                           interpolation=cv2.INTER_NEAREST)

        iou = mean_iou(pred, gt_mask, NUM_CLASSES, IGNORE_INDEX)

        pred_overlay = cv2.addWeighted(frame, 0.55, colorize(pred, frame.shape), 0.45, 0)
        gt_overlay = cv2.addWeighted(frame, 0.55, colorize(gt_mask, frame.shape), 0.45, 0)
        cv2.putText(pred_overlay, "predicted", (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(gt_overlay, "ground truth", (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)
        combined = cv2.hconcat([pred_overlay, gt_overlay])

        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), combined)
        results.append((float(iou) if not np.isnan(iou) else -1.0, img_path.name, out_path))

    print(f"Wrote {len(results)} overlays to {output_dir}")

    scored = sorted([r for r in results if r[0] >= 0], key=lambda r: r[0])
    if scored:
        worst, median, best = scored[0], scored[len(scored) // 2], scored[-1]
        print(f"Best:   {best[1]} (mIoU {best[0]:.3f})")
        print(f"Median: {median[1]} (mIoU {median[0]:.3f})")
        print(f"Worst:  {worst[1]} (mIoU {worst[0]:.3f})")

        if args.examples_dir:
            examples_dir = Path(args.examples_dir)
            examples_dir.mkdir(parents=True, exist_ok=True)
            for label, (iou, name, path) in zip(
                ["best", "median", "worst"], [best, median, worst]
            ):
                dest = examples_dir / f"segmentation_{label}.jpg"
                cv2.imwrite(str(dest), cv2.imread(str(path)))
                print(f"Copied example '{label}' -> {dest}")


if __name__ == "__main__":
    main()
