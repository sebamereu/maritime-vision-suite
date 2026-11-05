#!/usr/bin/env python3
"""Draw the gust/calm zone overlay on sampled frames from a video, for
visual inspection -- the qualitative counterpart to
detect_wind_zones.py's statistics.

Red = likely gust (rougher than this frame's average), blue = likely
calm/lull, unmarked = neutral.

Usage:
    python scripts/visualize_wind_zones.py --video data/datasets/wind_test_videos/sea_texture.mp4
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wind_detection" / "src"))
from detector import detect_frame_pair, zone_mask


def draw_overlay(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    overlay = frame.copy()
    red = np.zeros_like(frame)
    red[:, :, 2] = 255
    blue = np.zeros_like(frame)
    blue[:, :, 0] = 255

    gust_blend = cv2.addWeighted(frame, 1 - alpha, red, alpha, 0)
    calm_blend = cv2.addWeighted(frame, 1 - alpha, blue, alpha, 0)
    overlay = np.where(mask[..., None] == 1, gust_blend, overlay)
    overlay = np.where(mask[..., None] == -1, calm_blend, overlay)
    return overlay.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=5, help="how many sample frames to save")
    parser.add_argument("--gust-threshold", type=float, default=0.5)
    parser.add_argument("--calm-threshold", type=float, default=-0.3)
    parser.add_argument("--output", default="data/outputs/wind_zone_overlays")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_at = {
        int(i) for i in np.linspace(args.stride, total_frames - 1,
                                     num=args.n_samples, dtype=int)
    }

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    prev_gray = None
    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None and frame_idx in sample_at:
            gm = detect_frame_pair(prev_gray, frame, gray)
            mask = zone_mask(gm, gust_threshold=args.gust_threshold,
                              calm_threshold=args.calm_threshold)
            overlay = draw_overlay(frame, mask)
            out_path = output_dir / f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(str(out_path), overlay)
            print(f"Saved {out_path} (gust frac {float((mask == 1).mean()):.1%}, "
                  f"calm frac {float((mask == -1).mean()):.1%})")
            saved += 1

        prev_gray = gray
        frame_idx += 1
        if saved >= args.n_samples:
            break

    cap.release()
    print(f"\n{saved} overlays written to {output_dir}")


if __name__ == "__main__":
    main()
