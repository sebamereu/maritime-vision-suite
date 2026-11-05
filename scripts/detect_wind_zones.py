#!/usr/bin/env python3
"""Run the unsupervised gust/calm detector over a video and report
internal-consistency statistics -- NOT an accuracy number, which would
need real wind measurements synced to the footage (no such dataset
exists for this project; see README).

What "internal consistency" means here: frame-to-frame correlation of
the gust-likelihood map. If the signal were just noise, consecutive
frames (a fraction of a second apart) would correlate close to zero;
real gust/lull patches persist for at least a few frames, so a real
signal should correlate noticeably above zero without being perfectly
static (the patches do move and change). This is a sanity check on the
method, not a claim about matching reality.

Usage:
    python scripts/detect_wind_zones.py --video data/datasets/wind_test_videos/sea_texture.mp4
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wind_detection" / "src"))
from detector import detect_frame_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--stride", type=int, default=5,
                         help="process every Nth frame pair (consecutive frames look "
                              "nearly identical at 25fps; skipping gives more visible motion)")
    parser.add_argument("--max-pairs", type=int, default=60)
    parser.add_argument("--patch-size", type=int, default=32)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    prev_gray = None
    prev_score = None
    correlations = []
    gust_fractions, calm_fractions = [], []

    frame_idx = 0
    pairs_done = 0
    while pairs_done < args.max_pairs:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % args.stride != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            gm = detect_frame_pair(prev_gray, frame, gray, patch_size=args.patch_size)

            gust_fractions.append(float((gm.score >= 0.75).mean()))
            calm_fractions.append(float((gm.score <= -0.75).mean()))

            if prev_score is not None:
                corr = float(np.corrcoef(prev_score.ravel(), gm.score.ravel())[0, 1])
                if np.isfinite(corr):
                    correlations.append(corr)

            prev_score = gm.score
            pairs_done += 1

        prev_gray = gray
        frame_idx += 1

    cap.release()

    if not correlations:
        print("Not enough frame pairs processed to report statistics.")
        return

    print(f"Processed {pairs_done} frame pairs (stride={args.stride}) from {args.video}")
    print(f"Mean gust-zone fraction (score >= 0.75):  {np.mean(gust_fractions):.1%}")
    print(f"Mean calm-zone fraction (score <= -0.75): {np.mean(calm_fractions):.1%}")
    print(f"Frame-to-frame score correlation: mean {np.mean(correlations):.3f}, "
          f"std {np.std(correlations):.3f}, min {np.min(correlations):.3f}, "
          f"max {np.max(correlations):.3f}")
    print("(Correlation near 0 would mean noise; near 1 would mean a static, "
          "non-responsive map. This is an internal sanity check, not an accuracy "
          "measurement against real wind data -- see README.)")


if __name__ == "__main__":
    main()
