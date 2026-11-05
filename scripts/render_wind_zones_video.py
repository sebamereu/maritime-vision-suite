#!/usr/bin/env python3
"""Render a full video with the gust/calm overlay on every frame, not
just sampled stills -- lets you actually watch whether zones persist
and move coherently over time, instead of inferring it from the
frame-to-frame correlation number alone.

Usage:
    python scripts/render_wind_zones_video.py --video data/datasets/wind_test_videos/sea_texture.mp4 \
        --output data/outputs/sea_texture_annotated.mp4 --exclude-glare
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seconds", type=float, default=15.0,
                         help="cap output length -- keeps file size and render time sane")
    parser.add_argument("--gust-threshold", type=float, default=0.5)
    parser.add_argument("--calm-threshold", type=float, default=-0.3)
    parser.add_argument("--segmentation-checkpoint", default=None)
    parser.add_argument("--shore-buffer-px", type=int, default=0)
    parser.add_argument("--exclude-glare", action="store_true")
    parser.add_argument("--glare-min-area-fraction", type=float, default=0.01)
    args = parser.parse_args()

    masker = None
    if args.segmentation_checkpoint:
        from water_mask import WaterMasker
        masker = WaterMasker(args.segmentation_checkpoint)
    if args.shore_buffer_px and not masker:
        raise SystemExit("--shore-buffer-px requires --segmentation-checkpoint")

    open_water_mask = None
    if args.shore_buffer_px:
        from shoreline import open_water_mask as _open_water_mask
        open_water_mask = _open_water_mask

    specular_mask = None
    if args.exclude_glare:
        from glare import specular_mask as _specular_mask
        specular_mask = _specular_mask

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(args.max_seconds * fps)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    prev_gray = None
    frame_idx = 0
    written = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            writer.write(frame)  # first frame has no previous frame to compare -- pass through
        else:
            water_mask = None
            if masker:
                if open_water_mask:
                    class_map = masker.predict_classes(frame)
                    water_mask = open_water_mask(class_map, min_shore_distance_px=args.shore_buffer_px)
                else:
                    water_mask = masker.predict(frame)

            if specular_mask:
                glare = specular_mask(frame, min_area_fraction=args.glare_min_area_fraction)
                not_glare = ~glare
                water_mask = not_glare if water_mask is None else (water_mask & not_glare)

            gm = detect_frame_pair(prev_gray, frame, gray, water_mask=water_mask)
            mask = zone_mask(gm, gust_threshold=args.gust_threshold, calm_threshold=args.calm_threshold)
            writer.write(draw_overlay(frame, mask))

        prev_gray = gray
        frame_idx += 1
        written += 1

    cap.release()
    writer.release()
    print(f"Wrote {written} frames ({written / fps:.1f}s at {fps:.1f}fps) to {out_path}")


if __name__ == "__main__":
    main()
