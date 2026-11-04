#!/usr/bin/env python3
"""Qualitative domain-generalization check: run a boat-detection
checkpoint on real sailboat photos instead of SeaShips (commercial
shipping) test images.

No ground truth here -- these are free-license stock photos, not an
annotated dataset -- so this prints confidence scores and draws boxes
for visual inspection, not a metric. That's the whole point: SeaShips
has zero sailboats in it, so this is the actual domain-gap test flagged
as untested in the README.

Usage:
    python scripts/test_on_real_sailing_photos.py --images-dir /path/to/photos \
        --checkpoint data/checkpoints/yolov8n_boats.pt
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "boat_detection" / "src"))
from model import BoatDetector


def draw_boxes(frame, boxes, confs, label):
    out = frame.copy()
    for (xmin, ymin, xmax, ymax), conf in zip(boxes, confs):
        cv2.rectangle(out, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 3)
        cv2.putText(out, f"{conf:.2f}", (int(xmin), max(int(ymin) - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"{label} ({len(boxes)} boxes)", (6, out.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--checkpoint", default="data/checkpoints/yolov8n_boats.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--output", default="data/outputs/real_sailing_test")
    args = parser.parse_args()

    detector = BoatDetector(args.checkpoint)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(images_dir.glob("*.jpg")):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Skipping unreadable file: {img_path}")
            continue

        preds = detector.predict(frame, conf=args.conf)[0]
        boxes = preds.boxes.xyxy.tolist() if len(preds.boxes) else []
        confs = preds.boxes.conf.tolist() if len(preds.boxes) else []

        overlay = draw_boxes(frame, boxes, confs, img_path.stem)
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), overlay)

        conf_str = ", ".join(f"{c:.2f}" for c in confs) if confs else "none"
        print(f"{img_path.name}: {len(boxes)} boxes, confidences: [{conf_str}]")

    print(f"\nOverlays written to {output_dir}")


if __name__ == "__main__":
    main()
