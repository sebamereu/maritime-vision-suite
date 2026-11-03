#!/usr/bin/env python3
"""Run a boat-detection checkpoint on the val sample and draw predicted
vs ground-truth boxes side by side -- best/median/worst picked by mean
best-IoU per ground-truth box, same spirit as visualize_segmentation.py.

This picking score is a simple per-image IoU average, not mAP (see
evaluate_boat_detection.py for why a hand-rolled AP isn't trustworthy
on this dataset's tiny validation set) -- good enough to rank images
relative to each other within one run, not meant as a reported metric.

Usage:
    python scripts/visualize_boat_detection.py
    python scripts/visualize_boat_detection.py --checkpoint data/checkpoints/yolov8n_boats.pt
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "boat_detection" / "src"))
from dataset import yolo_lines_to_boxes
from metrics import box_iou
from model import COCO_BOAT_CLASS_ID, BoatDetector


def draw_boxes(frame, boxes, color, label):
    out = frame.copy()
    for xmin, ymin, xmax, ymax in boxes:
        cv2.rectangle(out, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
    cv2.putText(out, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def mean_best_iou(gt_boxes, pred_boxes):
    if not gt_boxes:
        return -1.0
    if not pred_boxes:
        return 0.0
    return sum(max(box_iou(g, p) for p in pred_boxes) for g in gt_boxes) / len(gt_boxes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data/datasets/seaships_sample/images/val")
    parser.add_argument("--labels", default="data/datasets/seaships_sample/labels/val")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--output", default="data/outputs/boat_detection_overlays")
    parser.add_argument("--examples-dir", default=None)
    args = parser.parse_args()

    weights = args.checkpoint or "yolov8n.pt"
    classes = None if args.checkpoint else [COCO_BOAT_CLASS_ID]
    detector = BoatDetector(weights)

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for img_path in sorted(images_dir.glob("*.jpg")):
        frame = cv2.imread(str(img_path))
        height, width = frame.shape[:2]
        label_path = labels_dir / f"{img_path.stem}.txt"
        gt_lines = ([line for line in label_path.read_text().splitlines() if line.strip()]
                    if label_path.exists() else [])
        gt_boxes = yolo_lines_to_boxes(gt_lines, width, height)

        preds = detector.predict(frame, conf=args.conf, classes=classes)[0]
        pred_boxes = preds.boxes.xyxy.tolist() if len(preds.boxes) else []

        score = mean_best_iou(gt_boxes, pred_boxes)

        pred_panel = draw_boxes(frame, pred_boxes, (0, 0, 255), "predicted")
        gt_panel = draw_boxes(frame, gt_boxes, (0, 255, 0), "ground truth")
        combined = cv2.hconcat([pred_panel, gt_panel])

        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), combined)
        results.append((score, img_path.name, out_path))

    print(f"Wrote {len(results)} overlays to {output_dir}")

    scored = sorted([r for r in results if r[0] >= 0], key=lambda r: r[0])
    if scored:
        worst, median, best = scored[0], scored[len(scored) // 2], scored[-1]
        print(f"Best:   {best[1]} (mean best IoU {best[0]:.3f})")
        print(f"Median: {median[1]} (mean best IoU {median[0]:.3f})")
        print(f"Worst:  {worst[1]} (mean best IoU {worst[0]:.3f})")

        if args.examples_dir:
            examples_dir = Path(args.examples_dir)
            examples_dir.mkdir(parents=True, exist_ok=True)
            for label, (score, name, path) in zip(
                ["best", "median", "worst"], [best, median, worst]
            ):
                dest = examples_dir / f"boat_detection_{label}.jpg"
                cv2.imwrite(str(dest), cv2.imread(str(path)))
                print(f"Copied example '{label}' -> {dest}")


if __name__ == "__main__":
    main()
