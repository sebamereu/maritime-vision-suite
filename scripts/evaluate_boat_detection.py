#!/usr/bin/env python3
"""Evaluate a BoatDetector against the SeaShips sample's val split,
using ultralytics' own val() (standard mAP50/mAP50-95/precision/recall)
rather than a hand-rolled metric.

An earlier version of this script computed its own IoU-matching AP; on
this dataset's tiny validation set (34 ground-truth boxes) that turned
out to be extremely sensitive to which single detection gets credited
as the match on a handful of hard/cluttered images, producing AP
numbers far off from ultralytics' own (0.43 vs 0.975 on the same
checkpoint) despite the matching and AP-integral logic itself being
correct (verified against ultralytics' internal implementation
directly). Small eval sets are simply not a regime where a from-scratch
mAP implementation is trustworthy -- using the library's own, far
better-tested evaluator avoids re-litigating that instability.

With no --checkpoint, evaluates the stock COCO-pretrained yolov8n.pt
zero-shot: data_zeroshot.yaml (written by fetch_seaships_sample.py)
relabels the ground truth at COCO's own "boat" class id so the
pretrained model's class indices line up, letting the same standard
metric score it fairly, before any fine-tuning happens. With
--checkpoint, evaluates a fine-tuned checkpoint against the normal
(class 0) data.yaml instead.

Usage:
    python scripts/evaluate_boat_detection.py
    python scripts/evaluate_boat_detection.py --checkpoint data/checkpoints/yolov8n_boats.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "boat_detection" / "src"))
from model import BoatDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/datasets/seaships_sample/data.yaml")
    parser.add_argument("--zeroshot-data", default="data/datasets/seaships_sample_zeroshot/data_zeroshot.yaml")
    parser.add_argument("--checkpoint", default=None,
                         help="fine-tuned weights; default evaluates stock yolov8n.pt zero-shot")
    args = parser.parse_args()

    weights = args.checkpoint or "yolov8n.pt"
    data_yaml = args.data if args.checkpoint else args.zeroshot_data
    detector = BoatDetector(weights)

    metrics = detector.val(data_yaml)

    label = (f"fine-tuned ({weights})" if args.checkpoint
             else f"zero-shot pretrained ({weights}, COCO 'boat' class)")
    print(f"Model: {label}")
    print(f"  mAP50:     {metrics.box.map50:.4f}")
    print(f"  mAP50-95:  {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
