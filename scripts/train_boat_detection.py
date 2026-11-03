#!/usr/bin/env python3
"""Fine-tune yolov8n on the SeaShips sample's single "boat" class.

Not run in CI (too slow/heavy for every push) -- run it locally, it
saves a checkpoint and prints final val metrics, which go into the
README by hand alongside the honest caveats.

Usage:
    python scripts/train_boat_detection.py --epochs 30
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "boat_detection" / "src"))
from model import BoatDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/datasets/seaships_sample/data.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default="data/checkpoints/yolov8n_boats.pt")
    args = parser.parse_args()

    detector = BoatDetector("yolov8n.pt")
    detector.train(args.data, epochs=args.epochs, imgsz=args.imgsz, seed=args.seed)

    best = Path(detector.model.trainer.best)
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path} (from {best})")

    metrics = detector.val(args.data, imgsz=args.imgsz)
    print(f"Final val mAP50: {metrics.box.map50:.4f}, mAP50-95: {metrics.box.map:.4f}, "
          f"precision: {metrics.box.mp:.4f}, recall: {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
