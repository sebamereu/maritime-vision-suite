#!/usr/bin/env python3
"""Train the TinyUNet sea/sky/obstacle segmentation model on MaSTr1325.

Not run in CI (too slow/heavy for every push) — run it locally, it saves
a checkpoint and prints final val metrics, which go into the README by
hand alongside the honest caveats.

Usage:
    python scripts/train_segmentation.py --epochs 20
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "segmentation" / "src"))
from dataset import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES, MaSTrDataset
from metrics import mean_iou, per_class_iou, pixel_accuracy
from model import TinyUNet


def split_stems(images_dir: Path, val_fraction: float, seed: int):
    stems = sorted(p.stem for p in images_dir.glob("*.jpg"))
    rng = random.Random(seed)
    rng.shuffle(stems)
    n_val = max(1, int(len(stems) * val_fraction))
    return stems[n_val:], stems[:n_val]


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(masks.numpy())
    preds = np.concatenate([p.reshape(-1) for p in all_preds])
    targets = np.concatenate([t.reshape(-1) for t in all_targets])
    return preds, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data/datasets/mastr1325_sample/images")
    parser.add_argument("--masks", default="data/datasets/mastr1325_sample/masks")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, nargs=2, default=[192, 256],
                         metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--checkpoint", default="data/checkpoints/tiny_unet.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, threads: {torch.get_num_threads()}")

    train_stems, val_stems = split_stems(Path(args.images), args.val_fraction, args.seed)
    print(f"Train: {len(train_stems)} images, Val: {len(val_stems)} images")

    train_ds = MaSTrDataset(args.images, args.masks, train_stems, size=tuple(args.size))
    val_ds = MaSTrDataset(args.images, args.masks, val_stems, size=tuple(args.size))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4)

    model = TinyUNet(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    best_miou = -1.0
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)
        epoch_loss /= len(train_ds)

        preds, targets = evaluate(model, val_loader, device)
        val_miou = mean_iou(preds, targets, NUM_CLASSES, IGNORE_INDEX)
        val_acc = pixel_accuracy(preds, targets, IGNORE_INDEX)

        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss {epoch_loss:.4f} "
              f"| val_mIoU {val_miou:.4f} | val_pixel_acc {val_acc:.4f}")

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), checkpoint_path)

    elapsed = time.time() - start
    print(f"\nTraining done in {elapsed / 60:.1f} min. Best val mIoU: {best_miou:.4f}")
    print(f"Checkpoint saved to {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    preds, targets = evaluate(model, val_loader, device)
    ious = per_class_iou(preds, targets, NUM_CLASSES, IGNORE_INDEX)
    print("\nFinal per-class IoU (best checkpoint):")
    for name, iou in zip(CLASS_NAMES, ious):
        print(f"  {name:10s}: {iou:.4f}" if not np.isnan(iou) else f"  {name:10s}: n/a")
    print(f"  mean       : {mean_iou(preds, targets, NUM_CLASSES, IGNORE_INDEX):.4f}")
    print(f"  pixel acc  : {pixel_accuracy(preds, targets, IGNORE_INDEX):.4f}")


if __name__ == "__main__":
    main()
