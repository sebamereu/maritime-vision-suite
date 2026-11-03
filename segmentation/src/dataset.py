"""PyTorch dataset for MaSTr1325-style sea/sky/obstacle segmentation.

Mask labels on disk: 0=obstacle, 1=water, 2=sky, 4=ignore (see
ground_truth.py). Remapped here to contiguous class indices for training
with nn.CrossEntropyLoss(ignore_index=...):
    0 -> 0 (obstacle), 1 -> 1 (water), 2 -> 2 (sky), 4 -> IGNORE_INDEX
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IGNORE_INDEX = 255
_LABEL_MAP = {0: 0, 1: 1, 2: 2, 4: IGNORE_INDEX}
NUM_CLASSES = 3
CLASS_NAMES = ("obstacle", "water", "sky")


def remap_mask(raw_mask: np.ndarray) -> np.ndarray:
    out = np.full_like(raw_mask, IGNORE_INDEX, dtype=np.uint8)
    for src, dst in _LABEL_MAP.items():
        out[raw_mask == src] = dst
    return out


class MaSTrDataset(Dataset):
    def __init__(self, image_dir, mask_dir, stems, size=(192, 256)):
        """
        stems: list of file stems (e.g. "0014") to include, so callers
               control the train/val split rather than the dataset
               silently deciding it.
        size:  (height, width) images/masks are resized to.
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.stems = list(stems)
        self.height, self.width = size

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        img_path = self.image_dir / f"{stem}.jpg"
        mask_path = self.mask_dir / f"{stem}m.png"

        frame_bgr = cv2.imread(str(img_path))
        if frame_bgr is None:
            raise FileNotFoundError(img_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(mask_path)

        frame_bgr = cv2.resize(frame_bgr, (self.width, self.height),
                                interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.width, self.height),
                           interpolation=cv2.INTER_NEAREST)
        mask = remap_mask(mask)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(frame_rgb.transpose(2, 0, 1)).contiguous()
        mask_tensor = torch.from_numpy(mask.astype(np.int64))

        return image_tensor, mask_tensor
