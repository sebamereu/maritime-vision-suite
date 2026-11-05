"""Wrap segmentation/'s trained TinyUNet to produce a water-only mask
for a frame, so texture/optical-flow roughness scoring can be computed
only where there actually is water -- the fix for the "sky/coastline
flagged as gust" failure mode documented in the README.

Not a new model: reuses the existing sea/sky/obstacle segmentation
checkpoint from segmentation/src/model.py (water = class 1, see
segmentation/src/dataset.py). This module only adds the inference +
upsampling plumbing to turn that into a boolean mask at the original
frame resolution.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_SEGMENTATION_SRC = Path(__file__).resolve().parents[2] / "segmentation" / "src"
if str(_SEGMENTATION_SRC) not in sys.path:
    sys.path.insert(0, str(_SEGMENTATION_SRC))

from dataset import NUM_CLASSES
from model import TinyUNet

OBSTACLE_CLASS = 0  # see segmentation/src/dataset.py CLASS_NAMES
WATER_CLASS = 1
SKY_CLASS = 2


class WaterMasker:
    def __init__(self, checkpoint_path: str, size: tuple[int, int] = (192, 256)):
        """`size` must match what the checkpoint was trained at
        (height, width) -- segmentation/scripts/train_segmentation.py's
        default, 192x256."""
        self.height, self.width = size
        self.model = TinyUNet(num_classes=NUM_CLASSES)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.model.eval()

    def predict_classes(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return the full per-pixel class map (0=obstacle, 1=water,
        2=sky) at the frame's own resolution."""
        small = cv2.resize(frame_bgr, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(tensor)
            pred_small = logits.argmax(dim=1).squeeze(0).numpy().astype(np.uint8)

        return cv2.resize(pred_small, (frame_bgr.shape[1], frame_bgr.shape[0]),
                           interpolation=cv2.INTER_NEAREST)

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return a boolean water mask at the frame's own resolution."""
        return self.predict_classes(frame_bgr) == WATER_CLASS
