"""Thin wrapper around ultralytics YOLO for boat detection.

Wraps weight loading, zero-shot pretrained inference (stock COCO "boat"
class) and fine-tuning/evaluation on the SeaShips sample, so
train/evaluate/visualize scripts share one place that calls into
ultralytics instead of three slightly different copies of the same API
calls.
"""
import torch
from ultralytics import YOLO

COCO_BOAT_CLASS_ID = 8  # "boat" in the 80-class COCO set YOLOv8 ships with


def default_device():
    """Ultralytics defaults to CPU and doesn't probe for Apple Silicon's
    GPU on its own -- it has to be asked for explicitly via device="mps".
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BoatDetector:
    def __init__(self, weights="yolov8n.pt"):
        self.model = YOLO(weights)

    def predict(self, source, conf=0.25, classes=None, device=None):
        return self.model.predict(source=source, conf=conf, classes=classes,
                                   device=device or default_device(), verbose=False)

    def train(self, data_yaml, epochs=30, imgsz=640, seed=42,
              project="runs/detect", name="boats", device=None):
        return self.model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, seed=seed,
                                 project=project, name=name, device=device or default_device(),
                                 verbose=False)

    def val(self, data_yaml, imgsz=640, classes=None, device=None):
        return self.model.val(data=data_yaml, imgsz=imgsz, classes=classes,
                               device=device or default_device(), verbose=False)
