"""Per-class IoU / mean IoU for semantic segmentation, ignoring a given
label (the dataset's ignore region)."""
import numpy as np


def per_class_iou(pred: np.ndarray, target: np.ndarray, num_classes: int,
                   ignore_index: int) -> np.ndarray:
    """pred, target: integer arrays of the same shape (any shape, e.g.
    HxW or NxHxW). Returns an array of length num_classes; a class with
    no pixels in the union is NaN (excluded from the mean, not scored as
    zero — it simply didn't occur)."""
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    ious = np.full(num_classes, np.nan, dtype=np.float64)
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        intersection = np.logical_and(pred_c, target_c).sum()
        union = np.logical_or(pred_c, target_c).sum()
        if union > 0:
            ious[c] = intersection / union
    return ious


def mean_iou(pred: np.ndarray, target: np.ndarray, num_classes: int,
             ignore_index: int) -> float:
    ious = per_class_iou(pred, target, num_classes, ignore_index)
    valid = ~np.isnan(ious)
    if not valid.any():
        return float("nan")
    return float(np.mean(ious[valid]))


def pixel_accuracy(pred: np.ndarray, target: np.ndarray, ignore_index: int) -> float:
    valid = target != ignore_index
    if not valid.any():
        return float("nan")
    return float((pred[valid] == target[valid]).mean())
