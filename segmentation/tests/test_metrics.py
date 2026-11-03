import numpy as np
from metrics import mean_iou, per_class_iou, pixel_accuracy


def test_perfect_prediction_gives_iou_one():
    target = np.array([0, 0, 1, 1, 2, 2])
    pred = target.copy()

    ious = per_class_iou(pred, target, num_classes=3, ignore_index=255)

    assert np.allclose(ious, [1.0, 1.0, 1.0])
    assert mean_iou(pred, target, 3, 255) == 1.0
    assert pixel_accuracy(pred, target, 255) == 1.0


def test_completely_wrong_prediction_gives_iou_zero_for_present_classes():
    target = np.array([0, 0, 1, 1])
    pred = np.array([1, 1, 0, 0])

    ious = per_class_iou(pred, target, num_classes=2, ignore_index=255)

    assert np.allclose(ious, [0.0, 0.0])


def test_ignore_index_is_excluded_from_scoring():
    target = np.array([0, 255, 1, 255])
    pred = np.array([0, 1, 1, 0])  # wrong guesses on ignored pixels don't count

    assert pixel_accuracy(pred, target, ignore_index=255) == 1.0


def test_class_absent_from_union_is_nan_not_zero():
    target = np.array([0, 0, 0])
    pred = np.array([0, 0, 0])

    ious = per_class_iou(pred, target, num_classes=2, ignore_index=255)

    assert ious[0] == 1.0
    assert np.isnan(ious[1])  # class 1 never appears in pred or target
