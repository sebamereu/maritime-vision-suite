"""Box IoU for the "boat" class.

A from-scratch AP/mAP implementation used to live here too (greedy IoU
matching + a precision/recall AP integral). It was mathematically
correct -- verified against ultralytics' own ap_per_class fed the same
inputs -- but on this dataset's tiny validation set (34 ground-truth
boxes, with dozens of overlapping candidate boxes on a few cluttered
images) the AP it produced was extremely sensitive to which single
detection got credited as the match in those cluttered cases, landing
far from ultralytics' own number (0.43 vs 0.975 on the same checkpoint)
despite both being "correct". See evaluate_boat_detection.py, which now
uses ultralytics' own val() instead. box_iou survives because
visualize_boat_detection.py still wants it for a simple, honestly-
labeled per-image ranking (mean best IoU), not a reported metric.
"""


def box_iou(a, b):
    """a, b: (xmin, ymin, xmax, ymax). Returns IoU in [0, 1]."""
    xmin = max(a[0], b[0])
    ymin = max(a[1], b[1])
    xmax = min(a[2], b[2])
    ymax = min(a[3], b[3])
    inter = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
