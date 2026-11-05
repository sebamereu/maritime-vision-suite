"""Restrict a water mask to open water, excluding a buffer near the
shoreline -- the fix attempt for the surf-foam failure mode: breaking
waves on shallow water near the coast produce the same kind of
high-frequency texture and locally inconsistent motion that wind over
open water does, and the segmentation model correctly calls that water
(it is), so masking to water-only doesn't separate the two. Proximity
to the shore is a cheap, physically-motivated signal that does: surf
break only happens near shallow water close to land, wind gusts can
happen anywhere.

Known simplification, not hidden: "obstacle" in the segmentation
model's output also covers things like buoys and boats out in open
water, not only the coastline -- this buffers a ring around every
detected obstacle, not specifically "land". For a frame dominated by
one real shoreline (the case this was built and tested against) that's
the effect that matters; a stray boat picking up a small open-water
exclusion ring around itself is a minor, acceptable side effect, not a
correctness bug.
"""
import cv2
import numpy as np

# Duplicated from water_mask.py rather than imported, so this module
# (pure array math) stays usable without torch installed -- same
# reasoning as detector.py not importing water_mask directly.
OBSTACLE_CLASS = 0
WATER_CLASS = 1


def open_water_mask(class_map: np.ndarray, min_shore_distance_px: int = 60) -> np.ndarray:
    """Return a boolean mask: water pixels at least `min_shore_distance_px`
    away (Euclidean, in pixels) from the nearest obstacle/shoreline
    pixel. `class_map` is WaterMasker.predict_classes()'s output.
    """
    is_obstacle = (class_map == OBSTACLE_CLASS).astype(np.uint8)

    # cv2.distanceTransform gives, for each nonzero pixel, the distance
    # to the nearest zero pixel -- so obstacle pixels are the zeros we
    # measure distance to.
    src = np.where(is_obstacle, 0, 255).astype(np.uint8)
    distance_to_obstacle = cv2.distanceTransform(src, cv2.DIST_L2, 5)

    is_water = class_map == WATER_CLASS
    return is_water & (distance_to_obstacle >= min_shore_distance_px)
