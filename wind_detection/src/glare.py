"""Detect and exclude sun glare (specular reflection) from gust
scoring -- the fix attempt for the failure mode found by testing on
the "success" example itself (see README): a band of sun glint read as
a confident gust, when it's reflected sunlight, not necessarily wind.

The physical distinction from a genuine whitecap: glint is both bright
AND desaturated (specular reflection washes out color) AND forms a
large, contiguous region (it follows the sun-water-camera geometry, a
broad band or patch, not scattered pixels). A real whitecap is also
bright/desaturated, but typically appears as small, scattered patches.
Size of the connected region is what separates the two here -- value
and saturation alone would also flag genuine whitecaps, which this
project has no interest in suppressing (they're a real wind signal).
"""
import cv2
import numpy as np
from zscore import zscore


def specular_mask(frame_bgr: np.ndarray, min_area_fraction: float = 0.01,
                   score_threshold: float = 1.0) -> np.ndarray:
    """Return a boolean mask: pixels that are part of a large,
    contiguous bright+desaturated region (likely glare), at the
    frame's own resolution.

    `min_area_fraction`: a connected bright/desaturated region smaller
    than this fraction of the frame is treated as a whitecap, not
    glare, and left unmasked.
    `score_threshold`: z-score cutoff (relative to this frame, not an
    absolute brightness) for "candidate" bright+desaturated pixels
    before the size filter is applied.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    value, saturation = hsv[..., 2], hsv[..., 1]

    # High value, low saturation -> positive score. z-scored per frame,
    # same reasoning as texture.py: an absolute brightness cutoff would
    # need retuning for every lighting condition.
    specular_score = zscore(value) - zscore(saturation)
    candidates = (specular_score >= score_threshold).astype(np.uint8)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    min_area = min_area_fraction * frame_bgr.shape[0] * frame_bgr.shape[1]

    mask = np.zeros(candidates.shape, dtype=bool)
    for label in range(1, n_labels):  # label 0 is background
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == label] = True
    return mask
