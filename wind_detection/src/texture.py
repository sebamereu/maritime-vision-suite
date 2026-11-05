"""Relative water-surface roughness from a single frame.

Per the roadmap: a gust ("cat's paw") reads as a rougher/more detailed
patch of water than its surroundings, and a lull reads as smoother. This
measures that *locally*, within one frame, rather than against a fixed
brightness/contrast threshold — a threshold set for one lighting
condition breaks the moment a cloud shadow or sun glare crosses the
frame, since those change absolute brightness without changing surface
roughness at all.
"""
import cv2
import numpy as np
from zscore import zscore


def roughness_energy(frame_bgr: np.ndarray, patch_size: int = 32, blur_sigma: float = 8.0) -> np.ndarray:
    """Raw (un-normalized) local texture energy, same height/width as
    the input. Higher = more small-scale detail in that patch.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # High-pass: subtract a heavily blurred copy. What's left is
    # small-scale detail (ripples, foam texture) -- large-scale
    # brightness gradients (glare, shadow) are removed with the blur.
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=blur_sigma)
    highpass = gray - blurred

    # Local energy of that detail, averaged over a patch -- a single
    # bright pixel isn't "rough", a patch full of small bright/dark
    # transitions is.
    return cv2.boxFilter(highpass ** 2, ddepth=-1, ksize=(patch_size, patch_size))


def roughness_map(frame_bgr: np.ndarray, patch_size: int = 32, blur_sigma: float = 8.0,
                   water_mask: np.ndarray | None = None) -> np.ndarray:
    """Return a per-pixel z-scored roughness map, same height/width as
    the input. Positive values = rougher than this frame's average,
    negative = smoother. Z-scoring per frame (not a fixed scale) is
    what makes this relative rather than absolute.

    If `water_mask` is given (same H x W, truthy = water), the mean/std
    used for z-scoring are computed from water pixels only, and
    non-water pixels are reported as 0 (neutral) rather than scored
    against a distribution they aren't part of.
    """
    energy = roughness_energy(frame_bgr, patch_size=patch_size, blur_sigma=blur_sigma)
    return zscore(energy, water_mask)
