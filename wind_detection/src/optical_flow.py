"""Local ripple-motion variability between two consecutive frames.

A second, independent signal from texture.py's single-frame roughness:
a gust patch's ripples move differently frame-to-frame than calm
water, even when lighting makes the two look deceptively similar in a
single still frame (e.g. a cloud shadow sitting over a genuinely calm
patch can look "textured" in brightness terms without actually moving
like one).
"""
import cv2
import numpy as np
from zscore import zscore


def dense_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """Farneback dense optical flow between two grayscale frames.
    Returns an (H, W, 2) array of (dx, dy) per pixel.
    """
    return cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )


def flow_local_variance(flow: np.ndarray, patch_size: int = 32) -> np.ndarray:
    """Raw (un-normalized) local variance of flow magnitude, same
    height/width as the flow field. Higher = ripple motion within that
    patch is locally inconsistent (characteristic of rougher water);
    lower = motion is locally uniform. This is about consistency of
    motion within a patch, not raw speed, so it isn't fooled by the
    whole frame drifting uniformly (e.g. camera motion).
    """
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    mean_sq = cv2.boxFilter(magnitude ** 2, ddepth=-1, ksize=(patch_size, patch_size))
    mean = cv2.boxFilter(magnitude, ddepth=-1, ksize=(patch_size, patch_size))
    return np.clip(mean_sq - mean ** 2, 0, None)


def flow_variability_map(flow: np.ndarray, patch_size: int = 32,
                          water_mask: np.ndarray | None = None) -> np.ndarray:
    """Return a per-pixel z-scored map of local flow-magnitude variance.
    See texture.roughness_map for the water_mask semantics -- same
    pattern: normalize only against water pixels when a mask is given.
    """
    local_var = flow_local_variance(flow, patch_size=patch_size)
    return zscore(local_var, water_mask)
