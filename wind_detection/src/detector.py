"""Combine texture roughness and flow variability into a single
gust-likelihood map, and threshold it into gust/calm/neutral zones.

This is deliberately unsupervised (see README): no gust/lull dataset
exists to train a classifier on, and a "ground truth" label for one
frame would have to come from a boat's anemometer at that exact place
and moment, which this project doesn't have. What's validated here is
internal consistency (does the score respond to the phenomenon it's
supposed to, on a frame-by-frame basis) -- not a measured accuracy
number against reality, which scripts/validate_wind_zones.py is
explicit about not providing.
"""
from dataclasses import dataclass

import numpy as np
from optical_flow import dense_flow, flow_variability_map
from texture import roughness_map


@dataclass
class GustMap:
    score: np.ndarray          # combined z-scored gust-likelihood, same H x W as the frame
    texture_z: np.ndarray      # texture.py's contribution alone
    flow_z: np.ndarray | None  # optical_flow.py's contribution alone (None for a single frame)


def combine(texture_z: np.ndarray, flow_z: np.ndarray | None,
            texture_weight: float = 0.6, flow_weight: float = 0.4) -> np.ndarray:
    if flow_z is None:
        return texture_z
    return texture_weight * texture_z + flow_weight * flow_z


def detect_single_frame(frame_bgr: np.ndarray, patch_size: int = 32) -> GustMap:
    """Texture-only detection -- usable on a single still image, no
    previous frame required. Weaker signal than detect_frame_pair
    (see module docstring: only one of the two independent cues)."""
    tz = roughness_map(frame_bgr, patch_size=patch_size)
    return GustMap(score=tz, texture_z=tz, flow_z=None)


def detect_frame_pair(prev_gray: np.ndarray, curr_frame_bgr: np.ndarray, curr_gray: np.ndarray,
                       patch_size: int = 32,
                       texture_weight: float = 0.6, flow_weight: float = 0.4) -> GustMap:
    """Full two-signal detection between a previous grayscale frame and
    the current frame (both color, for texture, and grayscale, for
    flow -- caller already has both when iterating a video)."""
    tz = roughness_map(curr_frame_bgr, patch_size=patch_size)
    flow = dense_flow(prev_gray, curr_gray)
    fz = flow_variability_map(flow, patch_size=patch_size)
    score = combine(tz, fz, texture_weight, flow_weight)
    return GustMap(score=score, texture_z=tz, flow_z=fz)


def zone_mask(gust_map: GustMap, gust_threshold: float = 0.75, calm_threshold: float = -0.75) -> np.ndarray:
    """Classify each pixel as 1 (likely gust), -1 (likely calm/lull), or
    0 (neutral/unclear), using thresholds on the frame's own z-scored
    distribution -- not fixed absolute values, so this doesn't need
    retuning as lighting or sea state changes between videos.
    """
    mask = np.zeros_like(gust_map.score, dtype=np.int8)
    mask[gust_map.score >= gust_threshold] = 1
    mask[gust_map.score <= calm_threshold] = -1
    return mask
