"""Shared frame-relative z-score, optionally restricted to a mask.

Pulled out of texture.py/optical_flow.py so both use the exact same
masking logic: when a water mask is given, mean/std (and therefore
what counts as "rough" or "calm") are computed only from water pixels.
Without this, a frame with sky or coastline in it would have its
z-scores skewed by regions that were never water to begin with --
exactly the failure mode documented in the README (sky/coastline
flagged as "gust").
"""
import numpy as np


def zscore(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Z-score `values` against the mean/std of values[mask] (or the
    whole array if mask is None). Pixels outside the mask are set to 0
    (neutral) in the output, not left to the std/mean of a region they
    don't belong to.
    """
    if mask is None:
        sample = values
    else:
        sample = values[mask.astype(bool)]
        if sample.size == 0:
            return np.zeros_like(values)

    mean, std = float(sample.mean()), float(sample.std())
    if std < 1e-6:
        return np.zeros_like(values)

    z = (values - mean) / std
    if mask is not None:
        z = np.where(mask.astype(bool), z, 0.0)
    return z
