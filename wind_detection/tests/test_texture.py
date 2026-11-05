import numpy as np
from texture import roughness_map


def _half_rough_half_smooth(height=128, width=128, seed=0):
    """Left half: high-frequency noise (rough). Right half: flat gray
    (smooth). Both halves share the same mean brightness, so a
    brightness-threshold detector would see no difference -- only a
    texture-based one should."""
    rng = np.random.default_rng(seed)
    frame = np.full((height, width, 3), 120, dtype=np.uint8)
    noise = rng.integers(-40, 41, size=(height, width // 2, 3))
    frame[:, : width // 2] = np.clip(120 + noise, 0, 255).astype(np.uint8)
    return frame


def test_rough_half_scores_higher_than_smooth_half():
    frame = _half_rough_half_smooth()
    z = roughness_map(frame, patch_size=16)

    rough_region = z[:, :40]
    smooth_region = z[:, -40:]

    assert rough_region.mean() > smooth_region.mean()


def test_flat_frame_gives_zero_map_without_crashing():
    flat = np.full((64, 64, 3), 100, dtype=np.uint8)
    z = roughness_map(flat)

    assert z.shape == (64, 64)
    assert np.all(z == 0)  # std is ~0 on a perfectly flat frame -> defined fallback, not NaN/inf


def test_output_shape_matches_input():
    frame = _half_rough_half_smooth(height=96, width=160)
    z = roughness_map(frame)
    assert z.shape == (96, 160)
