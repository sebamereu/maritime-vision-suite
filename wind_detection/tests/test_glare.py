import numpy as np
from glare import specular_mask


def _frame_with_glint_band_and_whitecaps(height=200, width=300):
    """Dark-ish water background, one large contiguous bright/desaturated
    band (glint) plus several small scattered bright/desaturated dots
    (whitecaps)."""
    frame = np.full((height, width, 3), (90, 70, 40), dtype=np.uint8)  # dark bluish water (BGR)

    # Large contiguous glint band: bright, low-saturation (near-white)
    frame[20:60, 20:280] = (230, 225, 220)

    # Small scattered whitecaps: also bright/near-white, but tiny and separate
    for cy, cx in [(100, 50), (120, 150), (150, 220), (170, 90)]:
        frame[cy:cy + 4, cx:cx + 4] = (235, 230, 225)

    return frame


def test_large_contiguous_bright_region_is_flagged_as_glare():
    frame = _frame_with_glint_band_and_whitecaps()
    mask = specular_mask(frame, min_area_fraction=0.01)

    # the glint band (rows 20-60, cols 20-280) should be substantially flagged
    band_region = mask[20:60, 20:280]
    assert band_region.mean() > 0.5


def test_small_scattered_bright_spots_are_not_flagged():
    frame = _frame_with_glint_band_and_whitecaps()
    mask = specular_mask(frame, min_area_fraction=0.01)

    # each small whitecap patch is 4x4=16px, far below 1% of 200*300=600px frame
    for cy, cx in [(100, 50), (120, 150), (150, 220), (170, 90)]:
        assert not mask[cy:cy + 4, cx:cx + 4].any()


def test_flat_frame_has_no_glare():
    flat = np.full((100, 100, 3), (90, 70, 40), dtype=np.uint8)
    mask = specular_mask(flat)
    assert not mask.any()


def test_output_shape_matches_input():
    frame = _frame_with_glint_band_and_whitecaps(height=80, width=120)
    mask = specular_mask(frame)
    assert mask.shape == (80, 120)
