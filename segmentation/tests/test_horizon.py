import numpy as np
from horizon import detect_horizon


def _synthetic_frame(height=384, width=512, horizon_row=150, noise=6):
    """A flat, noisy-but-clean synthetic horizon: bright band on top
    (sky), darker band below (water), nothing else in the scene."""
    rng = np.random.default_rng(0)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    sky_color = np.array([235, 220, 180], dtype=np.int16)  # BGR, bright
    water_color = np.array([90, 70, 40], dtype=np.int16)  # BGR, dark

    for row in range(height):
        base = sky_color if row < horizon_row else water_color
        jitter = rng.integers(-noise, noise + 1, size=3)
        frame[row, :] = np.clip(base + jitter, 0, 255)

    return frame


def test_detect_horizon_matches_synthetic_flat_line():
    horizon_row = 150
    frame = _synthetic_frame(horizon_row=horizon_row)

    predicted = detect_horizon(frame)

    assert predicted.shape == (frame.shape[1],)
    # A flat, high-contrast horizon should be recovered within a few
    # pixels across (almost) the whole width.
    error = np.abs(predicted - horizon_row)
    assert np.median(error) < 5


def test_detect_horizon_handles_blank_frame_without_crashing():
    blank = np.full((384, 512, 3), 128, dtype=np.uint8)
    predicted = detect_horizon(blank)
    assert predicted.shape == (512,)
    # No contrast at all -> falls back to the midpoint, not a crash or NaN
    assert np.all(np.isfinite(predicted))


def test_detect_horizon_output_is_within_image_bounds():
    frame = _synthetic_frame()
    predicted = detect_horizon(frame)
    assert np.all(predicted >= 0)
    assert np.all(predicted <= frame.shape[0] - 1)
