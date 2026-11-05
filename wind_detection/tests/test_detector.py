import numpy as np
from detector import GustMap, combine, detect_frame_pair, detect_single_frame, zone_mask


def test_combine_weights_texture_and_flow():
    tz = np.array([1.0, -1.0])
    fz = np.array([1.0, 1.0])

    combined = combine(tz, fz, texture_weight=0.6, flow_weight=0.4)

    assert np.allclose(combined, [0.6 * 1.0 + 0.4 * 1.0, 0.6 * -1.0 + 0.4 * 1.0])


def test_combine_falls_back_to_texture_only_without_flow():
    tz = np.array([0.3, -0.7])
    assert np.array_equal(combine(tz, None), tz)


def test_zone_mask_thresholds():
    score = np.array([[2.0, 0.0], [-2.0, 0.5]])
    gm = GustMap(score=score, texture_z=score, flow_z=None)

    mask = zone_mask(gm, gust_threshold=1.0, calm_threshold=-1.0)

    assert mask.tolist() == [[1, 0], [-1, 0]]


def test_detect_single_frame_has_no_flow_component():
    frame = np.full((64, 64, 3), 128, dtype=np.uint8)
    gm = detect_single_frame(frame)

    assert gm.flow_z is None
    assert gm.score.shape == (64, 64)


def test_detect_frame_pair_combines_both_signals():
    rng = np.random.default_rng(1)
    frame0 = np.clip(128 + rng.integers(-20, 20, size=(64, 64, 3)), 0, 255).astype(np.uint8)
    frame1 = np.clip(128 + rng.integers(-20, 20, size=(64, 64, 3)), 0, 255).astype(np.uint8)
    gray0 = frame0.mean(axis=2).astype(np.uint8)
    gray1 = frame1.mean(axis=2).astype(np.uint8)

    gm = detect_frame_pair(gray0, frame1, gray1)

    assert gm.flow_z is not None
    assert gm.score.shape == (64, 64)
