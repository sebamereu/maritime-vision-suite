import pytest
from fps import compute_fps


def test_first_frame_returns_zero():
    assert compute_fps(None, 10.0) == 0.0


def test_normal_delta_returns_expected_fps():
    # half a second between frames -> 2.0 fps
    assert compute_fps(10.0, 10.5) == pytest.approx(2.0)


def test_zero_delta_returns_zero_instead_of_dividing_by_zero():
    assert compute_fps(10.0, 10.0) == 0.0


def test_negative_delta_returns_zero():
    # e.g. a monotonic clock reset between frames
    assert compute_fps(10.0, 9.0) == 0.0
