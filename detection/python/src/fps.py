# fps.py
"""Pure helper for computing an instantaneous FPS value from two timestamps.

Extracted out of main.py so the frame-timing logic can be unit tested
without a real video source or display.
"""


def compute_fps(prev_time, current_time: float) -> float:
    """Return the instantaneous FPS given the previous and current frame
    timestamps (seconds).

    Returns 0.0 for the first frame (prev_time is None) and for a
    non-positive delta (clock reset or duplicate timestamp), instead of
    raising a ZeroDivisionError.
    """
    if prev_time is None:
        return 0.0
    delta = current_time - prev_time
    return 1.0 / delta if delta > 0 else 0.0
