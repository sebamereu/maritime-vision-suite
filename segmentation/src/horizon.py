"""Classical sea/sky horizon detection.

Not a learned segmentation model — a geometric baseline: find the
near-horizontal line that best separates the image into an upper (sky)
and lower (water) region, using edge detection + Hough line search.
Deliberately simple and fast (no GPU, no training), to have a measurable
first result before reaching for a CNN.

Known limitation: a single straight line cannot represent a coastline or
an obstacle straddling the horizon — see evaluate_horizon.py, which only
scores columns where the ground truth is a clean sky-then-water
transition, and reports what fraction of columns qualify.
"""
import cv2
import numpy as np


def detect_horizon(frame_bgr: np.ndarray) -> np.ndarray:
    """Return the estimated horizon row (float y) for every column of
    `frame_bgr`, as a 1D array of length `width`.

    Falls back to the image's vertical midpoint if no reliable line can
    be found (near-blank frame, no contrast).
    """
    height, width = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=width // 3,
        maxLineGap=20,
    )

    fallback = np.full(width, height / 2.0, dtype=np.float64)
    if lines is None:
        return fallback

    # Keep only near-horizontal segments (horizon tilt in practice is a
    # few degrees at most, from boat roll) and pick the longest one as
    # the anchor line; average over all segments within a tight angle
    # band of it for robustness against a single noisy detection.
    candidates = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = line
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length < 1e-6:
            continue
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        angle = min(angle, 180 - angle)  # fold into [0, 90]
        if angle > 15:
            continue
        candidates.append((length, x1, y1, x2, y2))

    if not candidates:
        return fallback

    candidates.sort(key=lambda c: c[0], reverse=True)
    xs, ys, weights = [], [], []
    for length, x1, y1, x2, y2 in candidates:
        xs.extend([x1, x2])
        ys.extend([y1, y2])
        weights.extend([length, length])

    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)

    # Weighted least-squares line fit: y = m*x + b
    design = np.vstack([xs, np.ones_like(xs)]).T
    w_sqrt = np.sqrt(weights)
    m, b = np.linalg.lstsq(design * w_sqrt[:, None], ys * w_sqrt, rcond=None)[0]

    columns = np.arange(width, dtype=np.float64)
    horizon = m * columns + b
    return np.clip(horizon, 0, height - 1)
