"""Extract a per-column horizon row from MaSTr1325-style semantic masks.

Label values (per the dataset's documentation):
    0 = obstacle/environment, 1 = water, 2 = sky, 4 = ignore
"""
import numpy as np

OBSTACLE, WATER, SKY, IGNORE = 0, 1, 2, 4


def horizon_from_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (valid_columns, horizon_rows) for a semantic mask.

    A column is only considered "valid" ground truth when it is a clean
    sky-then-water transition with no obstacle/ignore pixels: sky from
    row 0 down to some row r-1, water from row r to the bottom. Columns
    with an obstacle straddling the horizon, or any ignore-labelled
    pixels, are excluded rather than guessed at — a straight-line horizon
    detector has no way to be "correct" on them anyway.

    `horizon_rows[i]` is the first water row for `valid_columns[i]`.
    """
    _, width = mask.shape
    valid_columns = []
    horizon_rows = []

    for col in range(width):
        column = mask[:, col]
        if np.any(column == IGNORE) or np.any(column == OBSTACLE):
            continue

        is_sky = column == SKY
        is_water = column == WATER
        if not is_sky.any() or not is_water.any():
            continue

        sky_rows = np.flatnonzero(is_sky)
        water_rows = np.flatnonzero(is_water)
        # Clean transition: every sky pixel strictly above every water
        # pixel in this column, i.e. sky is a prefix and water a suffix.
        if sky_rows[-1] >= water_rows[0]:
            continue

        valid_columns.append(col)
        horizon_rows.append(float(water_rows[0]))

    return np.array(valid_columns, dtype=np.int64), np.array(horizon_rows, dtype=np.float64)
