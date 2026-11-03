import numpy as np
from ground_truth import OBSTACLE, SKY, WATER, horizon_from_mask


def test_clean_flat_horizon_recovered_for_every_column():
    mask = np.full((100, 10), WATER, dtype=np.uint8)
    mask[:40, :] = SKY  # sky rows 0-39, water rows 40-99

    valid_columns, horizon_rows = horizon_from_mask(mask)

    assert len(valid_columns) == 10
    assert np.all(horizon_rows == 40)


def test_column_with_obstacle_is_excluded():
    mask = np.full((100, 10), WATER, dtype=np.uint8)
    mask[:40, :] = SKY
    mask[35:45, 5] = OBSTACLE  # obstacle straddles the horizon in column 5

    valid_columns, _ = horizon_from_mask(mask)

    assert 5 not in valid_columns
    assert len(valid_columns) == 9


def test_column_missing_sky_or_water_is_excluded():
    mask = np.full((100, 10), WATER, dtype=np.uint8)
    mask[:40, :] = SKY
    mask[:, 0] = SKY  # no water at all in column 0

    valid_columns, _ = horizon_from_mask(mask)

    assert 0 not in valid_columns
