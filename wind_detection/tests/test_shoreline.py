import numpy as np
from shoreline import OBSTACLE_CLASS, WATER_CLASS, open_water_mask


def test_water_far_from_obstacle_is_open_water():
    # left column obstacle (shore), rest water
    class_map = np.full((50, 100), WATER_CLASS, dtype=np.uint8)
    class_map[:, 0] = OBSTACLE_CLASS

    mask = open_water_mask(class_map, min_shore_distance_px=10)

    assert mask[25, 99]  # far right: far from shore, should be open water
    assert not mask[25, 5]  # near shore: should be excluded


def test_sky_pixels_are_never_open_water():
    class_map = np.full((50, 100), 2, dtype=np.uint8)  # all sky
    mask = open_water_mask(class_map, min_shore_distance_px=5)
    assert not mask.any()


def test_distance_threshold_is_respected():
    class_map = np.full((1, 200), WATER_CLASS, dtype=np.uint8)
    class_map[0, 0] = OBSTACLE_CLASS

    mask_strict = open_water_mask(class_map, min_shore_distance_px=100)
    mask_loose = open_water_mask(class_map, min_shore_distance_px=5)

    # a stricter distance threshold can only exclude as much or more water
    assert mask_strict.sum() <= mask_loose.sum()


def test_no_obstacles_at_all_keeps_all_water():
    class_map = np.full((30, 30), WATER_CLASS, dtype=np.uint8)
    mask = open_water_mask(class_map, min_shore_distance_px=50)
    assert mask.all()
