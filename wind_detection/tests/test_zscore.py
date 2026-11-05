import numpy as np
from zscore import zscore


def test_zscore_without_mask_normalizes_whole_array():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    z = zscore(values)
    assert np.isclose(z.mean(), 0.0, atol=1e-6)


def test_zscore_with_mask_ignores_values_outside_it():
    # Two outlier values that would skew the stats if included
    values = np.array([1.0, 2.0, 3.0, 1000.0, -1000.0])
    mask = np.array([True, True, True, False, False])

    z = zscore(values, mask)

    # masked-out positions are reported as neutral (0), not scored
    assert z[3] == 0.0
    assert z[4] == 0.0
    # in-mask values are z-scored against {1,2,3} only, mean=2, std=sqrt(2/3)
    expected_std = np.std([1.0, 2.0, 3.0])
    assert np.isclose(z[0], (1.0 - 2.0) / expected_std)


def test_zscore_flat_array_returns_zeros_not_nan():
    values = np.full(10, 5.0)
    z = zscore(values)
    assert np.all(z == 0)


def test_zscore_empty_mask_returns_zeros():
    values = np.array([1.0, 2.0, 3.0])
    mask = np.array([False, False, False])
    z = zscore(values, mask)
    assert np.all(z == 0)
