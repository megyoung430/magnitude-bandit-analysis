import math

import numpy as np
import pytest

from src.behavior_analysis.stats import mean_se_n, nan_standard_error

nan = float("nan")


def test_nan_standard_error_uses_the_non_nan_rows_of_each_column():
    stack = np.array([[1.0, 1.0, nan], [3.0, nan, nan], [nan, nan, 5.0]])
    se = nan_standard_error(stack)
    assert se[0] == pytest.approx(1.0)  # sd 1.414 / sqrt(2)
    assert np.isnan(se[1]) and np.isnan(se[2])  # one or no values: never 0


def test_nan_standard_error_of_a_1d_input_is_one_value_over_all_elements():
    se = nan_standard_error(np.array([1.0, 3.0, nan]))
    assert se.shape == ()
    assert float(se) == pytest.approx(1.0)
    assert math.isnan(float(nan_standard_error(np.array([1.0, nan]))))


def test_mean_se_n_counts_only_non_nan_values():
    mean, se, n = mean_se_n([1.0, 3.0, nan])
    assert (mean, n) == (2.0, 2)
    assert se == pytest.approx(1.0)


def test_mean_se_n_with_fewer_than_two_values():
    mean, se, n = mean_se_n([4.0, nan])
    assert (mean, n) == (4.0, 1) and math.isnan(se)
    mean, se, n = mean_se_n([nan])
    assert n == 0 and math.isnan(mean) and math.isnan(se)
    assert n == mean_se_n([])[2] == 0
