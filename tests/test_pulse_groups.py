"""Tests for splitting outcome-aligned trial windows by the number of reward pulses."""

import numpy as np
import pytest

from src.photometry_analysis.pulse_groups import (
    categories_with_min_trials,
    pulse_categories,
    pulse_response_correlation,
    split_by_pulses,
)

N_TIME = 4


def make_views(pulses):
    """Two views whose row i is all i (and all 10 + i), so rows can be traced after a split."""
    rows = np.arange(len(pulses), dtype=float)[:, None] * np.ones((1, N_TIME))
    return {"A_z": rows, "B_z": rows + 10}


def test_categories_are_the_sorted_observed_values_including_zero():
    assert pulse_categories(np.array([4, 0, 1, 4, 0])) == [0, 1, 4]


def test_categories_reject_negative_or_fractional_counts():
    with pytest.raises(ValueError, match="non-negative whole"):
        pulse_categories(np.array([1, -1]))
    with pytest.raises(ValueError, match="non-negative whole"):
        pulse_categories(np.array([1.5, 2.0]))


def test_split_keeps_each_trial_with_its_own_count_in_every_view():
    pulses = np.array([4, 0, 1, 4])
    split = split_by_pulses(make_views(pulses), pulses, [0, 1, 4])
    np.testing.assert_array_equal(split[4]["A_z"][:, 0], [0.0, 3.0])
    np.testing.assert_array_equal(split[4]["B_z"][:, 0], [10.0, 13.0])
    np.testing.assert_array_equal(split[0]["A_z"][:, 0], [1.0])
    assert split[1]["A_z"].shape == (1, N_TIME)


def test_split_returns_an_empty_array_for_a_category_without_trials():
    pulses = np.array([0, 1])
    split = split_by_pulses(make_views(pulses), pulses, [0, 1, 4])
    assert split[4]["A_z"].shape == (0, N_TIME)


def test_split_refuses_trials_whose_count_is_not_a_listed_category():
    pulses = np.array([0, 3])
    with pytest.raises(ValueError, match="not in the categories"):
        split_by_pulses(make_views(pulses), pulses, [0, 1])


def test_split_refuses_a_count_per_trial_mismatch():
    with pytest.raises(ValueError, match="one pulse count per trial"):
        split_by_pulses(make_views(np.arange(3)), np.array([0, 1]), [0, 1])


def test_min_trials_keeps_categories_at_the_threshold_and_drops_below():
    counts = {0: 5, 1: 4, 4: 120}
    assert categories_with_min_trials(counts, 5) == [0, 4]


def test_min_trials_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        categories_with_min_trials({0: 3}, 0)


def test_correlation_is_one_for_a_response_rising_with_pulses_and_ignores_zero_pulse_trials():
    pulses = np.array([0, 0, 1, 2, 3, 4])
    responses = np.array([9.0, -9.0, 1.0, 2.0, 3.0, 4.0])  # the unrewarded trials are noise
    out = pulse_response_correlation(responses, pulses)
    assert out["rho"] == pytest.approx(1.0)
    assert out["n_trials"] == 4
    assert out["n_categories"] == 4


def test_correlation_is_negative_for_a_response_falling_with_pulses():
    out = pulse_response_correlation(np.array([3.0, 2.0, 1.0]), np.array([1, 2, 4]))
    assert out["rho"] == pytest.approx(-1.0)


def test_correlation_with_tied_pulse_counts_uses_average_ranks():
    # x ranks 1.5, 1.5, 3.5, 3.5 and y ranks 3, 4, 1, 2 give rho = -2 / sqrt(5)
    out = pulse_response_correlation(np.array([3.0, 3.1, 1.0, 1.1]), np.array([1, 1, 4, 4]))
    assert out["rho"] == pytest.approx(-2 / np.sqrt(5))


def test_correlation_is_nan_with_fewer_than_two_rewarded_categories():
    out = pulse_response_correlation(np.array([1.0, 2.0, 3.0]), np.array([0, 4, 4]))
    assert np.isnan(out["rho"])
    assert out["n_trials"] == 2 and out["n_categories"] == 1


def test_correlation_refuses_mismatched_lengths():
    with pytest.raises(ValueError, match="one pulse count per trial"):
        pulse_response_correlation(np.array([1.0, 2.0]), np.array([1]))
