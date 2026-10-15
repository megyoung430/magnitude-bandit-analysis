"""Psychometric curves and heatmap tables: weightings, SE rule, empty cells, ground truth."""

import warnings

import numpy as np
import pytest

from src.behavior_analysis.get_choice_probs_by_value import (
    BEST_VALS,
    SECOND_VALS,
    MouseCounts,
    count_choice_probs,
)
from src.behavior_analysis.stats import nan_mean
from src.behavior_analysis.value_psychometrics import heatmap_tables, psychometric_curves


def counts_for(cells):
    """MouseCounts from ``{(best, second): (n_trials, n_best, n_second)}``."""
    counts = MouseCounts.empty()
    for (b, s), (n, n_best, n_second) in cells.items():
        counts.denom.loc[b, s] = n
        counts.num_best.loc[b, s] = n_best
        counts.num_second.loc[b, s] = n_second
    return counts


# --- stats.nan_mean ------------------------------------------------------------------------


def test_nan_mean_is_nan_for_all_nan_columns_without_warning():
    stack = np.array([[1.0, np.nan], [3.0, np.nan]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = nan_mean(stack, axis=0)
    assert result[0] == 2.0 and np.isnan(result[1])


# --- weightings ----------------------------------------------------------------------------

# x = 4 pools a 10-trial cell (always best) with a 90-trial cell (never best)
UNEVEN = {(4, 1): (10, 10, 0), (4, 2): (90, 0, 90)}


def best_val_point(curves, choice, x):
    curve = curves[choice]["best_val"]
    return curve.mean[list(curve.x).index(x)]


def test_pooled_weighting_pools_trials_across_cells_sharing_an_x_value():
    curves = psychometric_curves({"m1": counts_for(UNEVEN)}, pool_trials=True)
    assert best_val_point(curves, "best", 4) == pytest.approx(0.1)  # 10 / 100
    assert best_val_point(curves, "second", 4) == pytest.approx(0.9)


def test_unpooled_weighting_averages_cell_probabilities_equally():
    curves = psychometric_curves({"m1": counts_for(UNEVEN)}, pool_trials=False)
    assert best_val_point(curves, "best", 4) == pytest.approx(0.5)  # mean of 1.0 and 0.0
    assert best_val_point(curves, "second", 4) == pytest.approx(0.5)


def test_x_values_without_trials_are_nan_not_zero():
    for pool in (True, False):
        curves = psychometric_curves({"m1": counts_for(UNEVEN)}, pool_trials=pool)
        assert np.isnan(best_val_point(curves, "best", 2))


def test_axes_are_best_value_second_value_and_their_difference():
    curves = psychometric_curves({"m1": counts_for(UNEVEN)}, pool_trials=True)
    assert list(curves["best"]["best_val"].x) == BEST_VALS
    assert list(curves["best"]["second_val"].x) == SECOND_VALS
    assert list(curves["best"]["diff"].x) == [0, 1, 2, 3]


def test_difference_curve_pools_the_cells_with_that_difference():
    cells = {(4, 3): (10, 10, 0), (3, 2): (10, 0, 10), (2, 1): (20, 10, 10), (3, 3): (4, 2, 2)}
    curves = psychometric_curves({"m1": counts_for(cells)}, pool_trials=True)
    diff = curves["best"]["diff"]
    by_diff = dict(zip(diff.x, diff.mean, strict=True))
    assert by_diff[1] == pytest.approx(20 / 40)
    assert by_diff[0] == pytest.approx(0.5)  # ties are kept, as on the value axes
    assert np.isnan(by_diff[3])
    unpooled = psychometric_curves({"m1": counts_for(cells)}, pool_trials=False)
    by_diff = dict(zip(diff.x, unpooled["best"]["diff"].mean, strict=True))
    assert by_diff[1] == pytest.approx((1.0 + 0.0 + 0.5) / 3)


def test_cell_with_best_below_second_and_trials_raises():
    counts = counts_for({(2, 3): (5, 1, 1)})
    with pytest.raises(ValueError, match="best.*second"):
        psychometric_curves({"m1": counts}, pool_trials=True)


# --- mice are the replicates ---------------------------------------------------------------


def test_mean_and_se_are_taken_across_mice_whatever_the_weighting():
    mice = {
        "m1": counts_for({(4, 1): (10, 8, 0)}),
        "m2": counts_for({(4, 1): (1000, 400, 0)}),  # many trials, still one replicate
        "m3": counts_for({(4, 1): (5, 1, 0)}),
    }
    expected = np.array([0.8, 0.4, 0.2])
    for pool in (True, False):
        curve = psychometric_curves(mice, pool_trials=pool)["best"]["best_val"]
        i = list(curve.x).index(4)
        assert curve.mean[i] == pytest.approx(expected.mean())
        assert curve.se[i] == pytest.approx(expected.std(ddof=1) / np.sqrt(3))
        assert curve.n_mice[i] == 3
        assert curve.n_trials[i] == 1015
        assert curve.mice == ["m1", "m2", "m3"]


def test_se_is_nan_with_a_single_mouse_and_mean_is_still_drawn():
    curve = psychometric_curves({"m1": counts_for(UNEVEN)}, pool_trials=True)["best"]["best_val"]
    i = list(curve.x).index(4)
    assert np.isnan(curve.se[i]) and not np.isnan(curve.mean[i])


def test_mice_missing_an_x_value_are_left_out_of_that_mean_and_n():
    mice = {"m1": counts_for({(4, 1): (10, 5, 0)}), "m2": counts_for({(3, 1): (10, 5, 0)})}
    curve = psychometric_curves(mice, pool_trials=True)["best"]["best_val"]
    assert list(curve.n_mice) == [0, 1, 1]
    assert np.isnan(curve.se).all()


def test_no_warnings_when_every_mouse_is_empty_at_some_x():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for pool in (True, False):
            psychometric_curves({"m1": counts_for(UNEVEN)}, pool_trials=pool)
            psychometric_curves({}, pool_trials=pool)


# --- heatmap tables ------------------------------------------------------------------------


def test_heatmap_tables_average_mice_equally_and_sum_trials():
    mice = {
        "m1": counts_for({(4, 3): (10, 10, 0)}),
        "m2": counts_for({(4, 3): (30, 0, 15), (3, 1): (4, 1, 1)}),
    }
    tables = heatmap_tables(mice)
    assert tables.p_best.loc[4, 3] == pytest.approx(0.5)  # mean(1.0, 0.0), not 10/40
    assert tables.p_second.loc[4, 3] == pytest.approx(0.25)
    assert tables.n_trials.loc[4, 3] == 40 and tables.n_trials.loc[3, 1] == 4
    assert tables.p_best.loc[3, 1] == pytest.approx(0.25)  # only m2 has trials here
    assert tables.per_mouse_n["m2"].loc[3, 1] == 4
    assert np.isnan(tables.per_mouse_p_best["m1"].loc[3, 1])


def test_heatmap_tables_do_not_warn_for_cells_empty_for_every_mouse():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tables = heatmap_tables({"m1": counts_for({(4, 3): (10, 10, 0)})})
    assert np.isnan(tables.p_best.loc[2, 1]) and tables.n_trials.loc[2, 1] == 0


def test_heatmap_tables_with_no_mice_are_empty_grids():
    tables = heatmap_tables({})
    assert tables.p_best.isna().all().all() and (tables.n_trials == 0).all().all()
    assert tables.per_mouse_p_best == {}


# --- ground truth recovery -----------------------------------------------------------------

TRUE_P_BEST = {(4, 1): 0.9, (4, 2): 0.75, (4, 3): 0.6, (3, 1): 0.8, (3, 2): 0.65, (2, 1): 0.7}


def synthetic_sessions(rng, n_per_cell=2000):
    """One mouse choosing best with a known probability per cell, else second."""
    mags, ranks, choices = [], [], []
    for (b, s), p in TRUE_P_BEST.items():
        for _ in range(n_per_cell):
            mags.append({"A": b, "B": s, "C": 0})
            pick_best = rng.random() < p
            ranks.append("best" if pick_best else "second")
            choices.append("A" if pick_best else "B")
    order = rng.permutation(len(mags))
    tv = {
        "reward_magnitudes": [mags[i] for i in order],
        "chosen_rank": [ranks[i] for i in order],
        "choice": [choices[i] for i in order],
    }
    return {"ses-1": {"trial_variables": [tv]}}


def test_pipeline_recovers_known_choice_probabilities_in_both_weightings():
    rng = np.random.default_rng(0)
    data = {1: {f"m{k}": synthetic_sessions(rng) for k in range(3)}}
    counts = count_choice_probs(data)
    tables = heatmap_tables(counts)
    for (b, s), p in TRUE_P_BEST.items():
        assert tables.p_best.loc[b, s] == pytest.approx(p, abs=0.03)
        assert tables.p_second.loc[b, s] == pytest.approx(1 - p, abs=0.03)
    # every cell has equal trial counts, so the two weightings agree with the cell average
    for pool in (True, False):
        curve = psychometric_curves(counts, pool_trials=pool)["best"]["diff"]
        by_diff = dict(zip(curve.x, curve.mean, strict=True))
        assert by_diff[1] == pytest.approx(np.mean([0.6, 0.65, 0.7]), abs=0.03)
        assert by_diff[3] == pytest.approx(0.9, abs=0.03)
