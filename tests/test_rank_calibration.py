"""Calibration of a mouse's readouts against replays, on numbers worked out by hand."""

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis import rank_calibration as rc


def null(mouse, values):
    return pd.DataFrame(dict(mouse=mouse, replicate=range(len(values)), a=values, b=values[::-1]))


OBSERVED = pd.DataFrame(dict(mouse=["m1", "m2"], a=[5.0, 1.0], b=[2.0, np.nan]))
NULL = pd.concat([null("m1", [0.0, 1.0, 2.0, 3.0, 4.0]), null("m2", [0.0, 1.0, 2.0, 3.0, 4.0])])


@pytest.fixture(scope="module")
def table():
    return rc.calibrate(OBSERVED, NULL).set_index(["mouse", "readout"])


def test_z_is_the_distance_from_the_null_mean_in_null_sds(table):
    row = table.loc[("m1", "a")]
    assert row.null_mean == 2 and row.null_sd == pytest.approx(np.sqrt(2.5))
    assert row.z == pytest.approx(3 / np.sqrt(2.5)) and row["diff"] == 3


def test_the_empirical_p_counts_draws_as_far_from_the_null_mean_as_the_observation(table):
    # observed 5: no draw is 3 or more from the mean 2, so only the observation counts: 1/6
    assert table.loc[("m1", "a"), "p_empirical"] == pytest.approx(1 / 6)
    # observed 1: draws 0, 1, 2, 3, 4 are all at least 1 from... |d - 2| >= 1: four of five
    assert table.loc[("m2", "a"), "p_empirical"] == pytest.approx(5 / 6)


def test_the_percentile_counts_ties_half(table):
    assert table.loc[("m2", "a"), "percentile"] == pytest.approx(0.3)  # one below, one tied


def test_a_nan_observation_has_nan_statistics_but_a_row(table):
    row = table.loc[("m2", "b")]
    assert np.isnan(row.z) and np.isnan(row.p_empirical)


def test_a_null_without_spread_gives_no_z_but_a_p():
    flat = pd.DataFrame(dict(mouse="m1", replicate=range(4), a=[1.0] * 4))
    row = rc.calibrate(pd.DataFrame(dict(mouse=["m1"], a=[3.0])), flat).iloc[0]
    assert np.isnan(row.z) and row.p_empirical == pytest.approx(1 / 5)


def test_a_mouse_without_replicates_raises():
    with pytest.raises(ValueError, match="mouse m3"):
        rc.calibrate(pd.DataFrame(dict(mouse=["m3"], a=[1.0])), NULL)


def test_a_mouse_passes_when_it_is_beyond_its_null_in_the_predicted_direction_only():
    calibration = rc.calibrate(OBSERVED, NULL)
    up = rc.passes(calibration, {"a": +1, "b": +1}).set_index(["mouse", "readout"])["passes"]
    down = rc.passes(calibration, {"a": -1, "b": -1}).set_index(["mouse", "readout"])["passes"]
    big_null = pd.concat([null("m1", list(np.linspace(0, 4, 60))), null("m2", [0.0, 1, 2, 3, 4])])
    strong = rc.passes(rc.calibrate(OBSERVED, big_null), {"a": +1}).set_index(["mouse", "readout"])
    assert not up[("m1", "a")]  # p = 1/6 is above alpha with five draws
    assert strong.loc[("m1", "a"), "passes"]  # sixty draws: p = 1/61
    assert not rc.passes(rc.calibrate(OBSERVED, big_null), {"a": -1}).loc[0, "passes"]
    assert not down[("m1", "a")]


def test_a_readout_without_a_prediction_never_passes():
    table = rc.passes(rc.calibrate(OBSERVED, NULL), {"a": +1})
    assert not table.loc[table["readout"] == "b", "passes"].any()


def test_the_summary_gives_mean_z_wilcoxon_and_who_passes():
    mice = [f"m{i}" for i in range(8)]
    observed = pd.DataFrame(dict(mouse=mice, a=[10.0 + i for i in range(8)]))
    draws = pd.concat([null(m, list(np.linspace(0, 4, 80))) for m in mice])
    summary = rc.summarize(rc.calibrate(observed, draws), {"a": +1}).iloc[0]
    assert summary.n == 8 and summary["mean"] > 5 and summary.p_wilcoxon < 0.05
    assert summary.n_pass == 8 and summary.passing.split(", ") == mice and summary.predicted == 1


def test_the_summary_of_few_mice_has_no_mean_or_test():
    summary = rc.summarize(rc.calibrate(OBSERVED, NULL), {"a": +1}).iloc[0]
    assert summary.n == 2 and np.isnan(summary["mean"]) and np.isnan(summary.p_wilcoxon)
