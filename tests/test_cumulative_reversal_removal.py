import re

import numpy as np
import pytest

from src.behavior_analysis.cumulative_reversal_removal import (
    cumulative_counts,
    cumulative_removal_series,
    good_bad_removal_curves,
    next_reversal_delays,
    post_window_steps,
    total_removal_curves,
)

GOODBAD_MESSAGE = "For mode='goodbad', provide all_good_idx and all_bad_idx."


def windows(**anchors):
    return {subject: [{"reversal_idx": i} for i in idxs] for subject, idxs in anchors.items()}


X = np.arange(-3, 20)  # post offsets 0..19


def test_post_window_steps_maps_each_post_offset_to_its_step():
    x = np.array([-2.0, np.nan, 0.0, 1.4, 1.6, 5.0, 5.0])
    post_mask, steps, step_idx = post_window_steps(x)
    assert post_mask.tolist() == [False, False, True, True, True, True, True]
    assert steps.tolist() == [0, 1, 2, 5]  # rounded, deduplicated, sorted
    assert step_idx.tolist() == [0, 1, 2, 3, 3]


def test_post_window_steps_without_post_offsets():
    post_mask, steps, _ = post_window_steps(np.array([-3.0, -1.0]))
    assert not post_mask.any()
    assert steps.size == 0


@pytest.mark.parametrize(
    ("exclude_anchor", "delays", "next_revs"),
    [(True, [3, 15], [15, 30]), (False, [3, 0], [15, 15])],
)
def test_next_reversal_delays_anchor_eligibility(exclude_anchor, delays, next_revs):
    # with exclude_anchor=False an anchor that is itself a reversal is its own next
    got_delays, got_next = next_reversal_delays(
        np.array([12, 15]), np.array([30, 10, 15]), exclude_anchor
    )
    assert got_delays.tolist() == delays
    assert got_next.tolist() == next_revs


def test_next_reversal_delays_ignores_duplicates_and_empty_reversals():
    delays, _ = next_reversal_delays(np.array([1]), np.array([4, 4, 4]), True)
    assert delays.tolist() == [3]
    delays, nxt = next_reversal_delays(np.array([1, 2]), np.array([]), True)
    assert delays.size == 0 and nxt.size == 0


def test_cumulative_counts_steps_up_at_each_delay_and_drops_delays_beyond_last_step():
    steps = np.array([0, 1, 2, 5])
    counts = cumulative_counts(np.array([1, 1, 3, 9]), steps)
    # delay 3 lands on the step at 5; delay 9 is past the last step and is dropped
    assert counts.tolist() == [0.0, 2.0, 2.0, 3.0]


def test_good_bad_removal_curves_hand_computed():
    wins = windows(A=[10, 20, 35])
    cum_good, cum_bad, frac_good, frac_bad = good_bad_removal_curves(
        wins, {"A": [10, 35]}, {"A": [20]}, X
    )
    post = X >= 0
    assert np.isnan(cum_good[~post]).all() and np.isnan(frac_bad[~post]).all()
    # anchor 10 is cut by the bad reversal at +10; anchor 20 by the good one at +15
    assert cum_bad[post].tolist() == [0.0] * 10 + [1.0] * 10
    assert cum_good[post].tolist() == [0.0] * 15 + [1.0] * 5
    np.testing.assert_allclose(frac_bad[post], cum_bad[post] / 3)
    np.testing.assert_allclose(frac_good[post], cum_good[post] / 3)


def test_good_bad_removal_curves_include_anchor_when_not_excluded():
    wins = windows(A=[10])
    cum_good, cum_bad, _, _ = good_bad_removal_curves(
        wins, {"A": [10]}, {"A": []}, X, exclude_anchor=False
    )
    assert cum_good[X >= 0].tolist() == [1.0] * 20  # the anchor itself, delay 0
    assert cum_bad[X >= 0].sum() == 0


@pytest.mark.parametrize(
    ("good", "bad"),
    [
        ({"A": [12]}, {"A": [], "B": []}),  # B missing from good_idx
        ({"A": [12], "B": []}, {"A": []}),  # B missing from bad_idx
    ],
)
def test_removal_curves_name_a_subject_missing_from_the_index_dicts(good, bad):
    wins = windows(A=[10], B=[5])
    with pytest.raises(KeyError, match="'B'"):
        good_bad_removal_curves(wins, good, bad, X)
    with pytest.raises(KeyError, match="'B'"):
        total_removal_curves(wins, X, good_idx=good, bad_idx=bad)


def test_removal_curves_skip_anchors_without_reversal_idx():
    wins = {"A": [{"reversal_idx": 4}, {"reversal_idx": None}, {}]}
    _, frac = total_removal_curves(wins, X, good_idx={"A": [4]}, bad_idx={"A": []})
    assert np.nanmax(frac) == 0.0  # one anchor and no later reversal


def test_removal_curves_without_post_window_are_all_nan():
    x = np.arange(-5, 0)
    for curve in good_bad_removal_curves(windows(A=[1]), {"A": [1]}, {"A": []}, x):
        assert np.isnan(curve).all() and curve.shape == x.shape
    for curve in total_removal_curves(windows(A=[1]), x, good_idx={"A": [1]}, bad_idx={"A": []}):
        assert np.isnan(curve).all()


def test_total_removal_curves_hand_computed_counts_good_and_bad_reversals():
    wins = windows(A=[10, 20, 35], B=[3, 4])
    good = {"A": [10, 20, 35], "B": [3, 4]}
    bad = {"A": [15, 50], "B": [9]}
    cum, frac = total_removal_curves(wins, X, good_idx=good, bad_idx=bad)
    # A: 10->15 bad (+5), 20->35 (+15), 35->50 (+15); B: 3->4 (+1), 4->9 bad (+5)
    expected = np.array([0] + [1] * 4 + [3] * 10 + [5] * 5, dtype=float)
    assert cum[X >= 0].tolist() == expected.tolist()
    np.testing.assert_allclose(frac[X >= 0], expected / 5)


def test_total_equals_good_plus_bad_at_every_offset_on_a_synthetic_mix():
    rng = np.random.default_rng(11)
    for _ in range(25):
        every = {s: rng.choice(300, 12, replace=False) for s in "AB"}
        is_good = {s: rng.random(12) < 0.6 for s in "AB"}
        good = {s: sorted(every[s][is_good[s]].tolist()) for s in "AB"}
        bad = {s: sorted(every[s][~is_good[s]].tolist()) for s in "AB"}
        wins = windows(**good)  # windows are built from good reversals only
        exclude = bool(rng.integers(2))
        cum_good, cum_bad, frac_good, frac_bad = good_bad_removal_curves(
            wins, good, bad, X, exclude
        )
        cum_total, frac_total = total_removal_curves(wins, X, exclude, good_idx=good, bad_idx=bad)
        np.testing.assert_array_equal(cum_total, cum_good + cum_bad)  # nan pattern included
        np.testing.assert_allclose(frac_total, frac_good + frac_bad, equal_nan=True)


def test_total_removal_curves_require_both_index_dicts():
    wins = windows(A=[10])
    for kwargs in ({}, {"good_idx": {"A": [10]}}, {"bad_idx": {"A": []}}):
        with pytest.raises(ValueError, match="good_idx and bad_idx"):
            total_removal_curves(wins, X, **kwargs)


def test_removal_curves_match_brute_force_on_random_data():
    rng = np.random.default_rng(3)
    for _ in range(25):
        good = {s: sorted(rng.choice(200, 6, replace=False).tolist()) for s in "AB"}
        bad = {s: sorted(rng.choice(200, 4, replace=False).tolist()) for s in "AB"}
        wins = windows(**{s: g for s, g in good.items()})
        exclude = bool(rng.integers(2))
        cum_good, cum_bad, _, _ = good_bad_removal_curves(wins, good, bad, X, exclude)
        for t_i, t in enumerate(X[X >= 0]):
            want_good = want_bad = 0
            for s in "AB":
                every = sorted(set(good[s]) | set(bad[s]))
                for g in good[s]:
                    later = [r for r in every if (r > g if exclude else r >= g)]
                    if later and later[0] - g <= t:
                        want_good += later[0] in good[s]
                        want_bad += later[0] not in good[s]
            assert cum_good[X >= 0][t_i] == want_good
            assert cum_bad[X >= 0][t_i] == want_bad


def test_cumulative_removal_series_total_and_good_bad_modes():
    wins = windows(A=[10, 20])
    idx = ({"A": [10]}, {"A": [20]})
    total = cumulative_removal_series(wins, *idx, X, use_total=True, fraction=True)
    assert list(total) == ["total"]
    good_bad = cumulative_removal_series(wins, *idx, X, use_total=False, fraction=False)
    assert list(good_bad) == ["good", "bad"]
    assert np.nanmax(good_bad["good"]) == 0  # the bad reversal at 20 follows the anchor at 10
    assert np.nanmax(good_bad["bad"]) == 1
    assert np.nanmax(total["total"]) == 0.5  # total counts that bad reversal too


def test_cumulative_removal_series_fraction_or_counts():
    wins = windows(A=[10, 20])
    idx = ({"A": [10]}, {"A": [20]})
    counts = cumulative_removal_series(wins, *idx, X, use_total=True, fraction=False)
    fractions = cumulative_removal_series(wins, *idx, X, use_total=True, fraction=True)
    assert np.nanmax(counts["total"]) == 1 and np.nanmax(fractions["total"]) == 0.5


def test_cumulative_removal_series_total_mode_requires_both_index_dicts():
    wins = windows(A=[10])
    with pytest.raises(ValueError, match="good_idx and bad_idx"):
        cumulative_removal_series(wins, {"A": [10]}, None, X, use_total=True, fraction=True)
    with pytest.raises(ValueError, match="good_idx and bad_idx"):
        cumulative_removal_series(wins, None, None, X, use_total=True, fraction=True)


def test_cumulative_removal_series_good_bad_mode_requires_both_index_dicts():
    wins = windows(A=[10])
    with pytest.raises(ValueError, match=re.escape(GOODBAD_MESSAGE)):
        cumulative_removal_series(wins, {"A": [10]}, None, X, use_total=False, fraction=True)
    with pytest.raises(ValueError, match=re.escape(GOODBAD_MESSAGE)):
        cumulative_removal_series(wins, None, {"A": [10]}, X, use_total=None, fraction=True)


def test_no_anchors_reports_no_data_rather_than_zero_removed():
    empty = {"A": []}
    *_, frac_good, frac_bad = good_bad_removal_curves(empty, {"A": [5]}, {"A": []}, X)
    assert np.isnan(frac_good).all() and np.isnan(frac_bad).all()
    _, frac_total = total_removal_curves(empty, X, good_idx={"A": [5]}, bad_idx={"A": []})
    assert np.isnan(frac_total).all()


def test_total_removal_counts_bad_reversals_that_are_not_anchors():
    wins = windows(A=[10])
    kwargs = {"good_idx": {"A": [10]}, "bad_idx": {"A": [15]}}
    cum_total, _ = total_removal_curves(wins, X, **kwargs)
    assert cum_total[X >= 0].tolist() == [0.0] * 5 + [1.0] * 15
