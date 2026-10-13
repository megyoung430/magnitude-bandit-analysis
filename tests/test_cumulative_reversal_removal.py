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


def test_good_bad_removal_curves_subject_missing_from_index_dicts_has_no_removal():
    wins = windows(A=[10], B=[5])
    _, _, frac_good, frac_bad = good_bad_removal_curves(wins, {"A": [12]}, {}, X)
    post = X >= 0
    assert frac_good[post][-1] == pytest.approx(0.5)  # only A's anchor (of 2) is cut
    assert frac_bad[post].sum() == 0


def test_removal_curves_skip_anchors_without_reversal_idx():
    wins = {"A": [{"reversal_idx": 4}, {"reversal_idx": None}, {}]}
    _, frac = total_removal_curves(wins, X)
    assert np.nanmax(frac) == 0.0  # one anchor (denominator 1) and no later reversal


def test_removal_curves_without_post_window_are_all_nan():
    x = np.arange(-5, 0)
    for curve in good_bad_removal_curves(windows(A=[1]), {"A": [1]}, {"A": []}, x):
        assert np.isnan(curve).all() and curve.shape == x.shape
    for curve in total_removal_curves(windows(A=[1]), x):
        assert np.isnan(curve).all()


def test_total_removal_curves_hand_computed():
    wins = windows(A=[10, 20, 35], B=[3, 4])
    cum, frac = total_removal_curves(wins, X)
    # A: 10->20 (+10), 20->35 (+15), 35 none; B: 3->4 (+1), 4 none
    expected = np.array([0] + [1] * 9 + [2] * 5 + [3] * 5, dtype=float)
    assert cum[X >= 0].tolist() == expected.tolist()
    np.testing.assert_allclose(frac[X >= 0], expected / 5)


def test_total_equals_good_plus_bad_when_windows_list_every_reversal():
    wins = windows(A=[3, 9, 14, 30], B=[2, 6])
    good = {"A": [3, 14], "B": [2]}
    bad = {"A": [9, 30], "B": [6]}
    cum_good, cum_bad, _, _ = good_bad_removal_curves(wins, good, bad, X)
    cum_total, _ = total_removal_curves(wins, X)
    np.testing.assert_array_equal(np.nan_to_num(cum_good + cum_bad), np.nan_to_num(cum_total))


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
    total = cumulative_removal_series(X, wins, None, None, use_total=True, fraction=True)
    assert list(total) == ["Total Rev"]
    good_bad = cumulative_removal_series(
        X, wins, {"A": [10]}, {"A": [20]}, use_total=False, fraction=False
    )
    assert list(good_bad) == ["Good Rev", "Bad Rev"]
    assert np.nanmax(good_bad["Good Rev"]) == 0  # the bad reversal at 20 follows the anchor at 10
    assert np.nanmax(good_bad["Bad Rev"]) == 1


def test_cumulative_removal_series_good_bad_mode_requires_both_index_dicts():
    wins = windows(A=[10])
    with pytest.raises(ValueError, match="all_good_idx and all_bad_idx"):
        cumulative_removal_series(X, wins, {"A": [10]}, None, use_total=False, fraction=True)
    with pytest.raises(ValueError, match="all_good_idx and all_bad_idx"):
        cumulative_removal_series(X, wins, None, {"A": [10]}, use_total=None, fraction=True)
