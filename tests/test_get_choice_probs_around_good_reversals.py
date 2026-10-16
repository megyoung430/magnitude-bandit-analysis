"""Tests for the three public functions of get_choice_probs_around_good_reversals.

Everything goes through the public API; the small helpers behind it are
exercised by the inputs below (truncated/NaN windows, skips, unsorted indices...).
"""

import numpy as np
import pytest

from src.behavior_analysis.get_choice_probs_around_good_reversals import (
    NoReversalsError,
    apply_moving_average_to_choice_probs,
    get_choice_probs_around_good_reversals,
    remove_trials_after_bad_rev,
)
from src.behavior_analysis.get_good_reversal_info import ReversalWarning

ROLES = ("prev_best", "next_best", "third")


def make_reversal(idx, pre, post, prev_best="A", next_best="B"):
    """Reversal whose prev-best tower is ``prev_best`` and next-best is ``next_best``.

    ``pre``/``post`` are the chosen tower per trial; choices are one-hot.
    """
    towers = ["A", "B", "C"]
    before = {t: 1.0 for t in towers}
    after = {t: 1.0 for t in towers}
    before[prev_best] = 3.0
    after[next_best] = 3.0
    return {
        "reversal_idx": idx,
        "reward_magnitudes_by_tower_before": before,
        "reward_magnitudes_by_tower_after": after,
        "trial_window_idx": {
            "pre": list(range(idx - len(pre), idx)),
            "post": list(range(idx, idx + len(post))),
        },
        "choices_by_tower": {
            t: {"pre": [float(p == t) for p in pre], "post": [float(p == t) for p in post]}
            for t in towers
        },
        "choices_by_rank": {"best": {"pre": [1.0] * len(pre), "post": [1.0] * len(post)}},
    }


def role_sum(mean_by_role):
    return sum(mean_by_role[role] for role in ROLES)


# ---------------------------------------------------------------- choice probabilities


def test_reversals_with_unchanged_best_tower_are_excluded_and_counted():
    kept = make_reversal(10, ["A", "A"], ["B", "B", "C"])
    unchanged = make_reversal(40, ["A", "A"], ["A", "A", "A"], prev_best="A", next_best="A")
    windows = {"m1": [kept, unchanged], "m2": [unchanged]}
    with pytest.warns(
        ReversalWarning, match=r"excluded 1 reversal.*m1.*best tower.*unchanged"
    ) as w:
        _, per_subject, across = get_choice_probs_around_good_reversals(windows, pre=2, post=3)
    assert any("excluded 1 reversal" in str(x.message) and "m2" in str(x.message) for x in w)
    assert set(per_subject) == {"m1"}
    assert across["num_reversals"] == 1
    np.testing.assert_allclose(role_sum(across["mean"]), 1.0)


def test_first_block_records_are_dropped_silently():
    first_block = make_reversal(0, [], ["A", "A"], prev_best="A", next_best="A")
    first_block["first_block"] = True
    kept = make_reversal(10, ["A", "A"], ["B", "B", "C"])
    _, per_subject, _ = get_choice_probs_around_good_reversals(
        {"m": [first_block, kept]}, pre=2, post=3
    )
    assert per_subject["m"]["num_reversals"] == 1


def test_get_choice_probs_end_to_end():
    windows = {
        "m1": [
            make_reversal(10, ["A", "A"], ["B", "B", "C"]),
            make_reversal(40, ["A", "A"], ["B", "C", "C"]),
        ],
        "m2": [make_reversal(15, ["A", "A"], ["B", "B", "B"])],
        "none": [],
    }
    x, per_subject, across = get_choice_probs_around_good_reversals(windows, pre=2, post=3)
    np.testing.assert_array_equal(x, [-2, -1, 0, 1, 2])
    assert set(per_subject) == {"m1", "m2"}
    assert across["num_subjects"] == 2
    assert across["num_reversals"] == 3
    # m1 next-best post = [1, 0.5, 0], m2 = [1, 1, 1]
    np.testing.assert_allclose(across["mean"]["next_best"], [0, 0, 1, 0.75, 0.5])
    np.testing.assert_allclose(across["mean"]["prev_best"], [1, 1, 0, 0, 0])
    np.testing.assert_allclose(role_sum(across["mean"]), 1.0)


def test_get_choice_probs_assigns_tower_roles_and_summarises_per_subject():
    rev = make_reversal(10, pre=["A", "A"], post=["B", "C", "B"], prev_best="A", next_best="B")
    _, per_subject, _ = get_choice_probs_around_good_reversals({"m1": [rev]}, pre=2, post=3)
    subject = per_subject["m1"]
    np.testing.assert_array_equal(subject["prev_best"], [[1, 1, 0, 0, 0]])
    np.testing.assert_array_equal(subject["next_best"], [[0, 0, 1, 0, 1]])
    np.testing.assert_array_equal(subject["third"], [[0, 0, 0, 1, 0]])
    np.testing.assert_array_equal(subject["third_mean"], [0, 0, 0, 1, 0])
    assert subject["num_reversals"] == 1


def test_get_choice_probs_follows_tower_roles_when_best_towers_change():
    rev = make_reversal(10, ["C", "C"], ["A", "A"], prev_best="C", next_best="A")
    _, per_subject, _ = get_choice_probs_around_good_reversals({"m1": [rev]}, pre=2, post=2)
    np.testing.assert_array_equal(per_subject["m1"]["prev_best"], [[1, 1, 0, 0]])
    np.testing.assert_array_equal(per_subject["m1"]["next_best"], [[0, 0, 1, 1]])


def test_get_choice_probs_stacks_one_row_per_reversal():
    windows = {"m1": [make_reversal(10, ["A"], ["B", "B"]), make_reversal(30, ["A"], ["C", "B"])]}
    _, per_subject, _ = get_choice_probs_around_good_reversals(windows, pre=1, post=2)
    assert per_subject["m1"]["next_best"].shape == (2, 3)
    np.testing.assert_array_equal(per_subject["m1"]["third"], [[0, 0, 0], [0, 1, 0]])


def test_get_choice_probs_nan_pads_truncated_windows_and_averages_the_finite_rows():
    windows = {
        "m1": [
            make_reversal(10, ["A"], ["B", "B", "B"]),
            make_reversal(30, ["A"], ["B"]),  # post window cut short: 2 trailing bins are NaN
        ],
        "m2": [make_reversal(50, ["A"], ["B", "B", "B"])],
    }
    _, per_subject, across = get_choice_probs_around_good_reversals(windows, pre=1, post=3)
    assert np.isnan(per_subject["m1"]["next_best"][1, 2:]).all()
    # nanmean over rows, so a NaN in one row does not hide the other row's value
    np.testing.assert_allclose(per_subject["m1"]["next_best_mean"], [0, 1, 1, 1])
    np.testing.assert_allclose(across["mean"]["next_best"], [0, 1, 1, 1])
    np.testing.assert_allclose(role_sum(across["mean"]), 1.0)


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_get_choice_probs_right_aligns_a_short_pre_window():
    # Reversal 3 trials into the data: pre holds 3 trials but the axis has 10 pre bins.
    rev = make_reversal(3, pre=["A", "A", "C"], post=["B", "B"], prev_best="A", next_best="B")
    x, per_subject, _ = get_choice_probs_around_good_reversals({"m": [rev]}, pre=10, post=2)
    nan7 = [np.nan] * 7
    assert list(x[:10]) == list(range(-10, 0))
    np.testing.assert_array_equal(per_subject["m"]["prev_best"], [[*nan7, 1, 1, 0, 0, 0]])
    np.testing.assert_array_equal(per_subject["m"]["third"], [[*nan7, 0, 0, 1, 0, 0]])
    np.testing.assert_array_equal(per_subject["m"]["next_best"], [[*nan7, 0, 0, 0, 1, 1]])


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_get_choice_probs_short_pre_does_not_move_the_post_values():
    short = make_reversal(2, pre=["A", "A"], post=["B", "C", "B"])
    full = make_reversal(20, pre=["A"] * 4, post=["B", "C", "B"])
    _, per_subject, _ = get_choice_probs_around_good_reversals(
        {"s": [short], "f": [full]}, pre=4, post=3
    )
    for role in ROLES:
        np.testing.assert_array_equal(per_subject["s"][role][0, 4:], per_subject["f"][role][0, 4:])


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_get_choice_probs_reversal_at_data_start_has_empty_pre_bins():
    rev = make_reversal(0, pre=[], post=["B", "B"])
    _, per_subject, _ = get_choice_probs_around_good_reversals({"m": [rev]}, pre=3, post=2)
    assert np.isnan(per_subject["m"]["next_best"][0, :3]).all()
    np.testing.assert_array_equal(per_subject["m"]["next_best"][0, 3:], [1, 1])


def test_get_choice_probs_keeps_the_last_trials_of_a_pre_window_longer_than_the_axis():
    rev = make_reversal(10, pre=["C", "C", "A", "A"], post=["B"])
    _, per_subject, _ = get_choice_probs_around_good_reversals({"m": [rev]}, pre=2, post=1)
    np.testing.assert_array_equal(per_subject["m"]["prev_best"], [[1, 1, 0]])
    np.testing.assert_array_equal(per_subject["m"]["third"], [[0, 0, 0]])


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_get_choice_probs_short_pre_and_short_post_with_skip():
    rev = make_reversal(2, pre=["A", "C"], post=["B", "A", "A", "C"])
    _, per_subject, _ = get_choice_probs_around_good_reversals(
        {"m": [rev]}, pre=4, post=6, skip_n_trials_after_reversal=2
    )
    # Post trials 1 and 2 are dropped; trials 0 and 3 land at offsets 0 and 1.
    nan2 = [np.nan] * 2
    np.testing.assert_array_equal(
        per_subject["m"]["next_best"], [[*nan2, 0, 0, 1, 0, np.nan, np.nan]]
    )
    np.testing.assert_array_equal(per_subject["m"]["third"], [[*nan2, 0, 1, 0, 1, np.nan, np.nan]])


def test_get_choice_probs_truncates_long_post_windows_to_the_axis():
    windows = {"m1": [make_reversal(10, ["A"], ["B"] * 6)]}
    x, per_subject, _ = get_choice_probs_around_good_reversals(windows, pre=1, post=2)
    assert len(x) == 3
    np.testing.assert_array_equal(per_subject["m1"]["prev_best"], [[1, 0, 0]])
    np.testing.assert_array_equal(per_subject["m1"]["next_best"], [[0, 1, 1]])


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_get_choice_probs_se_is_nan_for_one_subject_and_sample_sd_over_rows_otherwise():
    one = {"m1": [make_reversal(10, ["A"], ["B", "B"])]}
    _, _, across = get_choice_probs_around_good_reversals(one, pre=1, post=2)
    assert np.isnan(across["se"]["next_best"]).all()
    np.testing.assert_allclose(across["mean"]["next_best"], [0, 1, 1])

    # next-best last bin across subjects: 1, 0 and a truncated (NaN) third subject
    three = {
        "m1": [make_reversal(10, ["A"], ["B", "B"])],
        "m2": [make_reversal(10, ["A"], ["B", "C"])],
        "m3": [make_reversal(10, ["A"], ["B"])],
    }
    _, _, across = get_choice_probs_around_good_reversals(three, pre=1, post=2)
    np.testing.assert_allclose(across["mean"]["next_best"][-1], 0.5)
    # Two subjects have data in this bin, so the SE divides by sqrt(2).
    np.testing.assert_allclose(across["se"]["next_best"][-1], np.sqrt(0.5) / np.sqrt(2))


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_get_choice_probs_se_is_nan_where_fewer_than_two_subjects_have_data():
    windows = {
        "m1": [make_reversal(10, ["A"], ["B", "B"])],
        "m2": [make_reversal(10, ["A"], ["B"])],
    }
    _, _, across = get_choice_probs_around_good_reversals(windows, pre=1, post=2)
    assert across["mean"]["next_best"][-1] == 1.0  # a single value: mean defined, SE not
    assert np.isnan(across["se"]["next_best"][-1])
    assert np.isfinite(across["se"]["next_best"][:2]).all()


def test_get_choice_probs_skip_shortens_the_axis():
    windows = {"m1": [make_reversal(10, ["A"] * 3, ["B"] * 6)]}
    x, per_subject, _ = get_choice_probs_around_good_reversals(
        windows, pre=3, post=6, skip_n_trials_after_reversal=4
    )
    np.testing.assert_array_equal(x, [-3, -2, -1, 0, 1])
    assert per_subject["m1"]["prev_best"].shape == (1, 5)


@pytest.mark.parametrize("skip", [0, 1, 15])
def test_get_choice_probs_skip_axis_length(skip):
    windows = {"m1": [make_reversal(10, ["A"] * 2, ["B"] * 20)]}
    x, per_subject, _ = get_choice_probs_around_good_reversals(
        windows, pre=2, post=20, skip_n_trials_after_reversal=skip
    )
    assert len(x) == 2 + 20 - skip
    assert per_subject["m1"]["next_best"].shape == (1, len(x))


def test_get_choice_probs_skip_keeps_the_reversal_trial():
    rev = make_reversal(10, pre=["A"], post=["B", "C", "B", "A"], prev_best="A", next_best="B")
    _, per_subject, _ = get_choice_probs_around_good_reversals(
        {"m1": [rev]}, pre=1, post=4, skip_n_trials_after_reversal=2
    )
    # skip=2 drops post trials 1 and 2; the row keeps trials [0, 3] (its last trial too).
    np.testing.assert_array_equal(per_subject["m1"]["next_best"], [[0, 1, 0]])
    np.testing.assert_array_equal(per_subject["m1"]["prev_best"], [[1, 0, 1]])


@pytest.mark.parametrize("skip", [1, 2, 3, 4])
def test_get_choice_probs_skip_removes_exactly_skip_trials_and_keeps_the_last(skip):
    post = ["B", "A", "A", "A", "A", "C"]  # trial 0 is B, the last trial is C
    rev = make_reversal(10, pre=["A"], post=post, prev_best="A", next_best="B")
    _, per_subject, _ = get_choice_probs_around_good_reversals(
        {"m1": [rev]}, pre=1, post=6, skip_n_trials_after_reversal=skip
    )
    third = per_subject["m1"]["third"][0]
    assert len(third) == 1 + 6 - skip  # no row is cut or padded
    assert third[-1] == 1.0  # the last post trial (tower C) is kept
    assert not np.isnan(per_subject["m1"]["third"]).any()
    assert per_subject["m1"]["next_best"][0, 1] == 1.0  # the reversal trial is kept


@pytest.mark.parametrize("skip", [-1, 3, 4])
def test_get_choice_probs_rejects_bad_skip(skip):
    windows = {"m1": [make_reversal(10, ["A"], ["B", "B", "B"])]}
    with pytest.raises(ValueError, match="skip_n_trials_after_reversal"):
        get_choice_probs_around_good_reversals(
            windows, pre=1, post=3, skip_n_trials_after_reversal=skip
        )


def test_get_choice_probs_asserts_probabilities_sum_to_one():
    rev = make_reversal(10, ["A"], ["B", "B"])
    rev["choices_by_tower"]["A"]["post"] = [1.0, 1.0]  # tower A also chosen: sums to 2
    with pytest.raises(AssertionError, match="do not sum to 1"):
        get_choice_probs_around_good_reversals({"m1": [rev]}, pre=1, post=2)


def test_get_choice_probs_rejects_windows_with_no_finite_bins():
    windows = {"m1": [make_reversal(10, [], [])]}  # every bin is NaN padding
    with pytest.warns(RuntimeWarning), pytest.raises(AssertionError, match="No finite bins"):
        get_choice_probs_around_good_reversals(windows, pre=1, post=2)


def test_get_choice_probs_rejects_no_usable_subjects():
    with pytest.raises(ValueError):
        get_choice_probs_around_good_reversals({"none": []}, pre=1, post=2)


def test_get_choice_probs_without_usable_subjects_raises_the_specific_error():
    with pytest.raises(NoReversalsError):
        get_choice_probs_around_good_reversals({"none": []}, pre=1, post=2)


# ---------------------------------------------------------------- moving average


def curves_from_prev(prev):
    """Per-subject curves for which the three roles sum to 1 in every bin."""
    prev = np.array(prev, dtype=float)
    return {
        "prev_best_mean": prev,
        "next_best_mean": 1.0 - prev,
        "third_mean": np.zeros_like(prev),
        "num_reversals": 2,
    }


def per_subject_curves(prev, nxt, third, n=2):
    return {
        "prev_best_mean": np.array(prev, dtype=float),
        "next_best_mean": np.array(nxt, dtype=float),
        "third_mean": np.array(third, dtype=float),
        "num_reversals": n,
    }


def smoothed_prev(x, prev, **kwargs):
    """The smoothed prev-best curve of a single subject, via the public function."""
    _, smoothed, _ = apply_moving_average_to_choice_probs(
        x, {"m1": curves_from_prev(prev)}, **kwargs
    )
    return smoothed["m1"]["prev_best_mean_sm"]


def test_apply_moving_average_smooths_and_summarises():
    x = np.array([-2, -1, 0, 1])
    per_subject = {
        "m1": per_subject_curves([1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0], n=2),
        "m2": per_subject_curves([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], n=3),
    }
    x_out, smoothed, across = apply_moving_average_to_choice_probs(
        x, per_subject, moving_avg_window=2, mode="trailing"
    )
    np.testing.assert_array_equal(x_out, x)
    np.testing.assert_allclose(smoothed["m2"]["next_best_mean_sm"], [0, 0, 0, 0.5])
    assert "next_best_mean_sm" not in per_subject["m1"]  # input untouched
    assert across["num_subjects"] == 2
    assert across["num_reversals"] == 5
    assert (across["moving_avg_window"], across["mode"], across["split_at"]) == (2, "trailing", 0)
    np.testing.assert_allclose(role_sum(across["mean"]), 1.0)


def test_apply_moving_average_default_mode_is_centered_with_window_five():
    x = np.arange(-6, 6)  # 6 pre bins and 6 post bins
    prev = np.zeros(12)
    prev[8] = 5.0  # a spike at the third post bin
    out = smoothed_prev(x, prev)
    # window 5 centred, smoothed within the post segment only (edges average fewer bins)
    np.testing.assert_allclose(out[6:], [5 / 3, 5 / 4, 1, 1, 5 / 4, 0])
    np.testing.assert_allclose(out[:6], 0.0)


def test_apply_moving_average_centered_odd_window():
    x = np.arange(0, 6)
    out = smoothed_prev(x, [0, 0, 3, 0, 0, 0], moving_avg_window=3, mode="centered", split_at=-1)
    np.testing.assert_allclose(out, [0, 1, 1, 1, 0, 0])


def test_apply_moving_average_centered_even_window_covers_two_before_and_one_after():
    # Pinned alignment: with an even window the centred kernel spans bins [i-2, i+1].
    x = np.arange(0, 6)
    out = smoothed_prev(x, [0, 0, 4, 0, 0, 0], moving_avg_window=4, mode="centered", split_at=-1)
    np.testing.assert_allclose(out, [0, 4 / 3, 1, 1, 1, 0])


def test_apply_moving_average_split_at_moves_the_segment_boundary():
    x = np.arange(-4, 4)
    prev = [0, 0, 0, 0, 3, 3, 3, 3]
    # split_at=0: the step sits on the boundary, so nothing bleeds across it
    np.testing.assert_allclose(smoothed_prev(x, prev, moving_avg_window=3, split_at=0), prev)
    # split_at=1: bin x=0 (value 3) now ends the pre segment and is averaged with its neighbours
    np.testing.assert_allclose(
        smoothed_prev(x, prev, moving_avg_window=3, split_at=1), [0, 0, 0, 1, 1.5, 3, 3, 3]
    )


def test_apply_moving_average_split_at_pre_and_post_are_smoothed_independently():
    x = np.array([-2, -1, 0, 1])
    out = smoothed_prev(x, [1, 0, 1, 0], moving_avg_window=2, mode="trailing", split_at=0)
    np.testing.assert_allclose(out, [1, 0.5, 1, 0.5])
    out = smoothed_prev(x, [1, 0, 1, 0], moving_avg_window=2, mode="trailing", split_at=-1)
    np.testing.assert_allclose(out, [1, 0, 0.5, 0.5])


def test_apply_moving_average_skips_nan_bins():
    x = np.arange(0, 4)
    per_subject = {
        "m1": per_subject_curves([1, np.nan, 1, 0], [0, np.nan, 0, 1], [0, np.nan, 0, 0])
    }
    _, smoothed, across = apply_moving_average_to_choice_probs(
        x, per_subject, moving_avg_window=3, mode="centered"
    )
    # NaN bins are left out of the average, so the middle bins still get a value
    np.testing.assert_allclose(smoothed["m1"]["prev_best_mean_sm"], [1, 1, 0.5, 0.5])
    np.testing.assert_allclose(role_sum(across["mean"]), 1.0)


@pytest.mark.parametrize("window", [None, 0, 1])
def test_apply_moving_average_small_window_leaves_the_curve_unchanged(window):
    x = np.array([-1, 0, 1])
    out = smoothed_prev(x, [1, 0.25, 0.5], moving_avg_window=window)
    np.testing.assert_array_equal(out, [1, 0.25, 0.5])


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_apply_moving_average_all_nan_window_stays_nan():
    x = np.array([-1, 0, 1])
    nan = [np.nan] * 3
    per_subject = {"m1": per_subject_curves(nan, nan, nan)}
    _, smoothed, across = apply_moving_average_to_choice_probs(
        x, per_subject, moving_avg_window=2, mode="trailing"
    )
    assert np.isnan(smoothed["m1"]["prev_best_mean_sm"]).all()
    assert np.isnan(across["mean"]["prev_best"]).all()  # no finite bins is allowed here


def test_apply_moving_average_centered_window_longer_than_segment_keeps_the_length():
    # pre segment = [1], post segment = [0, 1]; each is averaged over its own finite bins
    x = np.array([-1, 0, 1])
    out = smoothed_prev(x, [1, 0, 1], moving_avg_window=5, mode="centered")
    np.testing.assert_allclose(out, [1, 0.5, 0.5])
    # a trailing window of the same width gives the same-length output too
    out = smoothed_prev(x, [1, 0, 1], moving_avg_window=5, mode="trailing")
    np.testing.assert_allclose(out, [1, 0, 0.5])


def test_apply_moving_average_custom_keys_and_missing_keys():
    x = np.array([-1, 0])
    per_subject = {
        "m1": {"a": np.array([0.5, 0.5]), "b": np.array([0.5, 0.5]), "c": np.array([0.0, 0.0])},
        "no_keys": {"num_reversals": 1},  # lacks the requested keys: dropped
    }
    _, smoothed, across = apply_moving_average_to_choice_probs(
        x, per_subject, moving_avg_window=1, use_keys=("a", "b", "c"), out_keys=("p", "q", "r")
    )
    assert set(smoothed) == {"m1"}
    assert set(smoothed["m1"]) >= {"a", "p", "q", "r"}
    assert across["num_subjects"] == 1


def test_apply_moving_average_drops_subjects_carrying_only_some_of_the_keys():
    # "Usable" means every requested curve is present: a subject with just a, b would
    # otherwise fail on the missing c instead of being dropped.
    x = np.array([-1, 0])
    full = {"a": np.array([0.5, 0.5]), "b": np.array([0.5, 0.5]), "c": np.array([0.0, 0.0])}
    per_subject = {"full": full, "partial": {"a": full["a"], "b": full["b"]}}
    _, smoothed, across = apply_moving_average_to_choice_probs(
        x, per_subject, moving_avg_window=1, use_keys=("a", "b", "c"), out_keys=("p", "q", "r")
    )
    assert set(smoothed) == {"full"}
    assert across["num_subjects"] == 1


def test_apply_moving_average_chained_keys_read_the_already_smoothed_value():
    # out_keys[1] ("c") is also use_keys[2], so the third curve is smoothed from the
    # value just written to "c": q = smooth(smooth(b)), not smooth(b).
    x = np.array([-2, -1, 0, 1])
    curves = {"a": [1, 0.4, 0.8, 0.4], "b": [0, 0.4, 0, 0.4], "c": [0, 0, 0, 0]}
    per_subject = {"m1": {k: np.array(v, dtype=float) for k, v in curves.items()}}
    _, smoothed, across = apply_moving_average_to_choice_probs(
        x,
        per_subject,
        moving_avg_window=2,
        mode="trailing",
        use_keys=("a", "b", "c"),
        out_keys=("p", "c", "q"),
        split_at=-10,  # every bin is post-reversal: one segment
    )
    np.testing.assert_allclose(smoothed["m1"]["c"], [0, 0.2, 0.2, 0.2])
    np.testing.assert_allclose(smoothed["m1"]["q"], [0, 0.1, 0.2, 0.2])
    np.testing.assert_array_equal(per_subject["m1"]["c"], [0, 0, 0, 0])  # input untouched
    np.testing.assert_allclose(role_sum(across["mean"]), 1.0)


def test_apply_moving_average_without_usable_subjects():
    x = np.array([-1, 0])
    _, smoothed, across = apply_moving_average_to_choice_probs(x, {"m1": {"num_reversals": 1}})
    assert smoothed == {}
    assert across["num_subjects"] == 0
    assert across["mean"] == dict.fromkeys(ROLES)
    assert across["se"] == dict.fromkeys(ROLES)
    assert (across["moving_avg_window"], across["mode"], across["split_at"]) == (5, "centered", 0)


def test_apply_moving_average_rejects_probabilities_off_one():
    x = np.array([-1, 0])
    per_subject = {"m1": per_subject_curves([0.9, 0.9], [0, 0], [0, 0])}
    with pytest.raises(AssertionError, match="Smoothed probabilities"):
        apply_moving_average_to_choice_probs(x, per_subject, moving_avg_window=1)


def test_apply_moving_average_counts_subjects_without_num_reversals_as_zero():
    x = np.array([-1, 0])
    curves = curves_from_prev([1, 0])
    del curves["num_reversals"]
    _, _, across = apply_moving_average_to_choice_probs(
        x, {"m1": curves, "m2": curves_from_prev([1, 0])}, moving_avg_window=1
    )
    assert (across["num_subjects"], across["num_reversals"]) == (2, 2)


# ------------------------------- removing trials after bad reversals


def post_lengths(reversal):
    return (
        len(reversal["choices_by_tower"]["A"]["post"]),
        len(reversal["choices_by_rank"]["best"]["post"]),
        len(reversal["trial_window_idx"]["post"]),
    )


def test_remove_trials_after_bad_rev_cuts_all_post_lists_and_reports(capsys):
    windows = {
        "m1": [make_reversal(30, ["A"], ["B"] * 6), make_reversal(10, ["A"], ["B"] * 6)],
        "empty": [],
    }
    out = remove_trials_after_bad_rev(
        windows, {"m1": [10, 30], "empty": []}, {"m1": [13]}, include_bad_trial=True
    )
    assert [r["reversal_idx"] for r in out["m1"]] == [10, 30]  # sorted by reversal_idx
    cut, untouched = out["m1"]
    assert (cut["removed_bad_idx"], cut["post_len_after_removal"]) == (13, 4)
    assert post_lengths(cut) == (4, 4, 4)
    assert cut["trial_window_idx"]["post"] == [10, 11, 12, 13]
    assert len(cut["choices_by_tower"]["A"]["pre"]) == 1  # pre window is never cut
    assert untouched["post_len_after_removal"] is None and untouched["removed_bad_idx"] is None
    assert post_lengths(untouched) == (6, 6, 6)
    assert out["empty"] == []
    assert "[REMOVE] m1 good@10: cut post at bad@13 (kept 4 post trials)" in capsys.readouterr().out
    assert post_lengths(windows["m1"][1]) == (6, 6, 6)  # input untouched


def test_remove_trials_after_bad_rev_can_exclude_the_bad_trial(capsys):
    windows = {"m1": [make_reversal(10, ["A"], ["B"] * 8)]}
    out = remove_trials_after_bad_rev(
        windows, {"m1": [10, 30]}, {"m1": [14]}, include_bad_trial=False
    )
    (rev,) = out["m1"]
    assert (rev["removed_bad_idx"], rev["post_len_after_removal"]) == (14, 4)
    assert post_lengths(rev) == (4, 4, 4)
    assert rev["trial_window_idx"]["post"] == [10, 11, 12, 13]
    assert "(kept 4 post trials)" in capsys.readouterr().out


def test_remove_trials_after_bad_rev_includes_the_bad_trial_by_default():
    windows = {"m1": [make_reversal(10, ["A"], ["B"] * 8)]}
    (rev,) = remove_trials_after_bad_rev(windows, {"m1": [10]}, {"m1": [14]})["m1"]
    assert rev["post_len_after_removal"] == 5


def test_remove_trials_after_bad_rev_bad_right_after_the_reversal():
    windows = {"m1": [make_reversal(10, ["A"], ["B"] * 8)]}
    kept = remove_trials_after_bad_rev(windows, {"m1": [10]}, {"m1": [11]}, include_bad_trial=True)
    cut = remove_trials_after_bad_rev(windows, {"m1": [10]}, {"m1": [11]}, include_bad_trial=False)
    assert kept["m1"][0]["post_len_after_removal"] == 2
    assert cut["m1"][0]["post_len_after_removal"] == 1
    assert post_lengths(cut["m1"][0]) == (1, 1, 1)


@pytest.mark.parametrize("bad_idx", [10.5, 10.2])
def test_remove_trials_after_bad_rev_fractional_bad_index_leaves_the_window_whole(bad_idx, capsys):
    # A bad index less than one trial after the reversal gives a cut length below 1 when
    # the bad trial is excluded: the cut is recorded but the window is not sliced or reported.
    windows = {"m1": [make_reversal(10, ["A"], ["B"] * 8)]}
    (rev,) = remove_trials_after_bad_rev(
        windows, {"m1": [10]}, {"m1": [bad_idx]}, include_bad_trial=False
    )["m1"]
    assert rev["removed_bad_idx"] == bad_idx
    assert rev["post_len_after_removal"] == pytest.approx(bad_idx - 10)
    assert post_lengths(rev) == (8, 8, 8)
    assert capsys.readouterr().out == ""


def test_remove_trials_after_bad_rev_uses_the_first_bad_of_unsorted_indices():
    windows = {"m1": [make_reversal(10, ["A"], ["B"] * 8)]}
    (rev,) = remove_trials_after_bad_rev(windows, {"m1": [10]}, {"m1": [16, 12, 14]})["m1"]
    assert rev["removed_bad_idx"] == 12


def test_remove_trials_after_bad_rev_sorts_good_indices_to_find_the_next_good():
    windows = {"m1": [make_reversal(10, ["A"], ["B"] * 8)]}
    # the next good reversal (12) comes before the bad one (13), so nothing is cut
    (rev,) = remove_trials_after_bad_rev(windows, {"m1": [40, 12, 10]}, {"m1": [13]})["m1"]
    assert rev["removed_bad_idx"] is None
    assert post_lengths(rev) == (8, 8, 8)
    # with the next good reversal far away the same bad reversal does cut
    (rev,) = remove_trials_after_bad_rev(windows, {"m1": [40, 10]}, {"m1": [13]})["m1"]
    assert rev["removed_bad_idx"] == 13


def test_remove_trials_after_bad_rev_ignores_bads_at_or_outside_the_good_interval():
    windows = {"m1": [make_reversal(10, ["A"], ["B"] * 8)]}
    for bads in ([10], [3], [20], []):  # on the reversal, before it, after the next good, none
        (rev,) = remove_trials_after_bad_rev(windows, {"m1": [10, 20]}, {"m1": bads})["m1"]
        assert rev["removed_bad_idx"] is None and post_lengths(rev) == (8, 8, 8)


def test_remove_trials_after_bad_rev_subject_missing_from_index_dicts_is_left_whole():
    windows = {"m1": [make_reversal(10, ["A"], ["B"] * 8)]}
    (rev,) = remove_trials_after_bad_rev(windows, {}, {})["m1"]
    assert rev["post_len_after_removal"] is None and post_lengths(rev) == (8, 8, 8)


def test_remove_trials_after_bad_rev_reversal_without_idx_passes_through():
    rev = make_reversal(10, ["A"], ["B"] * 3)
    del rev["reversal_idx"]
    (out,) = remove_trials_after_bad_rev({"m1": [rev]}, {"m1": [10]}, {"m1": [11]})["m1"]
    assert out["removed_bad_idx"] is None and post_lengths(out) == (3, 3, 3)


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_remove_trials_after_bad_rev_output_feeds_get_choice_probs():
    windows = {"m1": [make_reversal(10, ["A"], ["B", "B", "B", "C", "C"])]}
    cut = remove_trials_after_bad_rev(windows, {"m1": [10]}, {"m1": [12]}, include_bad_trial=False)
    _, per_subject, across = get_choice_probs_around_good_reversals(cut, pre=1, post=5)
    # post window cut to 2 trials; the remaining bins are NaN, so only 2 post bins are finite
    np.testing.assert_array_equal(per_subject["m1"]["next_best"][0, 1:3], [1, 1])
    assert np.isnan(per_subject["m1"]["next_best"][0, 3:]).all()
    np.testing.assert_allclose(role_sum(across["mean"])[:3], 1.0)
