"""Tests for reversal-aligned choice and magnitude curves of the three arms.

Reversals are small hand-built dicts shaped like ``get_good_reversal_info`` output (plus the
``"magnitudes_by_tower"`` windows), so every expected value can be read off the fixture.
"""

import copy

import numpy as np
import pytest

from src.behavior_analysis.get_good_reversal_info import get_good_reversal_info
from src.behavior_analysis.reversal_aligned_arms import (
    EmptySelectionWarning,
    add_magnitude_windows,
    align_window_values,
    compute_reversal_aligned_arms,
)
from src.behavior_analysis.stats import nan_standard_error
from tests.plot_fixtures import make_mouse, make_session, session_key
from tests.reversal_arm_fixtures import make_reversal


def curves(windows, **kwargs):
    defaults = dict(scheme="prev", reversal_type="all", pre=4, post=3, moving_avg_window=1)
    return compute_reversal_aligned_arms(windows, **{**defaults, **kwargs})


# ------------------------------------------------------------------ alignment


def test_a_short_pre_window_is_right_aligned_to_the_reversal_trial():
    row = align_window_values([7, 8], [1, 2, 3], pre=4, post=3)
    np.testing.assert_array_equal(row, [np.nan, np.nan, 7, 8, 1, 2, 3])


def test_a_short_post_window_is_nan_padded_at_the_end():
    row = align_window_values([5, 6], [1], pre=2, post=3)
    np.testing.assert_array_equal(row, [5, 6, 1, np.nan, np.nan])


def test_skip_k_keeps_post_trial_zero_and_drops_exactly_trials_1_to_k():
    row = align_window_values([9], [1, 2, 3, 4, 5], pre=1, post=5, skip=2)
    np.testing.assert_array_equal(row, [9, 1, 4, 5])


def test_a_window_longer_than_pre_or_post_raises():
    with pytest.raises(ValueError, match="pre window"):
        align_window_values([1, 2, 3], [1], pre=2, post=2)
    with pytest.raises(ValueError, match="post window"):
        align_window_values([1], [1, 2, 3], pre=2, post=2)
    with pytest.raises(ValueError, match="post window"):
        align_window_values([1], [1, 2, 3, 4], pre=2, post=3, skip=1)


def test_skip_must_leave_a_post_window():
    with pytest.raises(ValueError, match="skip"):
        align_window_values([1], [1, 2], pre=1, post=2, skip=2)
    with pytest.raises(ValueError, match="skip"):
        align_window_values([1], [1, 2], pre=1, post=2, skip=-1)


def test_curves_of_a_truncated_pre_window_sit_at_the_right_offsets():
    short = make_reversal(["A", "B"], ["C", "C", "C"])  # only offsets -2, -1 exist
    full = make_reversal(["A", "A", "A", "A"], ["C", "C", "C"])
    result = curves({"m": [short, full]})
    np.testing.assert_array_equal(result.trial_offsets, [-4, -3, -2, -1, 0, 1, 2])
    # offsets -4, -3: only the full window has data; -2: both choose A; -1: A once, B once.
    np.testing.assert_allclose(result.choice_mean["prev_best"], [1, 1, 1, 0.5, 0, 0, 0])
    np.testing.assert_allclose(result.choice_mean["prev_second"], [0, 0, 0, 0.5, 0, 0, 0])
    np.testing.assert_allclose(result.choice_mean["prev_third"], [0, 0, 0, 0, 1, 1, 1])


def test_curves_with_a_skip_keep_the_reversal_trial_and_label_later_bins_by_true_offset():
    rev = make_reversal(["A"], ["A", "B", "C", "B", "C"])
    result = curves({"m": [rev]}, pre=1, post=5, skip_n_trials_after_reversal=2)
    np.testing.assert_array_equal(result.trial_offsets, [-1, 0, 3, 4])
    np.testing.assert_allclose(result.choice_mean["prev_best"], [1, 1, 0, 0])
    np.testing.assert_allclose(result.choice_mean["prev_second"], [0, 0, 1, 0])
    np.testing.assert_allclose(result.choice_mean["prev_third"], [0, 0, 0, 1])


# ------------------------------------------------------------ labelling schemes


def test_prev_and_new_schemes_label_the_same_choices_by_different_rankings():
    rev = make_reversal(["A", "A", "A", "A"], ["C", "C", "C"])
    prev = curves({"m": [rev]}, scheme="prev")
    new = curves({"m": [rev]}, scheme="new")
    assert tuple(prev.choice_mean) == ("prev_best", "prev_second", "prev_third")
    assert tuple(new.choice_mean) == ("new_best", "new_second", "new_third")
    np.testing.assert_allclose(prev.choice_mean["prev_best"], [1, 1, 1, 1, 0, 0, 0])
    np.testing.assert_allclose(new.choice_mean["new_third"], [1, 1, 1, 1, 0, 0, 0])
    np.testing.assert_allclose(new.choice_mean["new_best"], [0, 0, 0, 0, 1, 1, 1])


def test_magnitude_traces_follow_the_arms_of_the_scheme():
    rev = make_reversal(["A"] * 4, ["C"] * 3)
    prev = curves({"m": [rev]}, scheme="prev")
    new = curves({"m": [rev]}, scheme="new")
    # prev_best is arm A: 3 before the reversal, 1 after.
    np.testing.assert_allclose(prev.magnitude_mean["prev_best"], [3, 3, 3, 3, 1, 1, 1])
    # new_best is arm C: 1 before, 3 after. new_second is arm B throughout.
    np.testing.assert_allclose(new.magnitude_mean["new_best"], [1, 1, 1, 1, 3, 3, 3])
    np.testing.assert_allclose(new.magnitude_mean["new_second"], [2] * 7)


def test_magnitude_traces_of_a_short_pre_window_use_the_same_offsets_as_the_choices():
    short = make_reversal(["A", "B"], ["C"] * 3)
    full = make_reversal(["A"] * 4, ["C"] * 3, pre_scale=2.0)
    result = curves({"m": [short, full]})
    # prev_best magnitude: only the full window at -4, -3; both at -2, -1 (3 and 6).
    np.testing.assert_allclose(result.magnitude_mean["prev_best"], [6, 6, 4.5, 4.5, 1, 1, 1])


# ------------------------------------------------------------------- statistics


def test_se_is_over_the_subjects_with_data_in_the_bin():
    windows = {
        "m1": [make_reversal(["A"] * 4, ["C"] * 3)],
        "m2": [make_reversal(["B"] * 4, ["C"] * 3)],
        "m3": [make_reversal([], ["C"] * 3)],  # no pre data: NaN at offsets -4..-1
    }
    result = curves(windows)
    # prev_best at -4: m1 = 1, m2 = 0, m3 missing -> sd(1, 0) / sqrt(2) = 0.5
    expected = nan_standard_error(np.array([[1.0], [0.0], [np.nan]]))[0]
    assert result.choice_se["prev_best"][0] == pytest.approx(expected)
    assert result.choice_se["prev_best"][0] == pytest.approx(0.5)
    # post bins: three subjects who all chose C.
    assert result.choice_se["prev_third"][4] == pytest.approx(0.0)


def test_se_is_nan_not_zero_for_one_subject():
    result = curves({"m": [make_reversal(["A"] * 4, ["C"] * 3)]})
    for role in result.roles:
        assert np.isnan(result.choice_se[role]).all()
        assert np.isnan(result.magnitude_se[role]).all()


def test_magnitude_se_is_across_subjects_of_their_per_subject_means():
    windows = {
        "m1": [make_reversal(["A"] * 4, ["C"] * 3), make_reversal(["A"] * 4, ["C"] * 3)],
        "m2": [make_reversal(["A"] * 4, ["C"] * 3, pre_scale=2.0)],
    }
    result = curves(windows)
    # per-subject means of prev_best before the reversal: 3 and 6 -> mean 4.5, SE 1.5
    np.testing.assert_allclose(result.magnitude_mean["prev_best"][:4], 4.5)
    np.testing.assert_allclose(result.magnitude_se["prev_best"][:4], 1.5)


def test_counts_are_reported_with_the_result():
    windows = {
        "m1": [make_reversal(["A"] * 4, ["C"] * 3)],
        "m2": [make_reversal(["B"] * 4, ["C"] * 3)],
    }
    result = curves(windows)
    assert result.num_subjects == 2 and result.num_reversals == 2
    assert result.counts.selected == 2 and result.counts.tied_labels == 0


# ------------------------------------------------------------------------- ties


def test_tied_reversals_are_left_out_and_counted():
    tied_before = make_reversal(["A"] * 4, ["C"] * 3, before={"A": 3.0, "B": 2.0, "C": 2.0})
    clean = make_reversal(["A"] * 4, ["C"] * 3)
    result = curves({"m": [tied_before, clean]}, scheme="prev")
    assert result.num_reversals == 1
    assert result.counts.tied_labels == 1
    # the same reversal has a clean after-ranking, so it counts for the new labels
    assert curves({"m": [tied_before, clean]}, scheme="new").num_reversals == 2


def test_type_selection_drops_reversals_whose_type_depends_on_a_tie():
    # B and C tie for second before; C is third after, so by arm name B would be second ("other")
    # and with C second the type is second_to_third.
    ambiguous = make_reversal(
        ["A"] * 4,
        ["C"] * 3,
        before={"A": 3.0, "B": 2.0, "C": 2.0},
        after={"A": 2.0, "B": 3.0, "C": 1.0},
    )
    with pytest.warns(EmptySelectionWarning):
        result = curves({"m": [ambiguous]}, scheme="new", reversal_type="second_to_third")
    assert result.counts.tied_type == 1 and result.num_reversals == 0


def test_empty_selection_gives_nan_curves_and_a_warning():
    with pytest.warns(EmptySelectionWarning, match="second_to_third"):
        result = curves(
            {"m": [make_reversal(["A"] * 4, ["C"] * 3)]}, reversal_type="second_to_third"
        )
    assert result.num_reversals == 0 and result.num_subjects == 0
    assert np.isnan(result.choice_mean["prev_best"]).all()


# --------------------------------------------------------------- checks / errors


def test_choice_probabilities_that_do_not_sum_to_one_raise_value_error():
    bad = make_reversal(["A"] * 4, ["C"] * 3)
    bad["choices_by_tower"]["B"]["pre"] = [1.0] * 4  # two arms chosen on every pre trial
    with pytest.raises(ValueError, match="sum to 1"):
        curves({"m": [bad]})


def test_unknown_moving_average_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        curves({"m": [make_reversal(["A"] * 4, ["C"] * 3)]}, moving_avg_mode="causal")


def test_moving_average_mode_is_centered_by_default_and_selectable():
    rev = make_reversal(["A", "B", "A", "B"], ["C", "A", "C"])
    default = curves({"m": [rev]}, moving_avg_window=3)
    centered = curves({"m": [rev]}, moving_avg_window=3, moving_avg_mode="centered")
    trailing = curves({"m": [rev]}, moving_avg_window=3, moving_avg_mode="trailing")
    np.testing.assert_allclose(default.choice_mean["prev_best"], centered.choice_mean["prev_best"])
    assert not np.allclose(
        centered.choice_mean["prev_best"], trailing.choice_mean["prev_best"], equal_nan=True
    )
    # smoothing never mixes the pre and post segments: pre bins use only pre data
    np.testing.assert_allclose(centered.choice_mean["prev_best"][:4], [0.5, 2 / 3, 1 / 3, 0.5])
    assert centered.moving_avg_mode == "centered" and centered.moving_avg_window == 3


def test_smoothed_probabilities_still_sum_to_one():
    rev = make_reversal(["A", "B", "A", "B"], ["C", "A", "C"])
    result = curves({"m": [rev]}, moving_avg_window=3, scheme="new")
    total = sum(result.choice_mean[r] for r in result.roles)
    np.testing.assert_allclose(total, 1.0)


def test_windows_are_not_modified():
    windows = {"m": [make_reversal(["A"] * 4, ["C"] * 3)]}
    before = copy.deepcopy(windows)
    curves(windows, moving_avg_window=3)
    assert windows == before


def test_first_block_records_are_rejected():
    first = make_reversal(["A"] * 4, ["C"] * 3)
    first["first_block"] = True
    with pytest.raises(ValueError, match="first block"):
        curves({"m": [first]})


# ------------------------------------------------------- magnitude windows (merge)


def two_session_subject():
    """Two sessions whose towers' magnitudes differ, so the merge order shows in the values."""
    first = make_session()
    second = make_session()
    rewards = second["reward_magnitudes_by_tower"]
    rewards["A1"], rewards["B1"] = rewards["B1"], rewards["A1"]
    return {session_key(1, 1): first, session_key(2, 3): second}


def test_magnitude_windows_are_the_merged_series_over_the_window_indices():
    data = {"m": two_session_subject()}
    windows = get_good_reversal_info(data, pre=5, post=6)
    enriched = add_magnitude_windows(windows, data)
    for rev in enriched["m"]:
        pre_idx, post_idx = rev["trial_window_idx"]["pre"], rev["trial_window_idx"]["post"]
        merged = {
            t: data["m"][session_key(1, 1)]["reward_magnitudes_by_tower"][t]
            + data["m"][session_key(2, 3)]["reward_magnitudes_by_tower"][t]
            for t in ("A1", "B1", "C1")
        }
        for tower, window in rev["magnitudes_by_tower"].items():
            assert window["pre"] == [merged[tower][i] for i in pre_idx]
            assert window["post"] == [merged[tower][i] for i in post_idx]


def test_magnitude_windows_do_not_depend_on_the_dict_order_of_sessions():
    data = {"m": two_session_subject()}
    reversed_order = {"m": dict(reversed(list(data["m"].items())))}
    windows = get_good_reversal_info(data, pre=5, post=6)
    assert add_magnitude_windows(windows, reversed_order) == add_magnitude_windows(windows, data)


def test_magnitude_windows_do_not_modify_the_input_windows():
    data = {"m": two_session_subject()}
    windows = get_good_reversal_info(data, pre=5, post=6)
    add_magnitude_windows(windows, data)
    assert all("magnitudes_by_tower" not in rev for rev in windows["m"])


def test_magnitude_windows_from_a_different_merge_are_rejected():
    data = {"m": two_session_subject()}
    windows = get_good_reversal_info(data, pre=5, post=6)
    first, second = list(data["m"].values())
    swapped = {"m": {session_key(1, 1): second, session_key(2, 3): first}}
    with pytest.raises(ValueError, match="do not match"):
        add_magnitude_windows(windows, swapped)


def test_magnitude_windows_need_the_subject_s_sessions():
    data = {"m": two_session_subject()}
    windows = get_good_reversal_info(data, pre=5, post=6)
    with pytest.raises(ValueError, match="no sessions"):
        add_magnitude_windows(windows, {"other": data["m"]})


def test_end_to_end_curves_from_sessions():
    # The animal always picks the best tower; every block change moves each tower up one rank,
    # so the previous third tower becomes best and all reversals are second_to_third.
    data = {"m": make_mouse(1, pick=0)}
    windows = add_magnitude_windows(get_good_reversal_info(data, pre=10, post=6), data)
    prev = compute_reversal_aligned_arms(
        windows, scheme="prev", reversal_type="all", pre=10, post=6, moving_avg_window=1
    )
    assert prev.num_reversals == 2
    # The first reversal (trial 8) has only 8 pre trials, so offsets -10 and -9 come from the
    # second alone. There the animal still sat on the best arm of the block before last, which
    # is the second best arm of the block that ended at the reversal.
    np.testing.assert_allclose(prev.choice_mean["prev_best"][:2], 0.0)
    np.testing.assert_allclose(prev.choice_mean["prev_second"][:2], 1.0)
    np.testing.assert_allclose(prev.choice_mean["prev_best"][2:10], 1.0)
    np.testing.assert_allclose(prev.choice_mean["prev_best"][10:], 0.0)
    np.testing.assert_allclose(prev.choice_mean["prev_third"][10:], 1.0)
    new = compute_reversal_aligned_arms(
        windows, scheme="new", reversal_type="second_to_third", pre=10, post=6, moving_avg_window=1
    )
    np.testing.assert_allclose(new.choice_mean["new_best"][10:], 1.0)
    # before the change the animal sits on the arm that becomes second best
    np.testing.assert_allclose(new.choice_mean["new_second"][2:10], 1.0)
    # magnitudes: the best tower pays 4 before the change and 2 after (it moves to second)
    np.testing.assert_allclose(prev.magnitude_mean["prev_best"][2:10], 4.0)
    np.testing.assert_allclose(prev.magnitude_mean["prev_best"][10:], 2.0)
