"""Direct tests of the shared reversal-alignment helpers."""

import numpy as np
import pytest

from src.behavior_analysis.reversal_alignment import (
    align_window_values,
    bin_offsets,
    keep_zero_then_skip,
    nan_moving_average,
    smooth_pre_post,
    true_trial_offsets,
    window_indices,
)

nan = np.nan


# ---------------------------------------------------------------- alignment


def test_align_right_aligns_pre_and_left_aligns_post():
    row = align_window_values([1, 2, 3], [4, 5], pre=5, post=4)
    np.testing.assert_array_equal(row, [nan, nan, 1, 2, 3, 4, 5, nan, nan])


def test_align_with_skip_keeps_the_reversal_trial_and_drops_the_next_ones():
    row = align_window_values([1], [10, 11, 12, 13], pre=1, post=4, skip=2)
    np.testing.assert_array_equal(row, [1, 10, 13])


def test_align_empty_windows_give_all_nan():
    assert np.isnan(align_window_values([], [], pre=2, post=2)).all()


def test_align_raises_on_windows_longer_than_the_axis_unless_truncating():
    with pytest.raises(ValueError, match="pre window has 3"):
        align_window_values([1, 2, 3], [4], pre=2, post=2)
    with pytest.raises(ValueError, match="post window has 3"):
        align_window_values([1], [4, 5, 6], pre=2, post=2)
    row = align_window_values([1, 2, 3], [4, 5, 6], pre=2, post=2, truncate=True)
    np.testing.assert_array_equal(row, [2, 3, 4, 5])


@pytest.mark.parametrize("skip", [-1, 3, 4])
def test_align_rejects_bad_skip(skip):
    with pytest.raises(ValueError, match="skip_n_trials_after_reversal"):
        align_window_values([1], [2, 3, 4], pre=1, post=3, skip=skip)


def test_keep_zero_then_skip():
    assert keep_zero_then_skip([0, 1, 2, 3, 4], 2) == [0, 3, 4]
    assert keep_zero_then_skip([0, 1, 2], 0) == [0, 1, 2]
    assert keep_zero_then_skip([0, 1], 5) == [0]
    assert keep_zero_then_skip([], 2) == []


# ---------------------------------------------------------------- offsets


def test_bin_offsets_do_not_carry_the_skip_shift():
    np.testing.assert_array_equal(bin_offsets(pre=2, post=4, skip=2), [-2, -1, 0, 1])


def test_true_trial_offsets_shift_the_post_bins_after_the_first_by_skip():
    np.testing.assert_array_equal(true_trial_offsets(pre=2, post=4, skip=2), [-2, -1, 0, 3])
    np.testing.assert_array_equal(true_trial_offsets(pre=2, post=3), bin_offsets(pre=2, post=3))


def test_offsets_reject_a_skip_that_empties_the_post_window():
    with pytest.raises(ValueError, match="skip_n_trials_after_reversal"):
        bin_offsets(pre=1, post=2, skip=2)
    with pytest.raises(ValueError, match="skip_n_trials_after_reversal"):
        true_trial_offsets(pre=1, post=2, skip=-1)


# ---------------------------------------------------------------- windows


def test_window_indices_stop_at_the_previous_boundary():
    # Boundaries 3 and 6: the pre window of the second cannot reach back past trial 3.
    assert window_indices([3, 6], 1, 10, pre=5, post=None) == ([3, 4, 5], [6, 7, 8, 9])


def test_window_indices_stop_at_extra_block_changes():
    pre_idx, _ = window_indices([3, 12], 1, 20, pre=8, post=2, block_changes=[9])
    assert pre_idx == [9, 10, 11]


def test_window_indices_pre_is_clipped_at_the_data_start():
    assert window_indices([2, 9], 0, 12, pre=5, post=3) == ([0, 1], [2, 3, 4])


def test_window_indices_clip_post_to_the_trial_count():
    assert window_indices([3, 6], 1, 10, pre=5, post=None)[1] == [6, 7, 8, 9]
    assert window_indices([3, 6], 0, 10, pre=0, post=2) == ([], [3, 4])
    assert window_indices([3, 6], 1, 8, pre=1, post=100) == ([5], [6, 7])


def test_window_indices_post_stops_at_the_next_boundary():
    assert window_indices([3, 6], 0, 10, pre=0, post=100) == ([], [3, 4, 5])


# ---------------------------------------------------------------- smoothing


def test_nan_moving_average_centered_and_trailing():
    y = [1.0, 2.0, 3.0, 4.0]
    np.testing.assert_allclose(nan_moving_average(y, 3), [1.5, 2, 3, 3.5])
    np.testing.assert_allclose(nan_moving_average(y, 2, mode="trailing"), [1, 1.5, 2.5, 3.5])


def test_nan_moving_average_ignores_nan_and_leaves_all_nan_windows():
    out = nan_moving_average([1.0, nan, 3.0, nan, nan, nan], 3)
    np.testing.assert_allclose(out, [1, 2, 3, 3, nan, nan])


@pytest.mark.parametrize("window", [None, 0, 1])
def test_nan_moving_average_small_window_is_the_identity(window):
    np.testing.assert_array_equal(nan_moving_average([1, nan, 3], window), [1, nan, 3])


def test_nan_moving_average_window_longer_than_input_keeps_the_length():
    assert len(nan_moving_average([1.0, 2.0], 9)) == 2


def test_smooth_pre_post_does_not_mix_the_two_segments():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    out = smooth_pre_post(y, np.array([True, True, False, False]), 4)
    np.testing.assert_allclose(out, [0, 0, 1, 1])
