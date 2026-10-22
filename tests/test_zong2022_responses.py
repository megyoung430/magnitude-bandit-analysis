"""Zong et al. 2022 reward analysis: aligned responses and the shuffle test for responsive cells."""

import numpy as np
import pandas as pd
import pytest

from src.funcimg_analysis.zong2022_responses import (
    aligned_windows,
    chance_summary,
    circular_shifts,
    distinction_scores,
    pooled_responses,
    response_frames,
    responsive_cells,
    reward_frame_mask,
    shifted_distinction_scores,
    valid_span,
)

PERIOD = 0.133
N = 400


def frame_times(n=N, start=20.0):
    return start + PERIOD * np.arange(n)


def outcome_table(frames, pulses, phase="task"):
    return pd.DataFrame(
        {
            "outcome_time": frame_times()[np.asarray(frames)] + 0.01,
            "num_rew_pulses": pulses,
            "phase": phase,
        }
    )


# --- valid span -----------------------------------------------------------------------------------


def test_valid_span_is_the_first_to_last_frame_with_a_time():
    time_s = frame_times()
    time_s[:10] = np.nan
    time_s[-5:] = np.nan
    assert valid_span(time_s) == slice(10, N - 5)


def test_valid_span_refuses_a_gap_inside_the_recording():
    time_s = frame_times()
    time_s[100] = np.nan
    with pytest.raises(ValueError, match="gap"):
        valid_span(time_s)


def test_valid_span_refuses_a_recording_without_times():
    with pytest.raises(ValueError, match="no frame"):
        valid_span(np.full(10, np.nan))


# --- aligned windows ------------------------------------------------------------------------------


def test_windows_hold_the_frames_around_each_outcome_by_pulse_count():
    traces = np.vstack([np.arange(N, dtype=float), -np.arange(N, dtype=float)])
    outcomes = outcome_table([100, 200, 300], [0, 2, 4])
    aligned = aligned_windows(traces, frame_times(), outcomes, (-0.4, 0.8), ("task",))
    pre, post = 3, 6  # round(0.4/0.133), round(0.8/0.133)
    assert aligned.offsets.tolist() == list(range(-pre, post + 1))
    assert aligned.pulses.tolist() == [0, 2, 4]
    assert aligned.windows.shape == (2, 3, pre + post + 1)
    np.testing.assert_array_equal(aligned.windows[0, 1], np.arange(200 - pre, 200 + post + 1))
    np.testing.assert_array_equal(aligned.windows[1, 2], -np.arange(300 - pre, 300 + post + 1))
    assert aligned.n_dropped == 0
    assert aligned.period_s == pytest.approx(PERIOD)


def test_outcomes_without_a_whole_window_or_in_other_phases_are_dropped_and_counted():
    traces = np.arange(N, dtype=float)[None]
    outcomes = pd.concat(
        [
            outcome_table([1, 200], [1, 1]),  # frame 1 has no 2 s of pre-outcome frames
            outcome_table([250], [3], phase="port_finding"),
        ],
        ignore_index=True,
    )
    aligned = aligned_windows(traces, frame_times(), outcomes, (-2.0, 5.0), ("task",))
    assert aligned.pulses.tolist() == [1]
    assert aligned.n_dropped == 1  # port-finding trials are filtered out, not dropped


def test_pooled_responses_average_cells_after_averaging_trials():
    rng = np.random.default_rng(0)
    traces = rng.normal(size=(5, N))
    outcomes = outcome_table([100, 150, 200, 250], [1, 1, 2, 2])
    aligned = aligned_windows(traces, frame_times(), outcomes, (-0.4, 0.4), ("task",))
    pooled = pooled_responses(aligned, np.array([True, True, True, False, False]))
    per_cell = np.mean(aligned.windows[:3][:, [0, 1]], axis=1)  # trial means of the 1-pulse trials
    np.testing.assert_allclose(pooled[1]["mean"], per_cell.mean(axis=0))
    np.testing.assert_allclose(pooled[1]["sem"], per_cell.std(axis=0, ddof=1) / np.sqrt(3))
    assert pooled[1]["n_cells"] == 3
    assert 0 not in pooled  # no trial of that kind


def test_pooling_no_cells_gives_an_empty_result():
    traces = np.zeros((2, N))
    aligned = aligned_windows(
        traces, frame_times(), outcome_table([100], [1]), (-0.4, 0.4), ("task",)
    )
    assert pooled_responses(aligned, np.zeros(2, bool)) == {}


# --- response windows and scores ------------------------------------------------------------------


def test_response_window_covers_frames_whose_start_lies_within_the_window():
    assert response_frames(2.0, 0.133) == 15  # frames starting at 0 .. 1.86 s


def test_reward_mask_marks_the_response_window_after_rewarded_outcomes_only():
    event_index = np.array([10, 40, 70])
    pulses = np.array([0, 2, 4])
    rewarded = reward_frame_mask(100, event_index, pulses, 5, rewarded=True)
    assert np.flatnonzero(rewarded).tolist() == [*range(40, 45), *range(70, 75)]
    unrewarded = reward_frame_mask(100, event_index, pulses, 5, rewarded=False)
    assert np.flatnonzero(unrewarded).tolist() == list(range(10, 15))


def test_distinction_score_is_the_normalised_contrast_of_mean_activity():
    trace = np.array([[4.0, 4.0, 1.0, 1.0, 1.0, 1.0]])
    mask = np.array([True, True, False, False, False, False])
    # A_reward = 4, A_other = 1 -> (4 - 1) / (4 + 1)
    assert distinction_scores(trace, mask)[0] == pytest.approx(0.6)


def test_distinction_score_of_a_silent_cell_is_nan():
    assert np.isnan(distinction_scores(np.zeros((1, 6)), np.array([1, 1, 0, 0, 0, 0], bool))[0])


def test_scores_need_frames_in_and_out_of_the_window():
    with pytest.raises(ValueError, match="window"):
        distinction_scores(np.ones((1, 4)), np.zeros(4, bool))
    with pytest.raises(ValueError, match="window"):
        distinction_scores(np.ones((1, 4)), np.ones(4, bool))


def test_shifted_scores_equal_rolling_the_activity_and_rescoring():
    rng = np.random.default_rng(1)
    traces = np.abs(rng.normal(size=(3, 500)))
    mask = np.zeros(500, bool)
    mask[[20, 21, 22, 200, 201, 202, 350]] = True
    shifts = np.array([30, 77, 250, 470])
    fast = shifted_distinction_scores(traces, mask, shifts)
    slow = np.array(
        [[distinction_scores(np.roll(row, s)[None], mask)[0] for s in shifts] for row in traces]
    )
    np.testing.assert_allclose(fast, slow, atol=1e-10)


def test_circular_shifts_keep_30_s_from_both_ends():
    shifts = circular_shifts(2000, PERIOD, 1000, 30.0, np.random.default_rng(2))
    low = int(np.ceil(30.0 / PERIOD))
    assert shifts.min() >= low and shifts.max() <= 2000 - low
    assert len(shifts) == 1000 and len(np.unique(shifts)) > 500


def test_circular_shifts_refuse_a_recording_too_short_for_the_minimum_shift():
    with pytest.raises(ValueError, match="shorter"):
        circular_shifts(300, PERIOD, 10, 30.0, np.random.default_rng(0))


# --- shuffle test ---------------------------------------------------------------------------------


def make_cells(rng, n_noise, n_locked, n_frames=4000):
    mask = np.zeros(n_frames, bool)
    starts = np.arange(150, n_frames - 150, 130)
    for s in starts:
        mask[s : s + 15] = True
    smooth = np.abs(
        np.array(
            [
                np.convolve(rng.normal(size=n_frames + 8), np.ones(9), "valid")
                for _ in range(n_noise + n_locked)
            ]
        )
    )
    smooth[n_noise:] += 4.0 * mask  # these cells respond in every window
    return smooth, mask


def test_about_five_percent_of_cells_with_no_response_pass_the_shuffle_test():
    rng = np.random.default_rng(3)
    traces, mask = make_cells(rng, n_noise=600, n_locked=0)
    shifts = circular_shifts(traces.shape[1], PERIOD, 1000, 30.0, rng)
    result = responsive_cells(traces, mask, shifts)
    fraction = result["responsive"].mean()
    assert 0.025 < fraction < 0.085


def test_cells_locked_to_the_outcome_are_detected():
    rng = np.random.default_rng(4)
    traces, mask = make_cells(rng, n_noise=50, n_locked=10)
    shifts = circular_shifts(traces.shape[1], PERIOD, 1000, 30.0, rng)
    result = responsive_cells(traces, mask, shifts)
    assert result["responsive"][50:].all()
    assert result["responsive"][:50].mean() < 0.15
    assert (result["score"][50:] > result["null_95th"][50:]).all()
    assert (result["p_value"][50:] < 0.01).all()


def test_responsive_flag_is_false_for_a_silent_cell():
    traces = np.zeros((1, 500))
    mask = np.zeros(500, bool)
    mask[100:120] = True
    shifts = circular_shifts(500, PERIOD, 50, 5.0, np.random.default_rng(0))
    result = responsive_cells(traces, mask, shifts)
    assert not result["responsive"][0] and np.isnan(result["score"][0])


def test_chance_summary_compares_the_count_with_five_percent():
    responsive = np.zeros(200, bool)
    responsive[:10] = True
    summary = chance_summary(responsive)
    assert summary["n_cells"] == 200 and summary["n_responsive"] == 10
    assert summary["expected_by_chance"] == pytest.approx(10.0)
    assert summary["p_binomial"] > 0.4  # exactly the chance level
    responsive[:40] = True
    assert chance_summary(responsive)["p_binomial"] < 1e-6
