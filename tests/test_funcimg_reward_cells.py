"""Reward-responsive cells: response index, split halves, circular-shift null, cross-validation."""

import numpy as np
import pytest
from scipy.stats import spearmanr, ttest_ind

from src.funcimg_analysis.reward_cells import (
    TooFewTrialsError,
    analyse_reward_cells,
    exceeds_null,
    response_index,
    split_halves,
    trial_responses,
)
from tests.funcimg_reward_fixtures import (
    N_FRAMES,
    PERIOD,
    SETTINGS,
    frame_times,
    make_cells,
    outcomes,
)

# --- the response index -------------------------------------------------------------------------


def test_the_rewarded_index_is_the_mean_over_the_sd_of_the_rewarded_trials_only():
    per_trial = np.array([[1.0, 2.0, 3.0, 100.0]])  # the 0-pulse trial is not a rewarded trial
    pulses = np.array([1, 2, 3, 0])
    index = response_index(per_trial, pulses, "rewarded", np.ones(4, bool))
    assert index[0] == pytest.approx(2.0 / 1.0)


def test_the_four_vs_zero_index_is_a_welch_t():
    rng = np.random.default_rng(0)
    four, zero = rng.normal(1.0, 1.0, 12), rng.normal(0.0, 3.0, 9)
    per_trial = np.concatenate([four, zero, [50.0, 50.0]])[None, :]  # 1-pulse trials are ignored
    pulses = np.array([4] * 12 + [0] * 9 + [1, 1])
    index = response_index(per_trial, pulses, "4_vs_0", np.ones(len(pulses), bool))
    assert index[0] == pytest.approx(ttest_ind(four, zero, equal_var=False).statistic)


def test_only_the_trials_in_the_mask_count():
    per_trial = np.array([[1.0, 2.0, 3.0, 50.0, 60.0]])
    pulses = np.ones(5, int)
    mask = np.array([True, True, True, False, False])
    assert response_index(per_trial, pulses, "rewarded", mask)[0] == pytest.approx(2.0)


def test_an_undefined_index_is_nan_not_inf_or_a_guess():
    per_trial = np.array(
        [[1.0, 1.0, 1.0], [np.nan, np.nan, 2.0], [0.0, 5.0, 0.0]]  # no spread; one value; fine
    )
    index = response_index(per_trial, np.ones(3, int), "rewarded", np.ones(3, bool))
    assert np.isnan(index[0]) and np.isnan(index[1]) and np.isfinite(index[2])


def test_an_unknown_contrast_is_refused():
    with pytest.raises(ValueError, match="contrast"):
        response_index(np.zeros((1, 4)), np.ones(4, int), "all", np.ones(4, bool))


# --- trial responses ----------------------------------------------------------------------------


def step_trace():
    trace = np.zeros((1, 100))
    trace[0, 50:] = 1.0  # a step at frame 50
    return trace


def test_a_trials_response_is_the_response_mean_minus_the_baseline_mean():
    offsets = np.arange(-4, 5)
    out = trial_responses(
        step_trace(), np.array([50]), offsets, PERIOD, (-4 * PERIOD, 0.0), (0.0, 4.5 * PERIOD)
    )
    assert out.shape == (1, 1) and out[0, 0] == pytest.approx(1.0)


def test_the_frame_being_acquired_at_the_event_is_left_out_of_that_trials_baseline():
    trace = np.zeros((1, 100))
    trace[0, 49] = 8.0  # the last pre-event frame, partly acquired after the event
    offsets = np.arange(-4, 5)
    args = (trace, np.array([50]), offsets, PERIOD, (-4 * PERIOD, 0.0), (0.0, 4.5 * PERIOD))
    assert trial_responses(*args)[0, 0] == pytest.approx(-2.0)  # baseline includes the 8
    spanned = trial_responses(*args, baseline_spanned=np.array([True]))
    assert spanned[0, 0] == pytest.approx(0.0)


def test_circular_shifts_wrap_windows_around_the_end_of_the_trace():
    trace = np.zeros((1, 100))
    trace[0, :3] = 1.0
    out = trial_responses(
        trace, np.array([99]), np.arange(-2, 5), PERIOD, (-2 * PERIOD, 0.0), (0.0, 4.5 * PERIOD),
        wrap=True,
    )  # fmt: skip
    assert out[0, 0] == pytest.approx(3 / 5)  # frames 99, 0, 1, 2, 3: three of five are 1


def test_an_empty_baseline_or_response_range_is_refused():
    with pytest.raises(ValueError, match="no frames"):
        trial_responses(
            step_trace(), np.array([50]), np.arange(-4, 5), PERIOD, (-9.0, -8.0), (0.0, 1.0)
        )


# --- the split into halves ----------------------------------------------------------------------


def test_halves_alternate_within_each_pulse_category():
    pulses = np.arange(100) % 5
    half_a = split_halves(pulses, np.random.default_rng(0))
    for category in range(5):
        in_category = half_a[pulses == category]
        assert (in_category[:-1] != in_category[1:]).all()  # A, B, A, B, ... in trial order
        assert abs(in_category.sum() - (~in_category).sum()) <= 1


def test_the_split_is_deterministic_for_a_seed_and_the_starting_half_is_seeded():
    pulses = np.arange(100) % 5
    first = split_halves(pulses, np.random.default_rng(3))
    np.testing.assert_array_equal(first, split_halves(pulses, np.random.default_rng(3)))
    starts = {bool(split_halves(pulses, np.random.default_rng(s))[0]) for s in range(20)}
    assert starts == {True, False}


# --- the null and cross-validation --------------------------------------------------------------


def run(traces, seed=0, **overrides):
    outs, _, _ = outcomes(np.random.default_rng(1))
    return analyse_reward_cells(
        {"dF/F0": traces},
        frame_times(),
        outs,
        rng=np.random.default_rng(seed),
        **{**SETTINGS, **overrides},
    )


def planted_run(n_cells=100, n_planted=20, amplitude=1.0, seed=2):
    rng = np.random.default_rng(seed)
    outs, frames, pulses = outcomes(np.random.default_rng(1))
    return make_cells(rng, frames, pulses, n_cells, n_planted, amplitude)


def test_on_pure_noise_about_five_percent_of_cells_exceed_their_null():
    result = run(planted_run(n_cells=300, n_planted=0))["measures"]["dF/F0"]
    fraction = result["null"]["n_exceed_a"] / 300
    assert 0.01 < fraction < 0.11
    assert result["null"]["expected_a"] == pytest.approx(15.0)


def test_with_planted_responsive_cells_far_more_than_five_percent_exceed():
    result = run(planted_run(n_cells=100, n_planted=30))["measures"]["dF/F0"]["null"]
    assert result["n_exceed_a"] >= 25 and result["n_exceed_b_of_a"] >= 22
    assert result["n_exceed_b_of_a"] <= result["n_exceed_a"]


def test_cross_validated_top_cells_recover_the_planted_cells():
    measure = run(planted_run(n_cells=100, n_planted=20))["measures"]["dF/F0"]
    top = measure["order"][:15]
    assert np.sum(top < 20) >= 14
    noise = run(planted_run(n_cells=100, n_planted=0))["measures"]["dF/F0"]
    assert measure["rank_correlation"] > 0.35  # the ranking is stable across halves
    assert abs(noise["rank_correlation"]) < 0.25  # and unrelated between halves for noise


def test_cross_validated_top_cells_of_pure_noise_have_no_response_on_the_held_out_half():
    held_out, selected = [], []
    for seed in range(8):  # one dataset's chance alignment moves the averages by about 0.03
        noise = planted_run(n_cells=200, n_planted=0, seed=seed)
        cross_validated = run(noise, n_shifts=5)["measures"]["dF/F0"]
        biased = run(noise, n_shifts=5, rank_on="all")["measures"]["dF/F0"]
        held_out.append(np.nanmean(cross_validated["index_b"][cross_validated["order"][:20]]))
        selected.append(np.nanmean(biased["index_all"][biased["order"][:20]]))
    # the index of a noise cell has an SD of about 0.1 (112 rewarded trials), so the top tenth of
    # 200 cells ranked on the trials they are judged on averages about 0.17; held out, about 0
    assert abs(np.mean(held_out)) < 0.04  # no selection bias
    assert np.mean(selected) > 0.12 and np.mean(selected) > abs(np.mean(held_out)) + 0.08


def test_ranking_on_all_trials_plots_all_trials_and_says_it_is_not_cross_validated():
    result = run(planted_run(), rank_on="all")
    assert result["rank_on"] == "all" and result["cross_validated"] is False
    assert result["plot_mask"].all() and result["rank_mask"].all()


def test_the_cross_validated_ranking_uses_half_a_and_plots_half_b():
    result = run(planted_run())
    np.testing.assert_array_equal(result["rank_mask"], result["halves"])
    np.testing.assert_array_equal(result["plot_mask"], ~result["halves"])
    assert result["cross_validated"] is True


def test_the_rank_correlation_is_spearman_of_the_two_halves_indices():
    measure = run(planted_run())["measures"]["dF/F0"]
    ok = np.isfinite(measure["index_a"]) & np.isfinite(measure["index_b"])
    expected = spearmanr(measure["index_a"][ok], measure["index_b"][ok]).statistic
    assert measure["rank_correlation"] == pytest.approx(expected)


def test_plot_data_are_baseline_subtracted_means_per_pulse_count_over_the_plotted_trials():
    result = run(planted_run(amplitude=2.0))
    measure = result["measures"]["dF/F0"]
    offsets, means = result["offsets"], measure["means"]
    assert set(means) == {0, 1, 2, 3, 4}
    assert offsets[0] * PERIOD == pytest.approx(-2.0, abs=PERIOD) and offsets[-1] * PERIOD > 4.8
    after = (offsets >= 0) & (offsets * PERIOD < 2)
    before = offsets < 0
    assert np.abs(means[4][:20][:, before].mean()) < 0.1  # baseline-subtracted, planted cells
    assert means[4][:20][:, after].mean() > 0.5  # the planted response, in robust-z units
    assert abs(means[0][:20][:, after].mean()) < 0.3  # no response without reward
    assert measure["rewarded_mean"].shape == means[4].shape
    assert result["n_trials_plot"][4] == result["plot_mask"][result["pulses"] == 4].sum()


def test_the_order_lists_cells_by_descending_rank_index_and_leaves_out_undefined_ones():
    traces = planted_run(n_cells=30, n_planted=5)
    traces[7] = 1.0  # constant: no scale, no index
    measure = run(traces)["measures"]["dF/F0"]
    assert 7 not in measure["order"] and len(measure["order"]) == 29
    ranked = measure["index_a"][measure["order"]]
    assert (np.diff(ranked) <= 0).all()


def test_the_four_vs_zero_contrast_finds_the_planted_cells_too():
    measure = run(planted_run(), contrast="4_vs_0")["measures"]["dF/F0"]
    assert np.sum(measure["order"][:15] < 20) >= 13


def test_each_measure_gets_its_own_analysis():
    traces = planted_run()
    outs, _, _ = outcomes(np.random.default_rng(1))
    result = analyse_reward_cells(
        {"a": traces, "b": traces[::-1]},
        frame_times(),
        outs,
        rng=np.random.default_rng(0),
        **SETTINGS,
    )
    assert set(result["measures"]) == {"a", "b"}
    assert result["measures"]["a"]["order"][0] != result["measures"]["b"]["order"][0]


def test_too_few_trials_in_a_half_raise_instead_of_returning_a_noisy_index():
    outs, frames, pulses = outcomes(np.random.default_rng(1), np.array([100, 200, 300, 400]))
    with pytest.raises(TooFewTrialsError, match="at least 2"):
        analyse_reward_cells(
            {"m": np.ones((3, N_FRAMES))},
            frame_times(),
            outs,
            rng=np.random.default_rng(0),
            **SETTINGS,
        )


def test_the_null_needs_a_run_longer_than_twice_the_minimum_shift():
    with pytest.raises(ValueError, match="too few"):
        run(planted_run(), min_shift_s=500.0)


def test_exceeds_null_compares_each_cell_with_its_own_95th_percentile():
    null = np.tile(np.linspace(0, 1, 101)[:, None], (1, 3)) * np.array([1.0, 2.0, 3.0])
    index = np.array([0.97, 0.97, np.nan])  # the 95th percentiles are 0.95, 1.9 and 2.85
    np.testing.assert_array_equal(exceeds_null(index, null), [True, False, False])
