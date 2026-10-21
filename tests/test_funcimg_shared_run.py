"""Shared-signal analysis of one run: residual reward responses, fit frames, controls, errors."""

import numpy as np
import pytest

pytest.importorskip("mini2p_preprocessing")  # the optional funcimg extra

from src.funcimg_analysis.frame_times import RunMismatchError  # noqa: E402
from src.funcimg_analysis.shared_signal import detrended_z, row_correlation  # noqa: E402
from src.funcimg_analysis.shared_signal_run import (  # noqa: E402
    NoFitFramesError,
    SharedSignalUnavailableError,
    TooFewCellsError,
    TooShortForSharedSignalError,
)
from tests.funcimg_shared_fixtures import (  # noqa: E402
    N_CELLS,
    N_FRAMES,
    SETTINGS,
    analyse,
    response_2s,
    run_inputs,
)


def test_a_reward_response_that_is_only_in_the_background_vanishes_from_the_residual():
    result = analyse(run_inputs(np.random.default_rng(0), "background"))
    original, residual = response_2s(result, "original"), response_2s(result, "residual")
    assert original > 0.5
    assert -0.1 * original < residual < 0.25 * original  # gone, and not overcorrected below 0


def test_a_reward_response_that_is_only_in_the_cells_survives_background_removal():
    result = analyse(run_inputs(np.random.default_rng(1), "cells"))
    original, residual = response_2s(result, "original"), response_2s(result, "residual")
    assert original > 0.5
    assert 0.7 * original < residual < 1.3 * original  # same units: z of the detrended cell


def test_the_response_of_a_dff_view_is_removed_the_same_way():
    result = analyse(run_inputs(np.random.default_rng(2), "background"))
    view = ("green", "dF/F0 z")
    original, residual = (
        response_2s(result, "original", view=view),
        response_2s(result, "residual", view=view),
    )
    assert original > 0.5 and abs(residual) < 0.25 * original


def test_a_small_leak_of_the_cells_response_into_the_background_is_not_fitted_into_the_slope():
    """Fitting on all frames lets the leak raise the slope and eat the cells' own response."""
    inputs = run_inputs(np.random.default_rng(3), "spill")
    masked = response_2s(analyse(inputs), "residual")
    unmasked = response_2s(analyse(inputs, fit_exclude_outcomes=False), "residual")
    original = response_2s(analyse(inputs), "original")
    # the leak itself is rightly subtracted (about a fifth of the response), the rest survives
    assert 0.65 * original < masked < 0.95 * original
    assert masked > unmasked + 0.15 * original


def test_the_fraction_of_frames_the_slope_was_fitted_on_is_reported():
    inputs = run_inputs(np.random.default_rng(4), "none")
    assert 0.1 < analyse(inputs)["fit_fraction"] < 0.5
    assert analyse(inputs, fit_exclude_outcomes=False)["fit_fraction"] == 1.0


def test_outcome_windows_that_cover_the_whole_run_leave_nothing_to_fit_and_raise():
    inputs = run_inputs(np.random.default_rng(5), "none")
    with pytest.raises(NoFitFramesError, match="fit_exclude"):
        analyse(inputs, fit_exclude_window_s=None)  # the analysis window overlaps itself


def test_statistics_are_given_per_channel_and_raw_measure_with_the_requested_shifts():
    result = analyse(run_inputs(np.random.default_rng(6), "none"))
    assert set(result["stats"]) == {("green", "F"), ("green", "dF/F0")}
    stats = result["stats"][("green", "F")]
    assert stats["null_partial_average"].shape == (SETTINGS["n_shifts"], N_CELLS)
    assert stats["null_partial_neuropil_average"].shape == (SETTINGS["n_shifts"], N_CELLS)
    assert stats["r_bg_avg"] > 0.3  # the cells follow the field signal
    assert np.isfinite(stats["bg_reliability"]) and np.isfinite(stats["partial_r_neuropil"]).all()
    assert result["n_cells"] == N_CELLS


def test_only_the_requested_null_kinds_are_computed():
    inputs = run_inputs(np.random.default_rng(7), "none")
    stats = analyse(inputs, null_kinds=("r_bg",))["stats"][("green", "F")]
    assert {k for k in stats if k.startswith("null_")} == {"null_r_bg"}
    opt_in = analyse(inputs, null_kinds=("partial_background", "r_bg_avg"))["stats"]
    assert {"null_partial_background", "null_r_bg_avg"} <= set(opt_in[("green", "F")])


def test_only_the_requested_measures_are_analysed():
    result = analyse(run_inputs(np.random.default_rng(8), "none"), measures=("F z",))
    assert set(result["stats"]) == {("green", "F")}
    assert set(result["original"]["means"]) == {("green", "F z")}


def test_the_same_seed_gives_the_same_chance_levels_and_another_seed_does_not():
    inputs = run_inputs(np.random.default_rng(9), "none")
    first = analyse(inputs)["stats"][("green", "F")]["null_r_bg"]
    np.testing.assert_array_equal(first, analyse(inputs)["stats"][("green", "F")]["null_r_bg"])
    views, background, neuropil, halves = inputs
    from src.funcimg_analysis.shared_signal_run import analyse_shared_signal
    from tests.funcimg_shared_fixtures import MEASURES, WINDOW_S, frame_times, outcomes

    other = analyse_shared_signal(
        views,
        background,
        frame_times(),
        outcomes(),
        neuropil=neuropil,
        background_halves=halves,
        rng=np.random.default_rng(99),
        window_s=WINDOW_S,
        phases=("task",),
        measures=MEASURES,
        **SETTINGS,
    )["stats"][("green", "F")]["null_r_bg"]
    assert not np.array_equal(first, other)


# --- red is analysed against the red background ------------------------------------------------


def test_the_red_statistics_use_the_red_background_not_the_green_one():
    inputs = run_inputs(np.random.default_rng(10), "none", channels=("green", "red"))
    views, background, *_ = inputs
    result = analyse(inputs)
    red = result["stats"][("red", "F")]
    wrong = row_correlation(
        detrended_z(views[("red", "F")], 113)[0],
        detrended_z(background[("green", "F")][None], 113)[0][0],
    )
    assert np.median(red["r_bg"]) > 0.3  # against its own background
    assert abs(np.median(wrong)) < 0.3 < np.median(red["r_bg"])  # a green background would differ
    assert not np.allclose(red["r_bg"], result["stats"][("green", "F")]["r_bg"])


def test_a_reward_response_in_the_red_background_alone_is_removed_from_the_red_cells():
    inputs = run_inputs(
        np.random.default_rng(11), "none", channels=("green", "red"), reward_in_red="background"
    )
    result = analyse(inputs)
    view = ("red", "F z")
    original, residual = (
        response_2s(result, "original", view=view),
        response_2s(result, "residual", view=view),
    )
    assert original > 0.5 and abs(residual) < 0.25 * original
    green = response_2s(result, "residual", view=("green", "F z"))
    assert abs(green) < 0.3  # nothing in green to start with, and red's removal does not add any


# --- refusals -----------------------------------------------------------------------------------


def test_a_run_too_short_for_the_shifts_or_the_window_is_refused():
    inputs = run_inputs(np.random.default_rng(12), "none")
    with pytest.raises(TooShortForSharedSignalError, match="shift"):
        analyse(inputs, min_shift_s=70.0)  # 2 x 70 s > 120 s
    with pytest.raises(TooShortForSharedSignalError, match="window"):
        analyse(inputs, detrend_window_s=500.0)


def test_a_single_roi_is_refused_with_its_own_error():
    views, background, neuropil, halves = run_inputs(np.random.default_rng(13), "cells")
    one = {k: v[:1] for k, v in views.items()}
    neuropil = {k: v[:1] for k, v in neuropil.items()}
    with pytest.raises(TooFewCellsError, match="two"):
        analyse((one, background, neuropil, halves))


def test_all_refusals_share_one_base_class_so_a_run_can_be_listed_and_left_out():
    for error in (TooShortForSharedSignalError, TooFewCellsError, NoFitFramesError):
        assert issubclass(error, SharedSignalUnavailableError)


def test_a_background_of_another_length_than_the_frame_times_raises():
    views, background, neuropil, halves = run_inputs(np.random.default_rng(14), "none")
    background = {key: trace[:-5] for key, trace in background.items()}
    with pytest.raises(RunMismatchError, match="frames"):
        analyse((views, background, neuropil, halves))


def test_a_constant_background_raises():
    views, background, neuropil, halves = run_inputs(np.random.default_rng(15), "none")
    background = {key: np.full(N_FRAMES, 5.0) for key in background}
    with pytest.raises(ValueError, match="constant"):
        analyse((views, background, neuropil, halves))


def test_raw_measures_are_not_accepted_because_the_analysis_is_in_z_units():
    with pytest.raises(ValueError, match="F z"):
        analyse(run_inputs(np.random.default_rng(16), "none"), measures=("F",))


def test_a_missing_neuropil_or_background_half_names_what_is_missing():
    views, background, neuropil, halves = run_inputs(np.random.default_rng(17), "none")
    with pytest.raises(ValueError, match="neuropil"):
        analyse((views, background, {}, halves))
    with pytest.raises(ValueError, match="background halves"):
        analyse((views, background, neuropil, {}))


def test_the_input_traces_are_not_modified():
    inputs = run_inputs(np.random.default_rng(18), "background")
    before = [{k: v.copy() for k, v in d.items()} for d in inputs]
    analyse(inputs)
    for got, want in zip(inputs, before, strict=True):
        for key in want:
            np.testing.assert_array_equal(got[key], want[key])
