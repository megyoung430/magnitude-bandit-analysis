"""Zong et al. 2022 signal processing: baseline, transients, SNR, deconvolution, clear traces."""

import numpy as np
import pytest

pytest.importorskip("mini2p_preprocessing")  # the optional funcimg extra (suite2p's OASIS)

from src.funcimg_analysis.zong2022_signal import (  # noqa: E402
    Zong2022Settings,
    analyse_traces,
    clear_traces,
    deconvolve,
    event_counts_by_tau,
    half_window_frames,
    local_baseline_sd,
    min_transient_frames,
    moving_percentile,
    moving_std,
    neuropil_correct,
    running_baseline,
    significant_transients,
    transient_bounds,
    transient_snr,
)
from tests.zong2022_fixtures import (  # noqa: E402
    FS_HZ,
    N_FRAMES,
    PERIOD_S,
    kernel,
    planted_cell,
    planted_cells,
    spike_train,
)

HALF = half_window_frames(15.0, PERIOD_S)


# --- settings and small helpers -------------------------------------------------------------------


def test_defaults_are_the_papers_values():
    settings = Zong2022Settings()
    assert settings.neuropil_coefficient == 0.7
    assert settings.baseline_percentile == 8
    assert settings.baseline_half_window_s == 15
    assert settings.baseline_sd_fraction == 0.1
    assert settings.transient_sd == 2
    assert settings.transient_min_s == 0.75
    assert settings.min_snr == 3
    assert settings.min_events is None


@pytest.mark.parametrize(
    "bad",
    [
        {"baseline_percentile": 101},
        {"baseline_sd_fraction": 0},
        {"baseline_sd_fraction": 1.5},
        {"tau_s": 0},
        {"transient_min_s": -1},
        {"min_events": 0},
        {"neuropil_coefficient": -0.1},
    ],
)
def test_nonsense_settings_raise(bad):
    with pytest.raises(ValueError):
        Zong2022Settings(**bad)


def test_neuropil_correction_is_fcell_minus_a_fixed_multiple_of_fnp():
    f, fneu = np.array([[10.0, 20.0]]), np.array([[2.0, 4.0]])
    np.testing.assert_allclose(neuropil_correct(f, fneu, 0.7), [[8.6, 17.2]])


def test_neuropil_correction_refuses_nan_and_shape_mismatch():
    with pytest.raises(ValueError, match="non-finite"):
        neuropil_correct(np.array([[1.0, np.nan]]), np.ones((1, 2)), 0.7)
    with pytest.raises(ValueError, match="shape"):
        neuropil_correct(np.ones((2, 3)), np.ones((2, 4)), 0.7)


def test_half_window_is_seconds_in_whole_frames():
    assert half_window_frames(15.0, 0.133) == 113


def test_a_transient_must_last_longer_than_the_minimum_duration():
    # 0.75 s at 0.133 s per frame is 5.64 frames: six frames (0.798 s) pass, five (0.665 s) do not
    assert min_transient_frames(0.75, 0.133) == 6
    # a duration that is exactly a whole number of frames must be exceeded, not matched
    assert min_transient_frames(0.5, 0.25) == 3


# --- moving windows -------------------------------------------------------------------------------


def test_moving_percentile_matches_brute_force_including_truncated_edges():
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    half = 5
    expected = [np.percentile(x[max(0, i - half) : i + half + 1], 8) for i in range(len(x))]
    np.testing.assert_allclose(moving_percentile(x, half, 8), expected)


def test_moving_std_matches_brute_force_including_truncated_edges():
    rng = np.random.default_rng(1)
    x = rng.normal(5, 2, size=60)
    half = 5
    expected = [np.std(x[max(0, i - half) : i + half + 1]) for i in range(len(x))]
    np.testing.assert_allclose(moving_std(x, half), expected, atol=1e-9)


def test_moving_windows_refuse_a_window_longer_than_the_trace():
    with pytest.raises(ValueError, match="window"):
        moving_percentile(np.ones(10), 6, 8)
    with pytest.raises(ValueError, match="window"):
        moving_std(np.ones(10), 6)


# --- baseline -------------------------------------------------------------------------------------


def test_f0_tracks_the_drift_and_is_centred_on_the_baseline_points():
    cell = planted_cell(np.random.default_rng(2))
    f0, baseline = running_baseline(cell.fcorr, HALF, 8.0, 0.1)
    # centred: by construction the mean residual at baseline points is zero
    assert np.mean(cell.fcorr[baseline] - f0[baseline]) == pytest.approx(0, abs=1e-9)
    # tracking: away from transients F0 follows the planted drift to within the noise
    quiet = baseline & (cell.fcorr < cell.f0_true * 1.05)
    assert np.abs(f0[quiet] - cell.f0_true[quiet]).mean() < 0.6
    # and the slow drift (a ±5 swing) is followed, not flattened
    assert np.corrcoef(f0, cell.f0_true)[0, 1] > 0.9


def test_baseline_points_exclude_frames_near_transients():
    cell = planted_cell(np.random.default_rng(3))
    _, baseline = running_baseline(cell.fcorr, HALF, 8.0, 0.1)
    for frame in cell.spike_frames:
        assert not baseline[frame : frame + 20].any()
    assert baseline.mean() > 0.3  # but plenty of the recording is baseline


def test_a_constant_trace_is_all_baseline():
    f0, baseline = running_baseline(np.full(500, 50.0), 20, 8.0, 0.1)
    assert baseline.all()
    np.testing.assert_allclose(f0, 50.0)


def test_local_baseline_sd_is_the_sd_of_baseline_points_in_the_window():
    rng = np.random.default_rng(4)
    dff = rng.normal(0, 0.01, 2000)
    dff[900:950] += 1.0  # a transient, not baseline
    baseline = np.ones(2000, bool)
    baseline[895:960] = False
    sd = local_baseline_sd(dff, baseline, half=100, min_points=30)
    window = slice(1000 - 100, 1000 + 101)
    expected = np.std(dff[window][baseline[window]])
    assert sd[1000] == pytest.approx(expected, rel=1e-6)
    # a window dominated by the transient still reports the baseline points' spread, not its own
    assert sd[925] < 0.02


def test_local_baseline_sd_falls_back_to_the_global_value_when_a_window_has_too_few_points():
    rng = np.random.default_rng(5)
    dff = rng.normal(0, 0.01, 1000)
    baseline = np.zeros(1000, bool)
    baseline[:400] = True
    sd = local_baseline_sd(dff, baseline, half=20, min_points=30)
    assert sd[900] == pytest.approx(np.std(dff[baseline]))  # no baseline point within ±20 frames
    assert np.isfinite(sd).all()


def test_local_baseline_sd_needs_baseline_points():
    with pytest.raises(ValueError, match="baseline"):
        local_baseline_sd(np.ones(50), np.zeros(50, bool), half=5, min_points=3)


# --- transients -----------------------------------------------------------------------------------


def test_transient_bounds_lists_runs_of_true():
    mask = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], bool)
    assert transient_bounds(mask) == [(1, 3), (5, 6), (7, 10)]
    assert transient_bounds(np.zeros(4, bool)) == []
    assert transient_bounds(np.ones(3, bool)) == [(0, 3)]


def test_only_runs_above_the_threshold_and_long_enough_are_significant():
    dff = np.zeros(100)
    dff[10:20] = 1.0  # long and large
    dff[30:33] = 1.0  # large but 3 frames
    dff[50:70] = 0.15  # long but below 2 SD (SD = 0.1 -> threshold 0.2)
    mask = significant_transients(dff, np.full(100, 0.1), n_sd=2.0, min_frames=6)
    assert np.flatnonzero(mask).tolist() == list(range(10, 20))


def test_transient_threshold_uses_the_local_sd_at_each_frame():
    dff = np.zeros(100)
    dff[10:20] = 0.5
    dff[60:70] = 0.5
    sd = np.full(100, 0.1)
    sd[60:70] = 0.5  # noisier here: 0.5 is below 2 SD
    mask = significant_transients(dff, sd, n_sd=2.0, min_frames=6)
    assert mask[10:20].all() and not mask[60:70].any()


def test_nan_dff_is_never_a_transient():
    dff = np.full(30, np.nan)
    assert not significant_transients(dff, np.full(30, 0.1), 2.0, 3).any()


def test_planted_transients_are_found_and_decoys_ignored():
    cell = planted_cell(np.random.default_rng(6))
    settings = Zong2022Settings()
    result = analyse_traces(cell.fcorr[None], np.zeros((1, N_FRAMES)), PERIOD_S, settings)
    mask = result.transient_mask[0]
    for frame in cell.spike_frames:
        assert mask[frame : frame + 10].all(), f"transient at frame {frame} missed"
    bounds = transient_bounds(mask)
    assert len(bounds) == len(cell.spike_frames)  # nothing else
    assert not mask[cell.short_frames[0] - 5 : cell.short_frames[0] + 10].any()
    assert not mask[cell.weak_frames[0] : cell.weak_frames[0] + 40].any()


# --- SNR ------------------------------------------------------------------------------------------


def test_snr_matches_a_hand_computed_value():
    dff = np.tile([0.1, -0.1], 50)  # outside the transients every step is 0.2 in size
    first = np.arange(11) / 1.0  # 0..10: 90th percentile 9.0
    second = np.arange(1, 7) / 1.0  # 1..6: 90th percentile 5.5
    dff[20:31], dff[60:66] = first, second
    mask = np.zeros(100, bool)
    mask[20:31] = mask[60:66] = True
    out = transient_snr(dff, mask)
    assert out["n_transients"] == 2
    assert out["signal"] == pytest.approx((9.0 + 5.5) / 2)
    assert out["noise"] == pytest.approx(0.2)
    assert out["snr"] == pytest.approx(7.25 / 0.2)


def test_noise_ignores_steps_that_touch_a_transient():
    dff = np.tile([0.1, -0.1], 50)
    dff[40:45] = 50.0  # big jumps into and out of the transient must not count as noise
    mask = np.zeros(100, bool)
    mask[40:45] = True
    assert transient_snr(dff, mask)["noise"] == pytest.approx(0.2)


def test_snr_is_nan_without_transients_or_without_noise():
    dff = np.tile([0.1, -0.1], 50)
    none = transient_snr(dff, np.zeros(100, bool))
    assert none["n_transients"] == 0 and np.isnan(none["snr"])
    flat = np.zeros(100)
    flat[10:20] = 1.0
    mask = np.zeros(100, bool)
    mask[10:20] = True
    assert np.isnan(transient_snr(flat, mask)["snr"])  # zero noise: undefined, not infinite


# --- deconvolution --------------------------------------------------------------------------------


def test_oasis_recovers_planted_spike_times_for_the_matching_tau():
    rng = np.random.default_rng(7)
    frames = np.array([100, 260, 420, 600, 800, 1000])
    amplitudes = np.array([1.0, 2.0, 1.5, 1.0, 3.0, 2.0])
    tau = 0.7
    trace = np.convolve(spike_train(1200, frames, amplitudes), kernel(tau))[:1200]
    noisy = trace + rng.normal(0, 0.05, 1200)
    spikes = deconvolve(noisy[None], FS_HZ, tau)[0]
    found = np.flatnonzero(spikes > 0.5 * amplitudes.min())
    assert sorted(found.tolist()) == frames.tolist()
    np.testing.assert_allclose(spikes[frames], amplitudes, rtol=0.2)
    assert (spikes >= 0).all()


def test_oasis_with_a_wrong_tau_misplaces_amplitude():
    # the planted decay is 1 s; assuming 0.1 s explains each decaying tail with extra spikes
    trace = np.convolve(spike_train(600, [100], [1.0]), kernel(1.0))[:600]
    right = deconvolve(trace[None], FS_HZ, 1.0)[0]
    wrong = deconvolve(trace[None], FS_HZ, 0.1)[0]
    assert (right > 0.05).sum() == 1
    assert (wrong > 0.05).sum() > 3


# --- clear traces ---------------------------------------------------------------------------------


def test_clear_traces_zero_outside_transients_and_scale_events_to_the_dff_maximum():
    dff = np.array([[0.1, 2.0, 4.0, 1.0, 0.2, 0.3]])
    events = np.array([[0.5, 0.0, 1.0, 2.0, 3.0, 0.0]])
    mask = np.array([[False, True, True, True, False, False]])
    dff_clear, e_clear = clear_traces(dff, events, mask)
    np.testing.assert_allclose(dff_clear, [[0, 2.0, 4.0, 1.0, 0, 0]])
    # outside the transient: zero; inside: scaled so the maximum (2.0 raw) becomes 4.0
    np.testing.assert_allclose(e_clear, [[0, 0, 2.0, 4.0, 0, 0]])
    assert e_clear.max() == dff_clear.max()


def test_a_cell_with_no_events_in_its_transients_keeps_zeros():
    dff = np.array([[0.0, 3.0, 0.0]])
    mask = np.array([[False, True, False]])
    dff_clear, e_clear = clear_traces(dff, np.zeros((1, 3)), mask)
    assert (e_clear == 0).all() and dff_clear.max() == 3.0


def test_a_cell_without_transients_is_all_zero():
    dff_clear, e_clear = clear_traces(np.ones((1, 4)), np.ones((1, 4)), np.zeros((1, 4), bool))
    assert (dff_clear == 0).all() and (e_clear == 0).all()


# --- whole pipeline on synthetic cells ------------------------------------------------------------


def test_pipeline_gives_one_row_per_cell_and_finds_the_planted_events():
    rng = np.random.default_rng(8)
    fcorr, cells = planted_cells(rng, n_cells=3)
    fneu = np.zeros_like(fcorr)
    result = analyse_traces(fcorr, fneu, PERIOD_S, Zong2022Settings(tau_s=1.0))
    assert result.dff.shape == fcorr.shape
    assert result.f0_ok.all()
    assert (result.snr > 3).all()
    assert result.passes.all()
    assert (result.n_transients == len(cells[0].spike_frames)).all()
    for i, cell in enumerate(cells):
        # each planted transient holds a calcium event at (or just after) its onset
        for frame in cell.spike_frames:
            assert (result.e_clear[i, frame : frame + 4] > 0).any()
        assert result.n_events[i] >= len(cell.spike_frames)
        assert result.e_clear[i].max() == pytest.approx(result.dff_clear[i].max())
        outside = ~result.transient_mask[i]
        assert (result.dff_clear[i, outside] == 0).all()
        assert (result.e_clear[i, outside] == 0).all()


def test_neuropil_is_subtracted_before_everything_else():
    rng = np.random.default_rng(9)
    fcorr, _ = planted_cells(rng, n_cells=1)
    fneu = np.full_like(fcorr, 30.0)
    result = analyse_traces(fcorr + 0.7 * fneu, fneu, PERIOD_S, Zong2022Settings())
    reference = analyse_traces(fcorr, np.zeros_like(fcorr), PERIOD_S, Zong2022Settings())
    np.testing.assert_allclose(result.f0, reference.f0, rtol=1e-4)


def test_a_noise_only_cell_does_not_pass_the_snr_threshold():
    rng = np.random.default_rng(10)
    noise = 100 + rng.normal(0, 1.0, (1, N_FRAMES))
    result = analyse_traces(noise, np.zeros_like(noise), PERIOD_S, Zong2022Settings())
    assert not result.passes[0]


def test_min_events_is_applied_on_top_of_snr_when_set():
    rng = np.random.default_rng(11)
    fcorr, cells = planted_cells(rng, n_cells=1)
    fneu = np.zeros_like(fcorr)
    loose = analyse_traces(fcorr, fneu, PERIOD_S, Zong2022Settings())
    strict = analyse_traces(fcorr, fneu, PERIOD_S, Zong2022Settings(min_events=10_000))
    assert loose.passes[0] and not strict.passes[0]
    assert strict.n_events[0] == loose.n_events[0]


def test_cells_whose_f0_is_not_positive_are_excluded_and_listed_not_dropped():
    rng = np.random.default_rng(12)
    fcorr, _ = planted_cells(rng, n_cells=2)
    fcorr[1] -= 1000.0  # F0 negative throughout
    result = analyse_traces(fcorr, np.zeros_like(fcorr), PERIOD_S, Zong2022Settings())
    assert result.f0_ok.tolist() == [True, False]
    assert np.isnan(result.dff[1]).all() and np.isnan(result.dff_clear[1]).all()
    assert not result.passes[1] and np.isnan(result.snr[1])
    assert result.passes[0]  # the other cell is unaffected
    assert result.dff.shape[0] == 2  # rows keep the ROI numbering


def test_pipeline_refuses_traces_with_nan():
    bad = np.full((1, 400), 100.0)
    bad[0, 5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        analyse_traces(bad, np.zeros_like(bad), PERIOD_S, Zong2022Settings())


def test_pipeline_refuses_a_recording_shorter_than_the_baseline_window():
    short = np.full((1, 100), 100.0)
    with pytest.raises(ValueError, match="window"):
        analyse_traces(short, np.zeros_like(short), PERIOD_S, Zong2022Settings())


# --- tau sensitivity ------------------------------------------------------------------------------


def test_event_counts_by_tau_returns_a_count_per_cell_for_each_tau():
    rng = np.random.default_rng(13)
    fcorr, _ = planted_cells(rng, n_cells=2)
    fneu = np.zeros_like(fcorr)
    result = analyse_traces(fcorr, fneu, PERIOD_S, Zong2022Settings())
    counts = event_counts_by_tau(result.delta_f, result.transient_mask, FS_HZ, (0.45, 0.7, 1.0))
    assert set(counts) == {0.45, 0.7, 1.0}
    for per_cell in counts.values():
        assert per_cell.shape == (2,)
    # a faster assumed decay explains the same trace with more, smaller events
    assert counts[0.45].sum() >= counts[1.0].sum()
    # and the count at the pipeline's own tau equals the pipeline's n_events
    again = event_counts_by_tau(result.delta_f, result.transient_mask, FS_HZ, (0.7,))
    np.testing.assert_array_equal(again[0.7], result.n_events)
