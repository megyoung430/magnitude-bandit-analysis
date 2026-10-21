"""Controls for the shared-signal analysis: neuropil as a second control, reliabilities, and the
attenuation-corrected correlation with the other cells."""

import numpy as np
import pytest

from src.funcimg_analysis.run_statistics import shared_signal_stats
from src.funcimg_analysis.shared_signal import (
    background_reliability,
    leave_one_out_mean,
    leave_one_out_reliability,
    partial_row_correlation,
    spearman_brown,
)
from src.funcimg_analysis.shared_signal_null import shift_offsets
from tests.test_funcimg_shared_signal import N_FRAMES, make_cells, smooth_noise


def residual_partial(a, b, controls):
    """Partial correlation by explicit least squares, one row at a time (the reference)."""
    design = np.column_stack([np.ones(len(a)), *controls])
    resid = [x - design @ np.linalg.lstsq(design, x, rcond=None)[0] for x in (a, b)]
    return np.corrcoef(*resid)[0, 1]


# --- several controls ---------------------------------------------------------------------------


def test_partial_correlation_with_several_shared_controls_equals_regression_residuals():
    rng = np.random.default_rng(0)
    cells, field = make_cells(rng, n_cells=4, common_weight=0.5)
    extra = smooth_noise(rng)
    loo = leave_one_out_mean(cells)
    got = partial_row_correlation(cells, loo, [field, extra])
    want = [residual_partial(c, o, [field, extra]) for c, o in zip(cells, loo, strict=True)]
    np.testing.assert_allclose(got, want, atol=1e-10)


def test_partial_correlation_accepts_a_control_per_cell():
    rng = np.random.default_rng(1)
    cells, field = make_cells(rng, n_cells=4, common_weight=0.5)
    neuropil = np.array([field + 0.5 * smooth_noise(rng) for _ in range(4)])
    loo = leave_one_out_mean(cells)
    got = partial_row_correlation(cells, loo, [field, neuropil])
    want = [residual_partial(cells[i], loo[i], [field, neuropil[i]]) for i in range(4)]
    np.testing.assert_allclose(got, want, atol=1e-10)


def test_several_controls_use_only_frames_finite_in_every_trace():
    rng = np.random.default_rng(2)
    cells, field = make_cells(rng, n_cells=3, common_weight=0.5)
    extra = smooth_noise(rng)
    extra[100:160] = np.nan
    loo = leave_one_out_mean(cells)
    ok = np.isfinite(extra)
    got = partial_row_correlation(cells, loo, [field, extra])
    want = [
        residual_partial(c[ok], o[ok], [field[ok], extra[ok]])
        for c, o in zip(cells, loo, strict=True)
    ]
    np.testing.assert_allclose(got, want, atol=1e-10)


def test_a_cell_that_is_exactly_a_control_has_an_undefined_partial_r():
    rng = np.random.default_rng(3)
    cells, field = make_cells(rng, n_cells=3)
    cells[0] = 2 * field + 1
    assert np.isnan(partial_row_correlation(cells, leave_one_out_mean(cells), field)[0])


# --- a noisy background leaves field signal that the cells' neuropil can remove -----------------


def noisy_background_scenario(rng, n_cells=30, background_noise=0.5):
    """Field-only cells; the background is a noisy copy of the field, the neuropil carries it."""
    field = smooth_noise(rng)
    background = field + background_noise * smooth_noise(rng)
    cells = np.array([field + smooth_noise(rng) for _ in range(n_cells)])
    neuropil = np.array([field + 0.3 * smooth_noise(rng) for _ in range(n_cells)])
    return cells, background, neuropil


def test_a_noisy_background_gives_a_positive_partial_r_for_field_only_cells():
    rng = np.random.default_rng(4)
    cells, background, _ = noisy_background_scenario(rng)
    partial = partial_row_correlation(cells, leave_one_out_mean(cells), background)
    assert np.median(partial) > 0.1  # shared neural signal is absent, yet partial r is positive


def test_controlling_the_neuropil_too_removes_most_of_that_residual_field_signal():
    rng = np.random.default_rng(4)
    cells, background, neuropil = noisy_background_scenario(rng)
    loo = leave_one_out_mean(cells)
    background_only = partial_row_correlation(cells, loo, background)
    with_neuropil = partial_row_correlation(
        cells, loo, [background, neuropil, leave_one_out_mean(neuropil)]
    )
    assert np.median(with_neuropil) < 0.5 * np.median(background_only)
    assert abs(np.median(with_neuropil)) < 0.06


def test_the_neuropil_control_does_not_remove_a_signal_the_cells_really_share():
    rng = np.random.default_rng(5)
    cells, background, neuropil = noisy_background_scenario(rng)
    common = smooth_noise(rng)
    cells = cells + 1.5 * common  # neuropil does not carry it
    loo = leave_one_out_mean(cells)
    partial = partial_row_correlation(
        cells, loo, [background, neuropil, leave_one_out_mean(neuropil)]
    )
    assert np.median(partial) > 0.3


# --- reliabilities ------------------------------------------------------------------------------


def test_spearman_brown_steps_a_half_reliability_up_and_is_nan_without_one():
    assert spearman_brown(np.array([0.5]))[0] == pytest.approx(2 / 3)
    assert spearman_brown(np.array([1.0]))[0] == 1.0
    assert np.isnan(spearman_brown(np.array([0.0, -0.2]))).all()


def test_background_reliability_is_the_split_half_correlation_stepped_up():
    rng = np.random.default_rng(6)
    field = smooth_noise(rng)
    halves = np.array([field + 0.5 * smooth_noise(rng), field + 0.5 * smooth_noise(rng)])
    r_halves, reliability = background_reliability(halves[0], halves[1])
    assert r_halves == pytest.approx(0.8, abs=0.08)
    assert reliability == pytest.approx(2 * r_halves / (1 + r_halves))


def test_background_reliability_raises_on_a_length_mismatch():
    with pytest.raises(ValueError, match="frames"):
        background_reliability(np.zeros(10), np.zeros(9))


@pytest.mark.parametrize("n_cells", [3, 12, 40])
def test_leave_one_out_reliability_estimates_the_signal_fraction_of_the_average(n_cells):
    rng = np.random.default_rng(7)
    field = smooth_noise(rng, 6000)
    cells = np.array([0.5 * field + smooth_noise(rng, 6000) for _ in range(n_cells)])
    others = n_cells - 1
    true = 0.25 / (0.25 + 1 / others)
    got = leave_one_out_reliability(cells, np.random.default_rng(0), n_splits=10)
    assert got.shape == (n_cells,)
    assert np.median(got) == pytest.approx(true, abs=0.12)
    assert np.all(got[np.isfinite(got)] <= 1.0)


def test_leave_one_out_reliability_needs_three_cells_and_is_seeded():
    z = np.random.default_rng(8).normal(size=(2, 500))
    with pytest.raises(ValueError, match="three cells"):
        leave_one_out_reliability(z, np.random.default_rng(0))
    cells = np.random.default_rng(9).normal(size=(6, 500)) + smooth_noise(
        np.random.default_rng(9), 500
    )
    first = leave_one_out_reliability(cells, np.random.default_rng(3))
    np.testing.assert_array_equal(first, leave_one_out_reliability(cells, np.random.default_rng(3)))


# --- the attenuation-corrected correlation with the other cells ---------------------------------


@pytest.mark.parametrize("n_cells", [3, 5, 12, 50])
def test_field_only_cells_lie_on_the_identity_line_once_the_loo_correlation_is_corrected(n_cells):
    rng = np.random.default_rng(10)
    field = smooth_noise(rng, 6000)
    cells = np.array([0.5 * field + smooth_noise(rng, 6000) for _ in range(n_cells)])
    offsets = shift_offsets(6000, 450, 5, np.random.default_rng(1))
    stats = shared_signal_stats(cells, field, offsets, rng=np.random.default_rng(2), null_kinds=())
    corrected = np.median(stats["r_loo_corrected"] - stats["r_bg"])
    raw = np.median(stats["r_loo"] - stats["r_bg"])
    assert abs(corrected) < 0.1
    if n_cells == 3:
        assert raw < -0.1  # the uncorrected comparison is biased against the cells


def test_cells_with_an_extra_common_signal_stay_above_the_identity_line_after_correction():
    rng = np.random.default_rng(11)
    cells, field = make_cells(rng, n_cells=12, common_weight=1.5)
    offsets = shift_offsets(N_FRAMES, 450, 5, np.random.default_rng(1))
    stats = shared_signal_stats(cells, field, offsets, rng=np.random.default_rng(2), null_kinds=())
    assert np.median(stats["r_loo_corrected"] - stats["r_bg"]) > 0.2


def test_the_corrected_correlation_is_capped_at_one():
    rng = np.random.default_rng(12)
    field = smooth_noise(rng)
    cells = np.array([field + 0.05 * smooth_noise(rng) for _ in range(6)])
    offsets = shift_offsets(N_FRAMES, 450, 5, np.random.default_rng(1))
    stats = shared_signal_stats(cells, field, offsets, rng=np.random.default_rng(2), null_kinds=())
    assert np.nanmax(stats["r_loo_corrected"]) <= 1.0


# --- statistics with the neuropil control and the background's reliability ----------------------


def test_stats_report_the_neuropil_controlled_partial_r_and_the_background_reliability():
    rng = np.random.default_rng(13)
    cells, background, neuropil = noisy_background_scenario(rng, n_cells=15)
    halves = np.array([background + 0.2 * smooth_noise(rng), background + 0.2 * smooth_noise(rng)])
    offsets = shift_offsets(N_FRAMES, 450, 10, np.random.default_rng(1))
    stats = shared_signal_stats(
        cells,
        background,
        offsets,
        neuropil_z=neuropil,
        background_halves_z=halves,
        rng=np.random.default_rng(2),
        null_kinds=("partial_average", "partial_neuropil_average"),
    )
    assert np.median(stats["partial_r_neuropil"]) < 0.5 * np.median(stats["partial_r"])
    assert 0.5 < stats["bg_reliability"] <= 1.0 and 0.3 < stats["bg_half_r"] < 1.0
    assert stats["null_partial_neuropil_average"].shape == (10, 15)
    assert set(stats) >= {"null_partial_average", "null_partial_neuropil_average"}
    assert not {"null_r_bg", "null_r_loo", "null_partial_background"} & set(stats)


def test_without_a_neuropil_or_halves_those_statistics_are_nan_not_missing():
    rng = np.random.default_rng(14)
    cells, field = make_cells(rng, n_cells=6)
    offsets = shift_offsets(N_FRAMES, 450, 3, np.random.default_rng(1))
    stats = shared_signal_stats(cells, field, offsets, rng=np.random.default_rng(2), null_kinds=())
    assert np.isnan(stats["partial_r_neuropil"]).all() and np.isnan(stats["bg_reliability"])


def test_an_unknown_null_kind_raises():
    rng = np.random.default_rng(15)
    cells, field = make_cells(rng, n_cells=4)
    with pytest.raises(ValueError, match="null_kinds"):
        shared_signal_stats(
            cells, field, np.array([500]), rng=np.random.default_rng(0), null_kinds=("r_bgg",)
        )
