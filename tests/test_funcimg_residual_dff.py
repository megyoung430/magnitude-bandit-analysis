"""Background-corrected ΔF/F0: the background's fluctuation goes, the cell's offset stays."""

import numpy as np
import pytest

pytest.importorskip("mini2p_preprocessing")  # the optional funcimg extra

from mini2p_preprocessing.config import FluorescenceSettings  # noqa: E402
from mini2p_preprocessing.fluorescence import compute_dff  # noqa: E402

from src.funcimg_analysis.residual_dff import (  # noqa: E402
    flag_cells,
    residual_dff_channel,
    subtract_background_fluctuation,
)
from src.funcimg_analysis.shared_signal import row_correlation  # noqa: E402

N_FRAMES, FS = 1200, 7.5
SETTINGS = FluorescenceSettings(baseline="prctile", prctile_baseline=8.0)


def smooth(rng, n=N_FRAMES, width=9):
    out = np.convolve(rng.normal(size=n + width), np.ones(width), mode="valid")[:n]
    return out / out.std()


def background_and_cells(rng, slopes=(3.0, 1.0, 0.0), offset=2000.0):
    background = 1000 + 30 * smooth(rng)
    private = [10 * smooth(rng) for _ in slopes]
    cells = np.array(
        [
            offset + s * (background - background.mean()) + p
            for s, p in zip(slopes, private, strict=True)
        ]
    )
    return background, cells


def test_the_backgrounds_fluctuation_is_removed_and_the_cells_offset_is_kept():
    background, cells = background_and_cells(np.random.default_rng(0))
    corrected, slope = subtract_background_fluctuation(cells, background, np.ones(N_FRAMES, bool))
    np.testing.assert_allclose(slope, [3.0, 1.0, 0.0], atol=0.15)
    np.testing.assert_allclose(corrected.mean(axis=1), cells.mean(axis=1), rtol=1e-9)
    assert np.abs(row_correlation(corrected, background)).max() < 1e-6  # OLS leaves none


def test_the_slope_is_fitted_only_on_the_frames_in_the_fit_mask():
    rng = np.random.default_rng(1)
    background, cells = background_and_cells(rng, slopes=(2.0,))
    event_frames = np.zeros(N_FRAMES, bool)
    event_frames[100:140] = event_frames[600:640] = True
    cells[0, event_frames] += 400.0  # a response the background does not share
    _, masked = subtract_background_fluctuation(cells, background, ~event_frames)
    _, unmasked = subtract_background_fluctuation(cells, background, np.ones(N_FRAMES, bool))
    assert masked[0] == pytest.approx(2.0, abs=0.1)
    assert abs(unmasked[0] - 2.0) > 2 * abs(masked[0] - 2.0)


def test_a_constant_background_cannot_be_fitted():
    cells = np.random.default_rng(2).normal(100, 1, (2, N_FRAMES))
    with pytest.raises(ValueError, match="background is constant"):
        subtract_background_fluctuation(cells, np.full(N_FRAMES, 5.0), np.ones(N_FRAMES, bool))


def test_dff_of_the_corrected_trace_matches_compute_dff_of_the_same_trace():
    rng = np.random.default_rng(3)
    background, cells = background_and_cells(rng)
    mask = np.ones(N_FRAMES, bool)
    result = residual_dff_channel(cells, background, mask, FS, SETTINGS, min_f0_fraction=0.1)
    corrected, _ = subtract_background_fluctuation(cells, background, mask)
    # zero neuropil: the correction replaces the neuropil subtraction, it is not added to it
    expected = compute_dff(corrected, np.zeros_like(corrected), FS, SETTINGS).dff
    np.testing.assert_allclose(result["dff"], expected, rtol=1e-5, atol=1e-6)
    assert result["kept"].all() and result["excluded"] == {}


def test_a_known_trace_gives_the_prctile_baseline_dff():
    rng = np.random.default_rng(4)
    background = 1000 + 30 * smooth(rng)
    f = 2000.0 + 5 * smooth(np.random.default_rng(5))[None, :]
    result = residual_dff_channel(
        f, background, np.ones(N_FRAMES, bool), FS, SETTINGS, min_f0_fraction=0.1
    )
    corrected, _ = subtract_background_fluctuation(f, background, np.ones(N_FRAMES, bool))
    f0 = np.percentile(corrected, 8.0, axis=1, keepdims=True)
    np.testing.assert_allclose(result["dff"], (corrected - f0) / f0, rtol=1e-4, atol=1e-6)


def test_cells_with_a_tiny_or_negative_baseline_are_flagged_with_their_reason():
    rng = np.random.default_rng(6)
    background = 1000 + 30 * smooth(rng)
    normal = 2000 + 10 * smooth(rng)
    dim = 1000.0 + 0 * normal
    dim[: N_FRAMES // 5] = 50.0  # a fifth of the run is near dark: F0 (8th percentile) is tiny
    negative = 0.5 + 2.0 * smooth(rng)  # F0 below zero
    cells = np.array([normal, dim, negative])
    result = residual_dff_channel(
        cells, background, np.ones(N_FRAMES, bool), FS, SETTINGS, min_f0_fraction=0.1
    )
    assert result["kept"].tolist() == [True, False, False]
    assert "below 0.1 x median F" in result["excluded"][1]
    assert "F0 <= 0" in result["excluded"][2]
    assert np.isnan(result["dff"][[1, 2]]).all()  # an excluded cell never reaches a plot


def test_a_cell_without_variance_has_no_slope_and_is_excluded():
    background, cells = background_and_cells(np.random.default_rng(7))
    cells[1] = 2000.0
    result = residual_dff_channel(
        cells, background, np.ones(N_FRAMES, bool), FS, SETTINGS, min_f0_fraction=0.1
    )
    assert "no slope" in result["excluded"][1] and result["kept"].tolist() == [True, False, True]


def test_flag_cells_lists_every_failed_rule_of_a_cell():
    f = np.full((1, 10), 100.0)
    f0 = np.array([[-1.0] * 3 + [5.0] * 2 + [50.0] * 5])  # median F is 100, so 10 is the floor
    reasons = flag_cells(f, f0, np.array([1.0]), min_f0_fraction=0.1)
    assert reasons == {0: "F0 <= 0 in 3 frames; F0 below 0.1 x median F in 5 frames"}
