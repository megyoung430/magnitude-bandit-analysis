"""Flash check: peak images and peri-event traces of the most skewed cells, and a movie clip."""

import numpy as np
import pytest

from src.funcimg_analysis.flash_check import (
    binned_clip,
    clip_dff,
    crop_box,
    event_dff_image,
    flash_check,
    offset_frames,
    peak_frames,
    peri_event,
    roi_mask,
)

PERIOD = 0.1
SHAPE = (60, 80)
N_FRAMES = 1000
CELL = {"ypix": np.repeat(np.arange(28, 33), 5), "xpix": np.tile(np.arange(38, 43), 5)}
SETTINGS = {
    "n_cells": 1,
    "n_peaks": 3,
    "min_peak_z": 3.0,
    "min_peak_gap_s": 5.0,
    "baseline_s": (-2.0, -0.5),
    "response_s": (0.0, 1.0),
    "crop_px": 10,
    "smooth_px": 0.0,
}


def _movie_with_a_flashing_cell(peaks, amplitude=0.5, red=False):
    """A flat field of 100 with the cell rising by *amplitude* (fraction) for 1 s at each peak."""
    movie = np.full((N_FRAMES, *SHAPE), 100.0)
    if not red:
        for peak in peaks:
            movie[peak : peak + 10, CELL["ypix"], CELL["xpix"]] = 100.0 * (1 + amplitude)
    return movie


# --- peaks ---------------------------------------------------------------------------------------


def test_offset_frames_are_the_frames_starting_in_the_range():
    assert offset_frames((-2.0, -0.5), 0.5).tolist() == [-4, -3, -2]
    assert offset_frames((0.0, 1.0), 0.5).tolist() == [0, 1]


def test_peaks_are_the_largest_separate_events_with_the_whole_window_inside():
    z = np.zeros(100)
    z[[5, 30, 33, 60, 97]] = [50.0, 9.0, 8.0, 7.0, 40.0]  # 5 and 97 are too close to the ends
    peaks = peak_frames(
        z, n_peaks=3, min_gap_frames=10, before_frames=10, after_frames=10, min_z=3.0
    )
    assert peaks.tolist() == [30, 60]


def test_peaks_skip_events_whose_window_reaches_left_out_frames():
    z = np.zeros(100)
    z[[30, 60]] = [9.0, 5.0]
    z[25] = np.nan  # in the baseline of the event at 30
    assert peak_frames(z, 2, 5, 10, 10, min_z=3.0).tolist() == [60]


def test_a_trace_without_an_excursion_has_no_peaks_rather_than_its_largest_noise():
    z = np.random.default_rng(0).normal(size=500) * 0.5
    assert peak_frames(z, 3, 5, 5, 5, min_z=3.0).size == 0


def test_peaks_of_a_trace_with_nothing_finite_are_empty():
    assert peak_frames(np.full(50, np.nan), 3, 5, 5, 5).size == 0


# --- crops and images ----------------------------------------------------------------------------


def test_crop_box_centres_on_the_roi_and_stays_inside_the_image():
    rows, cols = crop_box(CELL["ypix"], CELL["xpix"], 10, SHAPE)
    assert (rows, cols) == (slice(20, 41), slice(30, 51))
    rows, cols = crop_box(np.array([1]), np.array([78]), 10, SHAPE)
    assert (rows, cols) == (slice(0, 12), slice(68, 80))


def test_roi_mask_marks_the_roi_inside_the_crop():
    box = crop_box(CELL["ypix"], CELL["xpix"], 10, SHAPE)
    mask = roi_mask(CELL["ypix"], CELL["xpix"], box)
    assert mask.shape == (21, 21) and mask.sum() == 25 and mask[8:13, 8:13].all()


def test_event_image_shows_the_cell_rising_and_the_rest_flat():
    peaks = [200, 500]
    movie = _movie_with_a_flashing_cell(peaks, amplitude=0.5)
    box = crop_box(CELL["ypix"], CELL["xpix"], 10, SHAPE)
    image = event_dff_image(movie, np.array(peaks), np.arange(-20, -5), np.arange(0, 10), box)
    mask = roi_mask(CELL["ypix"], CELL["xpix"], box)
    assert image[mask] == pytest.approx(0.5)
    assert image[~mask] == pytest.approx(0.0)


def test_event_image_is_nan_where_the_baseline_is_not_positive():
    movie = np.zeros((100, 4, 4))
    image = event_dff_image(
        movie, np.array([50]), np.arange(-5, 0), np.arange(0, 5), (slice(0, 4), slice(0, 4))
    )
    assert np.isnan(image).all()


def test_peri_event_traces_are_one_row_per_peak():
    z = np.arange(100.0)
    traces = peri_event(z, np.array([10, 50]), np.arange(-2, 3))
    assert traces.tolist() == [[8, 9, 10, 11, 12], [48, 49, 50, 51, 52]]


# --- the clip ------------------------------------------------------------------------------------


def test_binned_clip_averages_frames_and_pixels():
    movie = np.arange(8 * 4 * 4, dtype=float).reshape(8, 4, 4)
    clip = binned_clip(
        movie, start=2, n_frames=4, bin_frames=2, box=(slice(0, 4), slice(0, 4)), bin_px=2
    )
    assert clip.shape == (2, 2, 2)
    assert clip[0, 0, 0] == pytest.approx(movie[2:4, 0:2, 0:2].mean())


def test_binned_clip_past_the_end_fails_loudly():
    with pytest.raises(ValueError, match="frames"):
        binned_clip(np.zeros((10, 4, 4)), 8, 4, 1, (slice(0, 4), slice(0, 4)))


def test_clip_dff_is_each_pixel_over_its_mean():
    clip = np.stack([np.full((2, 2), 100.0), np.full((2, 2), 300.0)])
    assert clip_dff(clip)[:, 0, 0] == pytest.approx([-0.5, 0.5])


# --- the whole check -----------------------------------------------------------------------------


def test_the_flashing_cell_shows_in_green_and_not_in_red():
    peaks = [200, 500, 800]
    green = _movie_with_a_flashing_cell(peaks)
    red = _movie_with_a_flashing_cell(peaks, red=True)
    z_green = np.zeros((2, N_FRAMES))
    z_green[0, peaks] = 10.0
    z = {"green": z_green, "red": np.zeros((2, N_FRAMES))}
    stat = [CELL, {"ypix": np.array([5]), "xpix": np.array([5])}]
    [cell] = flash_check(
        {"green": green, "red": red}, stat, z, np.array([3.0, 0.1]), PERIOD, **SETTINGS
    )
    assert cell["roi"] == 0 and cell["peaks"].tolist() == peaks
    assert cell["image"]["green"][cell["mask"]].mean() == pytest.approx(0.5)
    assert np.nanmax(np.abs(cell["image"]["red"])) == pytest.approx(0.0)
    assert cell["peri"]["green"].shape == (3, len(cell["offsets"]))
    assert cell["mean_image"].shape == cell["mask"].shape


def test_a_cell_without_a_usable_event_is_reported_with_no_peaks():
    z = {"green": np.full((1, N_FRAMES), np.nan), "red": np.full((1, N_FRAMES), np.nan)}
    movie = np.full((N_FRAMES, *SHAPE), 100.0)
    [cell] = flash_check(
        {"green": movie, "red": movie}, [CELL], z, np.array([1.0]), PERIOD, **SETTINGS
    )
    assert cell["peaks"].size == 0 and np.isnan(cell["image"]["green"]).all()
