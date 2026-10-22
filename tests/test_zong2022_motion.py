"""Motion and z-drift quality control after Zong et al. 2022 / Suite2p's registration metrics."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift as shift_image

pytest.importorskip("mini2p_preprocessing")  # the optional funcimg extra (suite2p, torch)

from src.funcimg_analysis.zong2022_motion import (  # noqa: E402
    nonrigid_motion,
    pc_images,
    registration_metrics,
    registration_settings,
    sample_frame_indices,
)

L, N = 128, 300


def base_image(seed=0):
    rng = np.random.default_rng(seed)
    image = gaussian_filter(rng.random((L, L)), 2.5)
    return 50 + 100 * (image - image.min()) / (image.max() - image.min())


def movie_with(state, kind):
    """Frames switching between two states: shifted by 3 px in x, or just 20 % brighter."""
    rng = np.random.default_rng(1)
    base = base_image()
    shifted = shift_image(base, (0, 3), order=1, mode="nearest")
    frames = [(shifted if kind == "shift" else base * 1.2) if s else base for s in state]
    return (np.array(frames) + rng.normal(0, 2, (len(state), L, L))).astype(np.float32)


@pytest.fixture(scope="module")
def state():
    return np.random.default_rng(2).integers(0, 2, N)


# --- sampling and settings ------------------------------------------------------------------------


def test_sample_indices_are_evenly_spaced_through_the_whole_recording():
    indices = sample_frame_indices(30_000, 5000)
    assert len(indices) == 5000 and indices[0] == 0 and indices[-1] == 29_999
    assert np.all(np.diff(indices) > 0)
    assert np.ptp(np.diff(indices)) <= 1  # equally sampled


def test_a_short_recording_is_used_whole():
    np.testing.assert_array_equal(sample_frame_indices(120, 5000), np.arange(120))


def test_registration_settings_start_from_suite2p_defaults_and_take_ops_overrides():
    plain = registration_settings({})
    assert plain["block_size"] == (128, 128) and plain["smooth_sigma"] == 1.15
    changed = registration_settings({"registration": {"smooth_sigma": 2.0, "unknown_key": 1}})
    assert changed["smooth_sigma"] == 2.0 and "unknown_key" not in changed


# --- principal components and residual drift ------------------------------------------------------


def test_top_and_bottom_images_average_the_extreme_frames_of_each_component(state):
    movie = movie_with(state, "shift")
    pcs = pc_images(movie, n_pcs=3, n_extremes=40)
    assert pcs["i_top"].shape == (3, L, L) and pcs["i_bottom"].shape == (3, L, L)
    assert pcs["pc_time"].shape == (N, 3)
    order = np.argsort(pcs["pc_time"][:, 0])
    np.testing.assert_allclose(pcs["i_top"][0], movie[order[-40:]].mean(axis=0), atol=1e-3)
    np.testing.assert_allclose(pcs["i_bottom"][0], movie[order[:40]].mean(axis=0), atol=1e-3)
    assert pcs["n_extremes"] == 40


def test_the_number_of_extreme_frames_shrinks_for_short_movies(state):
    pcs = pc_images(movie_with(state, "shift"), n_pcs=2, n_extremes=500)
    assert pcs["n_extremes"] == N // 2


def test_an_injected_shift_is_found_as_residual_drift(state):
    metrics = registration_metrics(movie_with(state, "shift"), n_pcs=5, n_extremes=60)
    drift = metrics["drift"]
    assert list(drift["pc"]) == [1, 2, 3, 4, 5]
    assert drift["rigid_px"].iloc[0] == pytest.approx(3.0, abs=0.5)
    assert drift["rigid_px"].iloc[1:].max() < 1.0


def test_brightness_changes_without_motion_give_no_residual_drift(state):
    drift = registration_metrics(movie_with(state, "brightness"), n_pcs=5, n_extremes=60)["drift"]
    assert drift["rigid_px"].max() < 0.5
    assert drift["max_nonrigid_px"].max() < 0.5


def test_pc_images_refuse_more_components_than_frames():
    with pytest.raises(ValueError, match="components"):
        pc_images(np.zeros((4, 16, 16), np.float32), n_pcs=10, n_extremes=2)


# --- shifts recorded by the registration ----------------------------------------------------------


def test_nonrigid_motion_is_the_largest_distance_between_any_two_blocks_in_a_frame():
    xoff1 = np.array([[0.0, 0.0, 0.0], [0.0, 3.0, 0.0], [1.0, 1.0, 1.0]])
    yoff1 = np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0], [2.0, 2.0, 2.0]])
    np.testing.assert_allclose(nonrigid_motion(xoff1, yoff1), [0.0, 5.0, 0.0])


def test_nonrigid_motion_needs_matching_shapes():
    with pytest.raises(ValueError, match="shape"):
        nonrigid_motion(np.zeros((4, 3)), np.zeros((4, 2)))
