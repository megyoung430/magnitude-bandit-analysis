"""Motion check: find the early artefact, leave it out, and redo the transient check."""

import numpy as np
import pytest

from src.funcimg_analysis.frame_times import RunMismatchError
from src.funcimg_analysis.motion_check import (
    MotionUnavailableError,
    kept_frames,
    masked,
    motion_check,
    motion_correlations,
    ops_trace,
    rigid_shift,
    shared_z,
    suggest_artefact_end,
)

PERIOD = 0.133
N_FRAMES = 6000  # about 13 min
SETTINGS = {
    "detrend_window_s": 30.0,
    "z_threshold": 3.0,
    "min_gap_s": 1.0,
    "red_artefact_z": 3.0,
}


def _times(n=N_FRAMES):
    return np.arange(n) * PERIOD


def _noise(rng, n_cells, n=N_FRAMES):
    return 1.0 + 0.05 * rng.normal(size=(n_cells, n))


def _artefact_run(rng, n_cells=30, artefact_end_s=200.0, n_planted=15):
    """Both channels share motion-locked jumps until *artefact_end_s*; green also has transients.

    Returns:
        tuple: ``ops`` (rigid offsets that jump with the artefact) and ``dff`` per channel.
    """
    n = N_FRAMES
    t = _times()
    jumps = np.zeros(n)
    for start in rng.choice(np.flatnonzero(t < artefact_end_s - 5), 25, replace=False):
        jumps[start : start + 8] = 1.0
    xoff = 3 * jumps + 0.2 * rng.normal(size=n)
    yoff = -2 * jumps + 0.2 * rng.normal(size=n)
    red = _noise(rng, n_cells) + 1.0 * jumps
    green = _noise(rng, n_cells) + 1.0 * jumps
    kernel = 0.6 * np.exp(-np.arange(60) / 12)
    for cell in range(n_planted):
        for start in rng.choice(np.flatnonzero(t > artefact_end_s + 5)[:-60], 15, replace=False):
            green[cell, start : start + 60] += kernel
    ops = {"xoff": xoff, "yoff": yoff, "corrXY": 1 - 0.1 * jumps, "badframes": np.zeros(n, bool)}
    return ops, {"green": green - 1, "red": red - 1}


# --- ops entries ---------------------------------------------------------------------------------


def test_ops_trace_takes_the_median_block_of_a_non_rigid_offset():
    blocks = np.array([[0.0, 1.0, 5.0], [2.0, 3.0, 9.0]])
    assert ops_trace({"xoff1": blocks}, "xoff1", 2) == pytest.approx([1.0, 3.0])


def test_ops_trace_of_the_wrong_length_fails_loudly():
    with pytest.raises(RunMismatchError, match="xoff"):
        ops_trace({"xoff": np.zeros(9)}, "xoff", 10)


def test_ops_trace_without_the_entry_fails_loudly():
    with pytest.raises(MotionUnavailableError, match="yoff"):
        ops_trace({"xoff": np.zeros(10)}, "yoff", 10)


def test_rigid_shift_is_the_distance_from_the_median_position():
    xoff, yoff = np.array([1.0, 1.0, 1.0, 4.0]), np.array([2.0, 2.0, 2.0, 6.0])
    assert rigid_shift(xoff, yoff) == pytest.approx([0, 0, 0, 5])


# --- the shared signal and the cut ---------------------------------------------------------------


def test_shared_z_keeps_what_all_cells_do_and_not_what_one_cell_does():
    z = np.zeros((9, 100))
    z[:, 10] = 5.0  # every cell
    z[0, 50] = 20.0  # one cell
    shared = shared_z(z)
    assert shared[10] == 5.0 and shared[50] == 0.0


def test_suggested_end_is_the_last_artefact_in_the_first_half_plus_a_margin():
    t = _times()
    median_red = np.zeros(N_FRAMES)
    median_red[(t > 100) & (t < 101)] = 6.0
    median_red[(t > 150) & (t < 151)] = -6.0  # below as well as above counts
    end = suggest_artefact_end(median_red, PERIOD, threshold=3.0, margin_s=10.0)
    assert end["end_s"] == pytest.approx(151 + 10, abs=0.5)
    assert end["n_frames_over_late"] == 0


def test_without_an_artefact_the_suggested_end_is_zero():
    end = suggest_artefact_end(np.zeros(N_FRAMES), PERIOD, threshold=3.0)
    assert end == {"end_s": 0.0, "n_frames_over_late": 0}


def test_an_artefact_late_in_the_run_does_not_move_the_cut_but_is_counted():
    t = _times()
    median_red = np.zeros(N_FRAMES)
    median_red[(t > 600) & (t < 601)] = 6.0
    end = suggest_artefact_end(median_red, PERIOD, threshold=3.0)
    assert end["end_s"] == 0.0 and end["n_frames_over_late"] > 0


def test_kept_frames_leave_out_the_cut_badframes_and_large_shifts():
    t = np.arange(10.0)
    bad = np.zeros(10, bool)
    bad[6] = True
    shift = np.zeros(10)
    shift[8] = 4.0
    assert kept_frames(t, 3.0).tolist() == [False] * 3 + [True] * 7
    keep = kept_frames(t, 3.0, badframes=bad, shift_px=shift, max_shift_px=2.0)
    assert np.flatnonzero(keep).tolist() == [3, 4, 5, 7, 9]


def test_a_maximum_shift_without_the_shift_fails_loudly():
    with pytest.raises(ValueError, match="shift"):
        kept_frames(np.arange(5.0), 0.0, max_shift_px=2.0)


def test_masked_sets_left_out_frames_to_nan_without_touching_the_input():
    dff = np.ones((2, 4))
    out = masked(dff, np.array([True, False, True, False]))
    assert np.isnan(out[:, [1, 3]]).all() and (out[:, [0, 2]] == 1).all()
    assert (dff == 1).all()


def test_motion_correlations_report_each_part_and_channel():
    rng = np.random.default_rng(0)
    motion = rng.normal(size=200)
    median = {"red": motion + 0.1 * rng.normal(size=200), "green": rng.normal(size=200)}
    parts = {"before": np.arange(200) < 100, "after": np.arange(200) >= 100}
    rows = motion_correlations(median, {"shift": motion}, parts)
    assert [(r["frames"], r["channel"]) for r in rows] == [
        ("before", "red"),
        ("before", "green"),
        ("after", "red"),
        ("after", "green"),
    ]
    assert rows[0]["r(median z, shift)"] > 0.9 and abs(rows[1]["r(median z, shift)"]) < 0.3


# --- the whole check -----------------------------------------------------------------------------


def test_the_check_finds_the_artefact_and_green_transients_appear_once_it_is_left_out():
    rng = np.random.default_rng(3)
    ops, dff = _artefact_run(rng)
    result = motion_check(dff, ops, PERIOD, **SETTINGS)
    assert 170 < result["suggested_end_s"] < 210  # last jump before 195 s, plus 10 s
    assert result["cut_s"] == result["suggested_end_s"]
    before = next(
        r for r in result["correlations"] if r["frames"] == "before cut" and r["channel"] == "red"
    )
    assert before["r(median z, shift from median)"] > 0.5
    kept = result["comparison"]["kept frames"]
    assert kept["median skew green"] > kept["median skew red"] + 0.2
    assert kept["fraction green above red 95th"] > 0.3
    assert result["z_kept"]["green"].shape == dff["green"].shape
    assert np.isnan(result["z_kept"]["green"][:, ~result["keep"]]).all()


def test_a_cut_given_by_hand_replaces_the_suggestion():
    rng = np.random.default_rng(4)
    ops, dff = _artefact_run(rng)
    result = motion_check(dff, ops, PERIOD, artefact_end_s=300.0, **SETTINGS)
    assert result["cut_s"] == 300.0 and result["suggested_end_s"] != 300.0
    assert not result["keep"][_times() < 300].any()


def test_badframes_of_the_wrong_length_fail_loudly():
    rng = np.random.default_rng(5)
    ops, dff = _artefact_run(rng)
    ops["badframes"] = np.zeros(N_FRAMES - 1, bool)
    with pytest.raises(RunMismatchError, match="badframes"):
        motion_check(dff, ops, PERIOD, **SETTINGS)
