"""Shared-signal analysis of whole runs: residual reward responses and collection from disk."""

import json

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("mini2p_preprocessing")  # the optional funcimg extra

from src.funcimg_analysis.collect import collect_shared_signal, pool_shared_signal  # noqa: E402
from src.funcimg_analysis.event_frames import pooled_mean_sem  # noqa: E402
from src.funcimg_analysis.frame_times import FRAME_TIMES_FILE, RunMismatchError  # noqa: E402
from src.funcimg_analysis.run_responses import PULSE_CATEGORIES  # noqa: E402
from src.funcimg_analysis.shared_signal import mean_response  # noqa: E402
from src.funcimg_analysis.shared_signal_run import (  # noqa: E402
    TooShortForSharedSignalError,
    analyse_shared_signal,
)
from tests.test_funcimg_background import PERIOD, write_plane  # noqa: E402

N_FRAMES = 900
N_CELLS = 4  # the plane fixture's cell count
EVENT_STEP = 20
EVENT_FRAMES = np.arange(40, N_FRAMES - 30, EVENT_STEP)
PULSES = np.arange(len(EVENT_FRAMES)) % 5
SETTINGS = {"detrend_window_s": 15.0, "min_shift_s": 10.0, "n_shifts": 20}
WINDOW_S = (-1.5, 3.0)
MEASURES = ("F z", "dF/F0 z")


def smooth(rng, n=N_FRAMES, width=9):
    raw = rng.normal(size=n + width)
    out = np.convolve(raw, np.ones(width) / np.sqrt(width), mode="valid")[:n]
    return out / out.std()


def reward_bump():
    """A response after every outcome, growing with the number of reward pulses."""
    bump = np.zeros(N_FRAMES)
    for frame, pulses in zip(EVENT_FRAMES, PULSES, strict=True):
        bump[frame : frame + 8] += pulses  # ~1 s
    return bump


def make_traces(rng, reward_in: str, n_cells=N_CELLS):
    """Background and cell traces: shared field + private noise; the reward in one place only."""
    field, bump = smooth(rng), reward_bump()
    in_bg, in_cells = reward_in == "background", reward_in == "cells"
    background = 1000 + 30 * field + 60 * bump * in_bg
    weights = rng.uniform(0.8, 1.2, n_cells)
    cells = np.array(
        [
            2000
            + 40 * w * field
            + 30 * smooth(rng)
            + 80 * w * bump * in_bg  # the cells see the same reward signal as the background
            + 80 * bump * in_cells
            for w in weights
        ]
    )
    return background, cells


def frame_times():
    return 20.0 + PERIOD * np.arange(N_FRAMES)


def outcomes():
    return pd.DataFrame(
        {
            "outcome_time": frame_times()[EVENT_FRAMES] + 0.01,
            "num_rew_pulses": PULSES,
            "phase": "task",
        }
    )


def run_views(rng, reward_in):
    background, cells = make_traces(rng, reward_in)
    views = {("green", "F"): cells, ("green", "dF/F0"): cells / 2000 - 1}
    return views, {("green", "F"): background}


def analyse(views, background, **overrides):
    settings = {**SETTINGS, "window_s": WINDOW_S, "phases": ("task",), "measures": MEASURES}
    settings.update(overrides)
    return analyse_shared_signal(
        views, background, frame_times(), outcomes(), rng=np.random.default_rng(0), **settings
    )


def response_2s(result, which, pulses=4, view=("green", "F z")):
    pooled = {view: pooled_mean_sem([result[which]["means"][view]], PULSE_CATEGORIES)}
    return mean_response(pooled, result[which]["offsets"], PERIOD)[view][pulses]


# --- the pure per-run analysis ------------------------------------------------------------------


def test_a_reward_response_that_is_only_in_the_background_vanishes_from_the_residual():
    views, background = run_views(np.random.default_rng(0), "background")
    result = analyse(views, background)
    original, residual = response_2s(result, "original"), response_2s(result, "residual")
    assert original > 0.5
    assert abs(residual) < 0.25 * original


def test_a_reward_response_that_is_only_in_the_cells_survives_background_removal():
    views, background = run_views(np.random.default_rng(1), "cells")
    result = analyse(views, background)
    original, residual = response_2s(result, "original"), response_2s(result, "residual")
    assert original > 0.5
    assert residual > 0.7 * original


def test_the_response_of_a_dff_view_is_removed_the_same_way():
    views, background = run_views(np.random.default_rng(2), "background")
    result = analyse(views, background)
    original = response_2s(result, "original", view=("green", "dF/F0 z"))
    residual = response_2s(result, "residual", view=("green", "dF/F0 z"))
    assert original > 0.5 and abs(residual) < 0.25 * original


def test_statistics_are_given_per_channel_and_raw_measure_with_the_requested_shifts():
    rng = np.random.default_rng(3)
    views, background = run_views(rng, "none")
    result = analyse(views, background)
    assert set(result["stats"]) == {("green", "F"), ("green", "dF/F0")}
    stats = result["stats"][("green", "F")]
    assert stats["null_partial_average"].shape == (SETTINGS["n_shifts"], N_CELLS)
    assert stats["r_bg_avg"] > 0.3  # the cells follow the field signal
    assert result["n_cells"] == N_CELLS


def test_only_the_requested_measures_are_analysed():
    views, background = run_views(np.random.default_rng(4), "none")
    result = analyse(views, background, measures=("F z",))
    assert set(result["stats"]) == {("green", "F")}
    assert set(result["original"]["means"]) == {("green", "F z")}


def test_the_same_seed_gives_the_same_chance_levels_and_another_seed_does_not():
    views, background = run_views(np.random.default_rng(5), "none")
    first = analyse(views, background)["stats"][("green", "F")]["null_r_bg"]
    again = analyse(views, background)["stats"][("green", "F")]["null_r_bg"]
    np.testing.assert_array_equal(first, again)
    other = analyse_shared_signal(
        views,
        background,
        frame_times(),
        outcomes(),
        rng=np.random.default_rng(99),
        window_s=WINDOW_S,
        phases=("task",),
        measures=MEASURES,
        **SETTINGS,
    )["stats"][("green", "F")]["null_r_bg"]
    assert not np.array_equal(first, other)


def test_a_run_too_short_for_the_shifts_or_the_window_is_refused():
    views, background = run_views(np.random.default_rng(6), "none")
    with pytest.raises(TooShortForSharedSignalError, match="shift"):
        analyse(views, background, min_shift_s=70.0)  # 2 x 70 s > 120 s
    with pytest.raises(TooShortForSharedSignalError, match="window"):
        analyse(views, background, detrend_window_s=500.0)


def test_a_background_of_another_length_than_the_frame_times_raises():
    views, background = run_views(np.random.default_rng(7), "none")
    background = {key: trace[:-5] for key, trace in background.items()}
    with pytest.raises(RunMismatchError, match="frames"):
        analyse(views, background)


def test_a_constant_background_raises():
    views, background = run_views(np.random.default_rng(8), "none")
    background = {key: np.full(N_FRAMES, 5.0) for key in background}
    with pytest.raises(ValueError, match="constant"):
        analyse(views, background)


def test_raw_measures_are_not_accepted_because_the_analysis_is_in_z_units():
    views, background = run_views(np.random.default_rng(9), "none")
    with pytest.raises(ValueError, match="F z"):
        analyse(views, background, measures=("F",))


def test_the_input_traces_are_not_modified():
    views, background = run_views(np.random.default_rng(10), "background")
    before = {k: v.copy() for k, v in views.items()}, {k: v.copy() for k, v in background.items()}
    analyse(views, background)
    for got, want in zip((views, background), before, strict=True):
        for key in want:
            np.testing.assert_array_equal(got[key], want[key])


# --- from disk ----------------------------------------------------------------------------------


def write_run(funcimg, run, rng, reward_in, iscell=None):
    """A run folder: planes' binaries hold the background; F.npy holds the cells."""
    plane = funcimg / run / "without_destriping" / "suite2p" / "plane0"
    green_bg, green_cells = make_traces(rng, reward_in)
    red_bg, red_cells = make_traces(rng, "none")
    write_plane(plane, rng, green_bg, red_bg)
    for name, array in {
        "F": green_cells,
        "Fneu": green_cells * 0.2,
        "F_chan2": red_cells,
        "Fneu_chan2": red_cells * 0.1,
    }.items():
        np.save(plane / f"{name}.npy", array)
    if iscell is not None:
        np.save(plane / "iscell.npy", np.column_stack([iscell, np.ones(len(iscell))]))
    time_s = frame_times()
    np.savez(
        funcimg / run / FRAME_TIMES_FILE,
        time_s=time_s,
        frame_index=np.arange(N_FRAMES),
        daq_frame_s=time_s - 20.0,
        tiff="71rtask_00001.tif",
        h5="d.h5",
        tsv=str(funcimg / run / "behaviour.tsv"),
        quality=json.dumps({"fraction_matched": 1.0}),
    )
    (funcimg / run / "without_destriping" / "source_tiff.json").write_text(
        json.dumps({"source_tiff": "/raw/ses-224/funcimg/71rtask_00001.tif"})
    )


@pytest.fixture
def derivatives(tmp_path):
    funcimg = tmp_path / "der" / "ses-224_date-20260809" / "funcimg"
    write_run(funcimg, "run-01", np.random.default_rng(0), "background")
    write_run(funcimg, "run-02", np.random.default_rng(1), "cells")
    return tmp_path / "der"


def collect_runs(derivatives, tmp_path, **overrides):
    settings = {**SETTINGS, "measures": MEASURES, "cache_dir": tmp_path / "cache"}
    settings.update(overrides)
    return collect_shared_signal(
        derivatives,
        224,
        224,
        "without_destriping",
        WINDOW_S,
        ("task",),
        lambda _: outcomes(),
        **settings,
    )


def test_each_run_is_analysed_for_green_and_red_with_original_and_residual_responses(
    derivatives, tmp_path
):
    runs, skipped = collect_runs(derivatives, tmp_path)
    assert skipped == []
    assert [r["run"] for r in runs] == ["run-01", "run-02"]
    for run in runs:
        assert set(run["stats"]) == {
            (channel, measure) for channel in ("green", "red") for measure in ("F", "dF/F0")
        }
        assert run["n_cells"] == N_CELLS and run["rois"].tolist() == list(range(N_CELLS))
    original, residual = pool_shared_signal(runs, MEASURES)
    assert (
        set(original)
        == set(residual)
        == {(channel, measure) for channel in ("green", "red") for measure in MEASURES}
    )


def test_the_residual_response_is_gone_for_the_background_run_and_kept_for_the_cell_run(
    derivatives, tmp_path
):
    runs, _ = collect_runs(derivatives, tmp_path)
    in_background, in_cells = runs
    assert abs(response_2s(in_background, "residual")) < 0.25 * response_2s(
        in_background, "original"
    )
    assert response_2s(in_cells, "residual") > 0.7 * response_2s(in_cells, "original")


def test_cells_only_keeps_the_rois_the_classifier_accepts(derivatives, tmp_path):
    plane = derivatives / "ses-224_date-20260809" / "funcimg" / "run-01"
    plane = plane / "without_destriping" / "suite2p" / "plane0"
    np.save(plane / "iscell.npy", np.array([[1, 0.9], [0, 0.2], [1, 0.8], [1, 0.7]]))
    everything, _ = collect_runs(derivatives, tmp_path, runs=("run-01",))
    cells_only, _ = collect_runs(derivatives, tmp_path, runs=("run-01",), cells_only=True)
    assert everything[0]["n_cells"] == 4
    assert cells_only[0]["n_cells"] == 3 and cells_only[0]["rois"].tolist() == [0, 2, 3]


def test_a_run_too_short_for_the_shifts_is_listed_with_the_reason(derivatives, tmp_path):
    runs, skipped = collect_runs(derivatives, tmp_path, min_shift_s=70.0)
    assert runs == []
    assert [(s, r) for s, r, _ in skipped] == [
        ("ses-224_date-20260809", "run-01"),
        ("ses-224_date-20260809", "run-02"),
    ]
    assert "shift" in skipped[0][2]


def test_a_run_without_frame_times_is_listed_not_dropped(derivatives, tmp_path):
    (derivatives / "ses-224_date-20260809" / "funcimg" / "run-02" / FRAME_TIMES_FILE).unlink()
    runs, skipped = collect_runs(derivatives, tmp_path)
    assert [r["run"] for r in runs] == ["run-01"]
    assert skipped[0][1] == "run-02" and "frame_times" in skipped[0][2]


def test_the_chance_levels_do_not_depend_on_which_other_runs_are_collected(derivatives, tmp_path):
    both, _ = collect_runs(derivatives, tmp_path)
    alone, _ = collect_runs(derivatives, tmp_path, runs=("run-02",))
    np.testing.assert_array_equal(
        both[1]["stats"][("green", "F")]["null_r_bg"],
        alone[0]["stats"][("green", "F")]["null_r_bg"],
    )


def test_nothing_is_written_beside_the_data(derivatives, tmp_path):
    before = sorted(p for p in derivatives.rglob("*"))
    collect_runs(derivatives, tmp_path)
    assert sorted(p for p in derivatives.rglob("*")) == before
