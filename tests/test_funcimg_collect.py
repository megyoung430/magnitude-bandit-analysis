import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.funcimg_analysis.collect import collect_pulse_responses, top_cells  # noqa: E402
from src.funcimg_analysis.frame_times import FRAME_TIMES_FILE  # noqa: E402
from src.funcimg_visualization.plot_funcimg_responses import (  # noqa: E402
    plot_pulse_responses,
    plot_top_cells,
)

N_FRAMES, PERIOD, N_CELLS = 600, 0.133, 4
EVENT_FRAMES = np.array([100, 200, 300, 400, 500])
PULSES = [0, 2, 2, 4, 1]


def write_plane(plane_dir, rng, n_frames=N_FRAMES):
    """A plane in the layout mini2p_preprocessing writes, with a response at each event."""
    plane_dir.mkdir(parents=True)
    f = 200 + rng.normal(0, 2, (N_CELLS, n_frames))
    for frame, pulses in zip(EVENT_FRAMES, PULSES, strict=True):
        f[:, frame : frame + 4] += 40 * (pulses + 1)
    labels = np.zeros((20, 30), np.int32)
    stat = []
    for cell in range(N_CELLS):
        ypix, xpix = np.meshgrid(np.arange(2, 5), np.arange(3 + 6 * cell, 6 + 6 * cell))
        labels[ypix, xpix] = cell + 1
        stat.append({"ypix": ypix.ravel(), "xpix": xpix.ravel(), "npix": 9, "med": [3, 4]})
    ops = {"fs": 1 / PERIOD, "Ly": 20, "Lx": 30, "nframes": n_frames, "nchannels": 2}
    ops["meanImg"] = rng.normal(100, 5, (20, 30)).astype(np.float32)
    arrays = {"F": f, "Fneu": f * 0.2, "F_chan2": f * 0.5, "Fneu_chan2": f * 0.1}
    arrays.update(iscell=np.ones((N_CELLS, 2)), spks=np.zeros_like(f))
    for name, array in arrays.items():
        np.save(plane_dir / f"{name}.npy", array)
    np.save(plane_dir / "stat.npy", np.array(stat, dtype=object), allow_pickle=True)
    np.save(plane_dir / "ops.npy", ops, allow_pickle=True)
    settings = {"neuropil_coefficient": 0.7, "baseline": "prctile", "prctile_baseline": 8.0}
    (plane_dir / "cellpose_extraction.json").write_text(
        json.dumps({"fluorescence_settings": settings})
    )


def write_frame_times(run_dir, n_frames=N_FRAMES):
    time_s = 20.0 + PERIOD * np.arange(n_frames)
    np.savez(
        run_dir / FRAME_TIMES_FILE,
        time_s=time_s,
        frame_index=np.arange(n_frames),
        daq_frame_s=time_s - 20.0,
        tiff="t.tif",
        h5="d.h5",
        tsv=str(run_dir / "behaviour.tsv"),
        quality=json.dumps({"fraction_matched": 1.0}),
    )
    return time_s


@pytest.fixture
def derivatives(tmp_path):
    rng = np.random.default_rng(0)
    funcimg = tmp_path / "der" / "ses-224_date-20260809" / "funcimg"
    for run in ("run-01", "run-02"):
        write_plane(funcimg / run / "without_destriping" / "suite2p" / "plane0", rng)
        write_frame_times(funcimg / run)
    write_plane(funcimg / "run-03" / "without_destriping" / "suite2p" / "plane0", rng)  # no times
    return tmp_path / "der"


def outcomes_for(tsv_path):
    times = 20.0 + PERIOD * EVENT_FRAMES + 0.01
    return pd.DataFrame({"outcome_time": times, "num_rew_pulses": PULSES, "phase": "task"})


def collect(subject_dir):
    return collect_pulse_responses(
        subject_dir, 224, 224, "without_destriping", (-1.0, 2.0), ("task",), outcomes_for
    )


def test_cells_of_every_run_are_pooled_and_skipped_runs_say_why(derivatives):
    pooled, runs, skipped = collect(derivatives)
    assert [r["run"] for r in runs] == ["run-01", "run-02"]
    assert skipped == [("ses-224_date-20260809", "run-03", f"no {FRAME_TIMES_FILE}")]
    green = pooled[("green", "F")]
    assert green[2]["n_cells"] == 2 * N_CELLS and green[2]["n_runs"] == 2
    zero = int(np.flatnonzero(runs[0]["offsets"] == 0)[0])
    assert green[4]["mean"][zero] > green[2]["mean"][zero] > green[0]["mean"][zero]
    assert set(pooled) == {("green", "F"), ("green", "dF/F0"), ("red", "F"), ("red", "dF/F0")}


def test_a_run_whose_traces_do_not_match_its_frame_clock_is_skipped(derivatives):
    run_dir = derivatives / "ses-224_date-20260809" / "funcimg" / "run-02"
    write_frame_times(run_dir, n_frames=N_FRAMES + 3)
    _, runs, skipped = collect(derivatives)
    assert [r["run"] for r in runs] == ["run-01"]
    assert "frames" in skipped[0][2]


def test_top_cells_and_figures(derivatives):
    pooled, runs, _ = collect(derivatives)
    fig = plot_pulse_responses(pooled, runs[0]["offsets"], runs[0]["period_s"], x_units="frames")
    assert len(fig.axes) == 4 and len(fig.axes[0].lines) >= 4
    plt.close(fig)
    run_dir = derivatives / "ses-224_date-20260809" / "funcimg" / "run-01"
    top = top_cells(run_dir / "without_destriping" / "suite2p" / "plane0", run_dir, 3, outcomes_for)
    assert len(top["cells"]) == 3 and top["dff"].shape == (3, N_FRAMES)
    assert np.all(np.diff(top["peaks"]) <= 0)
    fig = plot_top_cells(top, x_units="s", title="run-01")
    plt.close(fig)
