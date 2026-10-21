import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("mini2p_preprocessing")  # the optional funcimg extra

from src.funcimg_analysis.collect import (  # noqa: E402
    collect_pulse_responses,
    top_cells,
    variant_runs,
)
from src.funcimg_analysis.frame_times import FRAME_TIMES_FILE  # noqa: E402
from src.funcimg_analysis.run_responses import VIEWS  # noqa: E402
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
        tiff="71rtask_00001.tif",
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
        write_manifest(funcimg / run / "without_destriping", "71rtask_00001.tif")
    write_plane(funcimg / "run-03" / "without_destriping" / "suite2p" / "plane0", rng)  # no times
    return tmp_path / "der"


def write_manifest(variant_dir, tiff_name):
    (variant_dir / "source_tiff.json").write_text(
        json.dumps({"source_tiff": f"/raw/ses-224/funcimg/{tiff_name}"})
    )


def outcomes_for(tsv_path):
    times = 20.0 + PERIOD * EVENT_FRAMES + 0.01
    return pd.DataFrame({"outcome_time": times, "num_rew_pulses": PULSES, "phase": "task"})


def collect(subject_dir, runs=None):
    return collect_pulse_responses(
        subject_dir,
        224,
        224,
        "without_destriping",
        (-1.0, 2.0),
        ("task",),
        outcomes_for,
        runs=runs,
    )


def test_cells_of_every_run_are_pooled_and_skipped_runs_say_why(derivatives):
    pooled, runs, skipped = collect(derivatives)
    assert [r["run"] for r in runs] == ["run-01", "run-02"]
    assert skipped == [("ses-224_date-20260809", "run-03", f"no {FRAME_TIMES_FILE}")]
    green = pooled[("green", "F")]
    assert green[2]["n_cells"] == 2 * N_CELLS and green[2]["n_runs"] == 2
    zero = int(np.flatnonzero(runs[0]["offsets"] == 0)[0])
    assert green[4]["mean"][zero] > green[2]["mean"][zero] > green[0]["mean"][zero]
    assert set(pooled) == set(VIEWS)  # raw F, dF/F0 and their robust z, per channel
    assert "flat_cells" in runs[0]


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


def test_a_run_registered_from_another_tiff_than_its_frame_times_is_left_out(derivatives):
    run_dir = derivatives / "ses-224_date-20260809" / "funcimg" / "run-02"
    write_manifest(run_dir / "without_destriping", "71rtask_00002.tif")
    _, runs, skipped = collect(derivatives)
    assert [r["run"] for r in runs] == ["run-01"]
    assert "71rtask_00002.tif" in skipped[0][2] and "71rtask_00001.tif" in skipped[0][2]


def test_unexpected_errors_stop_the_analysis(derivatives, monkeypatch):
    from src.funcimg_analysis import collect as collect_module

    def broken(*args, **kwargs):
        raise ValueError("operands could not be broadcast together")

    monkeypatch.setattr(collect_module, "pulse_responses", broken)
    with pytest.raises(ValueError, match="broadcast"):
        collect(derivatives)


def test_top_cells_mark_the_trials_of_the_chosen_phases(derivatives):
    run_dir = derivatives / "ses-224_date-20260809" / "funcimg" / "run-01"
    plane = run_dir / "without_destriping" / "suite2p" / "plane0"

    def two_phases(tsv_path):
        table = outcomes_for(tsv_path)
        table.loc[0, "phase"] = "port_finding"
        return table

    top = top_cells(plane, run_dir, 2, two_phases, phases=("task",))
    assert len(top["outcomes"]) == len(PULSES) - 1
    top = top_cells(plane, run_dir, 2, two_phases, phases=("task", "port_finding"))
    assert len(top["outcomes"]) == len(PULSES)


def test_chosen_runs_are_the_only_ones_found(derivatives):
    found = variant_runs(derivatives, 224, 224, "without_destriping", runs=("run-02", "run-03"))
    assert [run_dir.name for _, run_dir, _ in found] == ["run-02", "run-03"]


def test_only_the_chosen_runs_are_pooled_and_others_are_not_reported(derivatives):
    pooled, runs, skipped = collect(derivatives, runs=("run-02",))
    assert [r["run"] for r in runs] == ["run-02"]
    assert skipped == []
    assert pooled[("green", "F")][2]["n_runs"] == 1


def test_choosing_a_run_that_does_not_exist_names_the_runs_that_do(derivatives):
    with pytest.raises(ValueError, match="run-09.*run-01, run-02, run-03"):
        variant_runs(derivatives, 224, 224, "without_destriping", runs=("run-09",))


def test_a_single_run_name_must_be_given_as_a_collection(derivatives):
    with pytest.raises(TypeError, match="tuple"):
        variant_runs(derivatives, 224, 224, "without_destriping", runs="run-02")


def test_a_run_without_a_record_is_pooled_with_one_warning_naming_it(derivatives):
    variant_dir = (
        derivatives / "ses-224_date-20260809" / "funcimg" / "run-02" / "without_destriping"
    )
    (variant_dir / "source_tiff.json").unlink()
    (variant_dir / "71rtask_00001_dynamic_bidiphase.tif").write_bytes(b"")
    with pytest.warns(UserWarning, match="inferred from file names") as caught:
        _, runs, _ = collect(derivatives)
    assert [r["run"] for r in runs] == ["run-01", "run-02"]
    assert len(caught) == 1 and "run-02" in str(caught[0].message)


def test_pooled_figures_show_robust_z_by_default_and_raw_views_on_request(derivatives):
    pooled, runs, _ = collect(derivatives)
    fig = plot_pulse_responses(pooled, runs[0]["offsets"], runs[0]["period_s"])
    assert "robust z, baseline-subtracted" in fig.axes[0].get_title()
    assert "not neuropil-corrected" in fig.axes[0].get_title()
    assert "robust z, baseline-subtracted" in fig.axes[0].get_ylabel()
    plt.close(fig)
    fig = plot_pulse_responses(
        pooled, runs[0]["offsets"], runs[0]["period_s"], measures=("F", "dF/F0")
    )
    assert fig.axes[0].get_ylabel() == "F (a.u.)" and fig.axes[2].get_ylabel() == "ΔF/F0"
    plt.close(fig)


def set_p_cell(plane_dir, p_cell):
    iscell = np.ones((N_CELLS, 2))
    iscell[:, 1] = p_cell
    np.save(plane_dir / "iscell.npy", iscell)


def test_top_cells_can_be_ranked_by_the_classifier_probability(derivatives):
    run_dir = derivatives / "ses-224_date-20260809" / "funcimg" / "run-01"
    plane = run_dir / "without_destriping" / "suite2p" / "plane0"
    set_p_cell(plane, [0.2, 0.9, 0.9, 0.5])
    top = top_cells(plane, run_dir, 3, outcomes_for, rank_by="p_cell")
    assert top["cells"].tolist() == [1, 2, 3]  # the 0.9 tie goes to the lower ROI index
    assert top["p_cell"].tolist() == [0.9, 0.9, 0.5]
    assert top["dff"].shape == (3, N_FRAMES) and top["rank_by"] == "p_cell"
    by_max = top_cells(plane, run_dir, 3, outcomes_for)
    assert by_max["rank_by"] == "max_dff" and len(by_max["p_cell"]) == 3


def test_an_unknown_ranking_is_refused(derivatives):
    run_dir = derivatives / "ses-224_date-20260809" / "funcimg" / "run-01"
    plane = run_dir / "without_destriping" / "suite2p" / "plane0"
    with pytest.raises(ValueError, match="rank_by"):
        top_cells(plane, run_dir, 3, outcomes_for, rank_by="brightness")


def test_only_the_chosen_measures_are_pooled_and_a_window_from_zero_suits_raw_ones(derivatives):
    pooled, runs, _ = collect_pulse_responses(
        derivatives,
        224,
        224,
        "without_destriping",
        (0.0, 1.0),
        ("task",),
        outcomes_for,
        measures=("F", "dF/F0"),
    )
    assert {m for _, m in pooled} == {"F", "dF/F0"} and len(runs) == 2
