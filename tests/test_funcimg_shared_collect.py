"""Shared-signal analysis from disk: collection, left-out runs, pooling and the result cache."""

import json

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("mini2p_preprocessing")  # the optional funcimg extra

from src.funcimg_analysis import collect_shared, shared_cache  # noqa: E402
from src.funcimg_analysis.collect_shared import (  # noqa: E402
    collect_shared_signal,
    pool_shared_signal,
)
from src.funcimg_analysis.frame_times import FRAME_TIMES_FILE  # noqa: E402
from tests.funcimg_shared_fixtures import (  # noqa: E402
    MEASURES,
    N_CELLS,
    N_FRAMES,
    SETTINGS,
    WINDOW_S,
    analyse,
    frame_times,
    make_traces,
    neuropil_and_halves,
    outcomes,
    response_2s,
    run_inputs,
)
from tests.test_funcimg_background import write_plane  # noqa: E402


def write_run(funcimg, run, rng, reward_in, iscell=None):
    """A run folder: the plane's binaries hold the background; the .npy files hold the cells."""
    plane = funcimg / run / "without_destriping" / "suite2p" / "plane0"
    green_bg, green_cells = make_traces(rng, reward_in)
    red_bg, red_cells = make_traces(rng, "none")
    write_plane(plane, rng, green_bg, red_bg)
    green_neuropil, _ = neuropil_and_halves(rng, green_bg, green_cells)
    red_neuropil, _ = neuropil_and_halves(rng, red_bg, red_cells)
    for name, array in {
        "F": green_cells,
        "Fneu": green_neuropil,
        "F_chan2": red_cells,
        "Fneu_chan2": red_neuropil,
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


def plane_dir(derivatives, run="run-01"):
    funcimg = derivatives / "ses-224_date-20260809" / "funcimg"
    return funcimg / run / "without_destriping" / "suite2p" / "plane0"


@pytest.fixture
def derivatives(tmp_path):
    funcimg = tmp_path / "der" / "ses-224_date-20260809" / "funcimg"
    write_run(funcimg, "run-01", np.random.default_rng(0), "background")
    write_run(funcimg, "run-02", np.random.default_rng(1), "cells")
    return tmp_path / "der"


def collect_runs(derivatives, tmp_path, **overrides):
    settings = {
        **SETTINGS,
        "measures": MEASURES,
        "cache_dir": tmp_path / "cache",
        "results_cache_dir": tmp_path / "results",
    }
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
        assert np.isfinite(run["stats"][("red", "F")]["bg_reliability"])
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


def test_the_neuropil_and_background_halves_come_from_the_plane_and_the_binaries(
    derivatives, tmp_path
):
    runs, _ = collect_runs(derivatives, tmp_path, runs=("run-01",))
    stats = runs[0]["stats"][("green", "F")]
    # the fixture's background pixels all carry one trace, so its two halves agree exactly
    assert stats["bg_half_r"] == pytest.approx(1.0) and stats["bg_reliability"] == pytest.approx(
        1.0
    )
    assert np.isfinite(stats["partial_r_neuropil"]).all()


def test_cells_only_keeps_the_rois_the_classifier_accepts(derivatives, tmp_path):
    np.save(
        plane_dir(derivatives) / "iscell.npy", np.array([[1, 0.9], [0, 0.2], [1, 0.8], [1, 0.7]])
    )
    everything, _ = collect_runs(derivatives, tmp_path, runs=("run-01",))
    cells_only, _ = collect_runs(derivatives, tmp_path, runs=("run-01",), cells_only=True)
    assert everything[0]["n_cells"] == 4
    assert cells_only[0]["n_cells"] == 3 and cells_only[0]["rois"].tolist() == [0, 2, 3]


def test_cells_only_with_a_single_accepted_roi_leaves_the_run_out_with_a_reason(
    derivatives, tmp_path
):
    np.save(
        plane_dir(derivatives) / "iscell.npy", np.array([[0, 0.1], [1, 0.9], [0, 0.2], [0, 0.1]])
    )
    runs, skipped = collect_runs(derivatives, tmp_path, cells_only=True)
    assert [r["run"] for r in runs] == ["run-02"]  # the other run keeps all its ROIs
    assert skipped[0][1] == "run-01" and "two" in skipped[0][2]


def test_a_run_too_short_for_the_shifts_is_listed_with_the_reason(derivatives, tmp_path):
    runs, skipped = collect_runs(derivatives, tmp_path, min_shift_s=70.0)
    assert runs == []
    assert [(s, r) for s, r, _ in skipped] == [
        ("ses-224_date-20260809", "run-01"),
        ("ses-224_date-20260809", "run-02"),
    ]
    assert "shift" in skipped[0][2]


def test_a_run_whose_outcome_windows_cover_it_is_listed_with_the_fix(derivatives, tmp_path):
    runs, skipped = collect_runs(derivatives, tmp_path, fit_exclude_window_s=None)
    assert runs == [] and "fit_exclude_window_s" in skipped[0][2]


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


# --- the result cache ---------------------------------------------------------------------------


@pytest.fixture
def analysis_calls(monkeypatch):
    calls = []
    real = collect_shared.analyse_shared_signal

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(collect_shared, "analyse_shared_signal", counting)
    return calls


def test_a_second_collection_comes_from_the_cache_with_the_same_numbers(
    derivatives, tmp_path, analysis_calls
):
    first, _ = collect_runs(derivatives, tmp_path)
    assert len(analysis_calls) == 2
    second, _ = collect_runs(derivatives, tmp_path)
    assert len(analysis_calls) == 2  # nothing recomputed
    np.testing.assert_array_equal(
        first[1]["stats"][("red", "F")]["partial_r"], second[1]["stats"][("red", "F")]["partial_r"]
    )
    np.testing.assert_array_equal(first[0]["rois"], second[0]["rois"])
    assert list((tmp_path / "results").glob("*"))


def test_changing_a_trace_a_setting_or_the_seed_recomputes_only_that_run(
    derivatives, tmp_path, analysis_calls
):
    collect_runs(derivatives, tmp_path)
    analysis_calls.clear()
    f = plane_dir(derivatives) / "F.npy"
    edited = np.load(f)
    edited[0] += 1.0
    np.save(f, edited)
    collect_runs(derivatives, tmp_path)
    assert len(analysis_calls) == 1  # run-01 only
    analysis_calls.clear()
    collect_runs(derivatives, tmp_path, n_shifts=SETTINGS["n_shifts"] + 1)
    assert len(analysis_calls) == 2
    analysis_calls.clear()
    collect_runs(derivatives, tmp_path, n_shifts=SETTINGS["n_shifts"] + 1, seed=7)
    assert len(analysis_calls) == 2
    analysis_calls.clear()
    collect_runs(derivatives, tmp_path, n_shifts=SETTINGS["n_shifts"] + 1, seed=7)
    assert not analysis_calls


def test_other_null_kinds_or_fit_settings_do_not_reuse_a_result(
    derivatives, tmp_path, analysis_calls
):
    collect_runs(derivatives, tmp_path, runs=("run-01",))
    analysis_calls.clear()
    collect_runs(derivatives, tmp_path, runs=("run-01",), null_kinds=("r_bg",))
    collect_runs(derivatives, tmp_path, runs=("run-01",), fit_exclude_outcomes=False)
    collect_runs(derivatives, tmp_path, runs=("run-01",), cells_only=True)
    assert len(analysis_calls) == 3


def test_a_new_code_version_invalidates_the_cache(
    derivatives, tmp_path, analysis_calls, monkeypatch
):
    collect_runs(derivatives, tmp_path, runs=("run-01",))
    analysis_calls.clear()
    monkeypatch.setattr(shared_cache, "CODE_VERSION", "another version")
    collect_runs(derivatives, tmp_path, runs=("run-01",))
    assert len(analysis_calls) == 1


def test_the_cache_key_depends_on_array_content_and_not_on_dict_order():
    a = {"x": np.arange(5.0), "y": ("a", 1)}
    b = {"y": ("a", 1), "x": np.arange(5.0)}
    assert shared_cache.content_key(a) == shared_cache.content_key(b)
    assert shared_cache.content_key(a) != shared_cache.content_key({**a, "x": np.arange(5.0) + 1})
    assert shared_cache.content_key({"x": np.zeros(4)}) != shared_cache.content_key(
        {"x": np.zeros((2, 2))}
    )


def test_a_truncated_cache_file_is_recomputed_not_trusted(derivatives, tmp_path, analysis_calls):
    collect_runs(derivatives, tmp_path, runs=("run-01",))
    (cached,) = list((tmp_path / "results").glob("*"))
    cached.write_bytes(cached.read_bytes()[:20])
    analysis_calls.clear()
    collect_runs(derivatives, tmp_path, runs=("run-01",))
    assert len(analysis_calls) == 1


# --- pooling ------------------------------------------------------------------------------------


def test_runs_with_different_frame_periods_cannot_be_pooled():
    result = analyse(run_inputs(np.random.default_rng(0), "none", channels=("green", "red")))
    slower = {**result, "original": {**result["original"], "period_s": 0.2}}
    with pytest.raises(ValueError, match="frame period"):
        pool_shared_signal(
            [{"original": result["original"], "residual": result["residual"]}, slower], MEASURES
        )


def test_runs_whose_periods_differ_only_by_measurement_noise_pool_fine():
    result = analyse(run_inputs(np.random.default_rng(1), "none", channels=("green", "red")))
    nearly = {
        "original": {**result["original"], "period_s": result["original"]["period_s"] * 1.0001},
        "residual": result["residual"],
    }
    original, _ = pool_shared_signal([result, nearly], MEASURES)
    assert ("green", "F z") in original
