"""Background-corrected ΔF/F0 from disk: exclusions, responses next to Suite2p's, cache."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("mini2p_preprocessing")  # the optional funcimg extra

from src.funcimg_analysis import collect_residual  # noqa: E402
from src.funcimg_analysis.collect import pool_runs  # noqa: E402
from src.funcimg_analysis.collect_residual import (  # noqa: E402
    collect_residual_dff,
    summary_table,
)
from src.funcimg_analysis.frame_times import FRAME_TIMES_FILE  # noqa: E402
from tests.funcimg_shared_fixtures import N_CELLS, WINDOW_S, outcomes, response_2s  # noqa: E402
from tests.test_funcimg_shared_collect import plane_dir, write_run  # noqa: E402

VIEW = ("green", "dF/F0 z")


@pytest.fixture
def derivatives(tmp_path):
    funcimg = tmp_path / "der" / "ses-224_date-20260809" / "funcimg"
    write_run(funcimg, "run-01", np.random.default_rng(0), "background")
    write_run(funcimg, "run-02", np.random.default_rng(1), "cells")
    return tmp_path / "der"


def collect(derivatives, tmp_path, **overrides):
    settings = {
        "fit_exclude_window_s": (-0.3, 1.6),
        "cache_dir": tmp_path / "cache",
        "results_cache_dir": tmp_path / "results",
    }
    settings.update(overrides)
    return collect_residual_dff(
        derivatives,
        224,
        224,
        "without_destriping",
        WINDOW_S,
        ("task",),
        lambda _: outcomes(),
        **settings,
    )


def test_a_response_shared_with_the_background_is_gone_after_the_correction(derivatives, tmp_path):
    runs, skipped = collect(derivatives, tmp_path)
    assert skipped == [] and [r["run"] for r in runs] == ["run-01", "run-02"]
    in_background, in_cells = runs
    before = response_2s(in_background, "suite2p", view=VIEW)
    assert before > 0.5 and abs(response_2s(in_background, "corrected", view=VIEW)) < 0.25 * before


def test_a_response_private_to_the_cells_survives_the_correction(derivatives, tmp_path):
    _, in_cells = collect(derivatives, tmp_path)[0]
    assert response_2s(in_cells, "corrected", view=VIEW) > 0.7 * response_2s(
        in_cells, "suite2p", view=VIEW
    )


def test_slope_and_residual_correlation_are_reported_per_cell(derivatives, tmp_path):
    runs, _ = collect(derivatives, tmp_path, fit_exclude_outcomes=False)
    green = runs[0]["channels"]["green"]
    assert green["slope"].shape == (N_CELLS,) and np.isfinite(green["slope"]).all()
    assert np.abs(green["r_after"]).max() < 1e-6  # fitted on every frame: nothing shared is left
    assert runs[0]["rois"].tolist() == list(range(N_CELLS))
    assert runs[0]["fit_fraction"] == 1.0


def test_the_summary_table_counts_cells_and_exclusions(derivatives, tmp_path):
    runs, _ = collect(derivatives, tmp_path)
    rows = summary_table(runs)
    assert [(r["run"], r["channel"]) for r in rows] == [
        ("run-01", "green"),
        ("run-01", "red"),
        ("run-02", "green"),
        ("run-02", "red"),
    ]
    assert rows[0]["n cells"] == N_CELLS and rows[0]["n excluded"] == 0
    assert np.isfinite(rows[0]["median b"]) and np.isfinite(rows[0]["median r(F_corr, bg)"])


def test_a_dark_cell_is_excluded_and_never_reaches_the_responses(derivatives, tmp_path):
    f = plane_dir(derivatives) / "F.npy"
    edited = np.load(f)
    edited[2] = 0.1 * np.random.default_rng(5).normal(size=edited.shape[1])
    np.save(f, edited)
    runs, _ = collect(derivatives, tmp_path, runs=("run-01",))
    green = runs[0]["channels"]["green"]
    assert list(green["excluded"]) == [2] and "F0" in green["excluded"][2]
    assert runs[0]["corrected"]["means"][VIEW][4].shape[0] == N_CELLS - 1
    assert runs[0]["suite2p"]["means"][VIEW][4].shape[0] == N_CELLS - 1  # same cells compared
    assert summary_table(runs)[0]["n excluded"] == 1


def test_a_run_with_no_usable_cell_in_a_channel_is_left_out_with_the_reason(derivatives, tmp_path):
    f = plane_dir(derivatives) / "F.npy"
    np.save(f, np.random.default_rng(6).normal(0, 0.1, np.load(f).shape))  # dark: F0 is below 0
    runs, skipped = collect(derivatives, tmp_path)
    assert [r["run"] for r in runs] == ["run-02"]
    assert skipped[0][1] == "run-01" and "no usable cell" in skipped[0][2]


def test_example_traces_hold_both_dffs_of_the_same_cells(derivatives, tmp_path):
    runs, _ = collect(derivatives, tmp_path, runs=("run-01",))
    example = runs[0]["examples"]["green"]
    k = len(example["cells"])
    assert 1 <= k <= 4 and example["suite2p"].shape == example["corrected"].shape
    assert (
        example["suite2p"].shape[0] == k and len(runs[0]["time_s"]) == example["suite2p"].shape[1]
    )


def test_pooled_responses_cover_both_measures_for_both_channels(derivatives, tmp_path):
    runs, _ = collect(derivatives, tmp_path)
    for which in ("suite2p", "corrected"):
        pooled = pool_runs([r[which] for r in runs], ("dF/F0 z",))
        assert set(pooled) == {("green", "dF/F0 z"), ("red", "dF/F0 z")}


def test_a_run_without_frame_times_is_listed_not_dropped(derivatives, tmp_path):
    (derivatives / "ses-224_date-20260809" / "funcimg" / "run-02" / FRAME_TIMES_FILE).unlink()
    runs, skipped = collect(derivatives, tmp_path)
    assert [r["run"] for r in runs] == ["run-01"] and skipped[0][1] == "run-02"


def test_outcome_windows_that_cover_the_run_are_listed_with_the_fix(derivatives, tmp_path):
    runs, skipped = collect(derivatives, tmp_path, fit_exclude_window_s=None)
    assert runs == [] and "fit_exclude_window_s" in skipped[0][2]


def test_nothing_is_written_beside_the_data(derivatives, tmp_path):
    before = sorted(derivatives.rglob("*"))
    collect(derivatives, tmp_path)
    assert sorted(derivatives.rglob("*")) == before


def test_a_second_collection_comes_from_the_cache(derivatives, tmp_path, monkeypatch):
    calls = []
    real = collect_residual.residual_dff_channel
    monkeypatch.setattr(
        collect_residual,
        "residual_dff_channel",
        lambda *a, **k: calls.append(1) or real(*a, **k),
    )
    first, _ = collect(derivatives, tmp_path)
    n_first = len(calls)
    second, _ = collect(derivatives, tmp_path)
    assert n_first == 4 and len(calls) == n_first  # 2 runs x 2 channels, nothing recomputed
    np.testing.assert_array_equal(
        first[0]["channels"]["green"]["slope"], second[0]["channels"]["green"]["slope"]
    )
    calls.clear()
    collect(derivatives, tmp_path, min_f0_fraction=0.2)  # another setting is another result
    assert len(calls) == 4
