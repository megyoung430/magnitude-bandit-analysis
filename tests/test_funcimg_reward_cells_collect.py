"""Reward-responsive cells from disk: both ΔF/F0 measures, left-out runs, p(cell), determinism."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("mini2p_preprocessing")  # the optional funcimg extra

from src.funcimg_analysis.collect_residual import collect_residual_dff  # noqa: E402
from src.funcimg_analysis.collect_reward_cells import (  # noqa: E402
    BG_CORRECTED,
    SUITE2P,
    collect_reward_cells,
)
from src.funcimg_analysis.frame_times import FRAME_TIMES_FILE  # noqa: E402
from tests.funcimg_shared_fixtures import N_CELLS, outcomes  # noqa: E402
from tests.test_funcimg_shared_collect import plane_dir, write_run  # noqa: E402

ANALYSIS = {
    "phases": ("task",),
    "plot_window_s": (-1.5, 3.0),
    "baseline_s": (-1.5, 0.0),
    "response_s": (0.0, 2.0),
    "contrast": "rewarded",
    "rank_on": "split_half",
    "n_shifts": 20,
    "min_shift_s": 10.0,
}


@pytest.fixture
def derivatives(tmp_path):
    funcimg = tmp_path / "der" / "ses-224_date-20260809" / "funcimg"
    write_run(funcimg, "run-01", np.random.default_rng(0), "background")
    write_run(funcimg, "run-02", np.random.default_rng(1), "cells")
    return tmp_path / "der"


def residual(derivatives, tmp_path, **overrides):
    runs, _ = collect_residual_dff(
        derivatives,
        224,
        224,
        "without_destriping",
        (-1.5, 3.0),
        ("task",),
        lambda _: outcomes(),
        fit_exclude_window_s=(-0.3, 1.6),
        cache_dir=tmp_path / "cache",
        results_cache_dir=tmp_path / "results",
        **overrides,
    )
    return runs


def collect(derivatives, tmp_path, residual_runs="auto", **overrides):
    if isinstance(residual_runs, str):
        residual_runs = residual(derivatives, tmp_path)
    settings = {**ANALYSIS, "residual_runs": residual_runs, "seed": 0}
    settings.update(overrides)
    return collect_reward_cells(
        derivatives, 224, 224, "without_destriping", lambda _: outcomes(), **settings
    )


def test_each_run_gets_an_analysis_of_both_measures_and_the_plane_it_needs_for_plotting(
    derivatives, tmp_path
):
    runs, skipped = collect(derivatives, tmp_path)
    assert skipped == [] and [r["run"] for r in runs] == ["run-01", "run-02"]
    run = runs[0]
    assert set(run["measures"]) == {SUITE2P, BG_CORRECTED}
    assert run["p_cell"].shape == (N_CELLS,) and run["rois"].tolist() == list(range(N_CELLS))
    assert run["labels"].ndim == 2 and run["mean_image"].shape == run["labels"].shape
    assert run["cross_validated"] is True


def test_a_response_shared_with_the_background_ranks_high_before_and_low_after_the_correction(
    derivatives, tmp_path
):
    in_background = collect(derivatives, tmp_path)[0][0]["measures"]
    suite2p = np.nanmedian(in_background[SUITE2P]["index_a"])
    corrected = np.nanmedian(in_background[BG_CORRECTED]["index_a"])
    assert suite2p > 0.5 and corrected < 0.5 * suite2p


def test_a_private_response_keeps_a_high_index_after_the_correction(derivatives, tmp_path):
    in_cells = collect(derivatives, tmp_path)[0][1]["measures"]
    assert np.nanmedian(in_cells[BG_CORRECTED]["index_a"]) > 0.5


def test_an_excluded_cell_has_no_index_in_the_corrected_measure_only(derivatives, tmp_path):
    f = plane_dir(derivatives) / "F.npy"
    edited = np.load(f)
    edited[2] = 0.1 * np.random.default_rng(5).normal(size=edited.shape[1])
    np.save(f, edited)
    run = collect(derivatives, tmp_path, runs=("run-01",))[0][0]["measures"]
    assert np.isnan(run[BG_CORRECTED]["index_a"][2]) and 2 not in run[BG_CORRECTED]["order"]


def test_a_run_without_a_background_corrected_result_is_left_out_with_the_reason(
    derivatives, tmp_path
):
    only_first = residual(derivatives, tmp_path)[:1]
    runs, skipped = collect(derivatives, tmp_path, residual_runs=only_first)
    assert [r["run"] for r in runs] == ["run-01"]
    assert skipped[0][1] == "run-02" and "background-corrected" in skipped[0][2]


def test_only_suite2p_needs_no_residual_runs(derivatives, tmp_path):
    runs, _ = collect(derivatives, tmp_path, residual_runs=None, measures=(SUITE2P,))
    assert set(runs[0]["measures"]) == {SUITE2P}


def test_asking_for_the_corrected_measure_without_residual_runs_raises(derivatives, tmp_path):
    with pytest.raises(ValueError, match="residual_runs"):
        collect(derivatives, tmp_path, residual_runs=None)


def test_unknown_measures_are_refused(derivatives, tmp_path):
    with pytest.raises(ValueError, match="measure"):
        collect(derivatives, tmp_path, measures=("F",))


def test_a_run_too_short_for_the_shifts_or_with_too_few_trials_is_listed(derivatives, tmp_path):
    runs, skipped = collect(derivatives, tmp_path, min_shift_s=70.0)
    assert runs == [] and "shift" in skipped[0][2]
    runs, skipped = collect(derivatives, tmp_path, contrast="4_vs_0", phases=("port_finding",))
    assert runs == [] and skipped  # no task trial: left out, not crashed


def test_a_run_without_frame_times_is_listed_not_dropped(derivatives, tmp_path):
    residual_runs = residual(derivatives, tmp_path)
    (derivatives / "ses-224_date-20260809" / "funcimg" / "run-02" / FRAME_TIMES_FILE).unlink()
    runs, skipped = collect(derivatives, tmp_path, residual_runs=residual_runs)
    assert [r["run"] for r in runs] == ["run-01"] and skipped[0][1] == "run-02"


def test_the_split_and_shifts_do_not_depend_on_which_other_runs_are_collected(
    derivatives, tmp_path
):
    both = collect(derivatives, tmp_path)[0]
    alone = collect(derivatives, tmp_path, runs=("run-02",))[0]
    np.testing.assert_array_equal(both[1]["halves"], alone[0]["halves"])
    np.testing.assert_array_equal(
        both[1]["measures"][SUITE2P]["index_a"], alone[0]["measures"][SUITE2P]["index_a"]
    )


def test_nothing_is_written_beside_the_data(derivatives, tmp_path):
    residual_runs = residual(derivatives, tmp_path)
    before = sorted(derivatives.rglob("*"))
    collect(derivatives, tmp_path, residual_runs=residual_runs)
    assert sorted(derivatives.rglob("*")) == before
