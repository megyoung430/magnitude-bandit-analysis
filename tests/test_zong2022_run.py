"""One imaging run through the Zong et al. 2022 pipeline, and a folder of runs."""

import dataclasses

import numpy as np
import pytest

pytest.importorskip("mini2p_preprocessing")  # the optional funcimg extra

from src.funcimg_analysis import zong2022_io  # noqa: E402
from src.funcimg_analysis.frame_times import RunMismatchError  # noqa: E402
from src.funcimg_analysis.zong2022_motion import MotionSettings  # noqa: E402
from src.funcimg_analysis.zong2022_responses import ResponseSettings  # noqa: E402
from src.funcimg_analysis.zong2022_run import (  # noqa: E402
    RunLeftOut,
    analyse_run,
    analyse_runs,
    build_cell_table,
    top_responsive_cells,
)
from src.funcimg_analysis.zong2022_signal import Zong2022Settings  # noqa: E402
from tests.zong2022_run_fixtures import SESSION, outcome_table, write_synthetic_run  # noqa: E402

VARIANT = "without_destriping"
MOTION = MotionSettings(n_pcs=3, n_extremes=50, max_frames=400)


def load_outcomes(_tsv):
    return outcome_table()


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    derivatives = tmp_path_factory.mktemp("derivatives")
    return derivatives, write_synthetic_run(derivatives)


def run(paths, cache_dir, **overrides):
    kwargs = {
        "settings": Zong2022Settings(),
        "responses": ResponseSettings(),
        "motion": MOTION,
        "cache_dir": cache_dir,
        "load_outcomes": load_outcomes,
    }
    kwargs.update(overrides)
    return analyse_run(paths["plane_dir"], paths["run_dir"], VARIANT, **kwargs)


@pytest.fixture(scope="module")
def result(tree, tmp_path_factory):
    _, paths = tree
    return run(paths, tmp_path_factory.mktemp("cache"))


# --- the run --------------------------------------------------------------------------------------


def test_every_roi_gets_a_row_and_the_planted_cells_pass_the_snr_threshold(result):
    table = result["table"]
    assert table["roi"].tolist() == [0, 1, 2]
    assert table["passes"].all() and (table["snr"] > 3).all()
    assert (table["n_transients"] > 0).all()


def test_the_cell_that_fires_at_rewards_is_responsive_to_rewards_only(result):
    table = result["table"]
    assert table["reward_responsive"].tolist() == [True, False, False]
    assert table["reward_score"].iloc[0] > 0.5
    assert (table["reward_score"].iloc[0] > table["reward_null_95th"].iloc[0]).all()


def test_the_cell_that_fires_at_no_reward_cues_is_responsive_to_those_only(result):
    table = result["table"]
    assert table["no_reward_responsive"].tolist() == [False, False, True]


def test_pooled_responses_are_split_by_pulse_count_for_passing_and_all_cells(result):
    reward = result["reward"]
    assert reward["n_trials"] == {0: 2, 1: 1, 2: 1, 3: 1, 4: 1}
    for trace in ("dff_clear", "e_clear"):
        for group in ("snr_pass", "all"):
            assert set(reward["pooled"][trace][group]) == {0, 1, 2, 3, 4}
    pooled = reward["pooled"]["dff_clear"]["snr_pass"]
    assert pooled[3]["n_cells"] == 3
    offsets = reward["aligned"]["dff_clear"].offsets
    assert offsets[0] == -15 and offsets[-1] == 38  # -2 s to 5 s at 0.133 s per frame
    # ROI 0 fires at the rewarded outcomes: its mean response peaks at the outcome's frame
    window = reward["aligned"]["dff_clear"].windows[0]
    rewarded = reward["aligned"]["dff_clear"].pulses > 0
    assert np.argmax(window[rewarded].mean(axis=0)) in (15, 16)


def test_chance_summaries_count_responsive_cells_against_five_percent(result):
    summary = result["reward"]["tests"]["reward"]["summary_all"]
    assert summary["n_cells"] == 3 and summary["n_responsive"] == 1
    assert summary["expected_by_chance"] == pytest.approx(0.15)


def test_motion_qc_reports_residual_drift_and_the_recorded_shifts(result):
    motion = result["motion"]
    assert list(motion["drift"]["pc"]) == [1, 2, 3]
    assert motion["i_top"].shape[0] == 3 and motion["crop"] == (2, 126, 2, 126)
    assert motion["i_top"].shape[1:] == (124, 124)
    assert set(motion["offsets"]) == {"xoff", "yoff"}


def test_the_run_carries_what_the_figures_need(result):
    assert result["labels"].shape == (128, 128) and result["labels"].max() == 3
    assert result["alignment"]["dy"] == 0 and result["alignment"]["dx"] == 0
    assert result["mean_image"].shape == (128, 128)
    assert result["signal"].dff.shape == (3, result["n_frames"])
    assert result["period_s"] == pytest.approx(0.133)


def test_nothing_is_written_next_to_the_data(tree, tmp_path):
    derivatives, paths = tree
    before = sorted(p.relative_to(derivatives) for p in derivatives.rglob("*"))
    run(paths, tmp_path / "cache")
    assert sorted(p.relative_to(derivatives) for p in derivatives.rglob("*")) == before


def test_a_second_run_reads_the_cache_instead_of_extracting_again(tree, tmp_path, monkeypatch):
    _, paths = tree
    first = run(paths, tmp_path)
    assert {p.parent.name for p in tmp_path.rglob("*.pkl")} == {"extraction", "signal", "motion"}

    def refuse(*args, **kwargs):
        raise AssertionError("extraction was repeated")

    monkeypatch.setattr(zong2022_io, "extract_channels", refuse)
    second = run(paths, tmp_path)
    np.testing.assert_array_equal(second["table"]["snr"], first["table"]["snr"])
    np.testing.assert_array_equal(second["signal"].dff_clear, first["signal"].dff_clear)


def test_progress_lines_name_each_stage(tree, tmp_path):
    _, paths = tree
    lines = []
    run(paths, tmp_path, progress=lines.append)
    text = "\n".join(lines)
    for stage in ("masks", "extract", "ΔF/F", "responses", "motion"):
        assert stage in text


# --- refusals -------------------------------------------------------------------------------------


def test_a_run_without_a_cellpose_segmentation_is_left_out_with_the_path(tree, tmp_path):
    _, paths = tree
    seg = paths["run_dir"] / "suite2p/plane0/meanImg_green_functional_contrast_seg.npy"
    hidden = seg.with_suffix(".hidden")
    seg.rename(hidden)
    try:
        with pytest.raises(RunLeftOut, match="meanImg_green_functional_contrast_seg.npy"):
            run(paths, tmp_path)
    finally:
        hidden.rename(seg)


def test_traces_that_do_not_match_the_frame_clock_are_refused(tmp_path):
    other = write_synthetic_run(tmp_path / "der3", n_frames=600, run="run-03")
    path = other["run_dir"] / "frame_times.npz"
    with np.load(path) as saved:
        kept = {key: saved[key] for key in saved.files}
    for key in ("time_s", "frame_index", "daq_frame_s"):
        kept[key] = kept[key][:-5]  # the clock has 5 frames fewer than the movie
    np.savez(path, **kept)
    with pytest.raises(RunMismatchError, match="frame"):
        run(other, tmp_path / "cache")


def test_a_run_without_frame_times_is_left_out(tmp_path):
    other = write_synthetic_run(tmp_path / "der2", n_frames=600, run="run-07")
    (other["run_dir"] / "frame_times.npz").unlink()
    with pytest.raises(RunLeftOut, match="frame_times"):
        run(other, tmp_path / "cache")


# --- many runs ------------------------------------------------------------------------------------


def test_runs_are_collected_in_order_and_the_ones_left_out_are_listed_with_reasons(
    tmp_path,
):
    derivatives = tmp_path / "derivatives"
    write_synthetic_run(derivatives, n_frames=2400, run="run-01")
    broken = write_synthetic_run(derivatives, n_frames=2400, run="run-02")
    (broken["run_dir"] / "frame_times.npz").unlink()
    lines = []
    runs, skipped = analyse_runs(
        derivatives, 224, 224, VARIANT, runs=None, settings=Zong2022Settings(),
        responses=ResponseSettings(), motion=MOTION, cache_dir=tmp_path / "cache",
        load_outcomes=load_outcomes, report=lines.append,
    )  # fmt: skip
    assert [(r["session"], r["run"]) for r in runs] == [(SESSION, "run-01")]
    assert skipped[0][:2] == (SESSION, "run-02") and "frame_times" in skipped[0][2]
    assert any("[1/2]" in line for line in lines) and any("[2/2]" in line for line in lines)


# --- tables ---------------------------------------------------------------------------------------


def test_top_responsive_cells_rank_snr_passing_responsive_cells_by_score():
    import pandas as pd

    table = pd.DataFrame(
        {
            "roi": [0, 1, 2, 3, 4],
            "passes": [True, True, False, True, True],
            "reward_responsive": [True, True, True, False, True],
            "reward_score": [0.3, 0.9, 0.99, 0.95, 0.5],
        }
    )
    assert top_responsive_cells(table, "reward", 2).tolist() == [1, 4]
    assert top_responsive_cells(table, "reward", 10).tolist() == [1, 4, 0]


def test_the_cell_table_has_one_row_per_roi_with_the_documented_columns(result):
    table = build_cell_table(result["signal"], result["reward"]["tests"])
    expected = {
        "roi", "f0_ok", "excluded", "snr", "signal", "noise", "n_transients", "n_events", "passes",
        "reward_score", "reward_null_95th", "reward_p_value", "reward_responsive",
        "no_reward_score", "no_reward_null_95th", "no_reward_p_value", "no_reward_responsive",
    }  # fmt: skip
    assert set(table.columns) == expected
    assert dataclasses.is_dataclass(result["signal"])
