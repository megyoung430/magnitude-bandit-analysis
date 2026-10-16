"""Tests for the reversal-aligned arm entry point on synthetic sessions."""

import copy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis.get_good_reversal_info import get_good_reversal_info
from src.behavior_analysis.reversal_aligned_arms import (
    EmptySelectionWarning,
    add_magnitude_windows,
    build_windows_with_magnitudes,
)
from src.pipelines.reversal_aligned_arms import run_reversal_aligned_arm_analysis
from tests.plot_capture import capture_figures
from tests.plot_fixtures import make_mouse

KWARGS = dict(pre=10, post=6)


@pytest.fixture
def real_files():
    """Ask for this fixture to let a test write its figure files (slow: about a second each)."""


@pytest.fixture(autouse=True)
def capture(request, monkeypatch):
    """Record figure saving and showing instead of rendering, unless the test wants real files."""
    capture = None if "real_files" in request.fixturenames else capture_figures(monkeypatch)
    yield capture
    plt.close("all")


def one_problem():
    return {1: {"m": make_mouse(1, pick=0)}}


def two_problems():
    return {
        4: {"m": make_mouse(1, pick=0), "n": make_mouse(1, pick=1)},
        5: {"m": make_mouse(1, pick=0, first_day=11)},
    }


def run(trials, out_dir, **kwargs):
    # The fixtures' reversals are all second_to_third, so best_to_third is empty.
    with pytest.warns(EmptySelectionWarning):
        return run_reversal_aligned_arm_analysis(trials, out_dir, **KWARGS, **kwargs)


def test_build_windows_with_magnitudes_matches_the_two_steps():
    data = {"m": make_mouse(1, pick=0)}
    built = build_windows_with_magnitudes(data, pre=10, post=6)
    plain = get_good_reversal_info(
        data, pre=10, post=6, include_first_block=False, stop_at_block_changes=True
    )
    expected = add_magnitude_windows(plain, data)
    recorded = {
        s: [{**r, "window_pre": 10, "window_post": 6} for r in v] for s, v in expected.items()
    }
    assert built == recorded
    assert all(not rev["first_block"] for rev in built["m"])


def test_one_problem_gives_pooled_figures_for_every_scheme_and_non_empty_type(tmp_path, real_files):
    result = run(one_problem(), tmp_path)
    names = sorted(p.name for p in tmp_path.iterdir())
    expected = [
        f"reversal_aligned_arms_pooled_{scheme}_{kind}.{ext}"
        for scheme in ("new", "prev")
        for kind in ("all", "second_to_third")
        for ext in ("pdf", "png")
    ]
    assert names == sorted([*expected, "reversal_aligned_arms_counts.csv"])
    assert list(result["results"]) == ["pooled"]
    assert set(result["results"]["pooled"]) == {"prev", "new"}
    assert set(result["results"]["pooled"]["prev"]) == {"all", "best_to_third", "second_to_third"}
    assert len(result["figure_paths"]) == 8


def test_the_counts_table_is_returned_and_saved_and_says_what_was_empty(tmp_path):
    result = run(one_problem(), tmp_path)
    counts = result["counts"]
    saved = pd.read_csv(tmp_path / "reversal_aligned_arms_counts.csv")
    assert len(counts) == len(saved) == 6
    row = counts.query("label_scheme == 'prev' and reversal_type == 'best_to_third'").iloc[0]
    assert row["n_reversals"] == 0 and row["n_candidates"] == 2 and row["n_wrong_type"] == 2
    assert pd.isna(row["figure"]) or row["figure"] == ""
    row = counts.query("label_scheme == 'new' and reversal_type == 'all'").iloc[0]
    assert row["n_reversals"] == 2 and row["n_subjects"] == 1


def test_several_problems_add_per_problem_figures_and_pool_per_subject(tmp_path, capture):
    result = run(two_problems(), tmp_path)
    assert list(result["results"]) == ["pooled", "problem-04", "problem-05"]
    pooled = result["results"]["pooled"]["prev"]["all"]
    per_problem = [result["results"][p]["prev"]["all"] for p in ("problem-04", "problem-05")]
    assert pooled.num_subjects == 2
    assert pooled.num_reversals == sum(r.num_reversals for r in per_problem) == 6
    assert "reversal_aligned_arms_problem-05_new_all.png" in capture.saved_names()


def test_trials_and_prebuilt_windows_give_the_same_curves(tmp_path):
    trials = two_problems()
    windows = {p: build_windows_with_magnitudes(t, pre=10, post=6) for p, t in trials.items()}
    from_trials = run(trials, tmp_path / "a")
    from_windows = run(windows, tmp_path / "b")
    for scope in from_trials["results"]:
        for scheme in ("prev", "new"):
            a = from_trials["results"][scope][scheme]["all"]
            b = from_windows["results"][scope][scheme]["all"]
            for role in a.roles:
                np.testing.assert_array_equal(a.choice_mean[role], b.choice_mean[role])
                np.testing.assert_array_equal(a.magnitude_mean[role], b.magnitude_mean[role])


def test_the_inputs_are_not_modified(tmp_path):
    trials = two_problems()
    windows = {p: build_windows_with_magnitudes(t, pre=10, post=6) for p, t in trials.items()}
    before_trials, before_windows = copy.deepcopy(trials), copy.deepcopy(windows)
    run(trials, tmp_path / "a")
    run(windows, tmp_path / "b")
    assert trials == before_trials and windows == before_windows


def test_moving_average_mode_and_window_reach_the_curves(tmp_path):
    centered = run(one_problem(), tmp_path / "c", moving_avg_window=3)
    trailing = run(one_problem(), tmp_path / "t", moving_avg_window=3, moving_avg_mode="trailing")
    a = centered["results"]["pooled"]["prev"]["all"]
    b = trailing["results"]["pooled"]["prev"]["all"]
    assert a.moving_avg_mode == "centered" and b.moving_avg_mode == "trailing"
    assert a.moving_avg_window == b.moving_avg_window == 3
    assert not np.allclose(a.choice_mean["prev_best"], b.choice_mean["prev_best"], equal_nan=True)


def test_a_subset_of_schemes_and_types(tmp_path, real_files):
    result = run_reversal_aligned_arm_analysis(
        one_problem(),
        tmp_path,
        **KWARGS,
        label_schemes=("new",),
        reversal_types=("all",),
    )
    assert list(result["results"]["pooled"]) == ["new"]
    assert list(result["results"]["pooled"]["new"]) == ["all"]
    assert sorted(p.name for p in tmp_path.glob("*.png")) == [
        "reversal_aligned_arms_pooled_new_all.png"
    ]


@pytest.mark.parametrize("out_dir", [None, ""])
def test_no_out_dir_shows_the_figures_and_writes_nothing(monkeypatch, tmp_path, capture, out_dir):
    monkeypatch.chdir(tmp_path)
    result = run(one_problem(), out_dir)
    assert len(capture.shown) == 4 and capture.saved == []
    assert list(tmp_path.iterdir()) == []
    assert result["figure_paths"] == []


def test_unknown_scheme_type_or_mode_fail_before_any_work(tmp_path):
    with pytest.raises(ValueError, match="label scheme"):
        run_reversal_aligned_arm_analysis({}, tmp_path, label_schemes=("old",))
    with pytest.raises(ValueError, match="reversal_type"):
        run_reversal_aligned_arm_analysis({}, tmp_path, reversal_types=("third_to_best",))
    with pytest.raises(ValueError, match="moving_avg_mode"):
        run_reversal_aligned_arm_analysis({}, tmp_path, moving_avg_mode="causal")


def test_windows_without_magnitudes_are_rejected(tmp_path):
    bare = {1: get_good_reversal_info({"m": make_mouse(1)}, pre=10, post=6)}
    with pytest.raises(ValueError, match="build_windows_with_magnitudes"):
        run_reversal_aligned_arm_analysis(bare, tmp_path, **KWARGS)


def test_mixing_trials_and_windows_is_rejected(tmp_path):
    windows = build_windows_with_magnitudes({"m": make_mouse(1)}, pre=10, post=6)
    mixed = {1: windows, 2: {"m": make_mouse(1)}}
    with pytest.raises(ValueError, match="mix"):
        run_reversal_aligned_arm_analysis(mixed, tmp_path, **KWARGS)


def test_no_problems_raise(tmp_path):
    with pytest.raises(ValueError, match="no problems"):
        run_reversal_aligned_arm_analysis({}, tmp_path, **KWARGS)


def test_windows_built_with_other_pre_or_post_are_rejected(tmp_path):
    small = {1: build_windows_with_magnitudes({"m": make_mouse(1)}, pre=3, post=4)}
    with pytest.raises(ValueError, match="rebuild"):
        run_reversal_aligned_arm_analysis(small, tmp_path, **KWARGS)


def test_moving_avg_window_is_checked_before_any_work(tmp_path):
    with pytest.raises(ValueError, match="moving_avg_window"):
        run_reversal_aligned_arm_analysis({}, tmp_path, moving_avg_window=0)


def test_numpy_integer_problem_numbers_name_the_scopes(tmp_path):
    trials = {np.int64(4): {"m": make_mouse(1)}, np.int64(5): {"m": make_mouse(1, first_day=11)}}
    result = run(trials, tmp_path)
    assert list(result["results"]) == ["pooled", "problem-04", "problem-05"]
