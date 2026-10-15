"""File names, formats, dpi and the empty-path-means-show rule of the multi-figure wrappers."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from test_reversal_boundaries import session, snapshot

from src.behavior_import.reconcile_reversals import reconcile_reversals
from src.behavior_visualization.plot_averaged_choice_probs import (
    plot_choice_probs_with_fraction_removed_axis,
)
from src.behavior_visualization.plot_num_reversals import (
    plot_moving_avg_reversals_over_time,
    plot_num_reversals,
    plot_num_reversals_over_time,
)
from src.behavior_visualization.plot_problem_summary import plot_metric_by_problem
from src.behavior_visualization.plot_reversals_by_trial import plot_reversals_by_trial
from tests.plot_capture import capture_figures
from tests.plot_fixtures import make_mouse

PDF = {"bbox_inches": "tight"}
PNG_300 = {"dpi": 300, "bbox_inches": "tight"}
BOTH_180 = {"dpi": 180, "bbox_inches": "tight"}


@pytest.fixture
def capture(monkeypatch):
    return capture_figures(monkeypatch)


@pytest.fixture
def reconciled():
    data = {
        "m1": {
            "ses-1": session(snapshot(), snapshot(bad=1, tb=1), good=(0, 1, 1), bad=(0, 0, 1)),
            "ses-2": session(snapshot(tb=1), snapshot(tb=1), good=(0, 0), bad=(1, 1)),
        },
        "m2": {
            "ses-1": session(snapshot(), snapshot(good=1, tg=1), good=(0, 0, 1, 1), bad=(0,) * 4)
        },
    }
    reconcile_reversals(data)
    return data


def saved(capture):
    return [(s.path.rsplit("/", 1)[-1], s.kwargs) for s in capture.saved]


def test_reversals_by_trial_saves_png_then_pdf_at_180_dpi_for_each_figure(
    capture, reconciled, tmp_path
):
    plot_reversals_by_trial(reconciled, save_path=str(tmp_path / "Rev"))
    assert saved(capture) == [
        ("Rev Across Mice.png", BOTH_180),
        ("Rev Across Mice.pdf", BOTH_180),
        ("Rev By Mouse.png", BOTH_180),
        ("Rev By Mouse.pdf", BOTH_180),
    ]
    assert capture.shown == [] and plt.get_fignums() == []


def test_reversals_by_trial_shows_both_figures_once_for_a_missing_or_empty_path(
    capture, reconciled, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    for path in (None, ""):
        plot_reversals_by_trial(reconciled, save_path=path)
    assert [len(group) for group in capture.shown] == [2, 2]
    assert capture.saved == [] and list(tmp_path.iterdir()) == [] and plt.get_fignums() == []


def test_num_reversals_saves_pdf_then_png_at_300_dpi(capture, tmp_path):
    plot_num_reversals({"m": make_mouse(2)}, save_path=str(tmp_path / "Rev"))
    assert saved(capture) == [("Rev.pdf", PDF), ("Rev.png", PNG_300)]


def test_num_reversals_over_time_names_its_figures_and_trial_companions(
    capture, reconciled, tmp_path
):
    plot_num_reversals_over_time(reconciled, save_path=str(tmp_path / "Rev"))
    assert saved(capture) == [
        ("Rev Across Mice.pdf", PDF),
        ("Rev Across Mice.png", PNG_300),
        ("Rev By Mouse.pdf", PDF),
        ("Rev By Mouse.png", PNG_300),
        ("Rev (By Trial) Across Mice.png", BOTH_180),
        ("Rev (By Trial) Across Mice.pdf", BOTH_180),
        ("Rev (By Trial) By Mouse.png", BOTH_180),
        ("Rev (By Trial) By Mouse.pdf", BOTH_180),
    ]


def test_moving_average_names_its_trial_companions_with_the_window(capture, reconciled, tmp_path):
    plot_moving_avg_reversals_over_time(
        reconciled, window=2, trial_window=2, save_path=str(tmp_path / "Mov")
    )
    assert capture.saved_names() == [
        "Mov Across Mice.pdf",
        "Mov Across Mice.png",
        "Mov By Mouse.pdf",
        "Mov By Mouse.png",
        "Mov (By Trial, 2-Trial Window) Across Mice.png",
        "Mov (By Trial, 2-Trial Window) Across Mice.pdf",
        "Mov (By Trial, 2-Trial Window) By Mouse.png",
        "Mov (By Trial, 2-Trial Window) By Mouse.pdf",
    ]


def test_an_empty_path_means_show_in_every_wrapper(capture, reconciled, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plot_num_reversals({"m": make_mouse(2)}, save_path="")
    plot_num_reversals_over_time(reconciled, save_path="")
    plot_moving_avg_reversals_over_time(reconciled, window=2, trial_window=2, save_path="")
    summary = dict(
        blocks=[1, 2], means=[1.0, 2.0], ses=[0.1, 0.2], ns=[2, 2],
        per_mouse_blocklens={"a": {1: 1.0, 2: 2.0}}, mice=["a"],
    )  # fmt: skip
    plot_metric_by_problem(summary, title="t", ylabel="y", save_path="")
    keys = ("prev_best", "next_best", "third")
    across = dict(
        mean={k: [0.3, 0.3, 0.4] for k in keys},
        se={k: [0.01] * 3 for k in keys},
        num_subjects=1,
        num_reversals=1,
        num_problems=1,
    )
    plot_choice_probs_with_fraction_removed_axis(
        [-1.0, 0.0, 1.0], across, [0.0, 0.1, 0.2], save_path=""
    )
    assert capture.saved == [] and list(tmp_path.iterdir()) == [] and plt.get_fignums() == []
    assert len(capture.shown) == 7  # 1 + 2 + 2 + 1 + 1
