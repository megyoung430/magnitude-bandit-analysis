"""File names, formats, dpi and the empty-path-means-show rule of the multi-figure wrappers."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from test_reversal_boundaries import session, snapshot

from src.behavior_import.reconcile_reversals import reconcile_reversals
from src.behavior_visualization.plot_averaged_choice_probs import (
    plot_choice_probs_with_fraction_removed_axis,
)
from src.behavior_visualization.plot_helpers import figure_caption
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
SVG = {"bbox_inches": "tight"}


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


def test_reversals_by_trial_saves_pdf_and_png_at_180_dpi_and_svg_for_each_figure(
    capture, reconciled, tmp_path
):
    plot_reversals_by_trial(reconciled, save_path=str(tmp_path / "Rev"))
    assert saved(capture) == [
        ("Rev Across Mice.pdf", BOTH_180),
        ("Rev Across Mice.png", BOTH_180),
        ("Rev Across Mice.svg", SVG),
        ("Rev By Mouse.pdf", BOTH_180),
        ("Rev By Mouse.png", BOTH_180),
        ("Rev By Mouse.svg", SVG),
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


def test_num_reversals_saves_pdf_png_at_300_dpi_and_svg(capture, tmp_path):
    plot_num_reversals({"m": make_mouse(2)}, save_path=str(tmp_path / "Rev"))
    assert saved(capture) == [("Rev.pdf", PDF), ("Rev.png", PNG_300), ("Rev.svg", SVG)]


def test_num_reversals_over_time_names_its_figures_and_trial_companions(
    capture, reconciled, tmp_path
):
    plot_num_reversals_over_time(reconciled, save_path=str(tmp_path / "Rev"))
    assert saved(capture) == [
        ("Rev Across Mice.pdf", PDF),
        ("Rev Across Mice.png", PNG_300),
        ("Rev Across Mice.svg", SVG),
        ("Rev By Mouse.pdf", PDF),
        ("Rev By Mouse.png", PNG_300),
        ("Rev By Mouse.svg", SVG),
        ("Rev (By Trial) Across Mice.pdf", BOTH_180),
        ("Rev (By Trial) Across Mice.png", BOTH_180),
        ("Rev (By Trial) Across Mice.svg", SVG),
        ("Rev (By Trial) By Mouse.pdf", BOTH_180),
        ("Rev (By Trial) By Mouse.png", BOTH_180),
        ("Rev (By Trial) By Mouse.svg", SVG),
    ]


def test_moving_average_names_its_trial_companions_with_the_window(capture, reconciled, tmp_path):
    plot_moving_avg_reversals_over_time(
        reconciled, window=2, trial_window=2, save_path=str(tmp_path / "Mov")
    )
    assert capture.saved_names() == [
        "Mov Across Mice.pdf",
        "Mov Across Mice.png",
        "Mov Across Mice.svg",
        "Mov By Mouse.pdf",
        "Mov By Mouse.png",
        "Mov By Mouse.svg",
        "Mov (By Trial, 2-Trial Window) Across Mice.pdf",
        "Mov (By Trial, 2-Trial Window) Across Mice.png",
        "Mov (By Trial, 2-Trial Window) Across Mice.svg",
        "Mov (By Trial, 2-Trial Window) By Mouse.pdf",
        "Mov (By Trial, 2-Trial Window) By Mouse.png",
        "Mov (By Trial, 2-Trial Window) By Mouse.svg",
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


def test_reversal_curves_across_mice_state_the_mean_and_n_and_caption_the_rest(
    capture, reconciled, tmp_path
):
    plot_moving_avg_reversals_over_time(
        reconciled, window=2, trial_window=2, save_path=str(tmp_path / "Mov")
    )
    figures = {Path(s.path).name: s.figure for s in capture.saved}
    across = figures["Mov Across Mice.png"].axes[0]
    assert (
        across.get_title()
        == "Moving-Average Good and Bad Reversals Over Time\nmean across mice, n = 2 mice"
    )
    caption = figure_caption(across.figure)
    assert "Centred moving average over 2 sessions." in caption
    assert "Sessions per mouse: 1–2." in caption
    by_mouse = figures["Mov By Mouse.png"]
    assert [ax.get_title() for ax in by_mouse.axes if ax.get_visible() and ax.axison] == [
        "m1",
        "m2",
    ]
    assert "Sessions per mouse: m1 2, m2 1." in figure_caption(by_mouse)
    trials = figures["Mov (By Trial, 2-Trial Window) Across Mice.png"].axes[0]
    assert trials.get_title() == "Trailing 2-trial reversal rate\nmean across mice, n = 2 mice"
