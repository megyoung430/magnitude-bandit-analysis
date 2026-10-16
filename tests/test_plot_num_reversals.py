import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.behavior_visualization.plot_num_reversals import (
    THRESHOLD_STYLE,
    plot_num_reversals,
    plot_reversal_curves_across_mice,
    plot_reversal_curves_for_mouse,
    plot_reversal_totals,
)
from tests.plot_capture import capture_figures
from tests.plot_fixtures import make_mouse


def test_panels_draw_into_given_axes():
    fig, (left, right) = plt.subplots(1, 2)
    curves = {"good": {"a": np.array([0.0, 1, 2]), "b": np.array([1.0])}}
    assert plot_reversal_curves_across_mice(curves, ax=left, threshold=2) is left
    assert (
        plot_reversal_curves_for_mouse(
            {"total": np.array([1.0, 2])}, ["ses-01", "ses-02"], ax=right
        )
        is right
    )
    assert len(fig.axes) == 2
    mean_line = [ln for ln in left.lines if ln.get_label() == "Good Rev"][0]
    np.testing.assert_allclose(mean_line.get_ydata(), [0.5, 1.0, 1.5])
    plt.close(fig)


def threshold_lines(ax):
    return [ln for ln in ax.lines if ln.get_label() == "Threshold"]


def test_total_bars_mark_the_threshold_in_the_shared_style():
    totals = pd.DataFrame({"total": [4, 12]}, index=["a", "b"])
    fig, ax = plt.subplots()
    plot_reversal_totals(totals, ("total",), ax=ax, threshold=10)
    (line,) = threshold_lines(ax)
    np.testing.assert_allclose(line.get_ydata(), [10, 10])
    assert line.get_color() == THRESHOLD_STYLE["color"]
    assert line.get_linestyle() == THRESHOLD_STYLE["linestyle"]
    assert "Threshold" in [t.get_text() for t in ax.get_legend().get_texts()]
    plt.close(fig)


def test_threshold_line_is_a_visible_mid_grey():
    rgb = mcolors.to_rgb(THRESHOLD_STYLE["color"])
    assert max(rgb) < 0.7 and np.ptp(rgb) == 0  # grey, and clearly darker than #DCDCDC
    assert THRESHOLD_STYLE.get("alpha", 1.0) == 1.0


@pytest.mark.parametrize("totals", [[4, 12], [3, 5], [10, 10]], ids=["above", "below", "at"])
def test_legend_leaves_the_threshold_line_uncovered(totals):
    df = pd.DataFrame({"total": totals}, index=["a", "b"])
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_reversal_totals(df, ("total",), ax=ax, threshold=10)
    fig.canvas.draw()
    legend_box = ax.get_legend().get_window_extent().transformed(ax.transData.inverted())
    assert legend_box.y0 > 10
    plt.close(fig)


def test_total_bars_have_no_line_without_a_threshold():
    fig, ax = plt.subplots()
    plot_reversal_totals(pd.DataFrame({"total": [4]}, index=["a"]), ("total",), ax=ax)
    assert threshold_lines(ax) == [] and ax.get_legend() is None
    plt.close(fig)


def test_drifting_total_reversals_figure_has_the_10_reversal_line(monkeypatch):
    capture = capture_figures(monkeypatch)
    plot_num_reversals({"m": make_mouse(2, counters=False)}, save_path="")
    (ax,) = capture.shown[0][0].axes
    (line,) = threshold_lines(ax)
    assert line.get_ydata()[0] == 10


def test_blocked_total_reversals_figure_is_unchanged(monkeypatch):
    capture = capture_figures(monkeypatch)
    plot_num_reversals({"m": make_mouse(2, counters=True)}, save_path="")
    (ax,) = capture.shown[0][0].axes
    assert threshold_lines(ax) == []
