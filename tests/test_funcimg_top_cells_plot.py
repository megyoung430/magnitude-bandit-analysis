import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_hex, to_rgba  # noqa: E402

from src.funcimg_visualization.plot_funcimg_responses import (  # noqa: E402
    plot_top_cells,
    roi_colors,
    trace_baselines,
)

TAB10 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]  # fmt: skip
N_FRAMES, N_ROIS = 300, 12


def synthetic_top(n_rois=N_ROIS, with_big_transient=True):
    """Noisy ΔF/F0 traces of ROIs 5, 6, ... (largest first), one with a large transient."""
    rng = np.random.default_rng(0)
    dff = rng.normal(0, 0.2, (n_rois, N_FRAMES))
    if with_big_transient:
        dff[3, 100:110] += 8.0
    cells = np.arange(n_rois) + 5
    labels = np.zeros((40, 5 * n_rois), np.int32)
    for i, cell in enumerate(cells):
        labels[5:15, 5 * i : 5 * i + 4] = cell + 1
    time_s = 10.0 + 0.133 * np.arange(N_FRAMES)
    outcomes = pd.DataFrame({"outcome_time": time_s[[50, 150, 250]], "num_rew_pulses": [0, 2, 4]})
    return {
        "cells": cells,
        "peaks": np.nanmax(dff, axis=1),
        "dff": dff,
        "time_s": time_s,
        "period_s": 0.133,
        "labels": labels,
        "mean_image": rng.normal(100, 5, labels.shape),
        "outcomes": outcomes,
    }


@pytest.fixture
def figure():
    fig = plot_top_cells(synthetic_top(), x_units="s", title="ses run", variant="with_destriping")
    yield fig
    plt.close(fig)


def trace_lines(fig, n=N_ROIS):
    return fig.axes[0].lines[:n]


def test_colours_cycle_through_tab10_in_rank_order():
    assert roi_colors(12) == [TAB10[i % 10] for i in range(12)]


def test_each_roi_has_its_own_colour_and_a_thin_line(figure):
    lines = trace_lines(figure)
    assert [to_hex(line.get_color()) for line in lines] == roi_colors(N_ROIS)
    assert all(0.6 <= line.get_linewidth() <= 1.0 for line in lines)


def test_fov_outlines_and_labels_use_the_trace_colour_of_their_roi(figure):
    ax_fov = figure.axes[1]
    top = synthetic_top()
    overlay = ax_fov.images[1].get_array()
    for rank, cell in enumerate(top["cells"]):
        ys, xs = np.nonzero(top["labels"] == cell + 1)
        pixel = overlay[ys[0], xs[0]]  # the corner pixel of the ROI is on its outline
        assert tuple(pixel) == pytest.approx(to_rgba(roi_colors(N_ROIS)[rank]))
    texts = {t.get_text(): to_hex(t.get_color()) for t in ax_fov.texts}
    assert texts == {str(c): roi_colors(N_ROIS)[i] for i, c in enumerate(top["cells"])}


def test_y_ticks_are_roi_labels_at_each_baseline_with_the_best_roi_on_top(figure):
    ax = figure.axes[0]
    top = synthetic_top()
    labels = [t.get_text() for t in ax.get_yticklabels()]
    order = np.argsort(-ax.get_yticks())  # top of the axis first
    assert [labels[i] for i in order] == [f"ROI {c}" for c in top["cells"]]
    assert ax.get_ylabel() == "ROI"
    assert not ax.texts


def test_only_left_and_bottom_spines_are_drawn(figure):
    spines = figure.axes[0].spines
    assert [name for name, s in spines.items() if s.get_visible()] == ["left", "bottom"]


def test_traces_do_not_overlap_even_with_one_large_transient():
    dff = synthetic_top()["dff"]
    baselines = trace_baselines(dff)
    shifted = baselines[:, None] + dff
    assert np.all(np.diff(baselines) < 0)  # rank 0 first, each lower than the one before
    assert np.all(shifted[1:].max(axis=1) < shifted[:-1].min(axis=1))


def test_spacing_is_set_by_the_largest_traces_not_by_noise():
    quiet = trace_baselines(synthetic_top(with_big_transient=False)["dff"])
    big = trace_baselines(synthetic_top()["dff"])
    assert abs(big[1]) > abs(quiet[1]) * 3


@pytest.mark.parametrize(("x_units", "label"), [("s", "pyControl time (s)"), ("frames", "Frames")])
def test_x_label_names_the_units(x_units, label):
    fig = plot_top_cells(synthetic_top(), x_units=x_units)
    assert fig.axes[0].get_xlabel() == label
    plt.close(fig)


def test_titles_name_the_variant_and_count_and_the_run(figure):
    assert figure.axes[0].get_title() == f"with_destriping green: top {N_ROIS} ROIs by maximum dF/F"
    assert figure.get_suptitle() == "ses run"


def test_outcome_markers_sit_above_the_traces_and_a_legend_lists_the_pulses(figure):
    ax = figure.axes[0]
    markers = ax.lines[N_ROIS : N_ROIS + 3]
    top_of_traces = max(line.get_ydata().max() for line in trace_lines(figure))
    assert all(min(m.get_ydata()) >= top_of_traces for m in markers)
    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend == ["0 pulses", "2 pulses", "4 pulses"]
