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
    top_cells_pdf_name,
    trace_baselines,
)

TAB10 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]  # fmt: skip
N_FRAMES, N_ROIS = 300, 12


def synthetic_top(n_rois=N_ROIS, with_big_transient=True, rank_by="max_dff"):
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
        "p_cell": np.linspace(0.99, 0.5, n_rois),
        "rank_by": rank_by,
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
    expected = [f"ROI {c} (p={p:.2f})" for c, p in zip(top["cells"], top["p_cell"], strict=True)]
    assert [labels[i] for i in order] == expected
    assert labels[order[0]] == "ROI 5 (p=0.99)"
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
    fig = plot_top_cells(synthetic_top(rank_by="p_cell"), variant="with_destriping")
    assert fig.axes[0].get_title() == f"with_destriping green: top {N_ROIS} ROIs by p(cell)"
    plt.close(fig)
    assert figure.get_suptitle() == "ses run"


def test_outcome_markers_sit_above_the_traces_and_a_legend_lists_the_pulses(figure):
    ax = figure.axes[0]
    markers = ax.lines[N_ROIS : N_ROIS + 3]
    top_of_traces = max(line.get_ydata().max() for line in trace_lines(figure))
    assert all(min(m.get_ydata()) >= top_of_traces for m in markers)
    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend == ["0 pulses", "2 pulses", "4 pulses"]


def test_each_ranking_has_its_own_pdf_name():
    args = ("20-20", "with_destriping", "s")
    by_p = top_cells_pdf_name(30, "p_cell", *args)
    by_max = top_cells_pdf_name(30, "max_dff", *args)
    assert by_p == "top30_cells_by_p_cell_20-20_with_destriping_s.pdf"
    assert by_max == "top30_cells_by_max_dff_20-20_with_destriping_s.pdf"
