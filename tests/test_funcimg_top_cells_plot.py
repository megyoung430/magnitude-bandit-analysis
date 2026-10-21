import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_hex, to_rgba  # noqa: E402

from src.funcimg_visualization.plot_funcimg_responses import (  # noqa: E402
    clip_traces,
    plot_top_cells,
    roi_colors,
    top_cells_pdf_name,
    trace_baselines,
    trace_spacing,
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


def long_traces(n=30, n_frames=3000, transient=8.0, seed=0):
    """Noisy ΔF/F0 traces, the fourth with one brief (10-frame) large transient."""
    dff = np.random.default_rng(seed).normal(0, 0.2, (n, n_frames))
    dff[3, 100:110] += transient
    return dff


def test_spacing_ignores_one_brief_transient_among_many_traces():
    quiet = trace_spacing(long_traces(transient=0.0))
    big = trace_spacing(long_traces(transient=8.0))
    assert big == pytest.approx(quiet, rel=0.15)  # not the transient (8)
    assert quiet < 2.5  # about one trace's own range, nowhere near the 8 of the transient


def test_the_bulk_of_each_trace_stays_within_its_own_row():
    dff = long_traces()
    baselines = trace_baselines(dff)
    shifted = baselines[:, None] + dff
    low, high = np.percentile(shifted, [1, 99], axis=1)
    assert np.all(high[1:] < low[:-1])
    assert np.all(np.diff(baselines) < 0)  # rank 0 first, each lower than the one before


def test_spacing_clears_an_asymmetric_pair():
    rng = np.random.default_rng(0)
    reaches_down = rng.uniform(-4.0, 0.1, 1000)  # the upper row
    reaches_up = rng.uniform(-0.1, 4.0, 1000)  # the lower row
    dff = np.vstack([reaches_down, reaches_up])
    needed = np.percentile(reaches_up, 99.5) - np.percentile(reaches_down, 0.5)
    assert trace_spacing(dff) >= needed  # about 8; each trace's own range is only 4.1


def test_flat_traces_get_a_nonzero_spacing():
    assert trace_spacing(np.zeros((3, 50))) > 0
    assert trace_spacing(np.zeros((1, 50))) > 0


def test_clipping_limits_each_trace_to_its_own_percentile_range():
    dff = long_traces(n=5)
    clipped = clip_traces(dff, (1, 99))
    low, high = np.percentile(dff, [1, 99], axis=1)
    assert np.allclose(clipped.min(axis=1), low) and np.allclose(clipped.max(axis=1), high)
    inside = (dff >= low[:, None]) & (dff <= high[:, None])
    assert np.array_equal(clipped[inside], dff[inside])  # values in range are untouched


def test_no_clipping_is_the_identity():
    dff = long_traces(n=5)
    assert clip_traces(dff, None) is dff


def test_clipped_rows_do_not_overlap_in_the_figure():
    top = synthetic_top(n_rois=30)
    top["dff"] = long_traces(n=30)[:, :N_FRAMES]
    top["peaks"] = top["dff"].max(axis=1)
    top["p_cell"] = np.linspace(0.99, 0.5, 30)
    fig = plot_top_cells(top, clip_percentiles=(1, 99))
    lines = fig.axes[0].lines[:30]
    ys = [line.get_ydata() for line in lines]
    assert all(
        np.all(lower.max() < upper.min() + 1e-9) for upper, lower in zip(ys, ys[1:], strict=False)
    )
    plt.close(fig)


def test_unclipped_figure_shows_the_raw_values():
    top = synthetic_top()
    fig = plot_top_cells(top, clip_percentiles=None)
    first = fig.axes[0].lines[0].get_ydata()
    assert np.allclose(first, top["dff"][0])  # row 0 has baseline 0
    assert "clipped" not in fig.axes[0].get_title()
    plt.close(fig)


def test_clipping_is_named_in_the_title_and_does_not_change_the_ranking_inputs(figure):
    assert "clipped to 1st–99th percentile per ROI" in figure.axes[0].get_title()


def test_no_top_cells_gives_an_empty_panel_with_a_note():
    top = synthetic_top()
    top.update(
        cells=np.array([], dtype=int),
        peaks=np.array([]),
        p_cell=np.array([]),
        dff=np.empty((0, N_FRAMES)),
    )
    fig = plot_top_cells(top, title="empty", variant="v")
    assert any("no ROIs" in t.get_text() for t in fig.axes[0].texts)
    plt.close(fig)


def test_a_top_roi_hidden_by_overlaps_is_noted_and_does_not_break_the_overlay():
    top = synthetic_top(n_rois=4)
    top["labels"] = np.where(top["labels"] == top["cells"][2] + 1, 0, top["labels"])
    top["cells"] = top["cells"].copy()
    fig = plot_top_cells(top)
    assert f"ROI {top['cells'][2]}" in fig.axes[1].get_title()
    plt.close(fig)


def test_a_top_roi_numbered_beyond_the_labels_does_not_break_the_overlay():
    top = synthetic_top(n_rois=4)
    top["cells"] = top["cells"] + 500  # no such labels in the image
    fig = plot_top_cells(top)
    plt.close(fig)


@pytest.mark.parametrize(("x_units", "label"), [("s", "pyControl time (s)"), ("frames", "Frames")])
def test_x_label_names_the_units(x_units, label):
    fig = plot_top_cells(synthetic_top(), x_units=x_units)
    assert fig.axes[0].get_xlabel() == label
    plt.close(fig)


def test_titles_name_the_variant_and_count_and_the_run(figure):
    first_line = figure.axes[0].get_title().split("\n")[0]
    assert first_line == f"with_destriping green: top {N_ROIS} ROIs by maximum dF/F"
    fig = plot_top_cells(synthetic_top(rank_by="p_cell"), variant="with_destriping")
    assert (
        fig.axes[0].get_title().startswith(f"with_destriping green: top {N_ROIS} ROIs by p(cell)")
    )
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
