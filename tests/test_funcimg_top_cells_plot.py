import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_hex, to_rgba  # noqa: E402

from src.funcimg_visualization.plot_funcimg_responses import (  # noqa: E402
    SPACING_FLOOR,
    clip_traces,
    middle_frames,
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
    expected = [
        f"ROI {c} (p={p:.2f}, max {peak:.3g})"
        for c, p, peak in zip(top["cells"], top["p_cell"], top["peaks"], strict=True)
    ]
    assert [labels[i] for i in order] == expected
    assert labels[order[0]].startswith("ROI 5 (p=0.99, max ")
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


def test_clipping_is_named_in_the_title(figure):
    assert "clipped to 1st–99th percentile per ROI" in figure.axes[0].get_title()


def test_clipping_leaves_the_input_and_the_rank_order_unchanged():
    top = synthetic_top()
    before = {key: np.copy(top[key]) for key in ("dff", "cells", "peaks")}
    labels = {}
    for clip in ((1, 99), None):
        fig = plot_top_cells(top, clip_percentiles=clip)
        ax = fig.axes[0]
        order = np.argsort(-ax.get_yticks())
        labels[clip] = [ax.get_yticklabels()[i].get_text() for i in order]
        plt.close(fig)
    assert all(np.array_equal(top[key], before[key]) for key in before)
    assert labels[(1, 99)] == labels[None]  # same ROIs in the same order, with the same max
    assert labels[None][0].startswith(f"ROI {top['cells'][0]} ")


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
    hidden = top["cells"][2]
    top["labels"] = np.where(top["labels"] == hidden + 1, 0, top["labels"])
    fig = plot_top_cells(top)
    ax_fov = fig.axes[1]
    note = ax_fov.get_title().split("\n")[1]
    assert note == f"not visible (hidden by overlaps): ROI {hidden}"
    assert {t.get_text() for t in ax_fov.texts} == {str(c) for c in top["cells"] if c != hidden}
    plt.close(fig)


def test_a_top_roi_numbered_beyond_the_labels_does_not_break_the_overlay():
    top = synthetic_top(n_rois=4)
    top["cells"] = top["cells"] + 500  # no such labels in the image
    fig = plot_top_cells(top)
    ax_fov = fig.axes[1]
    assert not ax_fov.texts  # no ROI has any pixel to number
    for cell in top["cells"]:
        assert f"ROI {cell}" in ax_fov.get_title()
    assert not ax_fov.images[1].get_array().any()  # nothing outlined
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


def test_middle_frames_is_a_centred_slice_or_everything():
    assert middle_frames(10_000, 8000) == slice(1000, 9000)
    assert middle_frames(10_001, 8000) == slice(1000, 9000)
    assert middle_frames(5000, 8000) == slice(0, 5000)
    assert middle_frames(5000, None) == slice(0, 5000)
    assert middle_frames(5000, np.int64(1000)) == slice(2000, 3000)


@pytest.mark.parametrize(("n_frames", "n_shown"), [(5000, 0), (5000, -3), (0, 0), (10, -1)])
def test_a_non_positive_middle_frame_count_is_refused_whatever_the_run_length(n_frames, n_shown):
    with pytest.raises(ValueError, match="positive"):
        middle_frames(n_frames, n_shown)


@pytest.mark.parametrize("n_shown", [True, 2.5, 8000.0, "100"])
def test_a_middle_frame_count_that_is_not_an_integer_is_refused(n_shown):
    with pytest.raises(TypeError, match="integer"):
        middle_frames(1, n_shown)  # True >= 1 would otherwise read as "everything"


@pytest.mark.parametrize(
    "clip",
    [(1,), (1, 2, 3), (99, 1), (5, 5), (-1, 50), (0, 101), (float("nan"), 99), (True, 99), "ab"],
)
def test_clip_percentiles_must_be_two_ordered_percentiles(clip):
    with pytest.raises(ValueError, match="clip_percentiles"):
        clip_traces(long_traces(n=4), clip)
    with pytest.raises(ValueError, match="clip_percentiles"):
        plot_top_cells(synthetic_top(), clip_percentiles=clip)


def test_clip_percentiles_may_span_the_whole_range_or_be_a_list():
    clip_traces(long_traces(n=4), (0, 100))
    clip_traces(long_traces(n=4), [0.5, 99.5])


def test_only_the_middle_frames_are_drawn_and_named_in_the_title():
    top = synthetic_top()
    shown = middle_frames(N_FRAMES, 100)
    fig = plot_top_cells(top, x_units="frames", middle_frames_shown=100)
    first = trace_lines(fig)[0]
    assert np.array_equal(first.get_xdata(), np.arange(N_FRAMES)[shown])
    assert f"middle 100 of {N_FRAMES} frames" in fig.axes[0].get_title()
    plt.close(fig)


def outcome_marker_positions(fig):
    """x of each outcome marker (a vertical segment); NaN means nothing is drawn."""
    return [
        line.get_xdata()[0]
        for line in fig.axes[0].lines
        if len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1]
    ]


def test_outcome_markers_outside_the_middle_frames_are_not_drawn():
    fig = plot_top_cells(synthetic_top(), x_units="frames", middle_frames_shown=100)
    drawn = [x for x in outcome_marker_positions(fig) if np.isfinite(x)]
    assert drawn == [150.0]  # the outcomes at frames 50 and 250 are outside frames 100-199
    plt.close(fig)


@pytest.mark.parametrize(("middle", "frames"), [(None, [50, 150, 250]), (100, [150])])
def test_outcome_markers_in_seconds_mode_are_filtered_to_the_frames_shown(middle, frames):
    top = synthetic_top()
    fig = plot_top_cells(top, x_units="s", middle_frames_shown=middle)
    drawn = [x for x in outcome_marker_positions(fig) if np.isfinite(x)]
    assert drawn == pytest.approx(list(top["time_s"][frames]))
    plt.close(fig)


def test_shown_frames_without_a_time_warn_and_are_named_in_the_title():
    top = synthetic_top()
    top["time_s"][100:200] = np.nan  # the middle 100 frames lie outside the sync pulses
    with pytest.warns(UserWarning, match="none of the frames shown has a pyControl time"):
        fig = plot_top_cells(top, x_units="s", middle_frames_shown=100)
    assert "no pyControl time in the frames shown" in fig.axes[0].get_title()
    plt.close(fig)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # frames mode needs no times
        plt.close(plot_top_cells(top, x_units="frames", middle_frames_shown=100))


def test_a_row_without_data_in_the_frames_shown_is_an_empty_labelled_row():
    top = synthetic_top()
    top["dff"][2, 100:200] = np.nan  # finite elsewhere, so the run ranked it
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no numpy all-NaN warnings either
        fig = plot_top_cells(top, x_units="frames", middle_frames_shown=100)
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    order = np.argsort(-ax.get_yticks())
    empty = [i for i, tick in enumerate(order) if "no data in shown frames" in labels[tick]]
    assert empty == [2]
    assert np.isnan(trace_lines(fig)[2].get_ydata()).all()
    assert np.isfinite(ax.get_ylim()).all()
    plt.close(fig)


@pytest.mark.parametrize("empty_rows", [[0], [N_ROIS - 1], list(range(N_ROIS))])
def test_empty_rows_at_the_ends_or_everywhere_still_give_a_finite_axis(empty_rows):
    top = synthetic_top()
    top["dff"][empty_rows, 100:200] = np.nan
    fig = plot_top_cells(top, x_units="frames", middle_frames_shown=100)
    ax = fig.axes[0]
    low, high = ax.get_ylim()
    assert np.isfinite([low, high]).all() and low < high
    assert low <= ax.get_yticks().min() and ax.get_yticks().max() <= high  # every label is visible
    plt.close(fig)


def test_spacing_ignores_rows_without_data_and_falls_back_to_the_floor():
    dff = long_traces(n=6)
    with_gap = dff.copy()
    with_gap[2] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert trace_spacing(with_gap) > SPACING_FLOOR
        assert trace_spacing(np.full((3, 50), np.nan)) == SPACING_FLOOR
        clipped = clip_traces(with_gap, (1, 99))
    assert np.isnan(clipped[2]).all()
    assert np.array_equal(clipped[0], clip_traces(dff, (1, 99))[0])


def test_the_tick_label_carries_the_true_max_that_clipping_flattens():
    top = synthetic_top(with_big_transient=False)
    top["dff"][3, 150] += 8.0  # one frame in 300: above the 99th percentile
    top["peaks"] = np.nanmax(top["dff"], axis=1)
    peak = top["peaks"][3]
    assert peak > 8
    fig = plot_top_cells(top, clip_percentiles=(1, 99))
    ax = fig.axes[0]
    assert np.ptp(trace_lines(fig)[3].get_ydata()) < peak / 4  # flattened to the 99th percentile
    expected = f"ROI {top['cells'][3]} (p={top['p_cell'][3]:.2f}, max {peak:.3g})"
    assert expected in [t.get_text() for t in ax.get_yticklabels()]
    plt.close(fig)


def peak_ticks(fig):
    return [line for line in fig.axes[0].lines if line.get_marker() == "|"]


@pytest.mark.parametrize(("x_units", "middle"), [("frames", None), ("frames", 100), ("s", 100)])
def test_each_roi_has_a_tick_in_its_colour_where_its_maximum_falls_if_shown(x_units, middle):
    top = synthetic_top()
    shown = middle_frames(N_FRAMES, middle)
    where = np.argmax(top["dff"], axis=1)
    x_all = np.arange(N_FRAMES) if x_units == "frames" else top["time_s"]
    colors = roi_colors(N_ROIS)
    expected = [(colors[i], x_all[w]) for i, w in enumerate(where) if shown.start <= w < shown.stop]
    assert 0 < len(expected) < N_ROIS or middle is None
    fig = plot_top_cells(top, x_units=x_units, middle_frames_shown=middle)
    got = [(to_hex(t.get_color()), t.get_xdata()[0]) for t in peak_ticks(fig)]
    assert [c for c, _ in got] == [c for c, _ in expected]
    assert [x for _, x in got] == pytest.approx([x for _, x in expected])
    plt.close(fig)
