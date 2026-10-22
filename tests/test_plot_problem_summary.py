"""Tests for the per-problem summary plots, through their public functions.

The figure functions close their figure before returning, so the ``drawn``
fixture records each figure as it is closed (still inspectable afterwards) along
with every save and show call.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from src.behavior_analysis import blocklen_distributions
from src.behavior_visualization import plot_problem_summary as pps
from src.behavior_visualization.plot_helpers import figure_caption

nan = float("nan")

# One-value panels have identical low/high limits; matplotlib widens them and warns.
pytestmark = pytest.mark.filterwarnings("ignore:Attempting to set identical low and high xlims")


class Drawn:
    def __init__(self):
        self.figures: list[Figure] = []
        self.saved: list[tuple[str, dict]] = []
        self.shows = 0


@pytest.fixture
def drawn(monkeypatch):
    record = Drawn()
    real_close, real_savefig = plt.close, Figure.savefig

    def close(fig=None):
        record.figures.append(fig)
        real_close(fig)

    def savefig(self, fname, **kwargs):
        record.saved.append((Path(fname).name, kwargs))
        real_savefig(self, fname, **kwargs)

    def show():
        record.shows += 1

    monkeypatch.setattr(plt, "close", close)
    monkeypatch.setattr(plt, "show", show)
    monkeypatch.setattr(Figure, "savefig", savefig)
    yield record
    real_close("all")


def summary():
    return {
        "blocks": [1, 2, 3],
        "means": [2.0, nan, 4.5],
        "ses": [0.5, nan, 0.0],
        "ns": [2, 0, 1],
        "per_mouse_blocklens": {"A": {1: 2.5, 3: 4.5}, "B": {1: 1.5}, "Z": {9: 1.0}},
        "mice": ["A", "B", "C", "Z"],
    }


def block_stats():
    return {
        1: {"block_stats": {"per_mouse_blocklens": {"A": {1: 10.0, 2: nan}, "B": {1: 20.0}}}},
        2: {"block_stats": {"per_mouse_blocklens": {"A": {1: 30.0, 2: 40.0}}}},
    }


def block_stats_three_problems():
    """A has problems 1-3; B has 1-2, so each mouse's last problem is a different one."""
    return {
        1: {"block_stats": {"per_mouse_blocklens": {"A": {1: 10.0}, "B": {1: 20.0}}}},
        2: {"block_stats": {"per_mouse_blocklens": {"A": {1: 30.0, 2: 40.0}, "B": {1: 25.0}}}},
        3: {"block_stats": {"per_mouse_blocklens": {"A": {1: 50.0}}}},
    }


def rank_counts():
    return {
        "A": [{"best_prop": 0.6, "second_prop": 0.3, "third_prop": 0.1}],
        "B": [{"best_prop": 0.8, "second_prop": 0.1, "third_prop": 0.1}],
    }


def visible(fig):
    return [ax for ax in fig.axes if ax.axison]


def title_pad_points(ax):
    # Reads matplotlib's private titleOffsetTrans: the public API has no getter for the pad.
    return ax.titleOffsetTrans.get_matrix()[1, 2] / ax.figure.dpi * 72


def mouse_lines(ax):
    return [ln for ln in ax.lines if ln.get_marker() == "o"]


def same_random_state(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]


# ---- per-problem bar charts ----


def test_metric_figure_text_scales_and_style(drawn):
    pps.plot_metric_by_problem(summary(), title="My title", ylabel="My Y", jitter=0)
    (fig,) = drawn.figures
    (ax,) = fig.axes
    assert tuple(fig.get_size_inches()) == (10, 4.8)
    assert ax.get_title() == "My title\nmean ± SE across mice, n = 4 mice"
    assert ax.get_ylabel() == "My Y"
    assert ax.get_xlabel() == "Problem Number"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["1", "2", "3"]
    assert [t.get_text() for t in ax.texts] == ["2.00", "4.50"]  # the NaN bar has no label
    assert title_pad_points(ax) == pytest.approx(18)
    # tallest bar + error = 4.5; labels default to max(top, 1) * 1.5; 2% headroom above that
    assert ax.get_ylim() == pytest.approx((0, 6.75 * 1.02))
    assert {t.get_position()[1] for t in ax.texts} == {6.75}
    legend = ax.get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["A (n=2)", "B (n=1)"]
    assert legend.get_texts()[0].get_fontsize() == 9


def test_metric_explicit_annotate_y_sets_label_height_and_limit(drawn):
    pps.plot_metric_by_problem(summary(), title="T", ylabel="Y", jitter=0, annotate_y=10.0)
    (ax,) = drawn.figures[0].axes
    assert {t.get_position()[1] for t in ax.texts} == {10.0}
    assert ax.get_ylim() == pytest.approx((0, 10.0 * 1.02))


def test_metric_all_nan_bars_fall_back_to_unit_top(drawn):
    all_nan = {**summary(), "means": [nan] * 3, "ses": [nan] * 3}
    pps.plot_metric_by_problem(all_nan, title="T", ylabel="Y")
    (ax,) = drawn.figures[0].axes
    assert len(ax.texts) == 0
    assert ax.get_ylim() == pytest.approx((0, 1.5 * 1.02))


def test_metric_with_nan_error_still_places_labels_and_limits_from_the_mean(drawn):
    nan_se = {**summary(), "ses": [0.5, nan, nan]}
    pps.plot_metric_by_problem(nan_se, title="T", ylabel="Y", jitter=0)
    (ax,) = drawn.figures[0].axes
    assert [t.get_text() for t in ax.texts] == ["2.00", "4.50"]
    # the tallest bar is 4.5 (no error bar to add); labels default to 1.5x that
    assert {t.get_position()[1] for t in ax.texts} == {6.75}


def test_lengths_figure_defaults_and_style(drawn):
    pps.plot_problem_lengths(summary(), jitter=0)
    (fig,) = drawn.figures
    (ax,) = fig.axes
    assert tuple(fig.get_size_inches()) == (10, 4.8)
    assert ax.get_ylabel() == "Sessions to Complete"
    assert ax.get_title() == "Sessions to Complete per Problem\nmean ± SE across mice, n = 4 mice"
    assert [t.get_text() for t in ax.texts] == ["2.0", "4.5"]
    assert title_pad_points(ax) == pytest.approx(20)
    assert ax.get_ylim() == pytest.approx((0, 6.75))  # no headroom, unlike the metric chart
    legend = ax.get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["A (n=2 problems)", "B (n=1 problems)"]
    assert legend.get_texts()[0].get_fontsize() == 10


def test_lengths_custom_ylabel_title_and_annotate_y(drawn):
    pps.plot_problem_lengths(
        summary(), ylabel="Mean Block Length", title="Custom", jitter=0, annotate_y=9.0
    )
    (ax,) = drawn.figures[0].axes
    assert ax.get_ylabel() == "Mean Block Length"
    assert ax.get_title() == "Custom\nmean ± SE across mice, n = 4 mice"
    assert ax.get_ylim() == pytest.approx((0, 9.0))


def test_lengths_default_title_follows_ylabel(drawn):
    pps.plot_problem_lengths(summary(), ylabel="Mean Block Length")
    assert drawn.figures[0].axes[0].get_title().startswith("Mean Block Length per Problem\n")


def test_mouse_lines_are_placed_at_their_problem_index(drawn):
    pps.plot_metric_by_problem(summary(), title="T", ylabel="Y", jitter=0)
    lines = mouse_lines(drawn.figures[0].axes[0])
    # A has problems 1 and 3 (indices 0, 2); B problem 1; C has no data and Z's problem is absent
    assert [list(ln.get_xdata()) for ln in lines] == [[0, 2], [0]]
    assert [list(ln.get_ydata()) for ln in lines] == [[2.5, 4.5], [1.5]]


@pytest.mark.parametrize("figure_fn", ["plot_metric_by_problem", "plot_problem_lengths"])
def test_value_labels_clear_bars_below_one_at_the_1_5_floor(drawn, figure_fn):
    # tops are 0.3 < 1, so the label height is the floor of 1 * 1.5, not 1.5 * top
    low = {**summary(), "means": [0.1, 0.2, 0.3], "ses": [0.0, 0.0, 0.0]}
    kwargs = {"title": "T", "ylabel": "Y"} if figure_fn == "plot_metric_by_problem" else {}
    getattr(pps, figure_fn)(low, **kwargs)
    (ax,) = drawn.figures[0].axes
    headroom = 1.02 if figure_fn == "plot_metric_by_problem" else 1.0
    assert {t.get_position()[1] for t in ax.texts} == {1.5}
    assert ax.get_ylim()[1] == pytest.approx(1.5 * headroom)


@pytest.mark.parametrize("figure_fn", ["plot_metric_by_problem", "plot_problem_lengths"])
def test_only_the_first_len_blocks_means_get_value_labels(drawn, figure_fn):
    # one block broadcasts over three means (stacked bars), but only one label is written
    one_block = {**summary(), "blocks": [1]}
    kwargs = {"title": "T", "ylabel": "Y"} if figure_fn == "plot_metric_by_problem" else {}
    getattr(pps, figure_fn)(one_block, **kwargs)
    (ax,) = drawn.figures[0].axes
    assert len(ax.patches) == 3 and len(ax.texts) == 1


def test_negative_means_are_clipped_by_the_zero_ylim_bottom_known_issue(drawn):
    negative = {**summary(), "means": [-2.0, -1.0, -3.0], "ses": [0.1, 0.1, 0.1]}
    pps.plot_metric_by_problem(negative, title="T", ylabel="Y")
    assert drawn.figures[0].axes[0].get_ylim()[0] == 0


@pytest.mark.parametrize("figure_fn", ["plot_metric_by_problem", "plot_problem_lengths"])
def test_missing_summary_key_raises_before_any_figure_is_made(drawn, figure_fn):
    broken = {k: v for k, v in summary().items() if k != "ns"}
    kwargs = {"title": "T", "ylabel": "Y"} if figure_fn == "plot_metric_by_problem" else {}
    with pytest.raises(KeyError, match="summary is missing 'ns'"):
        getattr(pps, figure_fn)(broken, **kwargs)
    assert plt.get_fignums() == [] and drawn.figures == []


@pytest.mark.parametrize("figure_fn", ["plot_metric_by_problem", "plot_problem_lengths"])
def test_non_numeric_means_are_reported_before_a_missing_ns(drawn, figure_fn):
    broken = {k: v for k, v in summary().items() if k != "ns"}
    kwargs = {"title": "T", "ylabel": "Y"} if figure_fn == "plot_metric_by_problem" else {}
    with pytest.raises(ValueError, match="could not convert string to float"):
        getattr(pps, figure_fn)({**broken, "means": ["a", "b", "c"]}, **kwargs)
    assert plt.get_fignums() == [] and drawn.figures == []


def test_unusable_summary_values_raise_before_any_figure_is_made(drawn):
    with pytest.raises(ValueError):
        pps.plot_metric_by_problem({**summary(), "means": ["a", "b", "c"]}, title="T", ylabel="Y")
    with pytest.raises(TypeError):  # a mouse label that cannot key the colour map
        pps.plot_problem_lengths({**summary(), "mice": [["A"]]})
    assert plt.get_fignums() == [] and drawn.figures == []


def test_jitter_draws_from_one_seeded_generator_in_mouse_order(drawn):
    np.random.seed(7)
    before = np.random.get_state()
    pps.plot_metric_by_problem(summary(), title="T", ylabel="Y", jitter=0.1)
    lines = mouse_lines(drawn.figures[0].axes[0])

    rng = np.random.default_rng(0)
    expected_a = np.array([0.0, 2.0]) + rng.uniform(-0.1, 0.1, size=2)
    expected_b = np.array([0.0]) + rng.uniform(-0.1, 0.1, size=1)
    np.testing.assert_array_equal(lines[0].get_xdata(), expected_a)
    np.testing.assert_array_equal(lines[1].get_xdata(), expected_b)
    assert same_random_state(before, np.random.get_state())  # the global stream is not used


def test_zero_jitter_leaves_the_random_stream_untouched(drawn):
    np.random.seed(7)
    before = np.random.get_state()
    pps.plot_metric_by_problem(summary(), title="T", ylabel="Y", jitter=0)
    pps.plot_problem_lengths(summary(), jitter=0)
    assert same_random_state(before, np.random.get_state())


def test_bar_panel_draws_into_the_axes_it_is_given(drawn):
    fig, (left, right) = plt.subplots(1, 2)
    out = pps.plot_problem_bars(summary(), ax=left, ylabel="Y", title="T", jitter=0)
    assert out is left
    assert len(fig.axes) == 2 and not right.lines and not right.patches
    assert len(left.patches) == 3
    assert drawn.figures == []  # the caller owns the figure


def test_bar_panel_makes_its_own_figure_only_without_ax(drawn):
    plt.close("all")
    ax = pps.plot_problem_bars(summary(), ylabel="Y", title="T")
    assert plt.get_fignums() == [ax.figure.number]


# ---- saving and showing ----


def test_metric_and_lengths_save_pdf_png_at_300_dpi_and_svg(drawn, tmp_path):
    pps.plot_metric_by_problem(summary(), title="T", ylabel="Y", save_path=tmp_path / "a" / "fig")
    pps.plot_problem_lengths(summary(), save_path=tmp_path / "len")
    assert [name for name, _ in drawn.saved] == [
        "fig.pdf",
        "fig.png",
        "fig.svg",
        "len.pdf",
        "len.png",
        "len.svg",
    ]
    assert [kw for _, kw in drawn.saved] == [
        {"bbox_inches": "tight"},
        {"bbox_inches": "tight", "dpi": 300},
        {"bbox_inches": "tight"},
    ] * 2
    assert drawn.shows == 0 and len(drawn.figures) == 2 and plt.get_fignums() == []
    assert (tmp_path / "a" / "fig.pdf").exists() and (tmp_path / "len.png").exists()


def test_without_a_save_path_the_figure_is_shown_then_closed(drawn):
    pps.plot_metric_by_problem(summary(), title="T", ylabel="Y")
    pps.plot_problem_lengths(summary())
    assert drawn.shows == 2 and drawn.saved == []
    assert len(drawn.figures) == 2 and plt.get_fignums() == []


# ---- block-length distributions ----


def test_pooled_distribution_panels_titles_labels_and_layout(drawn):
    pps.plot_distributions_per_problem_all_mice(block_stats())
    (fig,) = drawn.figures
    assert [ax.get_title() for ax in visible(fig)] == [
        "Problem 1 (blocks=2, mice=2)",
        "Problem 2 (blocks=2, mice=1)",
    ]
    assert len(fig.axes) == 4 and not fig.axes[3].axison  # default 4 columns, spare panels off
    assert tuple(fig.get_size_inches()) == pytest.approx((3.6 * 4, 2.8))
    assert sorted(t.get_text() for t in fig.texts) == [
        "Block Length Distributions — Pooled Across Mice",
        "Block length",
        "Counts",
    ]


def test_distribution_grid_wraps_rows_and_sizes_the_figure(drawn):
    five = {p: {"block_stats": {"per_mouse_blocklens": {"A": {1: float(p)}}}} for p in range(1, 6)}
    pps.plot_distributions_per_problem_all_mice(five, ncols=2)
    (fig,) = drawn.figures
    assert len(fig.axes) == 6 and len(visible(fig)) == 5
    assert tuple(fig.get_size_inches()) == pytest.approx((3.6 * 2, 2.8 * 3))


def test_pooled_panels_share_the_data_range_and_summarise_each_problem(drawn):
    pps.plot_distributions_per_problem_all_mice(block_stats())
    first, second = visible(drawn.figures[0])
    assert first.get_xlim() == second.get_xlim() == (10.0, 40.0)
    assert first.texts[0].get_text().startswith("n = 2\nmedian = 15.00")
    np.testing.assert_allclose(first.lines[0].get_xdata(), 15.0)  # mean line
    np.testing.assert_allclose(second.lines[0].get_xdata(), 35.0)


def test_sharex_and_sharey_link_the_panels(drawn):
    pps.plot_distributions_per_problem_all_mice(block_stats(), sharex=True, sharey=True)
    pps.plot_distributions_per_problem_all_mice(block_stats())
    shared, independent = drawn.figures
    a, b = visible(shared)
    assert a.get_shared_x_axes().joined(a, b) and a.get_shared_y_axes().joined(a, b)
    a, b = visible(independent)
    assert not a.get_shared_x_axes().joined(a, b) and not a.get_shared_y_axes().joined(a, b)


def test_per_mouse_figures_one_per_mouse_with_no_data_panels(drawn):
    pps.plot_distributions_per_mouse(block_stats(), ncols=2)
    fig_a, fig_b = drawn.figures
    assert sorted(t.get_text() for t in fig_a.texts) == [
        "Block Length Distributions — Mouse A",
        "Block length",
        "Count",
    ]
    assert [ax.get_title() for ax in visible(fig_a)] == ["Problem 1", "Problem 2"]
    assert [ax.get_title() for ax in visible(fig_b)] == ["Problem 1", "Problem 2"]
    assert [t.get_text() for t in fig_b.axes[1].texts] == ["no data"]  # B has nothing in problem 2
    assert tuple(fig_a.get_size_inches()) == pytest.approx((3.6 * 2, 2.8))


def test_per_mouse_panels_share_that_mouses_range(drawn):
    pps.plot_distributions_per_mouse(block_stats())
    fig_a = drawn.figures[0]
    assert all(ax.get_xlim() == (10.0, 40.0) for ax in visible(fig_a))


def test_per_mouse_saves_every_format_per_mouse_in_mouse_order(drawn, tmp_path):
    pps.plot_distributions_per_mouse(block_stats(), save_dir=tmp_path / "out")
    assert [name for name, _ in drawn.saved] == [
        f"blocklen_distributions_mouse_{m}.{ext}" for m in "AB" for ext in ("pdf", "png", "svg")
    ]
    assert [kw.get("dpi") for _, kw in drawn.saved] == [None, 300, None] * 2
    assert drawn.shows == 0 and plt.get_fignums() == []


def test_pooled_distribution_saves_pdf_png_and_svg(drawn, tmp_path):
    pps.plot_distributions_per_problem_all_mice(block_stats(), save_path=tmp_path / "d" / "pooled")
    assert [name for name, _ in drawn.saved] == ["pooled.pdf", "pooled.png", "pooled.svg"]
    assert drawn.saved[1][1] == {"bbox_inches": "tight", "dpi": 300}
    assert (tmp_path / "d" / "pooled.pdf").exists()


def test_distributions_are_shown_when_there_is_nothing_to_save(drawn):
    pps.plot_distributions_per_mouse(block_stats())
    pps.plot_distributions_per_problem_all_mice(block_stats())
    assert drawn.shows == 3 and len(drawn.figures) == 3 and plt.get_fignums() == []


def test_drop_last_problem_removes_each_mouses_last_problem_pooled(drawn):
    pps.plot_distributions_per_problem_all_mice(
        block_stats_three_problems(), drop_last_problem=True
    )
    (fig,) = drawn.figures
    # A loses problem 3 and B loses problem 2: problem 3 vanishes, problem 2 keeps only A
    assert [ax.get_title() for ax in visible(fig)] == [
        "Problem 1 (blocks=2, mice=2)",
        "Problem 2 (blocks=2, mice=1)",
    ]
    assert visible(fig)[1].texts[0].get_text().startswith("n = 2\nmedian = 35.00")


def test_drop_last_problem_removes_each_mouses_last_problem_per_mouse(drawn):
    pps.plot_distributions_per_mouse(block_stats_three_problems(), drop_last_problem=True)
    fig_a, fig_b = drawn.figures
    assert [ax.get_title() for ax in visible(fig_a)] == ["Problem 1", "Problem 2"]
    assert not any(t.get_text() == "no data" for ax in fig_a.axes for t in ax.texts)
    # B keeps only problem 1; its problem-2 panel is empty
    assert [t.get_text() for t in fig_b.axes[1].texts] == ["no data"]


def test_without_drop_last_problem_all_problems_are_drawn(drawn):
    pps.plot_distributions_per_problem_all_mice(block_stats_three_problems())
    assert len(visible(drawn.figures[0])) == 3


def test_pooled_distribution_with_nothing_left_after_dropping_raises_known_issue(drawn):
    # Characterises a known defect: the accidental matplotlib error below is what the
    # pooled plot raises, whereas the per-mouse plot (next test) silently draws nothing.
    one_problem = {1: {"block_stats": {"per_mouse_blocklens": {"A": {1: 5.0}}}}}
    with pytest.raises(ValueError, match="Number of rows must be a positive integer, not 0"):
        pps.plot_distributions_per_problem_all_mice(one_problem, drop_last_problem=True)
    assert drawn.figures == []


def test_per_mouse_distribution_with_nothing_left_after_dropping_draws_nothing(drawn):
    one_problem = {1: {"block_stats": {"per_mouse_blocklens": {"A": {1: 5.0}}}}}
    pps.plot_distributions_per_mouse(one_problem, drop_last_problem=True)
    assert drawn.figures == [] and drawn.shows == 0


def test_blocklen_histogram_panel_empty_says_no_data_and_ignores_xlim():
    _, ax = plt.subplots()
    pps.plot_blocklen_histogram(np.array([]), ax=ax, title="Problem 1", xlim=(0, 5))
    assert [t.get_text() for t in ax.texts] == ["no data"]
    assert ax.get_xlim() == (0.0, 1.0)  # the empty-axes default, not the (0, 5) asked for
    assert ax.get_title() == "Problem 1"


def test_blocklen_histogram_panel_draws_stats_and_xlim():
    _, ax = plt.subplots()
    pps.plot_blocklen_histogram(np.array([10.0, 20.0, 30.0]), ax=ax, title="P", xlim=(0, 50))
    assert ax.get_xlim() == (0, 50) and ax.get_title() == "P"
    assert ax.texts[0].get_text().startswith("n = 3")
    assert len(ax.lines) == 1  # mean line


def test_blocklen_histogram_panel_prints_median_sample_sd_and_draws_the_mean_line():
    _, ax = plt.subplots()
    pps.plot_blocklen_histogram(np.array([1.0, 2.0, 3.0, 10.0]), ax=ax, title="P")
    # median 2.5; sd with ddof=1 is sqrt(50 / 3) = 4.08 (ddof=0 would print 3.54)
    assert ax.texts[0].get_text() == "n = 4\nmedian = 2.50\nsd = 4.08\nmin = 1\nmax = 10"
    assert list(ax.lines[0].get_xdata()) == [4.0, 4.0]  # the mean, not the median


def test_blocklen_histogram_panel_prints_no_sd_for_a_single_value():
    # One value has no sample sd (as its standard error is NaN), so the line is left out
    # instead of printing 0.00.
    _, ax = plt.subplots()
    pps.plot_blocklen_histogram(np.array([7.0]), ax=ax, title="P")
    assert ax.texts[0].get_text() == "n = 1\nmedian = 7.00\nmin = 7\nmax = 7"


# ---- rank proportions ----


def test_rank_figure_text_scales_and_style(drawn):
    pps.plot_rank_proportions_average_without_blocks(rank_counts())
    (fig,) = drawn.figures
    (ax,) = fig.axes
    assert tuple(fig.get_size_inches()) == (7, 5.5)
    assert ax.get_title() == "Average Across Mice\nmean ± SE across mice, n = 2 mice"
    assert title_pad_points(ax) == pytest.approx(14)
    assert ax.get_ylabel() == "Proportion of Choices"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Best", "Second", "Third"]
    assert ax.get_ylim() == pytest.approx((0.0, 1.07))
    assert [p.get_height() for p in ax.patches] == pytest.approx([0.7, 0.2, 0.1])
    assert [t.get_text() for t in ax.texts] == ["0.70", "0.20", "0.10"]
    legend = ax.get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["A", "B"]
    assert legend.get_texts()[0].get_fontsize() == 10
    assert len(mouse_lines(ax)) == 2


def test_rank_figure_prints_the_available_pvalues(drawn):
    pps.plot_rank_proportions_average_without_blocks(
        rank_counts(), average_across_mice_pvalues={"best_vs_second": 0.02, "second_vs_third": 0.5}
    )
    texts = [t.get_text() for t in drawn.figures[0].axes[0].texts]
    assert "p(Best vs Second):\n0.020" in texts and "p(Second vs Third):\n0.500" in texts
    assert not any(t.startswith("p(Best vs Third)") for t in texts)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # nanmean of an all-NaN rank
def test_rank_pvalue_label_lands_on_the_wrong_bar_when_a_mean_is_nan_known_issue(drawn):
    # Characterises a known defect: labels anchor on the i-th finite bar, so with a NaN
    # "best" mean the "Best vs Second" label sits on the Second bar (x=1), not Best (x=0).
    no_best = {"A": [{"best_prop": nan, "second_prop": 0.6, "third_prop": 0.4}]}
    pps.plot_rank_proportions_average_without_blocks(
        no_best, average_across_mice_pvalues={"best_vs_second": 0.02}
    )
    ax = drawn.figures[0].axes[0]
    (label,) = [t for t in ax.texts if t.get_text().startswith("p(Best vs Second)")]
    assert label.get_position()[0] == 1


def test_rank_jitter_consumes_three_draws_per_mouse_in_order(drawn):
    np.random.seed(3)
    before = np.random.get_state()
    pps.plot_rank_proportions_average_without_blocks(rank_counts())
    lines = mouse_lines(drawn.figures[0].axes[0])

    rng = np.random.default_rng(0)
    for line in lines:
        expected = np.arange(3) + rng.uniform(-0.03, 0.03, size=3)
        np.testing.assert_array_equal(line.get_xdata(), expected)
    assert same_random_state(before, np.random.get_state())


def test_rank_figure_with_unusable_counts_raises_before_any_figure_is_made(drawn):
    with pytest.raises(AttributeError):
        pps.plot_rank_proportions_average_without_blocks([1, 2])
    assert plt.get_fignums() == [] and drawn.figures == []


def test_rank_figure_saves_pdf_png_and_svg(drawn, tmp_path):
    pps.plot_rank_proportions_average_without_blocks(rank_counts(), save_path=tmp_path / "r" / "x")
    assert [name for name, _ in drawn.saved] == ["x.pdf", "x.png", "x.svg"]
    assert drawn.saved[1][1] == {"bbox_inches": "tight", "dpi": 300}
    assert drawn.shows == 0 and plt.get_fignums() == []


def test_rank_panel_computes_its_bars_from_per_mouse_means(drawn):
    per_mouse = {
        "A": {"best": 0.6, "second": 0.3, "third": 0.1},
        "B": {"best": 0.8, "second": 0.1, "third": 0.1},
    }
    fig, ax = plt.subplots()
    out = pps.plot_rank_proportions_average_panel(
        per_mouse, ax=ax, pvalues={"best_vs_second": 0.02}, jitter=0
    )
    assert out is ax and len(fig.axes) == 1
    assert [p.get_height() for p in ax.patches] == pytest.approx([0.7, 0.2, 0.1])
    assert "p(Best vs Second):\n0.020" in [t.get_text() for t in ax.texts]
    np.testing.assert_array_equal(mouse_lines(ax)[0].get_xdata(), [0, 1, 2])  # no jitter


# ---- pvalue labels ----


def test_pvalue_annotations_anchor_on_finite_bars_and_skip_missing():
    out = pps.pvalue_annotations(
        {"best_vs_second": 0.01234, "second_vs_third": 0.5}, finite_x=[0, 1, 2]
    )
    assert out == [
        (0, 0.75, "p(Best vs Second):\n0.012"),
        (1, 0.5, "p(Second vs Third):\n0.500"),
    ]
    assert pps.pvalue_annotations(None, [0, 1, 2]) == []
    # best-vs-third needs three finite bars
    assert pps.pvalue_annotations({"best_vs_third": 0.1}, [0, 1]) == []


# ---- notebooks import the analysis helpers from here ----


def test_block_length_helpers_are_reexported():
    assert pps.collect_blocklens is blocklen_distributions.collect_blocklens
    assert (
        pps.drop_last_problem_per_mouse_raw
        is blocklen_distributions.drop_last_problem_per_mouse_raw
    )


def test_extra_lines_of_a_given_title_go_to_the_caption(drawn):
    pps.plot_metric_by_problem(summary(), title="T\n(cohort 3)", ylabel="Y")
    ax = drawn.figures[0].axes[0]
    assert ax.get_title() == "T\nmean ± SE across mice, n = 4 mice"
    assert figure_caption(ax.figure) == "(cohort 3)"
