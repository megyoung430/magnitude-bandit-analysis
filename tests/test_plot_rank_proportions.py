import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex, to_rgba

from src.behavior_visualization.plot_rank_proportions import (
    _MOUSE_PANEL_BAR_STYLE,
    BLOCK_COLORS,
    plot_average_across_blocks,
    plot_average_across_mice,
    plot_block_panel,
    plot_mouse_panel,
    plot_rank_proportions,
    plot_rank_proportions_average,
    plot_rank_proportions_by_block,
    plot_rank_proportions_by_mouse,
)
from src.behavior_visualization.plot_style import MOUSE_COLORS
from tests.rank_rows import RANKS, make_row

LABEL_Y = 1.05  # height at which the bar means are written


@pytest.fixture
def counts():
    return {
        "m1": [make_row(6, 3, 1), make_row(2, 2, 6), make_row(1, 1, 1)],
        "m2": [make_row(4, 4, 2), make_row(1, 2, 7), make_row(9, 1, 1), make_row(1, 1, 1)],
        "m3": [make_row(5, 5, 5), make_row(3, 3, 4), make_row(1, 1, 1)],
    }


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def shown_figures(monkeypatch):
    """Figures that were shown (shown figures are closed right after)."""
    figs = []
    monkeypatch.setattr(plt, "show", lambda: figs.append(plt.gcf()))
    return figs


# ---- what a drawn axis contains -------------------------------------------------


def data_lines(ax):
    # error-bar caps and stems are lines too; the data lines are the marked ones
    return [line for line in ax.lines if line.get_marker() == "o"]


def bar_heights(ax):
    return [p.get_height() for p in ax.patches]


def error_bar_lengths(ax):
    """Half-length of each error bar (the SE), read from the drawn error-bar caps."""
    # caps sit at mean - se and mean + se, and keep NaN errors as NaN (stem segments drop them)
    lower, upper = (cap.get_ydata() for cap in ax.containers[0].lines[1])
    return list((upper - lower) / 2)


def mean_labels(ax):
    """``[((x, y), text)]`` of the bar-mean labels, in drawing order."""
    return [(t.get_position(), t.get_text()) for t in ax.texts if t.get_position()[1] == LABEL_Y]


def pvalue_labels(ax):
    """``{text: (x, y)}`` of the p-value annotations."""
    return {t.get_text(): t.get_position() for t in ax.texts if t.get_position()[1] != LABEL_Y}


def tick_labels(ax):
    return [t.get_text() for t in ax.get_xticklabels()]


def legend_texts(ax):
    return [t.get_text() for t in ax.get_legend().get_texts()]


# ---- independent expectations ---------------------------------------------------


def props(rows, rank):
    return [r[f"{rank}_prop"] for r in rows]


def mean_and_se(values_by_rank):
    """Mean and SE across units, from ``{rank: [value per unit]}``."""
    means = [np.mean(values_by_rank[rank]) for rank in RANKS]
    ses = [
        np.std(values_by_rank[rank], ddof=1) / np.sqrt(len(values_by_rank[rank])) for rank in RANKS
    ]
    return means, ses


def pooled(rows, rank):
    return sum(r[rank] for r in rows) / sum(r["total"] for r in rows)


# ---- panels draw into a given ax -------------------------------------------------


def test_panels_draw_into_given_ax_without_new_figure(counts):
    fig, axes = plt.subplots(1, 4)
    plot_average_across_mice(counts, ax=axes[0])
    plot_average_across_blocks(counts, ax=axes[1])
    plot_mouse_panel("m1", counts["m1"], ax=axes[2])
    plot_block_panel(counts, 0, ax=axes[3])
    assert plt.get_fignums() == [fig.number]
    assert all(len(ax.patches) == 3 for ax in axes)


def test_panels_create_a_figure_when_no_ax_is_given(counts):
    ax = plot_average_across_mice(counts)
    assert ax.figure.number in plt.get_fignums()


def test_panels_take_flags_by_keyword_only(counts):
    with pytest.raises(TypeError):
        plot_average_across_mice(counts, None)
    with pytest.raises(TypeError):
        plot_mouse_panel("m1", counts["m1"], True)
    with pytest.raises(TypeError):
        plot_block_panel(counts, 0, None)
    with pytest.raises(TypeError):
        plot_average_across_blocks(counts, None)


# ---- average across mice ---------------------------------------------------------


def test_average_across_mice_bars_and_errors_use_every_row_of_every_mouse(counts):
    ax = plot_average_across_mice(counts)
    mouse_means = {rank: [np.mean(props(rows, rank)) for rows in counts.values()] for rank in RANKS}
    means, ses = mean_and_se(mouse_means)
    assert bar_heights(ax) == pytest.approx(means)
    assert error_bar_lengths(ax) == pytest.approx(ses)
    assert [line.get_label() for line in ax.get_legend().get_lines()] == ["m1", "m2", "m3"]


def test_average_across_mice_pooled_bars_use_each_mouses_pooled_proportion(counts):
    ax = plot_average_across_mice(counts, pool_trials_within_subject=True)
    mouse_pooled = {rank: [pooled(rows, rank) for rows in counts.values()] for rank in RANKS}
    means, ses = mean_and_se(mouse_pooled)
    assert bar_heights(ax) == pytest.approx(means)
    assert error_bar_lengths(ax) == pytest.approx(ses)
    assert ax.get_title() == "Pooled Trials Within Mouse"


def test_average_across_mice_lines_are_one_per_mouse_in_rank_order(counts):
    ax = plot_average_across_mice(counts)
    ys = [list(line.get_ydata()) for line in data_lines(ax)]
    expected = [[np.mean(props(rows, rank)) for rank in RANKS] for rows in counts.values()]
    assert np.array(ys) == pytest.approx(np.array(expected))


def test_average_across_mice_title_names_the_aggregation(counts):
    assert plot_average_across_mice(counts).get_title() == "Average Across Mice"


def test_mean_labels_are_two_decimals_above_each_bar():
    exact = {
        "a": [make_row(2, 1, 1), make_row(1, 1, 2)],
        "b": [make_row(2, 1, 1), make_row(1, 1, 2)],
    }
    ax = plot_average_across_mice(exact)
    # a and b are identical, each with block means (0.375, 0.25, 0.375)
    assert mean_labels(ax) == [
        ((0, LABEL_Y), "0.38"),
        ((1, LABEL_Y), "0.25"),
        ((2, LABEL_Y), "0.38"),
    ]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_average_across_mice_skips_the_label_of_a_nan_bar():
    nan_second = [{**make_row(3, 2, 1), "second_prop": np.nan} for _ in range(2)]
    ax = plot_average_across_mice({"a": nan_second, "b": nan_second})
    assert np.isnan(bar_heights(ax)[1])
    assert [pos for pos, _ in mean_labels(ax)] == [(0, LABEL_Y), (2, LABEL_Y)]


def test_pvalues_are_written_on_the_bars_at_fixed_positions(counts):
    pvalues = {"best_vs_second": 0.012, "best_vs_third": 0.0004, "second_vs_third": 0.5}
    ax = plot_average_across_mice(counts, pvalues=pvalues)
    assert pvalue_labels(ax) == {
        "p(Best vs Second):\n0.012": (0, 0.75),
        "p(Best vs Third):\n0.000": (2, 0.25),
        "p(Second vs Third):\n0.500": (1, 0.5),
    }


def test_pvalues_missing_none_or_empty_are_not_written(counts):
    assert pvalue_labels(plot_average_across_mice(counts)) == {}
    assert pvalue_labels(plot_average_across_mice(counts, pvalues={})) == {}
    partial = {"best_vs_second": None, "second_vs_third": 0.2}
    assert pvalue_labels(plot_average_across_mice(counts, pvalues=partial)) == {
        "p(Second vs Third):\n0.200": (1, 0.5)
    }


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_known_defect_pvalue_with_a_nan_bar_raises_index_error():
    """Characterises a known defect: p-values are placed by index into the labelled bars."""
    nan_second = [{**make_row(3, 2, 1), "second_prop": np.nan} for _ in range(2)]
    counts = {"a": nan_second, "b": nan_second}
    with pytest.raises(IndexError):
        plot_average_across_mice(counts, pvalues={"best_vs_third": 0.5})
    # without p-values the same data draws fine
    plot_average_across_mice(counts)


# ---- average across blocks -------------------------------------------------------


def test_average_across_blocks_bars_use_every_block_of_every_mouse(counts):
    ax = plot_average_across_blocks(counts)
    n_blocks = max(len(rows) for rows in counts.values())
    block_units = {
        rank: [
            np.mean([props(rows, rank)[b] for rows in counts.values() if b < len(rows)])
            for b in range(n_blocks)
        ]
        for rank in RANKS
    }
    means, ses = mean_and_se(block_units)
    assert bar_heights(ax) == pytest.approx(means)
    assert error_bar_lengths(ax) == pytest.approx(ses)
    assert mean_labels(ax) == [((i, LABEL_Y), f"{m:.2f}") for i, m in enumerate(means)]


def test_average_across_blocks_has_one_line_per_block(counts):
    ax = plot_average_across_blocks(counts)
    assert len(data_lines(ax)) == 4
    expected = [
        [
            np.mean([props(rows, rank)[b] for rows in counts.values() if b < len(rows)])
            for rank in RANKS
        ]
        for b in range(4)
    ]
    ys = [list(line.get_ydata()) for line in data_lines(ax)]
    assert np.array(ys) == pytest.approx(np.array(expected))
    assert legend_texts(ax) == ["Block 1", "Block 2", "Block 3", "Block 4"]


def test_average_across_blocks_with_every_subject_empty_draws_no_blocks():
    ax = plot_average_across_blocks({"a": [], "b": []})
    assert len(data_lines(ax)) == 0


# ---- one mouse -------------------------------------------------------------------


def test_mouse_panel_bars_and_errors_are_the_mean_and_se_across_rows(counts):
    rows = counts["m2"]
    ax = plot_mouse_panel("m2", rows)
    means, ses = mean_and_se({rank: props(rows, rank) for rank in RANKS})
    assert bar_heights(ax) == pytest.approx(means)
    assert error_bar_lengths(ax) == pytest.approx(ses)
    assert mean_labels(ax) == [((i, LABEL_Y), f"{m:.2f}") for i, m in enumerate(means)]


def test_mouse_panel_pooled_bars_and_errors_are_both_across_trials(counts):
    rows = counts["m2"]
    ax = plot_mouse_panel("m2", rows, pool_trials_within_subject=True)
    n = sum(r["total"] for r in rows)
    ps = [pooled(rows, rank) for rank in RANKS]
    assert bar_heights(ax) == pytest.approx(ps)
    # SE across the n trials of the 0/1 indicator of the rank: sd * sqrt(n/(n-1)) / sqrt(n)
    expected = [np.sqrt(p * (1 - p) / (n - 1)) for p in ps]
    assert error_bar_lengths(ax) == pytest.approx(expected)
    unpooled = plot_mouse_panel("m2", rows)
    assert bar_heights(unpooled) != pytest.approx(bar_heights(ax))


def test_mouse_panel_unpooled_error_bars_skip_nan_blocks_in_n(counts):
    rows = counts["m2"] + [make_row(0, 0, 0)]
    ax = plot_mouse_panel("m2", rows)
    _, ses = mean_and_se({rank: props(counts["m2"], rank) for rank in RANKS})
    assert error_bar_lengths(ax) == pytest.approx(ses)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_mouse_panel_labels_a_nan_mean_as_nan():
    rows = [make_row(0, 0, 0), make_row(0, 0, 0)]
    ax = plot_mouse_panel("silent", rows, pool_trials_within_subject=True)
    assert [text for _, text in mean_labels(ax)] == ["nan", "nan", "nan"]
    assert np.isnan(bar_heights(ax)).all()


def test_mouse_panel_draws_one_line_per_row_and_labels_blocks(counts):
    ax = plot_mouse_panel("m2", counts["m2"])
    assert len(data_lines(ax)) == 4
    assert legend_texts(ax) == ["Block 1", "Block 2", "Block 3", "Block 4"]
    assert ax.get_title() == "m2"
    assert ax.get_ylabel() == "Proportion of Choices"
    assert plot_mouse_panel("m2", counts["m2"], show_ylabel=False).get_ylabel() == ""


def test_mouse_panel_lines_are_the_rows_proportions_in_rank_order(counts):
    rows = counts["m2"]
    ax = plot_mouse_panel("m2", rows)
    ys = [list(line.get_ydata()) for line in data_lines(ax)]
    assert np.array(ys) == pytest.approx(np.array([[r[f"{k}_prop"] for k in RANKS] for r in rows]))


def test_mouse_panel_single_row_has_nan_error_bars(counts):
    ax = plot_mouse_panel("m1", counts["m1"][:1])
    assert np.isnan(error_bar_lengths(ax)).all()
    assert bar_heights(ax) == pytest.approx([props(counts["m1"][:1], rank)[0] for rank in RANKS])


# ---- one block -------------------------------------------------------------------


def test_block_panel_bars_average_the_mice_that_have_the_block(counts):
    ax = plot_block_panel(counts, 1)
    assert ax.get_title() == "Block 2"
    assert legend_texts(ax) == ["m1 (n=10 trials)", "m2 (n=10 trials)", "m3 (n=10 trials)"]
    means, ses = mean_and_se(
        {rank: [props(rows, rank)[1] for rows in counts.values()] for rank in RANKS}
    )
    assert bar_heights(ax) == pytest.approx(means)
    assert error_bar_lengths(ax) == pytest.approx(ses)
    assert mean_labels(ax) == [((i, LABEL_Y), f"{m:.2f}") for i, m in enumerate(means)]


def test_block_panel_bars_and_lines_cover_the_same_mice_at_a_last_index(counts):
    # index 2 is the last row of m1 and m3, but not of m2
    ax = plot_block_panel(counts, 2)
    assert legend_texts(ax) == ["m1 (n=3 trials)", "m2 (n=11 trials)", "m3 (n=3 trials)"]
    assert len(data_lines(ax)) == 3
    means, ses = mean_and_se(
        {rank: [props(rows, rank)[2] for rows in counts.values()] for rank in RANKS}
    )
    assert bar_heights(ax) == pytest.approx(means)
    assert error_bar_lengths(ax) == pytest.approx(ses)


def test_block_panel_at_the_longest_mouses_last_index_has_its_bar(counts):
    ax = plot_block_panel(counts, 3)
    assert legend_texts(ax) == ["m2 (n=3 trials)"]
    assert bar_heights(ax) == pytest.approx([counts["m2"][3][f"{rank}_prop"] for rank in RANKS])
    assert np.isnan(error_bar_lengths(ax)).all()
    assert len(mean_labels(ax)) == 3


def test_block_panel_lines_are_each_mouses_proportions_at_that_block(counts):
    ax = plot_block_panel(counts, 1)
    ys = [list(line.get_ydata()) for line in data_lines(ax)]
    expected = [[props(rows, rank)[1] for rank in RANKS] for rows in counts.values()]
    assert np.array(ys) == pytest.approx(np.array(expected))


def test_block_panel_without_total_labels_lines_with_the_subject_only(counts):
    bare = {
        s: [{k: v for k, v in r.items() if k != "total"} for r in rows]
        for s, rows in counts.items()
    }
    assert legend_texts(plot_block_panel(bare, 0)) == ["m1", "m2", "m3"]


# ---- colours and bar style -------------------------------------------------------


def line_colors(ax):
    return [to_hex(line.get_color()) for line in data_lines(ax)]


def test_mouse_lines_follow_mouse_colors_and_block_lines_follow_block_colors(counts):
    mouse_colors = [to_hex(c) for c in MOUSE_COLORS[:3]]
    block_colors = [to_hex(c) for c in BLOCK_COLORS[:3]]
    assert line_colors(plot_average_across_mice(counts)) == mouse_colors
    assert line_colors(plot_block_panel(counts, 0)) == mouse_colors
    assert line_colors(plot_average_across_blocks(counts)) == [to_hex(c) for c in BLOCK_COLORS[:4]]
    assert line_colors(plot_mouse_panel("m2", counts["m2"][:3])) == block_colors


def test_mouse_panel_bars_have_their_own_style(counts):
    def style(ax):
        bar = ax.patches[0]
        return bar.get_facecolor(), bar.get_linewidth()

    mouse_panel = plot_mouse_panel("m1", counts["m1"])
    shared_style_panel = plot_block_panel(counts, 0)
    expected = _MOUSE_PANEL_BAR_STYLE
    assert style(mouse_panel) == (
        to_rgba(expected["color"], expected["alpha"]),
        expected["linewidth"],
    )
    assert style(mouse_panel) != style(shared_style_panel)


# ---- shared axis styling ---------------------------------------------------------


@pytest.fixture(params=["mice", "blocks", "mouse", "block"])
def any_panel(request, counts):
    return {
        "mice": lambda: plot_average_across_mice(counts),
        "blocks": lambda: plot_average_across_blocks(counts),
        "mouse": lambda: plot_mouse_panel("m1", counts["m1"]),
        "block": lambda: plot_block_panel(counts, 0),
    }[request.param]()


def test_every_panel_has_rank_ticks_and_fixed_y_limits(any_panel):
    assert list(any_panel.get_xticks()) == [0, 1, 2]
    assert tick_labels(any_panel) == ["Best", "Second", "Third"]
    assert any_panel.get_ylim() == (0.0, 1.07)
    assert not any_panel.spines["top"].get_visible()
    assert not any_panel.spines["right"].get_visible()


# ---- figures: saving and showing -------------------------------------------------


def test_save_path_writes_pdf_and_png_and_creates_directories(counts, tmp_path):
    base = tmp_path / "nested" / "RP"
    # the by-mouse figure is drawn first and creates the folder the average figure needs
    plot_rank_proportions(counts, save_path=str(base))
    for view in ("by Mouse", "by Block", "by Average"):
        assert (tmp_path / "nested" / f"RP {view}.pdf").exists()
        assert (tmp_path / "nested" / f"RP {view}.png").exists()
    assert plt.get_fignums() == []


@pytest.mark.parametrize("figure", [plot_rank_proportions_by_mouse, plot_rank_proportions_by_block])
def test_by_mouse_and_by_block_create_missing_folders_and_accept_a_path(figure, counts, tmp_path):
    figure(counts, save_path=tmp_path / "a" / "b" / "RP")
    assert (tmp_path / "a" / "b" / "RP.pdf").exists()
    assert (tmp_path / "a" / "b" / "RP.png").exists()
    assert plt.get_fignums() == []


def test_average_figure_does_not_create_missing_folders(counts, tmp_path):
    """Characterises the average figure's save behaviour: no mkdir, str only."""
    with pytest.raises(FileNotFoundError):
        plot_rank_proportions_average(counts, save_path=str(tmp_path / "missing" / "RP"))
    assert not (tmp_path / "missing").exists()
    assert len(plt.get_fignums()) == 1  # left open by the failure


def test_average_figure_save_path_must_be_a_str(counts, tmp_path):
    """Characterises the average figure's save behaviour: a Path raises TypeError."""
    with pytest.raises(TypeError):
        plot_rank_proportions_average(counts, save_path=tmp_path / "RP")
    assert len(plt.get_fignums()) == 1


def test_average_figure_saves_to_an_existing_folder(counts, tmp_path):
    plot_rank_proportions_average(counts, save_path=str(tmp_path / "RP"))
    assert (tmp_path / "RP.pdf").exists() and (tmp_path / "RP.png").exists()
    assert plt.get_fignums() == []


def test_public_plots_show_and_close_when_no_save_path(counts, monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(len(plt.get_fignums())))
    plot_rank_proportions_average(counts)
    plot_rank_proportions_by_mouse(counts)
    plot_rank_proportions_by_block(counts)
    assert shown == [1, 1, 1]
    assert plt.get_fignums() == []


def test_by_block_without_blocks_warns_and_draws_nothing(capsys):
    plot_rank_proportions_by_block({"m1": []})
    assert "[WARN] No blocks found to plot." in capsys.readouterr().out
    assert plt.get_fignums() == []


def test_by_mouse_turns_off_unused_axes_and_labels_only_the_first_column(counts, shown_figures):
    plot_rank_proportions_by_mouse(counts)
    axes = shown_figures[0].axes
    assert len(axes) == 4  # 2 rows x ceil(3 / 2) columns
    assert [ax.axison for ax in axes] == [True, True, True, False]
    assert [ax.get_ylabel() for ax in axes[:3]] == [
        "Proportion of Choices",
        "",
        "Proportion of Choices",
    ]
    assert [ax.get_title() for ax in axes[:3]] == ["m1", "m2", "m3"]


def test_by_mouse_turns_off_the_axis_of_a_mouse_with_no_blocks(shown_figures):
    no_blocks = {"a": [], "b": [make_row(1, 1, 1), make_row(2, 1, 1)]}
    plot_rank_proportions_by_mouse(no_blocks)
    assert [ax.axison for ax in shown_figures[0].axes] == [False, True]


def test_by_mouse_draws_every_block_of_each_mouse(counts, shown_figures):
    plot_rank_proportions_by_mouse(counts)
    axes = shown_figures[0].axes
    assert [len(data_lines(ax)) for ax in axes[:3]] == [3, 4, 3]
    means, _ = mean_and_se({rank: props(counts["m1"], rank) for rank in RANKS})
    assert bar_heights(axes[0]) == pytest.approx(means)


def test_by_block_draws_one_panel_per_block_of_the_longest_mouse(counts, shown_figures):
    plot_rank_proportions_by_block(counts)
    axes = shown_figures[0].axes
    assert len(axes) == 4  # 2 rows x ceil(4 / 2) columns
    assert [ax.get_title() for ax in axes] == ["Block 1", "Block 2", "Block 3", "Block 4"]
    assert [ax.axison for ax in axes] == [True, True, True, True]
    assert [ax.get_ylabel() for ax in axes] == [
        "Proportion of Choices",
        "",
        "Proportion of Choices",
        "",
    ]


def test_dispatcher_passes_pooling_to_the_by_mouse_figure(counts, shown_figures):
    plot_rank_proportions(counts, by_block=False, average=False, pool_trials_within_subject=True)
    pooled_axes = shown_figures[0].axes
    plot_rank_proportions(counts, by_block=False, average=False)
    unpooled_axes = shown_figures[1].axes
    assert len(data_lines(pooled_axes[0])) == 3
    assert len(data_lines(unpooled_axes[0])) == 3
    assert bar_heights(unpooled_axes[0]) != pytest.approx(bar_heights(pooled_axes[0]))
    assert bar_heights(pooled_axes[0]) == pytest.approx([pooled(counts["m1"], r) for r in RANKS])


def test_dispatcher_passes_pooling_and_pvalues_to_the_average_figure(counts, shown_figures):
    pvalues = {"best_vs_second": 0.2}
    plot_rank_proportions(
        counts,
        average_across_mice_pvalues=pvalues,
        by_mouse=False,
        by_block=False,
        pool_trials_within_subject=True,
    )
    left = shown_figures[0].axes[0]
    assert left.get_title() == "Pooled Trials Within Mouse"
    assert pvalue_labels(left) == {"p(Best vs Second):\n0.200": (0, 0.75)}


def test_dispatcher_flags_choose_the_figures(counts, shown_figures):
    plot_rank_proportions(counts, by_mouse=False, average=False)
    assert len(shown_figures) == 1 and shown_figures[0].axes[0].get_title() == "Block 1"
    plot_rank_proportions(counts, by_mouse=False, by_block=False, average=False)
    assert len(shown_figures) == 1


# ---- warnings --------------------------------------------------------------------


def test_average_figure_of_one_mouse_warns_about_degrees_of_freedom(counts, shown_figures):
    with pytest.warns(RuntimeWarning):
        plot_rank_proportions_average({"m1": counts["m1"]})


def test_average_figure_of_no_data_warns(shown_figures):
    with pytest.warns(RuntimeWarning):
        plot_rank_proportions_average({})


def test_figures_of_several_mice_do_not_warn(counts, shown_figures):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot_rank_proportions_by_mouse(counts, pool_trials_within_subject=True)
        plot_rank_proportions_by_mouse(counts)
        plot_rank_proportions_by_block(counts)
        plot_rank_proportions_average(counts)


def test_block_panel_with_one_or_no_units_does_not_warn(counts):
    # fewer than two units: the SE is NaN, and with none the mean is NaN, both silently
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot_block_panel(counts, 2)
        plot_block_panel(counts, 3)
        plot_mouse_panel("m1", counts["m1"][:1])
        plot_mouse_panel("m1", counts["m1"][:1], pool_trials_within_subject=True)


def test_pooled_mouse_panel_does_not_average_proportions_it_discards():
    # pooled bars and error bars both come from the counts, so an all-NaN proportion
    # column must not trigger any warning
    rows = [
        {**make_row(3, 2, 1), "second_prop": np.nan},
        {**make_row(1, 2, 3), "second_prop": np.nan},
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot_mouse_panel("m", rows, pool_trials_within_subject=True)


# ---- star import -----------------------------------------------------------------


def test_star_import_exposes_plot_functions_but_not_drawing_constants():
    namespace = {}
    exec("from src.behavior_visualization.plot_rank_proportions import *", namespace)
    public = {name for name in namespace if not name.startswith("__")}
    assert {
        "plot_rank_proportions",
        "plot_rank_proportions_average",
        "plot_rank_proportions_by_mouse",
        "plot_rank_proportions_by_block",
        "BLOCK_COLORS",
        "MOUSE_COLORS",
        "np",
        "plt",
        "Path",
        "math",
        "Line2D",
    } <= public
    # a notebook's own `X` (e.g. a design matrix) must not be overwritten
    assert (
        not {"X", "RANK_LABELS", "BAR_WIDTH", "Y_LIMITS", "BAR_STYLE", "MOUSE_LINES", "RANKS"}
        & public
    )
