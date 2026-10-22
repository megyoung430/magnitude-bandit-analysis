"""Value heatmaps and psychometric curves: fixed colour range, annotations, saved files."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.behavior_analysis.get_choice_probs_by_value import MouseCounts
from src.behavior_analysis.value_psychometrics import heatmap_tables, psychometric_curves
from src.behavior_visualization.plot_helpers import figure_caption
from src.behavior_visualization.plot_value_psychometrics import (
    annotation_position,
    label_placements,
    plot_overall_heatmaps,
    plot_per_mouse_heatmaps,
    plot_prob_heatmap,
    plot_psychometric_curves,
    plot_psychometric_panel,
    tie_label,
    tie_summary,
)


@pytest.fixture(autouse=True)
def close_figures():
    plt.close("all")
    yield
    plt.close("all")


def counts_for(cells):
    counts = MouseCounts.empty()
    for (b, s), (n, n_best, n_second) in cells.items():
        counts.denom.loc[b, s] = n
        counts.num_best.loc[b, s] = n_best
        counts.num_second.loc[b, s] = n_second
    return counts


TIES = {"best_second": 4, "second_third": 0, "all_three": 2}
CELLS = {(4, 1): (20, 19, 1), (4, 3): (30, 15, 12), (3, 2): (10, 9, 0), (2, 1): (8, 2, 6)}


def two_mice():
    return {"m1": counts_for(CELLS), "m2": counts_for({(4, 1): (10, 10, 0), (3, 2): (5, 3, 2)})}


# --- annotation placement ------------------------------------------------------------------


@pytest.mark.parametrize(
    "mean, above, expected_y, expected_va",
    [
        (0.5, True, 0.55, "bottom"),
        (0.5, False, 0.45, "top"),
        (0.98, True, 0.93, "top"),  # no room above: flips below the point
        (0.02, False, 0.07, "bottom"),  # no room below: flips above the point
        (1.0, False, 0.95, "top"),
    ],
)
def test_annotation_stays_inside_the_unit_interval(mean, above, expected_y, expected_va):
    y, va = annotation_position(mean, above=above)
    assert (y, va) == (pytest.approx(expected_y), expected_va)
    height = 0.05  # text is drawn up (bottom) or down (top) from y, at most about this far
    low, high = (y, y + height) if va == "bottom" else (y - height, y)
    assert 0.0 <= low and high <= 1.0


# --- heatmaps ------------------------------------------------------------------------------


def test_heatmap_uses_a_fixed_unit_colour_range_and_the_given_axes():
    tables = heatmap_tables(two_mice())
    fig, ax = plt.subplots()
    out = plot_prob_heatmap(tables.p_best, tables.n_trials, "P(Best)", ax=ax)
    assert out is ax and plt.get_fignums() == [fig.number]
    assert ax.images[0].get_clim() == (0.0, 1.0)


def test_heatmap_cells_show_probability_and_trial_count_and_na_when_empty():
    tables = heatmap_tables({"m1": counts_for(CELLS)})
    _, ax = plt.subplots()
    plot_prob_heatmap(tables.p_best, tables.n_trials, "P(Best)", ax=ax)
    texts = [t.get_text() for t in ax.texts]
    assert "0.95\nn=20" in texts and "0.50\nn=30" in texts
    assert "NA\nn=0" in texts and len(texts) == 9


def test_heatmap_orientation_puts_best_value_4_on_top():
    tables = heatmap_tables({"m1": counts_for(CELLS)})
    _, ax = plt.subplots()
    plot_prob_heatmap(tables.p_best, tables.n_trials, "P(Best)", ax=ax)
    assert [t.get_text() for t in ax.get_yticklabels()] == ["4", "3", "2"]
    top_left = next(t for t in ax.texts if t.get_position() == (0, 0))
    assert top_left.get_text() == "0.95\nn=20"  # best 4, second 1


def test_heatmap_without_annotation_draws_no_text():
    tables = heatmap_tables(two_mice())
    _, ax = plt.subplots()
    plot_prob_heatmap(tables.p_best, tables.n_trials, "P(Best)", ax=ax, annotate=False)
    assert len(ax.texts) == 0


def test_overall_heatmaps_save_one_png_and_pdf(tmp_path):
    tables = heatmap_tables(two_mice())
    plot_overall_heatmaps(
        tables, "True", "split", save_path=tmp_path / "Heatmaps (True Value) Overall"
    )
    assert (tmp_path / "Heatmaps (True Value) Overall.png").exists()
    assert (tmp_path / "Heatmaps (True Value) Overall.pdf").exists()
    assert plt.get_fignums() == []


def test_per_mouse_heatmaps_have_a_row_per_mouse_and_a_column_per_choice(tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.extend(plt.get_fignums()))
    tables = heatmap_tables(two_mice())
    plot_per_mouse_heatmaps(tables, "Believed", "split")
    assert len(shown) == 1


def test_per_mouse_heatmap_layout(monkeypatch):
    from src.behavior_visualization import plot_value_psychometrics as module

    seen = {}
    monkeypatch.setattr(module, "save_or_show", lambda fig, path: seen.update(fig=fig))
    plot_per_mouse_heatmaps(heatmap_tables(two_mice()), "True", "split")
    heat_axes = [a for a in seen["fig"].axes if a.images]
    assert len(heat_axes) == 4
    assert [a.get_title() for a in heat_axes][:2] == ["P(Best) - m1", "P(Second) - m1"]


# --- psychometric curves -------------------------------------------------------------------


def draw_panel(counts, pool_trials=True, axis="best_val"):
    curves = psychometric_curves(counts, pool_trials=pool_trials, tie_mode="split")
    _, ax = plt.subplots()
    plot_psychometric_panel(curves["best"][axis], curves["second"][axis], "Value of Best", ax=ax)
    return ax


def test_panel_draws_thin_mouse_lines_a_thick_mean_and_an_se_band():
    ax = draw_panel(two_mice())
    widths = sorted({line.get_linewidth() for line in ax.lines if len(line.get_xdata()) > 1})
    assert widths[0] < widths[-1]
    assert len(ax.collections) == 2  # one SE band per choice (two mice give an SE)
    assert ax.get_ylim() == (0.0, 1.0)


def test_panel_omits_the_se_band_with_a_single_mouse_but_keeps_the_mean():
    ax = draw_panel({"m1": counts_for(CELLS)})
    assert len(ax.collections) == 0
    assert any(line.get_linewidth() > 2 for line in ax.lines)


def test_mean_annotations_are_inside_the_axes_even_at_the_extremes():
    extreme = {(4, 1): (10, 10, 0), (3, 1): (10, 0, 10), (2, 1): (10, 5, 5)}
    ax = draw_panel({"m1": counts_for(extreme), "m2": counts_for(extreme)})
    labels = [t for t in ax.texts if t.get_text()]
    assert labels, "means are annotated"
    for text in labels:
        y = text.get_position()[1]
        assert 0.0 <= y <= 1.0
    assert {"1.00", "0.00", "0.50"} <= {t.get_text() for t in labels}


def test_curves_figure_has_three_panels_and_is_saved(tmp_path):
    curves = psychometric_curves(two_mice(), pool_trials=False, tie_mode="split")
    base = tmp_path / "Psychometric Curves (True Value) (Pool Trials False)"
    plot_psychometric_curves(curves, "True", False, TIES, "split", save_path=base)
    assert base.with_suffix(".png").exists() and base.with_suffix(".pdf").exists()


def test_curves_figure_panels_and_title(monkeypatch):
    from src.behavior_visualization import plot_value_psychometrics as module

    seen = {}
    monkeypatch.setattr(module, "save_or_show", lambda fig, path: seen.update(fig=fig))
    curves = psychometric_curves(two_mice(), pool_trials=True, tie_mode="split")
    plot_psychometric_curves(curves, "Believed", True, TIES, "split")
    fig = seen["fig"]
    assert [a.get_xlabel() for a in fig.axes] == [
        "Value of Best",
        "Value of Second",
        "Value of Best - Second",
    ]
    assert fig._suptitle.get_text() == "Psychometric Curves (Believed Value)"
    assert figure_caption(fig).startswith("Trials pooled within mouse")
    assert [a.get_title() for a in fig.axes] == [
        f"{x}\nmean ± SE across mice, n = 2 mice"
        for x in ("Value of Best", "Value of Second", "Value of Best - Second")
    ]


def curve_panels(tie_mode):
    """The three axes of a psychometric figure drawn from two mice without tied trials."""
    from src.behavior_visualization import plot_value_psychometrics as module

    curves = psychometric_curves(two_mice(), pool_trials=True, tie_mode=tie_mode)
    captured = {}
    original = module.save_or_show
    module.save_or_show = lambda fig, path: captured.update(fig=fig)
    try:
        plot_psychometric_curves(curves, "True", True, TIES, tie_mode)
    finally:
        module.save_or_show = original
    return captured["fig"].axes


def relative_padding(ax):
    """Padding left and right of the first and last tick, as a fraction of the tick range."""
    ticks = ax.get_xticks()
    low, high = ax.get_xlim()
    span = ticks[-1] - ticks[0]
    return (ticks[0] - low) / span, (high - ticks[-1]) / span


def test_ties_excluded_panels_have_the_same_ticks_and_padding():
    best, second, diff = curve_panels("exclude")
    assert list(diff.get_xticks()) == [1, 2, 3]
    assert len(best.get_xticks()) == len(second.get_xticks()) == len(diff.get_xticks()) == 3
    paddings = [relative_padding(ax) for ax in (best, second, diff)]
    assert paddings[1] == pytest.approx(paddings[0])
    assert paddings[2] == pytest.approx(paddings[0])
    assert paddings[0][0] == pytest.approx(paddings[0][1]) and paddings[0][0] > 0


def test_ties_split_difference_panel_includes_zero_with_the_same_relative_padding():
    best, _, diff = curve_panels("split")
    assert list(diff.get_xticks()) == [0, 1, 2, 3]
    assert relative_padding(diff) == pytest.approx(relative_padding(best))


def test_limits_do_not_depend_on_which_end_points_have_data():
    # a missing last point must not shrink the axis (NaN is ignored by autoscaling)
    only_low = {"m1": counts_for({(3, 2): (10, 6, 4)})}
    curves = psychometric_curves(only_low, pool_trials=True, tie_mode="exclude")
    _, ax = plt.subplots()
    plot_psychometric_panel(curves["best"]["diff"], curves["second"]["diff"], "d", ax=ax)
    assert relative_padding(ax) == pytest.approx(relative_padding(curve_panels("exclude")[0]))


def test_no_mice_still_draws_empty_panels():
    curves = psychometric_curves({}, pool_trials=True, tie_mode="split")
    _, ax = plt.subplots()
    plot_psychometric_panel(curves["best"]["diff"], curves["second"]["diff"], "x", ax=ax)
    assert np.isnan(curves["best"]["diff"].mean).all()


# --- label placement -----------------------------------------------------------------------

LABEL_HEIGHT = 0.05  # the placement budget for one label, in data units


def span(placement):
    low = {"bottom": placement.y, "top": placement.y - LABEL_HEIGHT}.get(
        placement.va, placement.y - LABEL_HEIGHT / 2
    )
    return low, low + LABEL_HEIGHT


def test_labels_of_well_separated_means_go_outside_the_gap():
    best, second = label_placements(0.8, 0.3)
    assert (best.va, second.va) == ("bottom", "top") and best.y > 0.8 and second.y < 0.3


def test_crossed_curves_put_the_higher_label_above_whichever_curve_it_is():
    best, second = label_placements(0.40, 0.50)
    assert second.va == "bottom" and best.va == "top"


@pytest.mark.parametrize("gap", [0.0, 0.02, 0.05, 0.1])
def test_close_means_do_not_collide_anywhere_on_the_axis(gap):
    for low in np.arange(0.0, 1.0 - gap + 1e-9, 0.01):
        for mean_best, mean_second in ((low, low + gap), (low + gap, low)):
            best, second = label_placements(mean_best, mean_second)
            for placement in (best, second):
                lo, hi = span(placement)
                assert -1e-9 <= lo and hi <= 1 + 1e-9, (mean_best, mean_second)
            if best.dx_sign == second.dx_sign:  # stacked in one column
                (a0, a1), (b0, b1) = span(best), span(second)
                assert a1 <= b0 + 1e-9 or b1 <= a0 + 1e-9, (mean_best, mean_second)
                for (lo, hi), other_mean in (((a0, a1), mean_second), ((b0, b1), mean_best)):
                    assert not (lo - 0.01 <= other_mean <= hi + 0.01), (mean_best, mean_second)


def test_labels_beside_the_points_when_there_is_no_room_above_or_below():
    best, second = label_placements(0.97, 0.95)
    assert {best.dx_sign, second.dx_sign} == {1, -1}
    assert best.va == second.va == "center"


def test_panel_labels_of_close_means_are_separate_texts_in_range():
    close = {(4, 1): (100, 50, 45), (3, 1): (100, 40, 50)}
    ax = draw_panel({"m1": counts_for(close), "m2": counts_for(close)})
    positions = [t.get_position() for t in ax.texts]
    assert len(positions) == 4 and len(set(positions)) == 4


# --- tie reporting and layout --------------------------------------------------------------


def test_tie_label_names_the_mode_and_rejects_unknown_ones():
    assert (tie_label("exclude"), tie_label("split")) == ("Excluded", "Split")
    with pytest.raises(ValueError, match="tie_mode"):
        tie_label("drop")


def test_tie_summary_gives_the_three_totals_and_the_fate_of_the_trials():
    excluded = tie_summary(TIES, "exclude")
    assert "best/second only: 4" in excluded and "second/third only: 0" in excluded
    assert "all three: 2" in excluded and "left out" in excluded
    assert "split evenly" in tie_summary(TIES, "split")


def capture_figure(monkeypatch):
    from src.behavior_visualization import plot_value_psychometrics as module

    seen = {}
    monkeypatch.setattr(module, "save_or_show", lambda fig, path: seen.update(fig=fig))
    return seen


def tie_line(fig):
    return [
        line for line in figure_caption(fig).splitlines() if line.startswith("Trials with ties")
    ]


@pytest.mark.parametrize("tie_mode, word", [("exclude", "Excluded"), ("split", "Split")])
def test_every_figure_prints_the_tie_totals_and_the_mode(monkeypatch, tie_mode, word):
    seen = capture_figure(monkeypatch)
    counts = two_mice()
    counts["m1"].ties.update(TIES)
    tables = heatmap_tables(counts)
    curves = psychometric_curves(counts, pool_trials=True, tie_mode=tie_mode)
    drawers = [
        lambda: plot_overall_heatmaps(tables, "True", tie_mode),
        lambda: plot_per_mouse_heatmaps(tables, "True", tie_mode),
        lambda: plot_psychometric_curves(curves, "True", True, tables.ties, tie_mode),
    ]
    for draw in drawers:
        draw()
        fig = seen["fig"]
        assert len(tie_line(fig)) == 1 and "best/second only: 4" in tie_line(fig)[0]
        assert f"Ties {word.lower()}." in figure_caption(fig).splitlines()
        assert "Ties" not in fig._suptitle.get_text()
        assert not [t for t in fig.texts if t.get_visible() and t.get_text().startswith("Trials")]


def test_per_mouse_title_clears_the_first_row(monkeypatch):
    seen = capture_figure(monkeypatch)
    plot_per_mouse_heatmaps(heatmap_tables(two_mice()), "True", "split")
    fig = seen["fig"]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title_box = fig._suptitle.get_window_extent(renderer)
    for ax in fig.axes:
        assert ax.get_tightbbox(renderer).y1 <= title_box.y0 + 1


def test_one_mouse_is_not_pluralised(monkeypatch):
    seen = capture_figure(monkeypatch)
    plot_overall_heatmaps(heatmap_tables({"m1": counts_for(CELLS)}), "True", "split")
    assert seen["fig"]._suptitle.get_text().endswith("mean across mice, n = 1 mouse")
    curves = psychometric_curves({"m1": counts_for(CELLS)}, pool_trials=True, tie_mode="split")
    plot_psychometric_curves(curves, "True", True, TIES, "split")
    # one mouse has no SE band, so the panels say mean only
    assert seen["fig"].axes[0].get_title().endswith("mean across mice, n = 1 mouse")


# --- side labels, NaN means, centring, whitespace ------------------------------------------


@pytest.mark.parametrize(
    "mean_best, mean_second, best_side",
    [(0.97, 0.95, 1), (0.95, 0.97, -1)],
)
def test_side_labels_put_the_higher_mean_on_the_right(mean_best, mean_second, best_side):
    best, second = label_placements(mean_best, mean_second)
    assert (best.dx_sign, second.dx_sign) == (best_side, -best_side)
    right, left = (best, second) if best_side == 1 else (second, best)
    assert (right.ha, left.ha) == ("left", "right")


def test_side_labels_are_drawn_beside_their_points_with_a_nonzero_offset():
    edge = {(4, 1): (100, 97, 95), (3, 1): (100, 50, 40)}  # P(best) .97 and P(second) .95 at 4
    ax = draw_panel({"m1": counts_for(edge), "m2": counts_for(edge)})
    by_text = {t.get_text(): t for t in ax.texts}
    high, low = by_text["0.97"], by_text["0.95"]
    assert high.get_position()[0] > 4 > low.get_position()[0]
    assert (high.get_horizontalalignment(), low.get_horizontalalignment()) == ("left", "right")
    # stacked labels stay centred on their x
    assert by_text["0.50"].get_position()[0] == 3 and by_text["0.40"].get_position()[0] == 3


def test_a_mean_that_is_nan_for_only_one_choice_raises():
    from src.behavior_visualization.plot_value_psychometrics import _placements_at

    assert _placements_at(np.nan, np.nan) == (None, None)
    with pytest.raises(ValueError, match="NaN together"):
        _placements_at(0.5, np.nan)


def figure_after_drawing(monkeypatch, draw):
    seen = capture_figure(monkeypatch)
    draw()
    fig = seen["fig"]
    fig.canvas.draw()
    return fig, fig.canvas.get_renderer()


def panel_span(fig):
    positions = [ax.get_position() for ax in fig.axes]
    return min(p.x0 for p in positions), max(p.x1 for p in positions)


@pytest.mark.parametrize("which", ["overall", "per_mouse", "curves"])
def test_the_title_is_centred_over_the_panels(monkeypatch, which):
    tables = heatmap_tables(two_mice())
    curves = psychometric_curves(two_mice(), pool_trials=True, tie_mode="split")
    draw = {
        "overall": lambda: plot_overall_heatmaps(tables, "True", "split"),
        "per_mouse": lambda: plot_per_mouse_heatmaps(tables, "True", "split"),
        "curves": lambda: plot_psychometric_curves(curves, "True", True, tables.ties, "split"),
    }[which]
    fig, _ = figure_after_drawing(monkeypatch, draw)
    left, right = panel_span(fig)
    assert fig._suptitle.get_position()[0] == pytest.approx((left + right) / 2)


def test_the_gap_between_title_and_panel_titles_is_small(monkeypatch):
    tables = heatmap_tables(two_mice())
    fig, renderer = figure_after_drawing(
        monkeypatch, lambda: plot_overall_heatmaps(tables, "True", "split")
    )
    title_bottom = fig._suptitle.get_window_extent(renderer).y0
    panel_top = max(ax.get_tightbbox(renderer).y1 for ax in fig.axes)
    assert 0 <= (title_bottom - panel_top) / fig.dpi <= 0.3  # inches
