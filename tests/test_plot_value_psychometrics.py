"""Value heatmaps and psychometric curves: fixed colour range, annotations, saved files."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.behavior_analysis.get_choice_probs_by_value import MouseCounts
from src.behavior_analysis.value_psychometrics import heatmap_tables, psychometric_curves
from src.behavior_visualization.plot_value_psychometrics import (
    annotation_position,
    plot_overall_heatmaps,
    plot_per_mouse_heatmaps,
    plot_prob_heatmap,
    plot_psychometric_curves,
    plot_psychometric_panel,
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
    plot_overall_heatmaps(tables, "True", save_path=tmp_path / "Heatmaps (True Value) Overall")
    assert (tmp_path / "Heatmaps (True Value) Overall.png").exists()
    assert (tmp_path / "Heatmaps (True Value) Overall.pdf").exists()
    assert plt.get_fignums() == []


def test_per_mouse_heatmaps_have_a_row_per_mouse_and_a_column_per_choice(tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.extend(plt.get_fignums()))
    tables = heatmap_tables(two_mice())
    plot_per_mouse_heatmaps(tables, "Believed")
    assert len(shown) == 1


def test_per_mouse_heatmap_layout(monkeypatch):
    from src.behavior_visualization import plot_value_psychometrics as module

    seen = {}
    monkeypatch.setattr(module, "save_or_show", lambda fig, path: seen.update(fig=fig))
    plot_per_mouse_heatmaps(heatmap_tables(two_mice()), "True")
    heat_axes = [a for a in seen["fig"].axes if a.images]
    assert len(heat_axes) == 4
    assert [a.get_title() for a in heat_axes][:2] == ["P(Best) - m1", "P(Second) - m1"]


# --- psychometric curves -------------------------------------------------------------------


def draw_panel(counts, pool_trials=True, axis="best_val"):
    curves = psychometric_curves(counts, pool_trials=pool_trials)
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
    curves = psychometric_curves(two_mice(), pool_trials=False)
    base = tmp_path / "Psychometric Curves (True Value) (Pool Trials False)"
    plot_psychometric_curves(curves, "True", pool_trials=False, save_path=base)
    assert base.with_suffix(".png").exists() and base.with_suffix(".pdf").exists()


def test_curves_figure_panels_and_title(monkeypatch):
    from src.behavior_visualization import plot_value_psychometrics as module

    seen = {}
    monkeypatch.setattr(module, "save_or_show", lambda fig, path: seen.update(fig=fig))
    curves = psychometric_curves(two_mice(), pool_trials=True)
    plot_psychometric_curves(curves, "Believed", pool_trials=True)
    fig = seen["fig"]
    assert [a.get_xlabel() for a in fig.axes] == [
        "Value of Best",
        "Value of Second",
        "Value of Best - Second",
    ]
    title = fig._suptitle.get_text()
    assert "Believed" in title and "pooled" in title and "2 mice" in title


def test_no_mice_still_draws_empty_panels():
    curves = psychometric_curves({}, pool_trials=True)
    _, ax = plt.subplots()
    plot_psychometric_panel(curves["best"]["diff"], curves["second"]["diff"], "x", ax=ax)
    assert np.isnan(curves["best"]["diff"].mean).all()
