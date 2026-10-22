"""Significance brackets on the across-mice rank-proportion panel (the paper-ready figures)."""

import math
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.behavior_analysis.rank_proportions import proportions_by_mouse
from src.behavior_analysis.rank_significance import PairedTest, describe_rank_tests
from src.behavior_visualization.paper_style import KEEP_ANNOTATION_GID, paper_figures, polish
from src.behavior_visualization.plot_rank_proportions import (
    SIGNIFICANCE_GID,
    _bracket_levels,
    plot_average_across_mice,
    plot_rank_proportions_average_by_mouse,
)
from tests.rank_rows import make_row


def a_test(stars, p_holm, p=0.01, statistic=1.0):
    return PairedTest(8, 8, statistic, p, "exact", p_holm=p_holm, stars=stars)


UNTESTED = PairedTest(1, 1, math.nan, math.nan, None, reason="fewer than 2 mice")
SIGNIFICANCE = {
    "best_vs_second": a_test("*", 0.03),
    "best_vs_third": a_test("***", 0.0004),
    "second_vs_third": a_test("n.s.", 0.4),
}


@pytest.fixture
def counts():
    return {
        "m1": [make_row(6, 3, 1), make_row(9, 2, 1)],
        "m2": [make_row(8, 1, 1), make_row(7, 2, 1)],
        "m3": [make_row(9, 1, 0), make_row(5, 4, 1)],
    }


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def brackets(ax):
    return [line for line in ax.lines if line.get_gid() == SIGNIFICANCE_GID]


def marks(ax):
    return [t for t in ax.texts if t.get_gid() == KEEP_ANNOTATION_GID]


def draw(counts, significance=SIGNIFICANCE, **kwargs):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    plot_average_across_mice(counts, significance=significance, ax=ax, **kwargs)
    return fig, ax


def data_top(counts, ax):
    """Highest drawn data: bar tops plus error bars, and every mouse point."""
    mice = [line for line in ax.lines if line.get_marker() == "o"]
    points = max(np.nanmax(line.get_ydata()) for line in mice)
    error_tops = [
        np.nanmax(seg[:, 1]) for c in ax.collections for seg in c.get_segments() if len(seg)
    ]
    return max([points, *error_tops])


# --- levels ---------------------------------------------------------------------------------


def test_adjacent_brackets_share_the_lowest_level_and_the_long_one_goes_above():
    assert _bracket_levels([(0, 1), (0, 2), (1, 2)]) == [0, 1, 0]


def test_a_lone_bracket_is_on_the_lowest_level():
    assert _bracket_levels([(0, 2)]) == [0]


def test_overlapping_brackets_never_share_a_level():
    spans = [(0, 2), (0, 1), (1, 2), (0, 2)]
    levels = _bracket_levels(spans)
    for (a, la), (b, lb) in combinations(zip(spans, levels, strict=True), 2):
        if a[0] < b[1] and b[0] < a[1]:
            assert la != lb


# --- drawing --------------------------------------------------------------------------------


def test_three_brackets_with_their_marks_and_no_p_value_text(counts):
    _, ax = draw(counts)
    assert len(brackets(ax)) == 3
    assert sorted(t.get_text() for t in marks(ax)) == sorted(["*", "***", "n.s."])
    assert not [t for t in ax.texts if "p(" in t.get_text()]
    for text in marks(ax):
        assert text.get_fontsize() == 7


def test_brackets_are_thin_and_sit_above_every_bar_error_bar_and_mouse_point(counts):
    _, ax = draw(counts)
    lowest = min(np.min(line.get_ydata()) for line in brackets(ax))
    assert lowest > data_top(counts, ax)
    assert all(line.get_linewidth() == 0.5 for line in brackets(ax))


@pytest.mark.parametrize("paper", [False, True])
def test_no_mark_or_bracket_overlaps_another_and_all_fit_inside_the_axes(counts, paper):
    if paper:
        with paper_figures():
            fig, ax = draw(counts)
            polish(fig)
    else:
        fig, ax = draw(counts)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [a.get_window_extent(renderer) for a in [*brackets(ax), *marks(ax)]]
    for a, b in combinations(boxes, 2):
        assert not a.overlaps(b)
    for box in boxes:
        assert box.y1 <= ax.bbox.y1 + 0.5


def test_the_y_ticks_stay_within_proportions(counts):
    _, ax = draw(counts)
    ticks = ax.get_yticks()
    assert ticks.min() == 0 and ticks.max() == pytest.approx(1.0)


def test_show_ns_false_drops_the_non_significant_bracket(counts):
    _, ax = draw(counts, show_ns=False)
    assert len(brackets(ax)) == 2
    assert "n.s." not in [t.get_text() for t in marks(ax)]


def test_an_untested_comparison_gets_no_bracket(counts):
    _, ax = draw(counts, significance={**SIGNIFICANCE, "second_vs_third": UNTESTED})
    assert len(brackets(ax)) == 2


def test_p_values_and_significance_cannot_both_be_drawn(counts):
    with pytest.raises(ValueError, match="either"):
        plot_average_across_mice(counts, pvalues={"best_vs_second": 0.1}, significance=SIGNIFICANCE)


def test_with_mouse_legend_the_legend_moves_outside_the_axes(counts):
    _, ax = draw(counts)
    fig = ax.figure
    fig.canvas.draw()
    assert ax.get_legend().get_window_extent().x0 >= ax.bbox.x1


# --- the caption ----------------------------------------------------------------------------


def test_polish_keeps_the_marks_and_puts_the_statistics_and_means_in_the_caption(counts):
    with paper_figures():
        fig, ax = draw(counts)
        caption = polish(fig)
    assert sorted(t.get_text() for t in ax.texts) == sorted(["*", "***", "n.s."])
    assert describe_rank_tests(SIGNIFICANCE) in caption
    means = proportions_by_mouse(counts)
    best = np.mean([m["best"] for m in means.values()])
    assert f"Best {best:.2f}" in caption
    assert "p(Best" not in caption


def test_the_paper_figure_file_has_the_caption_sentence(counts, tmp_path):
    with paper_figures():
        plot_rank_proportions_average_by_mouse(
            counts, significance=SIGNIFICANCE, save_path=str(tmp_path / "rank")
        )
    caption = (tmp_path / "rank.caption.txt").read_text()
    assert "Two-sided Wilcoxon signed-rank tests across mice" in caption
    assert "p(Best vs Second)" not in caption
