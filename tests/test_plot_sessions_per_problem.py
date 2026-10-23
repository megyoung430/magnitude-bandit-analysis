"""The sessions-per-drifting-problem figures: what is drawn, labelled and saved."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.behavior_analysis.sessions_per_problem import summarize_sessions_per_problem
from src.behavior_visualization.plot_helpers import figure_caption
from src.behavior_visualization.plot_sessions_per_problem import (
    MEAN_LABEL,
    RUNNING_LABEL,
    plot_sessions_per_problem,
    plot_sessions_per_problem_by_cohort,
    plot_sessions_per_problem_by_cohort_panel,
    plot_sessions_per_problem_panel,
)
from tests.test_sessions_per_problem import ROWS, table_of


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def summary():
    return summarize_sessions_per_problem(table_of(ROWS))


def solid_line(ax, mouse):
    (line,) = [ln for ln in ax.lines if ln.get_label() == mouse]
    return line


def running_points(ax):
    """Hollow markers: the points of problems still running."""
    return [
        ln
        for ln in ax.lines
        if ln.get_marker() == "o"
        and ln.get_markerfacecolor() == "white"
        and ln.get_linestyle() == "None"
    ]


def dashed_segments(ax):
    return [ln for ln in ax.lines if ln.get_linestyle() == "--"]


def legend_labels(ax):
    return [t.get_text() for t in ax.get_legend().get_texts()]


# --- the panel -----------------------------------------------------------------------------


def test_each_mouse_is_a_line_through_its_completed_problems(summary):
    ax = plot_sessions_per_problem_panel(summary, scope_label="cohort 3")
    for mouse, expected in {"m1": [4, 6], "m2": [6], "m3": [8]}.items():
        line = solid_line(ax, mouse)
        assert line.get_marker() == "o"
        assert list(line.get_ydata()) == expected
        assert np.allclose(line.get_xdata(), np.arange(1, len(expected) + 1), atol=0.1)


def test_a_running_problem_is_a_hollow_point_joined_by_a_dashed_segment(summary):
    ax = plot_sessions_per_problem_panel(summary, scope_label="cohort 3")
    points = running_points(ax)
    assert sorted(float(p.get_ydata()[0]) for p in points) == [2, 9]
    segments = dashed_segments(ax)
    assert sorted(list(s.get_ydata()) for s in segments) == [[6, 2], [6, 9]]  # m2 and m1


def test_a_mouse_with_only_a_running_problem_has_no_dashed_segment():
    only_running = table_of([("m1", 1, 3, False)])
    ax = plot_sessions_per_problem_panel(
        summarize_sessions_per_problem(only_running), scope_label="cohort 3"
    )
    assert len(running_points(ax)) == 1 and dashed_segments(ax) == []


def test_the_mean_is_a_thick_black_line_over_completed_problems_only(summary):
    ax = plot_sessions_per_problem_panel(summary, scope_label="cohort 3")
    (mean,) = [ln for ln in ax.lines if ln.get_label() == MEAN_LABEL]
    assert mean.get_color() == "black" and mean.get_linewidth() >= 4
    xs, ys = mean.get_data()
    assert list(xs) == [1, 2] and list(ys) == pytest.approx([6.0, 6.0])  # problem 3: none complete


def test_the_mean_is_drawn_under_the_mouse_lines_so_a_single_mouse_stays_visible(summary):
    ax = plot_sessions_per_problem_panel(summary, scope_label="cohort 3")
    (mean,) = [ln for ln in ax.lines if ln.get_label() == MEAN_LABEL]
    for mouse in summary.mice:
        assert mean.get_zorder() < solid_line(ax, mouse).get_zorder()
    for point in running_points(ax):
        assert mean.get_zorder() < point.get_zorder()


def test_ticks_are_integers_with_n_mice_under_each_problem_and_the_axis_starts_at_zero(summary):
    ax = plot_sessions_per_problem_panel(summary, scope_label="cohort 3")
    assert list(ax.get_xticks()) == [1, 2, 3]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["1\nn=3", "2\nn=1", "3\nn=0"]
    assert ax.get_ylim()[0] == 0
    ticks = ax.get_yticks()
    assert np.all(ticks == np.round(ticks))
    assert ax.get_ylim()[1] >= 9


def test_the_title_has_the_heading_and_statistics_and_the_caption_the_scope_and_count(summary):
    ax = plot_sessions_per_problem_panel(summary, scope_label="cohort 3")
    n_mice = len(summary.mice)
    assert ax.get_title() == (
        f"Sessions per Drifting Problem\nmean ± se across mice, n={n_mice} mice"
    )
    caption = figure_caption(ax.figure)
    assert "Data: cohort 3." in caption and "Sessions counted: bandit sessions." in caption
    assert "completed problems only" in caption
    total = summarize_sessions_per_problem(table_of(ROWS), include_port_finding=True)
    ax = plot_sessions_per_problem_panel(total, scope_label="cohort 3")
    assert "port-finding-only" in figure_caption(ax.figure)


def test_the_legend_names_the_mice_the_mean_and_the_running_problems(summary):
    ax = plot_sessions_per_problem_panel(summary, scope_label="cohort 3")
    assert legend_labels(ax) == ["m1", "m2", "m3", MEAN_LABEL, RUNNING_LABEL]


def test_without_a_mouse_legend_the_lines_stay_and_the_other_entries_remain(summary):
    ax = plot_sessions_per_problem_panel(summary, scope_label="cohorts 1, 3", mouse_legend=False)
    assert legend_labels(ax) == [MEAN_LABEL, RUNNING_LABEL]
    assert len([ln for ln in ax.lines if ln.get_label() in summary.mice]) == 3


def test_there_is_no_running_entry_when_every_problem_is_complete():
    done = summarize_sessions_per_problem(table_of([("m1", 1, 3, True), ("m1", 2, 4, True)]))
    ax = plot_sessions_per_problem_panel(done, scope_label="cohort 3")
    assert RUNNING_LABEL not in legend_labels(ax)


def test_a_given_axes_is_drawn_into_and_no_figure_is_added(summary):
    fig, ax = plt.subplots()
    returned = plot_sessions_per_problem_panel(summary, ax=ax, scope_label="cohort 3")
    assert returned is ax and plt.get_fignums() == [fig.number]


def test_the_same_summary_always_draws_the_same_jitter(summary):
    first = plot_sessions_per_problem_panel(summary, scope_label="c")
    second = plot_sessions_per_problem_panel(summary, scope_label="c")
    assert list(solid_line(first, "m1").get_xdata()) == list(solid_line(second, "m1").get_xdata())


# --- by cohort ------------------------------------------------------------------------------


@pytest.fixture
def by_cohort():
    rows_a = [("a1", 1, 4, True), ("a1", 2, 6, True), ("a2", 1, 6, True), ("a2", 2, 8, False)]
    rows_b = [("b1", 1, 10, True), ("b1", 2, 12, False)]
    problems = [1, 2]
    return {
        "cohort-01": summarize_sessions_per_problem(table_of(rows_a), problems=problems),
        "cohort-03": summarize_sessions_per_problem(table_of(rows_b), problems=problems),
    }


def test_each_cohort_is_a_line_of_its_mean_with_its_own_colour(by_cohort):
    ax = plot_sessions_per_problem_by_cohort_panel(by_cohort, scope_label="cohorts 1, 3")
    lines = {ln.get_label(): ln for ln in ax.lines if ln.get_label().startswith("cohort-")}
    assert list(lines["cohort-01 (2 mice)"].get_ydata()) == pytest.approx([5.0, 6.0])
    assert list(lines["cohort-03 (1 mouse)"].get_ydata()) == [10.0]  # problem 2 is still running
    assert lines["cohort-01 (2 mice)"].get_color() != lines["cohort-03 (1 mouse)"].get_color()
    assert ax.get_title() == (
        "Sessions per Drifting Problem by Cohort\nmean ± se across mice, n=3 mice"
    )
    assert "Data: cohorts 1, 3." in figure_caption(ax.figure)
    assert ax.get_ylim()[0] == 0 and list(ax.get_xticks()) == [1, 2]


def test_each_cohort_point_says_how_many_mice_are_behind_it(by_cohort):
    ax = plot_sessions_per_problem_by_cohort_panel(by_cohort, scope_label="cohorts 1, 3")
    assert sorted(t.get_text() for t in ax.texts) == ["n=1", "n=1", "n=2"]


def test_a_cohort_without_a_completed_problem_is_named_so_in_the_legend():
    running_only = summarize_sessions_per_problem(table_of([("c1", 1, 3, False)]), problems=[1, 2])
    done = summarize_sessions_per_problem(table_of([("a1", 1, 4, True)]), problems=[1, 2])
    ax = plot_sessions_per_problem_by_cohort_panel(
        {"cohort-01": done, "cohort-05": running_only}, scope_label="cohorts 1, 5"
    )
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["cohort-01 (1 mouse)", "cohort-05 (1 mouse; no completed problem)"]


# --- saving ---------------------------------------------------------------------------------


def test_the_figure_is_saved_in_every_format_with_a_caption(summary, tmp_path):
    plot_sessions_per_problem(summary, tmp_path / "out" / "Sessions per Drifting Problem", "c")
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        f"Sessions per Drifting Problem.{ext}" for ext in ("caption.txt", "pdf", "png", "svg")
    ]
    assert not plt.get_fignums()


def test_the_cohort_figure_is_saved_in_every_format_with_a_caption(by_cohort, tmp_path):
    plot_sessions_per_problem_by_cohort(by_cohort, tmp_path / "by cohort", "cohorts 1, 3")
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        f"by cohort.{ext}" for ext in ("caption.txt", "pdf", "png", "svg")
    ]
