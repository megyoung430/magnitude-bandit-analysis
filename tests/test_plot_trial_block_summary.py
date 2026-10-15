import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.container import ErrorbarContainer

from src.behavior_analysis.trial_block_summary import build_trial_block_tables
from src.behavior_analysis.trial_count_summaries import (
    daily_mean_trials,
    ecdf,
    mean_trials_by_problem_and_mouse,
    mice_by_subject_number,
    summarize_trials_by_problem,
)
from src.behavior_visualization.plot_style import MOUSE_COLORS
from src.behavior_visualization.plot_trial_block_summary import (
    _plot_problem_means,
    plot_trial_block_summaries,
)
from tests.plot_capture import cap_size, capture_figures, was_tight_laid_out
from tests.plot_fixtures import make_mouse

NUMBERS = {"m1": 1, "m2": 2, "m3": 5}
PROBLEMS = {
    1: {
        "m1": make_mouse(2),
        "m2": make_mouse(3, pick=1, n_trials=16),
        "m3": make_mouse(1),
    },
    2: {"m1": make_mouse(1, first_day=11), "m2": make_mouse(2, n_trials=16, first_day=11)},
}


@pytest.fixture(scope="module")
def tables():
    return build_trial_block_tables(PROBLEMS, NUMBERS)


@pytest.fixture
def capture(monkeypatch):
    return capture_figures(monkeypatch)


def run(capture, tables, tmp_path, **kwargs):
    sessions, blocks = tables
    paths = plot_trial_block_summaries(sessions, blocks, tmp_path / "figs", **kwargs)
    figures = {png_stem(s.path): s.figure for s in capture.saved if s.path.endswith(".png")}
    return paths, figures


def single_mouse_tables(trials):
    """Session and block tables of one mouse with the given trial counts, one per day."""
    table = pd.DataFrame(
        {
            "mouse": "m1",
            "subject_number": 1,
            "group": "sub-01–04",
            "problem": 1,
            "date": pd.date_range("2026-09-01", periods=len(trials)),
            "trials": trials,
        }
    )
    return table, table.copy()


def png_stem(path):
    return path.rsplit("/", 1)[-1][: -len(".png")]


# ---------------------------------------------------------------- analysis helpers


def test_mice_are_ordered_by_subject_number(tables):
    sessions, blocks = tables
    assert mice_by_subject_number(sessions, blocks) == ["m1", "m2", "m3"]


def test_ecdf_is_sorted_values_with_cumulative_fractions():
    x, y = ecdf(np.array([3, 1, 2, 2]))
    assert x.tolist() == [1, 2, 2, 3] and y.tolist() == [0.25, 0.5, 0.75, 1.0]


def test_daily_means_average_observations_sharing_a_date():
    table = pd.DataFrame(
        {
            "mouse": ["a", "a", "a"],
            "date": pd.to_datetime(["2026-09-01"] * 2 + ["2026-09-02"]),
            "trials": [10, 20, 30],
        }
    )
    assert daily_mean_trials(table).trials.tolist() == [15.0, 30.0]


def test_problem_means_weigh_mice_equally_with_sem_and_count():
    table = pd.DataFrame(
        {"problem": [1, 1, 1, 1], "mouse": ["a", "a", "b", "c"], "trials": [10, 30, 20, 40]}
    )
    per_mouse = mean_trials_by_problem_and_mouse(table)
    assert per_mouse.trials.tolist() == [20.0, 20.0, 40.0]
    summary = summarize_trials_by_problem(per_mouse)
    assert summary.loc[1, "mean"] == pytest.approx(80 / 3)
    assert summary.loc[1, "sem"] == pytest.approx(np.std([20, 20, 40], ddof=1) / np.sqrt(3))
    assert summary.loc[1, "n_mice"] == 3


# ---------------------------------------------------------------- outputs


def test_returns_png_then_pdf_path_for_each_figure_in_order(capture, tables, tmp_path):
    paths, _ = run(capture, tables, tmp_path)
    stems = [
        "Trials per session - distributions",
        "Trials per session - over time",
        "Trials per session - across problems",
        "Trials per block - distributions",
        "Trials per block - over time",
        "Trials per block - across problems",
    ]
    assert paths == [tmp_path / "figs" / f"{s}.{e}" for s in stems for e in ("png", "pdf")]
    assert [s.path for s in capture.saved] == [str(p) for p in paths]
    assert all(s.kwargs == {"dpi": 180, "bbox_inches": "tight"} for s in capture.saved)


def test_source_tables_and_summaries_are_written(capture, tables, tmp_path):
    run(capture, tables, tmp_path)
    written = sorted(p.name for p in (tmp_path / "figs").iterdir())
    assert written == [
        "block_trial_counts.csv",
        "block_trials_by_problem_per_mouse.csv",
        "block_trials_by_problem_summary.csv",
        "session_trial_counts.csv",
        "session_trials_by_problem_per_mouse.csv",
        "session_trials_by_problem_summary.csv",
    ]
    summary = pd.read_csv(tmp_path / "figs" / "session_trials_by_problem_summary.csv")
    assert summary.columns.tolist() == ["problem", "mean", "sem", "n_mice"]
    assert summary.n_mice.tolist() == [3, 2]


def test_show_flag_controls_plt_show_and_figures_are_closed(capture, tables, tmp_path):
    run(capture, tables, tmp_path)  # show defaults to True
    assert len(capture.shown) == 6
    assert plt.get_fignums() == []
    capture.shown.clear()
    run(capture, tables, tmp_path, show=False)
    assert capture.shown == []


def test_empty_tables_are_rejected_after_creating_the_directory(capture, tables, tmp_path):
    sessions, blocks = tables
    with pytest.raises(ValueError, match="No block observations available for the selected"):
        plot_trial_block_summaries(sessions, blocks.iloc[:0], tmp_path / "made")
    assert (tmp_path / "made").is_dir()
    with pytest.raises(ValueError, match="No session observations available for the selected"):
        plot_trial_block_summaries(sessions.iloc[:0], blocks, tmp_path / "made")


# ---------------------------------------------------------------- distributions


def test_distribution_figure_layout_for_session_counts(capture, tables, tmp_path):
    _, figures = run(capture, tables, tmp_path)
    fig = figures["Trials per session - distributions"]
    assert tuple(fig.get_size_inches()) == (17, 5) and len(fig.axes) == 3
    assert fig.get_suptitle() == ("Trials per session: distributions\nEach point is one session.")
    assert fig._suptitle.get_fontsize() == 13
    boxes, low, high = fig.axes
    assert boxes.get_title() == "Distribution by mouse"
    assert boxes.get_ylabel() == "Trials per session"
    labels = boxes.get_xticklabels()
    assert [t.get_text() for t in labels] == [
        "sub-01\nm1\nn=3",
        "sub-02\nm2\nn=5",
        "sub-05\nm3\nn=1",
    ]
    assert all(t.get_rotation() == 45 and t.get_fontsize() == 9 for t in labels)
    assert boxes.get_position().width > 1.5 * low.get_position().width  # width ratio 2 : 1 : 1
    assert (low.get_title(), high.get_title()) == ("sub-01–04 (n=8)", "sub-05–08 (n=1)")
    for ax in (low, high):
        assert ax.get_xlabel() == "Trials per session"
        assert ax.get_ylabel() == "Cumulative fraction" and ax.get_ylim() == (0, 1.03)


def test_scatter_points_use_each_mouses_colour(capture, tables, tmp_path):
    _, figures = run(capture, tables, tmp_path)
    boxes = figures["Trials per session - distributions"].axes[0]
    scatter = [c for c in boxes.collections if len(c.get_offsets()) > 0]
    assert [len(c.get_offsets()) for c in scatter] == [3, 5, 1]
    colors = [matplotlib.colors.to_hex(c.get_facecolor()[0]).upper() for c in scatter]
    assert colors == [c.upper() for c in MOUSE_COLORS[:3]]
    assert all(c.get_alpha() == 0.4 for c in scatter)
    assert scatter[0].get_offsets()[:, 1].tolist() == [24, 24, 24]
    # points are jittered horizontally by at most 0.15 around the box position
    for position, points in enumerate(scatter, start=1):
        offsets = np.abs(points.get_offsets()[:, 0] - position)
        assert offsets.max() <= 0.15 and offsets.max() > 0


def test_group_ecdfs_show_each_mouse_and_the_pooled_observations(capture, tables, tmp_path):
    _, figures = run(capture, tables, tmp_path)
    low = figures["Trials per session - distributions"].axes[1]
    m1, m2, pooled = low.lines
    assert (m1.get_label(), m1.get_color(), m1.get_alpha()) == ("m1", MOUSE_COLORS[0], 0.65)
    assert (m2.get_label(), m2.get_color()) == ("m2", MOUSE_COLORS[1])
    assert m1.get_drawstyle() == "steps-post"
    assert pooled.get_label() == "Pooled observations"
    assert pooled.get_color() == "black" and pooled.get_linewidth() == 2
    assert pooled.get_xdata().tolist() == [16] * 5 + [24] * 3
    assert [t.get_text() for t in low.get_legend().get_texts()] == [
        "m1",
        "m2",
        "Pooled observations",
    ]


def test_block_distributions_use_the_block_note(capture, tables, tmp_path):
    _, figures = run(capture, tables, tmp_path)
    fig = figures["Trials per block - distributions"]
    assert fig.get_suptitle() == (
        "Trials per block: distributions\nBlocks include observed portions at problem edges."
    )


def test_without_subject_groups_one_pooled_panel_is_drawn(capture, tables, tmp_path):
    _, figures = run(capture, tables, tmp_path, split_subject_groups=False)
    fig = figures["Trials per session - distributions"]
    assert tuple(fig.get_size_inches()) == (13, 5) and len(fig.axes) == 2
    assert fig.axes[1].get_title() == "All mice (n=9)"


def test_group_without_mice_has_no_pooled_curve_or_legend(capture, tmp_path):
    sessions, blocks = build_trial_block_tables(PROBLEMS, {"m1": 1, "m2": 2, "m3": 3})
    _, figures = run(capture, (sessions, blocks), tmp_path)
    high = figures["Trials per session - distributions"].axes[2]
    assert high.get_title() == "sub-05–08 (n=0)" and len(high.lines) == 0
    assert high.get_legend() is None


# ---------------------------------------------------------------- over time


def test_over_time_plots_daily_means_per_mouse(capture, tables, tmp_path):
    _, figures = run(capture, tables, tmp_path)
    fig = figures["Trials per session - over time"]
    (ax,) = fig.axes
    assert tuple(fig.get_size_inches()) == (12, 5)
    assert ax.get_xlabel() == "Calendar date"
    assert ax.get_ylabel() == "Mean trials per session on that date"
    assert ax.get_title() == "Trials per session over time"
    lines = {ln.get_label(): ln for ln in ax.lines}
    assert list(lines) == ["m1", "m2", "m3"]
    assert lines["m2"].get_ydata().tolist() == [16] * 5  # three problem-1 and two problem-2 dates
    assert pd.to_datetime(lines["m1"].get_xdata()).day.tolist() == [
        1,
        3,
        11,
    ]  # one point per date, in calendar order
    assert lines["m1"].get_color() == MOUSE_COLORS[0] and lines["m1"].get_alpha() == 0.8
    assert lines["m1"].get_marker() == "." and lines["m1"].get_linestyle() == "-"
    assert ax.get_legend()._ncols == 4
    assert ax.get_legend().get_texts()[0].get_fontsize() == 9


def test_block_over_time_uses_the_block_end_date_label(capture, tables, tmp_path):
    _, figures = run(capture, tables, tmp_path)
    ax = figures["Trials per block - over time"].axes[0]
    assert ax.get_xlabel() == "Block end date (last observed trial)"
    assert ax.get_ylabel() == "Mean trials per block on that date"


# ---------------------------------------------------------------- across problems


def test_problem_figure_shows_mouse_means_and_mean_sem_with_mouse_counts(capture, tables, tmp_path):
    _, figures = run(capture, tables, tmp_path)
    fig = figures["Trials per session - across problems"]
    (ax,) = fig.axes
    assert tuple(fig.get_size_inches()) == (9, 5)
    assert ax.get_xlabel() == "Problem (number of mice)"
    assert ax.get_ylabel() == "Mean trials per session"
    assert ax.get_title() == "Trials per session across problems — equal weight per mouse"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["1\n(n=3)", "2\n(n=2)"]
    mouse_lines = [ln for ln in ax.lines if ln.get_label() in ("m1", "m2", "m3")]
    assert [ln.get_ydata().tolist() for ln in mouse_lines] == [[24, 24], [16, 16], [24]]
    assert [ln.get_alpha() for ln in mouse_lines] == [0.5] * 3
    (errorbar,) = [c for c in ax.containers if isinstance(c, ErrorbarContainer)]
    assert errorbar.get_label() == "Across mice: mean ± SEM"
    mean_line = errorbar.lines[0]
    assert mean_line.get_color() == "black" and mean_line.get_linewidth() == 2.5
    np.testing.assert_allclose(np.asarray(mean_line.get_ydata(), dtype=float), [64 / 3, 20])
    sem = np.std([24, 16, 24], ddof=1) / np.sqrt(3), np.std([24, 16], ddof=1) / np.sqrt(2)
    segments = errorbar.lines[2][0].get_segments()
    half_lengths = [float(seg[1, 1] - seg[0, 1]) / 2 for seg in segments]
    np.testing.assert_allclose(half_lengths, sem)
    assert ax.get_legend()._ncols == 3
    assert ax.get_legend().get_texts()[0].get_fontsize() == 8


# ---------------------------------------------------------------- styling that is easy to lose


def test_every_figure_is_tight_laid_out(capture, tables, tmp_path):
    _, figures = run(capture, tables, tmp_path)
    assert len(figures) == 6 and all(was_tight_laid_out(fig) for fig in figures.values())


def test_mouse_scatter_jitter_is_seeded_so_figures_repeat(capture, tables, tmp_path):
    _, figures = run(capture, tables, tmp_path)
    boxes = figures["Trials per session - distributions"].axes[0]
    scatter = [c for c in boxes.collections if len(c.get_offsets()) > 0]
    rng = np.random.default_rng(0)  # one generator per figure, drawn in mouse order
    for position, points in enumerate(scatter, start=1):
        expected = position + rng.uniform(-0.15, 0.15, len(points.get_offsets()))
        np.testing.assert_array_equal(points.get_offsets()[:, 0], expected)
    assert all(list(c.get_sizes()) == [10] for c in scatter)


def test_box_plots_hide_outliers_because_every_observation_is_a_point(capture, tmp_path):
    sessions, blocks = single_mouse_tables([10, 10, 10, 10, 10, 100])
    _, figures = run(capture, (sessions, blocks), tmp_path)
    boxes = figures["Trials per session - distributions"].axes[0]
    assert [ln for ln in boxes.lines if ln.get_marker() == "o"] == []  # no flier markers
    scatter = [c for c in boxes.collections if len(c.get_offsets()) > 0]
    assert sorted(scatter[0].get_offsets()[:, 1]) == [10, 10, 10, 10, 10, 100]


def test_distribution_panel_widths_follow_the_2_1_1_ratio(capture, tables, tmp_path):
    _, figures = run(capture, tables, tmp_path)
    boxes, low, high = figures["Trials per session - distributions"].axes
    widths = [ax.get_position().width for ax in (boxes, low, high)]
    assert widths[0] / widths[1] == pytest.approx(2) and widths[1] == pytest.approx(widths[2])


def test_over_time_dates_are_rotated_to_fit(capture, tables, tmp_path):
    _, figures = run(capture, tables, tmp_path)
    (ax,) = figures["Trials per session - over time"].axes
    labels = ax.get_xticklabels()
    assert all(t.get_rotation() == 30 and t.get_ha() == "right" for t in labels)


def test_problem_means_use_capped_error_bars(capture, tables, tmp_path):
    _, figures = run(capture, tables, tmp_path)
    (ax,) = figures["Trials per session - across problems"].axes
    (errorbar,) = [c for c in ax.containers if isinstance(c, ErrorbarContainer)]
    assert cap_size(errorbar) == 4


def test_problem_means_draw_the_mean_without_error_bar_for_a_single_mouse():
    per_mouse = pd.DataFrame(
        {"mouse": ["m1", "m2", "m1"], "problem": [1, 1, 2], "trials": [10.0, 14.0, 7.0]}
    )
    summary = summarize_trials_by_problem(per_mouse)
    _, ax = plt.subplots()
    _plot_problem_means(ax, per_mouse, summary, {"m1": "C0", "m2": "C1"})
    (errorbar,) = [c for c in ax.containers if isinstance(c, ErrorbarContainer)]
    assert errorbar.lines[0].get_ydata().tolist() == [12.0, 7.0]
    first, second = errorbar.lines[2][0].get_segments()
    assert first[:, 1].tolist() == [10.0, 14.0]
    assert len(second) == 0
    plt.close(ax.figure)
