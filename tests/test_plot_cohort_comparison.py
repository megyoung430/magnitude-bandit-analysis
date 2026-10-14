import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.container import ErrorbarContainer

from src.behavior_analysis.cohort_comparison_stats import (
    cap_problem_two,
    label_height,
    mean_and_sem,
    rank_proportions,
    reversal_values,
)
from src.behavior_visualization.plot_cohort_comparison import plot_cohort_comparison
from src.behavior_visualization.plot_style import MOUSE_COLORS
from tests.plot_capture import capture_figures
from tests.plot_fixtures import make_mouse

COHORTS = {
    "cohort-A": {
        1: {"a1": make_mouse(2), "a2": make_mouse(1, pick=1)},
        2: {"a1": make_mouse(2), "a2": make_mouse(2, pick=1)},
    },
    "cohort-B": {
        1: {"b1": make_mouse(1), "b2": make_mouse(1)},
        2: {"b1": make_mouse(1, pick=2), "b2": make_mouse(2)},
    },
}
NUMBERS = {"cohort-A": {"a1": 1, "a2": 2}, "cohort-B": {"b1": 5, "b2": 6}}
BUDGET = 30
CAPPED = f"problem 2 first {BUDGET} trials (or all available)"


# ---------------------------------------------------------------- statistics


def test_rank_proportions_pool_trials_within_a_mouse():
    rows = [
        {"best": 2, "second": 1, "third": 1, "total": 4},
        {"best": 0, "second": 2, "third": 2, "total": 4},
    ]
    assert rank_proportions(rows) == (8, [0.25, 0.375, 0.375])
    assert rank_proportions([{"best": 0, "second": 0, "third": 0, "total": 0}]) == (0, None)


def test_reversal_values_fill_missing_good_and_bad_with_nan():
    assert reversal_values({"total_reversals": 3, "good_reversals": 2, "bad_reversals": 1}) == [
        2,
        1,
        3,
    ]
    values = reversal_values({"total_reversals": 3})
    assert np.isnan(values[0]) and np.isnan(values[1]) and values[2] == 3


def test_mean_and_sem_ignore_nan_and_are_zero_for_one_mouse():
    with pytest.warns(RuntimeWarning):  # one finite value in the last column: SD undefined
        mean, sem = mean_and_sem([[1.0, 4.0, np.nan], [3.0, 6.0, 2.0]])
    np.testing.assert_allclose(mean, [2.0, 5.0, 2.0])
    np.testing.assert_allclose(
        sem[:2], [np.std([1.0, 3.0], ddof=1) / np.sqrt(2), np.std([4.0, 6.0], ddof=1) / np.sqrt(2)]
    )
    assert np.isnan(sem[2])
    mean, sem = mean_and_sem([[1.0, 2.0, 3.0]])
    assert sem.tolist() == [0, 0, 0] and mean.tolist() == [1, 2, 3]
    assert mean_and_sem([]) is None


def test_sem_denominator_counts_only_finite_mice():
    mean, sem = mean_and_sem([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [np.nan, 6.0, 7.0]])
    assert mean[0] == 2.0
    assert sem[0] == pytest.approx(np.std([1.0, 3.0], ddof=1) / np.sqrt(2))
    assert sem[1] == pytest.approx(np.std([2.0, 4.0, 6.0], ddof=1) / np.sqrt(3))


def test_label_height_sits_just_above_the_error_bar():
    assert label_height(0.5, 0.25) == pytest.approx(0.785)
    assert label_height(0.5, np.nan) == pytest.approx(0.535)


def test_problem_two_cap_reports_each_mouse_of_each_cohort():
    capped, reports = cap_problem_two(COHORTS, NUMBERS, ["cohort-A", "cohort-B"], 30)
    assert set(capped) == {"cohort-A", "cohort-B"} and set(capped["cohort-A"]) == {2}
    assert [(r["cohort"], r["mouse"], r["retained_trials"]) for r in reports] == [
        ("cohort-A", "a1", 30),
        ("cohort-A", "a2", 30),
        ("cohort-B", "b1", 24),
        ("cohort-B", "b2", 30),
    ]


# ---------------------------------------------------------------- figures


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("cohort") / "comparison"
    with pytest.MonkeyPatch.context() as monkeypatch:
        capture = capture_figures(monkeypatch)
        paths = plot_cohort_comparison(
            COHORTS, NUMBERS, out, problems=(1, 2), trial_budget=BUDGET, show=True
        )
    figures = {
        s.path.rsplit("/", 1)[-1][:-4]: s.figure for s in capture.saved if s.path.endswith(".png")
    }
    return paths, figures, capture, out


def test_paths_are_png_then_pdf_for_each_figure_in_order(run):
    paths, _, capture, out = run
    stems = [
        "Trials per session - distributions",
        "Trials per session - over time",
        "Trials per session - across problems",
        "Trials per block - distributions",
        "Trials per block - over time",
        "Trials per block - across problems",
        "Rank proportions - full data",
        "Reversal totals - full data",
        f"Rank proportions - {CAPPED}",
        f"Reversal totals - {CAPPED}",
    ]
    assert paths == [out / f"{s}.{e}" for s in stems for e in ("png", "pdf")]
    assert [s.path for s in capture.saved] == [str(p) for p in paths]
    assert all(s.kwargs == {"dpi": 180, "bbox_inches": "tight"} for s in capture.saved)


def test_each_figure_is_shown_once_and_closed(run):
    _, _, capture, _ = run
    assert len(capture.shown) == 10
    assert plt.get_fignums() == []


def test_source_tables_are_written(run):
    out = run[3]
    assert sorted(p.name for p in out.glob("*.csv")) == sorted(
        [
            f"{prefix}.csv"
            for prefix in (
                "cohort-A_block_means_by_mouse",
                "cohort-A_block_trials",
                "cohort-A_session_means_by_mouse",
                "cohort-A_session_trials",
                "cohort-B_block_means_by_mouse",
                "cohort-B_block_trials",
                "cohort-B_session_means_by_mouse",
                "cohort-B_session_trials",
                "Problem 2 trial cap",
                "Rank proportions - full data",
                "Reversal totals - full data",
                f"Rank proportions - {CAPPED}",
                f"Reversal totals - {CAPPED}",
            )
        ]
    )
    cap = pd.read_csv(out / "Problem 2 trial cap.csv")
    assert cap.retained_trials.tolist() == [30, 30, 24, 30]
    ranks = pd.read_csv(out / "Rank proportions - full data.csv")
    assert ranks.columns.tolist() == [
        "cohort",
        "problem",
        "mouse",
        "eligible_trials",
        "best",
        "second",
        "third",
    ]
    assert ranks.query("cohort == 'cohort-A' and problem == 1").mouse.tolist() == ["a1", "a2"]


def test_grid_figures_have_a_row_per_problem_and_a_column_per_cohort(run):
    _, figures, _, _ = run
    for name in ("Trials per session - distributions", "Rank proportions - full data"):
        fig = figures[name]
        assert tuple(fig.get_size_inches()) == (13, 8) and len(fig.axes) == 4
        assert fig._suptitle.get_fontsize() == 16
    fig = figures[f"Rank proportions - {CAPPED}"]
    assert tuple(fig.get_size_inches()) == (13, 4) and len(fig.axes) == 2


def test_distribution_panels_pool_observations_and_draw_faint_mouse_ecdfs(run):
    fig = run[1]["Trials per session - distributions"]
    assert fig.get_suptitle() == "Trials per session: distributions (full data)"
    a1, a2, b1, b2 = fig.axes
    assert a1.get_title() == "cohort-A — problem 1" and b2.get_title() == "cohort-B — problem 2"
    faint = a1.lines[:2]
    assert [ln.get_color() for ln in faint] == MOUSE_COLORS[:2]
    assert all(ln.get_alpha() == 0.35 for ln in faint)
    pooled = a1.lines[2]
    assert pooled.get_color() == "black" and pooled.get_linewidth() == 2
    assert pooled.get_label() == "2 mice; 3 sessions"
    assert pooled.get_xdata().tolist() == [24, 24, 24]
    assert a1.get_xlabel() == "Trials per session" and a1.get_ylabel() == "Cumulative fraction"
    assert a1.get_ylim() == (0, 1.03)
    assert a1.get_legend().get_texts()[0].get_fontsize() == 9


def test_block_distribution_axis_names_blocks(run):
    a1 = run[1]["Trials per block - distributions"].axes[0]
    assert a1.get_xlabel() == "Trials per block"
    assert a1.lines[2].get_label().endswith(" blocks")


def test_over_time_x_is_days_since_the_mouses_first_session_in_the_problem(run):
    fig = run[1]["Trials per session - over time"]
    assert fig.get_suptitle() == "Trials per session: over time (full data)"
    a1 = fig.axes[0]
    first, second = a1.lines
    assert (first.get_label(), first.get_color(), first.get_alpha()) == ("a1", MOUSE_COLORS[0], 0.8)
    assert first.get_xdata().tolist() == [0, 2] and first.get_ydata().tolist() == [24, 24]
    assert second.get_xdata().tolist() == [0]
    assert a1.get_xlabel() == "Days since first observed session in this problem"
    assert a1.get_ylabel() == "Daily mean trials per session"
    assert a1.get_legend()._ncols == 2 and a1.get_legend().get_texts()[0].get_fontsize() == 8


def test_across_problems_figure_has_one_panel_per_cohort_with_mouse_counts(run):
    fig = run[1]["Trials per session - across problems"]
    assert tuple(fig.get_size_inches()) == (13, 5) and len(fig.axes) == 2
    assert fig.get_suptitle() == "Trials per session: across problems (equal weight per mouse)"
    left, right = fig.axes
    assert (left.get_title(), right.get_title()) == ("cohort-A", "cohort-B")
    assert [t.get_text() for t in left.get_xticklabels()] == ["1\n(n=2)", "2\n(n=2)"]
    assert left.get_xlabel() == "Problem (number of mice)"
    assert left.get_ylabel() == "Mean trials per session"
    (errorbar,) = [c for c in left.containers if isinstance(c, ErrorbarContainer)]
    assert errorbar.get_label() == "Mean ± SEM across mice"
    assert errorbar.lines[0].get_color() == "black" and errorbar.lines[0].get_linewidth() == 2
    assert [ln.get_alpha() for ln in left.lines if ln.get_label() in ("a1", "a2")] == [0.45, 0.45]


def test_rank_panel_shows_pooled_proportions_with_mean_sem_and_labels(run):
    fig = run[1]["Rank proportions - full data"]
    assert fig.get_suptitle() == "Pooled true-value rank proportions — full data"
    ax = fig.axes[0]  # cohort-A, problem 1: a1 always best, a2 always second
    assert ax.get_title() == "cohort-A — problem 1 (n=2 mice)"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Best", "Second", "Third"]
    assert ax.get_ylabel() == "Proportion of choices" and ax.get_ylim() == (0, 1)
    a1, a2 = [ln for ln in ax.lines if ln.get_label() in ("a1", "a2")]
    assert a1.get_ydata().tolist() == [1, 0, 0] and a2.get_ydata().tolist() == [0, 1, 0]
    assert (a1.get_color(), a1.get_alpha(), a1.get_linestyle(), a1.get_marker()) == (
        MOUSE_COLORS[0],
        0.65,
        "-",
        "o",
    )
    assert [p.get_height() for p in ax.patches] == [0.5, 0.5, 0]
    assert all(p.get_alpha() == 0.2 and p.get_zorder() == 0 for p in ax.patches)
    (errorbar,) = [c for c in ax.containers if isinstance(c, ErrorbarContainer)]
    assert errorbar.get_label() == "Mean ± SEM"
    segments = errorbar.lines[2][0].get_segments()
    assert [float(seg[1, 1] - seg[0, 1]) / 2 for seg in segments] == pytest.approx([0.5, 0.5, 0])
    labels = [(t.get_text(), t.get_position()) for t in ax.texts]
    assert labels == [("0.50", (0, 1.035)), ("0.50", (1, 1.035)), ("0.00", (2, 0.035))]
    assert all(t.get_fontsize() == 11 for t in ax.texts)
    assert ax.get_legend()._ncols == 2


def test_reversal_panel_shows_good_bad_and_all_counts_per_mouse(run):
    fig = run[1]["Reversal totals - full data"]
    assert fig.get_suptitle() == "Total reversals per mouse — full data"
    ax = fig.axes[0]  # a1 has two sessions of two good reversals, a2 one
    assert ax.get_title() == "cohort-A — problem 1 (n=2 mice)"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Good", "Bad", "All"]
    assert ax.get_ylabel() == "Number of reversals"
    a1, a2 = [ln for ln in ax.lines if ln.get_label() in ("a1", "a2")]
    assert a1.get_ydata().tolist() == [4, 0, 4] and a2.get_ydata().tolist() == [2, 0, 2]
    assert [p.get_height() for p in ax.patches] == [3, 0, 3]
    assert len(ax.texts) == 0  # only rank panels carry value labels
    (errorbar,) = [c for c in ax.containers if isinstance(c, ErrorbarContainer)]
    segments = errorbar.lines[2][0].get_segments()
    assert [float(seg[1, 1] - seg[0, 1]) / 2 for seg in segments] == pytest.approx([1, 0, 1])


def test_capped_comparison_covers_problem_two_only(run):
    fig = run[1][f"Reversal totals - {CAPPED}"]
    assert fig.get_suptitle() == f"Total reversals per mouse — {CAPPED}"
    left, right = fig.axes
    assert (left.get_title(), right.get_title()) == (
        "cohort-A — problem 2 (n=2 mice)",
        "cohort-B — problem 2 (n=2 mice)",
    )
    # a1 keeps its first session (2 good reversals) and 6 trials of the second, before any reversal
    a1 = [ln for ln in left.lines if ln.get_label() == "a1"][0]
    assert a1.get_ydata().tolist() == [2, 0, 2]


# ---------------------------------------------------------------- argument checks


def test_exactly_two_cohorts_are_required(monkeypatch, tmp_path):
    capture_figures(monkeypatch)
    with pytest.raises(ValueError, match="Exactly two cohorts are required"):
        plot_cohort_comparison({"only": COHORTS["cohort-A"]}, NUMBERS, tmp_path / "x")
    assert (tmp_path / "x").is_dir()


def test_missing_problem_is_named(monkeypatch, tmp_path):
    capture_figures(monkeypatch)
    with pytest.raises(ValueError, match=r"cohort-A: missing problems \[3\]"):
        plot_cohort_comparison(COHORTS, NUMBERS, tmp_path / "x", problems=(1, 3))


def test_no_capped_comparison_without_budget_or_problem_two(monkeypatch, tmp_path):
    capture = capture_figures(monkeypatch)
    paths = plot_cohort_comparison(COHORTS, NUMBERS, tmp_path / "x", problems=(1,), trial_budget=30)
    assert len(paths) == 16 and not any("problem 2" in p.name for p in paths)
    assert not (tmp_path / "x" / "Problem 2 trial cap.csv").exists()
    assert capture.shown == []
