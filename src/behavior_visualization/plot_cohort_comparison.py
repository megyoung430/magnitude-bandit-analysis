"""Matched-axis descriptive comparisons of two cohorts (mice are independent)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.behavior_analysis.cohort_comparison_stats import (
    cap_problem_two,
    cohort_tables,
    good_reversal_rank_counts,
    mean_and_sem,
    mouse_rank_cell,
    mouse_reversal_cell,
    require_problems,
)
from src.behavior_analysis.trial_count_summaries import (
    ecdf,
    mean_trials_by_problem_and_mouse,
    summarize_trials_by_problem,
)
from src.behavior_visualization.plot_helpers import (
    FigureOutput,
    TableOutput,
    palette_color,
    write_outputs,
)

_RANK_NAMES = ("Best", "Second", "Third")
_REVERSAL_NAMES = ("Good", "Bad", "All")
_CATEGORIES = range(3)

# Known issue: panels colour mice by their position k within the panel (palette_color(k)),
# so the same mouse can get a different colour in another panel or problem.


def _label_height(value: float, error: float) -> float:
    """Height for a mean's text label: just above the top of its error bar."""
    return value + (error if np.isfinite(error) else 0) + 0.035


def _comparison_grid(labels: list[str], rows, title: str) -> tuple[plt.Figure, np.ndarray]:
    """One row per problem, one column per cohort; panels share axes within a row."""
    fig, axes = plt.subplots(
        len(rows), 2, figsize=(13, 4 * len(rows)), sharex="row", sharey="row", squeeze=False
    )
    fig.suptitle(title, fontsize=16)
    for i, p in enumerate(rows):
        for j, label in enumerate(labels):
            axes[i, j].set_title(f"{label} — problem {p}")
    return fig, axes


def _problem_rows(table: pd.DataFrame, problem) -> pd.DataFrame:
    return table[table.problem == problem]


# ---------------------------------------------------------------------------
# Trial-count panels
# ---------------------------------------------------------------------------


def _plot_distribution_panel(ax: plt.Axes, t: pd.DataFrame, unit: str) -> None:
    """Faint per-mouse ECDFs under a bold pooled ECDF."""
    for k, (_, rows) in enumerate(t.groupby("mouse")):
        x, y = ecdf(rows.trials.to_numpy())
        ax.step(x, y, where="post", alpha=0.35, color=palette_color(k))
    x, y = ecdf(t.trials.to_numpy())
    ax.step(
        x, y, where="post", color="black", lw=2, label=f"{t.mouse.nunique()} mice; {len(t)} {unit}s"
    )
    ax.set(xlabel=f"Trials per {unit}", ylabel="Cumulative fraction", ylim=(0, 1.03))
    ax.legend(fontsize=9)


def _plot_over_time_panel(
    ax: plt.Axes, t: pd.DataFrame, sessions: pd.DataFrame, problem, unit: str
) -> None:
    """Daily mean trials per mouse against days since its first session in the problem."""
    for k, (mouse, rows) in enumerate(t.groupby("mouse")):
        first = sessions[(sessions.problem == problem) & (sessions.mouse == mouse)]
        origin = first.date.min()
        daily = rows.groupby("date").trials.mean()
        ax.plot(
            (daily.index - origin).days,
            daily.values,
            ".-",
            alpha=0.8,
            label=mouse,
            color=palette_color(k),
        )
    ax.set(
        xlabel="Days since first observed session in this problem",
        ylabel=f"Daily mean trials per {unit}",
    )
    ax.legend(fontsize=8, ncol=2)


def _plot_across_problems_panel(
    ax: plt.Axes, per_mouse: pd.DataFrame, problems, label: str, unit: str
) -> None:
    """Per-mouse means by problem (faint) and the mean ± SEM across mice (black).

    Known issue: a problem with no data in this cohort raises ``KeyError`` when its
    mouse count is looked up.
    """
    for k, (mouse, rows) in enumerate(per_mouse.groupby("mouse")):
        ax.plot(rows.problem, rows.trials, "o-", alpha=0.45, label=mouse, color=palette_color(k))
    means = summarize_trials_by_problem(per_mouse)
    ax.errorbar(
        means.index,
        means["mean"],
        yerr=means["sem"].fillna(0),
        color="black",
        fmt="o-",
        capsize=4,
        lw=2,
        label="Mean ± SEM across mice",
    )
    ax.set_xticks(list(problems), [f"{p}\n(n={int(means.loc[p, 'n_mice'])})" for p in problems])
    ax.set(title=label, xlabel="Problem (number of mice)", ylabel=f"Mean trials per {unit}")
    ax.legend(fontsize=8, ncol=2)


# ---------------------------------------------------------------------------
# Trial-count figures
# ---------------------------------------------------------------------------


def _distributions_figure(tables: dict, labels, problems, unit: str) -> plt.Figure:
    fig, axes = _comparison_grid(labels, problems, f"Trials per {unit}: distributions (full data)")
    for i, p in enumerate(problems):
        for j, label in enumerate(labels):
            _plot_distribution_panel(axes[i, j], _problem_rows(tables[label][unit], p), unit)
    return fig


def _over_time_figure(tables: dict, labels, problems, unit: str) -> plt.Figure:
    fig, axes = _comparison_grid(labels, problems, f"Trials per {unit}: over time (full data)")
    for i, p in enumerate(problems):
        for j, label in enumerate(labels):
            t = _problem_rows(tables[label][unit], p)
            _plot_over_time_panel(axes[i, j], t, tables[label]["session"], p, unit)
    return fig


def _across_problems_outputs(
    tables: dict, labels, problems, unit: str
) -> Iterator[TableOutput | FigureOutput]:
    """Each cohort's per-mouse means table, then the figure of mean trials by problem.

    A cohort's table is yielded before its panel is drawn, so it is written even if
    the panel raises.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    for j, label in enumerate(labels):
        per_mouse = mean_trials_by_problem_and_mouse(tables[label][unit])
        yield TableOutput(f"{label}_{unit}_means_by_mouse.csv", per_mouse)
        _plot_across_problems_panel(axes[j], per_mouse, problems, label, unit)
    fig.suptitle(f"Trials per {unit}: across problems (equal weight per mouse)")
    yield FigureOutput(f"Trials per {unit} - across problems", fig)


def _trial_count_outputs(
    tables: dict, labels, problems, unit: str
) -> Iterator[TableOutput | FigureOutput]:
    """The three trial-count figures (and per-mouse means tables) for *unit*, in order."""
    name = f"Trials per {unit}"
    fig = _distributions_figure(tables, labels, problems, unit)
    yield FigureOutput(f"{name} - distributions", fig)
    yield FigureOutput(f"{name} - over time", _over_time_figure(tables, labels, problems, unit))
    yield from _across_problems_outputs(tables, labels, problems, unit)


# ---------------------------------------------------------------------------
# Rank-proportion and reversal-total figures
# ---------------------------------------------------------------------------


def _annotate_means(ax: plt.Axes, mean: np.ndarray, sem: np.ndarray) -> None:
    for x, value, error in zip(_CATEGORIES, mean, sem, strict=True):
        if np.isfinite(value):
            ax.text(
                x,
                _label_height(value, error),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=11,
            )


def _finish_summary_panel(
    ax: plt.Axes,
    values: list,
    *,
    names: tuple[str, ...],
    ylabel: str,
    title: str,
    annotate: bool,
) -> None:
    """Mean ± SEM over mice (bars, error bars, optional value labels), then axis text."""
    stats = mean_and_sem(values)
    if stats is not None:
        mean, sem = stats
        ax.bar(_CATEGORIES, mean, color="gray", alpha=0.2, zorder=0)
        ax.errorbar(_CATEGORIES, mean, yerr=sem, fmt="ko", capsize=4, label="Mean ± SEM")
        if annotate:
            _annotate_means(ax, mean, sem)
    ax.set_xticks(_CATEGORIES, names)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(f"{title} (n={len(values)} mice)")


def _draw_cell(
    subjects: dict, label: str, problem, rank_ax: plt.Axes, rev_ax: plt.Axes
) -> tuple[list[dict], list[dict]]:
    """Draw one cohort-by-problem cell onto both axes; returns its CSV records.

    Each mouse's values come from the pure ``mouse_*_cell`` functions, called one mouse
    at a time between draws so that a failure part-way leaves the same partly drawn
    figure as main.
    """
    counts = good_reversal_rank_counts(subjects)
    rank_values, rev_values, rank_records, rev_records = [], [], [], []
    for k, mouse in enumerate(sorted(subjects)):
        color = palette_color(k)
        rank = mouse_rank_cell(label, problem, mouse, counts[mouse])
        if rank is not None:
            rank_values.append(rank.proportions)
            rank_records.append(rank.record)
            rank_ax.plot(_CATEGORIES, rank.proportions, "o-", alpha=0.65, color=color, label=mouse)
        reversal = mouse_reversal_cell(label, problem, mouse, subjects[mouse])
        rev_values.append(reversal.values)
        rev_records.append(reversal.record)
        rev_ax.plot(_CATEGORIES, reversal.values, "o-", alpha=0.65, color=color, label=mouse)
    title = f"{label} — problem {problem}"
    _finish_summary_panel(
        rank_ax,
        rank_values,
        names=_RANK_NAMES,
        ylabel="Proportion of choices",
        title=title,
        annotate=True,
    )
    _finish_summary_panel(
        rev_ax,
        rev_values,
        names=_REVERSAL_NAMES,
        ylabel="Number of reversals",
        title=title,
        annotate=False,
    )
    rank_ax.set_ylim(0, 1)
    return rank_records, rev_records


def _rank_and_reversal_outputs(
    datasets: dict, labels, problems, suffix: str
) -> Iterator[TableOutput | FigureOutput]:
    """Rank-proportion and reversal-total figures and their tables for *datasets*."""
    rank_fig, rank_axes = _comparison_grid(
        labels, problems, f"Pooled true-value rank proportions — {suffix}"
    )
    rev_fig, rev_axes = _comparison_grid(labels, problems, f"Total reversals per mouse — {suffix}")
    rank_records, rev_records = [], []
    for i, p in enumerate(problems):
        for j, label in enumerate(labels):
            ranks, revs = _draw_cell(datasets[label][p], label, p, rank_axes[i, j], rev_axes[i, j])
            rank_records += ranks
            rev_records += revs
    yield TableOutput(f"Rank proportions - {suffix}.csv", pd.DataFrame(rank_records))
    yield TableOutput(f"Reversal totals - {suffix}.csv", pd.DataFrame(rev_records))
    yield FigureOutput(f"Rank proportions - {suffix}", rank_fig)
    yield FigureOutput(f"Reversal totals - {suffix}", rev_fig)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _comparison_outputs(
    cohorts: dict, subject_numbers: dict, problems, trial_budget
) -> Iterator[TableOutput | FigureOutput]:
    """Every table and figure of the comparison, in the order they are written."""
    labels = list(cohorts)
    tables = {}
    for label, data in cohorts.items():
        require_problems(label, data, problems)
        tables[label] = cohort_tables(data, subject_numbers[label], problems)
        for unit, table in tables[label].items():
            yield TableOutput(f"{label}_{unit}_trials.csv", table)

    for unit in ("session", "block"):
        yield from _trial_count_outputs(tables, labels, problems, unit)
    yield from _rank_and_reversal_outputs(cohorts, labels, problems, "full data")
    if 2 in problems and trial_budget is not None:
        capped, reports = cap_problem_two(cohorts, subject_numbers, labels, trial_budget)
        yield TableOutput("Problem 2 trial cap.csv", pd.DataFrame(reports))
        suffix = f"problem 2 first {trial_budget} trials (or all available)"
        yield from _rank_and_reversal_outputs(capped, labels, (2,), suffix)


def plot_cohort_comparison(
    cohorts, subject_numbers, output_dir, problems=(1, 2, 3), trial_budget=510, show=False
) -> list[Path]:
    """Save full-data comparisons, plus a separate capped problem-2 comparison.

    Columns are cohorts, rows are problems; paired panels share axes. Mouse
    means receive equal weight. Distributions pool observations within cohort;
    faint ECDFs show individual mice. Time is days since each mouse's first
    observed session in that problem. Block lengths include observed edges.
    Rank proportions use the existing good-block quality filters and pool
    retained trials within mouse. No between-cohort significance test is implied.

    Returns:
        The saved PNG and PDF paths, in figure order (PNG first for each figure).

    Known issue: a figure that raises while being drawn or saved stays open.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(cohorts) != 2:
        raise ValueError("Exactly two cohorts are required")
    outputs = _comparison_outputs(cohorts, subject_numbers, problems, trial_budget)
    return write_outputs(outputs, output_dir, show=show)
