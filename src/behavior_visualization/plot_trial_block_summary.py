"""Six cohort plots of observed session and block trial counts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.behavior_analysis.trial_count_summaries import (
    daily_mean_trials,
    ecdf,
    mean_trials_by_problem_and_mouse,
    mice_by_subject_number,
    summarize_trials_by_problem,
)
from src.behavior_visualization.plot_helpers import mouse_color_map, save_png_and_pdf

_SUBJECT_GROUPS = ("sub-01–04", "sub-05–08")
_ALL_MICE = ("All mice",)
_BOX_JITTER = 0.15  # half-width of the horizontal scatter around each box


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def _plot_mouse_distributions(
    ax: plt.Axes, table: pd.DataFrame, mice: list[str], colors: dict, ylabel: str
) -> None:
    """Box plot per mouse with its observations as jittered points; x labels name each mouse."""
    arrays = [table.loc[table.mouse == m, "trials"].to_numpy() for m in mice]
    ax.boxplot(arrays, showfliers=False)
    rng = np.random.default_rng(0)
    labels = []
    for i, (mouse, values) in enumerate(zip(mice, arrays, strict=True), 1):
        ax.scatter(
            i + rng.uniform(-_BOX_JITTER, _BOX_JITTER, len(values)),
            values,
            s=10,
            alpha=0.4,
            color=colors[mouse],
        )
        number = int(table.loc[table.mouse == mouse, "subject_number"].iloc[0])
        labels.append(f"sub-{number:02d}\n{mouse}\nn={len(values)}")
    ax.set_xticks(range(1, len(mice) + 1), labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title("Distribution by mouse")


def _plot_group_ecdfs(
    ax: plt.Axes, group_table: pd.DataFrame, group: str, colors: dict, xlabel: str
) -> None:
    """One ECDF per mouse plus the pooled ECDF of the group."""
    for mouse, rows in group_table.groupby("mouse"):
        x, y = ecdf(rows.trials.to_numpy())
        ax.step(x, y, where="post", color=colors[mouse], alpha=0.65, label=mouse)
    x, y = ecdf(group_table.trials.to_numpy())
    if len(x):
        ax.step(x, y, where="post", color="black", lw=2, label="Pooled observations")
        ax.legend(fontsize=8)
    ax.set(
        title=f"{group} (n={len(x)})", xlabel=xlabel, ylabel="Cumulative fraction", ylim=(0, 1.03)
    )


def _plot_daily_means(ax: plt.Axes, daily: pd.DataFrame, colors: dict) -> None:
    for mouse, rows in daily.groupby("mouse"):
        ax.plot(rows.date, rows.trials, ".-", color=colors[mouse], alpha=0.8, label=mouse)


def _plot_problem_means(
    ax: plt.Axes, per_mouse: pd.DataFrame, summary: pd.DataFrame, colors: dict
) -> None:
    """Per-mouse means by problem (faint) and the across-mouse mean ± SEM (black)."""
    for mouse, rows in per_mouse.groupby("mouse"):
        ax.plot(rows.problem, rows.trials, "o-", color=colors[mouse], alpha=0.5, label=mouse)
    ax.errorbar(
        summary.index,
        summary["mean"],
        yerr=summary["sem"].fillna(0),
        fmt="o-",
        color="black",
        lw=2.5,
        capsize=4,
        label="Across mice: mean ± SEM",
    )
    ax.set_xticks(summary.index, [f"{p}\n(n={int(row.n_mice)})" for p, row in summary.iterrows()])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _distribution_figure(
    table: pd.DataFrame, unit: str, mice: list[str], colors: dict, split_subject_groups: bool
) -> plt.Figure:
    title = f"Trials per {unit}"
    note = (
        "Blocks include observed portions at problem edges."
        if unit == "block"
        else "Each point is one session."
    )
    groups = _SUBJECT_GROUPS if split_subject_groups else _ALL_MICE
    fig, axes = plt.subplots(
        1,
        1 + len(groups),
        figsize=(17 if split_subject_groups else 13, 5),
        gridspec_kw={"width_ratios": [2] + [1] * len(groups)},
    )
    _plot_mouse_distributions(axes[0], table, mice, colors, title)
    for ax, group in zip(axes[1:], groups, strict=True):
        # Known issue: with split_subject_groups, mice numbered above 8 have group
        # "Other", which matches neither panel, so they are left out of both ECDFs.
        group_table = table[table.group == group] if split_subject_groups else table
        _plot_group_ecdfs(ax, group_table, group, colors, title)
    fig.suptitle(f"{title}: distributions\n{note}", fontsize=13)
    return fig


def _over_time_figure(table: pd.DataFrame, unit: str, colors: dict) -> plt.Figure:
    title = f"Trials per {unit}"
    fig, ax = plt.subplots(figsize=(12, 5))
    _plot_daily_means(ax, daily_mean_trials(table), colors)
    ax.set(
        xlabel="Calendar date" if unit == "session" else "Block end date (last observed trial)",
        ylabel=f"Mean trials per {unit} on that date",
        title=f"{title} over time",
    )
    ax.legend(fontsize=9, ncol=4)
    fig.autofmt_xdate()
    return fig


def _problem_figure(
    per_mouse: pd.DataFrame, summary: pd.DataFrame, unit: str, colors: dict
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_problem_means(ax, per_mouse, summary, colors)
    ax.set(
        xlabel="Problem (number of mice)",
        ylabel=f"Mean trials per {unit}",
        title=f"Trials per {unit} across problems — equal weight per mouse",
    )
    ax.legend(fontsize=8, ncol=3)
    return fig


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _plot_unit(
    unit: str,
    table: pd.DataFrame,
    mice: list[str],
    colors: dict,
    save_dir: Path,
    *,
    show: bool,
    split_subject_groups: bool,
) -> list[Path]:
    """Write the CSVs and save the three figures (distributions, over time, by problem)."""
    table.to_csv(save_dir / f"{unit}_trial_counts.csv", index=False)
    title = f"Trials per {unit}"
    paths = []
    fig = _distribution_figure(table, unit, mice, colors, split_subject_groups)
    paths += save_png_and_pdf(fig, save_dir, f"{title} - distributions", show=show)
    fig = _over_time_figure(table, unit, colors)
    paths += save_png_and_pdf(fig, save_dir, f"{title} - over time", show=show)

    per_mouse = mean_trials_by_problem_and_mouse(table)
    summary = summarize_trials_by_problem(per_mouse)
    summary.to_csv(save_dir / f"{unit}_trials_by_problem_summary.csv")
    per_mouse.to_csv(save_dir / f"{unit}_trials_by_problem_per_mouse.csv", index=False)
    fig = _problem_figure(per_mouse, summary, unit, colors)
    paths += save_png_and_pdf(fig, save_dir, f"{title} - across problems", show=show)
    return paths


def plot_trial_block_summaries(
    sessions, blocks, save_dir, show=True, split_subject_groups=True
) -> list[Path]:
    """Save PNG/PDF figures plus source tables and equal-mouse problem summaries.

    Distribution panels show individual observations per mouse and pooled group
    ECDFs. Time plots show each mouse (daily means when multiple observations
    share a date). Problem plots average within mouse first, then show mean ± SEM
    across mice; n is the number of mice. Blocks include observed edge portions.

    Returns:
        The saved PNG and PDF paths, in figure order (PNG first for each figure).

    Known issue: a figure that raises while being drawn or saved stays open.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    tables = {"session": sessions, "block": blocks}
    for unit, table in tables.items():
        if table.empty:
            raise ValueError(f"No {unit} observations available for the selected problems")
    mice = mice_by_subject_number(sessions, blocks)
    colors = mouse_color_map(mice)
    paths = []
    for unit, table in tables.items():
        paths += _plot_unit(
            unit,
            table,
            mice,
            colors,
            save_dir,
            show=show,
            split_subject_groups=split_subject_groups,
        )
    return paths
