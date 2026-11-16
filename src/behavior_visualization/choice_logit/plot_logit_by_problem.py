"""Figures of the choice logit fitted per problem through structured and drifting training.

Drawing only: the numbers come from :mod:`src.behavior_analysis.choice_logit.logit_by_problem`.
Structured problems (S1, S2, ...) come first on the x axis, then drifting ones (D1, D2, ...),
with a dashed line at the switch. Means are not joined across the switch.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.behavior_analysis.choice_logit import choice_logit as cl
from src.behavior_analysis.choice_logit import logit_by_problem as lbp
from src.behavior_analysis.data.stats import nan_mean, nan_standard_error
from src.behavior_visualization.choice_logit.plot_choice_logit import mice_stats
from src.behavior_visualization.common.plot_helpers import (
    Title,
    despine,
    error_drawn,
    mouse_color_map,
    save_or_show,
    set_title,
)
from src.behavior_visualization.value.plot_value_psychometrics import finish_figure, new_figure

WEIGHTS_BY_PROBLEM = "Logit Weights by Problem"
KERNELS_BY_PROBLEM = "Logit Kernels by Problem"
FIGURE_NAMES = (WEIGHTS_BY_PROBLEM, KERNELS_BY_PROBLEM)
# The figures of the paper's coding carry this suffix (file names avoid the "×" sign).
CODING_SUFFIX = {lbp.MAGNITUDE: "", lbp.OUTCOME: " (Outcome x Choice)"}
MIN_MEAN_MICE = 3  # a problem with fewer fitted mice shows its mice but no mean
GROUP_HEADINGS = {
    cl.REWARD: "Reward",
    lbp.OUTCOME_X_CHOICE: "Outcome × Choice",
    cl.CHOICE: "Choice",
}
SUMMARY_HEADINGS = {lbp.LAG_ONE: "Lag 1", lbp.OLDER: "Mean of Lags 2+"}
MEASURE_TEXT = {
    f"slope_{lbp.STRUCTURED}": "slope per structured problem",
    f"slope_{lbp.DRIFTING}": "slope per drifting problem",
    "step": "step at the switch (first drifting minus last structured)",
}
PHASE_LETTER = {lbp.STRUCTURED: "S", lbp.DRIFTING: "D"}


def problem_positions(table: pd.DataFrame) -> dict:
    """``{(phase, phase_problem): x}``: structured problems from 1, drifting after a gap of one.

    Args:
        table (pd.DataFrame): Any table with ``phase`` and ``phase_problem``.

    Returns:
        dict: The x position of every problem in *table*.
    """
    structured = sorted(table.loc[table["phase"] == lbp.STRUCTURED, "phase_problem"].unique())
    drifting = sorted(table.loc[table["phase"] == lbp.DRIFTING, "phase_problem"].unique())
    offset = (max(structured) if structured else 0) + 1
    positions = {(lbp.STRUCTURED, k): float(k) for k in structured}
    positions.update({(lbp.DRIFTING, k): float(offset + k) for k in drifting})
    return positions


def _switch_x(positions: dict) -> float | None:
    """The x of the switch line, halfway between the phases; ``None`` without both."""
    structured = [x for (phase, _), x in positions.items() if phase == lbp.STRUCTURED]
    drifting = [x for (phase, _), x in positions.items() if phase == lbp.DRIFTING]
    if not structured or not drifting:
        return None
    return (max(structured) + min(drifting)) / 2


def _tick_labels(table: pd.DataFrame, positions: dict) -> tuple[list[float], list[str]]:
    """Tick positions and labels: the problem (S1, D1, ...) over the number of mice fitted."""
    counts = table.groupby(["phase", "phase_problem"])["mouse"].nunique()
    keys = sorted(positions, key=positions.get)
    labels = [f"{PHASE_LETTER[phase]}{k}\n{counts.get((phase, k), 0)}" for phase, k in keys]
    return [positions[key] for key in keys], labels


def _mark_switch(ax: plt.Axes, positions: dict) -> None:
    """Draw the dashed line at the switch from structured to drifting problems."""
    x = _switch_x(positions)
    if x is not None:
        ax.axvline(x, color="0.4", ls="--", lw=1, zorder=1)


def _phase_means(
    ax: plt.Axes, rows: pd.DataFrame, positions: dict, value: str = "weight"
) -> list[np.ndarray]:
    """Mean ± se across mice per problem, one line per phase; returns the se drawn."""
    errors = []
    for phase in lbp.PHASES:
        mine = rows[rows["phase"] == phase]
        if mine.empty:
            continue
        matrix = mine.pivot(index="mouse", columns="phase_problem", values=value)
        matrix = matrix.loc[:, matrix.notna().sum() >= MIN_MEAN_MICE]
        if matrix.empty:
            continue
        x = np.array([positions[(phase, k)] for k in matrix.columns])
        mean = nan_mean(matrix.to_numpy(), axis=0)
        se = nan_standard_error(matrix.to_numpy())
        if error_drawn(se):
            ax.fill_between(x, mean - se, mean + se, color="black", alpha=0.2, lw=0, zorder=3)
            errors.append(se)
        ax.plot(x, mean, "-o", color="black", lw=2.2, ms=4.5, zorder=5)
    return errors


def _mouse_lines(ax: plt.Axes, rows: pd.DataFrame, positions: dict, colors: dict) -> None:
    """One thin line per mouse and phase, not joined across the switch."""
    for (mouse, _), mine in rows.groupby(["mouse", "phase"]):
        mine = mine.sort_values("phase_problem")
        x = [positions[(p, k)] for p, k in zip(mine["phase"], mine["phase_problem"], strict=True)]
        ax.plot(x, mine["weight"], "-", color=colors[mouse], lw=0.9, alpha=0.45, zorder=2)


def _test_lines(tests: pd.DataFrame) -> list[str]:
    """Caption lines with each trend test: mean ± se across mice, t and p."""
    lines = []
    for row in tests.itertuples():
        heading = f"{GROUP_HEADINGS[row.group]}, {SUMMARY_HEADINGS[row.summary].lower()}"
        if row.n_mice == 0:
            lines.append(f"{heading}, {MEASURE_TEXT[row.measure]}: no mouse has it.")
            continue
        se = f" ± se {row.se:.3f}" if np.isfinite(row.se) else ""
        test = f", t={row.t:.2f}, p={row.p:.3g}" if np.isfinite(row.p) else ""
        lines.append(
            f"{heading}, {MEASURE_TEXT[row.measure]}: {row.mean:.3f}{se} across mice "
            f"(n={row.n_mice} mice{test}; one-sample t-test against 0)."
        )
    return lines


def figure_name(base: str, coding: str) -> str:
    """The name of one of ``FIGURE_NAMES`` for a coding: the paper's coding adds a suffix."""
    return f"{base}{CODING_SUFFIX[coding]}"


def lagged_groups(table: pd.DataFrame) -> list[str]:
    """The lagged groups in *table*, history first (reward or outcome x choice), then choice."""
    present = set(table["group"])
    return [g for g in GROUP_HEADINGS if g in present]


def _coding_lines(coding: str, n_lags: int) -> tuple[str, ...]:
    """Caption lines that say how the history is coded."""
    if coding == lbp.OUTCOME:
        return (
            f"Per port and lag (1 to {n_lags}): choice (1 if chosen) and outcome x choice (+1 if "
            "chosen and paid anything, -1 if chosen and paid 0, else 0), as in Samborska et al. "
            "2022; the magnitude is ignored. The choice weight is the average pull of a past "
            "choice, rewarded or not; the outcome x choice weight is half the difference between "
            "a rewarded and an unrewarded one (win-stay versus lose-shift). The paper's "
            "outcome-alone regressor is the same for all three ports, so it cancels.",
        )
    return (
        f"Per port and lag (1 to {n_lags}): choice (1 if chosen) and reward (the pulses received "
        "there). The reward weight is the analogue of the paper's outcome x choice term, per "
        "pulse; the choice weight is the pull of a past choice that paid 0.",
        "Structured problems pay 0, 1 or 4 pulses (sometimes 3), drifting problems 0 to 4, so a "
        "step in the reward weight at the switch can reflect the payouts as well as learning; the "
        "slopes within a phase compare like with like.",
    )


def _method_lines(n_lags: int, l2: float, coding: str) -> tuple[str, ...]:
    """Caption lines shared by both figures: the model, the problem selection and the drawing."""
    return (
        f"A three-arm choice logit (a bias per tower; L2 {l2:g}) fitted to each mouse's problems "
        f"one at a time, problems with at least {lbp.MIN_TRIALS} trials. Structured problems (S) "
        "are the blocked problems before a mouse's first drifting problem (D); later blocked "
        "problems and drifting problems after the cohort's cutoff are left out.",
        *_coding_lines(coding, n_lags),
        "The number under each problem label is the number of mice fitted. Thin lines: single "
        f"mice; black: mean ± se across mice, drawn for problems with at least {MIN_MEAN_MICE} "
        "mice and not joined across the switch (dashed line).",
    )


def plot_weights_by_problem(
    summaries: pd.DataFrame,
    tests: pd.DataFrame,
    save_path: str | Path | None = None,
    *,
    n_lags: int = cl.DEFAULT_LAGS,
    l2: float = cl.DEFAULT_L2,
    coding: str = lbp.MAGNITUDE,
) -> None:
    """History and choice weights at lag 1 and older lags against problem, as Fig. 1k,l.

    Args:
        summaries (pd.DataFrame): :func:`logit_by_problem.weight_summaries`.
        tests (pd.DataFrame): :func:`logit_by_problem.trend_tests`, written into the caption.
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.
        n_lags (int): Lags of the fits, for the caption. Default ``choice_logit.DEFAULT_LAGS``.
        l2 (float): Penalty of the fits, for the caption. Default ``choice_logit.DEFAULT_L2``.
        coding (str): The coding of the fits (``logit_by_problem.CODINGS``), for the title and
            caption. Default ``logit_by_problem.MAGNITUDE``.

    Raises:
        ValueError: If *summaries* is empty.
    """
    if summaries.empty:
        raise ValueError("no weights to plot")
    positions = problem_positions(summaries)
    colors = mouse_color_map(sorted(summaries["mouse"].unique()))
    ticks, labels = _tick_labels(summaries, positions)
    width = max(9.0, 0.42 * len(positions) * 2 + 3.0)
    with new_figure(nrows=2, ncols=2, figsize=(width, 7.4), sharex=True) as (fig, axs):
        for i, group in enumerate(lagged_groups(summaries)):
            for j, summary in enumerate((lbp.LAG_ONE, lbp.OLDER)):
                ax = axs[i, j]
                rows = summaries[(summaries["group"] == group) & (summaries["summary"] == summary)]
                _mouse_lines(ax, rows, positions, colors)
                errors = _phase_means(ax, rows, positions)
                _mark_switch(ax, positions)
                ax.axhline(0, color="0.7", lw=0.8, zorder=1)
                heading = f"{GROUP_HEADINGS[group]}, {SUMMARY_HEADINGS[summary]}"
                n_mice = rows["mouse"].nunique()
                set_title(ax, Title(heading, mice_stats(n_mice, *errors)), fontsize=12)
                ax.set_xticks(ticks, labels, fontsize=8)
                ax.set_ylabel(f"{GROUP_HEADINGS[group]} Weight (Logit)", fontsize=12)
                despine(ax)
        for ax in axs[1]:
            ax.set_xlabel("Problem (Mice Fitted)", fontsize=12)
        for mouse, color in colors.items():
            axs[0, 1].plot([], [], "-", color=color, lw=1.2, label=mouse)
        axs[0, 1].plot([], [], "-o", color="black", lw=2.2, ms=4.5, label="Mean ± se")
        axs[0, 1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8)
        caption = (*_method_lines(n_lags, l2, coding),
                   "Lags 2+: the mean weight of lags 2 to the last lag.",
                   "Per mouse, the least-squares slope of the weight against the problem number "
                   f"within a phase (from {lbp.MIN_PROBLEMS} problems), and the step at the "
                   "switch:", *_test_lines(tests))  # fmt: skip
        finish_figure(fig, Title(figure_name(WEIGHTS_BY_PROBLEM, coding), None, caption))
    save_or_show(fig, save_path or None)


def plot_kernels_by_problem(
    weights: pd.DataFrame,
    save_path: str | Path | None = None,
    *,
    n_lags: int = cl.DEFAULT_LAGS,
    l2: float = cl.DEFAULT_L2,
    coding: str = lbp.MAGNITUDE,
) -> None:
    """The history and choice kernels (lags 1 to n) of every problem in turn, as Fig. 1i,j.

    Args:
        weights (pd.DataFrame): :func:`logit_by_problem.fit_by_problem`.
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.
        n_lags (int): Lags of the fits. Default ``choice_logit.DEFAULT_LAGS``.
        l2 (float): Penalty of the fits, for the caption. Default ``choice_logit.DEFAULT_L2``.
        coding (str): The coding of the fits, for the title and caption. Default
            ``logit_by_problem.MAGNITUDE``.

    Raises:
        ValueError: If *weights* is empty.
    """
    lagged = weights[weights["lag"].notna()]
    if lagged.empty:
        raise ValueError("no weights to plot")
    counts = lagged.groupby(["phase", "phase_problem"])["mouse"].nunique()
    positions = problem_positions(lagged)
    order = [key for key in sorted(positions, key=positions.get) if counts[key] >= MIN_MEAN_MICE]
    step = n_lags + 2  # lags of one problem, then a gap
    with new_figure(nrows=2, ncols=1, figsize=(max(9.0, 0.5 * len(order) * 2), 6.8)) as (
        fig,
        axs,
    ):
        for ax, group in zip(axs, lagged_groups(lagged), strict=True):
            errors = []
            for i, (phase, k) in enumerate(order):
                rows = lagged[(lagged["group"] == group) & (lagged["phase"] == phase)
                              & (lagged["phase_problem"] == k)]  # fmt: skip
                matrix = rows.pivot(index="mouse", columns="lag", values="weight").sort_index(
                    axis=1
                )
                x = i * step + matrix.columns.to_numpy(dtype=float)
                mean = nan_mean(matrix.to_numpy(), axis=0)
                se = nan_standard_error(matrix.to_numpy())
                color = "black" if phase == lbp.STRUCTURED else "0.45"
                if error_drawn(se):
                    ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.2, lw=0)
                    errors.append(se)
                ax.plot(x, mean, "-", color=color, lw=1.6)
            centers = [i * step + (n_lags + 1) / 2 for i in range(len(order))]
            labels = [f"{PHASE_LETTER[p]}{k}\n{counts[(p, k)]}" for p, k in order]
            ax.set_xticks(centers, labels, fontsize=8)
            n_structured = sum(p == lbp.STRUCTURED for p, _ in order)
            if 0 < n_structured < len(order):
                ax.axvline(n_structured * step - 0.5, color="0.4", ls="--", lw=1)
            ax.axhline(0, color="0.7", lw=0.8)
            n_mice = lagged.loc[lagged["group"] == group, "mouse"].nunique()
            heading = f"{GROUP_HEADINGS[group]} Kernel, Lags 1-{n_lags} per Problem"
            set_title(ax, Title(heading, mice_stats(n_mice, *errors)), fontsize=12)
            ax.set_ylabel(f"{GROUP_HEADINGS[group]} Weight (Logit)", fontsize=12)
            despine(ax)
        axs[1].set_xlabel("Problem (Mice Fitted)", fontsize=12)
        caption = (*_method_lines(n_lags, l2, coding),
                   f"Each problem's kernel is drawn over lags 1 to {n_lags} (left to right); "
                   "black: structured problems, gray: drifting problems; the number under each "
                   f"label is the number of mice fitted. Problems with fewer than {MIN_MEAN_MICE} "
                   "fitted mice are left out.")  # fmt: skip
        finish_figure(fig, Title(figure_name(KERNELS_BY_PROBLEM, coding), None, caption))
    save_or_show(fig, save_path or None)


def plot_logit_by_problem_figures(
    weights: pd.DataFrame,
    summaries: pd.DataFrame,
    tests: pd.DataFrame,
    out_dir: str | Path,
    *,
    n_lags: int = cl.DEFAULT_LAGS,
    l2: float = cl.DEFAULT_L2,
    coding: str = lbp.MAGNITUDE,
) -> None:
    """Save both figures of one coding into *out_dir* (named by :func:`figure_name`).

    Args:
        weights (pd.DataFrame): :func:`logit_by_problem.fit_by_problem`.
        summaries (pd.DataFrame): :func:`logit_by_problem.weight_summaries`.
        tests (pd.DataFrame): :func:`logit_by_problem.trend_tests`.
        out_dir (str | Path): Folder of the figures.
        n_lags (int): Lags of the fits. Default ``choice_logit.DEFAULT_LAGS``.
        l2 (float): Penalty of the fits. Default ``choice_logit.DEFAULT_L2``.
        coding (str): The coding of the fits. Default ``logit_by_problem.MAGNITUDE``.
    """
    out_dir = Path(out_dir)
    options = dict(n_lags=n_lags, l2=l2, coding=coding)
    plot_weights_by_problem(
        summaries, tests, out_dir / figure_name(WEIGHTS_BY_PROBLEM, coding), **options
    )
    plot_kernels_by_problem(weights, out_dir / figure_name(KERNELS_BY_PROBLEM, coding), **options)


__all__ = [
    "FIGURE_NAMES",
    "KERNELS_BY_PROBLEM",
    "WEIGHTS_BY_PROBLEM",
    "figure_name",
    "lagged_groups",
    "plot_kernels_by_problem",
    "plot_logit_by_problem_figures",
    "plot_weights_by_problem",
    "problem_positions",
]
