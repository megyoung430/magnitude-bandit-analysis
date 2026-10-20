"""Figures of the sessions a mouse needed to complete each drifting problem.

:func:`plot_sessions_per_problem` draws one line per mouse over the drifting problems, with the
mean ± SE across mice of the completed problems in thick black. A problem the mouse is still
running is a hollow point joined by a dashed segment; it is left out of the mean.
:func:`plot_sessions_per_problem_by_cohort` draws one mean line per cohort instead, for many
mice.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from src.behavior_analysis.sessions_per_problem import SessionsPerProblem
from src.behavior_visualization.plot_helpers import (
    axes_or_new,
    despine,
    jitter_rng,
    jittered,
    mouse_color_map,
    palette_color,
    save_or_show,
)

MEAN_LABEL = "Mean ± SE (completed problems)"
RUNNING_LABEL = "Still running"
_JITTER = 0.06  # keeps mice with the same count from drawing over each other
_MEAN_COLOR = "black"
_FIGSIZE = (11, 6)
_LEGEND = {"loc": "upper left", "bbox_to_anchor": (1.01, 1.0), "frameon": True, "fontsize": 10}


def _count_phrase(summary: SessionsPerProblem) -> str:
    return (
        "bandit + port-finding-only sessions" if summary.counts_port_finding else "bandit sessions"
    )


def _style_axes(ax: plt.Axes, problems: list[int], labels: list[str], top: float) -> None:
    ax.set_xticks(problems)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_xlim(min(problems) - 0.5, max(problems) + 0.5)
    ax.set_xlabel("Drifting problem")
    ax.set_ylabel("Number of sessions")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, top)
    ax.tick_params(axis="y", labelsize=13)
    despine(ax)


def _top(*arrays: np.ndarray) -> float:
    """An upper y-limit just above the largest finite value."""
    finite = np.concatenate([a[np.isfinite(a)] for a in arrays])
    return float(finite.max()) * 1.08 + 0.5 if finite.size else 1.0


def _draw_mean(ax: plt.Axes, problems, mean, se, *, color, linewidth, label) -> None:
    """The mean as a line over the problems that have one, with error bars where an SE exists.

    Drawn under everything else (zorder 2), so a mouse's own line stays visible on top of it.
    """
    x = np.asarray(problems, dtype=float)
    has_mean, has_se = np.isfinite(mean), np.isfinite(se)
    ax.plot(
        x[has_mean],
        mean[has_mean],
        color=color,
        linewidth=linewidth,
        marker="o",
        markersize=8,
        label=label,
        zorder=2,
    )
    ax.errorbar(
        x[has_se],
        mean[has_se],
        yerr=se[has_se],
        fmt="none",
        ecolor=color,
        elinewidth=2,
        capsize=6,
        zorder=2,
    )


def _draw_mouse(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray, done: np.ndarray, color: str, mouse: str
) -> None:
    """A solid line through the completed problems, a dashed hollow end for the running one."""
    finite = np.isfinite(y)
    completed, running = finite & done, finite & ~done
    ax.plot(
        x[completed],
        y[completed],
        color=color,
        linewidth=2.5,
        alpha=0.9,
        marker="o",
        markersize=6,
        label=mouse,
        zorder=4,
    )
    if running.any():
        # join the running point to the last completed one before it
        before = np.flatnonzero(completed & (np.arange(len(x)) < np.flatnonzero(running)[0]))
        if before.size:
            joined = [before[-1], *np.flatnonzero(running)]
            ax.plot(
                x[joined],
                y[joined],
                color=color,
                linestyle="--",
                linewidth=2,
                alpha=0.9,
                zorder=3,
                label=f"_{mouse} running segment",
            )
        ax.plot(
            x[running],
            y[running],
            color=color,
            linestyle="none",
            marker="o",
            markersize=8,
            markerfacecolor="white",
            markeredgewidth=2,
            zorder=5,
            label=f"_{mouse} running",
        )


def _legend_handles(summary: SessionsPerProblem, colors: dict, mouse_legend: bool) -> list:
    handles = []
    if mouse_legend:
        handles += [
            Line2D([0], [0], color=colors[m], lw=2.5, marker="o", markersize=6, label=m)
            for m in summary.mice
        ]
    handles.append(Line2D([0], [0], color=_MEAN_COLOR, lw=4, marker="o", label=MEAN_LABEL))
    if any(
        (~done & np.isfinite(summary.by_mouse[m])).any() for m, done in summary.complete.items()
    ):
        handles.append(
            Line2D(
                [0],
                [0],
                color="gray",
                ls="--",
                lw=2,
                marker="o",
                markersize=8,
                markerfacecolor="white",
                markeredgewidth=2,
                label=RUNNING_LABEL,
            )
        )
    return handles


def plot_sessions_per_problem_panel(
    summary: SessionsPerProblem,
    *,
    scope_label: str,
    ax: plt.Axes | None = None,
    mouse_legend: bool = True,
) -> plt.Axes:
    """Draw sessions per drifting problem: a line per mouse and the mean of the completed ones.

    The x tick labels give the number of mice behind each mean. The jitter is seeded, so the
    same summary always draws the same figure.

    Args:
        summary (SessionsPerProblem): From ``summarize_sessions_per_problem``.
        scope_label (str): The cohort(s) named in the title, e.g. ``"cohort 3"``.
        ax (plt.Axes | None): Axes to draw into; a new 11 x 6 inch figure is created if None.
            Default None.
        mouse_legend (bool): Name the mice in the legend; their lines are drawn either way.
            Switch off for many mice. Default True.

    Returns:
        plt.Axes: The axes drawn into.
    """
    x = np.asarray(summary.problems, dtype=float)
    colors = mouse_color_map(summary.mice)
    rng = jitter_rng()
    with axes_or_new(ax, figsize=_FIGSIZE) as ax:
        for mouse in summary.mice:
            _draw_mouse(
                ax,
                jittered(x, _JITTER, rng),
                summary.by_mouse[mouse],
                summary.complete[mouse],
                colors[mouse],
                mouse,
            )
        _draw_mean(
            ax, x, summary.mean, summary.se, color=_MEAN_COLOR, linewidth=4, label=MEAN_LABEL
        )
        labels = [f"{k}\nn={n}" for k, n in zip(summary.problems, summary.n, strict=True)]
        _style_axes(
            ax,
            summary.problems,
            labels,
            _top(*summary.by_mouse.values(), summary.mean + np.nan_to_num(summary.se)),
        )
        ax.set_title(
            "Sessions per Drifting Problem\n"
            f"({scope_label} | {_count_phrase(summary)} | mean ± se across mice, "
            "completed problems only)",
            fontsize=14,
            pad=15,
        )
        ax.legend(handles=_legend_handles(summary, colors, mouse_legend), **_LEGEND)
    return ax


def plot_sessions_per_problem(
    summary: SessionsPerProblem,
    save_path: str | Path | None,
    scope_label: str,
    *,
    mouse_legend: bool = True,
) -> None:
    """Save (or show) the figure of :func:`plot_sessions_per_problem_panel`.

    Args:
        summary (SessionsPerProblem): From ``summarize_sessions_per_problem``.
        save_path (str | Path | None): Base path without extension for the ``.pdf`` and ``.png``
            files (the parent folder is created); ``None`` shows the figure instead.
        scope_label (str): The cohort(s) named in the title, e.g. ``"cohort 3"``.
        mouse_legend (bool): As for :func:`plot_sessions_per_problem_panel`. Default True.
    """
    ax = plot_sessions_per_problem_panel(
        summary, scope_label=scope_label, mouse_legend=mouse_legend
    )
    ax.figure.tight_layout()
    save_or_show(ax.figure, save_path)


def _mice_phrase(n: int) -> str:
    return f"{n} mouse" if n == 1 else f"{n} mice"


def _cohort_label(cohort: str, summary: SessionsPerProblem) -> str:
    mice = _mice_phrase(len(summary.mice))
    return f"{cohort} ({mice}{'' if summary.n.any() else '; no completed problem'})"


def _annotate_n(ax: plt.Axes, problems: list[int], summary: SessionsPerProblem, color: str) -> None:
    """The number of mice behind each point of a cohort's mean, just above the point."""
    for k, mean, n in zip(problems, summary.mean, summary.n, strict=True):
        if np.isfinite(mean):
            ax.annotate(
                f"n={n}",
                (k, mean),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=9,
                color=color,
            )


def plot_sessions_per_problem_by_cohort_panel(
    summaries: Mapping[str, SessionsPerProblem], *, scope_label: str, ax: plt.Axes | None = None
) -> plt.Axes:
    """Draw one mean ± SE line per cohort over the drifting problems (completed problems only).

    Args:
        summaries (Mapping[str, SessionsPerProblem]): ``{cohort name: summary}``, all on the
            same problem axis (pass ``problems=`` to ``summarize_sessions_per_problem``).
        scope_label (str): The cohorts named in the title, e.g. ``"cohorts 1, 3"``.
        ax (plt.Axes | None): Axes to draw into; a new 11 x 6 inch figure is created if None.
            Default None.

    Returns:
        plt.Axes: The axes drawn into.

    Raises:
        ValueError: If there are no cohorts, or the summaries differ in problem axis or in the
            sessions they count.
    """
    if not summaries:
        raise ValueError("no cohorts to draw")
    axes_of = {tuple(s.problems) for s in summaries.values()}
    if len(axes_of) > 1:
        raise ValueError(f"the cohort summaries must share one problem axis, got {sorted(axes_of)}")
    problems = list(next(iter(summaries.values())).problems)
    if len({s.counts_port_finding for s in summaries.values()}) > 1:
        raise ValueError("the cohort summaries must all count the same sessions")
    phrase = _count_phrase(next(iter(summaries.values())))
    with axes_or_new(ax, figsize=_FIGSIZE) as ax:
        for i, (cohort, summary) in enumerate(summaries.items()):
            color = palette_color(i)
            _draw_mean(
                ax,
                problems,
                summary.mean,
                summary.se,
                color=color,
                linewidth=3,
                label=_cohort_label(cohort, summary),
            )
            _annotate_n(ax, problems, summary, color)
        _style_axes(
            ax,
            problems,
            [str(k) for k in problems],
            _top(*(s.mean + np.nan_to_num(s.se) for s in summaries.values())),
        )
        ax.set_title(
            "Sessions per Drifting Problem by Cohort\n"
            f"({scope_label} | {phrase} | mean ± se across mice, completed problems only)",
            fontsize=14,
            pad=15,
        )
        ax.legend(**_LEGEND)
    return ax


def plot_sessions_per_problem_by_cohort(
    summaries: Mapping[str, SessionsPerProblem], save_path: str | Path | None, scope_label: str
) -> None:
    """Save (or show) the figure of :func:`plot_sessions_per_problem_by_cohort_panel`.

    Args:
        summaries (Mapping[str, SessionsPerProblem]): ``{cohort name: summary}``.
        save_path (str | Path | None): Base path without extension for the ``.pdf`` and ``.png``
            files (the parent folder is created); ``None`` shows the figure instead.
        scope_label (str): The cohorts named in the title.
    """
    ax = plot_sessions_per_problem_by_cohort_panel(summaries, scope_label=scope_label)
    ax.figure.tight_layout()
    save_or_show(ax.figure, save_path)
