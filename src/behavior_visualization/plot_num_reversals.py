"""Plots of reversal counts per subject and over sessions.

Each panel function draws into an ``ax`` you pass (or a new figure if you
don't), so it can be one panel of a larger figure. The three ``plot_num_*``
functions compose those panels into the standard figures and save them.

- :func:`plot_num_reversals` — total good/bad (or total) reversals per subject.
- :func:`plot_num_reversals_over_time` — cumulative reversals by session,
  across mice and per mouse, plus trial-based companions.
- :func:`plot_moving_avg_reversals_over_time` — centered moving average of
  reversals per session, same layout.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.behavior_analysis.get_total_reversals import get_total_reversals
from src.behavior_analysis.reversals_by_session import (
    cumulative_by_session,
    curves_by_subject,
    mean_across_subjects,
    moving_average_by_session,
    reversal_counts_by_session,
    reversal_kinds,
    reversal_totals_by_subject,
)
from src.behavior_visualization.plot_reversals_by_trial import plot_reversals_by_trial
from src.behavior_visualization.plot_style import BAD_COLOR, GOOD_COLOR, TOTAL_COLOR

KIND_COLORS = {"good": GOOD_COLOR, "bad": BAD_COLOR, "total": TOTAL_COLOR}
KIND_LABELS = {"good": "Good Rev", "bad": "Bad Rev", "total": "Total Rev"}
THRESHOLD_STYLE = dict(color="#DCDCDC", linewidth=2, linestyle="--", alpha=0.7)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def _despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_reversal_totals(
    totals: pd.DataFrame, kinds: tuple[str, ...], ax: plt.Axes | None = None
) -> plt.Axes:
    """Bar chart of total reversals per subject (rows of *totals*)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(totals))
    if kinds == ("good", "bad"):
        bar_width = 0.35
        for offset, kind, label in [(-1, "good", "Good Reversals"), (1, "bad", "Bad Reversals")]:
            ax.bar(
                x_pos + offset * bar_width / 2,
                totals[kind].astype(int),
                width=bar_width,
                color=KIND_COLORS[kind],
                alpha=0.7,
                label=label,
                edgecolor="black",
            )
        ax.legend(loc="upper left", fontsize=10)
    else:
        ax.bar(x_pos, totals["total"].astype(int), color=TOTAL_COLOR, edgecolor="black")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(totals.index, rotation=45, ha="right")
    _despine(ax)
    ax.spines["left"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.set_ylabel("Number of Reversals", fontsize=12)
    ax.set_xlabel("Subject", fontsize=12)
    return ax


def plot_reversal_curves_across_mice(
    curves: dict[str, dict[str, np.ndarray]],
    *,
    ax: plt.Axes | None = None,
    x_start: int = 0,
    individual_alpha: float = 0.4,
    mean_styles: dict[str, tuple[str, str]] | None = None,
    threshold: float | None = None,
    ylabel: str = "",
    title: str = "",
) -> plt.Axes:
    """Faint per-mouse curves plus a bold across-mouse mean for each reversal kind.

    Args:
        curves: ``{kind: {subject: values}}`` with kinds from ``KIND_COLORS``.
        x_start: Session-axis value of the first point (0 or 1).
        mean_styles: Optional ``{kind: (color, label)}`` overrides for the mean line.
        threshold: Draw a dashed reference line at this y-value, if given.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    mean_styles = mean_styles or {}
    subjects = next(iter(curves.values())).keys()
    for subject in subjects:
        for kind, by_subject in curves.items():
            y = by_subject[subject]
            ax.plot(
                np.arange(x_start, x_start + len(y)),
                y,
                color=KIND_COLORS[kind],
                alpha=individual_alpha,
                linewidth=1.5,
            )
    for kind, by_subject in curves.items():
        mean = mean_across_subjects(by_subject)
        color, label = mean_styles.get(kind, (KIND_COLORS[kind], KIND_LABELS[kind]))
        ax.plot(
            np.arange(x_start, x_start + len(mean)), mean, color=color, linewidth=3, label=label
        )
    if threshold is not None:
        ax.axhline(threshold, label="Threshold", **THRESHOLD_STYLE)
    ax.set_xlabel("Session Number")
    ax.set_ylabel(ylabel)
    _despine(ax)
    ax.set_title(title)
    ax.legend(fontsize=10)
    return ax


def plot_reversal_curves_for_mouse(
    curves: dict[str, np.ndarray],
    session_labels: list[str],
    *,
    ax: plt.Axes | None = None,
    labels: dict[str, str] | None = None,
    threshold: float | None = None,
    show_labels: bool = True,
    ylabel: str = "",
    title: str = "",
) -> plt.Axes:
    """One mouse's reversal curves against its session labels.

    Args:
        curves: ``{kind: values}`` in session order.
        labels: Optional ``{kind: legend label}`` overrides.
        show_labels: Give lines legend labels; turn off for all but one panel
            of a grid so the shared legend lists each kind once.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    labels = {**KIND_LABELS, **(labels or {})}
    for kind, y in curves.items():
        ax.plot(y, color=KIND_COLORS[kind], label=labels[kind] if show_labels else None)
    if threshold is not None:
        ax.axhline(threshold, label="Threshold" if show_labels else None, **THRESHOLD_STYLE)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Session")
    ax.set_ylabel(ylabel)
    # at most ~10 tick labels so long training histories stay readable
    tick_idx = list(range(0, len(session_labels), max(1, len(session_labels) // 10)))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([session_labels[j] for j in tick_idx], rotation=45, ha="right")
    _despine(ax)
    return ax


# ---------------------------------------------------------------------------
# Standard figures
# ---------------------------------------------------------------------------


def _mouse_grid(n_subjects: int) -> tuple[plt.Figure, np.ndarray]:
    nrows = math.ceil(n_subjects / 2)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 3.5 * nrows), sharey=False)
    axes = np.array(axes).reshape(-1)
    for ax in axes[n_subjects:]:
        ax.axis("off")
    return fig, axes


def _by_mouse_figure(df, kinds, *, labels=None, threshold=None, ylabel, title_suffix, suptitle):
    groups = list(df.groupby("subject", sort=True))
    fig, axes = _mouse_grid(len(groups))
    for i, (subject, rows) in enumerate(groups):
        plot_reversal_curves_for_mouse(
            {kind: rows[kind].to_numpy(dtype=float) for kind in kinds},
            rows["session_label"].tolist(),
            ax=axes[i],
            labels=labels,
            threshold=threshold,
            show_labels=(i == 0),
            ylabel=ylabel,
            title=f"{subject}\n(n={len(rows)} {title_suffix})",
        )
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper right")
    fig.suptitle(suptitle, y=1.02, fontsize=14)
    fig.tight_layout()
    return fig


def _session_span(df: pd.DataFrame) -> str:
    sizes = df.groupby("subject").size()
    return f"n={len(sizes)} subjects; sessions per subject={sizes.min()}–{sizes.max()}"


def _kinds_title(kinds: tuple[str, ...]) -> str:
    return "Good and Bad" if kinds == ("good", "bad") else "Total"


def _save_or_show(figures: dict[str, plt.Figure], save_path) -> None:
    """Save each figure as ``<save_path><suffix>.pdf/.png``, or show them if no path."""
    if save_path:
        base = Path(save_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        for suffix, fig in figures.items():
            fig.savefig(f"{base}{suffix}.pdf", bbox_inches="tight")
            fig.savefig(f"{base}{suffix}.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()
    for fig in figures.values():
        plt.close(fig)


def _print_subject_totals(subjects_trials: dict) -> None:
    for subject in sorted(subjects_trials):
        print(f"{subject}: {get_total_reversals(subjects_trials[subject])}")


def plot_num_reversals(subjects_trials, save_path=None):
    """Bar chart of total good and bad (or total) reversals per subject.

    Args:
        subjects_trials: Nested dict ``{subject: {session_key: session_dict}}``.
        save_path: Base path (without extension) for ``.pdf`` and ``.png``
            output. If ``None``, the figure is shown.
    """
    _print_subject_totals(subjects_trials)
    totals = reversal_totals_by_subject(subjects_trials)
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_reversal_totals(totals, reversal_kinds(totals), ax=ax)
    fig.tight_layout()
    _save_or_show({"": fig}, save_path)


def plot_num_reversals_over_time(subjects_trials, threshold=10, save_path=None):
    """Cumulative reversals over sessions, across mice and per mouse.

    Saves ``<save_path> Across Mice`` and ``<save_path> By Mouse`` (``.pdf``
    and ``.png``), then trial-based companions with a ``(By Trial)`` suffix.
    Shorter mice are carried forward at their final count in the mean.

    Args:
        subjects_trials: Nested dict ``{subject: {session_key: session_dict}}``.
        threshold: Dashed reference line (e.g. the criterion reversal count);
            ``None`` for none.
        save_path: Base path without extension; if ``None``, figures are shown.

    Raises:
        ValueError: If there are no sessions, a session key lacks ``ses-<n>``,
            or good/bad and blocks-only sessions are mixed.
    """
    _print_subject_totals(subjects_trials)
    counts = reversal_counts_by_session(subjects_trials)
    kinds = reversal_kinds(counts)
    cumulative = cumulative_by_session(counts, kinds)
    name = _kinds_title(kinds)

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    plot_reversal_curves_across_mice(
        {kind: curves_by_subject(cumulative, kind) for kind in kinds},
        ax=ax1,
        mean_styles={"total": ("#000000", "Mean Total")},
        threshold=threshold,
        ylabel="Cumulative Reversals",
        title=f"Cumulative {name} Reversals Over Time\n({_session_span(cumulative)})",
    )
    fig1.tight_layout()
    fig2 = _by_mouse_figure(
        cumulative,
        kinds,
        labels={"total": "Total"},
        threshold=threshold,
        ylabel="Cumulative Reversals",
        title_suffix="sessions",
        suptitle=f"Cumulative {name} Reversals by Subject",
    )
    _save_or_show({" Across Mice": fig1, " By Mouse": fig2}, save_path)

    plot_reversals_by_trial(
        subjects_trials,
        threshold=threshold,
        save_path=f"{save_path} (By Trial)" if save_path else None,
    )


def plot_moving_avg_reversals_over_time(
    subjects_trials, *, window: int = 3, trial_window: int = 100, save_path=None
):
    """Centered moving average of reversals per session, across mice and per mouse.

    Same layout and file names as :func:`plot_num_reversals_over_time`. The
    trial-based companions use a trailing *trial_window* and report reversals
    per 100 trials (suffix ``(By Trial, <trial_window>-Trial Window)``).

    Args:
        subjects_trials: Nested dict ``{subject: {session_key: session_dict}}``.
        window: Width of the centered session moving average; it shrinks at
            the ends of each mouse's history (default 3).
        trial_window: Trailing trial window for the companion plots (default 100).
        save_path: Base path without extension; if ``None``, figures are shown.

    Raises:
        ValueError: As for :func:`plot_num_reversals_over_time`.
    """
    counts = reversal_counts_by_session(subjects_trials)
    kinds = reversal_kinds(counts)
    smoothed = moving_average_by_session(counts, kinds, window)
    name = _kinds_title(kinds)

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    plot_reversal_curves_across_mice(
        {kind: curves_by_subject(smoothed, kind) for kind in kinds},
        ax=ax1,
        x_start=1,
        individual_alpha=0.25,
        ylabel="Moving-Average Reversals / Session",
        title=(
            f"Moving-Average {name} Reversals Over Time\n"
            f"({_session_span(smoothed)}; window={window})"
        ),
    )
    fig1.tight_layout()
    fig2 = _by_mouse_figure(
        smoothed,
        kinds,
        ylabel="Moving-Avg Reversals / Session",
        title_suffix=f"sessions; window={window}",
        suptitle=f"Moving-Average {name} Reversals by Subject",
    )
    _save_or_show({" Across Mice": fig1, " By Mouse": fig2}, save_path)

    plot_reversals_by_trial(
        subjects_trials,
        window=trial_window,
        save_path=f"{save_path} (By Trial, {trial_window}-Trial Window)" if save_path else None,
    )
