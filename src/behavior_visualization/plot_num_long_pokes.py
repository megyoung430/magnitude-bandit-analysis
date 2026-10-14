"""Visualise non-choice long-poke counts across sessions and subjects.

A "long poke" is an exploratory poke into a port that is not the animal's
final choice for that trial.  This module provides a session-by-session bar
chart with per-mouse overlaid lines.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.behavior_analysis.long_pokes_by_session import PokesBySession, summarize_pokes_by_session
from src.behavior_import.sort_by_session import get_short_session_label
from src.behavior_visualization.plot_helpers import despine, mouse_color_map, save_or_show

_BAR_COLOR = "#999999"
_JITTER = 0.03


def _plot_session_bars(ax: plt.Axes, x: np.ndarray, summary: PokesBySession, y_text: float) -> None:
    """Bars of the across-mouse mean with SE caps, each labelled with its value at *y_text*."""
    ax.bar(
        x,
        summary.mean,
        yerr=summary.se,
        capsize=6,
        color=_BAR_COLOR,
        alpha=0.55,
        edgecolor="black",
        linewidth=1.5,
        zorder=1,
    )
    for xi, m in zip(x, summary.mean, strict=True):
        if np.isfinite(m):
            ax.text(
                xi,
                y_text,
                f"{m:.2f}",
                ha="center",
                va="bottom",
                fontsize=12,
                color="black",
                zorder=10,
            )


def _plot_mouse_lines(
    ax: plt.Axes, x: np.ndarray, summary: PokesBySession, rng: np.random.Generator | None
) -> list[Line2D]:
    """One jittered line per mouse with data; returns matching legend handles."""
    mouse_to_color = mouse_color_map(summary.mice)
    handles = []
    rng = np.random.default_rng() if rng is None else rng
    for m in summary.mice:
        c = mouse_to_color[m]
        y = summary.by_mouse[m]
        mask = np.isfinite(y)
        if not np.any(mask):
            continue
        xs = x[mask].astype(float)
        ys = y[mask].astype(float)
        xs = xs + rng.uniform(-_JITTER, _JITTER, size=len(xs))
        ax.plot(xs, ys, color=c, linewidth=2.5, alpha=0.9, marker="o", markersize=6, zorder=4)
        handles.append(Line2D([0], [0], color=c, lw=2.5, marker="o", markersize=6, label=m))
    return handles


def _style_axes(ax: plt.Axes, x: np.ndarray, summary: PokesBySession, y_text: float) -> None:
    # Known issue: the title's subject count includes mice without finite data,
    # which are not drawn.
    ax.set_xlabel("Session")
    ax.set_ylabel("Non-Choice Long Pokes")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [get_short_session_label(s) for s in summary.sessions], fontsize=12, rotation=45
    )
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylim(0, y_text)
    despine(ax)
    title = (
        "Mean Non-Choice Long Pokes Across Sessions\n"
        f" (mean ± se across subjects | n={len(summary.mice)} subjects)"
    )
    ax.set_title(title, fontsize=16, pad=25)


def plot_pokes_by_session(
    avg_num_pokes_across_subjects: dict,
    y_text: float,
    *,
    ax: plt.Axes | None = None,
    rng: np.random.Generator | None = None,
) -> plt.Axes:
    """Draw mean non-choice long pokes per session with per-mouse overlays.

    Args:
        avg_num_pokes_across_subjects: ``{mouse: {session_key: mean_pokes}}``; see
            :func:`plot_num_pokes_across_mice`.
        y_text: Y-coordinate for bar-value annotations and the upper y-limit.
        ax: Axes to draw into; a new 12 x 5.5 inch figure is created if ``None``.
        rng: Generator for the per-mouse x-jitter. Known issue: when ``None`` an
            unseeded generator is used, so the jitter differs on every call.

    Known issue: when the figure is created here, it stays open if drawing raises.
    """
    summary = summarize_pokes_by_session(avg_num_pokes_across_subjects)
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(summary.sessions))
    _plot_session_bars(ax, x, summary, y_text)
    handles = _plot_mouse_lines(ax, x, summary, rng)
    _style_axes(ax, x, summary, y_text)
    if handles:
        ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=10)
    return ax


def plot_num_pokes_across_mice(
    avg_num_pokes_across_subjects: dict, y_text: float, save_path: str | Path | None = None
) -> None:
    """Plot mean non-choice long-poke counts per session with per-mouse overlays.

    Draws a bar for each session (mean ± SE across mice) and overlays
    individual mouse lines with jitter.  Sessions are sorted by numeric
    ``ses-XX`` index; x-tick labels show the short ``ses-XX`` form.

    Args:
        avg_num_pokes_across_subjects: Nested dict
            ``{mouse: {session_key: mean_pokes_float}}``, optionally including
            an ``"average_across_subjects"`` key which is excluded from the
            per-mouse overlay.
        y_text: Y-coordinate for bar-value annotations and the upper y-limit
            of the axis.
        save_path: Base path (without extension) for saving ``.pdf`` and
            ``.png`` output.  If ``None``, the figure is shown interactively.

    Known issue: the parent directory of *save_path* is not created, so saving
    into a missing directory raises (and leaves the figure open).
    """
    ax = plot_pokes_by_session(avg_num_pokes_across_subjects, y_text)
    fig = ax.figure
    fig.tight_layout()
    # An empty save_path means "show" here, so it is passed on as None.
    save_or_show(fig, save_path or None, make_parent=False)
