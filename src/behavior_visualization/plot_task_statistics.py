"""Visualise block-length statistics across mice and sessions.

Provides a bar chart of per-block median lengths with per-mouse overlaid
lines, drawn from the summary dict returned by
:func:`src.behavior_analysis.get_task_statistics.get_block_lengths`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.behavior_visualization.plot_helpers import (
    Title,
    axes_or_new,
    despine,
    error_drawn,
    jitter_rng,
    jittered,
    mouse_color_map,
    save_or_show,
    set_title,
    stats_line,
)

_BAR_COLOR = "#999999"


# ---------------------------------------------------------------------------
# Layout geometry (pure functions of the summary arrays)
# ---------------------------------------------------------------------------


def _tallest_bar(meds: np.ndarray, ses: np.ndarray) -> float:
    """Top of the highest error bar, ``max(median + SE)``; 1.0 when there are no bars.

    A NaN SE (fewer than two values) adds no error: the bar's median still counts.

    Known issue: when every median + SE is NaN the result is NaN, which makes the
    figure's y-limits invalid and plotting raises.
    """
    return float(np.nanmax(meds + np.nan_to_num(ses))) if meds.size else 1.0


def _annotation_height(meds: np.ndarray, ses: np.ndarray) -> float:
    """Default y for the bar-value labels: 5% above the tallest error bar (at least 1)."""
    return max(_tallest_bar(meds, ses), 1.0) * 1.05


def _y_upper_limit(annotate_y: float, meds: np.ndarray, ses: np.ndarray) -> float:
    """Upper y-limit that fits both the labels at *annotate_y* and the tallest bar."""
    return max(annotate_y, _tallest_bar(meds, ses)) * 1.08


def _mouse_points(blocklens: dict, blocks: list) -> tuple[np.ndarray, np.ndarray]:
    """Positions (index into *blocks*) and lengths of the blocks one mouse has data for.

    Args:
        blocklens: ``{block_number: length}`` for one mouse.
        blocks: All block numbers, in plotting order.
    """
    positions, lengths = [], []
    for position, block in enumerate(blocks):
        if block in blocklens:
            positions.append(position)
            lengths.append(blocklens[block])
    return np.asarray(positions, float), np.asarray(lengths, float)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _plot_median_bars(
    ax: plt.Axes, x: np.ndarray, meds: np.ndarray, ses: np.ndarray, n_blocks: int
) -> None:
    ax.bar(
        x,
        meds,
        yerr=ses,
        capsize=6,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.55,
        color=[_BAR_COLOR] * n_blocks,
        zorder=1,
    )


def _annotate_medians(ax: plt.Axes, x: np.ndarray, meds: np.ndarray, annotate_y: float) -> None:
    # Known issue: ``meds`` is not checked against the number of blocks; a mismatch
    # silently labels only the shorter of the two (``ax.bar`` broadcasts a length-1
    # ``meds``), so the zip is deliberately not strict.
    for xi, m in zip(x, meds, strict=False):
        ax.text(xi, annotate_y, f"{m:.1f}", ha="center", va="bottom", fontsize=11, clip_on=False)


def _plot_mouse_lines(
    ax: plt.Axes,
    per_mouse_blocklens: dict,
    mice: list,
    mouse_to_color: dict,
    blocks: list,
    jitter: float,
) -> list[Line2D]:
    """One connected line per mouse over the blocks it has data for; returns legend handles.

    The jitter comes from one seeded generator, so the x positions are the same on every run.
    """
    handles = []
    rng = jitter_rng()
    for m in mice:
        c = mouse_to_color[m]
        xs, ys = _mouse_points(per_mouse_blocklens[m], blocks)
        if len(xs) == 0:
            continue
        xs = jittered(xs, jitter, rng)
        ax.plot(xs, ys, color=c, linewidth=2.5, alpha=0.9, marker="o", markersize=6, zorder=3)
        handles.append(
            Line2D([0], [0], color=c, lw=2.5, marker="o", label=f"{m} (n={len(ys)} blocks)")
        )
    return handles


def _style_axes(ax: plt.Axes, x: np.ndarray, blocks: list, n_mice: int, ses: np.ndarray) -> None:
    # Known issue: n_mice counts every mouse in the summary, including those without
    # data that are not drawn.
    despine(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in blocks], fontsize=12)
    ax.set_xlabel("Block Number", fontsize=12)
    ax.set_ylabel("Block Length (Trials)", fontsize=12)
    average = "median ± se" if error_drawn(ses) else "median"
    stats = stats_line(average, "mice", [(n_mice, "mouse", "mice")])
    set_title(ax, Title("Block Lengths", stats), pad=20)


def plot_block_length_panel(
    block_lengths_summary: dict,
    *,
    jitter: float = 0.06,
    annotate_y: float | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Draw per-block median lengths (bars, ± SE) with per-mouse overlay lines.

    Args:
        block_lengths_summary (dict): Dict as returned by
            :func:`src.behavior_analysis.get_task_statistics.get_block_lengths`, with keys
            ``"blocks"`` (block numbers), ``"meds"`` and ``"ses"`` (one value per block),
            ``"per_mouse_blocklens"`` (``{mouse: {block: length}}``) and ``"mice"``.
        jitter (float): Maximum half-width of the uniform x-jitter on per-mouse points;
            0 disables it. The jitter is seeded, so it is the same on every run. Default 0.06.
        annotate_y (float | None): Y of the bar-value labels; derived from the tallest bar if
            None. Default None.
        ax (plt.Axes | None): Axes to draw into; a new 10 x 4.8 inch figure is created if None.
            Default None.

    Returns:
        plt.Axes: The axes drawn into.

    Raises:
        KeyError: If the summary lacks one of the keys above.
    """
    blocks = block_lengths_summary["blocks"]
    meds = np.asarray(block_lengths_summary["meds"], dtype=float)
    ses = np.asarray(block_lengths_summary["ses"], dtype=float)
    per_mouse_blocklens = block_lengths_summary["per_mouse_blocklens"]
    mice = block_lengths_summary["mice"]
    mouse_to_color = mouse_color_map(mice)
    x = np.arange(len(blocks))
    with axes_or_new(ax, figsize=(10, 4.8)) as ax:
        _plot_median_bars(ax, x, meds, ses, len(blocks))
        if annotate_y is None:
            annotate_y = _annotation_height(meds, ses)
        _annotate_medians(ax, x, meds, annotate_y)
        handles = _plot_mouse_lines(ax, per_mouse_blocklens, mice, mouse_to_color, blocks, jitter)
        ax.legend(handles=handles, fontsize=10, loc="upper right")
        _style_axes(ax, x, blocks, len(mice), ses)
        ax.set_ylim(0, _y_upper_limit(annotate_y, meds, ses))
    return ax


def plot_block_lengths(
    block_lengths_summary: dict,
    jitter: float = 0.06,
    annotate_y: float | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Plot per-block median lengths with per-mouse overlay lines.

    Draws a bar (median ± se across mice) for each block number and overlays
    connected per-mouse data points with optional x-jitter.

    Args:
        block_lengths_summary (dict): Dict as returned by
            :func:`src.behavior_analysis.get_task_statistics.get_block_lengths`,
            containing keys ``"blocks"``, ``"meds"``, ``"ses"``,
            ``"per_mouse_blocklens"``, and ``"mice"``.
        jitter (float): Maximum half-width of the uniform x-jitter applied to
            per-mouse data points. Set to 0 to disable. Default 0.06.
        annotate_y (float | None): Y-coordinate for bar-value text annotations. If None,
            the value is derived automatically from the tallest bar. Default None.
        save_path (str | Path | None): Base path (without extension) for saving ``.pdf`` and
            ``.png`` output. If None or empty, the figure is shown interactively. Default None.

    Raises:
        KeyError: If the summary lacks one of the keys above.
    """
    ax = plot_block_length_panel(block_lengths_summary, jitter=jitter, annotate_y=annotate_y)
    fig = ax.figure
    fig.tight_layout()
    # An empty save_path means "show" here, so it is passed on as None.
    save_or_show(fig, save_path or None)
