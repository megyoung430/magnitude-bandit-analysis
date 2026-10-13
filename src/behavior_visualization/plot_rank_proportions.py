"""Plotting functions for arm-rank choice proportions around good reversals.

Three views are available, all accessible through the top-level
:func:`plot_rank_proportions` dispatcher:

- *By Mouse* — one panel per subject, blocks as coloured lines.
- *By Block* — one panel per block number, subjects as coloured lines.
- *Average* — a two-panel summary: left panel averages across mice (mouse
  colours), right panel averages across blocks (block colours).

Each panel is a function that draws into an ``ax`` you pass (or a new figure if
you don't), so it can be one panel of a larger figure; the ``plot_rank_proportions*``
functions arrange those panels and save or show the figure. The numbers behind
the panels come from :mod:`src.behavior_analysis.rank_proportions`.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.behavior_analysis.rank_proportions import (
    RANKS,
    mean_and_se_by_rank,
    mouse_block_rows,
    n_plotted_blocks,
    prop_units,
    proportions_by_block,
    proportions_by_mouse,
    row_props,
    rows_in_block,
    subject_rank_proportions,
)
from src.behavior_visualization.plot_style import MOUSE_COLORS

BLOCK_COLORS = [
    "#1B9E77",
    "#D95F02",
    "#7570B3",
    "#E7298A",
    "#66A61E",
    "#E6AB02",
    "#A6761D",
    "#666666",
    "#1F78B4",
    "#33A02C",
    "#FB9A99",
    "#E31A1C",
    "#FDBF6F",
    "#CAB2D6",
    "#6A3D9A",
    "#FFFF99",
    "#B15928",
    "#A6CEE3",
    "#B2DF8A",
    "#FF7F00",
]


def plot_rank_proportions(
    rank_counts_by_good_reversal,
    average_across_mice_pvalues=None,
    save_path=None,
    by_mouse=True,
    by_block=True,
    average=True,
    pool_trials_within_subject=False,
):
    """Dispatcher that creates one or more rank-proportion figures.

    Calls the appropriate sub-functions based on the boolean flags.
    Each sub-function saves a separate file (or shows interactively).

    Args:
        rank_counts_by_good_reversal: ``{subject: list[rank_count_dict]}`` as
            returned by
            :func:`src.behavior_analysis.get_rank_counts_by_good_reversal.get_rank_counts_by_good_reversal`.
        average_across_mice_pvalues: Optional dict with keys
            ``"best_vs_second"``, ``"best_vs_third"``, ``"second_vs_third"``
            (floats) to annotate the average-across-mice panel.  If ``None``,
            no p-value annotations are added (default: ``None``).
        save_path: Base path (without extension) shared by all sub-functions.
            Each sub-function appends its own suffix before the file extension.
            If ``None``, figures are shown interactively.
        by_mouse: If ``True``, produce the by-mouse panel grid (default:
            ``True``).
        by_block: If ``True``, produce the by-block panel grid (default:
            ``True``).
        average: If ``True``, produce the two-panel average summary (default:
            ``True``).
    """
    if by_mouse:
        mouse_save_path = save_path + " by Mouse" if save_path else None
        plot_rank_proportions_by_mouse(
            rank_counts_by_good_reversal,
            save_path=mouse_save_path,
            pool_trials_within_subject=pool_trials_within_subject,
        )
    if by_block:
        block_save_path = save_path + " by Block" if save_path else None
        plot_rank_proportions_by_block(rank_counts_by_good_reversal, save_path=block_save_path)
    if average:
        average_save_path = save_path + " by Average" if save_path else None
        plot_rank_proportions_average(
            rank_counts_by_good_reversal,
            average_across_mice_pvalues=average_across_mice_pvalues,
            save_path=average_save_path,
            pool_trials_within_subject=pool_trials_within_subject,
        )


# ---------------------------------------------------------------------------
# Shared drawing pieces
# ---------------------------------------------------------------------------

RANK_LABELS = ["Best", "Second", "Third"]
X = np.arange(len(RANK_LABELS))
BAR_WIDTH = 0.55
Y_LIMITS = (0.0, 1.07)

BAR_STYLE = dict(color="#999999", linewidth=1.5, alpha=0.55)
MOUSE_PANEL_BAR_STYLE = dict(color="#9E9E9E", linewidth=1.4, alpha=0.45)

# jitter keeps overlapping lines distinguishable; the rest goes to ax.plot
MOUSE_LINES = dict(jitter=0.03, linewidth=2.5, alpha=0.9, markersize=6)
AVERAGE_BLOCK_LINES = dict(jitter=0.03, linewidth=2.2, alpha=0.85)
MOUSE_PANEL_LINES = dict(jitter=0.04, linewidth=2.2, alpha=0.85, markersize=5)
BLOCK_PANEL_LINES = dict(jitter=0.03, linewidth=2.5, alpha=0.9, markersize=5.5)


def _draw_bars(ax, means, ses, style=BAR_STYLE):
    ax.bar(X, means, yerr=ses, capsize=6, width=BAR_WIDTH, edgecolor="black", zorder=0, **style)


def _label_means(ax, means, fontsize, *, finite_only=True):
    """Write each mean above its bar; return the x positions that were labelled."""
    labelled = []
    for xi, m in zip(X, means, strict=True):
        if finite_only and not np.isfinite(m):
            continue
        ax.text(xi, 1.05, f"{m:.2f}", ha="center", va="bottom", fontsize=fontsize)
        labelled.append(xi)
    return labelled


def _draw_line(ax, y, color, label, *, jitter, linewidth, **style):
    """Plot one jittered rank line and return its legend handle."""
    ax.plot(
        X + np.random.uniform(-jitter, jitter, size=len(X)),
        y,
        color=color,
        linewidth=linewidth,
        marker="o",
        **style,
    )
    return Line2D([0], [0], color=color, lw=linewidth, marker="o", label=label)


def _style_axis(ax, title, title_pad, *, show_ylabel=False, label_fontsize=None):
    text_kw = {} if label_fontsize is None else {"fontsize": label_fontsize}
    ax.set_title(title, pad=title_pad)
    ax.set_xticks(X)
    ax.set_xticklabels(RANK_LABELS, **text_kw)
    ax.set_ylim(*Y_LIMITS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_ylabel:
        ax.set_ylabel("Proportion of Choices", **text_kw)


def _annotate_pvalues(ax, labelled_x, pvalues):
    """Write the pairwise p-values that are present in *pvalues* onto the bars."""
    pvalues = pvalues or {}
    # Known issue: labelled_x holds only the bars with a finite mean, but is
    # indexed by rank position, so a non-finite mean plus a p-value raises IndexError.
    annotations = [
        ("best_vs_second", 0, 0.75, "p(Best vs Second)"),
        ("best_vs_third", 2, 0.25, "p(Best vs Third)"),
        ("second_vs_third", 1, 0.5, "p(Second vs Third)"),
    ]
    for key, rank_idx, y, text in annotations:
        if pvalues.get(key) is not None:
            ax.text(
                labelled_x[rank_idx],
                y,
                f"{text}:\n{pvalues[key]:.3f}",
                ha="center",
                va="bottom",
                fontsize=12,
            )


def _save_or_show(fig, save_path):
    """Save ``<save_path>.pdf/.png`` (creating the folder), or show; always close *fig*."""
    if save_path:
        base = Path(save_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(base) + ".pdf", bbox_inches="tight")
        fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def plot_average_across_mice(rank_counts, pvalues=None, pool_trials_within_subject=False, ax=None):
    """Bars (mean ± SE across mice) with one coloured line per mouse.

    Args:
        rank_counts: ``{subject: list[rank_count_dict]}``.
        pvalues: Optional ``{"best_vs_second", "best_vs_third", "second_vs_third"}``
            floats to annotate on the bars.
        pool_trials_within_subject: Weight each mouse's blocks by trial count.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5.5))
    per_mouse = proportions_by_mouse(rank_counts, pool_trials_within_subject)
    means, ses = mean_and_se_by_rank(list(per_mouse.values()))
    _draw_bars(ax, means, ses)
    _annotate_pvalues(ax, _label_means(ax, means, 12), pvalues)
    handles = [
        _draw_line(
            ax,
            [v[rank] for rank in RANKS],
            MOUSE_COLORS[i % len(MOUSE_COLORS)],
            subj,
            **MOUSE_LINES,
        )
        for i, (subj, v) in enumerate(per_mouse.items())
    ]
    title = "Pooled Trials Within Mouse" if pool_trials_within_subject else "Average Across Mice"
    _style_axis(ax, title, 14, show_ylabel=True)
    ax.legend(handles=handles, fontsize=10, loc="upper right")
    return ax


def plot_average_across_blocks(rank_counts, ax=None):
    """Bars (mean ± SE across blocks) with one coloured line per block."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5.5))
    per_block = proportions_by_block(rank_counts)
    means, ses = mean_and_se_by_rank(per_block)
    _draw_bars(ax, means, ses)
    _label_means(ax, means, 12)
    handles = [
        _draw_line(
            ax,
            [b[rank] for rank in RANKS],
            BLOCK_COLORS[i % len(BLOCK_COLORS)],
            f"Block {i + 1}",
            **AVERAGE_BLOCK_LINES,
        )
        for i, b in enumerate(per_block)
    ]
    _style_axis(ax, "Average Across Blocks", 20)
    ax.legend(handles=handles, fontsize=10, ncols=2, loc="upper right")
    return ax


def plot_mouse_panel(subject, rows, pool_trials_within_subject=False, ax=None, show_ylabel=True):
    """One mouse: a coloured line per block row over bars of the mouse's mean ± SE.

    Args:
        subject: Title of the panel.
        rows: The mouse's rank-count dicts to draw (see
            :func:`src.behavior_analysis.rank_proportions.mouse_block_rows`).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 4))
    handles = [
        _draw_line(
            ax,
            row_props(r, strict=True),
            BLOCK_COLORS[i % len(BLOCK_COLORS)],
            f"Block {i + 1}",
            **MOUSE_PANEL_LINES,
        )
        for i, r in enumerate(rows)
    ]
    proportions = subject_rank_proportions(rows, pool_trials_within_subject)
    means = [proportions[rank] for rank in RANKS]
    _, ses = mean_and_se_by_rank(prop_units(rows))
    _draw_bars(ax, means, ses, MOUSE_PANEL_BAR_STYLE)
    # unlike the other panels, a NaN mean is written out as "nan"
    _label_means(ax, means, 12, finite_only=False)
    _style_axis(ax, subject, 20, show_ylabel=show_ylabel)
    ax.legend(handles=handles, ncols=2, fontsize=10, loc="upper right")
    return ax


def _block_legend_label(subject, row):
    n_trials = row.get("total", None)
    return f"{subject}" if n_trials is None else f"{subject} (n={n_trials} trials)"


def plot_block_panel(rank_counts, block_idx, ax=None, show_ylabel=True):
    """One block: a coloured line per mouse over bars of the mean ± SE across mice."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 4))
    handles = [
        _draw_line(
            ax,
            row_props(rows[block_idx]),
            MOUSE_COLORS[si % len(MOUSE_COLORS)],
            _block_legend_label(subject, rows[block_idx]),
            **BLOCK_PANEL_LINES,
        )
        for si, (subject, rows) in enumerate(rank_counts.items())
        if block_idx < len(rows)
    ]
    # Known inconsistency: the lines above include every mouse's last row, the bars exclude it.
    means, ses = mean_and_se_by_rank(prop_units(rows_in_block(rank_counts, block_idx)))
    _draw_bars(ax, means, ses)
    _label_means(ax, means, 11)
    _style_axis(ax, f"Block {block_idx + 1}", 20, show_ylabel=show_ylabel, label_fontsize=12)
    if handles:
        ax.legend(handles=handles, fontsize=10, loc="upper right")
    return ax


# ---------------------------------------------------------------------------
# Standard figures
# ---------------------------------------------------------------------------


def plot_rank_proportions_average(
    rank_counts_by_good_reversal,
    average_across_mice_pvalues=None,
    save_path=None,
    pool_trials_within_subject=False,
):
    """Two-panel summary of rank-proportion averages across mice and blocks.

    The left panel shows bars (mean ± SE across mice) with individual mouse
    lines overlaid in ``MOUSE_COLORS``.  The right panel shows bars (mean ± SE
    across blocks) with individual block lines overlaid in ``BLOCK_COLORS``.

    Args:
        rank_counts_by_good_reversal: ``{subject: list[rank_count_dict]}``
            where each dict contains at least ``"best_prop"``, ``"second_prop"``,
            and ``"third_prop"`` keys (floats).  The last element of each list
            is treated as a summary entry and is excluded from the by-block view.
        average_across_mice_pvalues: Optional dict with keys
            ``"best_vs_second"``, ``"best_vs_third"``, ``"second_vs_third"``
            (floats) to annotate the left panel with p-values.  If ``None``,
            no p-value annotations are added (default: ``None``).
        save_path: Base path (without extension) for saving ``.pdf`` and
            ``.png`` output.  If ``None``, the figure is shown interactively.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    plot_average_across_mice(
        rank_counts_by_good_reversal,
        average_across_mice_pvalues,
        pool_trials_within_subject,
        ax=axes[0],
    )
    plot_average_across_blocks(rank_counts_by_good_reversal, ax=axes[1])
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_rank_proportions_by_mouse(
    rank_counts_by_good_reversal, save_path=None, pool_trials_within_subject=False
):
    """Plot rank proportions in a per-subject grid, with blocks as coloured lines.

    Draws one subplot per subject arranged in a 2-row grid.  Within each
    subplot, bars (mean ± SE across blocks) are shown with individual block
    lines overlaid using ``BLOCK_COLORS``.

    Args:
        rank_counts_by_good_reversal: ``{subject: list[rank_count_dict]}``
            where each dict contains ``"best_prop"``, ``"second_prop"``, and
            ``"third_prop"`` keys.  The last element of each subject's list is
            treated as a summary entry and is excluded from the per-block lines.
        save_path: Base path (without extension) for saving ``.pdf`` and
            ``.png`` output.  If ``None``, the figure is shown interactively.
    """
    subjects = list(rank_counts_by_good_reversal)
    ncols = math.ceil(len(subjects) / 2)
    fig, axes = plt.subplots(2, ncols, figsize=(4.2 * ncols, 8), sharey=True)
    for ax_idx, ax in enumerate(axes.flatten()):
        rows = (
            mouse_block_rows(
                rank_counts_by_good_reversal[subjects[ax_idx]], pool_trials_within_subject
            )
            if ax_idx < len(subjects)
            else []
        )
        if not rows:
            ax.axis("off")
            continue
        plot_mouse_panel(
            subjects[ax_idx],
            rows,
            pool_trials_within_subject,
            ax=ax,
            show_ylabel=ax_idx % ncols == 0,
        )
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_rank_proportions_by_block(rank_counts_by_good_reversal, save_path=None):
    """Plot rank proportions in a per-block grid, with subjects as coloured lines.

    Draws one subplot per block number arranged in a 2-row grid.  Within each
    subplot, bars (mean ± SE across subjects) are shown with individual subject
    lines overlaid using ``MOUSE_COLORS``.

    Args:
        rank_counts_by_good_reversal: ``{subject: list[rank_count_dict]}``
            where each dict contains ``"best_prop"``, ``"second_prop"``,
            ``"third_prop"``, and optionally ``"total"`` keys.  Subjects that
            do not have data for a given block are silently skipped.
        save_path: Base path (without extension) for saving ``.pdf`` and
            ``.png`` output.  If ``None``, the figure is shown interactively.
    """
    n_blocks = n_plotted_blocks(rank_counts_by_good_reversal)
    if n_blocks == 0:
        print("[WARN] No blocks found to plot.")
        return
    ncols = math.ceil(n_blocks / 2)
    fig, axes = plt.subplots(2, ncols, figsize=(4.2 * ncols, 8), sharey=True)
    axes = np.array(axes).flatten()
    for block_idx in range(n_blocks):
        plot_block_panel(
            rank_counts_by_good_reversal,
            block_idx,
            ax=axes[block_idx],
            show_ylabel=block_idx % ncols == 0,
        )
    for ax in axes[n_blocks:]:
        ax.axis("off")
    fig.tight_layout()
    _save_or_show(fig, save_path)
