"""Plotting functions for arm-rank choice proportions around good reversals.

Three views are available, all accessible through the top-level
:func:`plot_rank_proportions` dispatcher:

- *By Mouse* — one panel per subject, blocks as coloured lines.
- *By Block* — one panel per block number, subjects as coloured lines.
- *Average across mice* — one panel: bars over mice, a line per mouse (mouse colours).
- *Average across blocks* — one panel: bars over blocks, a line per block (block colours).

:func:`plot_rank_proportions_average` still draws the two averages side by side in one
figure, but the dispatcher does not call it.

Each panel is a function that draws into an ``ax`` you pass (or a new figure if
you don't), so it can be one panel of a larger figure; the ``plot_rank_proportions*``
functions arrange those panels and save or show the figure. The numbers behind
the panels come from :mod:`src.behavior_analysis.rank_proportions`.

Notebooks star-import this module, so only the plotting functions, ``BLOCK_COLORS``
and the imports below are public; every other module-level name starts with ``_``
(a bare ``X`` would overwrite a user's design matrix).
"""

import math
from collections.abc import Hashable
from pathlib import Path  # noqa: F401  # public to star-importing notebooks

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.behavior_analysis import rank_proportions as _summary
from src.behavior_visualization import plot_helpers as _helpers

# Read at call time (not copied): notebooks reassign this module's MOUSE_COLORS to extend the
# palette, and they star-import it.
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
    by_mouse=False,
    by_block=False,
    average_by_mouse=True,
    average_by_block=False,
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
        by_mouse: If ``True``, produce the by-mouse panel grid, saved as
            ``<save_path> by Mouse`` (default: ``False``).
        by_block: If ``True``, produce the by-block panel grid, saved as
            ``<save_path> by Block`` (default: ``False``).
        average_by_mouse: If ``True``, produce the across-mice figure (with
            p-values), saved as ``<save_path> by Average Across Mice``
            (default: ``True``).
        average_by_block: If ``True``, produce the across-blocks figure, saved as
            ``<save_path> by Average Across Blocks`` (default: ``False``).
        pool_trials_within_subject: If ``True``, the by-mouse grid and the
            average-across-mice panel weight each mouse's blocks by trial count
            (pooled proportions) instead of averaging the block proportions
            (default: ``False``).  The by-block grid ignores it.

    See the sub-functions for known issues in what each view includes.
    """
    if by_mouse:
        plot_rank_proportions_by_mouse(
            rank_counts_by_good_reversal,
            save_path=_view_path(save_path, "by Mouse"),
            pool_trials_within_subject=pool_trials_within_subject,
        )
    if by_block:
        plot_rank_proportions_by_block(
            rank_counts_by_good_reversal, save_path=_view_path(save_path, "by Block")
        )
    if average_by_mouse:
        plot_rank_proportions_average_by_mouse(
            rank_counts_by_good_reversal,
            average_across_mice_pvalues=average_across_mice_pvalues,
            save_path=_view_path(save_path, "by Average Across Mice"),
            pool_trials_within_subject=pool_trials_within_subject,
        )
    if average_by_block:
        plot_rank_proportions_average_by_block(
            rank_counts_by_good_reversal,
            save_path=_view_path(save_path, "by Average Across Blocks"),
        )


def _view_path(save_path, view):
    """``"<save_path> <view>"``, or ``None`` (show the figure) when there is no *save_path*."""
    return f"{save_path} {view}" if save_path else None


# ---------------------------------------------------------------------------
# Shared drawing pieces
# ---------------------------------------------------------------------------

_RANK_LABELS = ["Best", "Second", "Third"]
_X = np.arange(len(_RANK_LABELS))
_BAR_WIDTH = 0.55
_Y_LIMITS = (0.0, 1.07)

# half the width of the combined two-panel average figure (13 x 5.5)
_SINGLE_PANEL_FIGSIZE = (6.5, 5.5)

_BAR_STYLE = dict(color="#999999", linewidth=1.5, alpha=0.55)
_MOUSE_PANEL_BAR_STYLE = dict(color="#9E9E9E", linewidth=1.4, alpha=0.45)

# jitter keeps overlapping lines distinguishable; the rest goes to ax.plot
_MOUSE_LINES = dict(jitter=0.03, linewidth=2.5, alpha=0.9, markersize=6)
_AVERAGE_BLOCK_LINES = dict(jitter=0.03, linewidth=2.2, alpha=0.85)
_MOUSE_PANEL_LINES = dict(jitter=0.04, linewidth=2.2, alpha=0.85, markersize=5)
_BLOCK_PANEL_LINES = dict(jitter=0.03, linewidth=2.5, alpha=0.9, markersize=5.5)


def _draw_bars(ax, means, ses, *, style=_BAR_STYLE):
    ax.bar(_X, means, yerr=ses, capsize=6, width=_BAR_WIDTH, edgecolor="black", zorder=0, **style)


def _label_means(ax, means, fontsize, *, finite_only=True):
    """Write each mean above its bar; return the x positions that were labelled."""
    labelled = []
    for xi, m in zip(_X, means, strict=True):
        if finite_only and not np.isfinite(m):
            continue
        ax.text(xi, 1.05, f"{m:.2f}", ha="center", va="bottom", fontsize=fontsize)
        labelled.append(xi)
    return labelled


def _draw_line(ax, y, color, label, *, jitter, linewidth, **style):
    """Plot one jittered rank line and return its legend handle."""
    ax.plot(
        _helpers.jittered(_X, jitter),
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
    ax.set_xticks(_X)
    ax.set_xticklabels(_RANK_LABELS, **text_kw)
    ax.set_ylim(*_Y_LIMITS)
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


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def plot_average_across_mice(
    rank_counts: _summary.RankCounts,
    *,
    pvalues: dict | None = None,
    pool_trials_within_subject: bool = False,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Bars (mean ± SE across mice) with one coloured line per mouse.

    Args:
        rank_counts: ``{subject: list[rank_count_dict]}``; every row of every
            subject is used.
        pvalues: Optional ``{"best_vs_second", "best_vs_third", "second_vs_third"}``
            floats to annotate on the bars. Known issue: raises ``IndexError``
            if a bar's mean is not finite and any p-value is given.
        pool_trials_within_subject: Each mouse's value is its pooled proportion
            (weighted by trial count) instead of the mean of its block proportions.
        ax: Axes to draw into; a new figure is created if ``None``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5.5))
    per_mouse = _summary.proportions_by_mouse(
        rank_counts, pool_trials_within_subject=pool_trials_within_subject
    )
    units = list(per_mouse.values())
    means = _summary.mean_by_rank(units, guard_empty=False)
    ses = _summary.se_by_rank(units, guard_single=False)
    _draw_bars(ax, means, ses)
    labelled_x = _label_means(ax, means, 12)
    _annotate_pvalues(ax, labelled_x, pvalues)
    handles = [
        _draw_line(
            ax,
            [v[rank] for rank in _summary.RANKS],
            _helpers.palette_color(i, MOUSE_COLORS),
            subj,
            **_MOUSE_LINES,
        )
        for i, (subj, v) in enumerate(per_mouse.items())
    ]
    title = "Pooled Trials Within Mouse" if pool_trials_within_subject else "Average Across Mice"
    _style_axis(ax, title, 14, show_ylabel=True)
    ax.legend(handles=handles, fontsize=10, loc="upper right")
    return ax


def plot_average_across_blocks(
    rank_counts: _summary.RankCounts, *, ax: plt.Axes | None = None
) -> plt.Axes:
    """Bars (mean ± SE across blocks) with one coloured line per block.

    Every row of every subject is a block; block *b* averages the subjects that have it.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5.5))
    per_block = _summary.proportions_by_block(rank_counts)
    means = _summary.mean_by_rank(per_block, guard_empty=True)
    ses = _summary.se_by_rank(per_block, guard_single=False)
    _draw_bars(ax, means, ses)
    _label_means(ax, means, 12)
    handles = [
        _draw_line(
            ax,
            [b[rank] for rank in _summary.RANKS],
            BLOCK_COLORS[i % len(BLOCK_COLORS)],
            f"Block {i + 1}",
            **_AVERAGE_BLOCK_LINES,
        )
        for i, b in enumerate(per_block)
    ]
    _style_axis(ax, "Average Across Blocks", 20)
    ax.legend(handles=handles, fontsize=10, ncols=2, loc="upper right")
    return ax


def plot_mouse_panel(
    subject: Hashable,
    rows: list[_summary.RankRow],
    *,
    pool_trials_within_subject: bool = False,
    ax: plt.Axes | None = None,
    show_ylabel: bool = True,
) -> plt.Axes:
    """One mouse: a coloured line per block row over bars of the mouse's mean ± SE.

    The bar height is the mouse's proportion and the error bar the SE across the
    same units: blocks, or with *pool_trials_within_subject* the pooled trials
    (see :func:`src.behavior_analysis.rank_proportions.subject_rank_ses`).

    Args:
        subject: Title of the panel.
        rows: The mouse's rank-count dicts to draw, one per block.
        pool_trials_within_subject: Pool the rows' counts for the bars and errors.
        ax: Axes to draw into; a new figure is created if ``None``.
        show_ylabel: Whether to write the y-axis label.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 4))
    handles = [
        _draw_line(
            ax,
            _summary.row_props(r, strict=True),
            BLOCK_COLORS[i % len(BLOCK_COLORS)],
            f"Block {i + 1}",
            **_MOUSE_PANEL_LINES,
        )
        for i, r in enumerate(rows)
    ]
    proportions = _summary.subject_rank_proportions(
        rows, pool_trials_within_subject=pool_trials_within_subject
    )
    means = [proportions[rank] for rank in _summary.RANKS]
    ses = _summary.subject_rank_ses(rows, pool_trials_within_subject=pool_trials_within_subject)
    _draw_bars(ax, means, ses, style=_MOUSE_PANEL_BAR_STYLE)
    # unlike the other panels, a NaN mean is written out as "nan"
    _label_means(ax, means, 12, finite_only=False)
    _style_axis(ax, subject, 20, show_ylabel=show_ylabel)
    ax.legend(handles=handles, ncols=2, fontsize=10, loc="upper right")
    return ax


def _block_legend_label(subject, row):
    n_trials = row.get("total", None)
    return f"{subject}" if n_trials is None else f"{subject} (n={n_trials} trials)"


def plot_block_panel(
    rank_counts: _summary.RankCounts,
    block_idx: int,
    *,
    ax: plt.Axes | None = None,
    show_ylabel: bool = True,
) -> plt.Axes:
    """One block: a coloured line per mouse over bars of the mean ± SE across mice.

    The lines and the bars cover the same mice: those with a row at *block_idx*.

    Args:
        rank_counts: ``{subject: list[rank_count_dict]}``.
        block_idx: Zero-based block number.
        ax: Axes to draw into; a new figure is created if ``None``.
        show_ylabel: Whether to write the y-axis label.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 4))
    handles = [
        _draw_line(
            ax,
            _summary.row_props(rows[block_idx]),
            _helpers.palette_color(si, MOUSE_COLORS),
            _block_legend_label(subject, rows[block_idx]),
            **_BLOCK_PANEL_LINES,
        )
        for si, (subject, rows) in enumerate(rank_counts.items())
        if block_idx < len(rows)
    ]
    units = _summary.prop_units(_summary.rows_in_block(rank_counts, block_idx))
    means = _summary.mean_by_rank(units, guard_empty=True)
    ses = _summary.se_by_rank(units, guard_single=True)
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

    Known issues:

    - If a bar's mean is not finite and p-values are given, ``IndexError``.
    - Fewer than two mice emit a ``RuntimeWarning`` from the across-mice SE (the
      error bars are then omitted, the bars still drawn); no mice at all also
      emit numpy's empty-slice ``RuntimeWarning``.
    - ``save_path`` must be a ``str`` in an existing folder (``Path`` raises
      ``TypeError``, a missing folder ``FileNotFoundError``); the other two figures
      accept a ``Path`` and create the folder.

    Args:
        rank_counts_by_good_reversal: ``{subject: list[rank_count_dict]}``
            where each dict contains at least ``"best_prop"``, ``"second_prop"``,
            and ``"third_prop"`` keys (floats).
        average_across_mice_pvalues: Optional dict with keys
            ``"best_vs_second"``, ``"best_vs_third"``, ``"second_vs_third"``
            (floats) to annotate the left panel with p-values.  If ``None``,
            no p-value annotations are added (default: ``None``).
        save_path: Base path (without extension) for saving ``.pdf`` and
            ``.png`` output.  If ``None``, the figure is shown interactively.
        pool_trials_within_subject: If ``True``, the left panel uses each
            mouse's pooled proportion (weighted by trial count) and is titled
            "Pooled Trials Within Mouse" (default: ``False``).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    plot_average_across_mice(
        rank_counts_by_good_reversal,
        pvalues=average_across_mice_pvalues,
        pool_trials_within_subject=pool_trials_within_subject,
        ax=axes[0],
    )
    plot_average_across_blocks(rank_counts_by_good_reversal, ax=axes[1])
    fig.tight_layout()
    if save_path and not isinstance(save_path, str):
        plt.close(fig)
        raise TypeError(f"save_path must be a str, got {save_path!r}")
    _helpers.save_or_show(fig, save_path or None, make_parent=False)


def plot_rank_proportions_average_by_mouse(
    rank_counts_by_good_reversal,
    average_across_mice_pvalues=None,
    save_path=None,
    pool_trials_within_subject=False,
):
    """Single-panel figure of :func:`plot_average_across_mice`.

    It draws what the left panel of :func:`plot_rank_proportions_average` draws.
    Known issues are those of :func:`plot_average_across_mice`.

    Args:
        rank_counts_by_good_reversal: ``{subject: list[rank_count_dict]}``.
        average_across_mice_pvalues: Optional ``"best_vs_second"``,
            ``"best_vs_third"``, ``"second_vs_third"`` p-values to annotate.
        save_path: Base path (``str`` or ``Path``, without extension) for the
            ``.pdf`` and ``.png`` files; missing folders are created. If ``None``,
            the figure is shown interactively.
        pool_trials_within_subject: Use each mouse's pooled proportion (weighted by
            trial count); the panel is then titled "Pooled Trials Within Mouse".
    """
    fig, ax = plt.subplots(figsize=_SINGLE_PANEL_FIGSIZE)
    plot_average_across_mice(
        rank_counts_by_good_reversal,
        pvalues=average_across_mice_pvalues,
        pool_trials_within_subject=pool_trials_within_subject,
        ax=ax,
    )
    fig.tight_layout()
    _helpers.save_or_show(fig, save_path or None)


def plot_rank_proportions_average_by_block(rank_counts_by_good_reversal, save_path=None):
    """Single-panel figure of :func:`plot_average_across_blocks`.

    It draws what the right panel of :func:`plot_rank_proportions_average` draws.

    Args:
        rank_counts_by_good_reversal: ``{subject: list[rank_count_dict]}``.
        save_path: Base path (``str`` or ``Path``, without extension) for the
            ``.pdf`` and ``.png`` files; missing folders are created. If ``None``,
            the figure is shown interactively.
    """
    fig, ax = plt.subplots(figsize=_SINGLE_PANEL_FIGSIZE)
    plot_average_across_blocks(rank_counts_by_good_reversal, ax=ax)
    fig.tight_layout()
    _helpers.save_or_show(fig, save_path or None)


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
            ``"third_prop"`` keys.
        save_path: Base path (without extension) for saving ``.pdf`` and
            ``.png`` output.  If ``None``, the figure is shown interactively.
        pool_trials_within_subject: If ``True``, draw the bars as the pooled
            proportion (weighted by trial count) with the SE across the pooled
            trials, instead of the mean of the block proportions with the SE
            across blocks (default: ``False``).
    """
    subjects = list(rank_counts_by_good_reversal)
    ncols = math.ceil(len(subjects) / 2)
    fig, axes = plt.subplots(2, ncols, figsize=(4.2 * ncols, 8), sharey=True)
    # the 2-row grid can have one more axis than subjects; leftovers are turned off below
    for ax, subject in zip(axes.flat, subjects, strict=False):
        rows = rank_counts_by_good_reversal[subject]
        if rows:
            plot_mouse_panel(
                subject,
                rows,
                pool_trials_within_subject=pool_trials_within_subject,
                ax=ax,
                show_ylabel=ax.get_subplotspec().is_first_col(),
            )
        else:
            ax.axis("off")
    for ax in axes.flat[len(subjects) :]:
        ax.axis("off")
    fig.tight_layout()
    _helpers.save_or_show(fig, save_path or None)


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
    n_blocks = _summary.n_blocks(rank_counts_by_good_reversal)
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
            show_ylabel=axes[block_idx].get_subplotspec().is_first_col(),
        )
    for ax in axes[n_blocks:]:
        ax.axis("off")
    fig.tight_layout()
    _helpers.save_or_show(fig, save_path or None)
