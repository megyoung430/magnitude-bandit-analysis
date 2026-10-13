"""Plotting functions for per-problem summary metrics.

Functions here take summary dicts (as produced by
:mod:`src.behavior_analysis.problem_summary`) and produce bar charts with
per-mouse overlay lines, block-length histograms, and an averaged
rank-proportions panel.

Each panel function draws into an ``ax`` you pass (or a new figure if you
don't), so it can be one panel of a larger figure. The public ``plot_*``
figure functions compose those panels and save or show the result. The small
pure helpers (no matplotlib) hold the numbers each panel draws.
"""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.behavior_visualization.plot_style import MOUSE_COLORS

RANKS = ("best", "second", "third")
RANK_LABELS = ["Best", "Second", "Third"]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_or_show(fig: plt.Figure, base: str | Path | None, formats: tuple[str, ...]) -> None:
    """Save *fig* as ``<base>.<fmt>`` for each of *formats* (in order), or show it if no base.

    The figure is closed afterwards either way.
    """
    if base is not None:
        base = Path(base)
        base.parent.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            extra = {"dpi": 300} if fmt == "png" else {}
            fig.savefig(f"{base}.{fmt}", bbox_inches="tight", **extra)
    else:
        plt.show()
    plt.close(fig)


def mouse_color_map(mice: list) -> dict:
    """Assign each mouse a colour from ``MOUSE_COLORS``, cycling if there are more mice."""
    return {m: MOUSE_COLORS[i % len(MOUSE_COLORS)] for i, m in enumerate(mice)}


def concat_nonempty(arrays: list[np.ndarray]) -> np.ndarray:
    """Concatenate the non-empty arrays; an empty float array if all are empty."""
    parts = [a for a in arrays if a.size > 0]
    return np.concatenate(parts, axis=0) if parts else np.array([])


def value_range(values: np.ndarray) -> tuple[float, float] | None:
    """``(min, max)`` of *values*, or ``None`` if there are none."""
    return (np.nanmin(values), np.nanmax(values)) if values.size > 0 else None


# ---------------------------------------------------------------------------
# Bar chart of a per-problem metric with per-mouse overlay lines
# ---------------------------------------------------------------------------


class BarStyle(NamedTuple):
    """Presentation differences between the per-problem bar charts."""

    value_fmt: str
    legend_suffix: str
    legend_fontsize: int
    title_pad: int
    ylim_headroom: float


METRIC_STYLE = BarStyle(
    value_fmt=".2f", legend_suffix="", legend_fontsize=9, title_pad=18, ylim_headroom=1.02
)
LENGTHS_STYLE = BarStyle(
    value_fmt=".1f", legend_suffix=" problems", legend_fontsize=10, title_pad=20, ylim_headroom=1.0
)


def annotation_ceiling(
    means: np.ndarray, ses: np.ndarray, annotate_y: float | None
) -> tuple[float, float]:
    """Return ``(top of the tallest bar + error, y for the value labels)``.

    The label height defaults to 1.5x the bar top (at least 1.5) so the labels
    sit clear of the bars; an all-NaN chart falls back to a bar top of 1.
    """
    tops = means + ses
    finite_top = float(np.nanmax(tops)) if np.any(np.isfinite(tops)) else 1.0
    if annotate_y is None:
        annotate_y = max(finite_top, 1.0) * 1.5
    return finite_top, annotate_y


def mouse_points(blocks: list, by_block: dict) -> tuple[np.ndarray, np.ndarray]:
    """x positions (indices into *blocks*) and values for the blocks a mouse has data for."""
    pairs = [(i, by_block[b]) for i, b in enumerate(blocks) if b in by_block]
    xs = np.asarray([i for i, _ in pairs], float)
    ys = np.asarray([v for _, v in pairs], float)
    return xs, ys


def jittered(xs: np.ndarray, jitter: float) -> np.ndarray:
    """*xs* plus uniform noise of half-width *jitter* (unchanged if ``jitter <= 0``).

    Draws from the global ``np.random`` stream, so ``np.random.seed`` makes it repeatable.
    """
    if jitter > 0:
        return xs + np.random.uniform(-jitter, jitter, size=len(xs))
    return xs


def default_length_title(ylabel: str, n_mice: int) -> str:
    return f"{ylabel} per Problem\n(mean ± se across mice | n={n_mice} subjects)"


def _draw_bars_with_values(
    ax: plt.Axes, means: np.ndarray, ses: np.ndarray, annotate_y: float, value_fmt: str
) -> None:
    x = np.arange(len(means))
    ax.bar(
        x,
        means,
        yerr=ses,
        capsize=6,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.55,
        color=["#999999"] * len(means),
        zorder=1,
    )
    for xi, mval in zip(x, means, strict=True):
        if np.isfinite(mval):
            ax.text(
                xi,
                annotate_y,
                f"{mval:{value_fmt}}",
                ha="center",
                va="bottom",
                fontsize=12,
                clip_on=False,
            )


def _draw_mouse_lines(ax: plt.Axes, summary: dict, jitter: float, style: BarStyle) -> None:
    """One jittered line per mouse across problems, with a legend entry each.

    An explicit loop: the jitter draws from a shared random stream, so the
    mouse order fixes which noise each line gets.
    """
    blocks, per_mouse, mice = summary["blocks"], summary["per_mouse_blocklens"], summary["mice"]
    colors = mouse_color_map(mice)
    handles = []
    for m in mice:
        xs, ys = mouse_points(blocks, per_mouse.get(m, {}))
        if xs.size == 0:
            continue
        ax.plot(
            jittered(xs, jitter),
            ys,
            color=colors[m],
            linewidth=2.5,
            alpha=0.9,
            marker="o",
            markersize=6,
            zorder=3,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=colors[m],
                lw=2.5,
                marker="o",
                label=f"{m} (n={len(ys)}{style.legend_suffix})",
            )
        )
    ax.legend(handles=handles, fontsize=style.legend_fontsize, loc="upper right")


def plot_problem_bars(
    summary: dict,
    ax: plt.Axes | None = None,
    *,
    ylabel: str,
    title: str,
    jitter: float = 0.06,
    annotate_y: float | None = None,
    style: BarStyle = METRIC_STYLE,
) -> plt.Axes:
    """Bar chart of a per-problem mean +/- se with one overlay line per mouse.

    Args:
        summary: Summary dict with ``"blocks"``, ``"means"``, ``"ses"`` and
            ``"per_mouse_blocklens"``, ``"mice"``.
        ax: Axes to draw into; a new figure is made if ``None``.
        annotate_y: Height of the value labels above the bars (default: from
            :func:`annotation_ceiling`).
        style: Number format, legend and spacing choices.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4.8))
    blocks = summary["blocks"]
    means = np.asarray(summary["means"], dtype=float)
    ses = np.asarray(summary["ses"], dtype=float)
    finite_top, annotate_y = annotation_ceiling(means, ses, annotate_y)

    _draw_bars_with_values(ax, means, ses, annotate_y, style.value_fmt)
    _draw_mouse_lines(ax, summary, jitter, style)
    _despine(ax)
    ax.set_xticks(np.arange(len(blocks)))
    ax.set_xticklabels([str(b) for b in blocks], fontsize=12)
    ax.set_xlabel("Problem Number", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, pad=style.title_pad)
    ax.set_ylim(0, max(annotate_y, finite_top) * style.ylim_headroom)
    return ax


def plot_metric_by_problem(
    summary: dict,
    *,
    title: str,
    ylabel: str,
    jitter: float = 0.06,
    annotate_y: float | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Plot a per-problem metric as a bar chart with per-mouse overlay lines.

    Args:
        summary: A summary dict with keys ``"blocks"``, ``"means"``, ``"ses"``,
            ``"ns"``, ``"per_mouse_blocklens"``, and ``"mice"``, as returned by
            :func:`src.behavior_analysis.problem_utils.summarize_by_problem`.
        title: Figure title.
        ylabel: Y-axis label.
        jitter: Horizontal jitter width applied to per-mouse markers
            (default: 0.06).
        annotate_y: Y-position for value annotations above bars.  Defaults to
            1.5 × the max bar top.
        save_path: If provided, the figure is saved as both ``.pdf`` and
            ``.png`` at this path (suffix added automatically).
    """
    fig, ax = plt.subplots(figsize=(10, 4.8))
    plot_problem_bars(
        summary,
        ax,
        ylabel=ylabel,
        title=title,
        jitter=jitter,
        annotate_y=annotate_y,
        style=METRIC_STYLE,
    )
    fig.tight_layout()
    _save_or_show(fig, save_path, ("pdf", "png"))


def plot_problem_lengths(
    problem_summary: dict,
    *,
    ylabel: str = "Sessions to Complete",
    title: str | None = None,
    jitter: float = 0.06,
    annotate_y: float | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Plot per-problem session counts (or any per-problem metric) as a bar chart.

    This is the unified replacement for the three copies of ``plot_problem_lengths``
    found in the original ``testing_code.ipynb`` notebook.  Pass ``ylabel`` and
    ``title`` to customise the axis labels.

    Args:
        problem_summary: Summary dict as returned by
            :func:`src.behavior_analysis.problem_summary.compute_problem_length_summary`
            or any compatible function.
        ylabel: Y-axis label (default: ``"Sessions to Complete"``).
        title: Figure title.  Auto-generated from *ylabel* if not provided.
        jitter: Horizontal jitter width for per-mouse markers (default: 0.06).
        annotate_y: Y-position for value annotations above bars.
        save_path: If provided, saves ``.pdf`` and ``.png`` at this path.
    """
    if title is None:
        title = default_length_title(ylabel, len(problem_summary["mice"]))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    plot_problem_bars(
        problem_summary,
        ax,
        ylabel=ylabel,
        title=title,
        jitter=jitter,
        annotate_y=annotate_y,
        style=LENGTHS_STYLE,
    )
    fig.tight_layout()
    _save_or_show(fig, save_path, ("pdf", "png"))


# ---------------------------------------------------------------------------
# Block-length distribution plots
# ---------------------------------------------------------------------------


def collect_blocklens(
    problem_block_stats: dict,
) -> tuple[list, list, dict]:
    """Collect raw block lengths per problem per mouse.

    Args:
        problem_block_stats: ``{problem: {"block_stats": {...}}}``

    Returns:
        A three-tuple ``(problems, mice, blocklens_by_problem_mouse)`` where
        *blocklens_by_problem_mouse* is
        ``{problem: {mouse: np.ndarray of block lengths}}``.
    """
    problems = sorted(problem_block_stats.keys())
    mice = sorted(
        {
            m
            for p in problems
            for m in problem_block_stats[p]["block_stats"]["per_mouse_blocklens"].keys()
        }
    )

    blocklens_by_problem_mouse: dict = {p: {} for p in problems}
    for p in problems:
        per_mouse = problem_block_stats[p]["block_stats"]["per_mouse_blocklens"]
        for m in mice:
            block_dict = per_mouse.get(m, {})
            vals = np.asarray(list(block_dict.values()), dtype=float)
            blocklens_by_problem_mouse[p][m] = vals[np.isfinite(vals)]

    return problems, mice, blocklens_by_problem_mouse


def drop_last_problem_per_mouse_raw(
    problems: list,
    mice: list,
    blocklens_by_problem_mouse: dict,
) -> tuple[list, list, dict]:
    """Drop each mouse's last problem from raw block-length data.

    Args:
        problems: Sorted list of problem numbers.
        mice: Sorted list of mouse identifiers.
        blocklens_by_problem_mouse: ``{problem: {mouse: np.ndarray}}``.

    Returns:
        Updated ``(problems_new, mice_new, data_new)`` with last problems
        removed.
    """
    data = {p: {m: blocklens_by_problem_mouse[p][m].copy() for m in mice} for p in problems}

    for m in mice:
        ps_with_data = [p for p in problems if data[p].get(m, np.array([])).size > 0]
        if ps_with_data:
            data[max(ps_with_data)][m] = np.array([], dtype=float)

    problems_new = [p for p in problems if any(data[p][m].size > 0 for m in mice)]
    mice_new = [m for m in mice if any(data[p][m].size > 0 for p in problems_new)]
    data_new = {p: {m: data[p][m] for m in mice_new} for p in problems_new}
    return problems_new, mice_new, data_new


def _auto_hist_bins(x: Any) -> int:
    """Compute Freedman–Diaconis bin count with fallback for small samples.

    Args:
        x: 1-D array-like of values.

    Returns:
        Suggested number of histogram bins (between 5 and 40).
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 5
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    if iqr <= 0:
        return min(20, max(5, int(np.sqrt(x.size))))
    bin_width = 2 * iqr / (x.size ** (1 / 3))
    if bin_width <= 0:
        return min(20, max(5, int(np.sqrt(x.size))))
    bins = int(np.ceil((x.max() - x.min()) / bin_width))
    return min(40, max(5, bins))


def _stats_text(vals: Any, show_n: bool = True) -> str:
    """Format summary statistics as a multi-line string.

    Args:
        vals: 1-D array-like of values.
        show_n: If ``True``, include sample size in the text (default: True).

    Returns:
        Multi-line string with median, sd, min, and max.
    """
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return "no data"
    mu = np.median(vals)
    sd = vals.std(ddof=1) if vals.size > 1 else 0.0
    lines = []
    if show_n:
        lines.append(f"n = {vals.size}")
    lines += [
        f"median = {mu:.2f}",
        f"sd = {sd:.2f}",
        f"min = {vals.min():.0f}",
        f"max = {vals.max():.0f}",
    ]
    return "\n".join(lines)


def pooled_blocklens(data: dict, problem: Any, mice: list) -> np.ndarray:
    """All mice's block lengths for one *problem*, concatenated."""
    return concat_nonempty([data[problem][m] for m in mice])


def pooled_title(problem: Any, pooled: np.ndarray, n_mice_with_data: int) -> str:
    """Panel title for a pooled histogram; bare when the problem has no data."""
    if pooled.size == 0:
        return f"Problem {problem}"
    return f"Problem {problem} (blocks={pooled.size}, mice={n_mice_with_data})"


def plot_blocklen_histogram(
    vals: np.ndarray,
    ax: plt.Axes | None = None,
    *,
    title: str,
    xlim: tuple[float, float] | None = None,
) -> plt.Axes:
    """Histogram of block lengths with mean line and summary stats; "no data" if empty.

    Args:
        vals: 1-D array of finite block lengths.
        ax: Axes to draw into; a new figure is made if ``None``.
        title: Panel title.
        xlim: Shared x-range to apply to a non-empty panel.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(3.6, 2.8))
    if vals.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title(title, fontsize=11)
        _despine(ax)
        return ax
    ax.hist(
        vals,
        bins=_auto_hist_bins(vals),
        density=False,
        alpha=0.65,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.axvline(np.mean(vals), linestyle="--", linewidth=1.5)
    ax.text(
        0.98,
        0.98,
        _stats_text(vals),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="none"),
    )
    ax.set_title(title, fontsize=11)
    _despine(ax)
    if xlim is not None:
        ax.set_xlim(xlim)
    return ax


def _histogram_grid(
    n_panels: int, ncols: int, sharex: bool, sharey: bool
) -> tuple[plt.Figure, np.ndarray]:
    """A grid of at least *n_panels* axes (flattened); the unused ones are switched off."""
    nrows = int(ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.6 * ncols, 2.8 * nrows), sharex=sharex, sharey=sharey
    )
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n_panels:]:
        ax.axis("off")
    return fig, axes


def _label_histogram_figure(fig: plt.Figure, suptitle: str, ylabel: str) -> None:
    fig.suptitle(suptitle, y=1.02, fontsize=14)
    fig.supxlabel("Block length", fontsize=12)
    fig.supylabel(ylabel, fontsize=12)
    fig.tight_layout()


def _mouse_histogram_figure(
    data: dict, mouse: Any, problems: list, *, ncols: int, sharex: bool, sharey: bool
) -> plt.Figure:
    """One mouse's block-length histogram per problem, on a shared x-range."""
    fig, axes = _histogram_grid(len(problems), ncols, sharex, sharey)
    xlim = value_range(concat_nonempty([data[p][mouse] for p in problems]))
    for ax, p in zip(axes, problems, strict=False):
        plot_blocklen_histogram(data[p][mouse], ax, title=f"Problem {p}", xlim=xlim)
    _label_histogram_figure(fig, f"Block Length Distributions — Mouse {mouse}", "Count")
    return fig


def _pooled_histogram_figure(
    data: dict, problems: list, mice: list, *, ncols: int, sharex: bool, sharey: bool
) -> plt.Figure:
    """Block-length histogram per problem pooled over mice, on a shared x-range."""
    fig, axes = _histogram_grid(len(problems), ncols, sharex, sharey)
    xlim = value_range(concat_nonempty([data[p][m] for p in problems for m in mice]))
    for ax, p in zip(axes, problems, strict=False):
        pooled = pooled_blocklens(data, p, mice)
        n_mice = sum(data[p][m].size > 0 for m in mice)
        plot_blocklen_histogram(pooled, ax, title=pooled_title(p, pooled, n_mice), xlim=xlim)
    _label_histogram_figure(fig, "Block Length Distributions — Pooled Across Mice", "Counts")
    return fig


def plot_distributions_per_mouse(
    problem_block_stats: dict,
    *,
    drop_last_problem: bool = False,
    ncols: int = 4,
    sharex: bool = False,
    sharey: bool = False,
    save_dir: str | Path | None = None,
) -> None:
    """Plot block-length distributions per problem, one figure per mouse.

    Args:
        problem_block_stats: ``{problem: {"block_stats": {...}}}``
        drop_last_problem: If ``True``, exclude each mouse's last problem
            (default: False).
        ncols: Number of subplot columns (default: 4).
        sharex: Share x-axis across subplots (default: False).
        sharey: Share y-axis across subplots (default: False).
        save_dir: Directory for saved figures.  If ``None``, figures are shown
            interactively.
    """
    problems, mice, data = collect_blocklens(problem_block_stats)
    if drop_last_problem:
        problems, mice, data = drop_last_problem_per_mouse_raw(problems, mice, data)

    for m in mice:
        fig = _mouse_histogram_figure(data, m, problems, ncols=ncols, sharex=sharex, sharey=sharey)
        stem = None if save_dir is None else Path(save_dir) / f"blocklen_distributions_mouse_{m}"
        _save_or_show(fig, stem, ("png", "pdf"))


def plot_distributions_per_problem_all_mice(
    problem_block_stats: dict,
    *,
    drop_last_problem: bool = False,
    ncols: int = 4,
    sharex: bool = False,
    sharey: bool = False,
    save_path: str | Path | None = None,
) -> None:
    """Plot block-length distributions per problem, pooling all mice.

    Args:
        problem_block_stats: ``{problem: {"block_stats": {...}}}``
        drop_last_problem: If ``True``, exclude each mouse's last problem
            (default: False).
        ncols: Number of subplot columns (default: 4).
        sharex: Share x-axis across subplots (default: False).
        sharey: Share y-axis across subplots (default: False).
        save_path: Path for saved figure (suffixes added automatically).
            If ``None``, the figure is shown interactively.
    """
    problems, mice, data = collect_blocklens(problem_block_stats)
    if drop_last_problem:
        problems, mice, data = drop_last_problem_per_mouse_raw(problems, mice, data)

    fig = _pooled_histogram_figure(data, problems, mice, ncols=ncols, sharex=sharex, sharey=sharey)
    _save_or_show(fig, save_path, ("png", "pdf"))


# ---------------------------------------------------------------------------
# Averaged rank-proportions plot (no per-block breakdown)
# ---------------------------------------------------------------------------


def mean_rank_proportions_by_subject(rank_counts_by_good_reversal: dict) -> dict:
    """Each subject's mean best/second/third proportion over its reversals.

    Subjects with no reversals are left out; a reversal missing a proportion
    counts as NaN, which the mean skips.

    Args:
        rank_counts_by_good_reversal: ``{subject: list[reversal_dict]}`` with
            ``"best_prop"``, ``"second_prop"``, ``"third_prop"`` in each dict.

    Returns:
        ``{subject: {"best": float, "second": float, "third": float}}``.
    """
    return {
        subj: {
            rank: np.nanmean([r.get(f"{rank}_prop", float("nan")) for r in rows]) for rank in RANKS
        }
        for subj, rows in rank_counts_by_good_reversal.items()
        if rows
    }


def rank_mean_and_se(per_mouse: dict) -> tuple[list[float], list[float]]:
    """Across-mouse mean and standard error for each rank (NaNs skipped in the mean and sd)."""
    columns = [[v[rank] for v in per_mouse.values()] for rank in RANKS]
    means = [np.nanmean(col) for col in columns]
    ses = [np.nanstd(col, ddof=1) / np.sqrt(len(per_mouse)) for col in columns]
    return means, ses


def pvalue_annotations(
    pvalues: dict | None, finite_x: list[float]
) -> list[tuple[float, float, str]]:
    """``(x, y, text)`` for each available pairwise p-value label.

    Labels are anchored on the *finite_x* bar positions (those with a finite
    mean), by position in that list.
    """
    pvalues = pvalues or {}
    # (key, label, index into finite_x, y); a label needs that many finite bars
    specs = [
        ("best_vs_second", "p(Best vs Second)", 0, 0.75),
        ("best_vs_third", "p(Best vs Third)", 2, 0.25),
        ("second_vs_third", "p(Second vs Third)", 1, 0.5),
    ]
    return [
        (finite_x[idx], y, f"{label}:\n{pvalues[key]:.3f}")
        for key, label, idx, y in specs
        if pvalues.get(key) is not None and len(finite_x) > idx
    ]


def _draw_rank_bars(
    ax: plt.Axes, means: list[float], ses: list[float], pvalues: dict | None
) -> None:
    """Grey bars with value labels, plus the p-value labels."""
    x = np.arange(len(RANK_LABELS))
    ax.bar(
        x,
        means,
        yerr=ses,
        capsize=6,
        width=0.55,
        color="#999999",
        edgecolor="black",
        linewidth=1.5,
        alpha=0.55,
        zorder=0,
    )
    finite_x = [xi for xi, m in zip(x, means, strict=True) if np.isfinite(m)]
    for xi in finite_x:
        ax.text(xi, 1.05, f"{means[xi]:.2f}", ha="center", va="bottom", fontsize=12)
    # Known quirk, kept as is: a label is anchored on the i-th *finite* bar, so when an
    # earlier rank's mean is NaN it lands on a different rank than its text names.
    for xi, y, text in pvalue_annotations(pvalues, finite_x):
        ax.text(xi, y, text, ha="center", va="bottom", fontsize=12)


def _draw_rank_mouse_lines(ax: plt.Axes, per_mouse: dict, jitter: float) -> None:
    """One jittered line per mouse across the three ranks, with a legend entry each.

    An explicit loop: the jitter draws from a shared random stream, so the
    mouse order fixes which noise each line gets.
    """
    x = np.arange(len(RANK_LABELS))
    handles = []
    for i, (subj, v) in enumerate(per_mouse.items()):
        c = MOUSE_COLORS[i % len(MOUSE_COLORS)]
        ax.plot(
            jittered(x, jitter),
            [v[rank] for rank in RANKS],
            color=c,
            linewidth=2.5,
            marker="o",
            alpha=0.9,
            markersize=6,
        )
        handles.append(Line2D([0], [0], color=c, lw=2.5, marker="o", label=subj))
    ax.legend(handles=handles, fontsize=10, loc="upper right")


def plot_rank_proportions_average_panel(
    per_mouse: dict,
    means: list[float],
    ses: list[float],
    ax: plt.Axes | None = None,
    *,
    pvalues: dict | None = None,
    jitter: float = 0.03,
) -> plt.Axes:
    """Mean best/second/third proportion bars with one jittered line per mouse.

    Args:
        per_mouse: ``{subject: {"best", "second", "third"}}`` from
            :func:`mean_rank_proportions_by_subject`.
        means, ses: Across-mouse mean and se per rank, from :func:`rank_mean_and_se`.
        ax: Axes to draw into; a new figure is made if ``None``.
        pvalues: Optional ``"best_vs_second"``, ``"best_vs_third"``,
            ``"second_vs_third"`` p-values to print on the panel.
        jitter: Half-width of the horizontal jitter on the mouse lines.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5.5))
    _draw_rank_bars(ax, means, ses, pvalues)
    _draw_rank_mouse_lines(ax, per_mouse, jitter)
    ax.set_title("Average Across Mice", pad=14)
    ax.set_xticks(np.arange(len(RANK_LABELS)))
    ax.set_xticklabels(RANK_LABELS)
    ax.set_ylabel("Proportion of Choices")
    ax.set_ylim(0.0, 1.07)
    _despine(ax)
    return ax


def plot_rank_proportions_average_without_blocks(
    rank_counts_by_good_reversal: dict,
    *,
    average_across_mice_pvalues: dict | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Plot average rank proportions across mice (single-panel summary).

    Unlike :func:`src.behavior_visualization.plot_rank_proportions.plot_rank_proportions`,
    this function produces a single panel averaging across mice without a
    separate per-block panel.

    Args:
        rank_counts_by_good_reversal: ``{subject: list[reversal_dict]}``
            where each reversal dict contains ``"best_prop"``,
            ``"second_prop"``, and ``"third_prop"``.
        average_across_mice_pvalues: Optional dict of p-values with keys
            ``"best_vs_second"``, ``"best_vs_third"``, and
            ``"second_vs_third"``.
        save_path: Path for saved figure (suffixes added automatically).
            If ``None``, the figure is shown interactively.
    """
    per_mouse = mean_rank_proportions_by_subject(rank_counts_by_good_reversal)
    means, ses = rank_mean_and_se(per_mouse)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
    plot_rank_proportions_average_panel(
        per_mouse, means, ses, ax, pvalues=average_across_mice_pvalues
    )
    fig.tight_layout()
    _save_or_show(fig, save_path, ("pdf", "png"))
