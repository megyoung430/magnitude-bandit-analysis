"""Plotting functions for per-problem summary metrics.

Functions here take summary dicts (as produced by
:mod:`src.behavior_analysis.problem_summary`) and produce bar charts with
per-mouse overlay lines.
"""
from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.behavior_visualization.plot_style import MOUSE_COLORS

# ---------------------------------------------------------------------------
# Core bar-chart helper
# ---------------------------------------------------------------------------

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
    blocks = summary["blocks"]
    means = np.asarray(summary["means"], dtype=float)
    ses = np.asarray(summary["ses"], dtype=float)
    ns = summary["ns"]
    per_mouse = summary["per_mouse_blocklens"]
    mice = summary["mice"]

    mouse_to_color = {m: MOUSE_COLORS[i % len(MOUSE_COLORS)] for i, m in enumerate(mice)}

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(blocks))

    ax.bar(
        x, means, yerr=ses, capsize=6,
        edgecolor="black", linewidth=1.5, alpha=0.55,
        color=["#999999"] * len(blocks), zorder=1,
    )

    finite_top = np.nanmax(means + ses) if np.any(np.isfinite(means + ses)) else 1.0
    if annotate_y is None:
        annotate_y = max(float(finite_top), 1.0) * 1.5

    for xi, mval in zip(x, means):
        if np.isfinite(mval):
            ax.text(xi, annotate_y, f"{mval:.2f}", ha="center", va="bottom",
                    fontsize=12, clip_on=False)

    legend_handles = []
    for m in mice:
        d = per_mouse.get(m, {})
        xs, ys = [], []
        for bi, b in enumerate(blocks):
            if b in d:
                xs.append(bi)
                ys.append(d[b])
        if not xs:
            continue
        xs_arr = np.asarray(xs, float)
        ys_arr = np.asarray(ys, float)
        if jitter > 0:
            xs_arr = xs_arr + np.random.uniform(-jitter, jitter, size=len(xs_arr))
        c = mouse_to_color[m]
        ax.plot(xs_arr, ys_arr, color=c, linewidth=2.5, alpha=0.9,
                marker="o", markersize=6, zorder=3)
        legend_handles.append(
            Line2D([0], [0], color=c, lw=2.5, marker="o", label=f"{m} (n={len(ys)})")
        )

    ax.legend(handles=legend_handles, fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in blocks], fontsize=12)
    ax.set_xlabel("Problem Number", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, pad=18)

    top_needed = max(annotate_y, float(finite_top))
    ax.set_ylim(0, top_needed * 1.02)

    plt.tight_layout()

    if save_path is not None:
        base = Path(save_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(base) + ".pdf", bbox_inches="tight")
        fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)

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
    blocks = problem_summary["blocks"]
    means = np.asarray(problem_summary["means"], dtype=float)
    ses = np.asarray(problem_summary["ses"], dtype=float)
    ns = problem_summary["ns"]
    per_mouse_blocklens = problem_summary["per_mouse_blocklens"]
    mice = problem_summary["mice"]

    mouse_to_color = {m: MOUSE_COLORS[i % len(MOUSE_COLORS)] for i, m in enumerate(mice)}

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(blocks))

    ax.bar(
        x, means, yerr=ses, capsize=6,
        edgecolor="black", linewidth=1.5, alpha=0.55,
        color=["#999999"] * len(blocks), zorder=1,
    )

    finite_top = np.nanmax(means + ses) if np.any(np.isfinite(means + ses)) else 1.0
    if annotate_y is None:
        annotate_y = max(float(finite_top), 1.0) * 1.5

    for xi, mval, _n in zip(x, means, ns):
        if np.isfinite(mval):
            ax.text(xi, annotate_y, f"{mval:.1f}", ha="center", va="bottom",
                    fontsize=12, clip_on=False)

    legend_handles = []
    for m in mice:
        bl = per_mouse_blocklens.get(m, {})
        c = mouse_to_color[m]
        xs, ys = [], []
        for bi, b in enumerate(blocks):
            if b in bl:
                xs.append(bi)
                ys.append(bl[b])
        if not xs:
            continue
        xs_arr = np.asarray(xs, float)
        ys_arr = np.asarray(ys, float)
        if jitter > 0:
            xs_arr = xs_arr + np.random.uniform(-jitter, jitter, size=len(xs_arr))
        ax.plot(xs_arr, ys_arr, color=c, linewidth=2.5, alpha=0.9,
                marker="o", markersize=6, zorder=3)
        legend_handles.append(
            Line2D([0], [0], color=c, lw=2.5, marker="o",
                   label=f"{m} (n={len(ys)} problems)")
        )

    ax.legend(handles=legend_handles, fontsize=10, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in blocks], fontsize=12)
    ax.set_xlabel("Problem Number", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    if title is None:
        title = (
            f"{ylabel} per Problem\n"
            f"(mean ± se across mice | n={len(mice)} subjects)"
        )
    ax.set_title(title, pad=20)

    top_needed = max(annotate_y, float(finite_top))
    ax.set_ylim(0, top_needed)

    plt.tight_layout()

    if save_path is not None:
        base = Path(save_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(base) + ".pdf", bbox_inches="tight")
        fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)

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
    mice = sorted({
        m
        for p in problems
        for m in problem_block_stats[p]["block_stats"]["per_mouse_blocklens"].keys()
    })

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
    data = {
        p: {m: blocklens_by_problem_mouse[p][m].copy() for m in mice}
        for p in problems
    }

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
        n = len(problems)
        nrows = int(ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(3.6 * ncols, 2.8 * nrows),
            sharex=sharex,
            sharey=sharey,
        )
        axes = np.atleast_1d(axes).ravel()

        all_vals = np.concatenate(
            [data[p][m] for p in problems if data[p][m].size > 0], axis=0
        ) if any(data[p][m].size > 0 for p in problems) else np.array([])
        xlim = (np.nanmin(all_vals), np.nanmax(all_vals)) if all_vals.size > 0 else None

        for i, p in enumerate(problems):
            ax = axes[i]
            vals = data[p][m]
            if vals.size == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10)
                ax.set_title(f"Problem {p}", fontsize=11)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                continue
            ax.hist(vals, bins=_auto_hist_bins(vals), density=False,
                    alpha=0.65, edgecolor="black", linewidth=0.8)
            ax.axvline(np.mean(vals), linestyle="--", linewidth=1.5)
            ax.text(
                0.98, 0.98, _stats_text(vals),
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          alpha=0.85, edgecolor="none"),
            )
            ax.set_title(f"Problem {p}", fontsize=11)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if xlim is not None:
                ax.set_xlim(xlim)

        for j in range(len(problems), len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"Block Length Distributions — Mouse {m}", y=1.02, fontsize=14)
        fig.supxlabel("Block length", fontsize=12)
        fig.supylabel("Count", fontsize=12)
        plt.tight_layout()

        if save_dir is not None:
            out = Path(save_dir)
            out.mkdir(parents=True, exist_ok=True)
            stem = out / f"blocklen_distributions_mouse_{m}"
            fig.savefig(str(stem) + ".png", dpi=300, bbox_inches="tight")
            fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
        else:
            plt.show()

        plt.close(fig)

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

    n = len(problems)
    nrows = int(ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.6 * ncols, 2.8 * nrows),
        sharex=sharex,
        sharey=sharey,
    )
    axes = np.atleast_1d(axes).ravel()

    pooled_all = np.concatenate(
        [data[p][m] for p in problems for m in mice if data[p][m].size > 0], axis=0
    ) if any(data[p][m].size > 0 for p in problems for m in mice) else np.array([])
    xlim = (np.nanmin(pooled_all), np.nanmax(pooled_all)) if pooled_all.size > 0 else None

    for i, p in enumerate(problems):
        ax = axes[i]
        pooled = np.concatenate(
            [data[p][m] for m in mice if data[p][m].size > 0], axis=0
        ) if any(data[p][m].size > 0 for m in mice) else np.array([])

        if pooled.size == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            ax.set_title(f"Problem {p}", fontsize=11)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            continue

        ax.hist(pooled, bins=_auto_hist_bins(pooled), density=False,
                alpha=0.65, edgecolor="black", linewidth=0.8)
        ax.axvline(np.mean(pooled), linestyle="--", linewidth=1.5)
        ax.text(
            0.98, 0.98, _stats_text(pooled),
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      alpha=0.85, edgecolor="none"),
        )
        n_mice = sum(data[p][m].size > 0 for m in mice)
        ax.set_title(f"Problem {p} (blocks={pooled.size}, mice={n_mice})", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if xlim is not None:
            ax.set_xlim(xlim)

    for j in range(len(problems), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Block Length Distributions — Pooled Across Mice", y=1.02, fontsize=14)
    fig.supxlabel("Block length", fontsize=12)
    fig.supylabel("Counts", fontsize=12)
    plt.tight_layout()

    if save_path is not None:
        base = Path(save_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)

# ---------------------------------------------------------------------------
# Averaged rank-proportions plot (no per-block breakdown)
# ---------------------------------------------------------------------------

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
    from src.behavior_visualization.plot_style import MOUSE_COLORS as _MOUSE_COLORS

    subjects = list(rank_counts_by_good_reversal.keys())
    labels = ["Best", "Second", "Third"]
    x = np.arange(len(labels))
    jitter = 0.03

    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))

    per_mouse: dict = {}
    for subj in subjects:
        rows = rank_counts_by_good_reversal.get(subj, [])
        if not rows:
            continue
        per_mouse[subj] = {
            "best": np.nanmean([r.get("best_prop", float("nan")) for r in rows]),
            "second": np.nanmean([r.get("second_prop", float("nan")) for r in rows]),
            "third": np.nanmean([r.get("third_prop", float("nan")) for r in rows]),
        }

    mouse_means = [
        np.nanmean([v["best"] for v in per_mouse.values()]),
        np.nanmean([v["second"] for v in per_mouse.values()]),
        np.nanmean([v["third"] for v in per_mouse.values()]),
    ]
    mouse_se = [
        np.nanstd([v[k] for v in per_mouse.values()], ddof=1) / np.sqrt(len(per_mouse))
        for k in ["best", "second", "third"]
    ]

    ax.bar(x, mouse_means, yerr=mouse_se, capsize=6, width=0.55,
           color="#999999", edgecolor="black", linewidth=1.5, alpha=0.55, zorder=0)

    p_bs = average_across_mice_pvalues.get("best_vs_second") if average_across_mice_pvalues else None
    p_bt = average_across_mice_pvalues.get("best_vs_third") if average_across_mice_pvalues else None
    p_st = average_across_mice_pvalues.get("second_vs_third") if average_across_mice_pvalues else None

    xi_s = [xi for xi, m in zip(x, mouse_means) if np.isfinite(m)]
    for xi, m in zip(x, mouse_means):
        if np.isfinite(m):
            ax.text(xi, 1.05, f"{m:.2f}", ha="center", va="bottom", fontsize=12)

    if p_bs is not None and len(xi_s) > 0:
        ax.text(xi_s[0], 0.75, f"p(Best vs Second):\n{p_bs:.3f}",
                ha="center", va="bottom", fontsize=12)
    if p_bt is not None and len(xi_s) > 2:
        ax.text(xi_s[2], 0.25, f"p(Best vs Third):\n{p_bt:.3f}",
                ha="center", va="bottom", fontsize=12)
    if p_st is not None and len(xi_s) > 1:
        ax.text(xi_s[1], 0.5, f"p(Second vs Third):\n{p_st:.3f}",
                ha="center", va="bottom", fontsize=12)

    legend_handles = []
    for i, (subj, v) in enumerate(per_mouse.items()):
        c = _MOUSE_COLORS[i % len(_MOUSE_COLORS)]
        y = [v["best"], v["second"], v["third"]]
        ax.plot(x + np.random.uniform(-jitter, jitter, size=len(x)), y,
                color=c, linewidth=2.5, marker="o", alpha=0.9, markersize=6)
        legend_handles.append(Line2D([0], [0], color=c, lw=2.5, marker="o", label=subj))

    ax.set_title("Average Across Mice", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Proportion of Choices")
    ax.set_ylim(0.0, 1.07)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=legend_handles, fontsize=10, loc="upper right")

    plt.tight_layout()

    if save_path is not None:
        base = Path(save_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(base) + ".pdf", bbox_inches="tight")
        fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)
