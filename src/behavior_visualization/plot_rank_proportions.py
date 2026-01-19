import numpy as np
from pathlib import Path
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial"]
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["axes.titlesize"] = 14

def plot_rank_proportions(rank_counts_by_good_reversal, save_path=None, by_mouse=True, by_block=True, average=True):
    if by_mouse:
        mouse_save_path = save_path + " by Mouse" if save_path else None
        plot_rank_proportions_by_mouse(rank_counts_by_good_reversal, save_path=mouse_save_path)
    if by_block:
        block_save_path = save_path + " by Block" if save_path else None
        plot_rank_proportions_by_block(rank_counts_by_good_reversal, save_path=block_save_path)
    if average:
        average_save_path = save_path + " by Average" if save_path else None
        plot_rank_proportions_average(rank_counts_by_good_reversal, save_path=average_save_path)

def plot_rank_proportions_average(rank_counts_by_good_reversal, save_path=None):
    """
    Two-panel summary:
      (Left)  Average across mice: bars = mean across mice, lines = individual mice (mouse colors).
      (Right) Average across blocks: bars = mean across blocks, lines = individual blocks (block colors).

    rank_counts_by_good_reversal[subj] should be a list of dicts with keys:
      best_prop, second_prop, third_prop, total, ...
    """

    # Mouse colors
    mouse_colors = [
        "#4C72B0",  # blue
        "#55A868",  # green
        "#C44E52",  # red
        "#8172B2",  # purple
        "#CCB974",  # yellow-brown
        "#64B5CD",  # cyan
        "#8C8C8C",  # gray
    ]

    # Block colors
    block_colors = [
        "#1B9E77",  # teal
        "#D95F02",  # orange
        "#7570B3",  # blue-purple
        "#E7298A",  # magenta
        "#66A61E",  # olive
        "#E6AB02",  # mustard
        "#A6761D",  # brown
        "#666666",  # dark gray
        "#1F78B4",  # steel blue
        "#33A02C",  # green
    ]

    subjects = list(rank_counts_by_good_reversal.keys())
    labels = ["Best", "Second", "Third"]
    x = np.arange(len(labels))
    jitter = 0.03

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    # ============================================================
    # LEFT: Average across mice
    # ============================================================
    ax = axes[0]

    per_mouse = {}
    for subj in subjects:
        rows = rank_counts_by_good_reversal.get(subj, [])
        if not rows:
            continue

        per_mouse[subj] = {
            "best": np.nanmean([r.get("best_prop", np.nan) for r in rows]),
            "second": np.nanmean([r.get("second_prop", np.nan) for r in rows]),
            "third": np.nanmean([r.get("third_prop", np.nan) for r in rows]),
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

    ax.bar(x, mouse_means, yerr=mouse_se, capsize=6, width=0.55, color="#999999", edgecolor="black", linewidth=1.5, alpha=0.55, zorder=0)

    for xi, m in zip(x, mouse_means):
        if np.isfinite(m):
            ax.text(xi, 1.05, f"{m:.2f}", ha="center", va="bottom", fontsize=12)

    legend_handles = []
    for i, (subj, v) in enumerate(per_mouse.items()):
        c = mouse_colors[i % len(mouse_colors)]
        y = [v["best"], v["second"], v["third"]]
        ax.plot(x + np.random.uniform(-jitter, jitter, size=len(x)), y, color=c, linewidth=2.5, marker="o", alpha=0.9, markersize=6,)
        legend_handles.append(Line2D([0], [0], color=c, lw=2.5, marker="o", label=subj))

    ax.set_title("Average Across Mice", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Proportion of Choices")
    ax.set_ylim(0.0, 1.07)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=legend_handles, fontsize=10, loc="upper right")

    # ============================================================
    # RIGHT: Average across blocks
    # ============================================================
    ax = axes[1]

    max_blocks = max(len(v) for v in rank_counts_by_good_reversal.values()) if rank_counts_by_good_reversal else 0

    per_block = []
    for b in range(max_blocks):
        block_rows = []
        for subj in subjects:
            rows = rank_counts_by_good_reversal.get(subj, [])
            if b < len(rows):
                block_rows.append(rows[b])

        if not block_rows:
            continue

        per_block.append({
            "best": np.nanmean([r.get("best_prop", np.nan) for r in block_rows]),
            "second": np.nanmean([r.get("second_prop", np.nan) for r in block_rows]),
            "third": np.nanmean([r.get("third_prop", np.nan) for r in block_rows]),
        })

    block_means = [
        np.nanmean([b["best"] for b in per_block]) if per_block else np.nan,
        np.nanmean([b["second"] for b in per_block]) if per_block else np.nan,
        np.nanmean([b["third"] for b in per_block]) if per_block else np.nan,
    ]
    block_se = [
        np.nanstd([b[k] for b in per_block], ddof=1) / np.sqrt(len(per_block))
        for k in ["best", "second", "third"]
    ]

    ax.bar(x, block_means, yerr=block_se, capsize=6, width=0.55, color="#999999", edgecolor="black", linewidth=1.5, alpha=0.55, zorder=0)
    for xi, m in zip(x, block_means):
        if np.isfinite(m):
            ax.text(xi, 1.05, f"{m:.2f}", ha="center", va="bottom", fontsize=12)

    block_legend = []
    for bi, b in enumerate(per_block):
        y = [b["best"], b["second"], b["third"]]
        c = block_colors[bi % len(block_colors)]
        ax.plot(x + np.random.uniform(-jitter, jitter, size=len(x)), y, color=c, linewidth=2.2, marker="o", alpha=0.85)
        block_legend.append(Line2D([0], [0], color=c, lw=2.2, marker="o", label=f"Block {bi + 1}"))

    ax.set_title("Average Across Blocks", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.07)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=block_legend, fontsize=10, loc="upper right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path + ".pdf", bbox_inches="tight")
        fig.savefig(save_path + ".png", dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)

def plot_rank_proportions_by_mouse(rank_counts_by_good_reversal, save_path=None):
    subjects = list(rank_counts_by_good_reversal.keys())
    n_subj = len(subjects)

    nrows = 2
    ncols = math.ceil(n_subj / 2)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 8), sharey=True)
    axes = axes.flatten()

    labels = ["Best", "Second", "Third"]
    x = np.arange(len(labels))

    # Block colors
    block_colors = [
        "#1B9E77",  # teal
        "#D95F02",  # orange
        "#7570B3",  # blue-purple
        "#E7298A",  # magenta
        "#66A61E",  # olive
        "#E6AB02",  # mustard
        "#A6761D",  # brown
        "#666666",  # dark gray
        "#1F78B4",  # steel blue
        "#33A02C",  # green
    ]

    for ax_idx, ax in enumerate(axes):
        if ax_idx >= n_subj:
            ax.axis("off")
            continue

        subj = subjects[ax_idx]
        rows = rank_counts_by_good_reversal[subj]

        legend_handles = []
        jitter = 0.04

        for rev_idx, r in enumerate(rows):
            y = [r["best_prop"], r["second_prop"], r["third_prop"]]
            c = block_colors[rev_idx % len(block_colors)]
            ax.plot(x + np.random.uniform(-jitter, jitter, size=len(x)), y, color=c, linewidth=2.2, alpha=0.85, marker="o", markersize=5)
            legend_handles.append(Line2D([0], [0], color=c, lw=2.2, marker="o", label=f"Block {rev_idx + 1}"))
        mean_y = [np.nanmean([r["best_prop"] for r in rows]), 
                  np.nanmean([r["second_prop"] for r in rows]), 
                  np.nanmean([r["third_prop"] for r in rows])]
        se_y = [np.nanstd([r["best_prop"] for r in rows], ddof=1) / np.sqrt(len(rows)) if len(rows) > 1 else np.nan,
                np.nanstd([r["second_prop"] for r in rows], ddof=1) / np.sqrt(len(rows)) if len(rows) > 1 else np.nan,
                np.nanstd([r["third_prop"] for r in rows], ddof=1) / np.sqrt(len(rows)) if len(rows) > 1 else np.nan]
        ax.bar(x, mean_y, yerr=se_y, capsize=6, width=0.55, color="#9E9E9E", edgecolor="black", linewidth=1.4, alpha=0.45, zorder=0)
        for xi, m in zip(x, mean_y):
            ax.text(xi, 1.05, f"{m:.2f}", ha="center", va="bottom", fontsize=12)

        ax.set_title(subj, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.07)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if ax_idx % ncols == 0:
            ax.set_ylabel("Proportion of Choices")

        ax.legend(handles=legend_handles[:8], fontsize=10, loc="upper right")

    plt.tight_layout()

    if save_path:
        base = Path(save_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(base) + ".pdf", bbox_inches="tight")
        fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)

def plot_rank_proportions_by_block(rank_counts_by_good_reversal, save_path=None):
    # Mouse colors
    mouse_colors = [
        "#4C72B0",  # blue
        "#55A868",  # green
        "#C44E52",  # red
        "#8172B2",  # purple
        "#CCB974",  # yellow-brown
        "#64B5CD",  # cyan
        "#8C8C8C",  # gray
    ]

    subjects = list(rank_counts_by_good_reversal.keys())

    max_blocks = 0
    for subj in subjects:
        rows = rank_counts_by_good_reversal.get(subj, [])
        max_blocks = max(max_blocks, len(rows))

    if max_blocks == 0:
        print("[WARN] No blocks found to plot.")
        return

    nrows = 2
    ncols = math.ceil(max_blocks / 2)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 8), sharey=True)
    axes = np.array(axes).flatten()

    labels = ["Best", "Second", "Third"]
    x = np.arange(len(labels))
    bar_width = 0.55
    jitter = 0.03

    for block_idx in range(max_blocks):
        ax = axes[block_idx]

        legend_handles = []

        for si, subj in enumerate(subjects):
            rows = rank_counts_by_good_reversal.get(subj, [])
            if block_idx >= len(rows):
                continue
            r = rows[block_idx]
            y = [r.get("best_prop", np.nan), r.get("second_prop", np.nan), r.get("third_prop", np.nan)]
            c = mouse_colors[si % len(mouse_colors)]
            ax.plot(x + np.random.uniform(-jitter, jitter, size=len(x)), y, color=c, linewidth=2.5, alpha=0.9, marker="o", markersize=5.5)
            n_trials = r.get("total", None)
            lbl = f"{subj}" if n_trials is None else f"{subj} (n={n_trials} trials)"
            legend_handles.append(Line2D([0], [0], color=c, lw=2.5, marker="o", label=lbl))

        block_best = []
        block_second = []
        block_third = []

        for subj in subjects:
            rows = rank_counts_by_good_reversal.get(subj, [])
            if block_idx >= len(rows):
                continue
            rr = rows[block_idx]
            block_best.append(rr.get("best_prop", np.nan))
            block_second.append(rr.get("second_prop", np.nan))
            block_third.append(rr.get("third_prop", np.nan))

        mean_y = [np.nanmean(block_best) if len(block_best) else np.nan, 
                  np.nanmean(block_second) if len(block_second) else np.nan, 
                  np.nanmean(block_third) if len(block_third) else np.nan]
        se_y = [np.nanstd(block_best, ddof=1) / np.sqrt(len(block_best)) if len(block_best) > 1 else np.nan,
                np.nanstd(block_second, ddof=1) / np.sqrt(len(block_second)) if len(block_second) > 1 else np.nan,
                np.nanstd(block_third, ddof=1) / np.sqrt(len(block_third)) if len(block_third) > 1 else np.nan]
        ax.bar(x, mean_y, yerr=se_y, capsize=6, width=bar_width, color="#999999", edgecolor="black", linewidth=1.5, alpha=0.55, zorder=0)

        for xi, m in zip(x, mean_y):
            if np.isfinite(m):
                ax.text(xi, 1.05, f"{m:.2f}", ha="center", va="bottom", fontsize=11)

        ax.set_title(f"Block {block_idx + 1}", pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylim(0.0, 1.07)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if block_idx % ncols == 0:
            ax.set_ylabel("Proportion of Choices", fontsize=12)

        if legend_handles:
            ax.legend(handles=legend_handles, fontsize=10, loc="upper right")

    for j in range(max_blocks, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if save_path:
        base = Path(save_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(base) + ".pdf", bbox_inches="tight")
        fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)