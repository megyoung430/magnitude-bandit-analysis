"""Visualise block-length statistics across mice and sessions.

Provides a bar chart of per-block median lengths with per-mouse overlaid
lines, drawn from the summary dict returned by
:func:`src.behavior_analysis.get_task_statistics.get_block_lengths`.
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from src.behavior_visualization.plot_style import MOUSE_COLORS
from matplotlib.lines import Line2D

def plot_block_lengths(block_lengths_summary, jitter=0.06, annotate_y=None, save_path=None):
    """Plot per-block median lengths with per-mouse overlay lines.

    Draws a bar (median ± SE across mice) for each block number and overlays
    connected per-mouse data points with optional x-jitter.

    Args:
        block_lengths_summary: Dict as returned by
            :func:`src.behavior_analysis.get_task_statistics.get_block_lengths`,
            containing keys ``"blocks"``, ``"meds"``, ``"ses"``,
            ``"per_mouse_blocklens"``, and ``"mice"``.
        jitter: Maximum half-width of the uniform random x-jitter applied to
            per-mouse data points (default: ``0.06``).  Set to ``0`` to disable.
        annotate_y: Y-coordinate for bar-value text annotations.  If ``None``,
            the value is derived automatically from the tallest bar (default:
            ``None``).
        save_path: Base path (without extension) for saving ``.pdf`` and
            ``.png`` output.  If ``None``, the figure is shown interactively.
    """
    blocks = block_lengths_summary["blocks"]
    meds = np.asarray(block_lengths_summary["meds"], dtype=float)
    ses = np.asarray(block_lengths_summary["ses"], dtype=float)
    per_mouse_blocklens = block_lengths_summary["per_mouse_blocklens"]
    mice = block_lengths_summary["mice"]

    mouse_to_color = {m: MOUSE_COLORS[i % len(MOUSE_COLORS)] for i, m in enumerate(mice)}

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(blocks))

    ax.bar(x, meds, yerr=ses, capsize=6, edgecolor="black", linewidth=1.5, alpha=0.55, color=["#999999"] * len(blocks), zorder=1)

    if annotate_y is None:
        top = max(float(np.nanmax(meds + ses)) if meds.size else 1.0, 1.0)
        annotate_y = top * 1.05

    for xi, m in zip(x, meds):
        ax.text(xi, annotate_y, f"{m:.1f}", ha="center", va="bottom",
                fontsize=11, clip_on=False)

    legend_handles = []
    for m in mice:
        bl = per_mouse_blocklens[m]
        c = mouse_to_color[m]

        xs, ys = [], []
        for bi, b in enumerate(blocks):
            if b in bl:
                xs.append(bi)
                ys.append(bl[b])

        if not xs:
            continue

        xs = np.asarray(xs, float)
        ys = np.asarray(ys, float)

        if jitter > 0:
            xs = xs + np.random.uniform(-jitter, jitter, size=len(xs))

        ax.plot(xs, ys, color=c, linewidth=2.5, alpha=0.9, marker="o", markersize=6, zorder=3)

        legend_handles.append(Line2D([0],[0], color=c, lw=2.5, marker="o", label=f"{m} (n={len(ys)} blocks)"))

    ax.legend(handles=legend_handles, fontsize=10, loc="upper right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in blocks], fontsize=12)
    ax.set_xlabel("Block Number", fontsize=12)
    ax.set_ylabel("Block Length (Trials)", fontsize=12)

    ax.set_title(
        f"Block Lengths\n"
        f"(median ± se across mice | n={len(mice)} mice)",
        pad=20
    )

    top_needed = max(annotate_y, float(np.nanmax(meds + ses)) if meds.size else 1.0)
    ax.set_ylim(0, top_needed * 1.08)

    plt.tight_layout()

    if save_path:
        base = Path(save_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(base) + ".pdf", bbox_inches="tight")
        fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)