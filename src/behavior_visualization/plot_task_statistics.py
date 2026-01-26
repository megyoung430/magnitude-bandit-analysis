import numpy as np
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial"]
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["axes.titlesize"] = 14

MOUSE_COLORS = [
    "#4C72B0","#55A868","#C44E52","#8172B2",
    "#CCB974","#64B5CD","#8C8C8C","#DD8452",
    "#937860","#DA8BC3","#8C6D31","#1F77B4",
]

def plot_block_lengths(block_lengths_summary, jitter=0.06, annotate_y=None, save_path=None):
    """
    Plots output from compute_block_length_summary(...).
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