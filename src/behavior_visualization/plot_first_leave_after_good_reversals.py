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

def plot_first_leave_after_good_reversals(mean, se, per_subject_counts, save_path=None):
    labels = ["New Best", "Third Arm"]
    means = [mean["new_best"], mean["third"]]
    errs = [se["new_best"], se["third"]]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    ax.bar(x,means,yerr=errs,capsize=6,edgecolor="black",linewidth=1.5,alpha=0.55,color=["#999999", "#999999"])
    for xi, m in zip(x, means):
        ax.text(xi,1.05,f"{m:.2f}",ha="center",va="bottom",fontsize=12)

    subjects = list(per_subject_counts.keys())
    colors = [
        "#4C72B0",  # blue
        "#55A868",  # green
        "#C44E52",  # red
        "#8172B2",  # purple
        "#CCB974",  # yellow-brown
        "#64B5CD",  # cyan
        "#8C8C8C",  # gray
    ]
    legend_handles = []

    for subj in subjects:
        counts = per_subject_counts[subj]
        if counts["total"] == 0:
            continue

        nv = counts["new_best"] / counts["total"]
        tv = counts["third"] / counts["total"]

        c = colors[subjects.index(subj)]

        jitter = 0.03
        ax.plot([0 + np.random.uniform(-jitter, jitter), 1 + np.random.uniform(-jitter, jitter)],[nv, tv],color=c,linewidth=2.5,alpha=0.9,marker="o",markersize=6)
        legend_handles.append(Line2D([0], [0], color=c, lw=2.5, marker="o", label=subj))

    ax.legend(handles=legend_handles, fontsize=10, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Fraction of Good Reversals", fontsize=12)
    ax.set_ylim(0.0, 1.07)
    ax.set_title(
        f"First Choice After Leaving Previous Best\n"
        f"(mean ± se across subjects | n={len(per_subject_counts)} subjects "
        f"and n={mean['num_reversals']} reversals)", pad=20
    )

    plt.tight_layout()

    if save_path:
        base = Path(save_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(base) + ".pdf", bbox_inches="tight")
        fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)