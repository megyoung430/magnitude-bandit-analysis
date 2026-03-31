"""Visualise the direction of the first leave from the previously-best arm.

After a good reversal the animal must eventually visit a different arm.  This
module plots a grouped bar chart showing what fraction of the time that first
post-reversal choice was the new best arm versus the third (lowest-reward) arm,
with individual subject lines overlaid.
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.behavior_visualization.plot_style import MOUSE_COLORS


def plot_first_leave_after_good_reversals(mean, se, per_subject_counts, p_value=None, save_path=None):
    """Plot a bar chart of first-leave proportions (new-best vs third arm).

    Draws two bars (New Best and Third Arm) with mean ± SE across subjects,
    overlays individual-subject lines with jitter, and optionally annotates
    with a p-value.

    Args:
        mean: Dict with keys ``"new_best"`` (float), ``"third"`` (float), and
            ``"num_reversals"`` (int) as returned by
            :func:`src.behavior_analysis.get_first_leave_after_good_reversals.average_first_leave_across_subjects`.
        se: Dict with keys ``"new_best"`` and ``"third"`` (floats), standard
            errors across subjects.
        per_subject_counts: ``{subject: {"new_best": int, "third": int,
            "total": int}}`` used to draw per-subject lines.
        p_value: Optional float p-value to annotate above the bars (default:
            ``None`` — no annotation).
        save_path: Base path (without extension) for saving ``.pdf`` and
            ``.png`` output.  If ``None``, the figure is shown interactively.
    """
    labels = ["New Best", "Third Arm"]
    means = [mean["new_best"], mean["third"]]
    errs = [se["new_best"], se["third"]]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    ax.bar(x,means,yerr=errs,capsize=6,edgecolor="black",linewidth=1.5,alpha=0.55,color=["#999999", "#999999"])
    for xi, m in zip(x, means):
        ax.text(xi,1.05,f"{m:.2f}",ha="center",va="bottom",fontsize=12)

    subjects = list(per_subject_counts.keys())
    legend_handles = []

    for subj in subjects:
        counts = per_subject_counts[subj]
        if counts["total"] == 0:
            continue

        nv = counts["new_best"] / counts["total"]
        tv = counts["third"] / counts["total"]

        c = MOUSE_COLORS[subjects.index(subj)]

        jitter = 0.03
        ax.plot([0 + np.random.uniform(-jitter, jitter), 1 + np.random.uniform(-jitter, jitter)],[nv, tv],color=c,linewidth=2.5,alpha=0.9,marker="o",markersize=6)
        legend_handles.append(Line2D([0], [0], color=c, lw=2.5, marker="o", label=f"{subj} (n={counts['total']} reversals)"))

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

    if p_value is not None:
        ax.text(0.5, 1.02, f"p-value: {p_value:.3f}", ha="center", va="bottom", fontsize=12)

    plt.tight_layout()

    if save_path:
        base = Path(save_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(base) + ".pdf", bbox_inches="tight")
        fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
