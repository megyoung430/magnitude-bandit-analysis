import numpy as np
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from src.behavior_analysis.get_total_reversals import *

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial"]
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["axes.titlesize"] = 14

def plot_num_reversals(subjects_trials, save_path=None):

    all_subjects = sorted(subjects_trials.keys())
    per_subj_stats = {}
    for subj in all_subjects:
        per_subj_stats[subj] = get_total_reversals(subjects_trials[subj])
        print(f"{subj}: {per_subj_stats[subj]}")
        
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(all_subjects))
    bar_width = 0.35

    if "good_reversals" in per_subj_stats[all_subjects[0]] and "bad_reversals" in per_subj_stats[all_subjects[0]]:
        good_values = [int(per_subj_stats[subj]["good_reversals"] or 0) for subj in all_subjects]
        bad_values = [int(per_subj_stats[subj]["bad_reversals"] or 0) for subj in all_subjects]

        ax.bar(x_pos - bar_width / 2, good_values, width=bar_width, color="#3A982E", alpha=0.7, label="Good Reversals", edgecolor="black")
        ax.bar(x_pos + bar_width / 2, bad_values, width=bar_width, color="#F97979", alpha=0.7, label="Bad Reversals", edgecolor="black")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(all_subjects)
        ax.legend(loc="upper left", fontsize=10)
    else:
        values = [int(per_subj_stats[subj].get("total_reversals", 0) or 0) for subj in all_subjects]
        ax.bar(x_pos, values, color="gray", edgecolor="black")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(all_subjects)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)

    ax.set_ylabel("Number of Reversals", fontsize=12)
    ax.set_xlabel("Subject", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        base = Path(save_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(base) + ".pdf", bbox_inches="tight")
        fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)