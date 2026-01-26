import re
import math
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib as mpl
import matplotlib.pyplot as plt
from src.behavior_analysis.get_total_reversals import *

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial"]
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["axes.titlesize"] = 14

GOOD_COLOR = "#3A982E"
BAD_COLOR = "#F97979"

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

        ax.bar(x_pos - bar_width / 2, good_values, width=bar_width, color=GOOD_COLOR, alpha=0.7, label="Good Reversals", edgecolor="black")
        ax.bar(x_pos + bar_width / 2, bad_values, width=bar_width, color=BAD_COLOR, alpha=0.7, label="Bad Reversals", edgecolor="black")

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

def plot_num_reversals_over_time(subjects_trials, threshold=10, save_path=None):
    """
    Inputs:
      subjects_trials[subj][session_key] -> trials (whatever get_total_reversals expects)
    Saves:
      1) Across Mice (individual faint + mean bold; carry-forward padded)
      2) By Mouse (grid of per-subject plots)
    """

    all_subjects = sorted(subjects_trials.keys())

    # ---- helper: parse session number from session key ----
    # works with keys like: "ses-12_date-20260116" OR "ses-12" OR "ses-12_date-..."
    ses_re = re.compile(r"ses-(\d+)")

    def session_int(session_key: str) -> int:
        m = ses_re.search(session_key)
        if m is None:
            raise ValueError(f"Session key does not contain 'ses-<n>': {session_key}")
        return int(m.group(1))

    # ---- build per-subject per-session reversal counts ----
    rows_by_subj = defaultdict(list)

    for subj in all_subjects:
        totals = get_total_reversals(subjects_trials[subj])
        print(f"{subj}: {totals}")

        for sess_key, trials in subjects_trials[subj].items():
            stats = get_total_reversals({sess_key: trials})

            s_int = session_int(sess_key)
            s_label = f"ses-{s_int:02d}"

            rows_by_subj[subj].append({
                "ses_int": s_int,
                "ses_label": s_label,
                "good": stats.get("good_reversals", 0),
                "bad": stats.get("bad_reversals", 0),
            })

    subjects = sorted(rows_by_subj.keys())
    if len(subjects) == 0:
        raise ValueError("No subjects found in subjects_trials")

    # ---- cumulative arrays per subject ----
    cum_good_by_subj = {}
    cum_bad_by_subj = {}

    for subj in subjects:
        rows = sorted(rows_by_subj[subj], key=lambda d: d["ses_int"])
        g_sum = b_sum = 0
        cg, cb = [], []
        for d in rows:
            g_sum += d["good"]
            b_sum += d["bad"]
            cg.append(g_sum)
            cb.append(b_sum)
        cum_good_by_subj[subj] = np.array(cg, dtype=float)
        cum_bad_by_subj[subj]  = np.array(cb, dtype=float)

    max_len = max(len(v) for v in cum_good_by_subj.values())
    min_len = min(len(v) for v in cum_good_by_subj.values())

    # carry-forward padding so mean stays monotonic nondecreasing
    def pad_carry_forward(arr, n):
        arr = np.asarray(arr, dtype=float)
        out = np.empty(n, dtype=float)
        out[:] = arr[-1]
        out[:len(arr)] = arr
        return out

    good_mat = np.vstack([pad_carry_forward(v, max_len) for v in cum_good_by_subj.values()])
    bad_mat  = np.vstack([pad_carry_forward(v, max_len) for v in cum_bad_by_subj.values()])

    mean_good = good_mat.mean(axis=0)
    mean_bad  = bad_mat.mean(axis=0)

    # =========================
    # Figure 1: Across mice
    # =========================
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    for subj in subjects:
        ax1.plot(cum_good_by_subj[subj], color=GOOD_COLOR, alpha=0.4, linewidth=1.5)
        ax1.plot(cum_bad_by_subj[subj],  color=BAD_COLOR,  alpha=0.4, linewidth=1.5)

    ax1.plot(mean_good, color=GOOD_COLOR, linewidth=3, label="Good Rev")
    ax1.plot(mean_bad,  color=BAD_COLOR,  linewidth=3, label="Bad Rev")

    if threshold is not None:
        ax1.axhline(threshold, color="#DCDCDC", linewidth=2, linestyle="--", alpha=0.7, label="Threshold")

    ax1.set_xlabel("Session Number")
    ax1.set_ylabel("Cumulative Reversals")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_title(
        "Cumulative Good and Bad Reversals Over Time\n"
        f"(n={len(subjects)} subjects; sessions per subject={min_len}–{max_len})"
    )
    ax1.legend(fontsize=10)
    fig1.tight_layout()

    # =========================
    # Figure 2: By mouse (grid)
    # =========================
    n_subj = len(subjects)
    ncols = 2
    nrows = math.ceil(n_subj / ncols)

    fig2, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3.5 * nrows), sharey=False)
    axes = np.array(axes).reshape(-1)

    for i, subj in enumerate(subjects):
        ax = axes[i]
        rows = sorted(rows_by_subj[subj], key=lambda d: d["ses_int"])

        x_labels = [d["ses_label"] for d in rows]
        good = [d["good"] for d in rows]
        bad = [d["bad"] for d in rows]

        cum_good, cum_bad = [], []
        g = b = 0
        for gg, bb in zip(good, bad):
            g += gg
            b += bb
            cum_good.append(g)
            cum_bad.append(b)

        ax.plot(cum_good, color=GOOD_COLOR, label="Good Rev" if i == 0 else None)
        ax.plot(cum_bad,  color=BAD_COLOR,  label="Bad Rev" if i == 0 else None)

        if threshold is not None:
            ax.axhline(threshold, color="#DCDCDC", linewidth=2, linestyle="--", alpha=0.7,
                       label="Threshold" if i == 0 else None)

        ax.set_title(f"{subj}\n(n={len(rows)} sessions)", fontsize=12)
        ax.set_xlabel("Session")
        ax.set_ylabel("Cumulative Reversals")

        step = max(1, len(x_labels) // 10)
        tick_idx = list(range(0, len(x_labels), step))
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([x_labels[j] for j in tick_idx], rotation=45, ha="right")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(n_subj, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc="upper right")
    fig2.suptitle("Cumulative Good and Bad Reversals by Subject", y=1.02, fontsize=14)
    fig2.tight_layout()

    # =========================
    # Save or show
    # =========================
    if save_path:
        base = Path(save_path)
        base.parent.mkdir(parents=True, exist_ok=True)

        fig1.savefig(str(base) + " Across Mice.pdf", bbox_inches="tight")
        fig1.savefig(str(base) + " Across Mice.png", dpi=300, bbox_inches="tight")

        fig2.savefig(str(base) + " By Mouse.pdf", bbox_inches="tight")
        fig2.savefig(str(base) + " By Mouse.png", dpi=300, bbox_inches="tight")

        plt.close(fig1)
        plt.close(fig2)
    else:
        plt.show()