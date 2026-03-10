"""Plotting functions for cumulative and session-by-session reversal counts.

Two high-level plotting modes are provided:

- :func:`plot_num_reversals` — a simple per-subject bar chart of total
  good/bad reversals (or total reversals for non-perf-dependent paradigms).
- :func:`plot_num_reversals_over_time` — cumulative line plots aligned by
  session number, showing both an across-mice mean and per-mouse faint lines.
- :func:`plot_moving_avg_reversals_over_time` — same layout but using a
  centered moving average instead of cumulative counts.
"""
import re
import math
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from src.behavior_analysis.get_total_reversals import get_total_reversals
from src.behavior_visualization.plot_style import GOOD_COLOR, BAD_COLOR, TOTAL_COLOR

def plot_num_reversals(subjects_trials, save_path=None):
    """Plot total good and bad reversal counts per subject as a grouped bar chart.

    For performance-dependent paradigms (sessions with ``has_good`` /
    ``has_bad``), draws side-by-side bars coloured by reversal type.  For
    non-performance-dependent paradigms, draws a single total-reversal bar
    per subject.

    Args:
        subjects_trials: Nested dict ``{subject: {session_key: session_dict}}``.
        save_path: Base path (without extension) for saving ``.pdf`` and
            ``.png`` output.  If ``None``, the figure is shown interactively.
    """

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
        ax.bar(x_pos, values, color=TOTAL_COLOR, edgecolor="black")
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
    """Plot cumulative reversal counts over sessions for all subjects.

    Produces two figures:

    1. *Across Mice* — faint individual lines plus a bold mean line,
       using carry-forward padding so all subjects span the same x-axis.
    2. *By Mouse* — a grid of per-subject cumulative plots.

    For perf-dependent paradigms, good and bad reversals are plotted as
    separate lines.  For non-perf-dependent paradigms, total reversals are
    used.

    Args:
        subjects_trials: Nested dict ``{subject: {session_key: session_dict}}``.
        threshold: Horizontal reference line drawn on the across-mice plot
            (e.g. the criterion reversal count).  Pass ``None`` to suppress
            (default: 10).
        save_path: Base path (without extension) for saving ``.pdf`` and
            ``.png`` files.  Two suffixes are appended: ``" Across Mice"``
            and ``" By Mouse"``.  If ``None``, figures are shown interactively.

    Raises:
        ValueError: If *subjects_trials* is empty or contains sessions with
            unparseable keys.
    """
    # carry-forward padding so mean stays monotonic nondecreasing
    def pad_carry_forward(arr, n):
        """Extend *arr* to length *n* by repeating the last value."""
        arr = np.asarray(arr, dtype=float)
        out = np.empty(n, dtype=float)
        out[:] = arr[-1]
        out[:len(arr)] = arr
        return out

    all_subjects = sorted(subjects_trials.keys())

    # ---- helper: parse session number from session key ----
    # works with keys like: "ses-12_date-20260116" OR "ses-12" OR "ses-12_date-..."
    ses_re = re.compile(r"ses-(\d+)")

    def session_int(session_key: str) -> int:
        """Return the integer session number from a ``ses-N`` session key."""
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

            if "good_reversals" in stats and "bad_reversals" in stats:
                rows_by_subj[subj].append({
                    "ses_int": s_int,
                    "ses_label": s_label,
                    "good": stats.get("good_reversals", 0),
                    "bad": stats.get("bad_reversals", 0),
                })
            else: 
                rows_by_subj[subj].append({
                    "ses_int": s_int,
                    "ses_label": s_label,
                    "total": stats.get("total_reversals", 0),
                })

    subjects = sorted(rows_by_subj.keys())
    if len(subjects) == 0:
        raise ValueError("No subjects found in subjects_trials")

    if all("good" in rows_by_subj[subjects[0]][0] and "bad" in rows_by_subj[subjects[0]][0] for subj in subjects):
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
    else:
        # If no subjects have good/bad reversals, use total reversals
        cum_total_by_subj = {}
        for subj in subjects:
            rows = sorted(rows_by_subj[subj], key=lambda d: d["ses_int"])
            t_sum = 0
            ct = []
            for d in rows:
                t_sum += d["total"]
                ct.append(t_sum)
            cum_total_by_subj[subj] = np.array(ct, dtype=float)
        max_len = max(len(v) for v in cum_total_by_subj.values())
        min_len = min(len(v) for v in cum_total_by_subj.values())
        
        total_mat = np.vstack([pad_carry_forward(v, max_len) for v in cum_total_by_subj.values()])
        mean_total = total_mat.mean(axis=0)

        # =========================
        # Figure 1: Across mice
        # =========================
        fig1, ax1 = plt.subplots(figsize=(10, 6))

        for subj in subjects:
            ax1.plot(cum_total_by_subj[subj], color=TOTAL_COLOR, alpha=0.4, linewidth=1.5)

        ax1.plot(mean_total, color="#000000", linewidth=3, label="Mean Total")

        if threshold is not None:
            ax1.axhline(threshold, color="#DCDCDC", linewidth=2, linestyle="--", alpha=0.7, label="Threshold")

        ax1.set_xlabel("Session Number")
        ax1.set_ylabel("Cumulative Reversals")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.set_title(
            "Cumulative Total Reversals Over Time\n"
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
            total = [d["total"] for d in rows]

            cum_total = []
            t = 0
            for tt in total:
                t += tt
                cum_total.append(t)

            ax.plot(cum_total, color=TOTAL_COLOR, label="Total" if i == 0 else None)

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
        fig2.suptitle("Cumulative Total Reversals by Subject", y=1.02, fontsize=14)
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

def plot_moving_avg_reversals_over_time(subjects_trials, *, window: int = 3, save_path=None):
    """Plot session-by-session moving-average reversal rates over time.

    Produces two figures (same layout as :func:`plot_num_reversals_over_time`
    but using a centered moving average instead of cumulative counts):

    1. *Across Mice* — individual faint lines plus bold mean, aligned by
       relative session index and padded with the last finite value.
    2. *By Mouse* — a grid of per-subject moving-average plots.

    Args:
        subjects_trials: Nested dict ``{subject: {session_key: session_dict}}``.
        window: Kernel width for the centered moving average (default: 3).
            Near the array edges the kernel shrinks to use only available
            data points.
        save_path: Base path (without extension) for saving ``.pdf`` and
            ``.png`` files.  Two suffixes are appended: ``" Across Mice"``
            and ``" By Mouse"``.  If ``None``, figures are shown interactively.

    Raises:
        ValueError: If *subjects_trials* is empty or sessions have
            unparseable keys.
    """

    all_subjects = sorted(subjects_trials.keys())
    if not all_subjects:
        raise ValueError("subjects_trials is empty")

    # ---- helper: parse session number from session key ----
    ses_re = re.compile(r"ses-(\d+)")

    def session_int(session_key: str) -> int:
        """Return the integer session number from a ``ses-N`` session key."""
        m = ses_re.search(session_key)
        if m is None:
            raise ValueError(f"Session key missing 'ses-<n>': {session_key}")
        return int(m.group(1))

    # ---- helper: centered moving average (NaN-aware) ----
    def moving_average_1d(x: np.ndarray, window: int) -> np.ndarray:
        """Apply a centered NaN-aware moving average of *window* width to *x*."""
        x = np.asarray(x, dtype=float)
        n = len(x)
        if window <= 1:
            return x.copy()

        out = np.full(n, np.nan)
        half = window // 2
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            w = x[lo:hi]
            if np.any(np.isfinite(w)):
                out[i] = np.nanmean(w)
        return out
    
    # ---- carry-forward padding (like cumulative) ----
    def pad_carry_forward_finite(arr: np.ndarray, n: int) -> np.ndarray:
        """
        Pads to length n by repeating the last FINITE value.
        If arr has no finite values, pads with NaNs.
        """
        arr = np.asarray(arr, dtype=float)
        out = np.empty(n, dtype=float)

        finite = np.isfinite(arr)
        if not np.any(finite):
            out[:] = np.nan
            return out

        last = arr[np.where(finite)[0][-1]]
        out[:] = last
        out[:len(arr)] = arr

        # If there are NaNs inside arr (unlikely here), forward-fill them too.
        for i in range(1, n):
            if not np.isfinite(out[i]):
                out[i] = out[i - 1]
        if not np.isfinite(out[0]):
            out[0] = last

        return out

    # ---- extract per-session reversal counts + labels ----
    rows_by_subj = defaultdict(list)

    for subj in all_subjects:
        for sess_key, trials in subjects_trials[subj].items():
            stats = get_total_reversals({sess_key: trials})

            s_int = session_int(sess_key)
            s_label = f"ses-{s_int:02d}"
            
            if "good_reversals" in stats and "bad_reversals" in stats:
                rows_by_subj[subj].append({
                    "ses_int": s_int,
                    "ses_label": s_label,
                    "good": stats.get("good_reversals", 0),
                    "bad": stats.get("bad_reversals", 0),
                })
            else:
                rows_by_subj[subj].append({
                    "ses_int": s_int,
                    "ses_label": s_label,
                    "total": stats.get("total_reversals", 0),
                })

    subjects = sorted([s for s in rows_by_subj.keys() if len(rows_by_subj[s]) > 0])
    if not subjects:
        raise ValueError("No valid subjects after parsing sessions")

    if all("good" in rows_by_subj[subjects[0]][0] and "bad" in rows_by_subj[subjects[0]][0] for subj in subjects):
        # ---- per-subject arrays ----
        good_by_subj, bad_by_subj = {}, {}
        for subj in subjects:
            rows = sorted(rows_by_subj[subj], key=lambda d: d["ses_int"])
            good_by_subj[subj] = np.array([r["good"] for r in rows], dtype=float)
            bad_by_subj[subj]  = np.array([r["bad"]  for r in rows], dtype=float)

        # ---- moving average per mouse ----
        ma_good = {s: moving_average_1d(good_by_subj[s], window) for s in subjects}
        ma_bad  = {s: moving_average_1d(bad_by_subj[s],  window) for s in subjects}

        max_len = max(len(v) for v in ma_good.values())
        min_len = min(len(v) for v in ma_good.values())

        good_mat = np.vstack([pad_carry_forward_finite(ma_good[s], max_len) for s in subjects])
        bad_mat  = np.vstack([pad_carry_forward_finite(ma_bad[s],  max_len) for s in subjects])

        mean_good = np.nanmean(good_mat, axis=0)
        mean_bad  = np.nanmean(bad_mat,  axis=0)

        # =========================
        # Figure 1: Across mice
        # =========================
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        x = np.arange(1, max_len + 1)

        for s in subjects:
            ax1.plot(x[:len(ma_good[s])], ma_good[s], color=GOOD_COLOR, alpha=0.25, linewidth=1.5)
            ax1.plot(x[:len(ma_bad[s])],  ma_bad[s],  color=BAD_COLOR,  alpha=0.25, linewidth=1.5)

        ax1.plot(x, mean_good, color=GOOD_COLOR, linewidth=3, label="Good Rev")
        ax1.plot(x, mean_bad,  color=BAD_COLOR,  linewidth=3, label="Bad Rev")

        ax1.set_xlabel("Session Number")
        ax1.set_ylabel("Moving-Average Reversals / Session")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        title = (
            "Moving-Average Good and Bad Reversals Over Time\n"
            f"(n={len(subjects)} subjects; sessions per subject={min_len}–{max_len}; window={window})"
        )
        ax1.set_title(title)
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
            good = np.array([d["good"] for d in rows], dtype=float)
            bad  = np.array([d["bad"]  for d in rows], dtype=float)

            ma_g = moving_average_1d(good, window)
            ma_b = moving_average_1d(bad,  window)

            ax.plot(ma_g, color=GOOD_COLOR, label="Good Rev" if i == 0 else None)
            ax.plot(ma_b, color=BAD_COLOR,  label="Bad Rev"  if i == 0 else None)

            ax.set_title(f"{subj}\n(n={len(rows)} sessions; window={window})", fontsize=12)
            ax.set_xlabel("Session")
            ax.set_ylabel("Moving-Avg Reversals / Session")

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
        fig2.suptitle("Moving-Average Good and Bad Reversals by Subject", y=1.02, fontsize=14)
        fig2.tight_layout()
    else:
        # ---- per-subject arrays ----
        total_by_subj = {}
        for subj in subjects:
            rows = sorted(rows_by_subj[subj], key=lambda d: d["ses_int"])
            total_by_subj[subj] = np.array([r["total"] for r in rows], dtype=float)

        # ---- moving average per mouse ----
        ma_total = {s: moving_average_1d(total_by_subj[s], window) for s in subjects}

        max_len = max(len(v) for v in ma_total.values())
        min_len = min(len(v) for v in ma_total.values())

        total_mat = np.vstack([pad_carry_forward_finite(ma_total[s], max_len) for s in subjects])
        mean_total = np.nanmean(total_mat, axis=0)

        # =========================
        # Figure 1: Across mice
        # =========================
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        x = np.arange(1, max_len + 1)

        for s in subjects:
            ax1.plot(x[:len(ma_total[s])], ma_total[s], color=TOTAL_COLOR, alpha=0.25, linewidth=1.5)

        ax1.plot(x, mean_total, color=TOTAL_COLOR, linewidth=3, label="Total Rev")

        ax1.set_xlabel("Session Number")
        ax1.set_ylabel("Moving-Average Reversals / Session")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        title = (
            "Moving-Average Total Reversals Over Time\n"
            f"(n={len(subjects)} subjects; sessions per subject={min_len}–{max_len}; window={window})"
        )
        ax1.set_title(title)
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
            total  = np.array([d["total"]  for d in rows], dtype=float)

            ma_total = moving_average_1d(total, window)

            ax.plot(ma_total, color=TOTAL_COLOR, label="Total Rev" if i == 0 else None)

            ax.set_title(f"{subj}\n(n={len(rows)} sessions; window={window})", fontsize=12)
            ax.set_xlabel("Session")
            ax.set_ylabel("Moving-Avg Reversals / Session")

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
        fig2.suptitle("Moving-Average Total Reversals by Subject", y=1.02, fontsize=14)
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
