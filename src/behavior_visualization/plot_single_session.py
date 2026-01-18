import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial"]
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12

def plot_single_session(session_data, mag_key="reward_magnitudes_by_tower", choice_key="choices_by_tower", 
                        ema_keys=("ema_best_arm_choices", "ema_second_arm_choices"), title=None, save_path=None):
    TOWER_COLORS = ["#A6CEE3", "#FDBF6F", "#B2DF8A"]
    EMA_LABELS = {
        "ema_best_arm_choices": "Best",
        "ema_second_arm_choices": "Second",
    }
    EMA_COLORS = {
        "ema_best_arm_choices": "#000000",
        "ema_second_arm_choices": "#858585",
    }

    mags_by = session_data[mag_key]
    choices_by = session_data[choice_key]
    towers = list(mags_by.keys())

    N = len(mags_by[towers[0]])
    x = np.arange(N)

    fig = plt.figure(figsize=(14, 11))
    gs = GridSpec(3, 2, height_ratios=[2.2, 1.8, 1.5], hspace=0.25, wspace=0.25)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])

    # --- Find Reversals ---
    good_idx = event_indices_from_cumulative(session_data.get("good_reversals"), N=N)
    bad_idx  = event_indices_from_cumulative(session_data.get("bad_reversals"), N=N)
    def add_reversal_lines(ax):
        first_good = True
        for i in good_idx:
            ax.axvline(i - 0.5, linestyle="--", color="#3A982E", linewidth=2.5, label="Good Rev" if first_good else None)
            first_good = False
        first_bad = True
        for i in bad_idx:
            ax.axvline(i - 0.5, linestyle="--", color="#F97979", linewidth=2.5, label="Bad Rev" if first_bad else None)
            first_bad = False

    # --- Session Summary ---
    for tower, color in zip(towers, TOWER_COLORS):
        y = np.asarray(mags_by[tower], dtype=float)
        ax1.plot(x, y, linewidth=2.5, color=color, label=f"{tower}")
    chosen_y = np.full(N, np.nan)
    for i in range(N):
        for tower in towers:
            if i < len(choices_by[tower]) and choices_by[tower][i] == 1:
                chosen_y[i] = mags_by[tower][i]
                break
    ax1.scatter(x, chosen_y, s=30, color="black", marker="o", label="Choice", zorder=5)
    add_reversal_lines(ax1)
    ax1.set_ylabel("Reward Magnitude", fontsize=12)
    ax1.set_yticks([0, 1, 4])
    ax1.set_title(title or "Reward Magnitudes & Choices", fontsize=14)
    ax1.legend(loc="upper left", fontsize=10)

    # --- EMA ---
    for k in ema_keys:
        if k in session_data:
            ax2.plot(x, np.asarray(session_data[k], dtype=float), linewidth=3, color=EMA_COLORS.get(k, None), label=EMA_LABELS.get(k, k),)
    ax2.axhline(0.7, linestyle="--", linewidth=2, color="#DCDCDC", alpha=0.7, label="Threshold")
    add_reversal_lines(ax2)
    ax2.set_ylabel("EMA", fontsize=12)
    ax2.set_xlabel("Trial", fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.set_yticks(np.linspace(0, 1, 6))
    ax2.legend(loc="upper left", fontsize=10)

    # --- Choices by Tower ---
    tower_counts = [int(np.nansum(np.asarray(choices_by[t][:N], dtype=float))) for t in towers]
    total_choices = max(int(np.sum(tower_counts)), 1)
    bars3 = ax3.bar(towers, tower_counts, color=TOWER_COLORS[:len(towers)])
    ax3.set_ylabel("Count by Tower", fontsize=12)
    for rect, c in zip(bars3, tower_counts):
        pct = 100.0 * c / total_choices
        ax3.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), f"{pct:.0f}%", fontsize=12, ha="center", va="bottom",)

    # --- Choices by Rank ---
    rank_counts = None
    if "rank_counts" in session_data and session_data["rank_counts"]:
        last = session_data["rank_counts"][-1]
        if isinstance(last, dict):
            rank_counts = last
    if rank_counts:
        preferred = ["best", "second", "third"]
        rank_keys = [k for k in preferred if k in rank_counts] + [k for k in rank_counts if k not in preferred]
        rank_vals = [int(rank_counts[k]) for k in rank_keys]
        total_rank = max(int(np.sum(rank_vals)), 1)
        rank_color_map = {
            "best": EMA_COLORS["ema_best_arm_choices"],
            "second": EMA_COLORS["ema_second_arm_choices"],
            "third": "#CCCCCC",
        }
        rank_colors = [rank_color_map.get(k, "#CCCCCC") for k in rank_keys]
        bars4 = ax4.bar(rank_keys, rank_vals, color=rank_colors)
        ax4.set_ylabel("Count by Rank", fontsize=12)
        for rect, c in zip(bars4, rank_vals):
            pct = 100.0 * c / total_rank
            ax4.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), f"{pct:.0f}%", ha="center", va="bottom",fontsize=12)
    else:
        ax4.text(0.5, 0.5, "No rank data", ha="center", va="center")
        ax4.set_xticks([])
    ax4.set_ylabel("Count by Rank", fontsize=12)

    for ax in (ax1, ax2, ax3, ax4):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
    plt.tight_layout()
    
    if save_path:
        base = Path(save_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(base) + ".pdf", bbox_inches="tight")
        fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)

# --- Reversal Identification Helper Function ---
def event_indices_from_cumulative(counter, N=None):
    """
    counter: list of cumulative counts per trial.
    Returns trial indices where the counter increments (event occurs at that trial index).
    """
    if counter is None:
        return []
    c = np.asarray(counter, dtype=float)
    if N is not None:
        c = c[:N]
    if len(c) < 2:
        return []
    d = np.diff(c)
    return list(np.where(d > 0)[0] + 1)
