import warnings
import numpy as np
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial"]
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["axes.titlesize"] = 14

COLOR_MAP = {
    "prev_best": "#5DA5DA",
    "next_best": "#60BD68",
    "third":     "#7f7f7f",
}

def plot_choice_probs_around_good_reversals(x, across, add_cumulative_axis=True, windows_for_cumulative_axis=None, 
                                            all_good_idx=None, all_bad_idx=None, skip_n_trials_after_reversal=0, save_path=None):
    mean = across.get("mean", {})
    se = across.get("se", {})

    x = np.asarray(x, dtype=float)

    x_disp = x.copy()
    if skip_n_trials_after_reversal > 0:
        x_disp[x_disp >= 1] += skip_n_trials_after_reversal

    def _plot_all(ax, xlim):
        ax.set_xlim(*xlim)

        for key, label in [("prev_best", "Previous Best"),("next_best", "New Best"),("third", "Third Arm")]:
            y = mean.get(key)
            s = se.get(key)
            if y is None or s is None:
                continue

            y = np.asarray(y, dtype=float)
            s = np.asarray(s, dtype=float)

            m = np.isfinite(x_disp) & np.isfinite(y) & np.isfinite(s)
            m &= (x_disp >= xlim[0]) & (x_disp <= xlim[1])
            if not np.any(m):
                continue

            ax.plot(x_disp[m], y[m], linewidth=2, color=COLOR_MAP[key], label=label)
            ax.fill_between(x_disp, y - s, y + s, where=m, alpha=0.2, color=COLOR_MAP[key])

        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.axhline(1 / 3, color="gray", linestyle=":", linewidth=1)

        ax.set_ylim(-0.05, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if skip_n_trials_after_reversal <= 0:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        _plot_all(ax, (np.nanmin(x_disp), np.nanmax(x_disp)))
        ax.set_xlabel("Trials from Good Reversal")
        ax.set_ylabel("Choice Probability")
        title = "Good Reversal-Aligned Choices\n" \
            f"(mean ± se across subjects | " \
            f"n={across.get('num_subjects', 0)} subjects, " \
            f"n={across.get('num_reversals', 0)} reversals)\n"
        ax.set_title(title, pad=20)
        ax.legend(loc="upper right", fontsize=10)
        plt.tight_layout()
    
        if add_cumulative_axis:
            add_cumulative_rev_axis(x, across, windows_for_cumulative_axis, all_good_idx, all_bad_idx, save_path)
            return
    
    else:
        left_xlim = (np.nanmin(x_disp), 0)
        right_start = skip_n_trials_after_reversal + 1
        right_xlim = (right_start, np.nanmax(x_disp))

        left_span = left_xlim[1] - left_xlim[0]
        right_span = right_xlim[1] - right_xlim[0]

        fig, (axL, axR) = plt.subplots(1,2,sharey=True,figsize=(10, 4.5),gridspec_kw={"width_ratios": [left_span, right_span],"wspace": 0.15,})

        _plot_all(axL, left_xlim)
        _plot_all(axR, right_xlim)

        axL.spines["right"].set_visible(False)
        axR.spines["left"].set_visible(False)
        axR.spines["right"].set_visible(False)
        axR.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)

        axR.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        rticks = [t for t in axR.get_xticks() if t >= right_start - 1e-9]
        if right_start not in rticks:
            rticks = sorted(set(rticks + [right_start]))
        axR.set_xticks(rticks)

        axL.set_ylabel("Choice Probability")
        fig.supxlabel("Trials from Good Reversal")

        fig.suptitle(
            "Good Reversal-Aligned Choices\n"
            f"(mean ± se across subjects | "
            f"n={across.get('num_subjects', 0)} subjects, "
            f"n={across.get('num_reversals', 0)} reversals)\n"
            f"(Skipping first {skip_n_trials_after_reversal} trials post-reversal)",
            y=1.03,
        )

        handles, labels = axL.get_legend_handles_labels()
        if handles:
            axR.legend(handles, labels, loc="upper right", fontsize=10)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plt.tight_layout()
        
    if save_path:
        base = Path(save_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(base) + ".pdf", bbox_inches="tight")
        fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)

def add_cumulative_rev_axis(x, across, windows_for_cumulative_axis=None, all_good_idx=None, all_bad_idx=None, save_path=None, plot_fraction_removed=True):
    mean = across.get("mean", {})
    se = across.get("se", {})

    x = np.asarray(x, dtype=float)

    x_disp = x.copy()

    cumulative_good = cumulative_bad = None
    if windows_for_cumulative_axis is None or all_good_idx is None or all_bad_idx is None:
        raise ValueError("To show_reversal_cum_axis, provide windows_for_cumulative_axis, all_good_idx, all_bad_idx.")
    _, cumulative_good, cumulative_bad, fraction_removed_good, fraction_removed_bad = cumulative_reversal_events_over_post(windows_for_cumulative_axis, all_good_idx, all_bad_idx, x, across, exclude_anchor_good=True)

    def _plot_all(ax, xlim):
        ax.set_xlim(*xlim)

        for key, label in [("prev_best", "Previous Best"), ("next_best", "New Best"), ("third", "Third Arm")]:
            y = mean.get(key)
            s = se.get(key)
            if y is None or s is None:
                continue

            y = np.asarray(y, dtype=float)
            s = np.asarray(s, dtype=float)

            m = np.isfinite(x_disp) & np.isfinite(y) & np.isfinite(s)
            m &= (x_disp >= xlim[0]) & (x_disp <= xlim[1])
            if not np.any(m):
                continue

            ax.plot(x_disp[m], y[m], linewidth=2, color=COLOR_MAP[key], label=label)
            ax.fill_between(x_disp, y - s, y + s, where=m, alpha=0.2, color=COLOR_MAP[key])

        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.axhline(1 / 3, color="gray", linestyle=":", linewidth=1)

        ax.set_ylim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def _add_cumulative_axis_single(ax):
        ax2 = ax.twinx()

        if plot_fraction_removed:
            m = np.isfinite(x_disp) & np.isfinite(fraction_removed_good) & np.isfinite(fraction_removed_bad)

            ax2.set_zorder(0)
            ax.set_zorder(1)
            ax.patch.set_visible(False)

            ax2.step(x_disp[m], fraction_removed_good[m], where="post", linewidth=2.0, color="#FF9100", alpha=0.7, label="Good Rev")
            ax2.step(x_disp[m], fraction_removed_bad[m],  where="post", linewidth=2.0, color="#CD0000", alpha=0.7, label="Bad Rev")

            ax2.set_ylabel("Fraction of Data Removed", rotation=-90, labelpad=15)
            ax2.set_ylim(0, 1.05)
            ax2.spines["top"].set_visible(False)
        else:
            m = np.isfinite(x_disp) & np.isfinite(cumulative_good) & np.isfinite(cumulative_bad)

            ax2.set_zorder(0)
            ax.set_zorder(1)
            ax.patch.set_visible(False)

            ax2.step(x_disp[m], cumulative_good[m], where="post", linewidth=2.0, color="#FF9100", alpha=0.7, label="Good Rev")
            ax2.step(x_disp[m], cumulative_bad[m],  where="post", linewidth=2.0, color="#CD0000", alpha=0.7, label="Bad Rev")

            ax2.set_ylabel("Cumulative Reversals", rotation=-90, labelpad=15)

            ymax = 0.0
            if np.any(m):
                ymax = max(float(np.nanmax(cumulative_good[m])), float(np.nanmax(cumulative_bad[m])))
            ax2.set_ylim(0, ymax * 1.05 + 1)
            ax2.spines["top"].set_visible(False)

        return ax2

    fig, ax = plt.subplots(figsize=(10, 6))

    _plot_all(ax, (np.nanmin(x_disp), np.nanmax(x_disp)))
    ax2 = _add_cumulative_axis_single(ax)

    ax.set_xlabel("Trials from Good Reversal")
    ax.set_ylabel("Choice Probability")

    title = (
        "Good Reversal-Aligned Choices\n"
        f"(mean ± se across subjects | "
        f"n={across.get('num_subjects', 0)} subjects, "
        f"n={across.get('num_reversals', 0)} reversals)\n"
    )
    ax.set_title(title, pad=20)

    h1, l1 = ax.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=10)
    else:
        ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()

    if save_path:
        base = Path(save_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(base) + ".pdf", bbox_inches="tight")
        fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)

def cumulative_reversal_events_over_post(good_windows, all_good_idx, all_bad_idx, x, across, exclude_anchor_good=True):
    """
    For each aligned good reversal at raw index g, count whether raw reversals occur at g+t
    for each post offset t (t>=0), summed across all aligned good reversals/subjects.

    Returns:
      x_disp, cum_good_full, cum_bad_full (NaN for pre indices)
    """
    x = np.asarray(x, dtype=float)

    x_disp = x.copy()

    post_mask = np.isfinite(x) & (x >= 0)
    x_post = x[post_mask]
    if x_post.size == 0:
        return x_disp, np.full_like(x, np.nan), np.full_like(x, np.nan)

    t_int = np.rint(x_post).astype(int)

    good_events = np.zeros_like(t_int, dtype=float)
    bad_events  = np.zeros_like(t_int, dtype=float)

    for subj, revs in good_windows.items():
        goods = set(all_good_idx.get(subj, []))
        bads  = set(all_bad_idx.get(subj, []))

        for r in revs:
            g = r.get("reversal_idx", None)
            if g is None:
                continue

            idxs = g + t_int

            if exclude_anchor_good:
                good_hits = [(ii in goods) and (ii != g) for ii in idxs]
            else:
                good_hits = [ii in goods for ii in idxs]

            bad_hits = [ii in bads for ii in idxs]

            good_events += np.asarray(good_hits, dtype=float)
            bad_events  += np.asarray(bad_hits, dtype=float)

    order = np.argsort(t_int)
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)

    cumulative_good_sorted = np.cumsum(good_events[order])
    cumulative_bad_sorted  = np.cumsum(bad_events[order])

    cumulative_good = cumulative_good_sorted[inv]
    cumulative_bad  = cumulative_bad_sorted[inv]

    cumulative_good_full = np.full_like(x, np.nan, dtype=float)
    cumulative_bad_full  = np.full_like(x, np.nan, dtype=float)
    cumulative_good_full[post_mask] = cumulative_good
    cumulative_bad_full[post_mask]  = cumulative_bad

    fraction_removed_good = cumulative_good_full/across.get("num_reversals", 1)
    fraction_removed_bad = cumulative_bad_full/across.get("num_reversals", 1)

    return x_disp, cumulative_good_full, cumulative_bad_full, fraction_removed_good, fraction_removed_bad