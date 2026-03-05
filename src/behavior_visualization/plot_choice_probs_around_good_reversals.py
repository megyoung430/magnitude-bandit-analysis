import warnings
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from src.behavior_visualization.plot_style import CHOICE_PROB_COLOR_MAP as COLOR_MAP
from matplotlib.ticker import MaxNLocator



def plot_choice_probs_around_good_reversals(x, across, add_cumulative_axis=True, only_good=False, use_total=False, windows_for_cumulative_axis=None, 
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
            add_cumulative_rev_axis(x, across, windows_for_cumulative_axis, all_good_idx, all_bad_idx, save_path, only_good=only_good, use_total=use_total)
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

def add_cumulative_rev_axis(x, across, windows_for_cumulative_axis=None, all_good_idx=None, all_bad_idx=None, save_path=None, plot_fraction_removed=True, only_good=False, use_total=None):
    mean = across.get("mean", {})
    se = across.get("se", {})

    x = np.asarray(x, dtype=float)

    x_disp = x.copy()

    if use_total:
        _, cumulative_total, fraction_removed_total = cumulative_total_events_over_post(windows_for_cumulative_axis, x, across, exclude_anchor=True)
    else:
        if all_good_idx is None or all_bad_idx is None:
            raise ValueError("For mode='goodbad', provide all_good_idx and all_bad_idx.")
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

        ax2.set_zorder(0)
        ax.set_zorder(1)
        ax.patch.set_visible(False)

        if use_total:
            # TOTAL mode: one line
            if plot_fraction_removed:
                m = np.isfinite(x_disp) & np.isfinite(fraction_removed_total)
                ax2.step(x_disp[m], fraction_removed_total[m], where="post",
                        linewidth=2.0, color="#FF9100", alpha=0.7, label="Total Rev")
                ax2.set_ylabel("Fraction of Data Removed", rotation=-90, labelpad=15)
                ax2.set_ylim(0, 1.05)
            else:
                m = np.isfinite(x_disp) & np.isfinite(cumulative_total)
                ax2.step(x_disp[m], cumulative_total[m], where="post",
                        linewidth=2.0, color="#FF9100", alpha=0.7, label="Total Rev")
                ax2.set_ylabel("Cumulative Reversals", rotation=-90, labelpad=15)

                ymax = float(np.nanmax(cumulative_total[m])) if np.any(m) else 0.0
                ax2.set_ylim(0, ymax * 1.05 + 1)

            ax2.spines["top"].set_visible(False)
            return ax2

        # GOOD/BAD mode: two lines (your existing behavior)
        if plot_fraction_removed:
            m = np.isfinite(x_disp) & np.isfinite(fraction_removed_good) & np.isfinite(fraction_removed_bad)
            ax2.step(x_disp[m], fraction_removed_good[m], where="post", linewidth=2.0,
                    color="#FF9100", alpha=0.7, label="Good Rev")
            if not only_good:
                ax2.step(x_disp[m], fraction_removed_bad[m],  where="post", linewidth=2.0,
                        color="#CD0000", alpha=0.7, label="Bad Rev")
            ax2.set_ylabel("Fraction of Data Removed", rotation=-90, labelpad=15)
            ax2.set_ylim(0, 1.05)
        else:
            m = np.isfinite(x_disp) & np.isfinite(cumulative_good) & np.isfinite(cumulative_bad)
            ax2.step(x_disp[m], cumulative_good[m], where="post", linewidth=2.0,
                    color="#FF9100", alpha=0.7, label="Good Rev")
            if not only_good:
                ax2.step(x_disp[m], cumulative_bad[m],  where="post", linewidth=2.0,
                        color="#CD0000", alpha=0.7, label="Bad Rev")
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
    For each aligned good reversal at index g, find the FIRST subsequent reversal after g
    (good or bad). Then mark that anchor as "removed" for all offsets t >= (next_rev - g).

    This makes fraction_removed_good + fraction_removed_bad <= 1 by construction.
    """
    x = np.asarray(x, dtype=float)
    x_disp = x.copy()

    post_mask = np.isfinite(x) & (x >= 0)
    x_post = x[post_mask]
    if x_post.size == 0:
        nan = np.full_like(x, np.nan, dtype=float)
        return x_disp, nan, nan, nan, nan

    t_int = np.rint(x_post).astype(int)
    t_sorted = np.sort(np.unique(t_int))

    removed_good_sorted = np.zeros_like(t_sorted, dtype=float)
    removed_bad_sorted  = np.zeros_like(t_sorted, dtype=float)

    num_anchors = 0

    for subj, revs in good_windows.items():
        goods = sorted(set(all_good_idx.get(subj, [])))
        bads  = sorted(set(all_bad_idx.get(subj, [])))
        all_revs = sorted(set(goods) | set(bads))

        goods_set = set(goods)
        bads_set = set(bads)

        for r in revs:
            g = r.get("reversal_idx", None)
            if g is None:
                continue

            num_anchors += 1

            if exclude_anchor_good:
                candidates = [ii for ii in all_revs if ii > g]
            else:
                candidates = [ii for ii in all_revs if ii >= g]

            if not candidates:
                continue

            next_rev = candidates[0]
            delta = next_rev - g 

            is_good_next = next_rev in goods_set
            is_bad_next  = next_rev in bads_set

            start_idx = np.searchsorted(t_sorted, delta, side="left")
            if start_idx >= t_sorted.size:
                continue

            if is_good_next:
                removed_good_sorted[start_idx:] += 1.0
            elif is_bad_next:
                removed_bad_sorted[start_idx:] += 1.0
            else:
                removed_bad_sorted[start_idx:] += 1.0

    step_idx = np.searchsorted(t_sorted, t_int, side="right") - 1
    step_idx = np.clip(step_idx, 0, t_sorted.size - 1)

    cumulative_good = removed_good_sorted[step_idx]
    cumulative_bad  = removed_bad_sorted[step_idx]

    cumulative_good_full = np.full_like(x, np.nan, dtype=float)
    cumulative_bad_full  = np.full_like(x, np.nan, dtype=float)
    cumulative_good_full[post_mask] = cumulative_good
    cumulative_bad_full[post_mask]  = cumulative_bad

    denom = max(int(num_anchors), 1)  
    fraction_removed_good = cumulative_good_full / denom
    fraction_removed_bad  = cumulative_bad_full  / denom

    assert np.nanmax(fraction_removed_good + fraction_removed_bad) <= 1.0 + 1e-6, \
        "Fraction removed exceeds 1.0"
    return x_disp, cumulative_good_full, cumulative_bad_full, fraction_removed_good, fraction_removed_bad

def cumulative_total_events_over_post(windows, x, across, exclude_anchor=True):
    """
    Fraction removed interpretation:
    For each aligned anchor reversal at index g, find the FIRST subsequent reversal after g.
    That anchor's aligned window is considered "removed" for all offsets t >= (next_rev - g).

    Returns:
      x_disp, cum_total_full (# anchors removed by each offset), frac_total_full (in [0,1])
    """
    x = np.asarray(x, dtype=float)
    x_disp = x.copy()

    post_mask = np.isfinite(x) & (x >= 0)
    x_post = x[post_mask]
    if x_post.size == 0:
        nan = np.full_like(x, np.nan, dtype=float)
        return x_disp, nan, nan

    t_int = np.rint(x_post).astype(int)
    t_sorted = np.sort(np.unique(t_int))

    removed_total_sorted = np.zeros_like(t_sorted, dtype=float)
    num_anchors = 0

    for subj, revs in windows.items():
        rev_idxs = sorted(
            set(r.get("reversal_idx") for r in revs if r.get("reversal_idx") is not None)
        )
        if not rev_idxs:
            continue

        for r in revs:
            g = r.get("reversal_idx", None)
            if g is None:
                continue

            num_anchors += 1

            if exclude_anchor:
                candidates = [ii for ii in rev_idxs if ii > g]
            else:
                candidates = [ii for ii in rev_idxs if ii >= g]

            if not candidates:
                continue

            next_rev = candidates[0]
            delta = next_rev - g

            start_idx = np.searchsorted(t_sorted, delta, side="left")
            if start_idx < t_sorted.size:
                removed_total_sorted[start_idx:] += 1.0

    step_idx = np.searchsorted(t_sorted, t_int, side="right") - 1
    step_idx = np.clip(step_idx, 0, t_sorted.size - 1)

    cum_total = removed_total_sorted[step_idx]

    cum_total_full = np.full_like(x, np.nan, dtype=float)
    cum_total_full[post_mask] = cum_total

    denom = max(int(num_anchors), 1) 
    frac_total = cum_total_full / denom

    assert np.nanmax(frac_total[post_mask]) <= 1.0 + 1e-6, "Fraction removed exceeds 1.0"
    return x_disp, cum_total_full, frac_total