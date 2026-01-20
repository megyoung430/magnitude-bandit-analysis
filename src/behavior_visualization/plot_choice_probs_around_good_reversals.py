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

def plot_reversal_probs_around_good_reversals(x, across, show_chance=True, skip_n_trials_after_reversal=0, save_path=None):
    COLOR_MAP = {
        "prev_best": "#5DA5DA",
        "next_best": "#60BD68",
        "third":     "#7f7f7f",
    }

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
        if show_chance:
            ax.axhline(1 / 3, color="gray", linestyle=":", linewidth=1)

        ax.set_ylim(-0.05, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if skip_n_trials_after_reversal <= 0:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        _plot_all(ax, (np.nanmin(x_disp), np.nanmax(x_disp)))
        ax.set_xlabel("Trials from Good Reversal")
        ax.set_ylabel("Choice Probability")
        ax.legend(loc="upper right", fontsize=10)
        plt.tight_layout()

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
            axR.legend(handles, labels, loc="upper right", fontsize=10, frameon=False)

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