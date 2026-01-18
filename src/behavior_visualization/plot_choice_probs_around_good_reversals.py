import numpy as np
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial"]
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["axes.titlesize"] = 14

def plot_reversal_probs_around_good_reversals(x, across, show_chance=True, save_path=None):
    """Plots across-subject mean with +/- 1 std fill."""
    COLOR_MAP = {
        "prev_best": "#5DA5DA",   
        "next_best": "#60BD68",
        "third":     "#7f7f7f",
    }
    mean = across["mean"]
    std = across["std"]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for label, plot_label in [("prev_best", "Previous Best"), ("next_best", "New Best"), ("third", "Third Arm"),]:
        y = np.asarray(mean.get(label, None), dtype=float)
        s = np.asarray(std.get(label, None), dtype=float)

        if y is None or s is None:
            continue

        m = np.isfinite(y) & np.isfinite(s)
        ax.plot(x, y, label=plot_label, linewidth=2, color=COLOR_MAP[label])
        ax.fill_between(x, y - s, y + s, where=m, alpha=0.2, color=COLOR_MAP[label])

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    if show_chance:
        ax.axhline(1/3, color="gray", linestyle=":", linewidth=1)

    ax.set_xlabel("Trials from Good Reversal", fontsize=12)
    ax.set_ylabel("Choice Probability", fontsize=12)
    ax.set_title(f"Good Reversal-Aligned Choices\n(mean ± std across subjects | n={across['num_subjects']} subjects and n={across['num_reversals']} reversals)")

    ax.legend(loc="upper right", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(x[0], x[-1])
    plt.tight_layout()

    if save_path:
        base = Path(save_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(base) + ".pdf", bbox_inches="tight")
        fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)