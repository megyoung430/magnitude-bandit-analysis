"""Choice probabilities around block changes per training stage (the analogue of Samborska Fig. 1f).

Drawing only: the numbers come from :mod:`src.behavior_analysis.reversals.reversal_curves_by_stage`.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.behavior_visualization.common.plot_style  # noqa: F401  # applies rcParams
from src.behavior_analysis.reversals import reversal_curves_by_stage as rcs
from src.behavior_visualization.common.plot_helpers import (
    Title,
    despine,
    save_or_show,
    set_suptitle,
    set_title,
    stats_line,
)
from src.behavior_visualization.common.training_stage_style import (
    STAGE_COLORS,
    STAGE_TEXT,
    STAGES,
    stage_label,
    test_lines,
)

REVERSAL_CURVES_BY_STAGE = "Choice Probabilities Around Block Changes by Training Stage"
ROLE_HEADINGS = {
    rcs.NEW_BEST: "P(New Best)",
    rcs.PREVIOUS_BEST: "P(Previous Best)",
    rcs.THIRD: "P(Third)",
}


def _n_mice(table: pd.DataFrame) -> int:
    """The number of mice in *table*."""
    return table["mouse"].nunique()


def _draw_role(ax: plt.Axes, curves: pd.DataFrame, role: str) -> int:
    """Mean ± se across mice of one role, a line per stage; returns the number of mice."""
    rows = curves[curves["role"] == role]
    for stage in STAGES:
        mine = rows[rows["stage"] == stage]
        if mine.empty:
            continue
        wide = mine.pivot(index="mouse", columns="offset", values="probability")
        stats = [rcs.nan_mean_se(wide[c].to_numpy(dtype=float)) for c in wide.columns]
        mean, se = (np.array(v) for v in zip(*stats, strict=True))
        x = wide.columns.to_numpy(dtype=float)
        color = STAGE_COLORS[stage]
        ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.2, lw=0)
        ax.plot(x, mean, color=color, lw=2, label=f"{stage_label(stage)} (n={len(wide)})")
    ax.axvline(-0.5, color="0.4", ls="--", lw=1)
    ax.set_xlabel("Trials From Block Change", fontsize=12)
    ax.set_ylim(0, 1)
    return _n_mice(rows)


def _draw_readout(ax: plt.Axes, readouts: pd.DataFrame, role: str) -> int:
    """Each mouse's readout per stage (gray lines) and the mean ± se across mice."""
    rows = readouts[readouts["role"] == role]
    wide = rows.pivot(index="mouse", columns="stage", values="probability").reindex(
        columns=list(STAGES)
    )
    x = np.arange(len(STAGES))
    for _, values in wide.iterrows():
        ax.plot(x, values.to_numpy(dtype=float), "-", color="0.6", lw=0.8, alpha=0.7)
    for i, stage in enumerate(STAGES):
        mean, se = rcs.nan_mean_se(wide[stage].to_numpy(dtype=float))
        ax.errorbar(i, mean, yerr=se, fmt="o", color=STAGE_COLORS[stage], ms=7, capsize=3,
                    lw=2, zorder=5)  # fmt: skip
    ax.set_xticks(x, ["\n".join(stage_label(s).rsplit(" ", 1)) for s in STAGES], fontsize=9)
    ax.set_xlim(-0.5, len(STAGES) - 0.5)
    return len(wide)


def plot_reversal_curves_by_stage(
    curves: pd.DataFrame,
    readouts: pd.DataFrame,
    tests: pd.DataFrame,
    save_path: str | Path | None = None,
    *,
    start: int = rcs.ADAPT_START,
    stop: int = rcs.ADAPT_STOP,
) -> None:
    """P(new best), P(previous best) and P(third) around block changes, a line per stage.

    The fourth panel shows each mouse's mean P(new best) over trials *start* to *stop* after
    the change.

    Args:
        curves (pd.DataFrame): :func:`reversal_curves_by_stage.stage_curves`.
        readouts (pd.DataFrame): :func:`reversal_curves_by_stage.adaptation_readouts`.
        tests (pd.DataFrame): ``stage_tests`` of the readouts by ``role``, for the caption.
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.
        start (int): First trial of the readout window, for the labels. Default
            ``reversal_curves_by_stage.ADAPT_START``.
        stop (int): Last trial of the readout window. Default
            ``reversal_curves_by_stage.ADAPT_STOP``.

    Raises:
        ValueError: If *curves* is empty.
    """
    if curves.empty:
        raise ValueError("no curves to plot")
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.4), gridspec_kw=dict(width_ratios=[1, 1, 1, 0.7]))
    try:
        for ax, role in zip(axes[:3], rcs.ROLES, strict=True):
            n = _draw_role(ax, curves, role)
            set_title(ax, Title(ROLE_HEADINGS[role], stats_line("mean ± se", "mice",
                                                                  [(n, "mouse", "mice")])),
                      fontsize=12)  # fmt: skip
            ax.set_ylabel("Choice Probability", fontsize=12)
            despine(ax)
        n = _draw_readout(axes[3], readouts, rcs.NEW_BEST)
        heading = f"P(New Best), Trials {start}-{stop}"
        set_title(axes[3], Title(heading, stats_line("mean ± se", "mice", [(n, "mouse", "mice")])),
                  fontsize=12)  # fmt: skip
        axes[3].set_ylabel("Choice Probability", fontsize=12)
        despine(axes[3])
        axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.18), frameon=False, fontsize=10,
                       ncol=len(STAGES))  # fmt: skip
        caption = (
            "Choice probabilities around the good block changes of each problem whose best arm "
            "changed (trial 0: the first trial of the new block); trials after a bad block change "
            "are removed. Per mouse, the mean over a problem's block changes, then over the "
            "problems of a stage; mean ± se across mice. The n in the legend counts the mice "
            "with that stage.",
            STAGE_TEXT,
            f"Readout: each mouse's mean P(new best) and P(previous best) over trials {start} to "
            f"{stop} after the change (gray lines: single mice in the right panel).",
            *test_lines(tests, lambda row: f"{ROLE_HEADINGS[row.role]}, trials {start}-{stop}"),
        )
        fig.subplots_adjust(left=0.05, right=0.98, bottom=0.27, top=0.8, wspace=0.3)
        set_suptitle(fig, Title(REVERSAL_CURVES_BY_STAGE, None, caption), fontsize=14)
    except BaseException:
        plt.close(fig)
        raise
    save_or_show(fig, save_path)
