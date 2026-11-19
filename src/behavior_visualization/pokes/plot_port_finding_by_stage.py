"""Port-category fractions per training stage (the analogue of Samborska Fig. 1h).

Drawing only: the numbers come from :mod:`src.behavior_analysis.pokes.port_finding_by_stage`.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.behavior_visualization.common.plot_style  # noqa: F401  # applies rcParams
from src.behavior_analysis.pokes.poke_counts import KINDS
from src.behavior_analysis.pokes.port_finding_by_stage import IN_SEQUENCE
from src.behavior_analysis.pokes.port_finding_summary import (
    CATEGORIES,
    N_FIRST_BLOCKS,
    N_FIRST_TRIALS,
    WINDOWS,
)
from src.behavior_analysis.reversals.reversal_curves_by_stage import nan_mean_se
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
from src.behavior_visualization.pokes.plot_port_finding_summary import (
    CATEGORY_LABELS,
    KIND_LABELS,
    window_labels,
)

PORT_FINDING_BY_STAGE = "Port Categories by Training Stage"


def _draw_category(ax: plt.Axes, rows: pd.DataFrame) -> int:
    """Mean ± se across mice per window, a line per stage; returns the number of mice."""
    x = np.arange(len(WINDOWS))
    for k, stage in enumerate(STAGES):
        mine = rows[rows["stage"] == stage]
        if mine.empty:
            continue
        wide = mine.pivot(index="mouse", columns="window", values="fraction").reindex(
            columns=list(WINDOWS)
        )
        stats = [nan_mean_se(wide[w].to_numpy(dtype=float)) for w in WINDOWS]
        mean, se = (np.array(v) for v in zip(*stats, strict=True))
        offset = (k - 1) * 0.06  # keep the error bars of the stages apart
        ax.errorbar(x + offset, mean, yerr=se, fmt="-o", color=STAGE_COLORS[stage], ms=5,
                    capsize=3, lw=1.8, label=stage_label(stage))  # fmt: skip
    labels = window_labels(N_FIRST_TRIALS, N_FIRST_BLOCKS)
    ax.set_xticks(x, [labels[w].replace(" After", "\nAfter") for w in WINDOWS], fontsize=9)
    ax.set_xlim(-0.4, len(WINDOWS) - 0.6)
    ax.set_ylim(0, 1)
    return rows["mouse"].nunique()


def plot_port_finding_by_stage(
    fractions: pd.DataFrame, tests: pd.DataFrame, save_path: str | Path | None = None
) -> None:
    """The fraction of pokes at each port category in the three windows, a line per stage.

    Args:
        fractions (pd.DataFrame): :func:`port_finding_by_stage.stage_fractions`.
        tests (pd.DataFrame): ``stage_tests`` of the in-sequence fraction by ``kind`` and
            ``window``, for the caption.
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.

    Raises:
        ValueError: If *fractions* is empty.
    """
    if fractions.empty:
        raise ValueError("no fractions to plot")
    fig, axes = plt.subplots(len(KINDS), len(CATEGORIES), figsize=(18, 8.5), sharey=True)
    try:
        for i, kind in enumerate(KINDS):
            for j, category in enumerate(CATEGORIES):
                ax = axes[i, j]
                rows = fractions[(fractions["kind"] == kind) & (fractions["category"] == category)]
                n = _draw_category(ax, rows)
                stats = stats_line("mean ± se", "mice", [(n, "mouse", "mice")])
                heading = CATEGORY_LABELS[category]
                if category == IN_SEQUENCE[kind]:
                    heading += " (In Sequence)"
                set_title(ax, Title(heading, stats), fontsize=11)
                if j == 0:
                    ax.set_ylabel(f"{KIND_LABELS[kind]}\nFraction of Pokes", fontsize=12)
                despine(ax)
        axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=10)
        windows = window_labels(N_FIRST_TRIALS, N_FIRST_BLOCKS)
        caption = (
            "The fraction of each mouse's pokes at the current problem's choice ports, the "
            "previous problem's choice ports (a port in both counts as current), the "
            "initiation port and other ports, in three windows: port finding, the first "
            f"{N_FIRST_TRIALS} task trials and the first {N_FIRST_BLOCKS} blocks of the problem. "
            "Long pokes after initiation: from initiation to the choice; pokes after choice: "
            "port visits until the next initiation (see the port-finding summary figures). A "
            "mouse's first problem has "
            "no previous ports, so the first structured problem adds nothing to the "
            "previous-port fractions.",
            "In sequence: the next step of the trial, a current choice port after initiation "
            "and the initiation port after the choice (the paper's out-of-sequence pokes are "
            "choice-port visits after a choice). Tests of the in-sequence fraction:",
            STAGE_TEXT,
            *test_lines(tests, lambda row: f"{KIND_LABELS[row.kind]}, "
                                           f"{CATEGORY_LABELS[IN_SEQUENCE[row.kind]].lower()}, "
                                           f"{windows[row.window]}"),
        )  # fmt: skip
        fig.tight_layout(rect=(0, 0, 0.88, 0.9))
        set_suptitle(fig, Title(PORT_FINDING_BY_STAGE, None, caption), fontsize=14)
    except BaseException:
        plt.close(fig)
        raise
    save_or_show(fig, save_path)
