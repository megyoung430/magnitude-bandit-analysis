"""Colors, labels and caption text of the training stages, shared by the stage figures.

The stages are those of :mod:`src.behavior_analysis.data.training_stages`. The colors are three
Okabe-Ito colors that stay apart in gray scale and for the common forms of color blindness.
"""

import numpy as np
import pandas as pd

from src.behavior_analysis.data.training_stages import EARLY, LATE, STAGE_LABELS, STAGES, SWITCH

STAGE_COLORS = {EARLY: "#E69F00", SWITCH: "#56B4E9", LATE: "#009E73"}
STAGE_TEXT = (
    "Training stages: the first two structured (blocked) problems, the first two drifting "
    "problems and the mouse's last two drifting problems (from the third on). Structured problems "
    "are the blocked problems before a mouse's first drifting problem; drifting problems after "
    "the cohort's cutoff are left out. Each mouse's value of a stage is the mean over its "
    "problems in the stage."
)


def stage_label(stage: str) -> str:
    """The legend label of a stage."""
    return STAGE_LABELS[stage]


def test_lines(tests: pd.DataFrame, describe) -> list[str]:
    """Caption lines of :func:`training_stages.stage_tests`.

    Args:
        tests (pd.DataFrame): :func:`~src.behavior_analysis.data.training_stages.stage_tests`.
        describe: ``row -> str``, what the row compares (its *by* columns), written first.

    Returns:
        list[str]: One line per test.
    """
    lines = []
    for row in tests.itertuples():
        pair = f"{STAGE_LABELS[row.second]} minus {STAGE_LABELS[row.first]}"
        if row.n_mice == 0:
            lines.append(f"{describe(row)}, {pair}: no mouse has both.")
            continue
        se = f" ± se {row.se:.3f}" if np.isfinite(row.se) else ""
        test = f", t={row.t:.2f}, p={row.p:.3g}" if np.isfinite(row.p) else ""
        lines.append(
            f"{describe(row)}, {pair}: {row.mean_difference:+.3f}{se} across mice "
            f"(n={row.n_mice} mice{test}; paired t-test)."
        )
    return lines


__all__ = ["STAGES", "STAGE_COLORS", "STAGE_TEXT", "stage_label", "test_lines"]
