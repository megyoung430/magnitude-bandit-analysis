"""What the drifting task did, measured from the trials: block changes, drift and block lengths.

These numbers serve two purposes. They document the task as it ran (which arm the task sets
to zero at a block change, and what the arm that was at zero gets), which matters for reading
the block-change figures. And they calibrate the simulated task of
:mod:`src.agents.drifting_simulation`, so that agents face the same task as the mice.

All functions are pure and take the table of
:func:`src.behavior_analysis.drifting_trial_table.drifting_trial_table`.

A *block change* is a trial whose zero arm differs from the previous trial's in the same
mouse's problem, with the same three ports offered. The arm that drops to zero is ranked by how
often the mouse chose each non-zero arm since the previous block change (or since the problem
started): ``"most"``, ``"least"`` or ``"tie"``.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd

BLOCK_CHANGE_COLUMNS = [
    "mouse",
    "problem",
    "trial",
    "zeroed_arm",
    "previous_zero_arm",
    "zeroed_choice_rank",
    "inherits_value",
    "block_length",
    "complete_block",
]


class DriftCounts(NamedTuple):
    """Within-block value changes of the non-zero arms.

    Attributes:
        n_changes (int): Trial-to-trial transitions in which a non-zero arm changed value.
        n_opportunities (int): Non-zero arm transitions counted (arms x trial pairs).
    """

    n_changes: int
    n_opportunities: int

    @property
    def rate(self) -> float:
        """Probability that a non-zero arm changes value from one trial to the next."""
        return self.n_changes / self.n_opportunities if self.n_opportunities else float("nan")


def _zero_arm(magnitudes: dict, where: str) -> str:
    zeros = [arm for arm, value in magnitudes.items() if value == 0]
    if len(zeros) != 1:
        raise ValueError(f"{where}: expected exactly one arm at 0, got {magnitudes}")
    return zeros[0]


def _choice_rank(arm: str, counts: dict, candidates: list) -> str:
    """``"most"``, ``"least"`` or ``"tie"``: *arm*'s choice count among the two *candidates*."""
    other = next(a for a in candidates if a != arm)
    if counts.get(arm, 0) == counts.get(other, 0):
        return "tie"
    return "most" if counts.get(arm, 0) > counts.get(other, 0) else "least"


def _problem_block_changes(mouse, problem, trials: pd.DataFrame) -> list[dict]:
    """Block changes of one mouse's problem; *trials* are its rows in trial order."""
    rows, counts, block_start, first_block = [], {}, 0, True
    previous = None
    columns = (trials["trial"], trials["magnitudes"], trials["choice"])
    for trial, magnitudes, choice in zip(*columns, strict=True):
        where = f"mouse {mouse!r}, problem {problem!r}, trial {trial}"
        if previous is not None and set(previous) == set(magnitudes):
            old_zero, new_zero = _zero_arm(previous, where), _zero_arm(magnitudes, where)
            if old_zero != new_zero:
                candidates = [arm for arm in previous if arm != old_zero]
                rows.append(
                    dict(
                        mouse=mouse,
                        problem=problem,
                        trial=trial,
                        zeroed_arm=new_zero,
                        previous_zero_arm=old_zero,
                        zeroed_choice_rank=_choice_rank(new_zero, counts, candidates),
                        inherits_value=magnitudes[old_zero] == previous[new_zero],
                        block_length=trial - block_start,
                        complete_block=not first_block,
                    )
                )
                counts, block_start, first_block = {}, trial, False
        counts[choice] = counts.get(choice, 0) + 1
        previous = magnitudes
    return rows


def block_changes(trial_table: pd.DataFrame) -> pd.DataFrame:
    """Every block change, with which arm was zeroed and what the old zero arm received.

    Args:
        trial_table (pd.DataFrame): Table of ``drifting_trial_table``.

    Returns:
        pd.DataFrame: One row per block change with columns ``BLOCK_CHANGE_COLUMNS``:
        ``trial`` is the first trial of the new block; ``zeroed_choice_rank`` is ``"most"``,
        ``"least"`` or ``"tie"`` (choice counts of the two non-zero arms in the block that
        ended); ``inherits_value`` is True if the old zero arm now has the zeroed arm's former
        value; ``block_length`` is the number of trials in the block that ended, and
        ``complete_block`` is False for a problem's first block, whose start was not observed
        as a block change.

    Raises:
        ValueError: If a trial does not have exactly one arm at 0.
    """
    rows = [
        row
        for (mouse, problem), trials in trial_table.groupby(["mouse", "problem"], sort=False)
        for row in _problem_block_changes(mouse, problem, trials.sort_values("trial"))
    ]
    return pd.DataFrame(rows, columns=BLOCK_CHANGE_COLUMNS)


def drift_counts(trial_table: pd.DataFrame) -> DriftCounts:
    """How often a non-zero arm changes value between consecutive trials of one block.

    Only consecutive trials with the same zero arm and the same offered ports count, so block
    changes and port changes are not mistaken for drift.

    Args:
        trial_table (pd.DataFrame): Table of ``drifting_trial_table``.

    Returns:
        DriftCounts: Changes and opportunities, pooled over mice and problems.
    """
    changes = opportunities = 0
    for _, trials in trial_table.groupby(["mouse", "problem"], sort=False):
        magnitudes = list(trials.sort_values("trial")["magnitudes"])
        for previous, current in zip(magnitudes[:-1], magnitudes[1:], strict=True):
            if set(previous) != set(current):
                continue
            moving = [arm for arm, value in previous.items() if value != 0]
            if any(current[arm] == 0 for arm in moving):
                continue  # a block change
            opportunities += len(moving)
            changes += sum(previous[arm] != current[arm] for arm in moving)
    return DriftCounts(changes, opportunities)


def complete_block_lengths(changes: pd.DataFrame) -> np.ndarray:
    """Lengths, in trials, of the blocks whose start and end were both observed.

    Args:
        changes (pd.DataFrame): Table of :func:`block_changes`.

    Returns:
        np.ndarray: ``(n_blocks,)`` int array.
    """
    return changes.loc[changes["complete_block"], "block_length"].to_numpy(dtype=int)


def block_change_summary(changes: pd.DataFrame) -> pd.DataFrame:
    """Counts and fractions of the zeroed-arm ranks and of value inheritance, per mouse and all.

    Args:
        changes (pd.DataFrame): Table of :func:`block_changes`.

    Returns:
        pd.DataFrame: One row per mouse plus a final ``mouse="all"`` row, with columns
        ``mouse, n_block_changes, n_most, n_least, n_tie, frac_most, frac_inherits_value``.
        ``frac_most`` is ``n_most / (n_most + n_least)``: ties are left out.
    """

    def summarise(mouse, rows):
        ranks = rows["zeroed_choice_rank"].value_counts()
        most, least = int(ranks.get("most", 0)), int(ranks.get("least", 0))
        return dict(
            mouse=mouse,
            n_block_changes=len(rows),
            n_most=most,
            n_least=least,
            n_tie=int(ranks.get("tie", 0)),
            frac_most=most / (most + least) if most + least else np.nan,
            frac_inherits_value=float(rows["inherits_value"].mean()) if len(rows) else np.nan,
        )

    per_mouse = [summarise(mouse, rows) for mouse, rows in changes.groupby("mouse")]
    return pd.DataFrame([*per_mouse, summarise("all", changes)])
