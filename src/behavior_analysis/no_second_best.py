"""A table of the blocks in which more than one arm is at magnitude 0.

With two or more arms at 0 only one arm (or none) pays, so there is no second best arm and
analyses that compare the best with the second best have nothing to compare. This lists those
blocks by hand-review.

Blocks are the merged, problem-long block numbers of
:func:`~src.behavior_analysis.get_variables_across_sessions.get_vars_across_all_sessions`
(a block that runs over a session boundary is one block), per mouse and problem. Columns:

- ``mouse``, ``task_type``, ``problem_in_type``, ``overall_problem``;
- ``sessions``: the sessions the block spans, separated by ``;``;
- ``block``: the block number within the problem, as in the merged data;
- ``first_trial``, ``last_trial``, ``n_trials``: the block's extent, trials counted from 1
  over the problem; ``first_trial_in_session`` / ``last_trial_in_session``: the same trials
  counted within their own session (the first trial in the first session, the last in the last);
- ``n_trials_multiple_zero``: how many of the block's trials have more than one arm at 0 (the
  magnitudes of a drifting block can change within it, a blocked block's cannot);
- ``rewarded_arm`` / ``rewarded_magnitude``: the arm with a magnitude above 0 on the block's
  first such trial and its magnitude (several arms separated by ``;``; blank if none pays);
- ``magnitudes``: all arms on that trial, such as ``A1=4, B1=0, C1=0``.

Trials with a missing magnitude are not counted.
"""

from itertools import groupby

import pandas as pd

from src.behavior_analysis.get_variables_across_sessions import get_vars_across_all_sessions
from src.behavior_analysis.review_context import (
    ProblemIndex,
    magnitudes_text,
    typed_frame,
)
from src.behavior_import.session_order import ordered_session_keys

_DTYPES = {
    "mouse": "object",
    "task_type": "object",
    "problem_in_type": "Int64",
    "overall_problem": "Int64",
    "sessions": "object",
    "block": "Int64",
    "first_trial": "Int64",
    "last_trial": "Int64",
    "n_trials": "Int64",
    "first_trial_in_session": "Int64",
    "last_trial_in_session": "Int64",
    "n_trials_multiple_zero": "Int64",
    "rewarded_arm": "object",
    "rewarded_magnitude": "object",
    "magnitudes": "object",
}
COLUMNS = tuple(_DTYPES)
ZERO_ARMS_FOR_NO_SECOND_BEST = 2  # this many arms at 0 leave no second best


def _has_multiple_zero_arms(magnitudes: dict) -> bool:
    """True if at least two arms are at 0 and no magnitude is missing."""
    if any(v is None for v in magnitudes.values()):
        return False
    return sum(v == 0 for v in magnitudes.values()) >= ZERO_ARMS_FOR_NO_SECOND_BEST


def _magnitudes_at(by_tower: dict, trial: int) -> dict:
    return {tower: values[trial] for tower, values in by_tower.items()}


def _session_positions(unmerged: dict) -> list[tuple[str, int]]:
    """``(session key, trial index within the session)`` for every merged trial, in order."""
    return [
        (key, i)
        for key, trials in zip(unmerged["session_keys"], unmerged["trial"], strict=True)
        for i in range(len(trials))
    ]


def _block_row(block: int, trials: list[int], by_tower: dict, positions: list) -> dict | None:
    """The table row of the block with merged trial indices *trials*, or ``None``."""
    flagged = [t for t in trials if _has_multiple_zero_arms(_magnitudes_at(by_tower, t))]
    if not flagged:
        return None
    magnitudes = _magnitudes_at(by_tower, flagged[0])
    rewarded = {tower: m for tower, m in magnitudes.items() if m > 0}
    first_in, last_in = positions[trials[0]][1], positions[trials[-1]][1]
    sessions = list(dict.fromkeys(positions[t][0] for t in trials))
    return {
        "sessions": ";".join(sessions),
        "block": block,
        "first_trial": trials[0] + 1,
        "last_trial": trials[-1] + 1,
        "n_trials": len(trials),
        "first_trial_in_session": first_in + 1,
        "last_trial_in_session": last_in + 1,
        "n_trials_multiple_zero": len(flagged),
        "rewarded_arm": ";".join(rewarded) or None,
        "rewarded_magnitude": ";".join(str(m) for m in rewarded.values()) or None,
        "magnitudes": magnitudes_text(magnitudes),
    }


def _mouse_problem_rows(mouse: str, sessions: dict) -> list[dict]:
    merged, unmerged = get_vars_across_all_sessions({mouse: sessions})
    blocks = merged[mouse]["blocks"]
    by_tower = merged[mouse]["reward_magnitudes_by_tower"]
    positions = _session_positions(unmerged[mouse])
    runs = (
        (block, list(trials)) for block, trials in groupby(range(len(blocks)), blocks.__getitem__)
    )
    rows = (
        {"mouse": mouse, **row}
        for block, trials in runs
        if (row := _block_row(block, trials, by_tower, positions))
    )
    return list(rows)


def no_second_best_table(grouped: dict, index: ProblemIndex) -> pd.DataFrame:
    """One row per block with more than one arm at 0.

    Args:
        grouped (dict): ``{(task_type, k): {mouse: {session_key: session_dict}}}`` from
            :func:`~src.behavior_import.task_types.regroup_by_task_type`; the sessions carry the
            merged-ready trial variables (``trial``, ``blocks``, ``reward_magnitudes_by_tower``).
        index (ProblemIndex): Gives each problem's overall number.

    Returns:
        pd.DataFrame: The columns of :data:`COLUMNS`, ordered by task type, problem, mouse and
        block; empty (with the columns) if no block qualifies.
    """
    rows = []
    for mice in grouped.values():
        for mouse in sorted(mice):
            first_session = ordered_session_keys(mouse, mice[mouse])[0]
            columns = index.columns(mouse, first_session)
            rows += [{**columns, **row} for row in _mouse_problem_rows(mouse, mice[mouse])]
    return typed_frame(rows, _DTYPES)
