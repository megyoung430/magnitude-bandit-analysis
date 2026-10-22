"""One row per drifting-task trial, built from the importer's session dicts.

The choice regression, the task-rule checks and the agent fits all start from this table, so
the details of the session dicts (merged recordings, session order, missing trials) are handled
once, here. Everything is pure: session dicts in, a DataFrame out.

Columns of the table:

- ``mouse`` (str), ``problem`` (hashable key of the problem, e.g. ``(1, 3)`` for cohort 1's
  drifting problem 3), ``session`` (session key), ``session_index`` (0, 1, ... in session order
  within the problem);
- ``trial`` (0, 1, ... within the problem, across sessions) and ``trial_in_session``;
- ``choice`` (chosen port), ``magnitudes`` (``{port: magnitude}`` offered before the choice),
  ``reward`` (magnitude of the chosen port), ``initiation`` (initiation port of the session);
- ``block`` (the session's block counter, ``None`` when not logged).

Trials without a choice or without magnitudes are dropped. They break the trial sequence, so
the next kept trial is flagged with ``after_gap=True``; analyses that use the previous trial
(stickiness, last-seen values) treat a gap like a session start.
"""

import math

import pandas as pd

from src.behavior_import.problem_numbering import DEFAULT_INITIATION_TOWER
from src.behavior_import.session_order import ordered_session_keys

COLUMNS = [
    "mouse",
    "problem",
    "session",
    "session_index",
    "trial",
    "trial_in_session",
    "choice",
    "magnitudes",
    "reward",
    "initiation",
    "block",
    "after_gap",
]


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _session_rows(mouse, problem, key, session_index, session) -> list[dict]:
    """Rows of one session; ``trial`` is filled in by the caller."""
    choices = session.get("choice") or []
    magnitudes = session.get("reward_magnitudes") or []
    blocks = session.get("blocks") or [None] * len(choices)
    if len(magnitudes) != len(choices) or len(blocks) != len(choices):
        raise ValueError(
            f"mouse {mouse!r}, problem {problem!r}, session {key!r}: {len(choices)} choices, "
            f"{len(magnitudes)} magnitude entries and {len(blocks)} block entries; the trial "
            "lists must align"
        )
    initiation = session.get("initiation_tower") or DEFAULT_INITIATION_TOWER
    rows, gap = [], False
    for t, (choice, mags, block) in enumerate(zip(choices, magnitudes, blocks, strict=True)):
        if _is_missing(choice) or not isinstance(mags, dict) or not mags:
            gap = True
            continue
        if choice not in mags:
            raise ValueError(
                f"mouse {mouse!r}, problem {problem!r}, session {key!r}, trial {t}: choice "
                f"{choice!r} is not among the offered ports {sorted(mags)}"
            )
        rows.append(
            dict(
                mouse=mouse,
                problem=problem,
                session=key,
                session_index=session_index,
                trial_in_session=t,
                choice=choice,
                magnitudes=dict(mags),
                reward=mags[choice],
                initiation=initiation,
                block=block,
                after_gap=gap,
            )
        )
        gap = False
    return rows


def drifting_trial_table(problems: dict) -> pd.DataFrame:
    """One row per trial of the drifting *problems*, in trial order within each mouse's problem.

    Args:
        problems (dict): ``{problem key: {mouse: {session_key: session}}}``, e.g. the
            ``problems`` of :func:`src.pipelines.all_cohorts_drifting.pool_cohorts`. A session
            holds the merged per-trial lists ``"choice"``, ``"reward_magnitudes"`` (one
            ``{port: magnitude}`` per trial) and optionally ``"blocks"``.

    Returns:
        pd.DataFrame: Columns ``COLUMNS`` (see the module docstring); empty with those columns
        if there are no trials.

    Raises:
        ValueError: If a session's trial lists differ in length, or a choice is not an offered
            port.
    """
    rows = []
    for problem, mice in problems.items():
        for mouse, sessions in mice.items():
            problem_rows = [
                row
                for index, key in enumerate(ordered_session_keys(mouse, sessions))
                for row in _session_rows(mouse, problem, key, index, sessions[key])
            ]
            for trial, row in enumerate(problem_rows):
                row["trial"] = trial
            rows.extend(problem_rows)
    return pd.DataFrame(rows, columns=COLUMNS)
