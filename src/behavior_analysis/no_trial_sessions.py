"""A table of the files, recordings and sessions the importer left out for lack of trials.

The importer skips data in several places and prints a ``[WARN]`` line for each; the
:class:`~src.behavior_import.import_log.ImportLog` it fills keeps one
:class:`~src.behavior_import.import_log.SkipRecord` per skip. This module turns them into one
row each:

- ``mouse``, ``session``: from the path (``session`` is the folder name, so a non-numbered
  folder such as ``ses-test_date-20260102`` appears as it is); blank where the path has none;
- ``file``: the ``.tsv`` file name, blank where a whole session has no file;
- ``skipped``: ``file`` (never became a recording: unreadable, no subject, non-numbered session
  folder), ``recording`` (one recording without trials in a session that has others) or
  ``session`` (the whole session is left out of every analysis);
- ``kind`` and ``reason``: the cause as a code and in plain words;
- ``task_type``, ``problem_in_type``, ``overall_problem``: see
  :meth:`~src.behavior_analysis.review_context.ProblemIndex.columns`. A session that is in no
  problem is placed after the problem the mouse was running before it.
"""

from pathlib import Path

import pandas as pd

from src.behavior_analysis.review_context import PROBLEM_DTYPES, ProblemIndex, typed_frame
from src.behavior_import.import_log import SkipRecord
from src.behavior_import.session_order import session_sort_key

_DTYPES = {
    "mouse": "object",
    "session": "object",
    "file": "object",
    "skipped": "object",
    "kind": "object",
    "reason": "object",
    **PROBLEM_DTYPES,
}
COLUMNS = tuple(_DTYPES)


def _row(skip: SkipRecord, index: ProblemIndex) -> dict:
    return {
        "mouse": skip.mouse,
        "session": skip.session,
        "file": Path(skip.file).name if skip.file else None,
        "skipped": skip.level,
        "kind": skip.kind,
        "reason": skip.reason,
        **index.columns(skip.mouse, skip.session, skip.task_name),
    }


def _order(row: dict) -> tuple:
    session = row["session"]
    return (
        row["mouse"] or "",
        session_sort_key(session) if session else (float("inf"), float("inf"), ""),
        row["file"] or "",
    )


def no_trial_sessions_table(skips: list[SkipRecord], index: ProblemIndex) -> pd.DataFrame:
    """One row per skipped file, recording or session.

    Args:
        skips (list[SkipRecord]): ``ImportLog.skips`` of the cohort's import.
        index (ProblemIndex): Where each session sits in its mouse's problems.

    Returns:
        pd.DataFrame: The columns of :data:`COLUMNS`, ordered by mouse, session and file; empty
        (with the columns) if nothing was skipped.
    """
    return typed_frame(sorted((_row(s, index) for s in skips), key=_order), _DTYPES)
