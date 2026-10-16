"""A table of the files, recordings and sessions the importer left out for lack of trials.

The importer skips data in several places and warns for each; the
:class:`~src.behavior_import.import_log.ImportLog` it fills keeps one
:class:`~src.behavior_import.import_log.SkipRecord` per skip. This module turns them into rows,
one per skipped file or recording and one per skipped session (however many recordings it has):

- ``mouse``, ``session``: from the path (``session`` is the folder name, so a non-numbered
  folder such as ``ses-test_date-20260102`` appears as it is); blank where the path has none;
- ``files``: the ``.tsv`` file names, sorted and separated by ``;``: all the recordings of a
  skipped session, or the one file of a file or recording skip; blank where a whole session has
  no file;
- ``skipped``: ``file`` (never became a recording: unreadable, no subject, non-numbered session
  folder), ``recording`` (one recording without trials in a session that has others) or
  ``session`` (the whole session is left out of every analysis);
- ``kind`` and ``reason``: the cause as a code and in plain words; a session whose recordings
  have different causes lists each once, separated by ``;`` (``kind``) and ``; `` (``reason``);
- ``task_type``, ``problem_in_type``, ``overall_problem``: see
  :meth:`~src.behavior_analysis.review_context.ProblemIndex.columns`. A session that is in no
  problem is placed after the problem the mouse was running before it.
"""

from pathlib import Path

import pandas as pd

from src.behavior_analysis.review_context import PROBLEM_DTYPES, ProblemIndex, typed_frame
from src.behavior_import.import_log import SESSION, SkipRecord
from src.behavior_import.session_order import session_sort_key

_DTYPES = {
    "mouse": "object",
    "session": "object",
    "files": "object",
    "skipped": "object",
    "kind": "object",
    "reason": "object",
    **PROBLEM_DTYPES,
}
COLUMNS = tuple(_DTYPES)


def _unique(values) -> list:
    """The distinct values in order of first appearance."""
    return list(dict.fromkeys(values))


def _file_names(skips: list[SkipRecord]) -> str | None:
    names = sorted({Path(skip.file).name for skip in skips if skip.file})
    return ";".join(names) or None


def _row(skips: list[SkipRecord], index: ProblemIndex) -> dict:
    """One row for *skips*, which share their mouse, session and level."""
    first = skips[0]
    return {
        "mouse": first.mouse,
        "session": first.session,
        "files": _file_names(skips),
        "skipped": first.level,
        "kind": ";".join(_unique(skip.kind for skip in skips)),
        "reason": "; ".join(_unique(skip.reason for skip in skips)),
        **index.columns(first.mouse, first.session, first.task_name),
    }


def _row_groups(skips: list[SkipRecord]) -> list[list[SkipRecord]]:
    """Each skipped session's records together; every other skip on its own."""
    sessions: dict[tuple, list[SkipRecord]] = {}
    groups = []
    for skip in skips:
        if skip.level == SESSION:
            sessions.setdefault((skip.mouse, skip.session), []).append(skip)
        else:
            groups.append([skip])
    return groups + list(sessions.values())


def _order(row: dict) -> tuple:
    session = row["session"]
    return (
        row["mouse"] or "",
        session_sort_key(session) if session else (float("inf"), float("inf"), ""),
        row["files"] or "",
    )


def no_trial_sessions_table(skips: list[SkipRecord], index: ProblemIndex) -> pd.DataFrame:
    """One row per skipped file or recording, and one per skipped session.

    Args:
        skips (list[SkipRecord]): ``ImportLog.skips`` of the cohort's import.
        index (ProblemIndex): Where each session sits in its mouse's problems.

    Returns:
        pd.DataFrame: The columns of :data:`COLUMNS`, ordered by mouse, session and file; empty
        (with the columns) if nothing was skipped.
    """
    rows = (_row(group, index) for group in _row_groups(skips))
    return typed_frame(sorted(rows, key=_order), _DTYPES)
