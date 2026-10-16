"""A table of the Python warnings raised during a cohort run.

The run prints its warnings to the console, where they scroll away. This keeps one row for each
distinct warning (same class and text), in the order first seen. Columns:

- ``category``, ``message``: the warning class name and text;
- ``n_times``: how often it was raised;
- ``source``: ``file.py:line`` that raised it (the first time);
- ``mouse``, ``session``, ``problem_in_message``: read from the text where it names them, blank
  otherwise. The mouse is one of the cohort's mice named as a whole word, and only when the text
  names exactly one; the session is the first ``ses-<N>_date-<YYYYMMDD>`` (with ``_part-NN``)
  in the text; ``problem_in_message`` is the number after "problem" (as the text counts
  problems, which is the importer's numbering for numbering warnings);
- ``task_type``, ``problem_in_type``, ``overall_problem``: where the named session sits in the
  mouse's problems, blank unless both mouse and session were read (see
  :meth:`~src.behavior_analysis.review_context.ProblemIndex.columns`).
"""

import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from src.behavior_analysis.review_context import PROBLEM_DTYPES, ProblemIndex, typed_frame
from src.behavior_import.warning_records import WarningRecord

_SESSION_RE = re.compile(r"ses-\d+_date-\d{8}(?:_part-\d+)?")
_PROBLEM_RE = re.compile(r"\bproblem[ -](\d+)", re.IGNORECASE)

_DTYPES = {
    "category": "object",
    "message": "object",
    "n_times": "Int64",
    "source": "object",
    "mouse": "object",
    "session": "object",
    "problem_in_message": "Int64",
    **PROBLEM_DTYPES,
}
COLUMNS = tuple(_DTYPES)


def parse_warning_message(message: str, mice: Iterable[str]) -> dict:
    """The mouse, session and problem number a warning text names, ``None`` where it names none.

    Args:
        message (str): The warning text.
        mice (Iterable[str]): The cohort's mouse ids. A mouse is found as a whole word, so ``M1``
            does not match inside ``M10``.

    Returns:
        dict: ``mouse`` (the one mouse named, ``None`` if none or several), ``session`` (the
        first session key) and ``problem_in_message`` (the first number after "problem").
    """
    named = {mouse for mouse in mice if re.search(rf"(?<![\w]){re.escape(mouse)}(?![\w])", message)}
    session = _SESSION_RE.search(message)
    problem = _PROBLEM_RE.search(message)
    return {
        "mouse": next(iter(named)) if len(named) == 1 else None,
        "session": session.group(0) if session else None,
        "problem_in_message": int(problem.group(1)) if problem else None,
    }


def warnings_log_table(records: list[WarningRecord], index: ProblemIndex) -> pd.DataFrame:
    """One row per distinct warning of a run.

    Args:
        records (list[WarningRecord]): The warnings in the order raised (see
            :func:`~src.behavior_import.warning_records.record_warnings`).
        index (ProblemIndex): The cohort's mice and problems, to read names out of the texts and
            to place the sessions they name.

    Returns:
        pd.DataFrame: The columns of :data:`COLUMNS`; empty (with the columns) without warnings.
    """
    mice = sorted(index.sessions)
    counts = Counter((r.category, r.message) for r in records)
    first = {}
    for r in records:
        first.setdefault((r.category, r.message), r)
    rows = []
    for (category, message), n_times in counts.items():
        named = parse_warning_message(message, mice)
        located = named["mouse"] and named["session"]
        rows.append(
            {
                "category": category,
                "message": message,
                "n_times": n_times,
                "source": f"{Path(first[(category, message)].filename).name}:"
                f"{first[(category, message)].lineno}",
                **named,
                **(
                    index.columns(named["mouse"], named["session"])
                    if located
                    else dict.fromkeys(PROBLEM_DTYPES)
                ),
            }
        )
    return typed_frame(rows, _DTYPES)
