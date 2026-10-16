"""A table of the good reversals that the analyses drop, to review by hand.

:func:`~src.behavior_analysis.get_good_reversal_info.get_good_reversal_info` leaves out every
good reversal whose best tower is unchanged or tied, or whose reward magnitude is missing. A
reversal logged while nothing was recording (between runs, or after a run's last trial) is
checked like any other and dropped only when that check fails. This module lists the dropped
reversals, one row each, with what is needed to find them in the raw files. The rows come from
the same function as the analyses' drops
(:func:`~src.behavior_analysis.get_good_reversal_info.collect_good_reversals`), so the table
and the drop warning cannot disagree.

Columns (see :data:`COLUMNS`):

- ``subject``, ``session``: the session key as in the data (``None`` if no session holds the trial);
- ``session_trial``: the session-local trial index, as in the warning and the reversal records
  (counted from 0, over all of the session's runs);
- ``merged_trial``: the trial index in the subject's merged sessions;
- ``run``: the recording run (counted from 0) holding the trial, when the session records its runs;
- ``file``: that run's file name, when the import kept the source path;
- ``num_t``: the trial number pyControl printed on that trial in the run's file. It is 1-based and
  restarts in every recording, so it is the number to search for in ``file``; it is not
  ``session_trial``. Empty when the raw recordings are not kept;
- ``print_time``: the time column of that print row in the file (seconds), empty with ``num_t``;
- ``reason``: ``best tower unchanged``, ``tie for best`` or ``missing magnitude``;
- ``logged_between_runs``: true if the reversal was logged while nothing was recording (between
  runs, or after a run's last trial). It was checked like any other, so ``reason`` is the real
  result of the check;
- ``good_reversal_number``: the good counter at the trial, as logged;
- ``magnitudes_before`` / ``magnitudes_after``: the reward magnitudes of every tower on the trial
  before and on the reversal trial, as one string such as ``A=4, B=1, C=2``. Tower names differ
  between subjects, so a string per side keeps one set of columns for all of them;
- ``best_before`` / ``best_after``: the best tower on each side (the first listed on a tie), empty
  if a magnitude on that side is missing;
- ``count``: the reversals in the row (a jump of two in the counter is two).

A trial that holds both reversals logged while nothing was recording and reversals inside a run
gives two rows (``logged_between_runs`` true and false), so ``count`` sums to the number the drop
warning reports.
"""

from pathlib import Path

import pandas as pd

from src.behavior_analysis.get_good_reversal_info import collect_good_reversals
from src.behavior_import.extract_trials import ALIASES, recording_trial_rows

_DTYPES = {
    "subject": "object",
    "session": "object",
    "session_trial": "Int64",
    "merged_trial": "int64",
    "run": "Int64",
    "file": "object",
    "num_t": "Int64",
    "print_time": "float64",
    "reason": "object",
    "logged_between_runs": "bool",
    "good_reversal_number": "Int64",
    "magnitudes_before": "object",
    "magnitudes_after": "object",
    "best_before": "object",
    "best_after": "object",
    "count": "int64",
}
COLUMNS = tuple(_DTYPES)
# Columns a caller may put in front of COLUMNS: none, the problem, or where the problem sits in
# a mouse's blocked/drifting numbering.
LEADING_COLUMN_SETS = ((), ("problem",), ("task_type", "problem_in_type", "overall_problem"))
_RAW_COLUMNS = {"time", "type", "subtype", "content"}


def _magnitudes_text(magnitudes):
    """``{"A": 4, "B": None}`` as ``"A=4, B=missing"``."""
    return ", ".join(f"{t}={'missing' if v is None else v}" for t, v in magnitudes.items())


def _best_tower(magnitudes):
    """First-listed tower with the highest magnitude, or ``None`` if any magnitude is missing."""
    if any(v is None for v in magnitudes.values()):
        return None
    return max(magnitudes, key=magnitudes.get)


def _printed_trial(sessions, where):
    """``(num_t, time)`` of the trial print row holding the reversal, or ``(None, None)``.

    The row is the ``where.run_trial``-th trial print of the run's raw frame, which is how the
    import numbered the run's trials. Unknown when the session does not keep one raw frame per
    run, or the frame is not a raw pyControl table.
    """
    session = sessions.get(where.subject, {}).get(where.session)
    if session is None or where.run is None:
        return None, None
    frames = session.get("data")
    frames = frames if isinstance(frames, list) else [frames]
    runs = session.get("trial_variables")
    if not isinstance(runs, list) or len(frames) != len(runs):
        return None, None
    frame = frames[where.run]
    if not isinstance(frame, pd.DataFrame) or not _RAW_COLUMNS <= set(frame.columns):
        return None, None
    rows = recording_trial_rows(frame)
    if where.run_trial >= len(rows):
        return None, None
    time, trial = rows[where.run_trial]
    num_t = next((trial[k] for k in ALIASES["trial"] if k in trial), None)
    return num_t, time


def _rows(dropped, sessions):
    """The table rows of one dropped boundary: its between-runs part, then its in-run part."""
    where = dropped.where
    num_t, print_time = _printed_trial(sessions, where)
    common = {
        "subject": where.subject,
        "session": where.session,
        "session_trial": where.trial,
        "merged_trial": where.merged,
        "run": where.run,
        "file": where.file,
        "num_t": num_t,
        "print_time": print_time,
        "reason": dropped.reason,
        "good_reversal_number": dropped.good_reversal_number,
        "magnitudes_before": _magnitudes_text(dropped.before),
        "magnitudes_after": _magnitudes_text(dropped.after),
        "best_before": _best_tower(dropped.before),
        "best_after": _best_tower(dropped.after),
    }
    parts = [(True, dropped.events), (False, dropped.count - dropped.events)]
    return [
        {**common, "logged_between_runs": between_runs, "count": n}
        for between_runs, n in parts
        if n > 0
    ]


def _typed(rows):
    """*rows* as a frame with the columns and dtypes of :data:`COLUMNS`."""
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    return frame.astype(_DTYPES)


def suspect_reversal_table(
    subjects_trials, pre=5, post=None, include_first_block=False, max_zero_count=1
):
    """One row per good reversal that :func:`get_good_reversal_info` drops.

    Takes the same data and window arguments as ``get_good_reversal_info`` and finds the
    reversals with the same code, so the rows are exactly the ones its warning lists. Nothing is
    warned about and the input is not changed.

    Args:
        subjects_trials (dict): Nested dict ``{subject: {session_key: session_dict}}``.
        pre (int): As in ``get_good_reversal_info``. Default 5.
        post (int | None): As in ``get_good_reversal_info``. Default ``None``.
        include_first_block (bool): As in ``get_good_reversal_info``. Default ``False``.
        max_zero_count (int): As in ``get_good_reversal_info``. Default 1.

    Returns:
        pd.DataFrame: A DataFrame with the columns of :data:`COLUMNS` (see the module
        docstring), in the order of the subjects and, within a subject, of the trials. Empty,
        with the same columns and dtypes, if nothing is dropped.
    """
    _, dropped, _ = collect_good_reversals(
        subjects_trials,
        pre=pre,
        post=post,
        include_first_block=include_first_block,
        max_zero_count=max_zero_count,
    )
    return _typed([row for d in dropped for row in _rows(d, subjects_trials)])


def suspect_reversal_table_by_problem(subjects_trials_by_problem, **kwargs):
    """:func:`suspect_reversal_table` of every problem, stacked, with a leading ``problem`` column.

    Args:
        subjects_trials_by_problem (dict): ``{problem: {subject: {session_key: session_dict}}}``.
        **kwargs: Passed on to :func:`suspect_reversal_table`.

    Returns:
        pd.DataFrame: The tables concatenated, with columns ``problem`` followed by
        :data:`COLUMNS`; empty, with those columns, if nothing is dropped in any problem.
    """
    tables = {p: suspect_reversal_table(d, **kwargs) for p, d in subjects_trials_by_problem.items()}
    # Empty tables are left out of the concatenation so that it keeps every column's dtype.
    tables = [t.assign(problem=p) for p, t in tables.items() if len(t)]
    if not tables:
        return _typed([]).assign(problem=pd.Series([], dtype="object"))[["problem", *COLUMNS]]
    return pd.concat(tables, ignore_index=True)[["problem", *COLUMNS]]


def write_suspect_reversal_table(table, path):
    """Write *table* to the CSV *path*, creating its folder.

    Args:
        table (pd.DataFrame): A table from :func:`suspect_reversal_table` or
            :func:`suspect_reversal_table_by_problem`, optionally with the leading columns of
            :data:`LEADING_COLUMN_SETS`.
        path (str | os.PathLike): Destination file.

    Returns:
        None: Nothing; writes the CSV without the index.

    Raises:
        ValueError: If *table* does not have exactly the expected columns.
    """
    columns = list(table.columns)
    if columns not in ([*leading, *COLUMNS] for leading in LEADING_COLUMN_SETS):
        raise ValueError(
            f"not a suspect-reversal table: columns {columns}, expected {list(COLUMNS)} "
            f"optionally after one of {list(LEADING_COLUMN_SETS[1:])}"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
