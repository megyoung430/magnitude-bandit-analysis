"""A table of the good reversals that the analyses drop, to review by hand.

:func:`~src.behavior_analysis.get_good_reversal_info.get_good_reversal_info` leaves out every
good reversal whose best tower is unchanged or tied, whose reward magnitude is missing, or that
was logged while nothing was recording (so it cannot be checked). This module lists those
reversals, one row each, with what is needed to find them in the raw files. The rows come from
the same function as the analyses' drops
(:func:`~src.behavior_analysis.get_good_reversal_info.collect_good_reversals`), so the table
and the drop warning cannot disagree.

Columns (see :data:`COLUMNS`):

- ``subject``, ``session``: the session key as in the data (``None`` if no session holds the trial);
- ``session_trial``: the session-local trial index, as in the warning and the reversal records
  (counted from 0);
- ``merged_trial``: the trial index in the subject's merged sessions;
- ``run``: the recording run (counted from 0) holding the trial, when the session records its runs;
- ``file``: that run's file name, when the import kept the source path;
- ``reason``: ``best tower unchanged``, ``tie for best``, ``missing magnitude`` or
  ``uncheckable between-run or terminal`` (logged while nothing was recording);
- ``good_reversal_number``: the good counter at the trial, as logged;
- ``magnitudes_before`` / ``magnitudes_after``: the reward magnitudes of every tower on the trial
  before and on the reversal trial, as one string such as ``A=4, B=1, C=2``. Tower names differ
  between subjects, so a string per side keeps one set of columns for all of them;
- ``best_before`` / ``best_after``: the best tower on each side (the first listed on a tie), empty
  if a magnitude on that side is missing;
- ``count``: the reversals in the row (a jump of two in the counter is two).

A trial that holds both reversals logged while nothing was recording and reversals inside a run
gives two rows, so ``count`` sums to the number the drop warning reports.
"""

from pathlib import Path

import pandas as pd

from src.behavior_analysis.get_good_reversal_info import collect_good_reversals

UNCHECKABLE_EVENT = "uncheckable between-run or terminal"

_DTYPES = {
    "subject": "object",
    "session": "object",
    "session_trial": "Int64",
    "merged_trial": "int64",
    "run": "Int64",
    "file": "object",
    "reason": "object",
    "good_reversal_number": "Int64",
    "magnitudes_before": "object",
    "magnitudes_after": "object",
    "best_before": "object",
    "best_after": "object",
    "count": "int64",
}
COLUMNS = tuple(_DTYPES)


def _magnitudes_text(magnitudes):
    """``{"A": 4, "B": None}`` as ``"A=4, B=missing"``."""
    return ", ".join(f"{t}={'missing' if v is None else v}" for t, v in magnitudes.items())


def _best_tower(magnitudes):
    """First-listed tower with the highest magnitude, or ``None`` if any magnitude is missing."""
    if any(v is None for v in magnitudes.values()):
        return None
    return max(magnitudes, key=magnitudes.get)


def _rows(dropped):
    """The table rows of one dropped boundary: its event part, then its in-run part."""
    where = dropped.where
    common = {
        "subject": where.subject,
        "session": where.session,
        "session_trial": where.trial,
        "merged_trial": where.merged,
        "run": where.run,
        "file": where.file,
        "good_reversal_number": dropped.good_reversal_number,
        "magnitudes_before": _magnitudes_text(dropped.before),
        "magnitudes_after": _magnitudes_text(dropped.after),
        "best_before": _best_tower(dropped.before),
        "best_after": _best_tower(dropped.after),
    }
    parts = [
        (UNCHECKABLE_EVENT, dropped.events),
        (dropped.reason, dropped.count - dropped.events),
    ]
    return [{**common, "reason": reason, "count": n} for reason, n in parts if n > 0]


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
        subjects_trials: Nested dict ``{subject: {session_key: session_dict}}``.
        pre, post, include_first_block, max_zero_count: As in ``get_good_reversal_info``.

    Returns:
        A DataFrame with the columns of :data:`COLUMNS` (see the module docstring), in the
        order of the subjects and, within a subject, of the trials. Empty, with the same
        columns and dtypes, if nothing is dropped.
    """
    _, dropped, _ = collect_good_reversals(
        subjects_trials,
        pre=pre,
        post=post,
        include_first_block=include_first_block,
        max_zero_count=max_zero_count,
    )
    return _typed([row for d in dropped for row in _rows(d)])


def suspect_reversal_table_by_problem(subjects_trials_by_problem, **kwargs):
    """:func:`suspect_reversal_table` of every problem, stacked, with a leading ``problem`` column.

    Args:
        subjects_trials_by_problem: ``{problem: {subject: {session_key: session_dict}}}``.
        **kwargs: Passed on to :func:`suspect_reversal_table`.
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
        table: A table from :func:`suspect_reversal_table` or
            :func:`suspect_reversal_table_by_problem`.
        path: Destination file.

    Raises:
        ValueError: If *table* does not have exactly the expected columns.
    """
    columns = list(table.columns)
    if columns not in (list(COLUMNS), ["problem", *COLUMNS]):
        raise ValueError(
            f"not a suspect-reversal table: columns {columns}, expected {list(COLUMNS)} "
            "(optionally after 'problem')"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
