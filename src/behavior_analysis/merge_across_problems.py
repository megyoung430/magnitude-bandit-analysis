"""Utilities for merging per-problem behavioral data across problems.

These helpers are used to pool reversal windows, first-leave counts, and
other per-subject structures that are computed independently for each problem
before being averaged across problems.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# First-leave merge helpers
# ---------------------------------------------------------------------------

def _add_count(counts: dict, key: str, value: Any) -> None:
    """Add *value* to ``counts[key]``; counts must be numeric so none are silently dropped."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"count {key!r} has non-numeric value {value!r}")
    counts[key] = (counts.get(key, 0) or 0) + value


def sum_count_dicts(dicts: list[dict]) -> dict:
    """Sum a list of count dicts into one, ensuring required keys exist.

    Args:
        dicts (list[dict]): Dicts mapping category names to numeric counts. Non-dict entries
            and ``None`` values are skipped.

    Returns:
        dict: A single dict with summed values across all input dicts.
        Always contains keys ``"new_best"``, ``"third"``, and ``"total"``.
    """
    out: dict[str, Any] = {}
    for d in dicts:
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            if v is None:
                continue
            _add_count(out, k, v)
    out.setdefault("new_best", 0)
    out.setdefault("third", 0)
    out.setdefault("total", 0)
    return out


def normalize_subject_value(c: Any) -> dict | None:
    """Normalise a subject entry from ``get_first_leave_after_good_reversals``.

    The raw output can be a single dict, a list of dicts, or a nested list of
    dicts.  This function collapses all forms into a single summed count dict.

    Args:
        c (dict | list | tuple | None): Raw subject value from
            ``get_first_leave_after_good_reversals``: a count dict, a list of count dicts, or a
            nested list of count dicts.

    Returns:
        dict | None: A single count dict, or ``None`` if *c* is ``None`` or holds only empty
        lists.

    Raises:
        TypeError: If *c* has any other structure or a count is non-numeric.
    """
    if c is None:
        return None
    if isinstance(c, dict):
        return sum_count_dicts([c])
    if isinstance(c, (list, tuple)):
        # flat list of dicts
        if all(isinstance(x, dict) for x in c):
            return sum_count_dicts(list(c))
        # nested list: [[{...}], [{...}]] — flatten one level
        flat: list[dict] = []
        for x in c:
            if isinstance(x, dict):
                flat.append(x)
            elif isinstance(x, (list, tuple)) and all(isinstance(xx, dict) for xx in x):
                flat.extend(list(x))
            else:
                raise TypeError(f"unsupported first-leave entry {x!r}")
        return sum_count_dicts(flat) if flat else None
    raise TypeError(f"unsupported first-leave value of type {type(c).__name__}")


def merge_first_leave_dicts(counts_by_problem: dict[Any, dict]) -> dict:
    """Merge per-problem first-leave count dicts into a single per-subject dict.

    Args:
        counts_by_problem (dict): ``{problem: {subject: count_dict_or_list}}``; empty problems
            are skipped.

    Returns:
        dict: ``{subject: summed_count_dict}`` with counts accumulated across all
        problems.  Subjects whose value is ``None`` are skipped.
    """
    merged: dict[str, dict] = {}
    for _p, d in counts_by_problem.items():
        if not d:
            continue
        for subj, c in d.items():
            cd = normalize_subject_value(c)
            if cd is None:
                continue
            if subj not in merged:
                merged[subj] = {"new_best": 0, "third": 0, "total": 0}
            for k, v in cd.items():
                if v is None:
                    continue
                _add_count(merged[subj], k, v)

    bad = {s: type(v) for s, v in merged.items() if not isinstance(v, dict)}
    if bad:
        warnings.warn(f"Non-dict merged values detected: {bad}", stacklevel=2)
    return merged


# ---------------------------------------------------------------------------
# Reversal-window merge helper
# ---------------------------------------------------------------------------

def merge_windows_by_subject(windows_by_problem: dict[Any, dict]) -> dict:
    """Concatenate per-problem reversal windows into a single per-subject list.

    Args:
        windows_by_problem (dict): ``{problem: {subject: list[reversal_dict]}}``; ``None``
            problems and empty lists are skipped.

    Returns:
        dict: ``{subject: list[reversal_dict]}`` with all reversals from all problems
        concatenated together.
    """
    merged: dict[str, list] = {}
    for _p, win in windows_by_problem.items():
        if win is None:
            continue
        for subj, revs in win.items():
            if not revs:
                continue
            merged.setdefault(subj, []).extend(revs)
    return merged
