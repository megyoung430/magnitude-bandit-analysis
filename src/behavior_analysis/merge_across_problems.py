"""Utilities for merging per-problem behavioral data across problems.

These helpers are used to pool reversal windows, first-leave counts, and
other per-subject structures that are computed independently for each problem
before being averaged across problems.
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# First-leave merge helpers
# ---------------------------------------------------------------------------

def sum_count_dicts(dicts: list[dict]) -> dict:
    """Sum a list of count dicts into one, ensuring required keys exist.

    Args:
        dicts: List of dicts mapping category names to numeric counts.

    Returns:
        A single dict with summed values across all input dicts.
        Always contains keys ``"new_best"``, ``"third"``, and ``"total"``.
    """
    out: dict[str, Any] = {}
    for d in dicts:
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            if v is None:
                continue
            try:
                out[k] = (out.get(k, 0) or 0) + v
            except Exception:
                pass
    out.setdefault("new_best", 0)
    out.setdefault("third", 0)
    out.setdefault("total", 0)
    return out


def normalize_subject_value(c: Any) -> dict | None:
    """Normalise a subject entry from ``get_first_leave_after_good_reversals``.

    The raw output can be a single dict, a list of dicts, or a nested list of
    dicts.  This function collapses all forms into a single summed count dict.

    Args:
        c: Raw subject value from ``get_first_leave_after_good_reversals``.

    Returns:
        A single count dict, or ``None`` if the input is unsupported.
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
        if flat:
            return sum_count_dicts(flat)
    return None


def merge_first_leave_dicts(counts_by_problem: dict[Any, dict]) -> dict:
    """Merge per-problem first-leave count dicts into a single per-subject dict.

    Args:
        counts_by_problem: ``{problem: {subject: count_dict_or_list}}``

    Returns:
        ``{subject: summed_count_dict}`` with counts accumulated across all
        problems.  Subjects that cannot be normalised are silently skipped.
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
                try:
                    merged[subj][k] = (merged[subj].get(k, 0) or 0) + v
                except Exception:
                    pass

    bad = {s: type(v) for s, v in merged.items() if not isinstance(v, dict)}
    if bad:
        print("[WARNING] Non-dict merged values detected:", bad)
    return merged


# ---------------------------------------------------------------------------
# Reversal-window merge helper
# ---------------------------------------------------------------------------

def merge_windows_by_subject(windows_by_problem: dict[Any, dict]) -> dict:
    """Concatenate per-problem reversal windows into a single per-subject list.

    Args:
        windows_by_problem: ``{problem: {subject: list[reversal_dict]}}``

    Returns:
        ``{subject: list[reversal_dict]}`` with all reversals from all problems
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
