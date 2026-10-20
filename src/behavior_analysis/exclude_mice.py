"""Leave chosen mice out of the pooled drifting analyses (no files, no plotting).

The pooled analyses take ``{problem: {mouse: sessions}}``. :func:`exclude_mice_from_problems`
drops mice from that dict before pooling, so every analysis (ranks, psychometrics, value
change, ...) sees the same reduced set without any of them knowing about exclusions. The
remaining functions name the extra output set (folder, figure scope, description file).
"""

import re
from collections.abc import Iterable, Mapping
from datetime import datetime

import pandas as pd

# Mouse ids become part of a folder name, so only plain name characters are accepted.
_MOUSE_ID = re.compile(r"[A-Za-z0-9_.-]+")


def check_mouse_ids(mice: Iterable[str]) -> tuple[str, ...]:
    """The mouse ids, sorted and once each, after checking they can name a folder.

    Args:
        mice (Iterable[str]): Mouse ids such as ``"MY_175_N"``.

    Returns:
        tuple[str, ...]: The distinct ids in sorted order.

    Raises:
        ValueError: If there are none, or an id is not a string of letters, digits, ``_``, ``.``
            and ``-`` (a path separator in an id would write outside the results folder).
    """
    ids = list(mice)
    if not ids:
        raise ValueError("no mice given to exclude; name at least one mouse id")
    for mouse in ids:
        if not isinstance(mouse, str) or not _MOUSE_ID.fullmatch(mouse) or mouse in (".", ".."):
            raise ValueError(
                f"{mouse!r} is not a mouse id: use letters, digits, '_', '.' and '-' only"
            )
    return tuple(sorted(set(ids)))


def exclude_mice_from_problems(problems: Mapping, mice: Iterable[str]) -> dict:
    """The problems without the named mice; problems left with no mouse are dropped.

    Args:
        problems (Mapping): ``{problem: {mouse: sessions}}``, the problem key of any type (the
            drifting problem number, or ``(cohort id, number)`` across cohorts).
        mice (Iterable[str]): Mouse ids to leave out; each must run in at least one problem.

    Returns:
        dict: ``{problem: {mouse: sessions}}`` with the same sessions objects, in the order of
        *problems*. The input is not modified.

    Raises:
        ValueError: If *mice* is empty or has an invalid id, a named mouse is in none of the
            problems (the message lists the mice that are, since a typo is the likely cause),
            or no mouse is left.
    """
    excluded = set(check_mouse_ids(mice))
    present = sorted({m for subjects in problems.values() for m in subjects})
    unknown = sorted(excluded - set(present))
    if unknown:
        raise ValueError(
            f"cannot exclude {', '.join(unknown)}: not among the mice of the drifting problems "
            f"({', '.join(present) or 'none'})"
        )
    kept = {
        problem: {m: sessions for m, sessions in subjects.items() if m not in excluded}
        for problem, subjects in problems.items()
    }
    kept = {problem: subjects for problem, subjects in kept.items() if subjects}
    if not kept:
        raise ValueError(
            f"no mice left after excluding {', '.join(sorted(excluded))}: "
            f"the drifting problems have only {', '.join(present)}"
        )
    return kept


def exclude_mice_from_table(
    table: pd.DataFrame, mice: Iterable[str], column: str = "mouse"
) -> pd.DataFrame:
    """The rows of *table* whose *column* is not one of *mice*.

    Args:
        table (pd.DataFrame): A table with one mouse id per row in *column*.
        mice (Iterable[str]): Mouse ids to drop.
        column (str): The column holding the mouse id. Default ``"mouse"``.

    Returns:
        pd.DataFrame: A copy of *table* without those rows, index reset.

    Raises:
        ValueError: If *table* has no *column*.
    """
    if column not in table:
        raise ValueError(f"table has no {column!r} column to drop mice by: {list(table.columns)}")
    return table[~table[column].isin(set(mice))].reset_index(drop=True)


def excluded_set_name(base: str, mice: Iterable[str]) -> str:
    """``"all-drifting-problems-without-MY_175_N-MY_66_LL"`` for the base name and the mice.

    Args:
        base (str): Folder name of the full set, e.g. ``"all-drifting-problems"``.
        mice (Iterable[str]): Excluded mouse ids; sorted so the name does not depend on order.

    Returns:
        str: *base* then ``-without-`` and the sorted ids joined by ``-``.
    """
    return f"{base}-without-{'-'.join(check_mouse_ids(mice))}"


def excluded_scope_text(mice: Iterable[str]) -> str:
    """``"without MY_175_N, MY_66_LL"``: the scope text figures of the reduced set carry."""
    return f"without {', '.join(check_mouse_ids(mice))}"


def excluded_mice_description(cohort: str, mice: Iterable[str], run_at: datetime) -> str:
    """The text of ``excluded_mice.txt``: what the folder holds, who is left out, and when.

    Args:
        cohort (str): What the set pools, e.g. ``"cohort-03"``.
        mice (Iterable[str]): Excluded mouse ids.
        run_at (datetime): When the set was written.

    Returns:
        str: Three lines ending in a newline.
    """
    ids = check_mouse_ids(mice)
    return (
        f"Drifting problems of {cohort} pooled without {', '.join(ids)}\n"
        f"Excluded mice: {', '.join(ids)}\n"
        f"Run at {run_at:%Y-%m-%d %H:%M:%S}\n"
    )
