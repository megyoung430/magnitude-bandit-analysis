"""How much each port is chosen relative to how often it is best (drifting task).

All functions are pure: they take the importer's session dicts or the count tables made from
them and return DataFrames. No files, no plots.

Preference of a port or category is ``P(chosen) / P(best)``: the fraction of trials on which it
was chosen divided by the fraction on which it was best, both over the trials where it was
offered. A value of 1 means it is chosen exactly as often as it is best; below 1 it is chosen
less than its value warrants.

Geometry. Ports sit on a 3 x 3 grid, letters are columns (A, B, C) and numbers are rows (1 at the
bottom), so A1 is bottom-left and C3 top-right. Relative to a problem's initiation port a port is

- ``"plus"`` if it is in the same row or column as the initiation port (from B2: B1, B3, A2, C2),
- ``"X"`` if it is on one of the initiation port's diagonals, ``|dcolumn| == |drow|`` (from B2:
  A1, C1, A3, C3),
- ``"initiation"`` if it is the initiation port, and ``"other"`` for every remaining port.

From B2 these are exactly the eight neighbours. From a corner such as A1 the plus ports are A2,
A3, B1, C1, the X ports are B2 and C3, and B3 and C2 are other.

Counting. Only trials with a choice and reward magnitudes are counted. On each trial every
offered port (a key of the trial's magnitudes) gets one offered count; the chosen port gets one
chosen count; the port(s) with the highest magnitude share one best count evenly, so a two-way
tie gives each port 0.5. A category is offered on a trial if any of its ports was offered; it is
chosen if the chosen port belongs to it, and it gets the best shares of its ports.

Counts are pooled across problems before any ratio is formed, never by averaging ratios.

A mouse whose pooled scope offers ports from one category only has a category ratio of 1 by
construction (that category is offered, chosen and best on every trial); see
:func:`mask_single_category_mice`.
"""

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.behavior_analysis.poke_events import ARMS9
from src.behavior_analysis.stats import nan_mean, nan_standard_error
from src.behavior_import.problem_numbering import DEFAULT_INITIATION_TOWER

logger = logging.getLogger(__name__)

CATEGORIES = ("initiation", "plus", "X", "other")
COMPARED_CATEGORIES = ("X", "plus")
COUNT_COLUMNS = ["n_trials", "chosen", "best"]


class PortCounts(NamedTuple):
    """Count tables of :func:`count_port_preference`.

    Attributes:
        by_port (pd.DataFrame): One row per mouse, problem and port (all nine, also those never
            offered) with columns ``mouse, problem, initiation, port, category, n_trials,
            chosen, best``.
        by_category (pd.DataFrame): One row per mouse, problem and category with columns
            ``mouse, problem, initiation, category, n_trials, chosen, best``.
        skipped (pd.Series): Trials without a choice or magnitudes, per mouse.
    """

    by_port: pd.DataFrame
    by_category: pd.DataFrame
    skipped: pd.Series


def _grid_position(port: str) -> tuple[int, int]:
    """(column, row) of *port*; raises ValueError for anything but the nine ports."""
    if port not in ARMS9:
        raise ValueError(f"unknown port {port!r}; expected one of {ARMS9}")
    return "ABC".index(port[0]), int(port[1]) - 1


def port_category(port: str, initiation: str) -> str:
    """``"plus"``, ``"X"``, ``"initiation"`` or ``"other"`` for *port* seen from *initiation*.

    Args:
        port (str): One of the nine ports, e.g. ``"A1"``.
        initiation (str): The initiation port of the problem, e.g. ``"B2"``.

    Returns:
        str: ``"initiation"`` for the initiation port itself, ``"plus"`` if in its row or
        column, ``"X"`` if on one of its diagonals, else ``"other"``.

    Raises:
        ValueError: If *port* or *initiation* is not one of the nine ports.
    """
    column, row = _grid_position(port)
    init_column, init_row = _grid_position(initiation)
    d_column, d_row = abs(column - init_column), abs(row - init_row)
    if d_column == d_row == 0:
        return "initiation"
    if d_column == 0 or d_row == 0:
        return "plus"
    if d_column == d_row:
        return "X"
    return "other"


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def _best_shares(magnitudes: dict) -> dict[str, float]:
    """``{port: share}`` of the best count: 1 for one best port, 1/k each for k tied."""
    top = max(magnitudes.values())
    best = [port for port, magnitude in magnitudes.items() if magnitude == top]
    return dict.fromkeys(best, 1 / len(best))


def _trial_tallies(magnitudes: dict, choice: str, initiation: str, where: str):
    """``(port_tally, category_tally)`` of one trial: ``{key: (n_trials, chosen, best)}`` each.

    Raises:
        ValueError: If *choice* is not an offered port; *where* locates the trial.
    """
    if choice not in magnitudes:
        raise ValueError(f"{where}: choice {choice!r} is not among the offered {list(magnitudes)}")
    best = _best_shares(magnitudes)
    ports = {port: (1.0, float(port == choice), best.get(port, 0.0)) for port in magnitudes}
    categories: dict[str, np.ndarray] = {}
    for port, values in ports.items():
        total = categories.setdefault(port_category(port, initiation), np.zeros(3))
        # a category is offered once per trial, however many of its ports are offered
        total += (0.0, values[1], values[2])
        total[0] = 1.0
    return ports, categories


def _session_recordings(session: dict, where: str) -> list[dict]:
    recordings = session.get("trial_variables") or []
    for recording in recordings:
        n_mags = len(recording.get("reward_magnitudes", []))
        n_choice = len(recording.get("choice", []))
        if n_mags != n_choice:
            raise ValueError(
                f"{where}: {n_mags} reward_magnitudes but {n_choice} choices in one recording"
            )
    return recordings


def _problem_initiation(mouse: str, problem, sessions: dict) -> str:
    """The one initiation port of a mouse's problem (default ``B2`` if not logged).

    Raises:
        ValueError: If the problem's sessions disagree.
    """
    found = {s.get("initiation_tower") or DEFAULT_INITIATION_TOWER for s in sessions.values()}
    if len(found) > 1:
        raise ValueError(
            f"mouse {mouse!r}, problem {problem!r}: sessions have different initiation "
            f"towers {sorted(found)}; port categories need one per problem"
        )
    return found.pop() if found else DEFAULT_INITIATION_TOWER


def _count_mouse_problem(mouse: str, problem, sessions: dict, initiation: str):
    """``(port_totals, category_totals, n_skipped)`` of one mouse's problem."""
    port_totals: dict[str, np.ndarray] = {p: np.zeros(3) for p in ARMS9}
    category_totals: dict[str, np.ndarray] = {c: np.zeros(3) for c in CATEGORIES}
    skipped = 0
    for key, session in sessions.items():
        for recording in _session_recordings(session, f"mouse {mouse!r}, session {key!r}"):
            for t, (magnitudes, choice) in enumerate(
                zip(
                    recording.get("reward_magnitudes", []), recording.get("choice", []), strict=True
                )
            ):
                if not magnitudes or _is_missing(choice):
                    skipped += 1
                    continue
                where = f"mouse {mouse!r}, problem {problem!r}, session {key!r}, trial {t}"
                port_tally, category_tally = _trial_tallies(magnitudes, choice, initiation, where)
                for port, values in port_tally.items():
                    port_totals[port] += values
                for category, values in category_tally.items():
                    category_totals[category] += values
    return port_totals, category_totals, skipped


def count_port_preference(subjects_trials_by_problem: dict) -> PortCounts:
    """Offered, chosen and best counts per port and per category, for every mouse and problem.

    Each mouse's problem is counted separately because the categories depend on that problem's
    initiation port; pool the problems afterwards with :func:`pool_problems`. The initiation
    port is the session's ``"initiation_tower"``, ``B2`` if the session does not log one.

    Args:
        subjects_trials_by_problem (dict): ``{problem: {mouse: {session_key: session}}}``;
            a session holds ``"trial_variables"``, a list with one dict per recording that has
            per-trial ``"reward_magnitudes"`` (``{port: magnitude}``) and ``"choice"`` lists.

    Returns:
        PortCounts: The two count tables and the number of skipped trials per mouse (see the
        module docstring for the counting rules).

    Raises:
        ValueError: If a problem's sessions have different initiation ports, a recording's
            lists differ in length, a choice is not an offered port, a port is unknown, or
            there is no countable trial at all.
    """
    port_rows, category_rows, skipped = [], [], {}
    for problem, subjects in subjects_trials_by_problem.items():
        for mouse, sessions in subjects.items():
            initiation = _problem_initiation(mouse, problem, sessions)
            ports, categories, n_skipped = _count_mouse_problem(
                mouse, problem, sessions, initiation
            )
            skipped[mouse] = skipped.get(mouse, 0) + n_skipped
            base = {"mouse": mouse, "problem": problem, "initiation": initiation}
            port_rows += [
                {
                    **base,
                    "port": port,
                    "category": port_category(port, initiation),
                    **dict(zip(COUNT_COLUMNS, values, strict=True)),
                }
                for port, values in ports.items()
            ]
            category_rows += [
                {**base, "category": category, **dict(zip(COUNT_COLUMNS, values, strict=True))}
                for category, values in categories.items()
            ]
    by_port = pd.DataFrame(port_rows)
    if by_port.n_trials.sum() == 0:
        raise ValueError("no trials with a choice and reward magnitudes to count")
    return PortCounts(by_port, pd.DataFrame(category_rows), pd.Series(skipped, dtype=int))


def pool_problems(counts: pd.DataFrame, key: str) -> pd.DataFrame:
    """Add each mouse's counts over problems, per *key* (``"port"`` or ``"category"``).

    Counts are summed, so the ratio of the result weights every trial equally.

    Args:
        counts (pd.DataFrame): ``by_port`` or ``by_category`` of :func:`count_port_preference`.
        key (str): ``"port"`` for ``by_port``, ``"category"`` for ``by_category``.

    Returns:
        pd.DataFrame: Columns ``mouse``, *key*, ``n_trials``, ``chosen``, ``best``.
    """
    return counts.groupby(["mouse", key], sort=True)[COUNT_COLUMNS].sum().reset_index()


def add_ratio(counts: pd.DataFrame) -> pd.DataFrame:
    """Add ``ratio = (chosen / n_trials) / (best / n_trials)``; NaN where it is undefined.

    Undefined means never offered or never best (a division by zero).

    Args:
        counts (pd.DataFrame): Any count table with ``n_trials``, ``chosen`` and ``best``.

    Returns:
        pd.DataFrame: A copy of *counts* with the ``ratio`` column.
    """
    offered = counts.n_trials.to_numpy(dtype=float)
    defined = (offered > 0) & (counts.best.to_numpy(dtype=float) > 0)
    safe_offered = np.where(defined, offered, 1.0)
    p_chosen = counts.chosen.to_numpy(dtype=float) / safe_offered
    p_best = counts.best.to_numpy(dtype=float) / safe_offered
    ratio = np.where(defined, p_chosen / np.where(defined, p_best, 1.0), np.nan)
    return counts.assign(ratio=ratio)


def category_summary(
    category_ratios: pd.DataFrame, categories: tuple[str, ...] = COMPARED_CATEGORIES
) -> pd.DataFrame:
    """Mean and standard error over mice of the per-mouse category ratios.

    Mice with an undefined ratio (category never best) are left out of that category.

    Args:
        category_ratios (pd.DataFrame): Columns ``mouse``, ``category``, ``ratio``.
        categories (tuple[str, ...]): Categories to summarise, in order.
            Default ``("X", "plus")``.

    Returns:
        pd.DataFrame: Indexed by category, with columns ``mean``, ``se`` (NaN with fewer than
        two mice) and ``n_mice``.
    """
    wide = category_ratios.pivot(index="mouse", columns="category", values="ratio")
    stack = wide.reindex(columns=list(categories)).to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "mean": nan_mean(stack, axis=0),
            "se": nan_standard_error(stack),
            "n_mice": (~np.isnan(stack)).sum(axis=0),
        },
        index=pd.Index(categories, name="category"),
    )


def mask_single_category_mice(category_ratios: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Set every ratio of a mouse to NaN if only one category was offered to it.

    With a single offered category that category is offered, chosen and best on every trial, so
    its ratio is 1 whatever the mouse did; it carries no information.

    Args:
        category_ratios (pd.DataFrame): Pooled per-mouse category counts with ``ratio``
            (``add_ratio`` of ``pool_problems(..., "category")``), over the scope to judge.

    Returns:
        tuple[pd.DataFrame, list[str]]: A copy of *category_ratios* with those mice's ratios NaN,
        and the sorted names of those mice.
    """
    offered = category_ratios[category_ratios.n_trials > 0]
    n_offered = offered.groupby("mouse")["category"].nunique()
    single = sorted(n_offered.index[n_offered == 1])
    masked = category_ratios.assign(
        ratio=category_ratios.ratio.where(~category_ratios.mouse.isin(single), np.nan)
    )
    return masked, single


def count_nan_ratios(port_ratios: pd.DataFrame, category_ratios: pd.DataFrame) -> dict[str, int]:
    """Number of mouse-port and mouse-category ratios that are NaN although the item was offered.

    These are the items that were offered at least once but never best.

    Args:
        port_ratios (pd.DataFrame): Per-mouse port counts with ``ratio`` (see :func:`add_ratio`).
        category_ratios (pd.DataFrame): Per-mouse category counts with ``ratio``.

    Returns:
        dict[str, int]: ``{"ports": n, "categories": n}``.
    """

    def n_undefined(table: pd.DataFrame) -> int:
        return int(((table.n_trials > 0) & table.ratio.isna()).sum())

    return {"ports": n_undefined(port_ratios), "categories": n_undefined(category_ratios)}
