"""Choosing which problems a partial cohort rerun must regenerate, and merging its tables.

A problem folder (``drifting/problem-02``) holds every mouse that ran that problem, so one new
session of one mouse changes the whole folder. After a few new sessions only those problems, the
pooled drifting analyses (when a drifting problem is among them) and the cohort-wide summaries
need redoing.

- :func:`select_recent_sessions` picks the sessions (each mouse's last N, or those on or after a
  date) and the ``(task_type, k)`` problems they belong to.
- :func:`describe_plan` says in words what a run will regenerate, and why.
- :func:`replace_problem_rows` and :func:`merge_diagnostic_tables` put the freshly computed
  diagnostic rows of the rerun problems into the stacked tables of the earlier run, keeping the
  rows of every other problem.

Everything here is pure: loading the existing CSVs and writing the merged ones is the caller's
job (:mod:`src.pipelines.cohort_analysis`).
"""

import re
from datetime import date
from typing import NamedTuple

import pandas as pd

from src.behavior_import.session_order import ordered_session_keys, session_date
from src.behavior_import.task_types import BLOCKED, DRIFTING

Problem = tuple[str, int]  # (task_type, k within the type)
SessionPair = tuple[str, str]  # (mouse, session key without any _part-NN)

PROBLEM_COLUMNS = ["task_type", "problem_in_type"]
_PART_RE = re.compile(r"_part-\d+$")
_TASK_TYPE_ORDER = {BLOCKED: 0, DRIFTING: 1}
_SINCE_FORMAT = "YYYY-MM-DD"


class NoSessionsSelectedError(ValueError):
    """No session of the cohort matches the selection (the message says what does exist)."""


class SessionSelection(NamedTuple):
    """The sessions picked for a partial rerun, and the problems they put in play.

    Attributes:
        criterion (str): The selection in words, e.g. ``"last 4 session(s) per mouse"``.
        problems (frozenset[tuple[str, int]]): ``(task_type, k)`` of every problem containing a
            selected session.
        sessions (list[tuple[str, str]]): Selected ``(mouse, session key)``, by mouse then
            chronologically. A session split into ``_part-NN`` segments appears once.
        sessions_by_problem (dict): ``{(task_type, k): [(mouse, session key), ...]}`` for the
            problems in *problems*, ordered blocked before drifting then by ``k``.
    """

    criterion: str
    problems: frozenset
    sessions: list
    sessions_by_problem: dict

    @property
    def runs_pooled_drifting(self) -> bool:
        """Whether a selected problem is drifting, so the pooled drifting analyses change."""
        return any(task_type == DRIFTING for task_type, _ in self.problems)


# ---------------------------------------------------------------------------
# Selecting sessions
# ---------------------------------------------------------------------------


def parse_since(text: str) -> int:
    """An ISO date as the ``YYYYMMDD`` integer that session keys carry.

    Args:
        text (str): A date such as ``"2026-10-16"``.

    Returns:
        int: ``20261016`` for the example.

    Raises:
        ValueError: If *text* is not a real date written ``YYYY-MM-DD``.
    """
    if not isinstance(text, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        raise ValueError(f"cannot read {text!r} as a date; write it as {_SINCE_FORMAT}")
    try:
        parsed = date.fromisoformat(text)
    except ValueError as err:
        raise ValueError(f"{text!r} is not a real date; write it as {_SINCE_FORMAT}") from err
    return parsed.year * 10_000 + parsed.month * 100 + parsed.day


def _problems_of_sessions(grouped: dict) -> dict[str, dict[str, list[Problem]]]:
    """``{mouse: {session key: [problems it appears in]}}``, ``_part-NN`` segments merged."""
    found: dict[str, dict[str, list[Problem]]] = {}
    for problem, subjects in grouped.items():
        for mouse, sessions in subjects.items():
            by_session = found.setdefault(mouse, {})
            for key in sessions:
                problems = by_session.setdefault(_PART_RE.sub("", key), [])
                if problem not in problems:
                    problems.append(problem)
    return found


def check_selection_options(last_sessions: int | None, since: str | None) -> None:
    """Raise unless exactly one valid selection option is given.

    Called once before any cohort is loaded, so a typo fails at once instead of after an import.

    Args:
        last_sessions (int | None): Sessions per mouse; must be an integer of 1 or more.
        since (str | None): A ``YYYY-MM-DD`` date.

    Returns:
        None: Returns nothing when the options are valid.

    Raises:
        ValueError: If not exactly one option is given, *last_sessions* is not an integer of 1
            or more, or *since* is not a real ``YYYY-MM-DD`` date.
    """
    if (last_sessions is None) == (since is None):
        raise ValueError("give exactly one of last_sessions and since")
    if last_sessions is not None and (
        isinstance(last_sessions, bool) or not isinstance(last_sessions, int) or last_sessions < 1
    ):
        raise ValueError(f"last_sessions must be an integer of 1 or more, got {last_sessions!r}")
    if since is not None:
        parse_since(since)


def _describe_existing(sessions_by_mouse: dict[str, list[str]]) -> str:
    """``"M1: 6 session(s), 2026-10-01 to 2026-10-16"`` for each mouse, one per line."""
    lines = []
    for mouse, keys in sorted(sessions_by_mouse.items()):
        dates = [session_date(k) for k in keys if session_date(k) is not None]
        span = f", {_iso(min(dates))} to {_iso(max(dates))}" if dates else ""
        lines.append(f"  {mouse}: {len(keys)} session(s){span}")
    return "\n".join(lines)


def _iso(yyyymmdd: int) -> str:
    text = f"{yyyymmdd:08d}"
    return f"{text[:4]}-{text[4:6]}-{text[6:]}"


def _pick(
    mouse: str, ordered: list[str], last_sessions: int | None, cutoff: int | None
) -> list[str]:
    """The selected sessions of one mouse.

    Args:
        mouse (str): Mouse id, named in the error message.
        ordered (list[str]): The mouse's session keys, oldest first.
        last_sessions (int | None): Keep this many of the newest; ``None`` to select by *cutoff*.
        cutoff (int | None): Keep sessions dated on or after this ``YYYYMMDD`` int; used only
            when *last_sessions* is ``None``.

    Returns:
        list[str]: The kept keys, oldest first.

    Raises:
        ValueError: With a *cutoff*, if a key has no date.
    """
    if last_sessions is not None:
        return ordered[-last_sessions:]
    undated = [k for k in ordered if session_date(k) is None]
    if undated:
        raise ValueError(f"{mouse}: session {undated[0]!r} has no date, so --since cannot place it")
    return [k for k in ordered if session_date(k) >= cutoff]


def select_recent_sessions(
    grouped: dict, *, last_sessions: int | None = None, since: str | None = None
) -> SessionSelection:
    """The recent sessions of a cohort and the problems they belong to.

    Args:
        grouped (dict): ``{(task_type, k): {mouse: {session_key: session}}}`` from
            :func:`~src.behavior_import.task_types.regroup_by_task_type`. Only the keys are read.
        last_sessions (int | None): Take each mouse's most recent N sessions, ordered by
            :func:`~src.behavior_import.session_order.ordered_session_keys` (across all of the
            mouse's problems; a mouse with fewer contributes all it has). Keyword-only.
        since (str | None): Take every session whose date in its key is on or after this
            ``YYYY-MM-DD`` date. Keyword-only.

    Returns:
        SessionSelection: The problems (any with at least one selected session) and the sessions.

    Raises:
        ValueError: If not exactly one of *last_sessions* and *since* is given, *last_sessions* is
            not an integer of 1 or more, *since* is not a date, a mouse's sessions cannot be
            ordered, or (with *since*) a session key has no date.
        NoSessionsSelectedError: If no session matches; the message lists every mouse with its
            number of sessions and date range.
    """
    check_selection_options(last_sessions, since)
    cutoff = parse_since(since) if since is not None else None
    problems_of = _problems_of_sessions(grouped)

    picked: list[SessionPair] = []
    for mouse in sorted(problems_of):
        ordered = ordered_session_keys(mouse, problems_of[mouse])
        picked.extend((mouse, key) for key in _pick(mouse, ordered, last_sessions, cutoff))
    if not picked:
        existing = {m: list(by_session) for m, by_session in problems_of.items()}
        raise NoSessionsSelectedError(
            f"no session matches {since!r}; the cohort has:\n{_describe_existing(existing)}"
            if since is not None
            else f"the cohort has no sessions to take the last {last_sessions} of"
        )

    by_problem: dict[Problem, list[SessionPair]] = {}
    for mouse, key in picked:
        for problem in problems_of[mouse][key]:
            by_problem.setdefault(problem, []).append((mouse, key))
    ordered_problems = sorted(by_problem, key=lambda p: (_TASK_TYPE_ORDER[p[0]], p[1]))
    criterion = f"last {last_sessions} session(s) per mouse" if since is None else f"since {since}"
    return SessionSelection(
        criterion,
        frozenset(by_problem),
        picked,
        {problem: by_problem[problem] for problem in ordered_problems},
    )


def describe_plan(cohort: str, selection: SessionSelection) -> str:
    """What a partial run of *cohort* regenerates and which sessions make it so.

    Args:
        cohort (str): Cohort folder name, e.g. ``"cohort-04"``.
        selection (SessionSelection): From :func:`select_recent_sessions`.

    Returns:
        str: A multi-line plan: the problems with the mice and sessions behind each, the pooled
        drifting scope (regenerated or skipped), and what is always redone.
    """
    lines = [
        f"{cohort}: partial run, {len(selection.sessions)} session(s) selected "
        f"({selection.criterion}), {len(selection.problems)} problem(s) to regenerate"
    ]
    for (task_type, k), pairs in selection.sessions_by_problem.items():
        lines.append(f"  {task_type} problem {k}: {_sessions_by_mouse(pairs)}")
    if selection.runs_pooled_drifting:
        lines.append(
            f"  {DRIFTING}/all-drifting-problems: will be regenerated "
            "(a selected problem is drifting; it pools all drifting problems)"
        )
    else:
        lines.append(f"  {DRIFTING}/all-drifting-problems: skipped (no drifting problem selected)")
    lines.append(
        "  always regenerated: review tables, trial-block-summaries/all-problems; the diagnostic "
        "tables keep the rows of the other problems"
    )
    return "\n".join(lines)


def _sessions_by_mouse(pairs: list[SessionPair]) -> str:
    by_mouse: dict[str, list[str]] = {}
    for mouse, key in pairs:
        by_mouse.setdefault(mouse, []).append(key)
    return "; ".join(f"{mouse}: {', '.join(keys)}" for mouse, keys in by_mouse.items())


def describe_written(
    cohort: str, rerun: frozenset, pooled_ran: bool, tables: dict[str, pd.DataFrame]
) -> str:
    """What a finished partial run of *cohort* wrote.

    Args:
        cohort (str): Cohort folder name, e.g. ``"cohort-04"``.
        rerun (frozenset): ``(task_type, k)`` of the regenerated problems.
        pooled_ran (bool): Whether the pooled drifting analyses were regenerated.
        tables (dict): ``{name: table}`` written after the problems ran (diagnostics, log).

    Returns:
        str: A one-paragraph summary of the figure folders and tables written.
    """
    ordered = sorted(rerun, key=lambda p: (_TASK_TYPE_ORDER[p[0]], p[1]))
    problems = ", ".join(f"{task_type}/problem-{k:02d}" for task_type, k in ordered)
    pooled = "regenerated" if pooled_ran else "not regenerated"
    rows = ", ".join(f"{name} ({len(table)} rows)" for name, table in tables.items())
    return (
        f"[DONE] {cohort} partial run wrote figures for {problems}; "
        f"{DRIFTING}/all-drifting-problems {pooled}; trial-block-summaries/all-problems and "
        f"the review tables rewritten; tables: {rows}"
    )


# ---------------------------------------------------------------------------
# Merging stacked tables
# ---------------------------------------------------------------------------


def _sorted_by_problem(table: pd.DataFrame) -> pd.DataFrame:
    """*table* in problem order (blocked before drifting, then ``k``), keeping row order within."""
    unknown = set(table["task_type"]) - set(_TASK_TYPE_ORDER)
    if unknown:
        raise ValueError(f"unknown task type(s) in a diagnostic table: {sorted(unknown)}")
    order = table.assign(_type_rank=table["task_type"].map(_TASK_TYPE_ORDER))
    order = order.sort_values(["_type_rank", "problem_in_type"], kind="stable")
    return order.drop(columns="_type_rank").reset_index(drop=True)


def replace_problem_rows(
    existing: pd.DataFrame | None, new: pd.DataFrame, rerun: set[Problem] | frozenset[Problem]
) -> tuple[pd.DataFrame, bool]:
    """*existing* with the rows of the *rerun* problems replaced by those of *new*.

    Args:
        existing (pd.DataFrame | None): The table of an earlier run; ``None`` if there is none.
        new (pd.DataFrame): Rows computed for the rerun problems (may be empty).
        rerun (set[tuple[str, int]]): ``(task_type, k)`` whose rows *new* holds.

    Returns:
        tuple[pd.DataFrame, bool]: The merged table, in problem order, and whether it is partial:
        ``True`` when *existing* is missing or lacks the ``task_type`` / ``problem_in_type``
        columns that rows are matched on, in which case only *new* is returned.
    """
    if existing is None or not set(PROBLEM_COLUMNS) <= set(existing.columns):
        return new.reset_index(drop=True), True
    in_rerun = [
        (task_type, k) in rerun
        for task_type, k in zip(existing["task_type"], existing["problem_in_type"], strict=True)
    ]
    kept = existing.loc[[not flag for flag in in_rerun]]
    frames = [f for f in (kept, new) if len(f)] or [new]
    merged = pd.concat(frames, ignore_index=True)
    return _sorted_by_problem(merged), False


def merge_diagnostic_tables(
    new: dict[str, pd.DataFrame],
    existing: dict[str, pd.DataFrame | None],
    rerun: set[Problem] | frozenset[Problem],
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """:func:`replace_problem_rows` for each stacked diagnostic table.

    Args:
        new (dict): ``{table name: rows of the rerun problems}``, from ``stack_diagnostics``.
        existing (dict): ``{table name: table of the earlier run or None}``; must have every name
            of *new*.
        rerun (set[tuple[str, int]]): The problems that were rerun.

    Returns:
        tuple[dict, list[str]]: ``{table name: merged table}`` and the names of the tables that
        are partial (no usable earlier table to merge into).
    """
    merged, partial = {}, []
    for name, rows in new.items():
        merged[name], is_partial = replace_problem_rows(existing[name], rows, rerun)
        if is_partial:
            partial.append(name)
    return merged, partial
