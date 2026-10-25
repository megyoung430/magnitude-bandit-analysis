"""Where mice poke during port finding and just after it: three windows per mouse and problem.

For one mouse in one problem (:func:`mouse_port_finding_summary`) the pokes are counted per port
in three windows:

- ``"port_finding"``: every port-finding trial of the problem, those of the port-finding-only
  sessions that belong to it included (see :mod:`src.behavior_analysis.port_finding_sessions`),
  as in the per-mouse port-finding maps;
- ``"first_trials"``: the first :data:`N_FIRST_TRIALS` task trials of the problem (trials with a
  trial print that are not port finding), in session order, over as many sessions as it takes;
- ``"first_blocks"``: the task trials of the problem's first :data:`N_FIRST_BLOCKS` blocks. The
  importer numbers blocks within a session and restarts at 1 in the next one, so each session
  starts a new block; a block is a run of task trials with the same block number in one session.

A window counts only once it is complete: fewer than 50 task trials, or no start of a fourth
block (the third may still be running), leave it ``None``, and the mouse is left out of the
summary of that problem (:attr:`MousePortFinding.reasons` says why). So is a mouse without
port-finding trials.

Both poke kinds of the poke analysis are counted, with its rules
(:mod:`src.behavior_analysis.poke_counts`): ``"after_initiation"`` (long pokes from initiation to
the choice, leading pokes at the initiation port dropped) and ``"after_choice"`` (port visits
in the ITI until the next initiation, leading visits to the chosen port dropped). Dropped trials
add no pokes but count towards the 50 trials and the blocks.

Across mice the maps are averaged as percentages (:func:`mean_percent_map`): each mouse's counts
are divided by its own total, so a mouse with many trials weighs as much as one with few.

Nothing here reads files or draws.
"""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.behavior_analysis.poke_counts import (
    KINDS,
    count_ports,
    previous_choice_towers_for_problem,
    previous_problem_choice_towers,
    problem_choice_ports,
)
from src.behavior_analysis.poke_events import (
    ARMS9,
    PokeParseError,
    TrialPokes,
    is_port_finding,
    parse_session,
    pokes_after_choice,
    pokes_after_initiation,
)
from src.behavior_analysis.port_finding_sessions import (
    PortFindingAssignment,
    port_finding_sessions_for_problem,
)
from src.behavior_import.problem_numbering import DEFAULT_INITIATION_TOWER
from src.behavior_import.session_order import ordered_session_keys
from src.behavior_import.task_types import number_within_task_type, regroup_by_task_type

N_FIRST_TRIALS = 50
N_FIRST_BLOCKS = 3
PORT_FINDING = "port_finding"
FIRST_TRIALS = "first_trials"
FIRST_BLOCKS = "first_blocks"
WINDOWS = (PORT_FINDING, FIRST_TRIALS, FIRST_BLOCKS)

NO_PORT_FINDING = "no port-finding trials"


class WindowPokes(NamedTuple):
    """The per-port poke counts of one window.

    Attributes:
        after_initiation (dict[str, int]): Long pokes after initiation, ``{port: count}``.
        after_choice (dict[str, int]): Port visits after the choice, ``{port: count}``.
        n_trials (int): Trials in the window, dropped ones included.
    """

    after_initiation: dict[str, int]
    after_choice: dict[str, int]
    n_trials: int


class MousePortFinding(NamedTuple):
    """One mouse's three windows in one problem, with its ports.

    Attributes:
        mouse (str): Mouse id.
        current_ports (frozenset[str]): The problem's choice ports.
        previous_ports (frozenset[str] | None): The choice ports of the mouse's previous problem;
            ``None`` for its first problem.
        initiation_port (str): The initiation port.
        n_port_finding_trials (int): Port-finding trials counted, dropped ones included.
        windows (dict[str, WindowPokes | None]): ``{window: counts}`` for the three
            :data:`WINDOWS`; ``None`` for a window that is not complete.
        reasons (tuple[str, ...]): Why the mouse cannot be summarised; empty when it can.
    """

    mouse: str
    current_ports: frozenset[str]
    previous_ports: frozenset[str] | None
    initiation_port: str
    n_port_finding_trials: int
    windows: dict[str, WindowPokes | None]
    reasons: tuple[str, ...]

    @property
    def layout(self) -> tuple[tuple[str, ...], tuple[str, ...], str]:
        """``(current ports, previous ports, initiation port)``, ports sorted; ``()`` for none."""
        return (
            tuple(sorted(self.current_ports)),
            tuple(sorted(self.previous_ports or ())),
            self.initiation_port,
        )


class TaskTrial(NamedTuple):
    """A task trial with where it sits: its session, block number and initiation port."""

    session: str
    block: int | None
    pokes: TrialPokes
    initiation: str


# ---------------------------------------------------------------------------- parsing


def _initiation(session: Mapping) -> str:
    return session.get("initiation_tower") or DEFAULT_INITIATION_TOWER


def _parse(label: str, session: Mapping) -> list[TrialPokes] | None:
    """The session's parsed trials; ``None`` (with a warning) if its raw events cannot be read."""
    try:
        return parse_session(session, _initiation(session), label=label)
    except PokeParseError as error:
        warnings.warn(
            f"{label}: skipped in the port-finding summary: {error}", UserWarning, stacklevel=3
        )
        return None


def _session_task_trials(
    mouse: str, key: str, session: Mapping, trials: list[TrialPokes]
) -> list[TaskTrial]:
    """The printed task trials of one parsed session, each with its block number.

    The block numbers come from the session's ``"blocks"``, one per printed trial. If it is
    missing or has another length (the importer drops prints without a block number) the block
    numbers of the session are ``None``, with a warning.
    """
    task = [t for t in trials if t.has_print and not is_port_finding(t)]
    blocks = list(session.get("blocks") or [])
    if len(blocks) != len(task):
        warnings.warn(
            f"{mouse} {key}: {len(blocks)} block numbers for {len(task)} task trials; the "
            "port-finding summary does not know its blocks",
            UserWarning,
            stacklevel=3,
        )
        blocks = [None] * len(task)
    initiation = _initiation(session)
    return [TaskTrial(key, block, t, initiation) for block, t in zip(blocks, task, strict=True)]


# ---------------------------------------------------------------------------- windows


def window_pokes(trials: Iterable[tuple[TrialPokes, str]]) -> WindowPokes:
    """Per-port counts of both poke kinds over *trials*, with the poke analysis's rules.

    Args:
        trials (Iterable[tuple[TrialPokes, str]]): ``(trial, initiation port)`` pairs.

    Returns:
        WindowPokes: The counts; dropped trials add no pokes, and a trial whose after-choice
        window never closed adds none to ``after_choice``, but every trial is in ``n_trials``.
    """
    trials = list(trials)
    valid = [(t, init) for t, init in trials if t.drop_reason is None]
    return WindowPokes(
        after_initiation=count_ports(pokes_after_initiation(t.long_pokes, i) for t, i in valid),
        after_choice=count_ports(
            pokes_after_choice(t.window_pokes, t.chosen)
            for t, _ in valid
            if t.window_pokes is not None
        ),
        n_trials=len(trials),
    )


def first_trials(task: list[TaskTrial], n: int = N_FIRST_TRIALS) -> list[TaskTrial] | None:
    """The first *n* task trials, or ``None`` if there are fewer."""
    return task[:n] if len(task) >= n else None


def _starts_block(task: list[TaskTrial], i: int) -> bool | None:
    """Whether trial *i* starts a block; ``None`` if that is unknown.

    A block starts at the first trial, at every new session (the importer restarts the block
    numbering in each session) and where the block number changes within a session; inside a
    session an unknown block number leaves it unknown.
    """
    if i == 0 or task[i].session != task[i - 1].session:
        return True
    if task[i].block is None or task[i - 1].block is None:
        return None
    return task[i].block != task[i - 1].block


def first_blocks(
    task: list[TaskTrial], n: int = N_FIRST_BLOCKS, *, more_after: bool = False
) -> list[TaskTrial] | None:
    """The task trials of the first *n* blocks; ``None`` if block *n* is not known to be over.

    Block *n* is over once block *n + 1* has started. Block numbers are needed only up to there:
    an unknown one inside a session before that point gives ``None``, later ones do not matter.

    Args:
        task (list[TaskTrial]): The problem's task trials in order.
        n (int): Number of blocks. Default :data:`N_FIRST_BLOCKS`.
        more_after (bool): A later session of the problem exists that is not in *task* (it could
            not be read); it starts a new block, so block *n* is over if *task* ends in it.
            Default False. Keyword-only.

    Returns:
        list[TaskTrial] | None: The trials of blocks 1 to *n*, or ``None``.
    """
    n_started = 0
    for i in range(len(task)):
        starts = _starts_block(task, i)
        if starts is None:
            return None
        if starts:
            n_started += 1
            if n_started == n + 1:
                return task[:i]
    return task if more_after and n_started == n else None


# ---------------------------------------------------------------------------- one mouse

WINDOW_TEXT = {
    PORT_FINDING: "port-finding",
    FIRST_TRIALS: "first-trials",
    FIRST_BLOCKS: "first-blocks",
}


class _ProblemTrials(NamedTuple):
    """The parsed trials of one mouse and problem, as :func:`mouse_port_finding_summary` uses them.

    ``task`` stops before the first session that could not be read (``truncated``);
    ``port_finding_lost`` says why port-finding trials may be missing, or is ``None``.
    """

    port_finding: list[tuple[TrialPokes, str]]
    task: list[TaskTrial]
    truncated: str | None
    port_finding_lost: str | None


def _problem_trials(
    mouse: str, sessions: Mapping[str, Mapping], found: Mapping[str, Mapping]
) -> _ProblemTrials:
    """Parse the port-finding-only sessions and the problem's sessions, oldest first."""
    port_finding: list[tuple[TrialPokes, str]] = []
    lost = None
    for key in ordered_session_keys(mouse, found):
        parsed = _parse(f"{mouse} {key}", found[key])
        if parsed is None:
            lost = lost or f"port-finding-only session {key} could not be read"
            continue
        port_finding += [(t, _initiation(found[key])) for t in parsed if is_port_finding(t)]
    task: list[TaskTrial] = []
    truncated = None
    for key in ordered_session_keys(mouse, sessions):
        parsed = _parse(f"{mouse} {key}", sessions[key])
        if parsed is None:
            if not task:  # port finding happens before the first task trial
                lost = lost or f"session {key} could not be read before port finding ended"
            truncated = truncated or key
            continue
        port_finding += [(t, _initiation(sessions[key])) for t in parsed if is_port_finding(t)]
        if truncated is None:
            task += _session_task_trials(mouse, key, sessions[key], parsed)
    return _ProblemTrials(port_finding, task, truncated, lost)


def _window_reason(window: str, n_trials: int, n_blocks: int, truncated: str | None) -> str:
    if truncated:
        return (
            f"session {truncated} could not be read before the {WINDOW_TEXT[window]} window "
            "was complete"
        )
    if window == FIRST_TRIALS:
        return f"fewer than {n_trials} task trials"
    return f"block {n_blocks} not complete, or its block numbers unknown"


def mouse_port_finding_summary(
    mouse: str,
    sessions: Mapping[str, Mapping],
    *,
    previous_ports: Iterable[str] | None,
    port_finding_sessions: Mapping[str, Mapping] | None = None,
    n_trials: int = N_FIRST_TRIALS,
    n_blocks: int = N_FIRST_BLOCKS,
) -> MousePortFinding:
    """The three windows of one mouse in one problem (see the module doc).

    Args:
        mouse (str): Mouse id, used in warnings.
        sessions (Mapping[str, Mapping]): ``{session_key: session_dict}`` of the problem, each
            with ``"df"`` and ``"choice_towers"`` (and ``"blocks"`` for the block window).
        previous_ports (Iterable[str] | None): Choice ports of the mouse's previous problem, or
            ``None`` for its first. Keyword-only.
        port_finding_sessions (Mapping[str, Mapping] | None): ``{session_key: record}`` of the
            port-finding-only sessions that belong to the problem. Default ``None``: none.
        n_trials (int): Trials in the first-trials window. Default :data:`N_FIRST_TRIALS`.
        n_blocks (int): Blocks in the first-blocks window. Default :data:`N_FIRST_BLOCKS`.

    Returns:
        MousePortFinding: The counts. A session whose raw events cannot be read is skipped with
        a warning: a task window that would reach past it is ``None``, and so is the port-finding
        window if the session may hold port-finding trials (a port-finding-only session, or a
        problem session before the first task trial).

    Raises:
        ValueError: If the session keys cannot be ordered, or a session has no choice towers.
        PokeAlignmentError: If a session's parsed trials do not match its imported trials.
    """
    trials = _problem_trials(mouse, sessions, port_finding_sessions or {})
    chosen = {
        FIRST_TRIALS: first_trials(trials.task, n_trials),
        FIRST_BLOCKS: first_blocks(trials.task, n_blocks, more_after=trials.truncated is not None),
    }
    port_finding_ok = trials.port_finding and trials.port_finding_lost is None
    windows = {PORT_FINDING: window_pokes(trials.port_finding) if port_finding_ok else None}
    windows |= {
        name: None if found is None else window_pokes((t.pokes, t.initiation) for t in found)
        for name, found in chosen.items()
    }
    reasons = [trials.port_finding_lost] if trials.port_finding_lost else []
    if not trials.port_finding and not reasons:
        reasons.append(NO_PORT_FINDING)
    reasons += [
        _window_reason(name, n_trials, n_blocks, trials.truncated)
        for name, found in chosen.items()
        if found is None
    ]
    return MousePortFinding(
        mouse=mouse,
        current_ports=problem_choice_ports(sessions),
        previous_ports=None if previous_ports is None else frozenset(previous_ports),
        initiation_port=_problem_initiation(mouse, sessions),
        n_port_finding_trials=len(trials.port_finding),
        windows=windows,
        reasons=tuple(reasons),
    )


def _problem_initiation(mouse: str, sessions: Mapping[str, Mapping]) -> str:
    """The problem's one initiation port (the protocol default for a mouse with no sessions)."""
    towers = {_initiation(s) for s in sessions.values()} or {DEFAULT_INITIATION_TOWER}
    if len(towers) != 1:
        raise ValueError(f"{mouse}: sessions of one problem use initiation towers {sorted(towers)}")
    return towers.pop()


# ---------------------------------------------------------------------------- problems


def problem_port_finding_summaries(
    subjects_trials: Mapping[str, Mapping[str, Mapping]],
    *,
    previous_choice_towers: Mapping[str, Iterable[str] | None] | None = None,
    port_finding_sessions: Mapping[str, Mapping[str, Mapping]] | None = None,
) -> dict[str, MousePortFinding]:
    """:func:`mouse_port_finding_summary` for every mouse of one problem.

    Args:
        subjects_trials (Mapping): ``{mouse: {session_key: session_dict}}`` of the problem.
        previous_choice_towers (Mapping | None): ``{mouse: previous-problem ports or None}``
            (see ``previous_choice_towers_for_problem``). Default ``None``: no mouse has a
            previous problem. Keyword-only.
        port_finding_sessions (Mapping | None): ``{mouse: {session_key: record}}`` of the
            problem's port-finding-only sessions. Default ``None``: none. Keyword-only.

    Returns:
        dict[str, MousePortFinding]: ``{mouse: summary}``, mice sorted.

    Raises:
        ValueError: If *previous_choice_towers* is given but misses a mouse.
    """
    previous = previous_choice_towers or {}
    if previous_choice_towers is not None:
        missing = sorted(set(subjects_trials) - set(previous))
        if missing:
            raise ValueError(f"previous_choice_towers has no entry for mice {missing}")
    found = port_finding_sessions or {}
    return {
        mouse: mouse_port_finding_summary(
            mouse,
            subjects_trials[mouse],
            previous_ports=previous.get(mouse),
            port_finding_sessions=found.get(mouse),
        )
        for mouse in sorted(subjects_trials)
    }


def cohort_port_finding_summaries(
    trials_by_problem: Mapping,
    assignment: PortFindingAssignment,
    *,
    task_types: Iterable[str] | None = None,
) -> dict[tuple[str, int], dict[str, MousePortFinding]]:
    """The summaries of the problems of a cohort, keyed like ``regroup_by_task_type``.

    Args:
        trials_by_problem (Mapping): ``{overall problem: {mouse: sessions}}`` of the whole
            cohort (the previous problem of a mouse is looked up in it).
        assignment (PortFindingAssignment): The cohort's port-finding-only sessions, from
            ``assign_port_finding_sessions``.
        task_types (Iterable[str] | None): Only the problems of these task types, e.g.
            ``["drifting"]``. Default ``None``: all. Keyword-only.

    Returns:
        dict: ``{(task type, k): {mouse: MousePortFinding}}``.
    """
    wanted = None if task_types is None else set(task_types)
    previous = previous_problem_choice_towers(trials_by_problem)
    grouped = regroup_by_task_type(trials_by_problem, number_within_task_type(trials_by_problem))
    return {
        key: problem_port_finding_summaries(
            subjects,
            previous_choice_towers=previous_choice_towers_for_problem(subjects, previous),
            port_finding_sessions=port_finding_sessions_for_problem(assignment, subjects),
        )
        for key, subjects in grouped.items()
        if wanted is None or key[0] in wanted
    }


# ---------------------------------------------------------------------------- averaging


def percent_of_pokes(counts: Mapping[str, float]) -> dict[str, float] | None:
    """Each port's share of all pokes, in percent; ``None`` if there are no pokes."""
    total = sum(counts[port] for port in ARMS9)
    return None if total == 0 else {port: 100.0 * counts[port] / total for port in ARMS9}


def exclusion_reasons(summary: MousePortFinding, kind: str) -> tuple[str, ...]:
    """Why *summary* is left out of the maps of *kind*; empty if it is in them.

    Raises:
        ValueError: If *kind* is not one of ``KINDS``.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    empty = [
        f"no {kind.replace('_', '-')} pokes in the {WINDOW_TEXT[name]} window"
        for name, window in summary.windows.items()
        if window is not None and percent_of_pokes(getattr(window, kind)) is None
    ]
    return summary.reasons + tuple(empty)


def mean_percent_map(
    summaries: Iterable[MousePortFinding],
    window: str,
    kind: str,
    mice: Sequence[Hashable] | None = None,
) -> dict[str, float]:
    """The mean across mice of each port's percentage of pokes in one window.

    Args:
        summaries (Iterable[MousePortFinding]): Mice without exclusion reasons for *kind*.
        window (str): One of :data:`WINDOWS`.
        kind (str): ``"after_initiation"`` or ``"after_choice"``.
        mice (Sequence[Hashable] | None): The mouse of each summary, when a mouse may have several
            (e.g. ``(cohort, mouse)`` over problems with the same layout): its maps are averaged
            first, so every mouse weighs once. Default ``None``: one summary per mouse.

    Returns:
        dict[str, float]: ``{port: mean percent}`` over the nine ports.

    Raises:
        ValueError: If there are no summaries, *mice* has another length, or a summary has
            exclusion reasons for *kind*.
    """
    summaries = list(summaries)
    if not summaries:
        raise ValueError("no mice to average")
    mice = list(range(len(summaries))) if mice is None else list(mice)
    if len(mice) != len(summaries):
        raise ValueError(f"{len(mice)} mice for {len(summaries)} summaries")
    excluded = [s.mouse for s in summaries if exclusion_reasons(s, kind)]
    if excluded:
        raise ValueError(f"mice {excluded} cannot be averaged for {kind}")
    by_mouse: dict = {}
    for mouse, s in zip(mice, summaries, strict=True):
        by_mouse.setdefault(mouse, []).append(percent_of_pokes(getattr(s.windows[window], kind)))
    per_mouse = [{p: np.mean([m[p] for m in maps]) for p in ARMS9} for maps in by_mouse.values()]
    return {port: float(np.mean([m[port] for m in per_mouse])) for port in ARMS9}


def group_by_layout(keys_and_summaries: Iterable[tuple[object, MousePortFinding]]) -> dict:
    """``{layout: [key, ...]}`` of the summaries, layouts with more mice first.

    Args:
        keys_and_summaries (Iterable[tuple[object, MousePortFinding]]): ``(key, summary)`` pairs;
            the key names the mouse (e.g. the mouse id, or ``(cohort, k, mouse)``).

    Returns:
        dict: ``{(current ports, previous ports, initiation port): keys}``, ordered by the number
        of mice (most first), then by layout.
    """
    groups: dict = {}
    for key, summary in keys_and_summaries:
        groups.setdefault(summary.layout, []).append(key)
    return dict(sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])))


# ---------------------------------------------------------------------------- port categories

CURRENT = "current"
PREVIOUS = "previous"
INITIATION = "initiation"
OTHER = "other"
CATEGORIES = (CURRENT, PREVIOUS, INITIATION, OTHER)


def port_category(
    port: str, current: Iterable[str], previous: Iterable[str] | None, initiation: str
) -> str:
    """Which of :data:`CATEGORIES` a port is in for one problem.

    A port in several categories takes the first that applies, in the order current, previous,
    initiation: a previous-problem port that is also a current one is a current port (poking it
    is a valid choice now).
    """
    if port in set(current):
        return CURRENT
    if port in set(previous or ()):
        return PREVIOUS
    return INITIATION if port == initiation else OTHER


def category_fractions(summary: MousePortFinding, window: str, kind: str) -> dict[str, float]:
    """The fraction of the mouse's pokes of *kind* in *window* at each category of port.

    A mouse in its first problem has no previous ports: its previous fraction is ``nan``, so it
    is left out of that category's mean rather than counted as 0.

    Raises:
        ValueError: If the window is incomplete or has no pokes of *kind*.
    """
    counts = summary.windows[window]
    if counts is None:
        raise ValueError(f"{summary.mouse}: the {window} window is not complete")
    counts = getattr(counts, kind)
    total = sum(counts.values())
    if total == 0:
        raise ValueError(f"{summary.mouse}: no {kind} pokes in the {window} window")
    fractions = dict.fromkeys(CATEGORIES, 0.0)
    for port in ARMS9:
        category = port_category(
            port, summary.current_ports, summary.previous_ports, summary.initiation_port
        )
        fractions[category] += counts[port] / total
    if summary.previous_ports is None:
        fractions[PREVIOUS] = float("nan")
    return fractions


def category_fraction_table(entries: Iterable[tuple[Mapping, MousePortFinding]]) -> pd.DataFrame:
    """One row per mouse, problem, poke kind, window and port category, with the fraction.

    Args:
        entries (Iterable[tuple[Mapping, MousePortFinding]]): ``(labels, summary)`` pairs;
            *labels* (e.g. ``{"cohort": ..., "task_type": ..., "k": ...}``) are copied into each
            row. Summaries with exclusion reasons for a kind add no rows of that kind.

    Returns:
        pd.DataFrame: The label columns, then ``mouse, kind, window, category, fraction``
        (``nan`` for the previous category of a mouse's first problem).
    """
    rows = [
        {**labels, "mouse": s.mouse, "kind": kind, "window": window, "category": c, "fraction": f}
        for labels, s in entries
        for kind in KINDS
        if not exclusion_reasons(s, kind)
        for window in WINDOWS
        for c, f in category_fractions(s, window, kind).items()
    ]
    columns = ["mouse", "kind", "window", "category", "fraction"]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)


def mouse_mean_fractions(
    table: pd.DataFrame, mouse_columns: Sequence[str] = ("mouse",)
) -> pd.DataFrame:
    """Each mouse's fractions averaged over its problems, with the number of problems used.

    Args:
        table (pd.DataFrame): Output of :func:`category_fraction_table`.
        mouse_columns (Sequence[str]): Columns that name a mouse (add ``"cohort"`` when mice of
            several cohorts are pooled). Default ``("mouse",)``.

    Returns:
        pd.DataFrame: ``*mouse_columns, kind, window, category, fraction, n_problems``;
        ``n_problems`` counts the problems with a value (a ``nan`` previous fraction is
        skipped), and a mouse with none has ``fraction`` ``nan``.
    """
    keys = [*mouse_columns, "kind", "window", "category"]
    grouped = table.groupby(keys, sort=False)["fraction"]
    return grouped.agg(fraction="mean", n_problems="count").reset_index()
