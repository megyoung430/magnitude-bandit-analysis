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
from collections.abc import Iterable, Mapping
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
    assign_port_finding_sessions,
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


def _session_task_trials(key: str, session: Mapping, trials: list[TrialPokes]) -> list[TaskTrial]:
    """The printed task trials of one parsed session, each with its block number.

    The block numbers come from the session's ``"blocks"``, one per printed trial; if it is
    missing or has another length they are ``None``.
    """
    task = [t for t in trials if t.has_print and not is_port_finding(t)]
    blocks = list(session.get("blocks") or [])
    if len(blocks) != len(task):
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


def block_starts(task: list[TaskTrial]) -> list[int] | None:
    """Indices of the trials that start a block; ``None`` if a block number is unknown.

    A block starts at the first trial, where the block number changes, and at every new session
    (the importer restarts the numbering in each session).
    """
    if any(t.block is None for t in task):
        return None
    return [
        i
        for i, t in enumerate(task)
        if i == 0 or (t.session, t.block) != (task[i - 1].session, task[i - 1].block)
    ]


def first_blocks(task: list[TaskTrial], n: int = N_FIRST_BLOCKS) -> list[TaskTrial] | None:
    """The task trials of the first *n* blocks; ``None`` if block *n* is not known to be over.

    Block *n* is over once block *n + 1* has started, so a mouse still in its *n*-th block (or
    one with unknown block numbers) gets ``None``.
    """
    starts = block_starts(task)
    if starts is None or len(starts) <= n:
        return None
    return task[: starts[n]]


# ---------------------------------------------------------------------------- one mouse


def _window_reason(window: str, n_trials: int, n_blocks: int, truncated: str | None) -> str:
    if truncated:
        return f"{truncated} could not be read before the {window} window was complete"
    if window == FIRST_TRIALS:
        return f"fewer than {n_trials} task trials"
    return f"block {n_blocks} not complete, or block numbers missing"


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
        a warning; a window that would reach past it is ``None``.

    Raises:
        ValueError: If the session keys cannot be ordered, or a session has no choice towers.
        PokeAlignmentError: If a session's parsed trials do not match its imported trials.
    """
    found = dict(port_finding_sessions or {})
    port_finding: list[tuple[TrialPokes, str]] = []
    for key in ordered_session_keys(mouse, found):
        parsed = _parse(f"{mouse} {key}", found[key])
        if parsed is not None:
            init = _initiation(found[key])
            port_finding += [(t, init) for t in parsed if is_port_finding(t)]

    task: list[TaskTrial] = []
    truncated = None  # the first session that could not be read
    for key in ordered_session_keys(mouse, sessions):
        parsed = _parse(f"{mouse} {key}", sessions[key])
        if parsed is None:
            truncated = truncated or f"session {key}"
            continue
        init = _initiation(sessions[key])
        port_finding += [(t, init) for t in parsed if is_port_finding(t)]
        if truncated is None:
            task += _session_task_trials(key, sessions[key], parsed)

    chosen = {
        FIRST_TRIALS: first_trials(task, n_trials),
        FIRST_BLOCKS: first_blocks(task, n_blocks),
    }
    windows = {PORT_FINDING: window_pokes(port_finding) if port_finding else None}
    windows |= {
        name: None if trials is None else window_pokes((t.pokes, t.initiation) for t in trials)
        for name, trials in chosen.items()
    }
    reasons = ([NO_PORT_FINDING] if not port_finding else []) + [
        _window_reason(name, n_trials, n_blocks, truncated)
        for name, trials in chosen.items()
        if trials is None
    ]
    return MousePortFinding(
        mouse=mouse,
        current_ports=problem_choice_ports(sessions),
        previous_ports=None if previous_ports is None else frozenset(previous_ports),
        initiation_port=_problem_initiation(mouse, sessions),
        n_port_finding_trials=len(port_finding),
        windows=windows,
        reasons=tuple(reasons),
    )


def _problem_initiation(mouse: str, sessions: Mapping[str, Mapping]) -> str:
    towers = {_initiation(s) for s in sessions.values()}
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
    trials_by_problem: Mapping, port_finding_log: Mapping[str, Mapping[str, Mapping]]
) -> dict[tuple[str, int], dict[str, MousePortFinding]]:
    """The summaries of every problem of a cohort, keyed like ``regroup_by_task_type``.

    Args:
        trials_by_problem (Mapping): ``{overall problem: {mouse: sessions}}`` of the whole
            cohort (the previous problem of a mouse is looked up in it).
        port_finding_log (Mapping): ``ImportLog.port_finding_sessions`` of the cohort.

    Returns:
        dict: ``{(task type, k): {mouse: MousePortFinding}}``.
    """
    previous = previous_problem_choice_towers(trials_by_problem)
    assignment = assign_port_finding_sessions(trials_by_problem, port_finding_log)
    grouped = regroup_by_task_type(trials_by_problem, number_within_task_type(trials_by_problem))
    return {
        key: problem_port_finding_summaries(
            subjects,
            previous_choice_towers=previous_choice_towers_for_problem(subjects, previous),
            port_finding_sessions=port_finding_sessions_for_problem(assignment, subjects),
        )
        for key, subjects in grouped.items()
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
        f"no {kind.replace('_', '-')} pokes in the {name.replace('_', '-')} window"
        for name, window in summary.windows.items()
        if window is not None and percent_of_pokes(getattr(window, kind)) is None
    ]
    return summary.reasons + tuple(empty)


def mean_percent_map(
    summaries: Iterable[MousePortFinding], window: str, kind: str
) -> dict[str, float]:
    """The mean across mice of each port's percentage of pokes in one window.

    Args:
        summaries (Iterable[MousePortFinding]): Mice without exclusion reasons for *kind*.
        window (str): One of :data:`WINDOWS`.
        kind (str): ``"after_initiation"`` or ``"after_choice"``.

    Returns:
        dict[str, float]: ``{port: mean percent}`` over the nine ports.

    Raises:
        ValueError: If there are no summaries, or one of them has exclusion reasons for *kind*.
    """
    summaries = list(summaries)
    if not summaries:
        raise ValueError("no mice to average")
    excluded = [s.mouse for s in summaries if exclusion_reasons(s, kind)]
    if excluded:
        raise ValueError(f"mice {excluded} cannot be averaged for {kind}")
    maps = [percent_of_pokes(getattr(s.windows[window], kind)) for s in summaries]
    return {port: float(np.mean([m[port] for m in maps])) for port in ARMS9}


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
    return fractions


def category_fraction_table(entries: Iterable[tuple[Mapping, MousePortFinding]]):
    """One row per mouse, problem, poke kind, window and port category, with the fraction.

    Args:
        entries (Iterable[tuple[Mapping, MousePortFinding]]): ``(labels, summary)`` pairs;
            *labels* (e.g. ``{"cohort": ..., "task_type": ..., "k": ...}``) are copied into each
            row. Summaries with exclusion reasons for a kind add no rows of that kind.

    Returns:
        pd.DataFrame: The label columns, then ``mouse, kind, window, category, fraction``.
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


def mouse_mean_fractions(table, mouse_columns=("mouse",)):
    """Each mouse's fractions averaged over its problems, with the number of problems.

    Args:
        table (pd.DataFrame): Output of :func:`category_fraction_table`.
        mouse_columns (Sequence[str]): Columns that name a mouse (add ``"cohort"`` when mice of
            several cohorts are pooled). Default ``("mouse",)``.

    Returns:
        pd.DataFrame: ``*mouse_columns, kind, window, category, fraction, n_problems``.
    """
    keys = [*mouse_columns, "kind", "window", "category"]
    grouped = table.groupby(keys, sort=False)["fraction"]
    return grouped.agg(fraction="mean", n_problems="size").reset_index()
