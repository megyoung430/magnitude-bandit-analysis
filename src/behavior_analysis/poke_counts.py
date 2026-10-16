"""Per-port poke counts for sessions, mice and problems, and each mouse's previous-problem ports.

Built on :mod:`src.behavior_analysis.poke_events`: that module reads the raw event log, this one
applies its counting rules to every trial and sums the result. Nothing here reads files or draws.
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import NamedTuple

from src.behavior_analysis.poke_events import (
    ARMS9,
    NO_PRINT,
    LongPokeCheck,
    PokeParseError,
    TrialPokes,
    cross_check_long_pokes,
    parse_session,
    pokes_after_choice,
    pokes_after_initiation,
)
from src.behavior_analysis.problem_utils import problem_to_int
from src.behavior_import.problem_numbering import DEFAULT_INITIATION_TOWER
from src.behavior_import.session_order import ordered_session_keys

KINDS = ("after_initiation", "after_choice")


class SessionPokeCounts(NamedTuple):
    """Poke counts of one session.

    ``after_initiation`` and ``after_choice`` map each of the nine ports to a count.
    ``n_trials`` is the number of trials with a print row (the session's trials);
    ``n_valid`` how many of them were analysed (the others were dropped, see
    :mod:`src.behavior_analysis.poke_events`); ``n_open_windows`` how many analysed trials have
    no closed after-choice window (so are missing from ``after_choice``).
    ``n_dropped_no_print`` counts trials dropped for having no usable summary print (including
    aborted trials that have no print row, which are not in ``n_trials``) and
    ``n_dropped_inconsistent`` the other drops. ``long_poke_check`` compares the raw long
    pokes of the analysed trials with the printed lists. ``parse_error`` is the reason the
    session's raw events could not be read (all counts are then zero), else ``None``.
    """

    after_initiation: dict[str, int]
    after_choice: dict[str, int]
    n_trials: int
    n_open_windows: int
    long_poke_check: LongPokeCheck
    n_valid: int
    n_dropped_no_print: int
    n_dropped_inconsistent: int
    parse_error: str | None = None


def count_ports(sequences: Iterable[Iterable[str]]) -> dict[str, int]:
    """Number of occurrences of each of the nine ports over all *sequences* (repeats count)."""
    counter = Counter(port for sequence in sequences for port in sequence)
    unknown = set(counter) - set(ARMS9)
    if unknown:
        raise ValueError(f"unknown ports {sorted(unknown)}; expected {ARMS9}")
    return {port: counter[port] for port in ARMS9}


def _after_initiation_sequences(trials: list[TrialPokes], initiation_tower: str) -> list[list[str]]:
    return [
        pokes_after_initiation(t.long_pokes, initiation_tower)
        for t in trials
        if t.drop_reason is None
    ]


def _after_choice_sequences(trials: list[TrialPokes]) -> list[list[str]]:
    return [
        pokes_after_choice(t.window_pokes, t.chosen)
        for t in trials
        if t.drop_reason is None and t.window_pokes is not None
    ]


def session_poke_counts(session: dict, *, label: str = "session") -> SessionPokeCounts:
    """Both per-port poke counts of one session, from its raw recordings.

    Args:
        session: A session record with the importer's ``"df"`` (and, optionally,
            ``"initiation_tower"``, default ``"B2"``, and ``"choice"`` to check alignment).
        label: Names the session in error messages.

    Returns:
        A :class:`SessionPokeCounts`. Dropped trials are left out of both counts; trials
        whose after-choice window never closed (a recording's last printed trial) are left out of
        ``after_choice`` only.
    """
    initiation = session.get("initiation_tower") or DEFAULT_INITIATION_TOWER
    trials = parse_session(session, initiation, label=label)
    valid = [t for t in trials if t.drop_reason is None]
    return SessionPokeCounts(
        after_initiation=count_ports(_after_initiation_sequences(trials, initiation)),
        after_choice=count_ports(_after_choice_sequences(trials)),
        n_trials=sum(t.has_print for t in trials),
        n_open_windows=sum(t.window_pokes is None for t in valid),
        long_poke_check=cross_check_long_pokes(trials),
        n_valid=len(valid),
        n_dropped_no_print=sum(t.drop_reason == NO_PRINT for t in trials),
        n_dropped_inconsistent=sum(t.drop_reason not in (None, NO_PRINT) for t in trials),
    )


def unparsed_session_counts(reason: str) -> SessionPokeCounts:
    """Zero counts for a session whose raw events could not be read, carrying *reason*."""
    zeros = count_ports([])
    return SessionPokeCounts(
        after_initiation=dict(zeros),
        after_choice=dict(zeros),
        n_trials=0,
        n_open_windows=0,
        long_poke_check=LongPokeCheck(0, 0, []),
        n_valid=0,
        n_dropped_no_print=0,
        n_dropped_inconsistent=0,
        parse_error=reason,
    )


def _counts_or_skip(session: dict, label: str) -> SessionPokeCounts:
    try:
        return session_poke_counts(session, label=label)
    except PokeParseError as error:
        warnings.warn(f"{label}: skipped in the poke analysis: {error}", UserWarning, stacklevel=3)
        return unparsed_session_counts(str(error))


def mouse_poke_counts(subject: str, sessions: Mapping[str, dict]) -> dict[str, SessionPokeCounts]:
    """:func:`session_poke_counts` for each session of a mouse, in session order.

    A session whose raw events raise :class:`PokeParseError` is skipped with a warning and
    gets :func:`unparsed_session_counts` (zero counts, the reason in ``parse_error``).

    Raises:
        ValueError: If the session keys cannot be ordered (see ``ordered_session_keys``).
        PokeAlignmentError: If a session's parsed trials do not match its imported trials.
    """
    return {
        key: _counts_or_skip(sessions[key], f"{subject} {key}")
        for key in ordered_session_keys(subject, sessions)
    }


def total_counts(per_session: Mapping[str, SessionPokeCounts], kind: str) -> dict[str, int]:
    """Per-port sum over sessions of *kind*, ``"after_initiation"`` or ``"after_choice"``."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    return {port: sum(getattr(c, kind)[port] for c in per_session.values()) for port in ARMS9}


# ---------------------------------------------------------------- choice ports


def session_choice_ports(session: Mapping) -> frozenset[str]:
    """The choice ports of a session.

    The importer stores ``"choice_towers"`` either as one set of ports or as a list holding the
    ports available on each trial; both give the union of the ports.

    Raises:
        ValueError: If the session has no choice towers.
    """
    value = session.get("choice_towers") or ()
    ports: set[str] = set()
    for item in value:
        if isinstance(item, str):
            ports.add(item)
        else:
            ports.update(item)
    if not ports:
        raise ValueError(f"session has no choice towers: {session.get('choice_towers')!r}")
    return frozenset(ports)


def problem_choice_ports(sessions: Mapping[str, dict]) -> frozenset[str]:
    """Union of the choice ports over the sessions of one mouse and problem."""
    return frozenset().union(*(session_choice_ports(s) for s in sessions.values()))


def previous_problem_choice_towers(
    subjects_trials_by_problem: Mapping,
) -> dict[str, dict[int, frozenset[str] | None]]:
    """Each mouse's choice ports in the problem before every problem it ran.

    "Before" is the mouse's previous problem in overall problem-number order (whatever its task
    type), not in dict order; a problem number the mouse did not run is skipped.

    Args:
        subjects_trials_by_problem: ``{problem: {mouse: {session_key: session_dict}}}`` for all
            problems, each session carrying ``"choice_towers"``.

    Returns:
        ``{mouse: {overall problem number: ports of its previous problem, or None for its first
        problem}}``.
    """
    ports_by_mouse: dict[str, dict[int, frozenset[str]]] = {}
    for problem, subjects in subjects_trials_by_problem.items():
        number = problem_to_int(problem)
        if number is None:
            raise ValueError(f"problem key {problem!r} has no number")
        for mouse, sessions in subjects.items():
            ports_by_mouse.setdefault(mouse, {})[number] = problem_choice_ports(sessions)
    previous: dict[str, dict[int, frozenset[str] | None]] = {}
    for mouse, by_problem in ports_by_mouse.items():
        numbers = sorted(by_problem)
        previous[mouse] = {
            number: by_problem[before] if before is not None else None
            for number, before in zip(numbers, [None, *numbers[:-1]], strict=True)
        }
    return previous


def previous_choice_towers_for_problem(
    subjects_sessions: Mapping[str, Mapping[str, dict]],
    previous: Mapping[str, Mapping[int, frozenset[str] | None]],
) -> dict[str, frozenset[str] | None]:
    """``{mouse: previous-problem ports}`` for one problem's mice, as `run_poke_analysis` takes.

    A mouse's overall problem number is read from its sessions' ``"problem"``, so this also
    serves a problem regrouped by task type, where mice may sit at different overall numbers.

    Args:
        subjects_sessions: ``{mouse: {session_key: session_dict}}`` of one problem.
        previous: The result of :func:`previous_problem_choice_towers` for the whole cohort.

    Raises:
        ValueError: If a mouse's sessions name more than one problem, or no problem.
        ValueError: If the mouse or its problem is not in *previous*.
    """
    result = {}
    for mouse, sessions in subjects_sessions.items():
        numbers = {s.get("problem") for s in sessions.values()}
        if len(numbers) != 1 or None in numbers:
            raise ValueError(
                f"{mouse}: sessions of one problem must name exactly one problem number, "
                f"got {sorted(map(str, numbers))}; found more than one problem or none"
            )
        problem = problem_to_int(next(iter(numbers)))
        if problem not in previous.get(mouse, {}):
            raise ValueError(
                f"previous has no entry for {mouse} problem {problem}; build it with "
                "previous_problem_choice_towers from all problems that include this one"
            )
        result[mouse] = previous[mouse][problem]
    return result
