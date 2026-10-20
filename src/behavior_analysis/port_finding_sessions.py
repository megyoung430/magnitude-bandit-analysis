"""Which problem a port-finding-only session belongs to.

A mouse that does not finish port finding in one session starts the next session in port finding
again, and the task remembers the ports already found. The first session has no bandit trials, so
the importer keeps it out of the trials by problem (see
:attr:`~src.behavior_import.import_log.ImportLog.port_finding_sessions`); its port-finding trials
still belong to the problem that follows, and the port-finding poke maps of that problem count
them.

The rule (:func:`assign_port_finding_sessions`): a port-finding-only session belongs to the problem
of the mouse's next session with bandit trials, if that session has the same choice towers (and
the same initiation tower) as the port-finding session. Several port-finding-only sessions in a
row before one problem all go to it. Any other session is not assigned: it is reported with a
warning and listed in :attr:`PortFindingAssignment.unassigned`.

Nothing here reads files or draws.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import NamedTuple

from src.behavior_analysis.poke_counts import session_choice_ports
from src.behavior_import.problem_numbering import DEFAULT_INITIATION_TOWER
from src.behavior_import.session_order import ordered_session_keys


class UnassignedPortFinding(NamedTuple):
    """A port-finding-only session that no problem could take.

    Attributes:
        mouse (str): Mouse id.
        session (str): Session key.
        towers (frozenset[str]): The towers the session's port finding ran on.
        reason (str): Why no problem took it, in plain words.
    """

    mouse: str
    session: str
    towers: frozenset[str]
    reason: str


class PortFindingAssignment(NamedTuple):
    """The result of :func:`assign_port_finding_sessions`.

    Attributes:
        targets (dict[str, dict[str, str]]): ``{mouse: {port-finding session: the session with
            bandit trials whose problem it belongs to}}``, sessions oldest first.
        records (dict[str, dict[str, dict]]): ``{mouse: {session: record}}`` of the assigned
            port-finding-only sessions.
        unassigned (list[UnassignedPortFinding]): The sessions no problem took.
    """

    targets: dict[str, dict[str, str]]
    records: dict[str, dict[str, dict]]
    unassigned: list[UnassignedPortFinding]


def _initiation(session: Mapping) -> str:
    return session.get("initiation_tower") or DEFAULT_INITIATION_TOWER


def _towers_text(towers) -> str:
    return ", ".join(sorted(towers)) or "none"


def _mismatch(record: Mapping, later_key: str, later: Mapping) -> str | None:
    """Why *record* is not the port finding of the session *later*, else ``None``."""
    later_towers = session_choice_ports(later)
    if frozenset(record["choice_towers"]) != later_towers:
        return (
            f"the next session with bandit trials, {later_key}, has other towers "
            f"({_towers_text(later_towers)})"
        )
    if _initiation(record) != _initiation(later):
        return (
            f"the next session with bandit trials, {later_key}, has another initiation tower "
            f"({_initiation(later)}, not {_initiation(record)})"
        )
    return None


def _blocker(
    key: str, later_key: str, order: list[str], found: Mapping[str, Mapping]
) -> str | None:
    """A port-finding-only session between *key* and *later_key* on other towers, else ``None``.

    Port finding of other towers in between means the mouse moved on and *key* was never finished
    on the towers of *later_key*.
    """
    record = found[key]
    between = order[order.index(key) + 1 : order.index(later_key)]
    for other in (k for k in between if k in found):
        same_towers = frozenset(found[other]["choice_towers"]) == frozenset(record["choice_towers"])
        if not same_towers or _initiation(found[other]) != _initiation(record):
            return other
    return None


def _assign_mouse(
    mouse: str, bandit: Mapping[str, Mapping], found: Mapping[str, Mapping]
) -> tuple[dict[str, str], list[UnassignedPortFinding]]:
    """``({port-finding session: target session}, unassigned)`` for one mouse."""
    overlap = sorted(set(bandit) & set(found))
    if overlap:
        raise ValueError(f"{mouse}: sessions {overlap} are both port-finding-only and in a problem")
    order = ordered_session_keys(mouse, [*bandit, *found])
    next_bandit: str | None = None
    next_of: dict[str, str | None] = {}
    for key in reversed(order):  # the next session with bandit trials after each session
        if key in bandit:
            next_bandit = key
        else:
            next_of[key] = next_bandit
    targets, unassigned = {}, []
    for key in (k for k in order if k in found):
        record, later_key = found[key], next_of[key]
        if later_key is None:
            reason = "there is no later session with bandit trials"
        else:
            blocker = _blocker(key, later_key, order, found)
            reason = _mismatch(record, later_key, bandit[later_key]) or (
                None
                if blocker is None
                else f"another port-finding-only session, {blocker}, with other towers lies "
                f"between it and {later_key}"
            )
        if reason is None:
            targets[key] = later_key
        else:
            unassigned.append(
                UnassignedPortFinding(mouse, key, frozenset(record["choice_towers"]), reason)
            )
    return targets, unassigned


def assign_port_finding_sessions(
    trials_by_problem: Mapping, port_finding_sessions: Mapping[str, Mapping[str, Mapping]]
) -> PortFindingAssignment:
    """Assign each port-finding-only session to the problem that follows it (see the module doc).

    Args:
        trials_by_problem (Mapping): ``{problem: {mouse: {session_key: session_dict}}}``, each
            session carrying ``"choice_towers"`` (and optionally ``"initiation_tower"``, default
            ``"B2"``).
        port_finding_sessions (Mapping[str, Mapping[str, Mapping]]): ``{mouse: {session_key:
            record}}`` as in ``ImportLog.port_finding_sessions``; each record carries
            ``"choice_towers"`` and optionally ``"initiation_tower"``.

    Returns:
        PortFindingAssignment: Where each session goes, and the ones that go nowhere. Every
        unassigned session also raises a ``UserWarning`` naming the mouse, the session and its
        towers.

    Raises:
        ValueError: If a mouse's session keys cannot be ordered (see ``ordered_session_keys``),
            a session is both port-finding-only and in a problem, or a session of a problem has no
            choice towers.
    """
    bandit_by_mouse: dict[str, dict[str, Mapping]] = {}
    for subjects in trials_by_problem.values():
        for mouse, sessions in subjects.items():
            bandit_by_mouse.setdefault(mouse, {}).update(sessions)
    targets, records, unassigned = {}, {}, []
    for mouse, found in port_finding_sessions.items():
        mouse_targets, mouse_unassigned = _assign_mouse(
            mouse, bandit_by_mouse.get(mouse, {}), found
        )
        if mouse_targets:
            targets[mouse] = mouse_targets
            records[mouse] = {key: found[key] for key in mouse_targets}
        unassigned.extend(mouse_unassigned)
    for item in unassigned:
        warnings.warn(
            f"{item.mouse} {item.session}: port-finding-only session (towers "
            f"{_towers_text(item.towers)}) is not used in the port-finding poke maps: "
            f"{item.reason}",
            UserWarning,
            stacklevel=2,
        )
    return PortFindingAssignment(targets, records, unassigned)


def port_finding_sessions_for_problem(
    assignment: PortFindingAssignment, subjects_sessions: Mapping[str, Mapping]
) -> dict[str, dict[str, dict]]:
    """The port-finding-only sessions assigned to the sessions of one problem.

    Looks up the sessions rather than a problem number, so it serves a problem regrouped by task
    type (where the number changes and the sessions do not).

    Args:
        assignment (PortFindingAssignment): The cohort's assignment.
        subjects_sessions (Mapping[str, Mapping]): ``{mouse: {session_key: session_dict}}`` of
            one problem.

    Returns:
        dict[str, dict[str, dict]]: ``{mouse: {session_key: record}}`` of the assigned sessions
        whose target session is in the problem, oldest first; mice with none are left out.
    """
    found = {}
    for mouse, sessions in subjects_sessions.items():
        mine = {
            key: assignment.records[mouse][key]
            for key, target in assignment.targets.get(mouse, {}).items()
            if target in sessions
        }
        if mine:
            found[mouse] = mine
    return found
