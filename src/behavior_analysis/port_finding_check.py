"""A sanity check on the number of port-finding trials of each mouse and problem.

The task counts how often each choice port was chosen during port finding (``found_port_counts``)
and ends port finding once every port reached ``found_t_thresh``. A mouse that went through port
finding from the start therefore has at least ``n_ports x found_t_thresh`` port-finding trials
(the printed ones: a trial that times out has no print and does not count towards finding). Fewer
means port finding began in a session that is missing from the data, or the counts were set by
hand (a ``user_set`` of ``found_port_counts``); the table flags those mouse-problems and says
which of the two it can see. It never raises on a flagged row.

Pure: it reads the raw recordings in memory and writes nothing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import pandas as pd

from src.behavior_analysis.poke_counts import session_choice_ports
from src.behavior_import.extract_trials import port_finding_prints, recording_trial_rows
from src.behavior_import.import_data import RunStartError, parse_content_json, read_run_start
from src.behavior_import.session_order import ordered_session_keys

COLUMNS = (
    "mouse",
    "n_port_finding_trials",
    "n_port_finding_only_sessions",
    "n_ports",
    "found_t_thresh",
    "expected_min_port_finding_trials",
    "below_expected_min",
    "flag_reason",
)


class PortFindingSettings(NamedTuple):
    """What a session's variables say about its port finding.

    Attributes:
        found_t_thresh (int | None): Times each port must be chosen for port finding to end;
            ``None`` if the session does not log it.
        start_counts (dict[str, int] | None): ``found_port_counts`` at the start of the session;
            ``None`` if not logged.
        user_set_counts (bool): Whether a ``user_set`` row sets ``found_port_counts`` by hand.
    """

    found_t_thresh: int | None
    start_counts: dict[str, int] | None
    user_set_counts: bool


def _frames(session: Mapping) -> list[pd.DataFrame]:
    frames = session["df"]
    return [frames] if isinstance(frames, pd.DataFrame) else list(frames)


def _run_start(frames: list[pd.DataFrame]) -> dict:
    """The earliest recording's ``run_start``; ``{}`` for a recording that has none.

    Imported sessions always have one. A session built without it has no settings, which the
    table reports as an unknown minimum rather than stopping.
    """
    try:
        return read_run_start(frames[0])
    except RunStartError:
        return {}


def _sets_found_port_counts(frame: pd.DataFrame) -> bool:
    is_user_set = (frame["type"] == "variable") & (frame["subtype"] == "user_set")
    return any(
        isinstance(content, dict) and "found_port_counts" in content
        for content in frame.loc[is_user_set, "content"].map(parse_content_json)
    )


def session_port_finding_settings(session: Mapping) -> PortFindingSettings:
    """The port-finding settings of one session, from its raw recordings.

    Args:
        session (Mapping): A session record with the importer's ``"df"`` (one DataFrame or a
            list of them).

    Returns:
        PortFindingSettings: ``found_t_thresh`` and ``found_port_counts`` of the earliest
        recording's ``run_start`` (``None`` where absent), and whether any recording of the
        session sets ``found_port_counts`` by hand.
    """
    frames = _frames(session)
    run_start = _run_start(frames)
    return PortFindingSettings(
        run_start.get("found_t_thresh"),
        run_start.get("found_port_counts"),
        any(_sets_found_port_counts(frame) for frame in frames),
    )


def _n_port_finding_prints(session: Mapping) -> int:
    return sum(len(port_finding_prints(frame)) for frame in _frames(session))


def _has_task_trials(session: Mapping) -> bool:
    return any(recording_trial_rows(frame) for frame in _frames(session))


def _carried_in(counts: Mapping[str, int] | None) -> bool:
    return bool(counts) and any(counts.values())


def _flag_reason(
    n_ports: int,
    thresh: int,
    first_key: str,
    first: PortFindingSettings,
    hand_set: list[str],
) -> str:
    """Why a mouse has fewer trials than that takes, naming the causes the recordings show."""
    reasons = [f"port finding ends once each of {n_ports} ports was chosen {thresh} times"]
    if hand_set:
        reasons.append(f"found_port_counts were set by hand (user_set) in {', '.join(hand_set)}")
    if _carried_in(first.start_counts):
        reasons.append(
            f"found_port_counts were already {first.start_counts} when {first_key} started: port "
            "finding began in an earlier session that is not in the data"
        )
    return "; ".join(reasons)


def _mouse_row(mouse: str, sessions: Mapping[str, Mapping]) -> dict | None:
    """The check row of one mouse, or ``None`` if it has no port-finding trial."""
    keys = ordered_session_keys(mouse, sessions)
    prints = {key: _n_port_finding_prints(sessions[key]) for key in keys}
    with_port_finding = [key for key in keys if prints[key]]
    if not with_port_finding:
        return None
    settings = {key: session_port_finding_settings(sessions[key]) for key in with_port_finding}
    first_key = with_port_finding[0]
    first = settings[first_key]
    ports = frozenset().union(*(session_choice_ports(sessions[k]) for k in with_port_finding))
    total, known = sum(prints.values()), first.found_t_thresh is not None
    minimum = len(ports) * first.found_t_thresh if known else float("nan")
    below = known and total < minimum
    hand_set = [key for key in with_port_finding if settings[key].user_set_counts]
    reason = ""
    if below:
        thresh = first.found_t_thresh
        reason = _flag_reason(len(ports), thresh, first_key, first, hand_set)
    elif not known:
        reason = "found_t_thresh is not logged, so the minimum is unknown"
    return {
        "mouse": mouse,
        "n_port_finding_trials": total,
        "n_port_finding_only_sessions": sum(
            not _has_task_trials(sessions[key]) for key in with_port_finding
        ),
        "n_ports": len(ports),
        "found_t_thresh": first.found_t_thresh if known else float("nan"),
        "expected_min_port_finding_trials": minimum,
        "below_expected_min": bool(below),
        "flag_reason": reason,
    }


def port_finding_check_table(sessions: Mapping[str, Mapping[str, Mapping]]) -> pd.DataFrame:
    """Per mouse, the port-finding trials of a problem against the minimum its settings imply.

    Args:
        sessions (Mapping[str, Mapping[str, Mapping]]): ``{mouse: {session_key: session}}`` of one
            problem, including the port-finding-only sessions assigned to it. Each session needs
            the importer's ``"df"`` and ``"choice_towers"``.

    Returns:
        pd.DataFrame: One row per mouse with port-finding trials, columns ``mouse``,
        ``n_port_finding_trials`` (printed ``num_t_found`` trials over all the mouse's sessions),
        ``n_port_finding_only_sessions`` (of those sessions, how many have no task trial),
        ``n_ports`` (choice ports of the sessions with port finding), ``found_t_thresh`` and
        ``expected_min_port_finding_trials`` (``n_ports x found_t_thresh``, taken from the first
        session with port finding; NaN when not logged), ``below_expected_min`` and
        ``flag_reason`` (empty unless flagged or the minimum is unknown).

    Raises:
        ValueError: If a mouse's session keys cannot be ordered, or a session with port finding
            has no choice towers.
    """
    rows = [_mouse_row(mouse, sessions[mouse]) for mouse in sorted(sessions)]
    return pd.DataFrame([row for row in rows if row], columns=list(COLUMNS))
