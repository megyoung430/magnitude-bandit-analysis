"""A sanity check on the number of port-finding trials of each mouse and problem.

The task counts how often each choice port was chosen during port finding (``found_port_counts``)
and ends port finding once every port reached ``found_t_thresh``. A mouse that went through port
finding from the start therefore has at least ``n_ports x found_t_thresh`` port-finding trials
(the printed ones: a trial that times out has no print and does not count towards finding).
Fewer means port finding began in a session that is missing from the data, or the counts were
set by hand (a ``user_set`` of ``found_port_counts``).

A mouse-problem is flagged (``flagged``) when any of these holds, whatever its total:

- it has fewer trials than the minimum (``below_expected_min``);
- the first session with port finding starts with non-zero ``found_port_counts``, so port
  finding began in an earlier session that is missing from the data or not assigned to the
  problem (``carried_in_counts`` lists the starting counts);
- ``found_port_counts``, ``found_t_thresh`` or ``found_ports`` were set by hand (``user_set``).

The minimum uses the latest ``found_t_thresh`` set by hand before the last port-finding print of
its recording, else the ``run_start`` value of the first session with port finding. The check
never raises on a flagged row.

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
    "carried_in_counts",
    "below_expected_min",
    "flagged",
    "flag_reason",
)

# Variables whose hand-setting changes how port finding ends.
HAND_SET_VARIABLES = ("found_port_counts", "found_t_thresh", "found_ports")


class PortFindingSettings(NamedTuple):
    """What a session's variables say about its port finding.

    Attributes:
        found_t_thresh (int | None): ``run_start`` value of the times each port must be chosen
            for port finding to end; ``None`` if the session does not log it.
        start_counts (dict[str, int] | None): ``found_port_counts`` at the start of the session;
            ``None`` if not logged.
        hand_set (tuple[str, ...]): The :data:`HAND_SET_VARIABLES` that a ``user_set`` row sets
            in the session.
        hand_set_thresh (int | None): The latest ``found_t_thresh`` set by hand before the last
            port-finding print of its recording; ``None`` if none.
    """

    found_t_thresh: int | None
    start_counts: dict[str, int] | None
    hand_set: tuple[str, ...]
    hand_set_thresh: int | None


def _frames(session: Mapping) -> list[pd.DataFrame]:
    frames = session["df"]
    return [frames] if isinstance(frames, pd.DataFrame) else list(frames)


def _user_sets(frame: pd.DataFrame) -> list[tuple[float, dict]]:
    """``(time, variables)`` of each ``user_set`` row of a recording, in file order."""
    is_user_set = (frame["type"] == "variable") & (frame["subtype"] == "user_set")
    rows = zip(frame.loc[is_user_set, "time"], frame.loc[is_user_set, "content"], strict=True)
    parsed = [(time, parse_content_json(content)) for time, content in rows]
    return [(time, content) for time, content in parsed if isinstance(content, dict)]


def _last_port_finding_time(frame: pd.DataFrame) -> float:
    """Time of the recording's last port-finding print (``-inf`` if it has none)."""
    is_print = (frame["type"] == "variable") & (frame["subtype"] == "print")
    prints = frame.loc[is_print]
    is_port_finding = prints["content"].map(
        lambda content: isinstance(d := parse_content_json(content), dict) and "num_t_found" in d
    )
    return (
        float(prints.loc[is_port_finding, "time"].max()) if is_port_finding.any() else -float("inf")
    )


def _hand_set(frames: list[pd.DataFrame]) -> tuple[tuple[str, ...], int | None]:
    """The hand-set port-finding variables of a session, and the latest hand-set threshold."""
    names: list[str] = []
    thresh = None
    for frame in frames:
        last_found = _last_port_finding_time(frame)
        for time, changed in _user_sets(frame):
            names.extend(v for v in HAND_SET_VARIABLES if v in changed and v not in names)
            if "found_t_thresh" in changed and time <= last_found:
                thresh = changed["found_t_thresh"]
    return tuple(names), thresh


def session_port_finding_settings(session: Mapping, label: str = "session") -> PortFindingSettings:
    """The port-finding settings of one session, from its raw recordings.

    Args:
        session (Mapping): A session record with the importer's ``"df"`` (one DataFrame or a
            list of them).
        label (str): Names the session in the error message. Default ``"session"``.

    Returns:
        PortFindingSettings: ``found_t_thresh`` and ``found_port_counts`` of the earliest
        recording's ``run_start`` (``None`` where absent), the variables a ``user_set`` row
        sets by hand, and the latest hand-set ``found_t_thresh``.

    Raises:
        ValueError: If the earliest recording has no readable ``run_start`` row (imported
            sessions always have one).
    """
    frames = _frames(session)
    try:
        run_start = read_run_start(frames[0])
    except RunStartError as err:
        raise ValueError(f"{label}: {err}") from err
    hand_set, hand_set_thresh = _hand_set(frames)
    return PortFindingSettings(
        run_start.get("found_t_thresh"),
        run_start.get("found_port_counts"),
        hand_set,
        hand_set_thresh,
    )


def _n_port_finding_prints(session: Mapping) -> int:
    return sum(len(port_finding_prints(frame)) for frame in _frames(session))


def _has_task_trials(session: Mapping) -> bool:
    return any(recording_trial_rows(frame) for frame in _frames(session))


def _carried_in(counts: Mapping[str, int] | None) -> bool:
    return bool(counts) and any(counts.values())


def _counts_text(counts: Mapping[str, int] | None) -> str:
    return ";".join(f"{port}={n}" for port, n in sorted(counts.items())) if counts else ""


def _effective_threshold(settings: Mapping[str, PortFindingSettings]) -> int | None:
    """The latest hand-set threshold, else the first session's ``run_start`` value."""
    hand_set = [s.hand_set_thresh for s in settings.values() if s.hand_set_thresh is not None]
    return hand_set[-1] if hand_set else next(iter(settings.values())).found_t_thresh


def _flag_reasons(
    settings: Mapping[str, PortFindingSettings], below: bool, n_ports: int, thresh
) -> list[str]:
    """The reasons a mouse-problem is flagged, in plain words; empty if it is not."""
    first_key, first = next(iter(settings.items()))
    reasons = []
    if below:
        reasons.append(f"port finding ends once each of {n_ports} ports was chosen {thresh} times")
    for name in HAND_SET_VARIABLES:
        keys = [key for key, s in settings.items() if name in s.hand_set]
        if keys:
            verb = "were" if name == "found_port_counts" else "was"
            reasons.append(f"{name} {verb} set by hand (user_set) in {', '.join(keys)}")
    if _carried_in(first.start_counts):
        reasons.append(
            f"found_port_counts were already {first.start_counts} when {first_key} started: port "
            "finding began in an earlier session that is missing from the data or not assigned "
            "(see the warnings)"
        )
    return reasons


def _mouse_row(mouse: str, sessions: Mapping[str, Mapping]) -> dict | None:
    """The check row of one mouse, or ``None`` if it has no port-finding trial."""
    prints = {
        key: _n_port_finding_prints(sessions[key]) for key in ordered_session_keys(mouse, sessions)
    }
    with_port_finding = [key for key, n in prints.items() if n]
    if not with_port_finding:
        return None
    settings = {
        key: session_port_finding_settings(sessions[key], f"{mouse} {key}")
        for key in with_port_finding
    }
    n_ports = len(
        frozenset().union(*(session_choice_ports(sessions[k]) for k in with_port_finding))
    )
    total, thresh = sum(prints.values()), _effective_threshold(settings)
    known = thresh is not None
    minimum = n_ports * thresh if known else float("nan")
    below = known and total < minimum
    reasons = _flag_reasons(settings, below, n_ports, thresh)
    notes = [] if known else ["found_t_thresh is not logged, so the minimum is unknown"]
    first_counts = next(iter(settings.values())).start_counts
    return {
        "mouse": mouse,
        "n_port_finding_trials": total,
        "n_port_finding_only_sessions": sum(
            not _has_task_trials(sessions[key]) for key in with_port_finding
        ),
        "n_ports": n_ports,
        "found_t_thresh": thresh if known else float("nan"),
        "expected_min_port_finding_trials": minimum,
        "carried_in_counts": _counts_text(first_counts) if _carried_in(first_counts) else "",
        "below_expected_min": bool(below),
        "flagged": bool(reasons),
        "flag_reason": "; ".join(reasons + notes),
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
        ``expected_min_port_finding_trials`` (``n_ports x found_t_thresh``, see the module doc
        for which threshold; NaN when not logged), ``carried_in_counts`` (``found_port_counts``
        at the start of the first session with port finding, as ``"A1=5;B1=29"``, empty when
        zero), ``below_expected_min``, ``flagged`` (any reason in the module doc) and
        ``flag_reason`` (empty unless flagged or the minimum is unknown).

    Raises:
        ValueError: If a mouse's session keys cannot be ordered, a session with port finding has
            no choice towers, or no ``run_start`` (the message names mouse and session).
    """
    rows = [_mouse_row(mouse, sessions[mouse]) for mouse in sorted(sessions)]
    return pd.DataFrame([row for row in rows if row], columns=list(COLUMNS))
