"""A table of sessions whose starting reward magnitudes differ from the previous session's end.

Reward magnitudes carry over from one session of a problem to the next, in both tasks. A session
that starts on other magnitudes than the previous session of its problem ended on had something
change in between. Only sessions that continue a problem are compared; the first session of a
problem starts from fresh magnitudes.

The "previous session" is the previous session *of the problem that has trials*, not the session
before it in time: a session without trial information (``no_trial_sessions.csv``) or one left
out of the problem as a settings error lies in between but is not compared. The comparison is
:func:`~src.behavior_import.run_boundaries.describe_boundary` of the last trial of the previous
session against the first trial of this one (the sessions' ``"session_edges"``), with the change
in the logged counter totals between them.

Columns: ``mouse``, ``session``, ``previous_session``, ``file`` (this session's first
recording), ``task_type``, ``problem_in_type``, ``overall_problem``,
``magnitudes_end_previous``, ``magnitudes_start``, ``towers_differing``,
``block_total_change`` / ``good_total_change`` / ``bad_total_change`` (the change in the logged
session totals, blank where the file logs no such total), ``verdict`` and ``boundary_reason``
(from ``describe_boundary``) and ``likely_cause``.

``likely_cause`` is the first of the following that applies, so the causes are ordered from the
most to the least likely (:data:`CAUSES`):

1. :data:`LOGGED_AND_CONSISTENT`: the task logged a reversal or block change between the two
   sessions (in the previous ``run_end`` after its last trial, or in this ``run_start``) and the
   magnitudes show it: for the drifting task the zero arm moved, for the blocked task the best
   tower changed at a good reversal and the magnitudes changed at a bad one.
2. :data:`LOGGED_BUT_INCONSISTENT`: a change was logged but the magnitudes do not fit it (the
   zero arm stayed, the best tower stayed at a good reversal), or a total went down without the
   magnitudes going back to the initial values.
3. :data:`DRIFT`: drifting task, nothing logged, the zero arm unchanged and every other arm
   moved by at most :data:`MAX_DRIFT_STEP`: the drift of one trial, which happens between
   consecutive trials of a session as well.
4. :data:`RESET`: the session starts with every arm at :data:`INITIAL_MAGNITUDE` while the
   previous session did not end that way: the task was restarted with its initial values. A
   restart also sets the logged totals back to 0, which is not read as a logged change.
5. :data:`CHANGE_FROM_MAGNITUDES`: the file logs no totals, so the change (the zero arm moved, or
   the blocked magnitudes differ) is inferred from the magnitudes alone, as the importer does.
6. :data:`PREVIOUS_SESSION_NO_TRIALS`: a session of the same mouse with no trial information lies
   between the two (see ``no_trial_sessions.csv``); whatever happened in it is not in the data.
7. :data:`NO_LOGGED_CHANGE`: the magnitudes changed and nothing explains it.
8. :data:`CANNOT_COMPARE`: a side has no complete magnitudes (a tower is missing), the two sides
   have different towers, or the session has no first or last trial to compare.
"""

from pathlib import Path

import pandas as pd

from src.behavior_analysis.review_context import ProblemIndex, magnitudes_text, typed_frame
from src.behavior_import.import_data import session_source_paths
from src.behavior_import.import_log import SESSION, SkipRecord
from src.behavior_import.run_boundaries import CONSISTENT, describe_boundary
from src.behavior_import.session_order import ordered_session_keys, session_sort_key

INITIAL_MAGNITUDE = 2  # every arm starts a problem at this magnitude
# The largest change of an arm between two consecutive trials in the staged drifting sessions.
MAX_DRIFT_STEP = 2

LOGGED_AND_CONSISTENT = (
    "reversal (good/bad) or block change logged between the sessions (run_end after the last "
    "trial or next run_start); magnitudes consistent with it"
)
LOGGED_BUT_INCONSISTENT = "change logged between the sessions but magnitudes inconsistent"
DRIFT = "drifting arms moved by normal drift (zero arm unchanged, no change logged)"
RESET = f"reset to the initial values (all {INITIAL_MAGNITUDE})"
CHANGE_FROM_MAGNITUDES = "change inferred from the magnitudes (no totals logged)"
PREVIOUS_SESSION_NO_TRIALS = "previous session has no trials"
NO_LOGGED_CHANGE = "magnitudes changed with no logged change"
CANNOT_COMPARE = "cannot compare: magnitudes missing, different towers, or no trial to compare"
CAUSES = (
    LOGGED_AND_CONSISTENT,
    LOGGED_BUT_INCONSISTENT,
    DRIFT,
    RESET,
    CHANGE_FROM_MAGNITUDES,
    PREVIOUS_SESSION_NO_TRIALS,
    NO_LOGGED_CHANGE,
    CANNOT_COMPARE,
)

_DTYPES = {
    "mouse": "object",
    "session": "object",
    "previous_session": "object",
    "file": "object",
    "task_type": "object",
    "problem_in_type": "Int64",
    "overall_problem": "Int64",
    "magnitudes_end_previous": "object",
    "magnitudes_start": "object",
    "towers_differing": "object",
    "block_total_change": "Int64",
    "good_total_change": "Int64",
    "bad_total_change": "Int64",
    "verdict": "object",
    "boundary_reason": "object",
    "likely_cause": "object",
}
COLUMNS = tuple(_DTYPES)


def differing_towers(before: dict, after: dict) -> list[str]:
    """The towers whose magnitude differs between two sides (a tower on one side only differs).

    Args:
        before (dict): ``{tower: magnitude}`` at the end of the previous session.
        after (dict): ``{tower: magnitude}`` at the start of the session.

    Returns:
        list[str]: The differing towers, sorted.
    """
    return sorted(t for t in before.keys() | after.keys() if before.get(t) != after.get(t))


def _total_deltas(boundary: dict) -> list[int]:
    return [delta for delta in boundary["total_delta"].values() if delta is not None]


def _is_restart(boundary: dict) -> bool:
    """A total went down and the magnitudes are back at the initial values: the task restarted."""
    return any(delta < 0 for delta in _total_deltas(boundary)) and _is_reset(boundary)


def _logged_any(boundary: dict) -> bool:
    """A total changed, other than by a restart setting the totals back to 0."""
    return any(_total_deltas(boundary)) and not _is_restart(boundary)


def _comparable(boundary: dict | None) -> bool:
    """Whether both sides have complete magnitudes of the same towers."""
    return boundary is not None and boundary["magnitude_change"] is not None


def _inferred_from_magnitudes(boundary: dict) -> bool:
    """No total is logged and the magnitudes show a block change."""
    return boundary["source"] == "magnitudes" and boundary["magnitude_change"] is True


def _is_normal_drift(boundary: dict) -> bool:
    """Drifting task, the zero arm unchanged, and every arm within one trial's drift."""
    before, after = boundary["magnitudes_before"], boundary["magnitudes_after"]
    return (
        boundary["mode"] == "drifting"
        and boundary["magnitude_change"] is False
        and all(abs(after[t] - before[t]) <= MAX_DRIFT_STEP for t in before)
    )


def _is_reset(boundary: dict) -> bool:
    before, after = boundary["magnitudes_before"], boundary["magnitudes_after"]
    started = set(after.values()) == {INITIAL_MAGNITUDE}
    return started and set(before.values()) != {INITIAL_MAGNITUDE}


def starting_magnitude_cause(boundary: dict | None, *, gap_without_trials: bool) -> str:
    """The most likely cause of differing magnitudes across a session boundary.

    Args:
        boundary (dict | None): ``describe_boundary`` of the previous session's last trial
            against the session's first, or ``None`` if either has no trial.
        gap_without_trials (bool): Whether a session without trial information lies between
            the two sessions.

    Returns:
        str: One of :data:`CAUSES`, the first that applies.
    """
    if not _comparable(boundary):
        return CANNOT_COMPARE
    if _logged_any(boundary):
        consistent = boundary["verdict"] == CONSISTENT
        return LOGGED_AND_CONSISTENT if consistent else LOGGED_BUT_INCONSISTENT
    if _is_normal_drift(boundary):
        return DRIFT
    if _is_reset(boundary):
        return RESET
    if _inferred_from_magnitudes(boundary):
        return CHANGE_FROM_MAGNITUDES
    if gap_without_trials:
        return PREVIOUS_SESSION_NO_TRIALS
    return NO_LOGGED_CHANGE


def _sessions_without_trials(skips: list[SkipRecord]) -> dict[str, list[str]]:
    """``{mouse: [session keys left out for having no trials]}``."""
    out: dict[str, list[str]] = {}
    for skip in skips:
        if skip.level == SESSION and skip.mouse and skip.session:
            out.setdefault(skip.mouse, []).append(skip.session)
    return out


def _has_gap(empty_sessions: list[str], previous: str, current: str) -> bool:
    low, high = session_sort_key(previous), session_sort_key(current)
    return any(low < session_sort_key(key) < high for key in empty_sessions)


def _differs(boundary: dict | None) -> bool:
    """Whether the boundary shows different magnitudes, or cannot be compared at all."""
    if not _comparable(boundary):
        return True
    return bool(differing_towers(boundary["magnitudes_before"], boundary["magnitudes_after"]))


def session_boundary(previous: dict, session: dict, mode: str) -> dict | None:
    """The last trial of *previous* against the first trial of *session*.

    Args:
        previous (dict): The previous session of the problem, with its ``"session_edges"``.
        session (dict): The session, with its ``"session_edges"``.
        mode (str): ``"drifting"`` or ``"blocked"``.

    Returns:
        dict | None: :func:`~src.behavior_import.run_boundaries.describe_boundary` of the two
        sides; ``None`` if either session has no trial.
    """
    before, after = previous.get("session_edges"), session.get("session_edges")
    if not (before and after):
        return None
    return describe_boundary(before["last"], after["first"], mode)


def _row(mouse, previous, key, sess, boundary, cause, index) -> dict:
    before = (boundary or {}).get("magnitudes_before")
    after = (boundary or {}).get("magnitudes_after")
    delta = (boundary or {}).get("total_delta", {})
    paths = session_source_paths(sess)
    return {
        "mouse": mouse,
        "session": key,
        "previous_session": previous,
        "file": Path(paths[0]).name if paths else None,
        **index.columns(mouse, key),
        "magnitudes_end_previous": magnitudes_text(before) if before else None,
        "magnitudes_start": magnitudes_text(after) if after else None,
        "towers_differing": ";".join(differing_towers(before, after)) if before and after else None,
        "block_total_change": delta.get("block"),
        "good_total_change": delta.get("good"),
        "bad_total_change": delta.get("bad"),
        "verdict": (boundary or {}).get("verdict"),
        "boundary_reason": (boundary or {}).get("reason"),
        "likely_cause": cause,
    }


def starting_magnitudes_table(
    grouped: dict, skips: list[SkipRecord], index: ProblemIndex
) -> pd.DataFrame:
    """One row per session whose start differs from the end of its problem's previous session.

    Args:
        grouped (dict): ``{(task_type, k): {mouse: {session_key: session_dict}}}`` from
            :func:`~src.behavior_import.task_types.regroup_by_task_type`; each session carries the
            importer's ``"session_edges"``.
        skips (list[SkipRecord]): ``ImportLog.skips``, to find sessions without trials between
            two sessions of a problem.
        index (ProblemIndex): Where each session sits in its mouse's problems.

    Returns:
        pd.DataFrame: The columns of :data:`COLUMNS`, ordered by task type, problem, mouse and
        session; empty (with the columns) if no session qualifies.
    """
    empty = _sessions_without_trials(skips)
    rows = []
    for (task_type, _), mice in grouped.items():
        for mouse in sorted(mice):
            keys = ordered_session_keys(mouse, mice[mouse])
            for previous, key in zip(keys, keys[1:], strict=False):
                sess = mice[mouse][key]
                boundary = session_boundary(mice[mouse][previous], sess, task_type)
                if not _differs(boundary):
                    continue
                gap = _has_gap(empty.get(mouse, []), previous, key)
                cause = starting_magnitude_cause(boundary, gap_without_trials=gap)
                rows.append(_row(mouse, previous, key, sess, boundary, cause, index))
    return typed_frame(rows, _DTYPES)
