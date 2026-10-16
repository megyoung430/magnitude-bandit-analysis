"""Count the logged good reversals whose best tower does not change.

Only the blocked task logs a good counter. The drifting task has block changes and no counter, so
a drifting session has nothing to count here: its best tower often stays the best at a block
change (the most-chosen arm drops to 0, which can be the second best), and the check that its
zero arm moves is made by :func:`get_good_reversal_info` and listed by ``suspect_reversal_table``.

The task logs its own good-reversal totals, which stay as they are. This module counts, beside
them, with the rule that :func:`get_good_reversal_info` uses when it drops reversals (see
:mod:`src.behavior_analysis.best_change`):

- *suspect*: a good-counter increment whose best tower is unchanged or tied;
- *uncheckable*: an increment with a missing magnitude (or in a session without any), and a
  reversal for which there really is no trial to check against.

The recordings of a session form one continuous session, so a reversal logged while nothing was
recording (a ``reversal_boundary_events`` entry) between two recordings is checked like any
other, on the last trial before it and the first trial after it. Only two kinds of event have no
such pair:

- one at the very start of a problem's first session, or after the last trial of its last
  session: there is no trial on the other side;
- one logged after a session's last trial (``source`` ``"run_end"``, index = the trial count)
  whose next session with trials does not show it at its first trial: the import marks it
  ``session_check = "inconsistent"``, it is not carried into the next session, and it counts as
  uncheckable. When the next session does show it (``"consistent"``) it starts the new block
  there and needs no further check.

An event in the next session's first ``run_start`` (index 0, ``"between_runs"``) is checked
against the previous session's last trial by the import (``session_check``): a failed check is
suspect.

:func:`get_good_reversal_info` drops the same reversals that this module counts as suspect or
uncheckable, counted per reversal. A reversal that the zero-reward gate or the missing-tower
check skips before that check is neither dropped nor in its warning, but is classified here.
"""

from collections import Counter

from src.behavior_analysis.best_change import best_change_problem, is_suspect, is_uncheckable


def _increments(counter):
    """``[(index, size)]`` where a cumulative counter rises above its previous value.

    A ``None`` entry, and the entry after it, are skipped.
    """
    steps, prev = [], None
    for i, v in enumerate(counter):
        if i > 0 and v is not None and prev is not None and v > prev:
            steps.append((i, v - prev))
        prev = v
    return steps


def _magnitudes_at(rewards, trial):
    """``{tower: magnitude}`` at *trial*, with ``None`` where an array does not reach it."""
    return {t: (arr[trial] if trial < len(arr) else None) for t, arr in rewards.items()}


def _good_events(session):
    """The session's good-reversal boundary events, as logged."""
    return [e for e in session.get("reversal_boundary_events", []) if e["kind"] == "good"]


def good_event_counts(session):
    """``{session-local trial index: good reversals logged while nothing was recording}``.

    The index is the trial at which the event enters the good counter; an index equal to the
    number of trials is an event after the session's last trial. An event after the last trial
    that the next session's first trial does not show (``session_check`` ``"inconsistent"``)
    is not carried into it and is left out.

    Args:
        session (dict): A session dict, optionally with ``reversal_boundary_events``
            (``{"kind": "good" | "bad" | "block", "index": int, "count": int}`` entries).

    Returns:
        Counter: ``{trial index: number of good reversals}``; empty if there are none.
    """
    events = Counter()
    n_trials = len(session.get("trial") or session.get("good_reversals") or [])
    for event in _good_events(session):
        if event.get("session_check") == "inconsistent" and event["index"] == n_trials:
            continue
        events[event["index"]] += event["count"]
    return events


def _count_session_end_events(events, n_trials):
    """``(suspect, uncheckable)`` of the events after the session's last trial.

    A consistent event starts the next session's block and is fine; any other has no trial pair
    that shows it.
    """
    pending = [e for e in events if e["index"] == n_trials]
    return 0, sum(e["count"] for e in pending if e.get("session_check") != "consistent")


def _count_session_start_events(events, n_trials):
    """``(suspect, uncheckable)`` of the events in the session's first ``run_start``.

    The import checked them against the previous session's last trial; without that check (no
    earlier session of the problem, or no magnitudes) there is no trial to compare with.
    """
    if n_trials == 0:
        return 0, 0
    at_start = [e for e in events if e["index"] == 0]
    suspect = sum(e["count"] for e in at_start if e.get("session_check") == "inconsistent")
    uncheckable = sum(e["count"] for e in at_start if "session_check" not in e)
    return suspect, uncheckable


def count_suspect_good_reversals(session):
    """Suspect and uncheckable good reversals of one session.

    Args:
        session (dict): A session dict with ``good_reversals`` (the corrected per-trial counter),
            ``reward_magnitudes_by_tower`` and optionally ``reversal_boundary_events``.

    Returns:
        dict[str, int]: ``{"suspect": int, "uncheckable": int}``. *suspect* counts the
        increments of the good counter where the best tower is unchanged or tied, including the
        ones that entered between recordings. *uncheckable* counts the increments with a missing
        magnitude (all of them if the session has no reward magnitudes), a reversal on the first
        trial with no earlier session to compare it with, and the events after the last trial
        that the next session does not show. A session without a good counter has neither.
    """
    rewards = session.get("reward_magnitudes_by_tower") or {}
    counter = session.get("good_reversals") or []
    events = _good_events(session)
    n_trials = len(counter)
    suspect, uncheckable = 0, 0
    for found in (
        _count_session_end_events(events, n_trials),
        _count_session_start_events(events, n_trials),
    ):
        suspect, uncheckable = suspect + found[0], uncheckable + found[1]
    # A counter that starts above what the events at trial 0 explain has a reversal on the
    # first trial, with no earlier trial in this session to compare it with.
    at_start = sum(e["count"] for e in events if e["index"] == 0)
    if counter and counter[0] is not None:
        uncheckable += max(0, counter[0] - at_start)
    for i, size in _increments(counter):
        if not rewards:
            uncheckable += size
            continue
        problem = best_change_problem(_magnitudes_at(rewards, i - 1), _magnitudes_at(rewards, i))
        if is_suspect(problem):
            suspect += size
        elif is_uncheckable(problem):
            uncheckable += size
    return {"suspect": suspect, "uncheckable": uncheckable}
