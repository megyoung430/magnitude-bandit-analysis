"""Count the logged good reversals whose best tower does not change.

The task logs its own good-reversal totals, which stay as they are. This module counts, beside
them, with the rule that :func:`get_good_reversal_info` uses when it drops reversals (see
:mod:`src.behavior_analysis.best_change`):

- *suspect*: a good-counter increment inside a recording run whose best tower is unchanged or
  tied;
- *uncheckable*: an increment with a missing magnitude (or in a session without any), and every
  reversal logged while nothing was recording (a ``reversal_boundary_events`` entry: between
  runs, or after a run's last trial).

A reversal logged while nothing was recording is uncheckable, not suspect, because the trials
on either side of it may be separated by any number of unlogged reversals, so an unchanged best
tower is not evidence of a bug on its own. :func:`get_good_reversal_info` still drops such a
reversal when its best tower looks unchanged (or a magnitude is missing), so every dropped
reversal is suspect or uncheckable, counted per reversal. A reversal that the zero-reward gate
or the missing-tower check skips before that check is neither dropped nor in its warning, but is
classified here.
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


def good_event_counts(session):
    """``{session-local trial index: good reversals logged while nothing was recording}``.

    The index is the trial at which the event enters the good counter; an index equal to the
    number of trials is an event after the session's last trial.
    """
    events = Counter()
    for event in session.get("reversal_boundary_events", []):
        if event["kind"] == "good":
            events[event["index"]] += event["count"]
    return events


def count_suspect_good_reversals(session):
    """Suspect and uncheckable good reversals of one session.

    Args:
        session: A session dict with ``good_reversals`` (the corrected per-trial counter),
            ``reward_magnitudes_by_tower`` and optionally ``reversal_boundary_events``.

    Returns:
        ``{"suspect": int, "uncheckable": int}``. *suspect* counts the within-run increments of
        the good counter where the best tower is unchanged or tied. *uncheckable* counts the
        increments with a missing magnitude (all of them if the session has no reward
        magnitudes), any reversal on the first trial, plus every event logged while nothing
        was recording. A session without a good counter has neither.
    """
    rewards = session.get("reward_magnitudes_by_tower") or {}
    events = good_event_counts(session)
    counter = session.get("good_reversals") or []
    suspect, uncheckable = 0, sum(events.values())
    # A counter that starts above what the events at trial 0 explain has a reversal on the
    # first trial, with no earlier trial in this session to compare it with.
    if counter and counter[0] is not None:
        uncheckable += max(0, counter[0] - events.get(0, 0))
    for i, size in _increments(counter):
        # The counter also jumps where an event enters; that part is counted above.
        size -= events.get(i, 0)
        if size <= 0:
            continue
        if not rewards:
            uncheckable += size
            continue
        problem = best_change_problem(_magnitudes_at(rewards, i - 1), _magnitudes_at(rewards, i))
        if is_suspect(problem):
            suspect += size
        elif is_uncheckable(problem):
            uncheckable += size
    return {"suspect": suspect, "uncheckable": uncheckable}
