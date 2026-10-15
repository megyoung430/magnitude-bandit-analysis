"""Count the logged good reversals whose best tower does not change.

The task logs its own good-reversal totals, which stay as they are. This module counts, beside
them, the suspect ones, by applying the check that :func:`get_good_reversal_info` uses when it
drops reversals to every within-session increment of the good-reversal counter. Reversals that
happen while nothing is recording (between runs, or after the last trial) have no trial to
compare, so they are reported as uncheckable instead of being guessed at.
"""

from collections import Counter

from src.behavior_analysis.best_change import best_change_problem


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


def _between_run_events(session):
    """``{trial index: good events}`` that happened between recording runs."""
    events = Counter()
    for e in session.get("reversal_boundary_events", []):
        if e["kind"] == "good" and e["source"] == "between_runs":
            events[e["index"]] += e["count"]
    return events


def _uncheckable_events(session):
    """Good events with no trial on either side: between runs, or after a run's last trial."""
    return sum(
        e["count"] for e in session.get("reversal_boundary_events", []) if e["kind"] == "good"
    )


def count_suspect_good_reversals(session):
    """Suspect and uncheckable good reversals of one session.

    Args:
        session: A session dict with ``good_reversals`` (the corrected per-trial counter),
            ``reward_magnitudes_by_tower`` and optionally ``reversal_boundary_events``.

    Returns:
        ``{"suspect": int, "uncheckable": int}``. *suspect* counts the within-session
        increments of the good counter where the best tower does not change (a tie for best
        and a missing magnitude count too). *uncheckable* counts the between-run and
        terminal events, plus the increments of a session that has no reward magnitudes at
        all. A session without a good counter has neither.
    """
    rewards = session.get("reward_magnitudes_by_tower") or {}
    between = _between_run_events(session)
    suspect, uncheckable = 0, _uncheckable_events(session)
    for i, size in _increments(session.get("good_reversals") or []):
        # The counter also jumps where a between-run event enters; that part is counted above.
        size -= between.get(i, 0)
        if size <= 0:
            continue
        if not rewards:
            uncheckable += size
        elif best_change_problem(_magnitudes_at(rewards, i - 1), _magnitudes_at(rewards, i)):
            suspect += size
    return {"suspect": suspect, "uncheckable": uncheckable}
