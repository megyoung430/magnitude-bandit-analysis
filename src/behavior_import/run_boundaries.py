"""What changed between two recordings: the counters and the reward magnitudes, side by side.

A *boundary* runs from the last trial of one recording to the first trial of the next recording,
of the same session or of the next session of the same problem. The task can log a block change
(drifting task) or a good or bad reversal (blocked task) in that gap, in a ``run_end`` or in the
next ``run_start``, without printing it in any trial. :func:`describe_boundary` says what the
counters and the magnitudes show on either side and whether they agree.

Both sides are plain dicts (``{"magnitudes", "totals", "counters"}``, see
:func:`describe_boundary`), so the description can be reused for any pair of recordings,
for example to review sessions whose starting magnitudes differ from the previous session's end.

The functions here are pure: they read no files and change nothing.
"""

from src.behavior_import.problem_numbering import BLOCKED, DRIFTING

KINDS = ("block", "good", "bad")

NO_CHANGE = "no change"
CONSISTENT = "consistent"
INCONSISTENT = "inconsistent"
UNVERIFIABLE = "unverifiable"


def _complete(magnitudes):
    """True if *magnitudes* is a non-empty ``{tower: number}`` dict with no missing value."""
    return bool(magnitudes) and all(v is not None for v in magnitudes.values())


def _comparable(before, after):
    """True if both sides are complete and describe the same towers."""
    return _complete(before) and _complete(after) and set(before) == set(after)


def zero_towers(magnitudes):
    """The towers whose magnitude is 0 (the drifting task's zero arm).

    Args:
        magnitudes (dict): ``{tower: magnitude}`` with no missing value.

    Returns:
        frozenset[str]: The towers at magnitude 0.
    """
    return frozenset(t for t, v in magnitudes.items() if v == 0)


def best_changed(before, after):
    """Whether the best tower differs; a tie resolves to the first-listed tower.

    This is the rule that decides whether a logged good reversal is real: it matches
    :func:`src.behavior_analysis.best_change.best_change_problem`.

    Args:
        before (dict | None): ``{tower: magnitude}`` on the earlier trial.
        after (dict | None): ``{tower: magnitude}`` on the later trial.

    Returns:
        bool | None: ``True`` if the first-listed best tower differs, ``False`` if not,
        ``None`` if a magnitude is missing or the towers differ.
    """
    if not _comparable(before, after):
        return None
    return max(before, key=before.get) != max(after, key=after.get)


def magnitude_change(before, after, mode):
    """Whether the magnitudes show a block change between two trials.

    Drifting: the zero arm moved (the other arms drift a little on every trial, so a
    difference elsewhere is not a block change). Blocked: the magnitudes are fixed within a
    block, so any difference is.

    Args:
        before (dict | None): ``{tower: magnitude}`` on the earlier trial.
        after (dict | None): ``{tower: magnitude}`` on the later trial.
        mode (str | None): ``"drifting"`` or ``"blocked"``; ``None`` if the task is unknown.

    Returns:
        bool | None: ``None`` if a magnitude is missing, the towers differ or *mode* is
        unknown.
    """
    if mode not in (DRIFTING, BLOCKED) or not _comparable(before, after):
        return None
    if mode == DRIFTING:
        return zero_towers(before) != zero_towers(after)
    return dict(before) != dict(after)


def event_matches(kind, before, after, mode):
    """Whether the magnitudes show a logged event of *kind*.

    Args:
        kind (str): ``"good"`` (the best tower changes), ``"bad"`` (the magnitudes change) or
            ``"block"`` (:func:`magnitude_change`).
        before (dict | None): ``{tower: magnitude}`` on the last trial before the event.
        after (dict | None): ``{tower: magnitude}`` on the first trial after it.
        mode (str | None): ``"drifting"`` or ``"blocked"``; ``None`` if the task is unknown.

    Returns:
        bool | None: ``None`` if it cannot be told (see :func:`magnitude_change`).

    Raises:
        ValueError: If *kind* is not ``"good"``, ``"bad"`` or ``"block"``.
    """
    if kind == "good":
        return best_changed(before, after)
    if kind == "bad":
        return magnitude_change(before, after, BLOCKED)
    if kind == "block":
        return magnitude_change(before, after, mode)
    raise ValueError(f"unknown event kind {kind!r}; expected 'good', 'bad' or 'block'")


def events_match(kind, count, before, after, mode):
    """Whether the magnitudes show *count* events of *kind* logged at one boundary.

    A single event must show (:func:`event_matches`). Two or more can take the zero arm (or the
    best tower) away and bring it back, so the two sides of the boundary look unchanged although
    every event happened; the counter delta says how many there were, and the net result cannot
    argue against them. Such a group counts as shown.

    Args:
        kind (str): ``"good"``, ``"bad"`` or ``"block"`` (see :func:`event_matches`).
        count (int): How many events of *kind* the counters log at the boundary.
        before (dict | None): ``{tower: magnitude}`` on the last trial before the events.
        after (dict | None): ``{tower: magnitude}`` on the first trial after them.
        mode (str | None): ``"drifting"``, ``"blocked"`` or ``None`` if the task is unknown.

    Returns:
        bool | None: ``None`` if it cannot be told.

    Raises:
        ValueError: If *kind* is unknown (see :func:`event_matches`).
    """
    shown = event_matches(kind, before, after, mode)
    return True if shown is False and count >= 2 else shown


def _delta(before, after):
    """``{kind: after - before}``, ``None`` where either side is not logged."""
    return {
        kind: None
        if before.get(kind) is None or after.get(kind) is None
        else int(after[kind] - before[kind])
        for kind in KINDS
    }


def _logged(total_delta, mode):
    """The ``{kind: change}`` that the totals log, for the kinds that count in *mode*."""
    kinds = ("good", "bad") if mode == BLOCKED else ("block",)
    logged = {k: total_delta[k] for k in kinds if total_delta[k] is not None}
    if not logged and mode == BLOCKED and total_delta["block"] is not None:
        # Only the block total is logged: the change is known, its good/bad kind is not.
        logged = {"block": total_delta["block"]}
    return logged


def _verdict(logged, change, before, after):
    """``(verdict, reason)`` from the logged totals and the magnitude change."""
    if any(d < 0 for d in logged.values()):
        return INCONSISTENT, "a session total rolled back"
    total = sum(logged.values())
    if change is None:
        return UNVERIFIABLE, "the magnitudes needed to compare the two sides are missing"
    if total == 0:
        if change:
            return INCONSISTENT, "the magnitudes changed but no change was logged"
        return NO_CHANGE, "no change logged and the magnitudes did not change"
    if not change:
        return INCONSISTENT, "a change was logged but the zero arm / magnitudes did not change"
    good_only = logged.get("good", 0) > 0 and logged.get("bad", 0) == 0
    if good_only and best_changed(before, after) is False:
        return INCONSISTENT, "a good reversal was logged but the best tower did not change"
    return CONSISTENT, "the logged change shows in the magnitudes"


def describe_boundary(before, after, mode):
    """Describe the gap between the last trial of one recording and the first of the next.

    Args:
        before (dict): The earlier side, as ``{"magnitudes": {tower: magnitude} | None,
            "totals": {kind: session total | None}, "counters": {kind: per-recording counter |
            None}}`` at its last trial (the totals as of that trial: ``run_start`` total plus
            the counter's progress since).
        after (dict): The later side, in the same layout, at its first trial (the totals of its
            ``run_start``). The kinds are ``"block"`` (``tot_num_blocks`` / ``num_blocks``),
            ``"good"`` and ``"bad"``.
        mode (str | None): ``"drifting"`` (block changes, zero arm moves) or ``"blocked"``
            (good and bad reversals, magnitudes permute); ``None`` if the task is unknown, in
            which case the magnitudes cannot be compared.

    Returns:
        dict: ``mode``; ``magnitudes_before`` / ``magnitudes_after``; ``counter_delta`` and
        ``total_delta`` (``{"block", "good", "bad"}`` of after minus before, ``None`` where a
        side does not log it; the counters restart in every recording, so only the totals count
        changes); ``logged_changes`` (the changes the totals log, ``None`` when no total is
        logged); ``source`` (``"totals"``, or ``"magnitudes"`` when the totals are absent and the
        magnitudes are the only evidence); ``magnitude_change`` (``True`` / ``False`` / ``None``,
        see :func:`magnitude_change`); ``verdict`` (``"consistent"``, ``"no change"``,
        ``"inconsistent"`` or ``"unverifiable"``) and a ``reason`` sentence.
    """
    magnitudes_before, magnitudes_after = before.get("magnitudes"), after.get("magnitudes")
    counter_delta = _delta(before.get("counters", {}), after.get("counters", {}))
    total_delta = _delta(before.get("totals", {}), after.get("totals", {}))
    change = magnitude_change(magnitudes_before, magnitudes_after, mode)
    logged = _logged(total_delta, mode)
    if logged:
        verdict, reason = _verdict(logged, change, magnitudes_before, magnitudes_after)
    elif change is None:
        verdict, reason = UNVERIFIABLE, "no totals are logged and the magnitudes cannot be compared"
    elif change:
        verdict, reason = CONSISTENT, "no totals are logged; the magnitudes show a block change"
    else:
        verdict, reason = NO_CHANGE, "no totals are logged and the magnitudes did not change"
    return {
        "mode": mode,
        "magnitudes_before": magnitudes_before,
        "magnitudes_after": magnitudes_after,
        "counter_delta": counter_delta,
        "total_delta": total_delta,
        "logged_changes": sum(logged.values()) if logged else None,
        "source": "totals" if logged else "magnitudes",
        "magnitude_change": change,
        "verdict": verdict,
        "reason": reason,
    }
