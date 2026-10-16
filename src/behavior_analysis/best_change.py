"""Check that a good reversal changes the best-rewarding tower.

Shared by the windowing of good reversals and by the counts of suspect reversals, so both
apply the same rule:

- *suspect*: checked, and the best tower is unchanged or tied (:func:`is_suspect`);
- *uncheckable*: a magnitude is missing (:func:`is_uncheckable`); counts also treat a reversal
  with no trial to check against as uncheckable (see
  :mod:`src.behavior_analysis.suspect_reversals`). A reversal logged while nothing was
  recording is checked on the last trial before it and the first trial after it, like any other.
"""

MISSING_MAGNITUDE = "missing magnitude"
TIE_FOR_BEST = "tie for best"
BEST_UNCHANGED = "best tower unchanged"
SUSPECT_PROBLEMS = (TIE_FOR_BEST, BEST_UNCHANGED)


def has_tie(magnitudes):
    """True if more than one tower has the highest magnitude.

    Args:
        magnitudes (dict): ``{tower: magnitude}``; values must be comparable
            (no ``None``).

    Returns:
        bool: Whether the maximum magnitude occurs for more than one tower.
    """
    values = list(magnitudes.values())
    return values.count(max(values)) > 1


def best_change_problem(before, after):
    """Why a reversal is suspect, or ``None`` when the best tower changes.

    Args:
        before (dict): ``{tower: magnitude}`` on the trial before the reversal.
        after (dict): ``{tower: magnitude}`` on the reversal trial. A magnitude of ``None`` means
            it was not logged.

    Returns:
        str | None: ``"missing magnitude"`` if any magnitude is ``None`` (the best tower is
        unknown), ``"tie for best"`` or ``"best tower unchanged"`` if the first-listed best tower is
        the same on both sides (the tie wording tells whoever investigates that a tie, not an
        off-by-one, may be behind it), else ``None``.
    """
    if any(v is None for v in (*before.values(), *after.values())):
        return MISSING_MAGNITUDE
    if max(before, key=before.get) != max(after, key=after.get):
        return None
    return TIE_FOR_BEST if has_tie(before) or has_tie(after) else BEST_UNCHANGED


def is_suspect(problem):
    """True if a :func:`best_change_problem` result says the best tower did not change.

    Args:
        problem (str | None): Result of :func:`best_change_problem`.

    Returns:
        bool: ``True`` for ``"tie for best"`` or ``"best tower unchanged"``.
    """
    return problem in SUSPECT_PROBLEMS


def is_uncheckable(problem):
    """True if a :func:`best_change_problem` result says a magnitude is missing.

    Args:
        problem (str | None): Result of :func:`best_change_problem`.

    Returns:
        bool: ``True`` for ``"missing magnitude"``.
    """
    return problem == MISSING_MAGNITUDE
