"""Check that a good reversal changes the best-rewarding tower.

Shared by the windowing of good reversals and by the counts of suspect reversals, so both
apply the same rule.
"""


def has_tie(magnitudes):
    """True if more than one tower has the highest magnitude."""
    values = list(magnitudes.values())
    return values.count(max(values)) > 1


def best_change_problem(before, after):
    """Why a reversal is suspect, or ``None`` when the best tower changes.

    Args:
        before: ``{tower: magnitude}`` on the trial before the reversal.
        after: ``{tower: magnitude}`` on the reversal trial. A magnitude of ``None`` means
            it was not logged.

    Returns:
        ``"missing magnitude"`` if any magnitude is ``None`` (the best tower is unknown),
        ``"tie for best"`` or ``"best tower unchanged"`` if the first-listed best tower is
        the same on both sides (the tie wording tells whoever investigates that a tie, not an
        off-by-one, may be behind it), else ``None``.
    """
    if any(v is None for v in (*before.values(), *after.values())):
        return "missing magnitude"
    if max(before, key=before.get) != max(after, key=after.get):
        return None
    return "tie for best" if has_tie(before) or has_tie(after) else "best tower unchanged"
