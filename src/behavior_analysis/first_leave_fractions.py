"""Per-subject first-leave fractions for plotting (no plotting)."""

from __future__ import annotations


def first_leave_fractions(counts: dict) -> tuple[float, float] | None:
    """Fractions of a subject's good reversals that were left toward the new best / third arm.

    Args:
        counts (dict): ``{"new_best": int, "third": int, "total": int}`` for one subject, as in
            :func:`src.behavior_analysis.get_first_leave_after_good_reversals.get_first_leave_after_good_reversals`.

    Returns:
        tuple[float, float] | None: ``(new_best_fraction, third_fraction)``, or ``None`` if the subject had no
        reversals (``total == 0``).
    """
    if counts["total"] == 0:
        return None
    return counts["new_best"] / counts["total"], counts["third"] / counts["total"]
