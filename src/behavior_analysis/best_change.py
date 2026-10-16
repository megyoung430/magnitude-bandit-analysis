"""Check that a block change or good reversal shows in the reward magnitudes.

Shared by the windowing of good reversals and by the counts of suspect reversals, so both
apply the same rule. The rule depends on the task:

- blocked task: a good reversal must change the best-rewarding tower;
- drifting task: at a block change the most-chosen arm drops to magnitude 0, and that arm can
  be the best or the second best, so the best tower often stays the best. The check is that the
  zero arm moves.

A reversal is

- *suspect*: checked, and the best tower is unchanged or tied (blocked), or the zero arm did
  not move (drifting); see :func:`is_suspect`;
- *uncheckable*: a magnitude is missing (:func:`is_uncheckable`); counts also treat a reversal
  with no trial to check against as uncheckable (see
  :mod:`src.behavior_analysis.suspect_reversals`). A reversal logged while nothing was
  recording is checked on the last trial before it and the first trial after it, like any other.
"""

from src.behavior_import.problem_numbering import DRIFTING
from src.behavior_import.run_boundaries import event_matches

MISSING_MAGNITUDE = "missing magnitude"
TIE_FOR_BEST = "tie for best"
BEST_UNCHANGED = "best tower unchanged"
ZERO_ARM_UNCHANGED = "zero arm unchanged"
SUSPECT_PROBLEMS = (TIE_FOR_BEST, BEST_UNCHANGED, ZERO_ARM_UNCHANGED)


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


def zero_arm_problem(before, after):
    """Why a drifting block change is suspect, or ``None`` when the zero arm moves.

    Args:
        before (dict): ``{tower: magnitude}`` on the trial before the block change.
        after (dict): ``{tower: magnitude}`` on the block change trial. A magnitude of ``None``
            means it was not logged.

    Returns:
        str | None: ``"missing magnitude"`` if any magnitude is ``None`` (the zero arm is
        unknown), ``"zero arm unchanged"`` if the same towers are at 0 on both sides, else
        ``None``.

    Raises:
        ValueError: If the two sides list different towers.
    """
    if any(v is None for v in (*before.values(), *after.values())):
        return MISSING_MAGNITUDE
    moved = event_matches("block", before, after, DRIFTING)
    if moved is None:
        raise ValueError(f"cannot compare the towers {sorted(before)} with {sorted(after)}")
    return None if moved else ZERO_ARM_UNCHANGED


def block_change_problem(before, after, task_type):
    """Why a block change is suspect for the task, or ``None`` when it shows in the magnitudes.

    Args:
        before (dict): ``{tower: magnitude}`` on the trial before the change.
        after (dict): ``{tower: magnitude}`` on the change trial; ``None`` for a magnitude that
            was not logged.
        task_type (str | None): ``"drifting"`` (:func:`zero_arm_problem`), or ``"blocked"`` /
            ``None`` for the best-tower rule (:func:`best_change_problem`). ``None`` is a
            session that carries no task information, such as hand-built data.

    Returns:
        str | None: As the rule that applies.
    """
    if task_type == DRIFTING:
        return zero_arm_problem(before, after)
    return best_change_problem(before, after)


def is_suspect(problem):
    """True if a check result says the best tower (blocked) or the zero arm (drifting) is unchanged.

    Args:
        problem (str | None): Result of :func:`best_change_problem` or
            :func:`block_change_problem`.

    Returns:
        bool: ``True`` for ``"tie for best"``, ``"best tower unchanged"`` or
        ``"zero arm unchanged"``.
    """
    return problem in SUSPECT_PROBLEMS


def is_uncheckable(problem):
    """True if a check result says a magnitude is missing.

    Args:
        problem (str | None): Result of :func:`best_change_problem` or
            :func:`block_change_problem`.

    Returns:
        bool: ``True`` for ``"missing magnitude"``.
    """
    return problem == MISSING_MAGNITUDE
