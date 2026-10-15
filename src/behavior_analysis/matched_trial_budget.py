"""Match problem exposure using the first N chronological trials per mouse."""

import math
from copy import deepcopy

from src.behavior_import.session_order import ordered_session_keys

# Analysis fields copied into a matched session; raw dataframes are left out.
_COPIED_FIELDS = (
    "trial",
    "blocks",
    "trials_in_block",
    "good_reversals",
    "bad_reversals",
    "reward_magnitudes_by_tower",
    "choices_by_tower",
    "choices_by_rank",
    "has_good",
    "has_bad",
    "reversal_totals",
    "reversal_boundary_events",
)
_PER_TRIAL_FIELDS = ("trial", "blocks", "trials_in_block", "good_reversals", "bad_reversals")
_PER_TOWER_FIELDS = ("reward_magnitudes_by_tower", "choices_by_tower", "choices_by_rank")


def _trial_counts(subjects: dict) -> dict:
    """``{mouse: number of trials over all its sessions}``."""
    return {
        mouse: sum(len(s.get("trial", [])) for s in sessions.values())
        for mouse, sessions in subjects.items()
    }


def _reference_budget(counts: dict, *, subject_numbers: dict, reference_numbers) -> tuple:
    """``(reference trial counts, their mean, floored mean)`` of the reference mice."""
    reference = {m: n for m, n in counts.items() if subject_numbers[m] in reference_numbers}
    missing = set(reference_numbers) - {subject_numbers[m] for m in reference}
    if missing or not reference or any(n == 0 for n in reference.values()):
        raise ValueError(
            f"Reference mice must all have trials; missing subject numbers: {sorted(missing)}"
        )
    mean = sum(reference.values()) / len(reference)
    return reference, mean, math.floor(mean)


def _resolve_budget(counts, *, subject_numbers, reference_numbers, trial_budget) -> tuple:
    """``(budget, reference trial counts, reference mean)`` from a fixed or reference budget."""
    reference, mean = {}, None
    if trial_budget is None:
        reference, mean, budget = _reference_budget(
            counts, subject_numbers=subject_numbers, reference_numbers=reference_numbers
        )
    else:
        # bool is an int subclass, but True is not a trial count
        if not isinstance(trial_budget, int) or isinstance(trial_budget, bool):
            raise ValueError("trial_budget must be a positive integer")
        budget = trial_budget
    if budget < 1:
        raise ValueError("Trial budget must be positive")
    return budget, reference, mean


def _partial_reversal_totals(source: dict, take: int) -> dict:
    """Reversal totals of a session cut after *take* trials.

    Each total is the counter at the next trial boundary (index *take*), so a
    reversal right after the last retained trial counts and later ones do not.
    """
    totals = {}
    for kind in ("good", "bad"):
        counter = source.get(kind + "_reversals", [])
        if counter:
            totals[kind] = next((v for v in reversed(counter[: take + 1]) if v is not None), 0)
    return totals


def _truncate_session(source: dict, take: int, *, is_partial: bool) -> dict:
    """Copy of the analysis fields of *source* cut to its first *take* trials."""
    target = {k: deepcopy(source[k]) for k in _COPIED_FIELDS if k in source}
    target["trial_info"] = True
    for field in _PER_TRIAL_FIELDS:
        if field in target:
            target[field] = target[field][:take]
    for field in _PER_TOWER_FIELDS:
        target[field] = {k: v[:take] for k, v in target.get(field, {}).items()}
    if is_partial:
        target["reversal_totals"] = _partial_reversal_totals(source, take)
        events = target.get("reversal_boundary_events", [])
        target["reversal_boundary_events"] = [e for e in events if e["index"] <= take]
    return target


def _retain_trials(mouse: str, sessions: dict, budget: int) -> dict:
    """``{session key: truncated session}`` for the first *budget* trials, in session order."""
    left = budget
    kept = {}
    for key in ordered_session_keys(mouse, sessions):
        if left <= 0:
            break
        source = sessions[key]
        n = len(source.get("trial", []))
        if not n:
            continue
        take = min(n, left)
        kept[key] = _truncate_session(source, take, is_partial=take < n)
        left -= take
    return kept


def match_trial_budget(
    subjects, subject_numbers, reference_numbers=(5, 6, 7, 8), *, trial_budget=None
):
    """Use a fixed ``trial_budget`` or the floored reference-mouse mean as a cap.

    Sessions are taken in date-then-number order (:mod:`src.behavior_import.session_order`);
    a mouse whose session numbers and dates disagree raises ``ValueError``.
    Mice with fewer trials retain all available trials and are flagged in the
    report. Full sessions retain terminal reversal counts; a partial session's
    totals use the counter at the next trial boundary, including a reversal
    after its final retained trial, but excluding later reversals.

    Args:
        subjects: ``{mouse: {session_key: session_dict}}``.
        subject_numbers: ``{mouse: subject number}``.
        reference_numbers: Subject numbers whose mean trial count sets the cap.
        trial_budget: Fixed positive int cap; ``None`` uses the reference mean.

    Returns:
        ``(matched, report)``: *matched* has the same layout as *subjects* and
        omits mice without trials; *report* holds the reference counts and mean,
        the budget, and one row per mouse under ``"mice"``.
    """
    counts = _trial_counts(subjects)
    budget, reference, mean = _resolve_budget(
        counts,
        subject_numbers=subject_numbers,
        reference_numbers=reference_numbers,
        trial_budget=trial_budget,
    )
    result, rows = {}, []
    for mouse, sessions in subjects.items():
        kept = _retain_trials(mouse, sessions, budget)
        if kept:
            result[mouse] = kept
        rows.append(
            dict(
                mouse=mouse,
                subject_number=subject_numbers[mouse],
                available_trials=counts[mouse],
                retained_trials=min(budget, counts[mouse]),
                below_budget=counts[mouse] < budget,
                reference_mouse=mouse in reference,
            )
        )
    report = dict(
        reference_trial_counts=reference, reference_mean=mean, trial_budget=budget, mice=rows
    )
    return result, report
