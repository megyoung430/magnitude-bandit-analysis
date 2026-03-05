"""Identify and aggregate the first arm chosen after leaving the previous best arm.

After a good reversal the animal must eventually stop visiting the arm that was
previously the best.  The functions here find the very first trial in the post
window where a non-previous-best arm is chosen, classify it as new-best or
third, and aggregate these classifications across reversals and subjects.
"""
import numpy as np


def get_first_leave_after_good_reversals(reversal_windows):
    """Count first-leave directions (new-best vs third arm) across good reversals.

    For each reversal in each subject's window data, iterates through the
    post-reversal trials to find the first trial on which the animal chose an
    arm other than the previous best, then records whether that arm was the
    new-best or the third arm.

    Args:
        reversal_windows: ``{subject: list[reversal_dict]}`` as returned by
            :func:`src.behavior_analysis.get_good_reversal_info.get_good_reversal_info`.

    Returns:
        Dict ``{subject: {"new_best": int, "third": int, "total": int}}``
        where ``"total"`` is the number of reversals examined, ``"new_best"``
        is the count where the animal first left toward the new best arm, and
        ``"third"`` is the count where it left toward the third arm.
    """
    per_subject = {}

    for subj, revs in reversal_windows.items():
        counts = {"new_best": 0, "third": 0, "total": 0}

        for r in revs:
            prev_best, new_best, third = classify_towers_at_good_reversals(r)
            towers = [prev_best, new_best, third]

            post_len = len(r["choices_by_tower"][prev_best]["post"])

            first_non_prev = None
            for j in range(post_len):
                chosen = chosen_tower_from_onehots(r, j, towers)
                if chosen is None:
                    continue
                if chosen != prev_best:
                    first_non_prev = chosen
                    break

            counts["total"] += 1
            if first_non_prev == new_best:
                counts["new_best"] += 1
            elif first_non_prev == third:
                counts["third"] += 1
        per_subject[subj] = counts
    return per_subject

def average_first_leave_across_subjects(per_subject_counts):
    """Compute mean and SE of first-leave fractions across subjects.

    Args:
        per_subject_counts: ``{subject: {"new_best": int, "third": int,
            "total": int}}`` as returned by
            :func:`get_first_leave_after_good_reversals`.

    Returns:
        A three-tuple ``(mean, se, n_subjects)`` where:

        - ``mean`` – dict with keys ``"new_best"``, ``"third"``, and
          ``"num_reversals"`` (total reversals pooled across all subjects).
        - ``se`` – dict with keys ``"new_best"`` and ``"third"``, each a
          float standard error across subjects.
        - ``n_subjects`` – number of subjects with at least one reversal.
    """
    new_vals = []
    third_vals = []

    for subj, c in per_subject_counts.items():
        if c["total"] == 0:
            continue
        new_vals.append(c["new_best"] / c["total"])
        third_vals.append(c["third"] / c["total"])

    mean = {
        "new_best": np.mean(new_vals),
        "third": np.mean(third_vals),
        "num_reversals": np.sum([c["total"] for c in per_subject_counts.values()]),
    }
    se = {
        "new_best": np.std(new_vals, ddof=1) / np.sqrt(len(new_vals)) if len(new_vals) > 1 else 0.0,
        "third": np.std(third_vals, ddof=1) / np.sqrt(len(third_vals)) if len(third_vals) > 1 else 0.0,
    }

    return mean, se, len(new_vals)

# ========== Classifying Towers at Good Reversals ==========
def classify_towers_at_good_reversals(reversal):
    """Classify towers as prev-best, next-best, and third around a good reversal.

    Args:
        reversal: A reversal dict containing ``"reward_magnitudes_by_tower_before"``
            and ``"reward_magnitudes_by_tower_after"`` keys (tower → magnitude).

    Returns:
        Three-tuple ``(prev_best, next_best, third)`` of tower key strings.
    """
    before = reversal["reward_magnitudes_by_tower_before"]
    after  = reversal["reward_magnitudes_by_tower_after"]

    prev_best = max(before, key=before.get)
    next_best = max(after, key=after.get)

    towers = set(before.keys())
    third = list(towers - {prev_best, next_best})[0]

    return prev_best, next_best, third

# ========== Get Chosen Tower from Dictionary of One Hots ==========
def chosen_tower_from_onehots(reversal, trial_idx, towers):
    """Infer the chosen tower at a given post-window trial from one-hot arrays.

    Reads the one-hot value for each tower from ``reversal["choices_by_tower"]``
    at position *trial_idx* in the post list and returns the tower with the
    highest value.

    Args:
        reversal: A reversal dict containing a ``"choices_by_tower"`` key.
        trial_idx: Integer index into the post-window lists.
        towers: Ordered list of tower key strings to check.

    Returns:
        The tower key string with the highest one-hot value, or ``None`` if
        all values are ``None`` (trial not recorded).
    """
    vals = []
    for t in towers:
        vals.append(reversal["choices_by_tower"][t]["post"][trial_idx])
    if all(v is None for v in vals):
        return None
    j = int(np.nanargmax(np.asarray(vals, dtype=float)))
    return towers[j]