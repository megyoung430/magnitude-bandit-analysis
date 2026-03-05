"""Split good reversal windows by the direction of the best-arm reward change.

After a good reversal the arm that was previously best either drops to second
place or drops all the way to third place.  The functions here classify each
reversal accordingly and partition the windows into two separate dicts for
independent downstream analysis.
"""
from copy import deepcopy


def split_good_reversals_by_best_change(reversal_windows):
    """Partition good-reversal windows by whether the best arm became second or third.

    Args:
        reversal_windows: ``{subject: list[reversal_dict]}`` as returned by
            :func:`src.behavior_analysis.get_good_reversal_info.get_good_reversal_info`.

    Returns:
        A two-tuple ``(best_to_second, best_to_third)`` where each element is
        a ``{subject: list[reversal_dict]}`` dict.  Each reversal dict in the
        output has an additional ``"best_change_type"`` key set to either
        ``"best_to_second"`` or ``"best_to_third"``.  Reversals that cannot
        be classified are excluded from both outputs.
    """
    best_to_second = {}
    best_to_third  = {}

    for subj, revs in reversal_windows.items():
        s_list, t_list = [], []
        for r in revs:
            label = classify_best_change(r)
            if label == "best_to_second":
                r2 = deepcopy(r)
                r2["best_change_type"] = label
                s_list.append(r2)
            elif label == "best_to_third":
                r2 = deepcopy(r)
                r2["best_change_type"] = label
                t_list.append(r2)

        best_to_second[subj] = s_list
        best_to_third[subj]  = t_list

    return best_to_second, best_to_third

def classify_best_change(rev, best_val=4, second_val=1, third_val=0):
    """Classify the direction of change for the previously-best arm at a reversal.

    Identifies the arm with the highest reward magnitude immediately before the
    reversal and checks what magnitude it received after the reversal.

    Args:
        rev: A reversal dict containing ``"reward_magnitudes_by_tower_before"``
            and ``"reward_magnitudes_by_tower_after"`` keys.
        best_val: Expected magnitude of the best arm (default: 4).
        second_val: Expected magnitude of the second-best arm (default: 1).
        third_val: Expected magnitude of the third arm (default: 0).

    Returns:
        ``"best_to_second"`` if the previously-best arm's post-reversal
        magnitude equals *second_val*, ``"best_to_third"`` if it equals
        *third_val*, or ``None`` if the values are missing or the pattern
        is unexpected.
    """
    before = rev.get("reward_magnitudes_by_tower_before", {})
    after  = rev.get("reward_magnitudes_by_tower_after", {})
    if not before or not after:
        return None

    prev_best = max(before, key=before.get)
    new_val = after.get(prev_best, None)

    if new_val == second_val:
        return "best_to_second"
    if new_val == third_val:
        return "best_to_third"
    return None