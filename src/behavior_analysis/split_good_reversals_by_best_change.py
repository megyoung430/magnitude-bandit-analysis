from copy import deepcopy

def split_good_reversals_by_best_change(reversal_windows):
    """
    Input: dict[subj] -> list[rev_dict]
    Output: (best_to_second_dict, best_to_third_dict)
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
    """
    Returns: "best_to_second" | "best_to_third" | None
    None means we can't classify (missing values, unexpected pattern, etc.)
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