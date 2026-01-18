import numpy as np

def get_first_leave_after_good_reversals(reversal_windows):
    """
    Returns per_subject counts:
      per_subject[subj] = {
        "new_best": int,
        "third": int,
        "total": int
      }
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
    """
    Returns:
      mean: dict[label] -> float
      std:  dict[label] -> float
      n_subjects: int
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
    std = {
        "new_best": np.std(new_vals, ddof=1) if len(new_vals) > 1 else 0.0,
        "third": np.std(third_vals, ddof=1) if len(third_vals) > 1 else 0.0,
    }

    return mean, std, len(new_vals)

# ========== Classifying Towers at Good Reversals ==========
def classify_towers_at_good_reversals(reversal):
    """
    Returns:
      prev_best, next_best, third
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
    """
    Given a trial index into the concatenated window and towers,
    infer which tower was chosen using one-hot arrays in choices_by_tower.
    Returns tower name or None if cannot infer.
    """
    vals = []
    for t in towers:
        vals.append(reversal["choices_by_tower"][t]["post"][trial_idx])
    if all(v is None for v in vals):
        return None
    j = int(np.nanargmax(np.asarray(vals, dtype=float)))
    return towers[j]