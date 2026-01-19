import numpy as np

def get_choice_probs_around_good_reversals(reversal_windows, pre=10, post=40):
    """
    NaN-padded aggregation of good reversals.
    Returns:
      x: np.array of relative trial indices, length pre+post ([-pre ... post-1])
      per_subject: dict[subj] -> dict with keys:
        - "prev_best": (num_reversals x T) np.array
        - "next_best": (num_reversals x T) np.array
        - "third": (num_reversals x T) np.array
        - "prev_best_mean": np.array
        - "next_best_mean": np.array
        - "third_mean": np.array
        - "num_reversals": int
      across: dict with keys:
        - "mean": dict[label]->np.array
        - "se":  dict[label]->np.array
        - "num_subjects": int
    """
    T = pre + post
    x = np.arange(-pre, post)
    per_subject = {}

    for subj, revs in reversal_windows.items():
        if not revs:
            print(f"[INFO] Skipping subject {subj} with no good reversals.")
            continue

        prev_mat, next_mat, third_mat = [], [], []

        for r in revs:
            prev_best, next_best, third = classify_towers_at_good_reversals(r)

            prev_raw  = r["choices_by_tower"][prev_best]["pre"] + r["choices_by_tower"][prev_best]["post"]
            next_raw  = r["choices_by_tower"][next_best]["pre"] + r["choices_by_tower"][next_best]["post"]
            third_raw = r["choices_by_tower"][third]["pre"] + r["choices_by_tower"][third]["post"]

            # --- NaN pad to length T ---
            def pad(arr):
                out = np.full(T, np.nan, dtype=float)
                n = min(len(arr), T)
                out[:n] = arr[:n]
                return out

            prev_mat.append(pad(prev_raw))
            next_mat.append(pad(next_raw))
            third_mat.append(pad(third_raw))

        if len(prev_mat) == 0:
            print(f"[INFO] Skipping subject {subj} with no good reversals.")
            continue

        prev_mat  = np.vstack(prev_mat)
        next_mat  = np.vstack(next_mat)
        third_mat = np.vstack(third_mat)

        per_subject[subj] = {
            "prev_best": prev_mat,
            "next_best": next_mat,
            "third": third_mat,
            "prev_best_mean": np.nanmean(prev_mat, axis=0),
            "next_best_mean": np.nanmean(next_mat, axis=0),
            "third_mean": np.nanmean(third_mat, axis=0),
            "num_reversals": prev_mat.shape[0],
        }

    subj_list = list(per_subject.keys())
    num_subjects = len(subj_list)

    if num_subjects == 0:
        return x, per_subject, {
            "mean": {"prev_best": None, "next_best": None, "third": None},
            "se":  {"prev_best": None, "next_best": None, "third": None},
            "num_subjects": 0
        }

    # --- Across-subject mean/se ---
    across_mean = {}
    across_se = {}

    mean_key_map = {
        "prev_best": "prev_best_mean",
        "next_best": "next_best_mean",
        "third": "third_mean",
    }

    for k, mk in mean_key_map.items():
        stack = np.vstack([per_subject[subj][mk] for subj in subj_list])
        across_mean[k] = np.nanmean(stack, axis=0)
        across_se[k]  = (
            np.nanstd(stack, axis=0, ddof=1) / np.sqrt(num_subjects)
            if num_subjects > 1 else np.zeros(T)
        )

    across = {
        "mean": across_mean,
        "se": across_se,
        "num_subjects": num_subjects,
        "num_reversals": sum(per_subject[subj]["num_reversals"] for subj in subj_list)
    }

    # --- Check: probabilities sum to 1 ---
    assert np.isclose(across["mean"]["prev_best"] + across["mean"]["next_best"] + across["mean"]["third"], 1.0, atol=1e-6).all(), "Choice probabilities do not sum to 1."

    return x, per_subject, across

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