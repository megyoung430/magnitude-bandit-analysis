import numpy as np

def get_choice_probs_around_good_reversals(reversal_windows, pre=10, post=40, skip_n_trials_after_reversal=0):

    def keep_zero_then_skip(post_list, skip):
        if skip <= 0:
            return post_list
        return post_list[:1] + post_list[skip:]

    if skip_n_trials_after_reversal < 0:
        raise ValueError("skip_n_trials_after_reversal must be >= 0")
    if skip_n_trials_after_reversal >= post:
        raise ValueError("skip_n_trials_after_reversal must be < post (otherwise post window is empty).")

    T = pre + (post - skip_n_trials_after_reversal)

    x = np.concatenate([
        np.arange(-pre, 0),
        np.arange(0, post - skip_n_trials_after_reversal)
    ])

    per_subject = {}

    for subj, revs in reversal_windows.items():
        if not revs:
            print(f"[INFO] Skipping subject {subj} with no good reversals.")
            continue

        prev_mat, next_mat, third_mat = [], [], []

        for r in revs:
            prev_best, next_best, third = classify_towers_at_good_reversals(r)

            prev_pre   = r["choices_by_tower"][prev_best]["pre"]
            prev_post  = keep_zero_then_skip(r["choices_by_tower"][prev_best]["post"], skip_n_trials_after_reversal)

            next_pre   = r["choices_by_tower"][next_best]["pre"]
            next_post  = keep_zero_then_skip(r["choices_by_tower"][next_best]["post"], skip_n_trials_after_reversal)

            third_pre  = r["choices_by_tower"][third]["pre"]
            third_post = keep_zero_then_skip(r["choices_by_tower"][third]["post"], skip_n_trials_after_reversal)

            prev_raw  = prev_pre  + prev_post
            next_raw  = next_pre  + next_post
            third_raw = third_pre + third_post

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

        if num_subjects > 1:
            across_se[k] = np.nanstd(stack, axis=0, ddof=1) / np.sqrt(num_subjects)
        else:
            across_se[k] = np.zeros(T)

    across = {
        "mean": across_mean,
        "se": across_se,
        "num_subjects": num_subjects,
        "num_reversals": sum(per_subject[subj]["num_reversals"] for subj in subj_list)
    }

    s = across["mean"]["prev_best"] + across["mean"]["next_best"] + across["mean"]["third"]
    finite = np.isfinite(s)
    assert finite.any(), "No finite bins to validate (all-NaN after padding/filtering)."
    assert np.isclose(s[finite], 1.0, atol=1e-6).all(), "Choice probabilities do not sum to 1 (finite bins)."

    return x, per_subject, across

def classify_towers_at_good_reversals(reversal):
    before = reversal["reward_magnitudes_by_tower_before"]
    after  = reversal["reward_magnitudes_by_tower_after"]

    prev_best = max(before, key=before.get)
    next_best = max(after, key=after.get)

    towers = set(before.keys())
    third = list(towers - {prev_best, next_best})[0]

    return prev_best, next_best, third