from copy import deepcopy
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

def apply_moving_average_to_choice_probs(x, per_subject, moving_avg_window=5, mode="centered",
    use_keys=("prev_best_mean", "next_best_mean", "third_mean"),
    out_keys=("prev_best_mean_sm", "next_best_mean_sm", "third_mean_sm"),
    split_at=0):
    """
    Applies NaN-aware moving average separately to pre (x < split_at) and post (x >= split_at)
    so there is no smoothing bleedthrough across the reversal boundary.
    """

    def nan_moving_average(y, window, mode="centered"):
        if window is None or window <= 1:
            return np.asarray(y, dtype=float)

        y = np.asarray(y, dtype=float)
        w = int(window)
        if w < 1:
            return y

        valid = np.isfinite(y).astype(float)
        y0 = np.nan_to_num(y, nan=0.0)
        kernel = np.ones(w, dtype=float)

        if mode == "trailing":
            num = np.convolve(y0, kernel, mode="full")[:len(y0)]
            den = np.convolve(valid, kernel, mode="full")[:len(y0)]
        else:
            num = np.convolve(y0, kernel, mode="same")
            den = np.convolve(valid, kernel, mode="same")

        out = np.full_like(y0, np.nan, dtype=float)
        m = den > 0
        out[m] = num[m] / den[m]
        return out

    pre_mask = x < split_at
    post_mask = ~pre_mask

    def smooth_pre_post(y):
        """Smooth y separately on pre and post segments and stitch back together."""
        y = np.asarray(y, dtype=float)
        out = np.full_like(y, np.nan, dtype=float)

        if pre_mask.any():
            out[pre_mask] = nan_moving_average(y[pre_mask], moving_avg_window, mode=mode)
        if post_mask.any():
            out[post_mask] = nan_moving_average(y[post_mask], moving_avg_window, mode=mode)

        return out

    per_subject_moving_avg = {}
    for subj, d in per_subject.items():
        d2 = dict(d)

        if any(k not in d2 for k in use_keys):
            continue

        for src, dst in zip(use_keys, out_keys):
            d2[dst] = smooth_pre_post(d2[src])

        per_subject_moving_avg[subj] = d2

    subj_list = list(per_subject_moving_avg.keys())
    n_subjects = len(subj_list)
    T = len(x)

    if n_subjects == 0:
        across_moving_avg = {
            "mean": {"prev_best": None, "next_best": None, "third": None},
            "se":   {"prev_best": None, "next_best": None, "third": None},
            "num_subjects": 0,
            "num_reversals": 0,
            "moving_avg_window": moving_avg_window,
            "mode": mode,
            "split_at": split_at,
        }
        return x, per_subject_moving_avg, across_moving_avg

    stack_prev  = np.vstack([per_subject_moving_avg[s][out_keys[0]] for s in subj_list])
    stack_next  = np.vstack([per_subject_moving_avg[s][out_keys[1]] for s in subj_list])
    stack_third = np.vstack([per_subject_moving_avg[s][out_keys[2]] for s in subj_list])

    mean_prev  = np.nanmean(stack_prev, axis=0)
    mean_next  = np.nanmean(stack_next, axis=0)
    mean_third = np.nanmean(stack_third, axis=0)

    if n_subjects > 1:
        se_prev  = np.nanstd(stack_prev, axis=0, ddof=1) / np.sqrt(n_subjects)
        se_next  = np.nanstd(stack_next, axis=0, ddof=1) / np.sqrt(n_subjects)
        se_third = np.nanstd(stack_third, axis=0, ddof=1) / np.sqrt(n_subjects)
    else:
        se_prev = se_next = se_third = np.zeros(T)

    across_moving_avg = {
        "mean": {"prev_best": mean_prev, "next_best": mean_next, "third": mean_third},
        "se":   {"prev_best": se_prev,   "next_best": se_next,   "third": se_third},
        "num_subjects": n_subjects,
        "num_reversals": sum(per_subject_moving_avg[s].get("num_reversals", 0) for s in subj_list),
        "moving_avg_window": moving_avg_window,
        "mode": mode,
        "split_at": split_at,
    }

    s = (across_moving_avg["mean"]["prev_best"] + across_moving_avg["mean"]["next_best"] + across_moving_avg["mean"]["third"])
    finite = np.isfinite(s)
    if finite.any():
        assert np.isclose(s[finite], 1.0, atol=1e-6).all(), \
            "Smoothed probabilities do not sum to 1 (finite bins)."

    return x, per_subject_moving_avg, across_moving_avg

def remove_trials_after_bad_rev(good_windows, all_good_idx, all_bad_idx, include_bad_trial=True):
    """
    Stops each good reversal's post window at the first bad reversal between it and the next good reversal.
    include_bad_trial: if True, keep trial b in post; if False, cut just before b
    """

    out = {}

    for subj, good_revs in good_windows.items():
        revs_sorted = sorted(good_revs, key=lambda r: r.get("reversal_idx", 0))
        goods = sorted(all_good_idx.get(subj, []))
        bads  = sorted(all_bad_idx.get(subj, []))

        if not revs_sorted:
            out[subj] = []
            continue

        subj_out = []
        for r in revs_sorted:
            r2 = deepcopy(r)
            g = r2.get("reversal_idx", None)
            if g is None:
                r2["removed_bad_idx"] = None
                r2["post_len_after_removal"] = None
                subj_out.append(r2)
                continue

            next_g = None
            for gg in goods:
                if gg > g:
                    next_g = gg
                    break
            if next_g is None:
                next_g = float("inf")

            cutoff = None
            for b in bads:
                if not (g < b < next_g):
                    continue
                cutoff = b
                break

            if cutoff is None:
                r2["removed_bad_idx"] = None
                r2["post_len_after_removal"] = None
                subj_out.append(r2)
                continue

            cut_len = (cutoff - g) + (1 if include_bad_trial else 0)

            r2["removed_bad_idx"] = cutoff
            r2["post_len_after_removal"] = cut_len

            if cut_len < 1:
                subj_out.append(r2)
                continue

            for t, dct in r2.get("choices_by_tower", {}).items():
                if isinstance(dct, dict) and "post" in dct:
                    dct["post"] = dct["post"][:cut_len]

            for rk, dct in r2.get("choices_by_rank", {}).items():
                if isinstance(dct, dict) and "post" in dct:
                    dct["post"] = dct["post"][:cut_len]

            tw = r2.get("trial_window_idx", {})
            if isinstance(tw, dict) and isinstance(tw.get("post", None), list):
                tw["post"] = tw["post"][:cut_len]

            blk_str = ""
            print(f"[REMOVE] {subj} good@{g}{blk_str}: cut post at bad@{cutoff} (kept {cut_len} post trials)")

            subj_out.append(r2)

        out[subj] = subj_out
    return out

def classify_towers_at_good_reversals(reversal):
    before = reversal["reward_magnitudes_by_tower_before"]
    after  = reversal["reward_magnitudes_by_tower_after"]

    prev_best = max(before, key=before.get)
    next_best = max(after, key=after.get)

    towers = set(before.keys())
    third = list(towers - {prev_best, next_best})[0]

    return prev_best, next_best, third