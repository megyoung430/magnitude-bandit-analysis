from src.behavior_analysis.get_variables_across_sessions import get_vars_across_all_sessions
from src.behavior_analysis.get_total_reversals import find_increment_indices
def get_good_reversal_info(data, pre=5, post=None, include_first_block=False, max_zero_count=1):
    def find_change_indices(seq):
        chg = []
        for i in range(1, len(seq)):
            if seq[i] != seq[i - 1]:
                chg.append(i)
        return chg

    merged_subject_data_across_all_sessions, _ = get_vars_across_all_sessions(data)

    out = {}
    for subj, d in merged_subject_data_across_all_sessions.items():
        good = d.get("good_reversals")          # may be None / missing
        num_blocks = d.get("num_blocks") or [] # cumulative block counter per trial (preferred)
        blocks = d.get("blocks") or []         # block id per trial (backup)

        reward_by_tower = d.get("reward_magnitudes_by_tower", {}) or {}
        choices_by_tower = d.get("choices_by_tower", {}) or {}
        choices_by_rank = d.get("choices_by_rank", {}) or {}

        # decide n_trials from something that exists (prefer blocks/num_blocks, else reward arrays)
        if len(num_blocks) > 0:
            n_trials = len(num_blocks)
        elif len(blocks) > 0:
            n_trials = len(blocks)
        elif reward_by_tower:
            n_trials = max((len(v) for v in reward_by_tower.values()), default=0)
        elif choices_by_tower:
            n_trials = max((len(v) for v in choices_by_tower.values()), default=0)
        else:
            n_trials = 0

        if n_trials == 0:
            out[subj] = []
            continue

        # towers
        towers = list(reward_by_tower.keys()) if reward_by_tower else list(choices_by_tower.keys())
        if not towers:
            print(f"[WARN] {subj}: no towers found in reward_magnitudes_by_tower or choices_by_tower")
            out[subj] = []
            continue

        # --- pick boundary indices ---
        subj_sessions = data.get(subj, {})
        any_good = any(sess.get("has_good", False) for sess in subj_sessions.values())
        if any_good and good is not None and len(good) >= 2:
            boundary_indices = find_increment_indices(good)
            boundary_source = "good_reversals"
        elif len(num_blocks) >= 2:
            boundary_indices = find_increment_indices(num_blocks)
            boundary_source = "num_blocks"
        elif len(blocks) >= 2:
            boundary_indices = find_change_indices(blocks)
            boundary_source = "blocks"
        else:
            boundary_indices = []
            boundary_source = None

        if include_first_block and n_trials > 0:
            boundary_indices = [0] + boundary_indices

        if not boundary_indices:
            out[subj] = []
            continue

        subj_results = []

        for k, idx in enumerate(boundary_indices):
            before_trial = idx - 1 if idx > 0 else 0

            missing = [t for t in towers if t not in reward_by_tower]
            if missing:
                print(f"[SKIP] {subj} boundary@{idx}: missing towers in reward_magnitudes_by_tower: {missing}")
                continue

            try:
                before_vals = [reward_by_tower[t][before_trial] for t in towers]
            except IndexError:
                print(f"[SKIP] {subj} boundary@{idx}: reward_magnitudes_by_tower too short for idx-1")
                continue
            
            num_zeros = sum(v == 0 for v in before_vals)
            if num_zeros > max_zero_count:
                block_id = blocks[idx] if idx < len(blocks) else None
                print(
                    f"[SKIP] {subj} boundary@{idx} ({boundary_source}, block {block_id}): "
                    f"more than {max_zero_count} zero rewards in {before_vals}"
                )
                continue

            pre_start = max(0, idx - pre)
            pre_end = idx
            next_boundary = boundary_indices[k + 1] if (k + 1) < len(boundary_indices) else n_trials
            post_start = idx
            if post is None:
                post_end = next_boundary
            else:
                post_end = min(n_trials, idx + post)
                post_end = min(post_end, next_boundary)

            pre_idx = list(range(pre_start, pre_end))
            post_idx = list(range(post_start, post_end))

            reward_before = {t: reward_by_tower[t][before_trial] for t in towers}
            try:
                reward_after = {t: reward_by_tower[t][idx] for t in towers}
            except IndexError:
                reward_after = {t: None for t in towers}

            tower_slices = {}
            for t, arr in choices_by_tower.items():
                n = len(arr)
                tower_slices[t] = {
                    "pre":  [arr[i] for i in pre_idx if i < n],
                    "post": [arr[i] for i in post_idx if i < n],
                }

            rank_slices = {}
            for rk, arr in choices_by_rank.items():
                n = len(arr)
                rank_slices[rk] = {
                    "pre":  [arr[i] for i in pre_idx if i < n],
                    "post": [arr[i] for i in post_idx if i < n],
                }

            block_id = blocks[idx] if idx < len(blocks) else None

            subj_results.append({
                "reversal_idx": idx,
                "boundary_source": boundary_source,
                "good_reversal_number": (good[idx] if good is not None and idx < len(good) else block_id - 1),
                "block_id": block_id,
                "towers": towers,
                "trial_window_idx": {"pre": pre_idx, "post": post_idx},
                "reward_magnitudes_by_tower_before": reward_before,
                "reward_magnitudes_by_tower_after": reward_after,
                "choices_by_tower": tower_slices,
                "choices_by_rank": rank_slices,
            })

        out[subj] = subj_results

    return out

def classify_towers_at_good_reversals(reversal):
    """
    Classify towers before and after a reversal based on reward magnitudes.

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
