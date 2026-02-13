from collections import Counter
from src.behavior_analysis.get_variables_across_sessions import get_vars_across_all_sessions

def get_good_reversal_info(data, pre=5, post=None, include_first_block=False, required_reward_patterns=((4, 1, 0), (3, 1, 0))):
    """
    For each subject:
      - Find GOOD reversal indices (where cumulative good_reversals increments).
      - Collect window indices:
          pre  = trials [idx-pre, ..., idx-1]
          post = trials [idx, ..., idx+post-1]  (includes the reversal trial, or goes until the next good reversal if post is None)
      - Keep only those reversals where the reward magnitudes on trial idx-1
        across towers match ANY of required_reward_patterns in ANY order (multiset match).
      - Infer towers from reward_magnitudes_by_tower keys (preferred), else choices_by_tower keys.
      - Include both reward_magnitudes_by_tower_before (idx-1) and
        reward_magnitudes_by_tower_after (idx).

    required_reward_patterns can be:
      - a single tuple like (4,1,0)
      - or an iterable of tuples like ((4,1,0), (3,1,0))
    """

    def find_increment_indices(cumulative_list):
        """Return indices i where cumulative_list[i] > cumulative_list[i-1]."""
        inc = []
        for i in range(1, len(cumulative_list)):
            if cumulative_list[i] > cumulative_list[i - 1]:
                inc.append(i)
        return inc

    # ---- normalize required_reward_patterns ----
    # Allow passing a single tuple like (4,1,0)
    if isinstance(required_reward_patterns, tuple) and all(
        isinstance(x, (int, float)) for x in required_reward_patterns
    ):
        required_reward_patterns = (required_reward_patterns,)

    required_counters = [Counter(pat) for pat in required_reward_patterns]

    merged_subject_data_across_all_sessions, _ = get_vars_across_all_sessions(data)

    out = {}
    for subj, d in merged_subject_data_across_all_sessions.items():
        good = d.get("good_reversals", [])
        blocks = d.get("blocks", [])
        reward_by_tower = d.get("reward_magnitudes_by_tower", {}) or {}
        choices_by_tower = d.get("choices_by_tower", {}) or {}
        choices_by_rank = d.get("choices_by_rank", {}) or {}

        n_trials = len(good)
        if n_trials == 0:
            out[subj] = []
            continue

        if len(blocks) != n_trials:
            print(f"[WARN] {subj}: blocks length {len(blocks)} != good_reversals length {n_trials}")

        if reward_by_tower:
            towers = list(reward_by_tower.keys())
        else:
            towers = list(choices_by_tower.keys())

        if not towers:
            print(f"[WARN] {subj}: no towers found in reward_magnitudes_by_tower or choices_by_tower")
            out[subj] = []
            continue

        good_rev_indices = find_increment_indices(good)
        if include_first_block and n_trials > 0 and good[0] == 0:
            good_rev_indices = [0] + good_rev_indices

        subj_results = []

        for k, idx in enumerate(good_rev_indices):
            before_trial = idx - 1 if idx > 0 else 0

            missing = [t for t in towers if t not in reward_by_tower]
            if missing:
                print(f"[SKIP] {subj} reversal@{idx}: missing towers in reward_magnitudes_by_tower: {missing}")
                continue

            try:
                before_vals = [reward_by_tower[t][before_trial] for t in towers]
            except IndexError:
                print(f"[SKIP] {subj} reversal@{idx}: reward_magnitudes_by_tower too short for idx-1")
                continue

            before_counter = Counter(before_vals)
            if not any(before_counter == rc for rc in required_counters):
                block_id = blocks[idx] if idx < len(blocks) else None
                print(
                    f"[SKIP] {subj} reversal@{idx} (block {block_id}): "
                    f"reward magnitudes before reversal were {before_vals} across towers {towers} "
                    f"(expected a permutation of one of {list(required_reward_patterns)})"
                )
                continue

            pre_start = max(0, idx - pre)
            pre_end = idx

            post_start = idx
            if post is None:
                post_end = good_rev_indices[k + 1] if (k + 1) < len(good_rev_indices) else n_trials
            else:
                post_end = min(n_trials, idx + post)

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
                "good_reversal_number": good[idx],
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