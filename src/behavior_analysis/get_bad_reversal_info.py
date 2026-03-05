"""Extract per-subject windowed trial data centred on bad reversal events.

The function here mirrors :mod:`src.behavior_analysis.get_good_reversal_info`
but targets bad-reversal boundaries, and applies a stricter reward-pattern
filter (multiset match against ``required_reward_pattern``) instead of the
zero-count gate used for good reversals.
"""
from collections import Counter
from src.behavior_analysis.get_variables_across_sessions import get_vars_across_all_sessions
from src.behavior_analysis.get_total_reversals import find_increment_indices


def get_bad_reversal_info(data, pre=5, post=None, include_first_block=False, required_reward_pattern=(4, 1, 0)):
    """Extract windowed trial data around each bad reversal for all subjects.

    Identifies bad-reversal boundaries from the merged session data for each
    subject and returns sliced pre/post windows.  Only reversals where the
    reward magnitudes immediately before the reversal form a permutation of
    *required_reward_pattern* are kept.

    Args:
        data: Nested dict ``{subject: {session_key: session_dict}}``.
        pre: Number of trials before the reversal index to include in the
            pre-window (default: 5).
        post: Number of trials after the reversal index to include in the
            post-window.  If ``None``, extends to the next bad reversal
            (default: ``None``).
        include_first_block: If ``True``, prepend index ``0`` to the boundary
            list (default: ``False``).
        required_reward_pattern: Multiset of reward values that must be
            present (in any order) across towers on the trial immediately
            before the reversal.  Reversals that do not match are skipped
            (default: ``(4, 1, 0)``).

    Returns:
        Dict ``{subject: list[reversal_dict]}`` where each *reversal_dict*
        contains:

        - ``"reversal_idx"`` (int): Trial index of the reversal.
        - ``"bad_reversal_number"`` (int): Cumulative bad-reversal count at
          this reversal.
        - ``"block_id"`` (int | None): Block identifier at the reversal trial.
        - ``"towers"`` (list[str]): Tower keys present in the reward data.
        - ``"trial_window_idx"`` (dict): ``{"pre": [...], "post": [...]}``
          lists of absolute trial indices.
        - ``"reward_magnitudes_by_tower_before"`` (dict): Tower → magnitude
          on the trial immediately before the reversal.
        - ``"reward_magnitudes_by_tower_after"`` (dict): Tower → magnitude on
          the reversal trial itself (``None`` if out of range).
        - ``"choices_by_tower"`` (dict): Tower → ``{"pre": [...], "post":
          [...]}`` one-hot choice slices.
        - ``"choices_by_rank"`` (dict): Rank → ``{"pre": [...], "post":
          [...]}`` one-hot choice slices.
    """

    merged_subject_data_across_all_sessions, _ = get_vars_across_all_sessions(data)
    
    out = {}
    for subj, d in merged_subject_data_across_all_sessions.items():
        bad = d.get("bad_reversals", [])
        blocks = d.get("blocks", [])
        reward_by_tower = d.get("reward_magnitudes_by_tower", {}) or {}
        choices_by_tower = d.get("choices_by_tower", {}) or {}
        choices_by_rank = d.get("choices_by_rank", {}) or {}

        n_trials = len(bad)
        if n_trials == 0:
            out[subj] = []
            continue

        if len(blocks) != n_trials:
            print(f"[WARN] {subj}: blocks length {len(blocks)} != bad_reversals length {n_trials}")

        if reward_by_tower:
            towers = list(reward_by_tower.keys())
        else:
            towers = list(choices_by_tower.keys())

        if not towers:
            print(f"[WARN] {subj}: no towers found in reward_magnitudes_by_tower or choices_by_tower")
            out[subj] = []
            continue

        bad_rev_indices = find_increment_indices(bad)
        if include_first_block and n_trials > 0 and bad[0] == 0:
            bad_rev_indices = [0] + bad_rev_indices

        subj_results = []

        for k, idx in enumerate(bad_rev_indices):
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

            if Counter(before_vals) != Counter(required_reward_pattern):
                block_id = blocks[idx] if idx < len(blocks) else None
                print(
                    f"[SKIP] {subj} reversal@{idx} (block {block_id}): "
                    f"reward magnitudes before reversal were {before_vals} across towers {towers} "
                    f"(expected a permutation of {list(required_reward_pattern)})"
                )
                continue

            pre_start = max(0, idx - pre)
            pre_end = idx

            post_start = idx
            if post is None:
                post_end = bad_rev_indices[k + 1] if (k + 1) < len(bad_rev_indices) else n_trials
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
                "bad_reversal_number": bad[idx],
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
