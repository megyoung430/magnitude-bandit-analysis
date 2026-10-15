"""Extract per-subject windowed trial data centred on bad reversal events.

The function here mirrors :mod:`src.behavior_analysis.get_good_reversal_info`
but targets bad-reversal boundaries, and applies a stricter reward-pattern
filter (multiset match against ``required_reward_pattern``) instead of the
zero-count gate used for good reversals.
"""

from collections import Counter
from dataclasses import dataclass

from src.behavior_analysis.get_total_reversals import find_increment_indices
from src.behavior_analysis.get_variables_across_sessions import get_vars_across_all_sessions


@dataclass(frozen=True)
class _SubjectSeries:
    """Per-trial series of one subject's merged sessions (missing fields become empty)."""

    bad: list
    blocks: list
    reward_by_tower: dict
    choices_by_tower: dict
    choices_by_rank: dict

    @classmethod
    def from_merged(cls, d):
        # Only the tower and choice fields are coerced with ``or``; ``blocks`` is always a
        # list here because get_vars_across_all_sessions already fails on a None one.
        return cls(
            bad=d.get("bad_reversals", []),
            blocks=d.get("blocks", []),
            reward_by_tower=d.get("reward_magnitudes_by_tower", {}) or {},
            choices_by_tower=d.get("choices_by_tower", {}) or {},
            choices_by_rank=d.get("choices_by_rank", {}) or {},
        )


def _list_towers(series):
    """Tower keys, taken from the reward data when present, else from the choice data."""
    by_tower = series.reward_by_tower if series.reward_by_tower else series.choices_by_tower
    return list(by_tower.keys())


def _find_bad_boundaries(series, *, include_first_block):
    """Trial indices where the bad-reversal counter increments, optionally led by trial 0."""
    boundaries = find_increment_indices(series.bad)
    # The first block counts as a boundary only when the counter starts at 0.
    if include_first_block and series.bad[0] == 0:
        boundaries = [0] + boundaries
    return boundaries


def _block_id_at(blocks, idx):
    """Block id at trial *idx*, or ``None`` when *blocks* does not reach it."""
    return blocks[idx] if idx < len(blocks) else None


def _before_trial(idx):
    """Trial whose reward magnitudes describe the state just before boundary *idx*.

    Boundary 0 has no preceding trial, so it returns 0: with ``include_first_block``
    the first record's "before" and "after" values both come from trial 0.
    """
    return idx - 1 if idx > 0 else 0


def _window_indices(boundaries, k, n_trials, *, pre, post):
    """Absolute ``(pre_idx, post_idx)`` trial indices around boundary number *k*.

    The post-window never extends past the next boundary (or the last trial).
    """
    idx = boundaries[k]
    next_boundary = boundaries[k + 1] if k + 1 < len(boundaries) else n_trials
    post_end = next_boundary if post is None else min(n_trials, idx + post, next_boundary)
    return list(range(max(0, idx - pre), idx)), list(range(idx, post_end))


def _slice_choices(choices, *, pre_idx, post_idx):
    """Cut each one-hot choice array into pre/post windows, dropping indices past its end."""
    return {
        key: {
            "pre": [arr[i] for i in pre_idx if i < len(arr)],
            "post": [arr[i] for i in post_idx if i < len(arr)],
        }
        for key, arr in choices.items()
    }


def _rewards_at(reward_by_tower, towers, trial):
    """Reward magnitude of every tower at *trial*; ``None`` for all if any array is too short."""
    if any(trial >= len(reward_by_tower[t]) for t in towers):
        return {t: None for t in towers}
    return {t: reward_by_tower[t][trial] for t in towers}


def _skip_message(subj, idx, series, towers, *, required_reward_pattern):
    """Why boundary *idx* must be skipped, or ``None`` if it is usable."""
    missing = [t for t in towers if t not in series.reward_by_tower]
    if missing:
        return (
            f"[SKIP] {subj} reversal@{idx}: missing towers in reward_magnitudes_by_tower: {missing}"
        )
    before_trial = _before_trial(idx)
    if any(before_trial >= len(series.reward_by_tower[t]) for t in towers):
        return f"[SKIP] {subj} reversal@{idx}: reward_magnitudes_by_tower too short for idx-1"
    before_vals = [series.reward_by_tower[t][before_trial] for t in towers]
    if Counter(before_vals) != Counter(required_reward_pattern):
        block_id = _block_id_at(series.blocks, idx)
        return (
            f"[SKIP] {subj} reversal@{idx} (block {block_id}): "
            f"reward magnitudes before reversal were {before_vals} across towers {towers} "
            f"(expected a permutation of {list(required_reward_pattern)})"
        )
    return None


def _build_reversal(series, towers, idx, *, pre_idx, post_idx):
    """Assemble the reversal dict for the boundary at trial *idx* (see the public docstring).

    Only called for boundaries that passed :func:`_skip_message`, so every tower's
    reward array reaches the trial before *idx*.
    """
    before = _before_trial(idx)
    return {
        "reversal_idx": idx,
        "bad_reversal_number": series.bad[idx],
        "block_id": _block_id_at(series.blocks, idx),
        "towers": list(towers),  # own copy: records must not alias each other
        "trial_window_idx": {"pre": pre_idx, "post": post_idx},
        "reward_magnitudes_by_tower_before": {t: series.reward_by_tower[t][before] for t in towers},
        "reward_magnitudes_by_tower_after": _rewards_at(series.reward_by_tower, towers, idx),
        "choices_by_tower": _slice_choices(
            series.choices_by_tower, pre_idx=pre_idx, post_idx=post_idx
        ),
        "choices_by_rank": _slice_choices(
            series.choices_by_rank, pre_idx=pre_idx, post_idx=post_idx
        ),
    }


def _subject_reversals(subj, series, *, pre, post, include_first_block, required_reward_pattern):
    """Reversal dicts for one subject; ``[]`` when there is nothing to window."""
    n_trials = len(series.bad)
    if n_trials == 0:
        return []
    if len(series.blocks) != n_trials:
        print(
            f"[WARN] {subj}: blocks length {len(series.blocks)} != bad_reversals length {n_trials}"
        )
    towers = _list_towers(series)
    if not towers:
        print(f"[WARN] {subj}: no towers found in reward_magnitudes_by_tower or choices_by_tower")
        return []

    boundaries = _find_bad_boundaries(series, include_first_block=include_first_block)
    reversals = []
    for k, idx in enumerate(boundaries):
        message = _skip_message(
            subj, idx, series, towers, required_reward_pattern=required_reward_pattern
        )
        if message is not None:
            print(message)
            continue
        # Windows are computed from the full boundary list, so a kept reversal's
        # post-window still stops at a neighbouring boundary that was skipped.
        pre_idx, post_idx = _window_indices(boundaries, k, n_trials, pre=pre, post=post)
        reversals.append(_build_reversal(series, towers, idx, pre_idx=pre_idx, post_idx=post_idx))
    return reversals


def get_bad_reversal_info(
    data, pre=5, post=None, include_first_block=False, required_reward_pattern=(4, 1, 0)
):
    """Extract windowed trial data around each bad reversal for all subjects.

    Identifies bad-reversal boundaries from the merged session data for each
    subject and returns sliced pre/post windows.  Only reversals where the
    reward magnitudes immediately before the reversal form a permutation of
    *required_reward_pattern* are kept.

    Args:
        data: Nested dict ``{subject: {session_key: session_dict}}``.
        pre: Number of trials before the reversal index to include in the
            pre-window (default: 5).
        post: Maximum number of trials after the reversal index to include in
            the post-window.  The window never extends past the next bad
            reversal; if ``None``, it extends exactly to it (default: ``None``).
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
    return {
        subj: _subject_reversals(
            subj,
            _SubjectSeries.from_merged(d),
            pre=pre,
            post=post,
            include_first_block=include_first_block,
            required_reward_pattern=required_reward_pattern,
        )
        for subj, d in merged_subject_data_across_all_sessions.items()
    }
