"""Extract per-subject windowed trial data centred on good reversal events.

Functions here identify good-reversal boundaries from merged session data, apply
quality filters (zero-reward gate), and slice choice/reward arrays into pre/post
windows for downstream analyses.
"""

import warnings
from dataclasses import dataclass

from src.behavior_analysis.get_total_reversals import find_increment_indices
from src.behavior_analysis.get_variables_across_sessions import (
    get_vars_across_all_sessions,
    session_order,
)


@dataclass(frozen=True)
class _SubjectSeries:
    """Per-trial series of one subject's merged sessions (missing fields become empty)."""

    good: list | None  # None when absent; the fallback in _good_reversal_number depends on it
    num_blocks: list
    blocks: list
    reward_by_tower: dict
    choices_by_tower: dict
    choices_by_rank: dict

    @classmethod
    def from_merged(cls, d):
        return cls(
            good=d.get("good_reversals"),
            num_blocks=d.get("num_blocks") or [],
            blocks=d.get("blocks") or [],
            reward_by_tower=d.get("reward_magnitudes_by_tower", {}) or {},
            choices_by_tower=d.get("choices_by_tower", {}) or {},
            choices_by_rank=d.get("choices_by_rank", {}) or {},
        )


def _find_change_indices(seq):
    """Return indices where consecutive elements differ in *seq*."""
    return [i for i in range(1, len(seq)) if seq[i] != seq[i - 1]]


def _count_trials(series):
    """Trial count from the first series that exists: num_blocks, blocks, rewards, choices."""
    if len(series.num_blocks) > 0:
        return len(series.num_blocks)
    if len(series.blocks) > 0:
        return len(series.blocks)
    for by_tower in (series.reward_by_tower, series.choices_by_tower):
        if by_tower:
            return max((len(v) for v in by_tower.values()), default=0)
    return 0


def _list_towers(series):
    """Tower keys, taken from the reward data when present, else from the choice data."""
    by_tower = series.reward_by_tower if series.reward_by_tower else series.choices_by_tower
    return list(by_tower.keys())


@dataclass(frozen=True)
class _SessionSpan:
    """The merged-trial range ``[start, stop)`` covered by one session."""

    key: str
    start: int
    stop: int
    bad_only: bool  # logs bad reversals but no good ones


def _session_spans(sessions):
    """Spans of a subject's sessions in merged order (sessions without trials are absent)."""
    spans, start = [], 0
    for key in sorted(sessions, key=session_order):
        sess = sessions[key]
        n = len(sess.get("trial") or [])
        if n:
            bad_only = bool(sess.get("has_bad", False)) and not sess.get("has_good", False)
            spans.append(_SessionSpan(key, start, start + n, bad_only))
            start += n
    return spans


def _locate_trial(spans, idx):
    """``(session key, trial index within that session)`` of merged trial *idx*.

    The session is ``None`` when no span contains *idx*.
    """
    for span in spans:
        if span.start <= idx < span.stop:
            return span.key, idx - span.start
    return None, idx


def _in_spans(idx, masked):
    """True if *idx* lies inside any ``(start, stop)`` range of *masked*."""
    return any(start <= idx < stop for start, stop in masked)


def _find_boundaries(series, any_good, masked):
    """Return ``(boundary_indices, boundary_source)`` using the priority in the public docstring.

    *masked* lists ``(start, stop)`` trial ranges whose block changes are not good
    reversals; it only applies to the block-change fallback, which is reached when no
    session of the subject has a good counter.
    """
    good = series.good
    if any_good and good is not None and len(good) >= 2:
        return find_increment_indices(good), "good_reversals"
    if len(series.num_blocks) >= 2:
        boundaries, source = find_increment_indices(series.num_blocks), "num_blocks"
    elif len(series.blocks) >= 2:
        boundaries, source = _find_change_indices(series.blocks), "blocks"
    else:
        return [], None
    return [b for b in boundaries if not _in_spans(b, masked)], source


def _block_id_at(blocks, idx):
    """Block id at trial *idx*, or ``None`` when *blocks* does not reach it."""
    return blocks[idx] if idx < len(blocks) else None


def _window_indices(boundaries, k, n_trials, *, pre, post):
    """Absolute ``(pre_idx, post_idx)`` trial indices around boundary number *k*.

    The post-window never extends past the next boundary (or the last trial).
    """
    idx = boundaries[k]
    next_boundary = boundaries[k + 1] if k + 1 < len(boundaries) else n_trials
    post_end = next_boundary if post is None else min(n_trials, idx + post, next_boundary)
    return list(range(max(0, idx - pre), idx)), list(range(idx, post_end))


def _slice_choices(choices, pre_idx, post_idx):
    """Cut each one-hot choice array into pre/post windows, dropping indices past its end."""
    return {
        key: {
            "pre": [arr[i] for i in pre_idx if i < len(arr)],
            "post": [arr[i] for i in post_idx if i < len(arr)],
        }
        for key, arr in choices.items()
    }


def _before_trial(idx):
    """Trial whose reward magnitudes describe the state just before boundary *idx*.

    Boundary 0 has no preceding trial, so it returns 0: with ``include_first_block``
    the first record's "before" and "after" values both come from trial 0.
    """
    return idx - 1 if idx > 0 else 0


def _rewards_at(reward_by_tower, towers, trial):
    """Reward magnitude of every tower at *trial*; ``None`` for all if any array is too short."""
    if any(trial >= len(reward_by_tower[t]) for t in towers):
        return {t: None for t in towers}
    return {t: reward_by_tower[t][trial] for t in towers}


def _skip_message(subj, idx, boundary_source, series, towers, *, max_zero_count):
    """Why boundary *idx* must be skipped, or ``None`` if it is usable."""
    missing = [t for t in towers if t not in series.reward_by_tower]
    if missing:
        return (
            f"[SKIP] {subj} boundary@{idx}: missing towers in reward_magnitudes_by_tower: {missing}"
        )
    before_trial = _before_trial(idx)
    if any(before_trial >= len(series.reward_by_tower[t]) for t in towers):
        return f"[SKIP] {subj} boundary@{idx}: reward_magnitudes_by_tower too short for idx-1"
    before_vals = [series.reward_by_tower[t][before_trial] for t in towers]
    if sum(v == 0 for v in before_vals) > max_zero_count:
        block_id = _block_id_at(series.blocks, idx)
        return (
            f"[SKIP] {subj} boundary@{idx} ({boundary_source}, block {block_id}): "
            f"more than {max_zero_count} zero rewards in {before_vals}"
        )
    return None


def _good_reversal_number(series, idx, block_id):
    """Cumulative good-reversal count at *idx*, falling back to ``block_id - 1``."""
    if series.good is not None and idx < len(series.good):
        return series.good[idx]
    # Known bug: block_id is None when blocks is shorter than idx, so this raises
    # TypeError instead of yielding None.
    return block_id - 1


def _best_unchanged(before, after):
    """True if the best-rewarding tower is the same before and after.

    ``False`` when a magnitude is missing (``None``), since the best tower is then unknown.
    """
    if any(v is None for v in (*before.values(), *after.values())):
        return False
    return max(before, key=before.get) == max(after, key=after.get)


def _build_reversal(series, towers, boundary_source, idx, pre_idx, post_idx, *, first_block):
    """Assemble the reversal dict for the boundary at trial *idx* (see the public docstring).

    Only called for boundaries that passed :func:`_skip_message`, so every tower's
    reward array reaches the trial before *idx*.
    """
    block_id = _block_id_at(series.blocks, idx)
    before_trial = _before_trial(idx)
    before = {t: series.reward_by_tower[t][before_trial] for t in towers}
    after = _rewards_at(series.reward_by_tower, towers, idx)
    return {
        "reversal_idx": idx,
        "boundary_source": boundary_source,
        "good_reversal_number": _good_reversal_number(series, idx, block_id),
        "block_id": block_id,
        "towers": towers,
        "first_block": first_block,
        # The first block is not a reversal: its "before" and "after" both come from trial 0,
        # so an unchanged best tower there is expected and not a flag.
        "best_unchanged": not first_block and _best_unchanged(before, after),
        "trial_window_idx": {"pre": pre_idx, "post": post_idx},
        "reward_magnitudes_by_tower_before": before,
        "reward_magnitudes_by_tower_after": after,
        "choices_by_tower": _slice_choices(series.choices_by_tower, pre_idx, post_idx),
        "choices_by_rank": _slice_choices(series.choices_by_rank, pre_idx, post_idx),
    }


def _warn_best_unchanged(subj, spans, idx):
    session, trial = _locate_trial(spans, idx)
    where = f"session {session} trial {trial}" if session is not None else f"trial {trial}"
    warnings.warn(
        f"{subj} {where}: best tower unchanged across a good reversal "
        f"(merged trial {idx}); flagged best_unchanged and left out of the "
        "tower-classified analyses. A good reversal should change the best tower.",
        UserWarning,
        stacklevel=3,
    )


def _warn_bad_only_sessions(subj, spans):
    for span in spans:
        if span.bad_only:
            warnings.warn(
                f"{subj} {span.key}: logs bad but no good reversals, so it has no good reversals.",
                UserWarning,
                stacklevel=3,
            )


def _subject_reversals(subj, series, sessions, *, pre, post, include_first_block, max_zero_count):
    """Reversal dicts for one subject; ``[]`` when there is nothing to window."""
    n_trials = _count_trials(series)
    if n_trials == 0:
        return []
    towers = _list_towers(series)
    if not towers:
        print(f"[WARN] {subj}: no towers found in reward_magnitudes_by_tower or choices_by_tower")
        return []

    any_good = any(sess.get("has_good", False) for sess in sessions.values())
    spans = _session_spans(sessions)
    _warn_bad_only_sessions(subj, spans)
    # Block changes of a bad-only session mix in bad reversals, so they are not good
    # reversals. The mask matters only for the block-change fallback (no good counter).
    masked = [] if any_good else [(s.start, s.stop) for s in spans if s.bad_only]
    boundaries, boundary_source = _find_boundaries(series, any_good, masked)
    with_first_block = include_first_block and not _in_spans(0, masked)
    if with_first_block:
        boundaries = [0] + boundaries

    reversals = []
    for k, idx in enumerate(boundaries):
        message = _skip_message(
            subj, idx, boundary_source, series, towers, max_zero_count=max_zero_count
        )
        if message is not None:
            print(message)
            continue
        # Windows are computed from the full boundary list, so a kept reversal's
        # post-window still stops at a neighbouring boundary that was skipped.
        pre_idx, post_idx = _window_indices(boundaries, k, n_trials, pre=pre, post=post)
        reversal = _build_reversal(
            series,
            towers,
            boundary_source,
            idx,
            pre_idx,
            post_idx,
            first_block=with_first_block and k == 0,
        )
        if reversal["best_unchanged"]:
            _warn_best_unchanged(subj, spans, idx)
        reversals.append(reversal)
    return reversals


def get_good_reversal_info(data, pre=5, post=None, include_first_block=False, max_zero_count=1):
    """Extract windowed trial data around each good reversal for all subjects.

    Identifies good-reversal boundaries from the merged session data for each
    subject, filters out reversals where too many towers had zero reward before
    the reversal, and returns sliced pre/post windows for choices and rewards.

    Boundary detection priority (first match wins):
      1. ``good_reversals`` increment — used when the subject has at least one
         session with ``has_good == True`` and a valid ``good_reversals`` array.
         Sessions of that subject without the counter contribute no reversals.
      2. ``num_blocks`` increment.
      3. ``blocks`` value-change.
    Steps 2 and 3 treat every block change as good, which only holds for
    sessions from before the counters existed. A bad-only session (``has_bad``
    set, no ``has_good``) has no recorded good reversals: its block changes are
    dropped from steps 2 and 3, and a ``UserWarning`` names the subject and
    session. Legacy sessions of the same subject keep their block changes.

    Args:
        data: Nested dict ``{subject: {session_key: session_dict}}``.
        pre: Number of trials before the reversal index to include in the
            pre-window (default: 5).
        post: Number of trials after the reversal index to include in the
            post-window.  If ``None``, extends to the next reversal boundary
            (default: ``None``).
        include_first_block: If ``True``, prepend index ``0`` to the boundary
            list so the very first block is also represented (default:
            ``False``). Not done when trial 0 belongs to a bad-only session
            and the subject has no good counter: that session has no good
            reversals, and its first block ends at a bad reversal.
        max_zero_count: Maximum number of towers allowed to have a zero reward
            magnitude on the trial immediately before the reversal.  Reversals
            exceeding this count are skipped (default: 1).

    Returns:
        Dict ``{subject: list[reversal_dict]}`` where each *reversal_dict*
        contains:

        - ``"reversal_idx"`` (int): Trial index of the reversal.
        - ``"boundary_source"`` (str | None): Which field was used to detect
          the boundary (``"good_reversals"``, ``"num_blocks"``, or
          ``"blocks"``).
        - ``"good_reversal_number"`` (int | None): Cumulative good-reversal
          count at this reversal, or ``block_id - 1`` as a fallback.
        - ``"block_id"`` (int | None): Block identifier at the reversal trial.
        - ``"towers"`` (list[str]): Tower keys from the reward data, or from the
          choice data when there is no reward data (boundaries are then skipped
          as missing towers).
        - ``"first_block"`` (bool): ``True`` for the record prepended by
          *include_first_block*. The first block is not a reversal.
        - ``"best_unchanged"`` (bool): ``True`` when the best-rewarding tower is
          the same before and after the reversal. A good reversal should change
          it, so such a record is a suspected data or detection bug: it is kept,
          a ``UserWarning`` names the subject, session and trial, and the
          tower-classified analyses leave it out. Always ``False`` for the
          first-block record.
        - ``"trial_window_idx"`` (dict): ``{"pre": [...], "post": [...]}``
          lists of absolute trial indices.
        - ``"reward_magnitudes_by_tower_before"`` (dict): Tower → magnitude on
          the trial immediately before the reversal.
        - ``"reward_magnitudes_by_tower_after"`` (dict): Tower → magnitude on
          the reversal trial itself (value is ``None`` if out of range).
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
            data.get(subj, {}),
            pre=pre,
            post=post,
            include_first_block=include_first_block,
            max_zero_count=max_zero_count,
        )
        for subj, d in merged_subject_data_across_all_sessions.items()
    }


def classify_towers_at_good_reversals(reversal):
    """Classify towers as prev-best, next-best, and third around a good reversal.

    Uses the reward magnitude dicts stored inside a reversal window dict to
    determine which tower was the best-rewarding arm immediately before the
    reversal, which became the best arm after, and which is neither.

    Args:
        reversal: A reversal dict as returned by :func:`get_good_reversal_info`,
            containing ``"reward_magnitudes_by_tower_before"`` and
            ``"reward_magnitudes_by_tower_after"`` keys.

    Returns:
        A three-tuple ``(prev_best, next_best, third)``. When the best tower
        changes, ``third`` is the remaining tower. When it is unchanged
        (``prev_best == next_best``) there is no meaningful third tower and
        ``third`` is ``None``. Ties for best resolve to the tower listed first.

    Raises:
        ValueError: If the best tower changes and no other tower is left.
    """
    before = reversal["reward_magnitudes_by_tower_before"]
    after = reversal["reward_magnitudes_by_tower_after"]

    prev_best = max(before, key=before.get)
    next_best = max(after, key=after.get)
    if prev_best == next_best:
        return prev_best, next_best, None

    third = next((t for t in before if t not in (prev_best, next_best)), None)
    if third is None:
        raise ValueError(
            f"no remaining tower besides {prev_best!r} and {next_best!r} in {list(before)}"
        )

    return prev_best, next_best, third


def exclude_best_unchanged(reversals, *, subject, caller):
    """Drop reversals that have no third tower; warn how many were best-unchanged.

    Used by the analyses that classify towers as prev-best / next-best / third.
    A record has no third tower when its best tower is unchanged, either because
    the reversal is suspect (``best_unchanged``) or because it is the first-block
    record, which is not a reversal. Only the former are counted and reported.

    Args:
        reversals: A subject's list of reversal dicts.
        subject: Subject name, for the warning.
        caller: Name of the calling analysis, for the warning.

    Returns:
        The reversals that can be classified, in their original order.
    """
    has_third = [classify_towers_at_good_reversals(r)[2] is not None for r in reversals]
    kept = [r for r, ok in zip(reversals, has_third, strict=True) if ok]
    n_excluded = sum(
        not ok and not r.get("first_block", False)
        for r, ok in zip(reversals, has_third, strict=True)
    )
    if n_excluded:
        warnings.warn(
            f"{caller}: excluded {n_excluded} reversal(s) for subject {subject!r} whose "
            "best tower was unchanged (no third tower to classify).",
            UserWarning,
            stacklevel=3,
        )
    return kept
