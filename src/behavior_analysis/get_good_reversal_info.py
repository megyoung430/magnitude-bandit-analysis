"""Extract per-subject windowed trial data centred on good reversal events.

Functions here identify good-reversal boundaries from merged session data, apply
quality filters (zero-reward gate), and slice choice/reward arrays into pre/post
windows for downstream analyses.

Importing this module sets the ``"always"`` warnings filter for :class:`ReversalWarning`.
"""

import warnings
from dataclasses import dataclass

from src.behavior_analysis.best_change import best_change_problem
from src.behavior_analysis.get_total_reversals import find_increment_indices
from src.behavior_analysis.get_variables_across_sessions import get_vars_across_all_sessions
from src.behavior_analysis.reversal_format import check_formats_agree
from src.behavior_import.session_order import ordered_session_keys


class ReversalWarning(UserWarning):
    """A good-reversal record was dropped or a session has none to offer.

    Registered as ``"always"`` so that rerunning an analysis in the same session shows the
    warning again (Python's default shows a warning once per source location).
    """


warnings.simplefilter("always", ReversalWarning)


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
    run_lengths: tuple  # trials in each recording run of the session (empty if unknown)


def _run_lengths(sess):
    """Trial count of each recording run of *sess*, or ``()`` when not recorded."""
    runs = sess.get("trial_variables")
    if not isinstance(runs, list) or not all(isinstance(r, dict) for r in runs):
        return ()
    return tuple(len(r.get("trial") or []) for r in runs)


def _session_spans(subject, sessions):
    """Spans of a subject's sessions in merged order (sessions without trials are absent)."""
    spans, start = [], 0
    for key in ordered_session_keys(subject, sessions):
        sess = sessions[key]
        n = len(sess.get("trial") or [])
        if n:
            bad_only = bool(sess.get("has_bad", False)) and not sess.get("has_good", False)
            spans.append(_SessionSpan(key, start, start + n, bad_only, _run_lengths(sess)))
            start += n
    return spans


@dataclass(frozen=True)
class _Where:
    """Where a reversal sits: subject, session, session-local trial, run and merged trial."""

    subject: str | None
    session: str | None
    trial: int | None
    run: int | None
    merged: int | None

    def text(self):
        if self.session is not None:
            head = f"{self.subject} {self.session} trial {self.trial}"
        else:
            head = f"{self.subject} merged trial {self.merged}"
        return head if self.run is None else f"{head} (run {self.run})"


def _run_of(span, local_trial):
    """Index of the recording run holding session-local *local_trial*, or ``None``."""
    first = 0
    for run, length in enumerate(span.run_lengths):
        if local_trial < first + length:
            return run
        first += length
    return None


def _locate_trial(subj, spans, idx):
    """:class:`_Where` of merged trial *idx*; the session is ``None`` outside every span."""
    for span in spans:
        if span.start <= idx < span.stop:
            local = idx - span.start
            return _Where(subj, span.key, local, _run_of(span, local), idx)
    return _Where(subj, None, None, None, idx)


def _in_spans(idx, masked):
    """True if *idx* lies inside any ``(start, stop)`` range of *masked*."""
    return any(start <= idx < stop for start, stop in masked)


def _find_boundaries(series, any_good):
    """Return ``(boundary_indices, boundary_source)`` using the priority in the public docstring."""
    good = series.good
    if any_good and good is not None and len(good) >= 2:
        return find_increment_indices(good), "good_reversals"
    if len(series.num_blocks) >= 2:
        return find_increment_indices(series.num_blocks), "num_blocks"
    if len(series.blocks) >= 2:
        return _find_change_indices(series.blocks), "blocks"
    return [], None


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


def _best_tower(magnitudes, label, where):
    """Tower with the highest magnitude (the first listed on a tie).

    Raises:
        ValueError: If a tower has no magnitude, naming where the reversal is.
    """
    missing = [t for t, v in magnitudes.items() if v is None]
    if missing:
        raise ValueError(
            f"{where.text()}: no reward magnitude {label} the reversal for towers {missing}"
        )
    return max(magnitudes, key=magnitudes.get)


def _build_reversal(
    series, towers, boundary_source, idx, pre_idx, post_idx, *, before, after, first_block, where
):
    """Assemble the reversal dict for the boundary at trial *idx* (see the public docstring)."""
    block_id = _block_id_at(series.blocks, idx)
    return {
        "reversal_idx": idx,
        "boundary_source": boundary_source,
        "good_reversal_number": _good_reversal_number(series, idx, block_id),
        "block_id": block_id,
        "towers": towers,
        "subject": where.subject,
        "session": where.session,
        "session_trial": where.trial,
        "first_block": first_block,
        "trial_window_idx": {"pre": pre_idx, "post": post_idx},
        "reward_magnitudes_by_tower_before": before,
        "reward_magnitudes_by_tower_after": after,
        "choices_by_tower": _slice_choices(series.choices_by_tower, pre_idx, post_idx),
        "choices_by_rank": _slice_choices(series.choices_by_rank, pre_idx, post_idx),
    }


@dataclass(frozen=True)
class _Dropped:
    """A reversal left out of the output because its best tower did not change."""

    where: _Where
    reason: str


def _subject_reversals(subj, series, sessions, *, pre, post, include_first_block, max_zero_count):
    """Reversal dicts for one subject, the dropped suspect ones, and its bad-only sessions.

    Returns ``(reversals, dropped, bad_only_sessions)``; the first is ``[]`` when there is
    nothing to window.
    """
    n_trials = _count_trials(series)
    if n_trials == 0:
        return [], [], []
    towers = _list_towers(series)
    if not towers:
        print(f"[WARN] {subj}: no towers found in reward_magnitudes_by_tower or choices_by_tower")
        return [], [], []

    any_good = any(sess.get("has_good", False) for sess in sessions.values())
    spans = _session_spans(subj, sessions)
    # Block changes of a bad-only session mix in bad reversals, so they are not good
    # reversals: no record is made for them. Only the block-change fallback (no good counter
    # anywhere) can reach them.
    masked = [] if any_good else [(s.start, s.stop) for s in spans if s.bad_only]
    boundaries, boundary_source = _find_boundaries(series, any_good)
    if include_first_block:
        boundaries = [0] + boundaries

    reversals, dropped = [], []
    for k, idx in enumerate(boundaries):
        if _in_spans(idx, masked):
            continue
        message = _skip_message(
            subj, idx, boundary_source, series, towers, max_zero_count=max_zero_count
        )
        if message is not None:
            print(message)
            continue
        where = _locate_trial(subj, spans, idx)
        first_block = include_first_block and k == 0
        before = _rewards_at(series.reward_by_tower, towers, _before_trial(idx))
        after = _rewards_at(series.reward_by_tower, towers, idx)
        # The first block is not a reversal: its "before" and "after" both come from trial 0,
        # so an unchanged best tower there is expected.
        problem = None if first_block else best_change_problem(before, after)
        if problem is not None:
            # A dropped reversal is still a real block change in the data, so (like a
            # skipped one) it keeps ending the window of the reversal before it.
            dropped.append(_Dropped(where, problem))
            continue
        # Windows come from the full boundary list, so a kept reversal's post-window
        # still stops at a neighbouring boundary that was skipped, masked or dropped.
        pre_idx, post_idx = _window_indices(boundaries, k, n_trials, pre=pre, post=post)
        reversals.append(
            _build_reversal(
                series,
                towers,
                boundary_source,
                idx,
                pre_idx,
                post_idx,
                before=before,
                after=after,
                first_block=first_block,
                where=where,
            )
        )
    return reversals, dropped, [s.key for s in spans if s.bad_only]


def _check_aligned(subj, merged):
    """Raise if a merged per-trial series does not have one entry per trial.

    The merge concatenates sessions by position, so sessions with different formats (a
    counter and a blocks-only session) or different tower sets silently shift or cut trials.
    An empty series counts as "not logged" and is not checked.
    """
    n_trials = len(merged.get("trial") or [])
    series = {
        "good_reversals": merged.get("good_reversals"),
        "blocks": merged.get("blocks"),
        **{
            f"reward_magnitudes_by_tower[{t!r}]": arr
            for t, arr in (merged.get("reward_magnitudes_by_tower") or {}).items()
        },
    }
    for name, values in series.items():
        if values and len(values) != n_trials:
            raise ValueError(
                f"{subj}: {name} has {len(values)} entries but the merged trial list has "
                f"{n_trials}; sessions merged by position must be aligned trial by trial."
            )


def _warn_dropped(dropped):
    """One warning listing every dropped reversal (a list of ``_Dropped``) and their count."""
    listing = "; ".join(
        f"{d.where.text()} [merged trial {d.where.merged}, {d.reason}]" for d in dropped
    )
    warnings.warn(
        f"dropped {len(dropped)} good reversal(s) because the best tower did not change "
        f"(a tied best counts) or a reward magnitude was missing: {listing}. A good reversal "
        "should change the best tower; these are left out of every analysis but still end "
        "the windows of their neighbours.",
        ReversalWarning,
        stacklevel=3,
    )


def get_good_reversal_info(data, pre=5, post=None, include_first_block=False, max_zero_count=1):
    """Extract windowed trial data around each good reversal for all subjects.

    Identifies good-reversal boundaries from the merged session data for each
    subject, filters out reversals where too many towers had zero reward before
    the reversal, and returns sliced pre/post windows for choices and rewards.

    Boundary detection priority (first match wins):
      1. ``good_reversals`` increment — used when the subject has at least one
         session with ``has_good == True`` and a valid ``good_reversals`` array.
      2. ``num_blocks`` increment.
      3. ``blocks`` value-change.
    Steps 2 and 3 treat every block change as good, which only holds for
    sessions from before the counters existed (blocks-only sessions). A subject
    that mixes sessions with the good counter and blocks-only sessions raises
    ``ValueError`` naming the subject and the sessions of each kind
    (:func:`src.behavior_analysis.reversal_format.check_formats_agree`): the good
    counter would lose the blocks-only sessions' block changes. A bad-only
    session (``has_bad`` set, no ``has_good``) has no recorded good reversals:
    its block changes are given no record (a ``ReversalWarning`` names the
    subject and session), but they still end the post-window of the reversal
    before them. Blocks-only sessions of the same subject keep their block
    changes.

    A good reversal should change the best-rewarding tower. A reversal where it
    does not (including a tie for best) is a suspected data or detection bug and
    is left out of the result, so every analysis downstream sees only valid
    reversals. One ``ReversalWarning`` per call lists the dropped reversals
    (subject, session, session-local trial, recording run) and their count.
    **A dropped reversal still ends the window of its neighbour and still counts
    as a removal** (:mod:`src.behavior_analysis.cumulative_reversal_removal`): it
    is a real block change in the data, so it only has no record of its own, as
    for a reversal skipped by the zero-reward gate.
    The first-block record is never dropped for this reason: it is not a reversal.
    A reversal with a missing reward magnitude on either side (an unlogged trial,
    a tower with no magnitude, or arrays that end before the trial) is dropped the
    same way, with the reason "missing magnitude".
    :mod:`src.behavior_analysis.suspect_reversals` counts the same reversals with
    the same rule: of the reversals dropped inside a recording run, the ones with
    reason "tie for best" or "best tower unchanged" are its *suspect* count and
    "missing magnitude" its *uncheckable* count. Reversals logged while nothing
    was recording (between runs, after a run's last trial) are uncheckable in both:
    they are never dropped for an unchanged best tower, only for a missing
    magnitude.

    Counters (good/bad reversals, block number) are assumed to restart at each
    session, as the rig's do; the merge across sessions offsets them accordingly.
    The merged per-trial series must have as many entries as there are trials:
    a mismatch (for example sessions with different tower sets) raises
    ``ValueError`` naming the subject.

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
        - ``"subject"``, ``"session"``, ``"session_trial"``: where the reversal
          is (``session`` and ``session_trial`` are ``None`` if no session
          holds the merged trial).
        - ``"first_block"`` (bool): ``True`` for the record prepended by
          *include_first_block*. The first block is not a reversal.
        - ``"trial_window_idx"`` (dict): ``{"pre": [...], "post": [...]}``
          lists of absolute trial indices.
        - ``"reward_magnitudes_by_tower_before"`` (dict): Tower → magnitude on
          the trial immediately before the reversal.
        - ``"reward_magnitudes_by_tower_after"`` (dict): Tower → magnitude on
          the reversal trial itself. Never ``None``: a reversal with a missing
          magnitude is dropped (only the first-block record can hold ``None``).
        - ``"choices_by_tower"`` (dict): Tower → ``{"pre": [...], "post":
          [...]}`` one-hot choice slices.
        - ``"choices_by_rank"`` (dict): Rank → ``{"pre": [...], "post":
          [...]}`` one-hot choice slices.
    """
    for subj, subject_sessions in data.items():
        check_formats_agree(subj, subject_sessions)
    merged_subject_data_across_all_sessions, _ = get_vars_across_all_sessions(data)
    result, dropped, bad_only = {}, [], []
    for subj, d in merged_subject_data_across_all_sessions.items():
        _check_aligned(subj, d)
        result[subj], subj_dropped, subj_bad_only = _subject_reversals(
            subj,
            _SubjectSeries.from_merged(d),
            data.get(subj, {}),
            pre=pre,
            post=post,
            include_first_block=include_first_block,
            max_zero_count=max_zero_count,
        )
        dropped += subj_dropped
        bad_only += [(subj, key) for key in subj_bad_only]
    # Warnings are raised here, not in the helpers, so stacklevel=2 is the caller's line.
    for subj, key in bad_only:
        warnings.warn(
            f"{subj} {key}: logs bad but no good reversals, so it has no good reversals.",
            ReversalWarning,
            stacklevel=2,
        )
    if dropped:
        _warn_dropped(dropped)
    return result


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
        ValueError: If a tower has no reward magnitude (the message names the
            subject, session and trial recorded on the reversal), or if the
            best tower changes and no other tower is left.
    """
    before = reversal["reward_magnitudes_by_tower_before"]
    after = reversal["reward_magnitudes_by_tower_after"]
    where = _Where(
        reversal.get("subject"),
        reversal.get("session"),
        reversal.get("session_trial"),
        None,
        reversal.get("reversal_idx"),
    )

    prev_best = _best_tower(before, "before", where)
    next_best = _best_tower(after, "after", where)
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
    :func:`get_good_reversal_info` already drops suspect reversals, so this only
    matters for hand-built windows and for the first-block record (not a
    reversal; dropped without a warning). A record has no third tower when its
    best tower is unchanged.

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
            ReversalWarning,
            stacklevel=3,
        )
    return kept
