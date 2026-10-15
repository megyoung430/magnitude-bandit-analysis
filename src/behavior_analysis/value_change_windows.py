"""Choice windows aligned to changes in the TRUE reward magnitude of the best or second-best arm.

An *event* is a trial ``i`` at which the magnitude held by the ``rank`` position (``"best"`` =
the largest of the arms' magnitudes, ``"second"`` = the middle one) moved in ``direction``
relative to trial ``i - 1``. Around each event the animal's choices are labelled for the
``pre`` trials before it, the event trial itself (t = 0) and the ``post`` trials after it.
Only the true magnitudes are used (the reward magnitudes the task set on each trial), never what
the animal could have learned about them.

Window definition
-----------------
- t = 0 is the first trial with the new value. The window is ``-pre .. +post`` inclusive, so it
  has ``pre + post + 1`` points and ``post`` counts the trials AFTER t = 0.
- Only complete windows are kept: an event needs ``pre`` trials before it and ``post`` trials
  after it inside its own session. Events without room are dropped and counted
  (``n_dropped_no_pre_room`` / ``n_dropped_no_post_room``); windows never cross sessions.
- Changes at a block boundary are reversals, not events, and a window's labels are ``None``
  for trials outside the event's block.

Filters
-------
- ``"all"``: every event.
- ``"rank_preserving"``: the arm that held ``rank`` before still holds it after.
- ``"rank_swap"``: the arm that held ``rank`` itself moved in ``direction`` and the best and
  second-best arms exchanged places. Possible only for (best, decrease) and (second, increase)
  when one arm changes at a time, so the other two combinations are rejected.
The filters do not partition the events: e.g. a third arm overtaking the second-best arm is in
neither.

Ties
----
Arm names never break ties. An event whose filter verdict depends on a tie (the arm holding a
position is not unique before or after the change) is skipped and counted in ``n_tie_skipped``.
The ``"all"`` filter needs no ordering, so ties never affect it.

Truncation
----------
``truncate_at_rank_swap`` sets the post-window labels to ``None`` from the first later trial at
which the two arms that were best and second-best at t = 0 have strictly exchanged places (a
tie is not an exchange). Which events exist never depends on truncation. An event whose best
and second-best arms are tied at t = 0 has no reference ordering; it is left untruncated and
flagged (``truncation_reference_tied``).

Label sets
----------
``"best"/"second"/"third"``: the importer's rank of the arm chosen on each trial.
``"prev_best"/"prev_second"/"third"`` (rank-swap filter): the chosen arm's rank at trial i - 1,
so a swap's two arms keep their identity across the event.
"""

from collections import Counter
from collections.abc import Mapping
from numbers import Real
from typing import Any, NamedTuple

import numpy as np

from src.behavior_analysis.merge_across_problems import merge_windows_by_subject
from src.behavior_analysis.stats import nan_standard_error
from src.behavior_import.extract_trials import RANK_LABEL_MAP, merge_trial_variables
from src.behavior_import.session_order import ordered_session_keys

RANKS = ("best", "second")
DIRECTIONS = ("increase", "decrease")
RANK_FILTERS = ("all", "rank_preserving", "rank_swap")
STANDARD_LABELS = ("best", "second", "third")
SWAP_LABELS = ("prev_best", "prev_second", "third")
# A rank swap needs the tracked arm to move in the direction while being passed, so these two
# combinations cannot occur.
_IMPOSSIBLE_SWAPS = frozenset({("best", "increase"), ("second", "decrease")})
_REQUIRED_VARIABLES = ("reward_magnitudes", "choice", "chosen_rank", "blocks")
COUNT_KEYS = (
    "n_changes",
    "n_filtered_out",
    "n_tie_skipped",
    "n_dropped_no_pre_room",
    "n_dropped_no_post_room",
    "n_events",
    "n_sessions_without_trials",
    "n_missing_magnitude_trials",
)


class ValueChangeConfig(NamedTuple):
    """One analysis configuration: the tracked rank, direction and event filter."""

    rank: str
    direction: str
    rank_filter: str

    @property
    def label(self) -> str:
        return f"{self.rank}/{self.direction}/{self.rank_filter}"

    @property
    def labels(self) -> tuple[str, ...]:
        """Label set of this configuration's windows (explicit from the filter)."""
        return SWAP_LABELS if self.rank_filter == "rank_swap" else STANDARD_LABELS


class ValueChangeEvent(NamedTuple):
    """One value-change event with its choice-label window.

    ``pre`` has ``pre`` labels (t = -pre .. -1), ``post`` has ``post + 1`` (t = 0 .. post);
    a label is ``None`` when the trial is outside the event's block, has no choice, or was
    truncated. ``rank_swap_offset`` is the t of the first later best/second exchange within the
    block and window (``None`` if there is none or the reference is tied).
    """

    subject: str
    session: str
    trial_idx: int
    pre: tuple[str | None, ...]
    post: tuple[str | None, ...]
    contaminated: bool
    rank_swap_offset: int | None
    truncation_reference_tied: bool
    truncated: bool = False


class ValueChangeWindows(NamedTuple):
    """Events of one configuration, ``{subject: [ValueChangeEvent]}``, plus accounting counts."""

    config: ValueChangeConfig
    pre: int
    post: int
    truncate_at_rank_swap: bool
    events: dict[str, list[ValueChangeEvent]]
    counts: dict[str, int]

    @property
    def labels(self) -> tuple[str, ...]:
        return self.config.labels


class _SessionTrials(NamedTuple):
    magnitudes: list
    choices: list
    ranks: list
    blocks: list


# ---------------------------------------------------------------------------
# Configurations and argument checks
# ---------------------------------------------------------------------------


def value_change_configs() -> list[ValueChangeConfig]:
    """Every (rank, direction, filter) configuration, without the impossible swap combinations.

    Returns:
        list[ValueChangeConfig]: 10 configurations, ordered rank, direction, filter.
    """
    return [
        ValueChangeConfig(rank, direction, rank_filter)
        for rank in RANKS
        for direction in DIRECTIONS
        for rank_filter in RANK_FILTERS
        if not (rank_filter == "rank_swap" and (rank, direction) in _IMPOSSIBLE_SWAPS)
    ]


def _check_config(rank: str, direction: str, rank_filter: str) -> ValueChangeConfig:
    """The configuration, or ValueError for an unknown value or an impossible swap."""
    for name, value, allowed in (
        ("rank", rank, RANKS),
        ("direction", direction, DIRECTIONS),
        ("rank_filter", rank_filter, RANK_FILTERS),
    ):
        if value not in allowed:
            raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
    if rank_filter == "rank_swap" and (rank, direction) in _IMPOSSIBLE_SWAPS:
        raise ValueError(
            f"rank_filter='rank_swap' with rank={rank!r} and direction={direction!r} is "
            "impossible: the arm that held the rank would have to move that way and still "
            "be passed. Possible swaps: (second, increase) and (best, decrease)."
        )
    return ValueChangeConfig(rank, direction, rank_filter)


def _check_window(pre: int, post: int) -> None:
    for name, value in (("pre", pre), ("post", post)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1, got {value!r}")


# ---------------------------------------------------------------------------
# Magnitude helpers (ties never resolved by arm name)
# ---------------------------------------------------------------------------


def _usable(magnitudes: Any) -> bool:
    """True for a ``{arm: finite number}`` dict of at least two arms."""
    return (
        isinstance(magnitudes, dict)
        and len(magnitudes) >= 2
        and all(isinstance(v, Real) and np.isfinite(v) for v in magnitudes.values())
    )


def _position(magnitudes: dict, arm: Any) -> int | None:
    """Rank position (0 = largest) of *arm*, or ``None`` if another arm has the same value."""
    value = magnitudes[arm]
    if sum(v == value for v in magnitudes.values()) > 1:
        return None
    return sum(v > value for v in magnitudes.values())


def _arm_at(magnitudes: dict, position: int) -> Any:
    """The arm uniquely holding *position*, or ``None`` if a tie leaves it undetermined."""
    for arm in magnitudes:
        if _position(magnitudes, arm) == position:
            return arm
    return None


def _tracked_value(magnitudes: dict, rank_idx: int) -> float:
    return sorted(magnitudes.values(), reverse=True)[rank_idx]


def _moved(before: float, after: float, direction: str) -> bool:
    return after > before if direction == "increase" else after < before


def _passes_filter(before: dict, after: dict, config: ValueChangeConfig) -> bool | None:
    """Whether the change from *before* to *after* belongs to the filter; ``None`` if a tie decides.

    Args:
        before: ``{arm: magnitude}`` at trial i - 1.
        after: ``{arm: magnitude}`` at trial i.
        config: Tracked rank, direction and filter.

    Returns:
        ``True``/``False``, or ``None`` when the verdict depends on a tie.
    """
    if config.rank_filter == "all":
        return True
    rank_idx = RANKS.index(config.rank)
    tracked = _arm_at(before, rank_idx)
    if tracked is None:
        return None
    if config.rank_filter == "rank_preserving":
        position = _position(after, tracked)
        return None if position is None else position == rank_idx
    if not _moved(before[tracked], after[tracked], config.direction):
        return False
    best, second = _arm_at(before, 0), _arm_at(before, 1)
    if best is None or second is None:
        return None
    best_after, second_after = _position(after, best), _position(after, second)
    if best_after is None or second_after is None:
        return None
    return second_after == 0 and best_after == 1


# ---------------------------------------------------------------------------
# Session trial lists
# ---------------------------------------------------------------------------


def _check_recording(tv: Mapping, where: str) -> None:
    """Raise ValueError unless the recording has the required lists, all equally long."""
    missing = [name for name in _REQUIRED_VARIABLES if name not in tv]
    if missing:
        raise ValueError(f"{where}: trial variables lack {missing}; blocks are required")
    lengths = {name: len(tv[name]) for name in _REQUIRED_VARIABLES}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"{where}: trial lists differ in length: {lengths}")


def _session_trials(tv_list: Any, where: str) -> _SessionTrials | None:
    """All recordings of a session merged into one trial list, or ``None`` if it has none.

    Uses the importer's ``merge_trial_variables``, so later recordings are not dropped.
    """
    if not tv_list:
        return None
    for tv in tv_list:  # per recording: equal totals could hide misaligned recordings
        _check_recording(tv, where)
    merged = merge_trial_variables(tv_list)
    _check_recording(merged, where)
    return _SessionTrials(*(list(merged[name]) for name in _REQUIRED_VARIABLES))


# ---------------------------------------------------------------------------
# Event windows
# ---------------------------------------------------------------------------


def _standard_label(rank: Any) -> str | None:
    """The importer's rank label of one trial, ``None`` for a trial without a choice."""
    if rank is None or rank == "":
        return None
    if rank not in RANK_LABEL_MAP:
        raise ValueError(f"unknown chosen_rank {rank!r}; expected one of {sorted(RANK_LABEL_MAP)}")
    return RANK_LABEL_MAP[rank]


def _swap_label(choice: Any, best: Any, second: Any, arms: dict) -> str | None:
    """``prev_best``/``prev_second``/``third`` for the chosen arm, ``None`` without a choice."""
    if choice is None or choice == "":
        return None
    if choice not in arms:
        raise ValueError(f"chosen arm {choice!r} is not one of the arms {sorted(arms)}")
    return "prev_best" if choice == best else "prev_second" if choice == second else "third"


def _window_labels(trials: _SessionTrials, i: int, config: ValueChangeConfig, lo: int, hi: int):
    """Labels of trials ``lo .. hi - 1`` around event *i*; ``None`` outside the event's block."""
    before = trials.magnitudes[i - 1]
    best, second = _arm_at(before, 0), _arm_at(before, 1)
    labels = []
    for j in range(lo, hi):
        if trials.blocks[j] != trials.blocks[i]:
            labels.append(None)
        elif config.rank_filter == "rank_swap":
            labels.append(_swap_label(trials.choices[j], best, second, before))
        else:
            labels.append(_standard_label(trials.ranks[j]))
    return tuple(labels)


def _later_changes(trials: _SessionTrials, i: int, post: int) -> tuple[bool, int | None, bool]:
    """Contamination of the post window of event *i*.

    Returns:
        ``(contaminated, rank_swap_offset, reference_tied)``: whether the magnitudes changed
        again at a later trial of the event's block within ``post`` trials; the offset of the
        first strict exchange of the t = 0 best and second-best arms (``None`` if none); and
        whether those two arms are undetermined at t = 0 because of a tie.
    """
    rms = trials.magnitudes
    top, second = _arm_at(rms[i], 0), _arm_at(rms[i], 1)
    reference_tied = top is None or second is None
    contaminated, swap_offset = False, None
    for offset in range(1, post + 1):
        j = i + offset
        if trials.blocks[j] != trials.blocks[i]:
            break
        if not _usable(rms[j]):
            continue
        if _usable(rms[j - 1]) and rms[j] != rms[j - 1]:
            contaminated = True
        if swap_offset is None and not reference_tied and rms[j][second] > rms[j][top]:
            swap_offset = offset
    return contaminated or swap_offset is not None, swap_offset, reference_tied


def _build_event(
    subject: str,
    session: str,
    trials: _SessionTrials,
    i: int,
    config: ValueChangeConfig,
    pre: int,
    post: int,
) -> ValueChangeEvent:
    contaminated, swap_offset, reference_tied = _later_changes(trials, i, post)
    return ValueChangeEvent(
        subject=subject,
        session=session,
        trial_idx=i,
        pre=_window_labels(trials, i, config, i - pre, i),
        post=_window_labels(trials, i, config, i, i + post + 1),
        contaminated=contaminated,
        rank_swap_offset=swap_offset,
        truncation_reference_tied=reference_tied,
    )


def _check_same_arms(before: dict, after: dict, where: str) -> None:
    if set(before) != set(after):
        raise ValueError(f"{where}: arms changed between consecutive trials: {before} -> {after}")


def _session_events(
    subject: str,
    session: str,
    trials: _SessionTrials,
    config: ValueChangeConfig,
    pre: int,
    post: int,
) -> tuple[list[ValueChangeEvent], Counter]:
    """Events and accounting counts of one session; indices are session-local."""
    rms, n = trials.magnitudes, len(trials.magnitudes)
    rank_idx = RANKS.index(config.rank)
    usable = [_usable(rm) for rm in rms]
    counts: Counter = Counter(n_missing_magnitude_trials=n - sum(usable))
    events = []
    for i in range(1, n):
        if not (usable[i] and usable[i - 1]) or trials.blocks[i] != trials.blocks[i - 1]:
            continue
        before, after = rms[i - 1], rms[i]
        _check_same_arms(before, after, f"{subject} {session} trial {i}")
        if not _moved(
            _tracked_value(before, rank_idx), _tracked_value(after, rank_idx), config.direction
        ):
            continue
        counts["n_changes"] += 1
        verdict = _passes_filter(before, after, config)
        if verdict is None:
            counts["n_tie_skipped"] += 1
        elif not verdict:
            counts["n_filtered_out"] += 1
        elif i < pre:
            counts["n_dropped_no_pre_room"] += 1
        elif i + post >= n:
            counts["n_dropped_no_post_room"] += 1
        else:
            counts["n_events"] += 1
            events.append(_build_event(subject, session, trials, i, config, pre, post))
    return events, counts


def _subject_events(
    subject: str,
    sessions: Mapping[str, Mapping],
    config: ValueChangeConfig,
    pre: int,
    post: int,
) -> tuple[list[ValueChangeEvent], Counter]:
    """Events of one subject, sessions visited in session order; none cross a session."""
    events, counts = [], Counter()
    for session in ordered_session_keys(subject, sessions):
        trials = _session_trials(sessions[session].get("trial_variables"), f"{subject} {session}")
        if trials is None:
            counts["n_sessions_without_trials"] += 1
            continue
        session_events, session_counts = _session_events(
            subject, session, trials, config, pre, post
        )
        events += session_events
        counts += session_counts
    return events, counts


def get_value_change_windows(
    subjects_trials: Mapping[str, Mapping[str, Mapping]],
    rank: str,
    direction: str,
    rank_filter: str = "all",
    *,
    pre: int = 5,
    post: int = 15,
    truncate_at_rank_swap: bool = False,
) -> ValueChangeWindows:
    """Events where the true magnitude at the *rank* position changed, with choice windows.

    Args:
        subjects_trials (dict): ``{subject: {session_key: session}}`` for one problem; each
            session holds ``"trial_variables"``, a list with one dict per recording.
        rank (str): ``"best"`` or ``"second"``.
        direction (str): ``"increase"`` or ``"decrease"``.
        rank_filter (str): ``"all"``, ``"rank_preserving"`` or ``"rank_swap"``.
        pre (int): Trials before the event in the window (>= 1).
        post (int): Trials after the event trial in the window (>= 1).
        truncate_at_rank_swap (bool): Blank the post window from the first later best/second
            exchange (see the module docstring).

    Returns:
        ValueChangeWindows: Events per subject (only subjects with events) and counts of
        changes found, filtered out, tie-skipped and dropped for lack of room.

    Raises:
        ValueError: On an unknown argument, an impossible swap combination, a session with
            missing or unequal-length trial lists (``blocks`` are required), a session key
            that cannot be ordered, or arms that change between consecutive trials.
    """
    config = _check_config(rank, direction, rank_filter)
    _check_window(pre, post)
    events: dict[str, list[ValueChangeEvent]] = {}
    counts: Counter = Counter({key: 0 for key in COUNT_KEYS})
    for subject, sessions in subjects_trials.items():
        subject_events, subject_counts = _subject_events(subject, sessions, config, pre, post)
        counts += subject_counts
        if subject_events:
            events[subject] = subject_events
    result = ValueChangeWindows(
        config, pre, post, False, events, {k: counts[k] for k in COUNT_KEYS}
    )
    return truncate_windows_at_rank_swap(result) if truncate_at_rank_swap else result


def _truncate_event(event: ValueChangeEvent) -> ValueChangeEvent:
    offset = event.rank_swap_offset
    if offset is None:
        return event
    post = event.post[:offset] + (None,) * (len(event.post) - offset)
    return event._replace(post=post, truncated=True)


def truncate_windows_at_rank_swap(windows: ValueChangeWindows) -> ValueChangeWindows:
    """The same events with each post window blanked from its first later best/second exchange.

    Args:
        windows (ValueChangeWindows): Windows to truncate.

    Returns:
        ValueChangeWindows: Same events and counts; ``truncate_at_rank_swap`` is True.
    """
    events = {s: [_truncate_event(e) for e in evs] for s, evs in windows.events.items()}
    return windows._replace(events=events, truncate_at_rank_swap=True)


def merge_value_change_windows(windows_list: list[ValueChangeWindows]) -> ValueChangeWindows:
    """Pool windows computed per problem into one per-mouse set of events.

    Events stay within their session, so pooling concatenates each mouse's events and sums the
    counts; session keys are never combined.

    Args:
        windows_list (list[ValueChangeWindows]): One per problem, same configuration,
            ``pre``, ``post`` and truncation setting.

    Returns:
        ValueChangeWindows: Events concatenated per mouse, counts summed.

    Raises:
        ValueError: If the list is empty or the settings differ.
    """
    if not windows_list:
        raise ValueError("no windows to merge")
    first = windows_list[0]
    settings = (first.config, first.pre, first.post, first.truncate_at_rank_swap)
    for w in windows_list:
        if (w.config, w.pre, w.post, w.truncate_at_rank_swap) != settings:
            raise ValueError(f"cannot merge windows with different settings: {settings} vs {w[:4]}")
    events = merge_windows_by_subject(dict(enumerate(w.events for w in windows_list)))
    counts = {k: sum(w.counts[k] for w in windows_list) for k in COUNT_KEYS}
    return first._replace(events=events, counts=counts)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _event_rows(events: list[ValueChangeEvent], pre: int, post: int, labels: tuple[str, ...]):
    """Validated label rows, one per event: ``(n_events, pre + post + 1)`` object list."""
    rows = []
    for e in events:
        if len(e.pre) != pre or len(e.post) != post + 1:
            raise ValueError(
                f"window length mismatch for {e.subject} {e.session} trial {e.trial_idx}: "
                f"expected pre={pre}, post={post + 1}, got {len(e.pre)}, {len(e.post)}"
            )
        row = e.pre + e.post
        bad = {label for label in row if label is not None and label not in labels}
        if bad:
            raise ValueError(f"labels {sorted(bad)} are not in the label set {labels}")
        rows.append(row)
    return rows


def _subject_means(rows: list, labels: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Per-time-point fraction of valid events choosing each label; NaN with no valid event."""
    valid = np.array([[label is not None for label in row] for row in rows], dtype=float)
    n_valid = valid.sum(axis=0)
    means = {}
    for label in labels:
        hits = np.array([[lab == label for lab in row] for row in rows], dtype=float)
        means[label] = np.divide(
            hits.sum(axis=0), n_valid, out=np.full(n_valid.shape, np.nan), where=n_valid > 0
        )
    return means


def _nan_mean(stack: np.ndarray) -> np.ndarray:
    """Column mean over non-NaN rows; NaN (no warning) for columns without data."""
    n_valid = np.sum(~np.isnan(stack), axis=0)
    total = np.where(np.isnan(stack), 0.0, stack).sum(axis=0)
    return np.divide(total, n_valid, out=np.full(n_valid.shape, np.nan), where=n_valid > 0)


def aggregate_windows(
    events_by_subject: Mapping[str, list[ValueChangeEvent]],
    pre: int,
    post: int,
    labels: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]], dict]:
    """Per-subject choice probabilities, then mean and SE across subjects.

    Args:
        events_by_subject (dict): ``{subject: [ValueChangeEvent]}``.
        pre (int): Trials before the event.
        post (int): Trials after the event trial.
        labels (tuple[str]): The label set (explicit, from the filter); every label in an
            event must be in it.

    Returns:
        tuple: ``(x, per_subject_means, across)``. ``x`` is ``-pre .. post``;
        ``per_subject_means`` is ``{subject: {label: array}}``; ``across`` has ``mean`` and
        ``se`` (``{label: array}``; SE is NaN where fewer than two subjects have data),
        ``num_subjects``, ``num_events`` and ``num_truncated``.

    Raises:
        ValueError: If an event's window has the wrong length or a label outside *labels*.
    """
    x = np.arange(-pre, post + 1, dtype=float)
    per_subject = {}
    for subject, events in events_by_subject.items():
        if events:
            per_subject[subject] = _subject_means(_event_rows(events, pre, post, labels), labels)
    mean, se = {}, {}
    for label in labels:
        stack = np.vstack([m[label] for m in per_subject.values()]) if per_subject else None
        mean[label] = _nan_mean(stack) if stack is not None else np.full(x.shape, np.nan)
        se[label] = nan_standard_error(stack) if stack is not None else np.full(x.shape, np.nan)
    all_events = [e for events in events_by_subject.values() for e in events]
    across = {
        "mean": mean,
        "se": se,
        "num_subjects": len(per_subject),
        "num_events": len(all_events),
        "num_truncated": sum(e.truncated for e in all_events),
    }
    return x, per_subject, across


class ValueChangeCurves(NamedTuple):
    """Aggregated curves of one configuration, ready to plot."""

    config: ValueChangeConfig
    truncate_at_rank_swap: bool
    pre: int
    post: int
    x: np.ndarray
    labels: tuple[str, ...]
    per_subject_mean: dict[str, dict[str, np.ndarray]]
    mean: dict[str, np.ndarray]
    se: dict[str, np.ndarray]
    num_subjects: int
    num_events: int
    num_truncated: int
    counts: dict[str, int]


def summarize_windows(windows: ValueChangeWindows) -> ValueChangeCurves:
    """Aggregate *windows* into curves with the label set of its filter.

    Args:
        windows (ValueChangeWindows): Events of one configuration.

    Returns:
        ValueChangeCurves: x axis, per-subject and across-subject mean and SE, event counts.
    """
    x, per_subject, across = aggregate_windows(
        windows.events, windows.pre, windows.post, windows.labels
    )
    return ValueChangeCurves(
        config=windows.config,
        truncate_at_rank_swap=windows.truncate_at_rank_swap,
        pre=windows.pre,
        post=windows.post,
        x=x,
        labels=windows.labels,
        per_subject_mean=per_subject,
        mean=across["mean"],
        se=across["se"],
        num_subjects=across["num_subjects"],
        num_events=across["num_events"],
        num_truncated=across["num_truncated"],
        counts=dict(windows.counts),
    )


# ---------------------------------------------------------------------------
# Contamination diagnostic
# ---------------------------------------------------------------------------


def contamination_summary(events_by_subject: Mapping[str, list[ValueChangeEvent]]) -> dict:
    """How many post windows contain a further value change, and how many a rank swap.

    A window is contaminated if the reward magnitudes change again at a later trial of the
    event's block within the post window; ``n_rank_swap`` counts windows where the t = 0 best
    and second-best arms strictly exchange places. Both come from the event's own session, so
    trial positions are session-local. ``n_reference_tied`` counts events whose t = 0
    best/second-best arms are tied, for which a swap is undefined (not counted in
    ``n_rank_swap``).

    Args:
        events_by_subject (dict): ``{subject: [ValueChangeEvent]}``.

    Returns:
        dict: ``n_events``, ``n_contaminated``, ``frac_contaminated`` (NaN with no events),
        ``n_rank_swap``, ``frac_rank_swap`` (of contaminated windows; NaN with none) and
        ``n_reference_tied``.
    """
    events = [e for evs in events_by_subject.values() for e in evs]
    n_events = len(events)
    n_contaminated = sum(e.contaminated for e in events)
    n_swap = sum(e.rank_swap_offset is not None for e in events)
    return {
        "n_events": n_events,
        "n_contaminated": n_contaminated,
        "frac_contaminated": n_contaminated / n_events if n_events else float("nan"),
        "n_rank_swap": n_swap,
        "frac_rank_swap": n_swap / n_contaminated if n_contaminated else float("nan"),
        "n_reference_tied": sum(e.truncation_reference_tied for e in events),
    }
