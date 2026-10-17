"""Choice windows aligned to changes in the TRUE reward magnitude of the best or second-best arm.

An *event* is a trial ``i`` at which the magnitude held by the ``rank`` position (``"best"`` =
the larger, ``"second"`` = the middle of the three arms' magnitudes) moved in ``direction``
relative to trial ``i - 1``. Around each event the animal's choices are labelled for the
``pre`` trials before it, the event trial itself (t = 0) and the ``post`` trials after it.
Only the true magnitudes are used (the reward magnitudes the task set on each trial), never what
the animal could have learned about them.

The third arm
-------------
The task has three arms; the *third* arm is the one with the lowest magnitude (the unrewarded
arm). It changes only at block changes, which the block-change-aligned analyses cover. So here
the third arm is held fixed:

- An event whose third arm at t = 0 differs from the third arm at t = -1 is excluded and
  counted (``n_third_arm_changed``).
- A window is cut where the third arm differs from the event's or the block changes: going
  back from t = 0 (and forward) the window extends over the trials that share the event's
  block and third arm, up to ``pre`` / ``post`` trials. Trials beyond that are ``None``. The
  window never skips over a trial that breaks it, and a trial with missing magnitudes breaks it.
- A trial whose lowest magnitude is tied has no determinable third arm: an event touching it
  is skipped and counted (``n_tie_skipped``) and it ends any window it falls in.

Window definition
-----------------
- t = 0 is the first trial with the new value. The window is ``-pre .. +post`` inclusive, so it
  has ``pre + post + 1`` points and ``post`` counts the trials AFTER t = 0.
- The window is cut as above and also at the start and end of the session. Every event is
  kept; missing trials are ``None`` and the number of events contributing at each t is
  reported (``n_events_per_bin``), so thinning is visible.
- Windows can span the recordings of one session (the task continues the same drifting values
  across a restart, so a value change at a restart is a real event) but never sessions.
- Changes at a block boundary are block changes, not events.
- The pre window also stops where the best arm last changed: only trials whose best arm is the
  best arm of t = -1 are kept (a tied best counts as its own state, so a run of tied trials
  continues only a tied t = -1). Without this, an event right after a rank change mixes in
  trials where the other arm was best, and mice still favour the arm that was best a few
  trials earlier.

Filters
-------
With the third arm fixed, the other two arms are the best and the second-best arm, and a
*rank swap* means they exchange places: the relative order of the two flips. One definition
(``_exchanged``) serves both the filter and the truncation.

- ``"all"``: every event.
- ``"rank_preserving"``: the arm that held ``rank`` before still holds it after.
- ``"rank_swap"``: the arm that held ``rank`` itself moved in ``direction`` and the best and
  second-best arms exchanged places. Possible only for (best, decrease) and (second, increase),
  so the other two combinations are rejected.
For (best, decrease) and (second, increase), the two swap-capable combinations, preserving and
swap partition the events whose best/second order is determinable.

Ties
----
Arm names never break ties. An event whose verdict depends on a tie (the best and second-best
arms equal before or after the change, or the lowest magnitude tied) is skipped and counted in
``n_tie_skipped``. The ``"all"`` filter only needs the third arm to be determinable.

Truncation
----------
``truncate_at_rank_swap`` sets the post-window labels to ``None`` from the first later trial at
which the best and second-best arms of t = 0 have strictly exchanged places (a tie is not an
exchange); a swap event is therefore not truncated by its own swap. Which events exist never
depends on truncation. An event whose best and second-best arms are tied at t = 0 has no
reference ordering; it is left untruncated and flagged (``truncation_reference_tied``).

Label sets
----------
``"best"/"second"/"third"``: the rank of the chosen arm on each trial, from the magnitudes of
that trial. When the best and second-best magnitudes tie on a trial, a choice of either tied arm
counts half for "best" and half for "second" (an even tie share); arm names never decide.
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
from src.behavior_import.extract_trials import merge_trial_variables
from src.behavior_import.session_order import ordered_session_keys

RANKS = ("best", "second")
DIRECTIONS = ("increase", "decrease")
RANK_FILTERS = ("all", "rank_preserving", "rank_swap")
STANDARD_LABELS = ("best", "second", "third")
SWAP_LABELS = ("prev_best", "prev_second", "third")
# A rank swap needs the arm that held the rank to move in the direction while being passed, so
# these two combinations cannot occur.
N_ARMS = 3
_IMPOSSIBLE_SWAPS = frozenset({("best", "increase"), ("second", "decrease")})
_REQUIRED_VARIABLES = ("reward_magnitudes", "choice", "blocks")
COUNT_KEYS = (
    "n_changes",
    "n_third_arm_changed",
    "n_tie_skipped",
    "n_filtered_out",
    "n_events",
    "n_sessions_without_trials",
    "n_missing_magnitude_trials",
)


class ValueChangeConfig(NamedTuple):
    """One analysis configuration: the tracked rank, direction and event filter.

    Attributes:
        rank (str): ``"best"`` or ``"second"``: the magnitude position that is tracked.
        direction (str): ``"increase"`` or ``"decrease"``: the direction of the change.
        rank_filter (str): ``"all"``, ``"rank_preserving"`` or ``"rank_swap"``.
    """

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


_RANK_WORDS = {"best": "Best-Value", "second": "Second-Best-Value"}
_DIRECTION_WORDS = {"increase": "Increases", "decrease": "Decreases"}
_FILTER_WORDS = {
    "all": "All Events",
    "rank_preserving": "Rank-Preserving",
    "rank_swap": "Rank Swap",
}


def describe_value_change(config: ValueChangeConfig, truncated: bool) -> str:
    """Readable name of a configuration and truncation mode, for figure titles and file names.

    Args:
        config (ValueChangeConfig): The rank, direction and event filter.
        truncated (bool): Whether the post windows are cut at the first rank swap.

    Returns:
        str: E.g. ``"Best-Value Decreases (Rank Swap, Truncated at Swap)"``.
    """
    mode = "Truncated at Swap" if truncated else "Full Window"
    rank, direction = _RANK_WORDS[config.rank], _DIRECTION_WORDS[config.direction]
    return f"{rank} {direction} ({_FILTER_WORDS[config.rank_filter]}, {mode})"


class ValueChangeEvent(NamedTuple):
    """One value-change event with its choice-label window.

    ``pre`` has ``pre`` labels (t = -pre .. -1), ``post`` has ``post + 1`` (t = 0 .. post);
    a label is ``None`` when the trial is outside the window (another block or third arm, or
    past the session edge), has no choice, or was truncated; a tied best/second trial is the
    tuple ``("best", "second")`` (half each). ``rank_swap_offset`` is the t of the first later
    best/second exchange within the
    block and window (``None`` if there is none or the reference is tied).

    Attributes:
        subject (str): Subject id.
        session (str): Session key of the event trial.
        trial_idx (int): Session-local index of the event trial (t = 0).
        pre (tuple[str | tuple[str, ...] | None, ...]): Labels at t = -pre .. -1.
        post (tuple[str | tuple[str, ...] | None, ...]): Labels at t = 0 .. post.
        contaminated (bool): Whether the magnitudes change again within the post window.
        rank_swap_offset (int | None): t of the first later best/second exchange, or ``None``.
        truncation_reference_tied (bool): Whether the best and second-best arms were tied at
            t = 0, so the window could not be truncated at a swap.
        truncated (bool): Whether the post labels were cut at a rank swap. Default False.
    """

    subject: str
    session: str
    trial_idx: int
    pre: tuple[str | tuple[str, ...] | None, ...]
    post: tuple[str | tuple[str, ...] | None, ...]
    contaminated: bool
    rank_swap_offset: int | None
    truncation_reference_tied: bool
    truncated: bool = False


class ValueChangeWindows(NamedTuple):
    """Events of one configuration, ``{subject: [ValueChangeEvent]}``, plus accounting counts.

    Attributes:
        config (ValueChangeConfig): The configuration.
        pre (int): Trials before the event in each window.
        post (int): Trials after the event trial in each window.
        truncate_at_rank_swap (bool): Whether post labels are cut at the first rank swap.
        events (dict[str, list[ValueChangeEvent]]): ``{subject: events}``.
        counts (dict[str, int]): Accounting counts, with the keys of ``COUNT_KEYS``
            (``n_changes``, ``n_third_arm_changed``, ``n_tie_skipped``, ``n_filtered_out``,
            ``n_events``, ``n_sessions_without_trials``, ``n_missing_magnitude_trials``).
    """

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
    blocks: list
    thirds: list  # third arm per trial; None where magnitudes are missing or tied


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


def check_window(pre: int, post: int) -> None:
    """Raise ValueError unless *pre* and *post* are integers >= 1.

    Args:
        pre (int): Trials before the event.
        post (int): Trials after the event trial.

    Raises:
        ValueError: If either is not an int (a bool is not) or is below 1.
    """
    for name, value in (("pre", pre), ("post", post)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1, got {value!r}")


# ---------------------------------------------------------------------------
# Magnitude helpers (ties never resolved by arm name)
# ---------------------------------------------------------------------------


def _usable(magnitudes: Any) -> bool:
    """True for a ``{arm: finite number}`` dict of three arms; False for a missing trial.

    Raises:
        ValueError: If a non-empty dict does not have exactly three arms.
    """
    if not isinstance(magnitudes, dict) or not magnitudes:
        return False
    if len(magnitudes) != N_ARMS:
        raise ValueError(f"expected {N_ARMS} arms per trial, got {magnitudes}")
    return all(isinstance(v, Real) and np.isfinite(v) for v in magnitudes.values())


def _third_arm(magnitudes: dict) -> Any:
    """The arm with the lowest magnitude, or ``None`` if that magnitude is tied."""
    lowest = min(magnitudes.values())
    arms = [arm for arm, v in magnitudes.items() if v == lowest]
    return arms[0] if len(arms) == 1 else None


def _ranked_pair(magnitudes: dict, third: Any) -> tuple[Any, Any] | None:
    """``(best, second)``: the two arms other than *third*, larger first; ``None`` on a tie."""
    first, other = [arm for arm in magnitudes if arm != third]
    if magnitudes[first] == magnitudes[other]:
        return None
    return (first, other) if magnitudes[first] > magnitudes[other] else (other, first)


def _exchanged(pair: tuple[Any, Any], magnitudes: dict) -> bool:
    """Whether the second arm of *pair* is now strictly above its first (a rank swap)."""
    best, second = pair
    return magnitudes[second] > magnitudes[best]


def _tracked_value(magnitudes: dict, rank_idx: int) -> float:
    return sorted(magnitudes.values(), reverse=True)[rank_idx]


def _moved(before: float, after: float, direction: str) -> bool:
    return after > before if direction == "increase" else after < before


def _passes_filter(before: dict, after: dict, third: Any, config: ValueChangeConfig) -> bool | None:
    """Whether the change from *before* to *after* belongs to the filter; ``None`` if a tie decides.

    Args:
        before: ``{arm: magnitude}`` at trial i - 1.
        after: ``{arm: magnitude}`` at trial i.
        third: The third arm, the same at both trials.
        config: Tracked rank, direction and filter.

    Returns:
        ``True``/``False``, or ``None`` when the verdict depends on a tie.
    """
    if config.rank_filter == "all":
        return True
    rank_idx = RANKS.index(config.rank)
    pair_before, pair_after = _ranked_pair(before, third), _ranked_pair(after, third)
    if pair_before is None or pair_after is None:
        return None
    tracked = pair_before[rank_idx]
    if config.rank_filter == "rank_preserving":
        return pair_after[rank_idx] == tracked
    # With the third arm fixed, an exchange in a swap-capable combination already implies the
    # tracked arm moved in the direction; `moved` only rejects the two impossible combinations'
    # lookalikes (e.g. best/increase, where the other arm jumped above the best arm).
    moved = _moved(before[tracked], after[tracked], config.direction)
    return moved and _exchanged(pair_before, after)


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

    Uses the importer's ``merge_trial_variables`` (every recording, in order).
    """
    if not tv_list:
        return None
    for tv in tv_list:  # per recording: equal totals could hide misaligned recordings
        _check_recording(tv, where)
    merged = merge_trial_variables(tv_list)
    _check_recording(merged, where)
    magnitudes = list(merged["reward_magnitudes"])
    thirds = [_third_arm(m) if _usable(m) else None for m in magnitudes]
    return _SessionTrials(magnitudes, list(merged["choice"]), list(merged["blocks"]), thirds)


# ---------------------------------------------------------------------------
# Event windows
# ---------------------------------------------------------------------------


def _standard_label(magnitudes: dict, choice: Any, third: Any) -> str | tuple[str, ...] | None:
    """Rank label of the chosen arm from this trial's magnitudes.

    ``None`` without a choice; a choice of one of two tied best/second arms gives
    ``("best", "second")``, which counts half for each.
    """
    if choice is None or choice == "":
        return None
    if choice not in magnitudes:
        raise ValueError(f"chosen arm {choice!r} is not one of the arms {sorted(magnitudes)}")
    if choice == third:
        return "third"
    pair = _ranked_pair(magnitudes, third)
    if pair is None:
        return ("best", "second")
    return "best" if choice == pair[0] else "second"


def swap_label_arms(magnitudes: dict) -> dict[str, Any]:
    """The arm behind each rank-swap label, from the magnitudes of trial i - 1.

    Args:
        magnitudes (dict): ``{arm: magnitude}`` of the three arms.

    Returns:
        dict[str, Any]: ``{"prev_best": arm, "prev_second": arm, "third": arm}``.

    Raises:
        ValueError: If the lowest magnitude or the best and second-best magnitudes tie, so the
            labels are not determined (such an event is never a rank-swap event).
    """
    third = _third_arm(magnitudes)
    pair = None if third is None else _ranked_pair(magnitudes, third)
    if pair is None:
        raise ValueError(f"ranks are tied, so swap labels are undetermined: {magnitudes}")
    return dict(zip(SWAP_LABELS, (*pair, third), strict=True))


def _swap_label(choice: Any, pair: tuple[Any, Any], arms: dict) -> str | None:
    """``prev_best``/``prev_second``/``third`` for the chosen arm, ``None`` without a choice."""
    if choice is None or choice == "":
        return None
    if choice not in arms:
        raise ValueError(f"chosen arm {choice!r} is not one of the arms {sorted(arms)}")
    best, second = pair
    return "prev_best" if choice == best else "prev_second" if choice == second else "third"


def _valid_run(trials: _SessionTrials, i: int, step: int, limit: int) -> int:
    """Trials from ``i + step`` outwards (up to *limit*) that share event *i*'s block and third arm.

    Stops at the first trial that does not, and at the session edge.
    """
    run = 0
    for k in range(1, limit + 1):
        j = i + step * k
        if not 0 <= j < len(trials.blocks):
            break
        if trials.blocks[j] != trials.blocks[i] or trials.thirds[j] != trials.thirds[i]:
            break
        run = k
    return run


def _best_arm(magnitudes: Any, third: Any) -> Any:
    """The best arm given the third arm; ``None`` if best and second tie (or no magnitudes)."""
    if not _usable(magnitudes):
        return None
    pair = _ranked_pair(magnitudes, third)
    return None if pair is None else pair[0]


def _stable_best_run(trials: _SessionTrials, i: int, limit: int) -> int:
    """Pre-window trials (up to *limit*) whose best arm is still the best arm of trial ``i - 1``.

    Counts back from ``i - 1`` (which always counts) and stops at the first earlier trial with
    a different best arm, so the pre window never reaches back across a change of the best arm.
    """
    third = trials.thirds[i]
    reference = _best_arm(trials.magnitudes[i - 1], third)
    run = 0
    for k in range(1, limit + 1):
        j = i - k
        if j < 0 or _best_arm(trials.magnitudes[j], third) != reference:
            break
        run = k
    return run


def _later_changes(
    trials: _SessionTrials, i: int, post_run: int, pair: tuple[Any, Any] | None
) -> tuple[bool, int | None]:
    """Contamination of the valid post window of event *i* (offsets 1 .. *post_run*).

    Returns:
        ``(contaminated, rank_swap_offset)``: whether the magnitudes changed again, and the
        offset of the first trial at which the t = 0 *pair* has exchanged places (``None`` if
        none or *pair* is ``None``).

    Raises:
        ValueError: If the arms in the window differ from the event's.
    """
    rms = trials.magnitudes
    contaminated, swap_offset = False, None
    for offset in range(1, post_run + 1):
        j = i + offset
        if set(rms[j]) != set(rms[i]):
            raise ValueError(f"arms changed within a window: {rms[i]} -> {rms[j]}")
        contaminated = contaminated or rms[j] != rms[j - 1]
        if swap_offset is None and pair is not None and _exchanged(pair, rms[j]):
            swap_offset = offset
    return contaminated, swap_offset


def _build_event(
    subject: str,
    session: str,
    trials: _SessionTrials,
    i: int,
    config: ValueChangeConfig,
    pre: int,
    post: int,
) -> ValueChangeEvent:
    """The event at trial *i*: its labelled pre/post windows and later-change flags.

    The windows are cut at the first trial of another block or third arm, and at the session
    edges; the pre window also at the last change of the best arm (see the module docstring).
    Swap labels name the arms by their rank at trial i - 1; the truncation reference is the
    best/second pair at trial i.
    """
    rms, third = trials.magnitudes, trials.thirds[i]
    pre_run = min(_valid_run(trials, i, -1, pre), _stable_best_run(trials, i, pre))
    post_run = _valid_run(trials, i, 1, post)
    pair_before, pair_at_event = _ranked_pair(rms[i - 1], third), _ranked_pair(rms[i], third)

    def label(j: int):
        if config.rank_filter == "rank_swap":
            return _swap_label(trials.choices[j], pair_before, rms[i])
        return _standard_label(rms[j], trials.choices[j], third)

    contaminated, swap_offset = _later_changes(trials, i, post_run, pair_at_event)
    return ValueChangeEvent(
        subject=subject,
        session=session,
        trial_idx=i,
        pre=tuple(label(i - k) if k <= pre_run else None for k in range(pre, 0, -1)),
        post=tuple(label(i + k) if k <= post_run else None for k in range(post + 1)),
        contaminated=contaminated,
        rank_swap_offset=swap_offset,
        truncation_reference_tied=pair_at_event is None,
    )


def _check_same_arms(before: dict, after: dict, where: str) -> None:
    if set(before) != set(after):
        raise ValueError(f"{where}: arms changed between consecutive trials: {before} -> {after}")


def _classify_change(
    trials: _SessionTrials, i: int, config: ValueChangeConfig, where: str
) -> str | None:
    """The fate of the change at trial *i*: a ``COUNT_KEYS`` name, or ``None`` if no candidate.

    ``None`` means no event candidate: a trial without magnitudes, a block change, or the
    tracked value did not move in the direction.
    """
    rms, thirds = trials.magnitudes, trials.thirds
    before, after = rms[i - 1], rms[i]
    if not (_usable(before) and _usable(after)) or trials.blocks[i] != trials.blocks[i - 1]:
        return None
    _check_same_arms(before, after, where)
    rank_idx = RANKS.index(config.rank)
    tracked_before, tracked_after = (
        _tracked_value(before, rank_idx),
        _tracked_value(after, rank_idx),
    )
    if not _moved(tracked_before, tracked_after, config.direction):
        return None
    if thirds[i - 1] is None or thirds[i] is None:
        return "n_tie_skipped"
    if thirds[i - 1] != thirds[i]:
        return "n_third_arm_changed"
    verdict = _passes_filter(before, after, thirds[i], config)
    return "n_tie_skipped" if verdict is None else "n_events" if verdict else "n_filtered_out"


def _session_events(
    subject: str,
    session: str,
    trials: _SessionTrials,
    config: ValueChangeConfig,
    pre: int,
    post: int,
) -> tuple[list[ValueChangeEvent], Counter]:
    """Events and accounting counts of one session; indices are session-local."""
    n_missing = sum(not _usable(m) for m in trials.magnitudes)
    counts: Counter = Counter(n_missing_magnitude_trials=n_missing)
    events = []
    for i in range(1, len(trials.magnitudes)):
        fate = _classify_change(trials, i, config, f"{subject} {session} trial {i}")
        if fate is None:
            continue
        counts["n_changes"] += 1
        counts[fate] += 1
        if fate == "n_events":
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
        changes found, third-arm changes, tie-skipped, filtered out and kept as events.

    Raises:
        ValueError: On an unknown argument, an impossible swap combination, a session with
            missing or unequal-length trial lists (``blocks`` are required), a session key
            that cannot be ordered, a trial without exactly three arms, or arms that change
            between consecutive trials or within a window.
    """
    config = _check_config(rank, direction, rank_filter)
    check_window(pre, post)
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


def _entry_shares(entry: Any, labels: tuple[str, ...]) -> np.ndarray:
    """Share of one trial per label: one-hot, an even split for a tuple, NaN for ``None``."""
    if entry is None:
        return np.full(len(labels), np.nan)
    names = (entry,) if isinstance(entry, str) else tuple(entry)
    bad = sorted(set(names) - set(labels))
    if bad:
        raise ValueError(f"labels {bad} are not in the label set {labels}")
    shares = np.zeros(len(labels))
    for name in names:
        shares[labels.index(name)] += 1 / len(names)
    return shares


def _event_matrix(
    event: ValueChangeEvent, pre: int, post: int, labels: tuple[str, ...]
) -> np.ndarray:
    """``(pre + post + 1, n_labels)`` shares of one event, NaN rows where the trial is missing."""
    if len(event.pre) != pre or len(event.post) != post + 1:
        raise ValueError(
            f"window length mismatch for {event.subject} {event.session} trial "
            f"{event.trial_idx}: expected pre={pre}, post={post + 1}, "
            f"got {len(event.pre)}, {len(event.post)}"
        )
    return np.array([_entry_shares(entry, labels) for entry in event.pre + event.post])


def _subject_means(
    events: list[ValueChangeEvent], pre: int, post: int, labels: tuple[str, ...]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """One subject's choice probability per label and the number of events valid at each t.

    The probability is NaN where no event has a trial.
    """
    stack = np.stack([_event_matrix(e, pre, post, labels) for e in events])  # event, t, label
    n_valid = np.sum(~np.isnan(stack[:, :, 0]), axis=0)
    total = np.nansum(stack, axis=0)  # t, label
    means = np.divide(
        total, n_valid[:, None], out=np.full(total.shape, np.nan), where=n_valid[:, None] > 0
    )
    return {label: means[:, k] for k, label in enumerate(labels)}, n_valid


def _nan_mean(stack: np.ndarray) -> np.ndarray:
    """Column mean over non-NaN rows; NaN (no warning) for columns without data."""
    n_valid = np.sum(~np.isnan(stack), axis=0)
    total = np.where(np.isnan(stack), 0.0, stack).sum(axis=0)
    return np.divide(total, n_valid, out=np.full(n_valid.shape, np.nan), where=n_valid > 0)


WEIGHTINGS = ("subject", "event")


def _subject_weighted(
    per_subject: dict[str, dict[str, np.ndarray]], labels: tuple[str, ...], shape: tuple
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Mean and SE across subjects of the per-subject probabilities."""
    mean, se = {}, {}
    for label in labels:
        stack = np.vstack([m[label] for m in per_subject.values()]) if per_subject else None
        mean[label] = _nan_mean(stack) if stack is not None else np.full(shape, np.nan)
        se[label] = nan_standard_error(stack) if stack is not None else np.full(shape, np.nan)
    return mean, se


def _event_weighted(
    events_by_subject: Mapping[str, list[ValueChangeEvent]],
    pre: int,
    post: int,
    labels: tuple[str, ...],
    shape: tuple,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Mean and SE across all events, every event counting once whatever its subject."""
    events = [e for evs in events_by_subject.values() for e in evs]
    if not events:
        empty = {label: np.full(shape, np.nan) for label in labels}
        return empty, dict(empty)
    stack = np.stack([_event_matrix(e, pre, post, labels) for e in events])  # event, t, label
    mean = {label: _nan_mean(stack[:, :, k]) for k, label in enumerate(labels)}
    se = {label: nan_standard_error(stack[:, :, k]) for k, label in enumerate(labels)}
    return mean, se


def aggregate_windows(
    events_by_subject: Mapping[str, list[ValueChangeEvent]],
    pre: int,
    post: int,
    labels: tuple[str, ...],
    *,
    weighting: str = "subject",
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]], dict]:
    """Per-subject choice probabilities, then mean and SE across subjects or across events.

    With ``weighting="subject"`` each subject's events are pooled into one probability per t,
    and ``mean``/``se`` are taken across subjects, so every subject counts once. With
    ``weighting="event"`` every event counts once: ``mean`` is the share over all events with
    a trial at t, whichever subject they come from, and ``se`` is across those events. That SE
    treats events as independent; events of one subject are not, so it is too narrow for
    inference and is meant for comparing the shape of the curves.

    Args:
        events_by_subject (dict): ``{subject: [ValueChangeEvent]}``.
        pre (int): Trials before the event.
        post (int): Trials after the event trial.
        labels (tuple[str]): The label set (explicit, from the filter); every label in an
            event must be in it.
        weighting (str): ``"subject"`` (default) or ``"event"``; see above.

    Returns:
        tuple: ``(x, per_subject_means, across)``. ``x`` is ``-pre .. post``;
        ``per_subject_means`` is ``{subject: {label: array}}``; ``across`` has ``mean`` and
        ``se`` (``{label: array}``; SE is NaN where fewer than two subjects, or with
        ``weighting="event"`` two events, have data), ``weighting``,
        ``num_subjects``, ``num_events``, ``num_truncated`` and, per t, ``n_events_per_bin``
        and ``n_subjects_per_bin`` (events and subjects with a trial at that t).

    Raises:
        ValueError: If an event's window has the wrong length, a label outside *labels*, or
            *weighting* is not ``"subject"`` or ``"event"``.
    """
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}, got {weighting!r}")
    x = np.arange(-pre, post + 1, dtype=float)
    per_subject, n_valid = {}, {}
    for subject, events in events_by_subject.items():
        if events:
            per_subject[subject], n_valid[subject] = _subject_means(events, pre, post, labels)
    if weighting == "event":
        mean, se = _event_weighted(events_by_subject, pre, post, labels, x.shape)
    else:
        mean, se = _subject_weighted(per_subject, labels, x.shape)
    counts = np.vstack(list(n_valid.values())) if n_valid else np.zeros((1, x.size), dtype=int)
    all_events = [e for events in events_by_subject.values() for e in events]
    across = {
        "mean": mean,
        "se": se,
        "weighting": weighting,
        "num_subjects": len(per_subject),
        "num_events": len(all_events),
        "num_truncated": sum(e.truncated for e in all_events),
        "n_events_per_bin": counts.sum(axis=0),
        "n_subjects_per_bin": (counts > 0).sum(axis=0),
    }
    return x, per_subject, across


class ValueChangeCurves(NamedTuple):
    """Aggregated curves of one configuration, ready to plot.

    Attributes:
        config (ValueChangeConfig): The configuration.
        truncate_at_rank_swap (bool): Whether post labels were cut at the first rank swap.
        pre (int): Trials before the event.
        post (int): Trials after the event trial.
        x (np.ndarray): Float array ``-pre .. post``, shape ``(pre + post + 1,)``.
        labels (tuple[str, ...]): The label set of the curves.
        per_subject_mean (dict[str, dict[str, np.ndarray]]): ``{subject: {label: choice
            probability per t}}``.
        mean (dict[str, np.ndarray]): ``{label: mean per t}``, across subjects or across
            events according to *weighting*.
        se (dict[str, np.ndarray]): ``{label: standard error per t}``, across the same units;
            NaN where fewer than two of them have data.

        num_subjects (int): Subjects with at least one event.
        num_events (int): Events averaged.
        num_truncated (int): Events whose post labels were truncated.
        n_events_per_bin (np.ndarray): Events with a trial at each t, shape ``(pre + post + 1,)``.
        n_subjects_per_bin (np.ndarray): Subjects with a trial at each t, same shape.
        counts (dict[str, int]): Accounting counts, as in :class:`ValueChangeWindows`.
        weighting (str): ``"subject"`` (every mouse counts once) or ``"event"`` (every event
            counts once; the SE then treats events as independent, so it understates the
            uncertainty).
    """

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
    n_events_per_bin: np.ndarray
    n_subjects_per_bin: np.ndarray
    counts: dict[str, int]
    weighting: str = "subject"


def summarize_windows(
    windows: ValueChangeWindows, *, weighting: str = "subject"
) -> ValueChangeCurves:
    """Aggregate *windows* into curves with the label set of its filter.

    Args:
        windows (ValueChangeWindows): Events of one configuration.
        weighting (str): ``"subject"`` (default; mean and SE across mice) or ``"event"``
            (mean and SE across events), as in :func:`aggregate_windows`.

    Returns:
        ValueChangeCurves: x axis, per-subject curves, mean and SE, event counts.

    Raises:
        ValueError: If *weighting* is not ``"subject"`` or ``"event"``.
    """
    x, per_subject, across = aggregate_windows(
        windows.events, windows.pre, windows.post, windows.labels, weighting=weighting
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
        weighting=weighting,
        num_subjects=across["num_subjects"],
        num_events=across["num_events"],
        num_truncated=across["num_truncated"],
        n_events_per_bin=across["n_events_per_bin"],
        n_subjects_per_bin=across["n_subjects_per_bin"],
        counts=dict(windows.counts),
    )


# ---------------------------------------------------------------------------
# Contamination diagnostic
# ---------------------------------------------------------------------------


def contamination_summary(events_by_subject: Mapping[str, list[ValueChangeEvent]]) -> dict:
    """How many post windows contain a further value change, and how many a rank swap.

    A window is contaminated if the reward magnitudes change again at a later trial of the
    event's block and third arm within the post window; ``n_rank_swap`` counts windows where the
    t = 0 best
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
