"""Symmetric, complete-window selection of value-change events.

The regular value-change windows keep every event and cut a window where it meets a block or
third-arm change, a session edge or (before the event) a change of the best arm, so the curves
at large ``|t|`` average fewer, different events than those near t = 0. The symmetric version
instead keeps only events whose whole ``-N .. +N`` window is intact:

- every one of the ``2N + 1`` trials exists in the event's block with the event's third arm
  (so no block change, third-arm change, tied or missing magnitudes, or session edge);
- the best arm does not change in the pre window (the pre-window rule of the regular windows:
  an event whose pre window that rule would cut is excluded, not truncated);
- every trial has a choice. A trial that exists but has no choice (a missed trial) leaves a
  hole in the window, so the event is excluded and counted as ``other``.

A best-arm change in the post window is allowed: it is the outcome the figure looks at, and the
regular windows allow it too. Rank-swap truncation is not applied, since blanking post trials
would break the symmetry.
"""

from collections import Counter

from src.behavior_analysis.value_change_windows import (
    CUT_BEST_ARM,
    CUT_BLOCK_OR_THIRD_ARM,
    CUT_COMPLETE,
    CUT_SESSION_EDGE,
    ValueChangeEvent,
    ValueChangeWindows,
)

EXCLUSION_KEYS = (
    "n_excluded_block_or_third_arm_change",
    "n_excluded_session_edge",
    "n_excluded_best_arm_change",
    "n_excluded_other",
)
# Cut reason -> exclusion count; the order is the precedence when an event has several causes.
_REASON_KEYS = {
    CUT_BLOCK_OR_THIRD_ARM: EXCLUSION_KEYS[0],
    CUT_SESSION_EDGE: EXCLUSION_KEYS[1],
    CUT_BEST_ARM: EXCLUSION_KEYS[2],
}
OTHER_KEY = EXCLUSION_KEYS[3]


def _exclusion_key(event: ValueChangeEvent) -> str | None:
    """The ``EXCLUSION_KEYS`` entry that explains why *event*'s window is incomplete, or None.

    An event with several causes is counted once, under the first of: block or third-arm
    change, session edge, best-arm change, other (a trial without a choice).

    Raises:
        ValueError: If the event does not record why its window was cut.
    """
    if event.pre_cut is None or event.post_cut is None:
        raise ValueError(
            f"{event.subject} {event.session} trial {event.trial_idx}: the event does not record "
            "why its window was cut (pre_cut / post_cut); build it with get_value_change_windows"
        )
    causes = {event.pre_cut, event.post_cut} - {CUT_COMPLETE}
    for reason, key in _REASON_KEYS.items():
        if reason in causes:
            return key
    return OTHER_KEY if None in event.pre + event.post else None


def select_symmetric_windows(windows: ValueChangeWindows) -> tuple[ValueChangeWindows, dict]:
    """Keep the events whose ``-N .. +N`` window is complete, and count why the others are not.

    Args:
        windows (ValueChangeWindows): Untruncated windows built with ``pre == post == N``.

    Returns:
        tuple: ``(kept, exclusions)``. *kept* has the same configuration and counts and only the
        complete events (subjects without any are dropped), with ``symmetric`` True.
        *exclusions* is ``{"n_events_before": ..., "n_events_kept": ..., <each of
        EXCLUSION_KEYS>: ...}``; the exclusion counts add up to ``n_events_before -
        n_events_kept``.

    Raises:
        ValueError: If the windows are truncated or already symmetric, ``pre != post``, or an
            event does not record why its window was cut.
    """
    if windows.truncate_at_rank_swap:
        raise ValueError("symmetric selection needs untruncated windows")
    if windows.symmetric:
        raise ValueError("windows are already symmetric")
    if windows.pre != windows.post:
        raise ValueError(
            f"symmetric selection needs pre == post, got {windows.pre}, {windows.post}"
        )
    excluded: Counter = Counter({key: 0 for key in EXCLUSION_KEYS})
    kept: dict[str, list[ValueChangeEvent]] = {}
    n_before = 0
    for subject, events in windows.events.items():
        for event in events:
            n_before += 1
            key = _exclusion_key(event)
            if key is None:
                kept.setdefault(subject, []).append(event)
            else:
                excluded[key] += 1
    n_kept = sum(map(len, kept.values()))
    exclusions = {"n_events_before": n_before, "n_events_kept": n_kept, **excluded}
    return windows._replace(events=kept, symmetric=True), exclusions
