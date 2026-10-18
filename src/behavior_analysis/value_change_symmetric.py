"""Symmetric, complete-window selection of value-change events.

The regular value-change windows keep every event and cut a window where it meets a block or
third-arm change, a session edge or (before the event) a change of the best arm, so the curves
at large ``|t|`` average fewer, different events than those near t = 0. The symmetric version
instead keeps only events whose whole ``-N .. +N`` window is intact:

- all ``2N + 1`` trials exist in the event's block with the event's third arm: no block change,
  third-arm change, tied or missing magnitudes, or session edge. A block that starts exactly at
  t = -N, or ends exactly at t = +N, keeps the event;
- the best arm does not change in the pre window: every trial of t = -N .. -1 has the best arm
  of t = -1 (the pre-window rule of the regular windows; an event whose pre window that rule
  would cut is excluded, not truncated). A best arm that starts exactly at t = -N keeps the
  event. A best/second tie in the pre window is a different state from a determined best arm,
  so it counts as a best-arm change (and only a tie at t = -1 continues a run of ties);
- every trial has a choice. A trial that exists but has no choice (a missed trial) leaves a
  hole in the window, so the event is excluded and counted as ``other``.

A best-arm change in the post window is allowed: it is the outcome the figure looks at, and the
regular windows allow it too. Rank-swap truncation is not applied, since blanking post trials
would break the symmetry.

The whole window is scanned for every cause. Each excluded event is counted once in
``n_excluded_*``, under the first of: block or third-arm change, session edge, best-arm change,
other; and in ``n_with_*`` under every cause it has, so a cause hidden behind a higher-priority
one on the same event is still visible.
"""

from collections import Counter

from src.behavior_analysis.value_change_windows import (
    CAUSE_BEST_ARM,
    CAUSE_BLOCK_OR_THIRD_ARM,
    CAUSE_NO_CHOICE,
    CAUSE_SESSION_EDGE,
    ValueChangeEvent,
    ValueChangeWindows,
)

# Cause -> (exclusive count key, any-cause count key); the order is the priority of the
# exclusive count when an event has several causes.
_CAUSE_KEYS = {
    CAUSE_BLOCK_OR_THIRD_ARM: (
        "n_excluded_block_or_third_arm_change",
        "n_with_block_or_third_arm_change",
    ),
    CAUSE_SESSION_EDGE: ("n_excluded_session_edge", "n_with_session_edge"),
    CAUSE_BEST_ARM: ("n_excluded_best_arm_change", "n_with_best_arm_change"),
    CAUSE_NO_CHOICE: ("n_excluded_other", "n_with_other"),
}
EXCLUSION_KEYS = tuple(exclusive for exclusive, _ in _CAUSE_KEYS.values())
ANY_CAUSE_KEYS = tuple(any_cause for _, any_cause in _CAUSE_KEYS.values())


def _causes(event: ValueChangeEvent) -> frozenset[str]:
    """The reasons *event*'s window is incomplete.

    Raises:
        ValueError: If the event does not record them.
    """
    if event.incomplete_causes is None:
        raise ValueError(
            f"{event.subject} {event.session} trial {event.trial_idx}: the event does not record "
            "why its window is incomplete (incomplete_causes); build it with "
            "get_value_change_windows"
        )
    return event.incomplete_causes


def _exclusive_key(causes: frozenset[str]) -> str:
    """The ``EXCLUSION_KEYS`` entry of the highest-priority cause in *causes* (not empty)."""
    return next(keys[0] for cause, keys in _CAUSE_KEYS.items() if cause in causes)


def select_symmetric_windows(windows: ValueChangeWindows) -> tuple[ValueChangeWindows, dict]:
    """Keep the events whose ``-N .. +N`` window is complete, and count why the others are not.

    Args:
        windows (ValueChangeWindows): Untruncated windows built with ``pre == post == N``.

    Returns:
        tuple: ``(kept, exclusions)``. *kept* has the same configuration and only the complete
        events (subjects without any are dropped), with ``symmetric`` True. Its ``counts`` are
        those of *windows* except ``n_events``, now the number kept, and the extra
        ``n_events_before_exclusion``, the number found. *exclusions* is
        ``{"n_events_before": ..., "n_events_kept": ..., <each of EXCLUSION_KEYS>: ...,
        <each of ANY_CAUSE_KEYS>: ...}``. The ``n_excluded_*`` counts count each event once and
        add up to ``n_events_before - n_events_kept``; the ``n_with_*`` counts are the events
        having each cause, so they can add up to more.

    Raises:
        ValueError: If the windows are truncated or already symmetric, ``pre != post``, or an
            event does not record why its window is incomplete.
    """
    if windows.truncate_at_rank_swap:
        raise ValueError("symmetric selection needs untruncated windows")
    if windows.symmetric:
        raise ValueError("windows are already symmetric")
    if windows.pre != windows.post:
        raise ValueError(
            f"symmetric selection needs pre == post, got {windows.pre}, {windows.post}"
        )
    tally: Counter = Counter({key: 0 for key in EXCLUSION_KEYS + ANY_CAUSE_KEYS})
    kept: dict[str, list[ValueChangeEvent]] = {}
    n_before = 0
    for subject, events in windows.events.items():
        for event in events:
            n_before += 1
            causes = _causes(event)
            if not causes:
                kept.setdefault(subject, []).append(event)
                continue
            tally[_exclusive_key(causes)] += 1
            tally.update(_CAUSE_KEYS[cause][1] for cause in causes)
    n_kept = sum(map(len, kept.values()))
    exclusions = {"n_events_before": n_before, "n_events_kept": n_kept, **tally}
    counts = {**windows.counts, "n_events_before_exclusion": n_before, "n_events": n_kept}
    return windows._replace(events=kept, counts=counts, symmetric=True), exclusions
