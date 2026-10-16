"""Per-event windows and arm labels behind the best-decrease rank-swap choice-probability plot.

For one session this returns, for every event of ``get_value_change_windows(..., "best",
"decrease", "rank_swap", truncate_at_rank_swap=True)``, the session-local trial indices that
the analysis actually counts (the bins whose label is not ``None``) and which arm each label
(``prev_best``, ``prev_second``, ``third``) names. Everything is read off the analysis's own
events, so what is drawn is what the analysis used.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NamedTuple

from src.behavior_analysis.value_change_windows import (
    ValueChangeEvent,
    get_value_change_windows,
    swap_label_arms,
)
from src.behavior_import.extract_trials import merge_trial_variables

# The window of the cohort pipeline's value-change figures (``run_value_change_analysis``).
PRE = 5
POST = 15
RANK, DIRECTION, RANK_FILTER = "best", "decrease", "rank_swap"


class EventDiagnostic(NamedTuple):
    """One event's window indices and arm labels.

    Attributes:
        number (int): 1-based event number within the session, in trial order.
        trial_idx (int): Session-local trial index of t = 0.
        pre_idx (tuple[int, ...]): Trials of the pre window the analysis counts (labelled bins).
        post_idx (tuple[int, ...]): Trials of the post window it counts, t = 0 included.
        labels_by_trial (dict[int, str]): Label of the choice on each counted trial.
        arms (dict[str, Any]): Arm named by each label, ``prev_best`` / ``prev_second`` /
            ``third``, from the magnitudes at t = -1.
        pre_requested (int): Pre length asked for.
        post_requested (int): Post length asked for (trials after t = 0).
        truncated (bool): Whether the post window was cut at a best/second exchange.
        swap_offset (int | None): t of that exchange, or ``None``.
    """

    number: int
    trial_idx: int
    pre_idx: tuple[int, ...]
    post_idx: tuple[int, ...]
    labels_by_trial: dict[int, str]
    arms: dict[str, Any]
    pre_requested: int
    post_requested: int
    truncated: bool
    swap_offset: int | None

    @property
    def pre_used(self) -> int:
        """Number of pre-window trials counted."""
        return len(self.pre_idx)

    @property
    def post_used(self) -> int:
        """Number of post-window trials counted after t = 0."""
        return sum(i > self.trial_idx for i in self.post_idx)


def contiguous_runs(indices: Iterable[int]) -> list[tuple[int, int]]:
    """``(start, length)`` of each run of consecutive integers in sorted *indices*."""
    runs: list[tuple[int, int]] = []
    for i in indices:
        if runs and i == runs[-1][0] + runs[-1][1]:
            runs[-1] = (runs[-1][0], runs[-1][1] + 1)
        else:
            runs.append((i, 1))
    return runs


def diagnose_event(
    event: ValueChangeEvent, number: int, magnitudes: Sequence[Mapping], pre: int, post: int
) -> EventDiagnostic:
    """Window indices and arm labels of one analysis event.

    Args:
        event (ValueChangeEvent): An event with ``pre`` / ``post`` of length *pre* and
            *post* + 1 (t = -pre .. -1 and t = 0 .. post).
        number (int): Event number to record.
        magnitudes (Sequence[Mapping]): ``{arm: magnitude}`` per trial of the event's session.
        pre (int): Pre length of the event's windows.
        post (int): Post length of the event's windows.

    Returns:
        EventDiagnostic: Indices are session-local, like ``event.trial_idx``.
    """
    t0 = event.trial_idx
    pre_labels = {t0 - pre + p: label for p, label in enumerate(event.pre) if label is not None}
    post_labels = {t0 + p: label for p, label in enumerate(event.post) if label is not None}
    return EventDiagnostic(
        number=number,
        trial_idx=t0,
        pre_idx=tuple(pre_labels),
        post_idx=tuple(post_labels),
        labels_by_trial={**pre_labels, **post_labels},
        arms=swap_label_arms(magnitudes[t0 - 1]),
        pre_requested=pre,
        post_requested=post,
        truncated=event.truncated,
        swap_offset=event.rank_swap_offset if event.truncated else None,
    )


def check_series_length(session: Mapping[str, Any], n_trials: int, where: str) -> None:
    """Raise ValueError unless the per-tower series the plot draws have *n_trials* trials."""
    for key in ("reward_magnitudes_by_tower", "choices_by_tower"):
        for tower, series in session[key].items():
            if len(series) != n_trials:
                raise ValueError(
                    f"{where}: {key}[{tower!r}] has {len(series)} trials but the trial "
                    f"variables have {n_trials}; event trial indices would not match the plot"
                )


def diagnose_session(
    subject: str, session_key: str, session: Mapping[str, Any], *, pre: int = PRE, post: int = POST
) -> list[EventDiagnostic]:
    """The diagnostics of every best-decrease rank-swap event of one session.

    Args:
        subject (str): Subject id.
        session_key (str): Session key.
        session (Mapping[str, Any]): An imported session with ``trial_variables`` and the
            per-tower series ``reward_magnitudes_by_tower`` and ``choices_by_tower``.
        pre (int): Pre length, as in the value-change figures. Default ``PRE``.
        post (int): Post length. Default ``POST``.

    Returns:
        list[EventDiagnostic]: One per event, in trial order; empty for a session without events.

    Raises:
        ValueError: If the plotted series and the trial variables differ in length, or as
            :func:`get_value_change_windows`.
    """
    windows = get_value_change_windows(
        {subject: {session_key: session}}, RANK, DIRECTION, RANK_FILTER,
        pre=pre, post=post, truncate_at_rank_swap=True,
    )  # fmt: skip
    return diagnose_events(
        windows.events.get(subject, []), session, pre, post, where=f"{subject} {session_key}"
    )


def diagnose_events(
    events: Sequence[ValueChangeEvent],
    session: Mapping[str, Any],
    pre: int,
    post: int,
    *,
    where: str,
) -> list[EventDiagnostic]:
    """:func:`diagnose_event` for the events of one session, numbered from 1 in trial order.

    Args:
        events (Sequence[ValueChangeEvent]): The session's events, from one call of
            :func:`get_value_change_windows` with the same *pre* and *post*.
        session (Mapping[str, Any]): The session they came from.
        pre (int): Pre length of the windows.
        post (int): Post length of the windows.
        where (str): Subject and session, named in errors.

    Raises:
        ValueError: If the plotted series and the trial variables differ in length.
    """
    magnitudes = merge_trial_variables(session["trial_variables"])["reward_magnitudes"]
    check_series_length(session, len(magnitudes), where)
    ordered = sorted(events, key=lambda e: e.trial_idx)
    return [diagnose_event(e, n, magnitudes, pre, post) for n, e in enumerate(ordered, start=1)]


def nearest_event_labels(diagnostics: Sequence[EventDiagnostic]) -> dict[int, str]:
    """Label of each counted trial from the event whose t = 0 is closest (the earlier on a tie).

    Windows of nearby events can overlap and name a trial's choice differently; one trial gets
    one colour on the session panel, and the strip below shows every event's own labels.
    """
    best: dict[int, tuple[int, str]] = {}
    for diagnostic in diagnostics:
        for trial, label in diagnostic.labels_by_trial.items():
            distance = abs(trial - diagnostic.trial_idx)
            if trial not in best or distance < best[trial][0]:
                best[trial] = (distance, label)
    return {trial: label for trial, (_, label) in best.items()}
