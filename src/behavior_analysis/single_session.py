"""Pure computations behind the single-session summary figure.

Everything here works on in-memory session dicts and returns plain lists /
arrays, so it can be tested without a display. The drawing lives in
``src.behavior_visualization.plot_single_session``.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

RANK_ORDER = ("best", "second", "third")

# (good_idx, bad_idx, block_idx): trial indices of good reversals, bad reversals and block starts.
Reversals = tuple[list[int], list[int], list[int]]


def event_indices_from_cumulative(
    counter: Sequence[float] | None, N: int | None = None
) -> list[int]:
    """Return trial indices where a cumulative counter increments.

    Args:
        counter (Sequence[float] | None): Cumulative counts, one per trial; ``None`` gives an
            empty list.
        N (int | None): Optional maximum trial count; the counter is truncated to *N*
            elements before processing. Default ``None``.

    Returns:
        list[int]: 0-based trial indices where the counter value increases
        relative to the previous trial, i.e. the index of the first trial
        after each increment.
    """
    if counter is None:
        return []
    c = np.asarray(counter, dtype=float)
    if N is not None:
        c = c[:N]
    if len(c) < 2:
        return []
    d = np.diff(c)
    return list(np.where(d > 0)[0] + 1)


def reversal_event_indices(session_data: Mapping[str, Any], N: int) -> Reversals:
    """Trial indices of good reversals, bad reversals and block starts.

    Good/bad indices come from the cumulative counters and are extended with
    ``reversal_boundary_events`` (``{"kind": "good" | "bad", "index": int}``)
    that the counters did not already mark.

    Args:
        session_data (Mapping[str, Any]): A session dict; optional keys ``"good_reversals"``,
            ``"bad_reversals"``, ``"blocks"`` (cumulative counters per trial) and
            ``"reversal_boundary_events"``.
        N (int): Number of trials to use from the counters.

    Returns:
        Reversals: ``(good_idx, bad_idx, block_idx)``, each a list of trial indices.

    Known limitations (kept so the figure matches the original):
    any boundary-event ``kind`` other than ``"good"`` is treated as a bad
    reversal, and event indices are not checked against *N*, so an event
    beyond the last trial still gets a line, which stretches the x-axis.
    """
    good_idx = event_indices_from_cumulative(session_data.get("good_reversals"), N=N)
    bad_idx = event_indices_from_cumulative(session_data.get("bad_reversals"), N=N)
    for event in session_data.get("reversal_boundary_events", []):
        target = good_idx if event["kind"] == "good" else bad_idx
        if event["index"] not in target:
            target.append(event["index"])
    block_idx = event_indices_from_cumulative(session_data.get("blocks"), N=N)
    return good_idx, bad_idx, block_idx


def _chosen_magnitude(mags_by, choices_by, towers, trial):
    """Magnitude of the first tower flagged on *trial* (NaN if none is)."""
    return next(
        (
            mags_by[tower][trial]
            for tower in towers
            if trial < len(choices_by[tower]) and choices_by[tower][trial] == 1
        ),
        np.nan,
    )


def chosen_magnitudes(
    mags_by: Mapping[str, Sequence[float]],
    choices_by: Mapping[str, Sequence[float]],
    *,
    towers: Sequence[str],
    N: int,
) -> np.ndarray:
    """Reward magnitude of the chosen tower on each of the first *N* trials.

    Trials without a recorded choice are NaN. If several towers are flagged on
    one trial, the first in *towers* order wins.

    Args:
        mags_by (Mapping[str, Sequence[float]]): ``{tower: reward magnitude per trial}``.
        choices_by (Mapping[str, Sequence[float]]): ``{tower: choice flag per trial}``; 1
            marks a choice.
        towers (Sequence[str]): Towers to look at, in priority order.
        N (int): Number of trials.

    Returns:
        np.ndarray: Float array of shape ``(N,)``.

    Only a flag of exactly 1 counts as a choice, whereas
    :func:`choice_counts_by_tower` sums the raw values, so a flag of 2 is
    counted twice there but is not a choice here.
    """
    return np.array(
        [_chosen_magnitude(mags_by, choices_by, towers, trial) for trial in range(N)],
        dtype=float,
    )


def choice_counts_by_tower(
    choices_by: Mapping[str, Sequence[float]], towers: Sequence[str], N: int
) -> list[int]:
    """Number of choices of each tower over the first *N* trials (NaN counts as 0).

    Sums the raw flag values (a 2 counts twice); see :func:`chosen_magnitudes`
    for the stricter ``== 1`` rule used for the dots on the magnitude panel.

    Args:
        choices_by (Mapping[str, Sequence[float]]): ``{tower: choice flag per trial}``.
        towers (Sequence[str]): Towers to count, in output order.
        N (int): Number of leading trials to use.

    Returns:
        list[int]: One count per tower in *towers*.
    """
    return [int(np.nansum(np.asarray(choices_by[t][:N], dtype=float))) for t in towers]


def has_ema_traces(session_data: Mapping[str, Any], ema_keys: Sequence[str]) -> bool:
    """True when any of *ema_keys* holds at least one non-``None`` value.

    This decides whether the figure gets an EMA panel. The panel then plots
    every key that is *present* (even if its value is ``None``), so one key
    with data next to another set to ``None`` passes this check and then
    fails inside matplotlib ("x and y must have same first dimension").

    Args:
        session_data (Mapping[str, Any]): A session dict.
        ema_keys (Sequence[str]): Keys of the EMA traces to look for.

    Returns:
        bool: Whether any key holds a non-``None`` value.

    Written with ``in`` and ``[]`` rather than ``.get`` so that a non-mapping
    *session_data* raises ``TypeError`` (as the original single function did),
    not ``AttributeError``.
    """
    return any(
        k in session_data
        and session_data[k] is not None
        and any(v is not None for v in session_data[k])
        for k in ema_keys
    )


def choice_probability_matrix(
    probabilities_by_tower: Mapping[str, Sequence[float]], towers: Sequence[str], N: int
) -> np.ndarray:
    """Stack per-tower probabilities into an ``(N, len(towers))`` array.

    Args:
        probabilities_by_tower (Mapping[str, Sequence[float]]): ``{tower: probability per
            trial}``; its keys must be exactly *towers*.
        towers (Sequence[str]): Towers, giving the column order.
        N (int): Number of trials in the session.

    Returns:
        np.ndarray: Float array of shape ``(N, len(towers))``.

    Raises:
        ValueError: if the labels differ from *towers*, or the probabilities
            are not finite, within [0, 1], aligned to every trial and
            summing to 1 on each trial.
    """
    if set(probabilities_by_tower) != set(towers):
        raise ValueError("Model probabilities must use the session's physical tower labels.")
    probabilities = np.column_stack([probabilities_by_tower[t] for t in towers])
    if (
        probabilities.shape != (N, len(towers))
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.allclose(probabilities.sum(axis=1), 1)
    ):
        raise ValueError(
            "Model probabilities must be normalized and aligned to every session trial."
        )
    return probabilities


def latest_rank_counts(session_data: Mapping[str, Any]) -> dict | None:
    """The final cumulative ``{rank: count}`` dict, or ``None`` if unavailable.

    A last entry that is not a dict (malformed data) also gives ``None``, which
    the figure shows as "No rank data" instead of raising.

    Args:
        session_data (Mapping[str, Any]): A session dict whose ``"rank_counts"`` is a list of
            cumulative ``{rank: count}`` dicts.

    Returns:
        dict | None: The last entry of ``"rank_counts"``, or ``None``.
    """
    history = session_data.get("rank_counts")
    if not history:
        return None
    last = history[-1]
    return last if isinstance(last, dict) else None


def ordered_rank_counts(rank_counts: Mapping[str, float]) -> tuple[list[str], list[int]]:
    """Rank names and integer counts, with best/second/third first.

    Ranks outside :data:`RANK_ORDER` keep their input order after the standard ones.

    Args:
        rank_counts (Mapping[str, float]): ``{rank: count}``.

    Returns:
        tuple[list[str], list[int]]: ``(rank names, integer counts)``, aligned.
    """
    keys = [k for k in RANK_ORDER if k in rank_counts]
    keys += [k for k in rank_counts if k not in RANK_ORDER]
    return keys, [int(rank_counts[k]) for k in keys]


def percent_of_total(counts: Sequence[float]) -> list[float]:
    """Each count as a percentage of the total (an all-zero total counts as 1).

    Args:
        counts (Sequence[float]): Counts.

    Returns:
        list[float]: Percentages (0 to 100), one per count.
    """
    # An all-zero panel (no choices, or all-zero rank counts) should label its
    # bars "0%" rather than divide by zero.
    total = max(int(np.sum(counts)), 1)
    return [100.0 * c / total for c in counts]
