"""Pure computations behind the single-session summary figure.

Everything here works on in-memory session dicts and returns plain lists /
arrays, so it can be tested without a display. The drawing lives in
``src.behavior_visualization.plot_single_session``.
"""

import numpy as np

RANK_ORDER = ("best", "second", "third")


def event_indices_from_cumulative(counter, N=None):
    """Return trial indices where a cumulative counter increments.

    Args:
        counter: List of cumulative integer counts, one per trial.
        N: Optional maximum trial count; the counter is truncated to *N*
            elements before processing (default: ``None``).

    Returns:
        List of trial indices (1-based) where the counter value increases
        relative to the previous trial.
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


def reversal_event_indices(session_data, N):
    """Trial indices of good reversals, bad reversals and block starts.

    Good/bad indices come from the cumulative counters and are extended with
    ``reversal_boundary_events`` (``{"kind": "good" | "bad", "index": int}``)
    that the counters did not already mark.

    Returns:
        ``(good_idx, bad_idx, block_idx)``, each a list of trial indices.
    """
    good_idx = event_indices_from_cumulative(session_data.get("good_reversals"), N=N)
    bad_idx = event_indices_from_cumulative(session_data.get("bad_reversals"), N=N)
    for event in session_data.get("reversal_boundary_events", []):
        target = good_idx if event["kind"] == "good" else bad_idx
        if event["index"] not in target:
            target.append(event["index"])
    block_idx = event_indices_from_cumulative(session_data.get("blocks"), N=N)
    return good_idx, bad_idx, block_idx


def chosen_magnitudes(mags_by, choices_by, towers, N):
    """Reward magnitude of the chosen tower on each trial.

    Trials without a recorded choice are NaN. If several towers are flagged on
    one trial, the first in *towers* order wins.
    """
    return np.array(
        [
            next(
                (
                    mags_by[tower][i]
                    for tower in towers
                    if i < len(choices_by[tower]) and choices_by[tower][i] == 1
                ),
                np.nan,
            )
            for i in range(N)
        ],
        dtype=float,
    )


def choice_counts_by_tower(choices_by, towers, N):
    """Number of choices of each tower over the first *N* trials (NaN counts as 0)."""
    return [int(np.nansum(np.asarray(choices_by[t][:N], dtype=float))) for t in towers]


def has_ema_traces(session_data, ema_keys):
    """True when any of *ema_keys* holds at least one non-``None`` value.

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


def validate_choice_probabilities(probabilities_by_tower, towers, N):
    """Stack per-tower probabilities into an ``(N, len(towers))`` array.

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


def latest_rank_counts(session_data):
    """The final cumulative ``{rank: count}`` dict, or ``None`` if unavailable."""
    history = session_data.get("rank_counts")
    if not history:
        return None
    last = history[-1]
    return last if isinstance(last, dict) else None


def ordered_rank_counts(rank_counts):
    """Rank names and integer counts, with best/second/third first.

    Ranks outside :data:`RANK_ORDER` keep their input order after the standard ones.
    """
    keys = [k for k in RANK_ORDER if k in rank_counts]
    keys += [k for k in rank_counts if k not in RANK_ORDER]
    return keys, [int(rank_counts[k]) for k in keys]


def percent_of_total(counts):
    """Each count as a percentage of the total (an all-zero total counts as 1)."""
    total = max(int(np.sum(counts)), 1)
    return [100.0 * c / total for c in counts]
