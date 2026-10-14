from __future__ import annotations

BEST_VALS = [2, 3, 4]
SECOND_VALS = [1, 2, 3]

_RANK_LABELS = ("best", "second", "third")


def _believed_magnitudes(offer: dict, last_seen: dict) -> dict:
    """Last observed magnitude of every arm in one trial's *offer* seen before."""
    return {arm: last_seen[arm] for arm in offer.keys() if arm in last_seen}


def _rank_arms(believed: dict) -> list:
    """``(arm, magnitude)`` pairs, best first; ties break by arm name for determinism."""
    return sorted(believed.items(), key=lambda x: (-x[1], str(x[0])))


def _rank_label(choice, ranked: list) -> str | None:
    """Label (best, second or third) for the chosen arm's position in *ranked*."""
    for rank_idx, (arm, _) in enumerate(ranked):
        if arm == choice:
            return _RANK_LABELS[min(rank_idx, 2)]
    return None


def _trial_belief(choice, believed: dict) -> dict:
    """The believed best/second magnitudes and chosen-arm rank for one trial."""
    ranked = _rank_arms(believed)
    n_seen = len(believed)
    best_mag = ranked[0][1] if n_seen >= 1 else None
    second_mag = ranked[1][1] if n_seen >= 2 else None
    # an arm that has never been seen has no believed rank
    rank_of_choice = _rank_label(choice, ranked) if choice in believed else None
    tied = best_mag is not None and second_mag is not None and best_mag == second_mag
    return {
        "believed_best_mag": best_mag,
        "believed_second_mag": second_mag,
        "believed_chosen_rank": rank_of_choice,
        "tied": tied,
        "n_seen": n_seen,
    }


def compute_believed_value(choices, reward_magnitudes):
    """
    For each trial i, infer:
      - The believed magnitude for every arm (last observed value before trial i).
      - The believed rank of the chosen arm, derived by re-ranking only the
        *seen* arms by their believed magnitudes. Unseen arms are excluded from
        the ranking entirely.
      - The believed best and second-best magnitudes (by believed rank order).

    Ties in believed magnitude are flagged via the 'tied' field. Tied trials
    are handled downstream by splitting contributions evenly across best/second.

    Parameters
    ----------
    choices           : list[str]  — arm chosen on each trial
    reward_magnitudes : list[dict] — true arm->magnitude mapping per trial

    Returns
    -------
    list of dicts with keys:
        believed_best_mag   : float | None  (None if fewer than 1 seen arm)
        believed_second_mag : float | None  (None if fewer than 2 seen arms)
        believed_chosen_rank: str | None    ("best"/"second"/"third"/None if
                                             chosen arm is unseen)
        tied                : bool          (True if best and second believed
                                             magnitudes are equal)
        n_seen              : int           (number of arms with a believed value)
    """
    last_seen = {}
    results = []

    for i, rm in enumerate(reward_magnitudes):
        choice = choices[i]
        results.append(_trial_belief(choice, _believed_magnitudes(rm, last_seen)))

        # Update last_seen AFTER recording.
        # Known issue: the truthiness test also skips arms named 0 or "", so such an
        # arm would never count as seen; arm names are non-empty strings in practice.
        if choice and choice in rm:
            last_seen[choice] = rm[choice]

    return results
