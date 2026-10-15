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


TIE_BEST_SECOND = "best_second"
TIE_SECOND_THIRD = "second_third"
TIE_ALL_THREE = "all_three"
TIE_CATEGORIES = (TIE_BEST_SECOND, TIE_SECOND_THIRD, TIE_ALL_THREE)


def tie_category(magnitudes) -> str | None:
    """Which of the top three values tie: a ``TIE_*`` name, or ``None`` if none do.

    ``all_three`` when the best three are equal, else ``best_second`` or ``second_third``
    when those two adjacent values are equal. Ties below the third value are ignored.
    With two values only a best/second tie is possible; fewer than two give ``None``.

    Args:
        magnitudes (Iterable[float]): Values of the ranked arms.

    Returns:
        str | None: The tie category.
    """
    top = sorted(magnitudes, reverse=True)[:3]
    if len(top) == 3 and top[0] == top[1] == top[2]:
        return TIE_ALL_THREE
    if len(top) >= 2 and top[0] == top[1]:
        return TIE_BEST_SECOND
    if len(top) == 3 and top[1] == top[2]:
        return TIE_SECOND_THIRD
    return None


def rank_weights(chosen_magnitude: float, magnitudes, n_ranks: int = 3) -> tuple[float, ...]:
    """Share of each of the first *n_ranks* rank positions held by an arm of *chosen_magnitude*.

    *magnitudes* are the values of all ranked arms, the chosen arm's included. Arms with equal
    magnitude occupy their rank positions together, so each gets an equal share of every
    position the group covers (a three-way tie gives each arm 1/3 of best, second and third).
    The shares depend only on the magnitudes, never on arm names.

    Args:
        chosen_magnitude (float): Magnitude of the arm that was chosen.
        magnitudes (Iterable[float]): Magnitudes of every ranked arm.
        n_ranks (int): Number of rank positions to report.

    Returns:
        tuple[float, ...]: One share in [0, 1] per rank position, best first.
    """
    values = list(magnitudes)
    above = sum(v > chosen_magnitude for v in values)
    tied = sum(v == chosen_magnitude for v in values)
    # the tied arms hold rank positions above .. above + tied - 1
    return tuple(1.0 / tied if above <= pos < above + tied else 0.0 for pos in range(n_ranks))


def rank_position_shares(chosen_magnitude: float, magnitudes) -> tuple[float, float]:
    """``(best_share, second_share)`` of an arm of *chosen_magnitude* (see :func:`rank_weights`)."""
    best, second = rank_weights(chosen_magnitude, magnitudes, n_ranks=2)
    return best, second


def _chosen_shares(choice, believed: dict) -> tuple[float | None, float | None]:
    """``(best_share, second_share)`` of the chosen arm, ``(None, None)`` if it is unseen."""
    if choice not in believed:
        return None, None
    return rank_position_shares(believed[choice], believed.values())


def _trial_belief(choice, believed: dict) -> dict:
    """The believed best/second magnitudes, chosen-arm rank and rank shares for one trial."""
    ranked = _rank_arms(believed)
    n_seen = len(believed)
    best_mag = ranked[0][1] if n_seen >= 1 else None
    second_mag = ranked[1][1] if n_seen >= 2 else None
    # an arm that has never been seen has no believed rank
    rank_of_choice = _rank_label(choice, ranked) if choice in believed else None
    tied = best_mag is not None and second_mag is not None and best_mag == second_mag
    best_share, second_share = _chosen_shares(choice, believed)
    return {
        "believed_best_mag": best_mag,
        "believed_second_mag": second_mag,
        "believed_chosen_rank": rank_of_choice,
        "tied": tied,
        "n_seen": n_seen,
        "believed_best_share": best_share,
        "believed_second_share": second_share,
        "believed_tie": tie_category(believed.values()),
    }


def compute_believed_value(choices, reward_magnitudes):
    """
    For each trial i, infer:
      - The believed magnitude for every arm (last observed value before trial i).
      - The believed rank of the chosen arm, derived by re-ranking only the
        *seen* arms by their believed magnitudes. Unseen arms are excluded from
        the ranking entirely.
      - The believed best and second-best magnitudes (by believed rank order).

    Ties in believed magnitude are flagged via the 'tied' field (best and second equal).
    'believed_chosen_rank' breaks ties by arm name, so it depends on arm labels when
    magnitudes tie; use 'believed_best_share' / 'believed_second_share' for anything that
    must not: they split each rank position evenly among the arms tied for it.

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
        believed_best_share : float | None  (share of "best" held by the chosen arm;
                                             None if the chosen arm is unseen)
        believed_second_share: float | None (share of "second", likewise)
        believed_tie        : str | None    (tie category among the believed values, see
                                             tie_category)
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
