BEST_VALS = [2, 3, 4]
SECOND_VALS = [1, 2, 3]

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
        all_arms = list(rm.keys())

        # Build believed magnitudes for seen arms only
        believed = {arm: last_seen[arm] for arm in all_arms if arm in last_seen}
        n_seen = len(believed)

        # Rank seen arms by believed magnitude descending,
        # with deterministic tie-break by arm name
        ranked = sorted(believed.items(), key=lambda x: (-x[1], str(x[0])))

        believed_best_mag   = ranked[0][1] if n_seen >= 1 else None
        believed_second_mag = ranked[1][1] if n_seen >= 2 else None

        # Believed rank of chosen arm (None if the chosen arm is unseen)
        believed_rank_of_choice = None
        if choice in believed:
            for rank_idx, (arm, _) in enumerate(ranked):
                if arm == choice:
                    believed_rank_of_choice = ["best", "second", "third"][min(rank_idx, 2)]
                    break

        tied = (
            believed_best_mag is not None and
            believed_second_mag is not None and
            believed_best_mag == believed_second_mag
        )

        results.append({
            "believed_best_mag":    believed_best_mag,
            "believed_second_mag":  believed_second_mag,
            "believed_chosen_rank": believed_rank_of_choice,
            "tied":                 tied,
            "n_seen":               n_seen,
        })

        # Update last_seen AFTER recording
        if choice and choice in rm:
            last_seen[choice] = rm[choice]

    return results
