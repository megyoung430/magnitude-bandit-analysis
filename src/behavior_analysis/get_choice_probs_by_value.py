from dataclasses import dataclass

import numpy as np
import pandas as pd

from .get_believed_value import compute_believed_value

BEST_VALS = [2, 3, 4]
SECOND_VALS = [1, 2, 3]


def init_matrix(fill=np.nan):
    return pd.DataFrame(
        fill,
        index=pd.Index(BEST_VALS, name="best_value"),
        columns=pd.Index(SECOND_VALS, name="second_value"),
    )


@dataclass(frozen=True)
class _MouseCounts:
    """Per-cell trial counts (best value x second value) for one mouse.

    ``denom`` counts the trials in each cell; ``num_best`` and ``num_second`` count how
    much of that was a best / second choice (fractional when believed values tie).
    """

    denom: pd.DataFrame
    num_best: pd.DataFrame
    num_second: pd.DataFrame

    @classmethod
    def empty(cls):
        return cls(init_matrix(fill=0.0), init_matrix(fill=0.0), init_matrix(fill=0.0))


def _in_value_grid(b, s):
    """Whether ``(b, s)`` is one of the matrix cells."""
    return b in BEST_VALS and s in SECOND_VALS


def _true_contribution(reward_magnitudes, chosen_rank):
    """``(b, s, best_share, second_share)`` for one trial binned by true magnitudes.

    Returns ``None`` when the trial does not belong in any cell: fewer than two arms,
    an unrecognised chosen rank, or magnitudes outside the grid.
    """
    sorted_arms = sorted(reward_magnitudes.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_arms) < 2:
        return None
    b, s = sorted_arms[0][1], sorted_arms[1][1]
    if chosen_rank not in ("best", "second", "third"):
        return None
    if not _in_value_grid(b, s):
        return None
    return b, s, float(chosen_rank == "best"), float(chosen_rank == "second")


def _believed_contribution(trial):
    """``(b, s, best_share, second_share)`` for one trial from ``compute_believed_value``.

    Returns ``None`` when fewer than two arms have been seen, the chosen arm has no
    believed rank, or the believed magnitudes are outside the grid. A tie between the
    best and second believed arms makes their rank labels arbitrary, so a choice of
    either tied arm contributes half to each numerator; a third-ranked choice
    contributes to the denominator only.
    """
    b, s = trial["believed_best_mag"], trial["believed_second_mag"]
    chosen_rank = trial["believed_chosen_rank"]
    if trial["n_seen"] < 2:
        return None
    if chosen_rank is None:
        return None
    if not _in_value_grid(b, s):
        return None
    if trial["tied"]:
        share = 0.5 if chosen_rank in ("best", "second") else 0.0
        return b, s, share, share
    return b, s, float(chosen_rank == "best"), float(chosen_rank == "second")


def _session_trial_lists(sess):
    """``(reward_magnitudes, chosen_ranks, choices)`` cut to a common length, or ``None``.

    ``None`` when the session has no trial variables or no trial common to all three lists.
    """
    tv_list = sess.get("trial_variables", [])
    if not tv_list:
        return None
    tv = tv_list[0]
    reward_magnitudes = tv.get("reward_magnitudes", [])
    chosen_ranks = tv.get("chosen_rank", [])
    choices = tv.get("choice", [])
    # Known issue: the lists are truncated to the shortest of the three even when
    # ``use_believed`` is set, where ``chosen_rank`` is otherwise unused, so a short or
    # missing ``chosen_rank`` list silently drops trials from the believed analysis.
    n = min(len(reward_magnitudes), len(chosen_ranks), len(choices))
    if n == 0:
        return None
    return reward_magnitudes[:n], chosen_ranks[:n], choices[:n]


def _session_contributions(sess, *, use_believed):
    """Yield ``(b, s, best_share, second_share)`` for every countable trial of a session."""
    trial_lists = _session_trial_lists(sess)
    if trial_lists is None:
        return
    reward_magnitudes, chosen_ranks, choices = trial_lists
    if use_believed:
        believed_trials = compute_believed_value(choices, reward_magnitudes)
        contributions = (_believed_contribution(trial) for trial in believed_trials)
    else:
        contributions = map(_true_contribution, reward_magnitudes, chosen_ranks)
    for contribution in contributions:
        if contribution is not None:
            yield contribution


def _add_contribution(counts, b, s, best_share, second_share):
    """Add one trial to the cell ``(b, s)``; zero shares leave their numerator untouched."""
    counts.denom.loc[b, s] += 1.0
    if best_share:
        counts.num_best.loc[b, s] += best_share
    if second_share:
        counts.num_second.loc[b, s] += second_share


def _count_mouse_sessions(counts, sessions_dict, *, use_believed):
    """Accumulate every session of one mouse into *counts* (in place)."""
    for sess in sessions_dict.values():
        for contribution in _session_contributions(sess, use_believed=use_believed):
            _add_contribution(counts, *contribution)


def _count_mice(subject_trials_by_value_problem, *, use_believed):
    """``{subject_id: _MouseCounts}``, pooling a mouse's sessions across all problems."""
    counts_by_mouse = {}
    for subjects_dict in subject_trials_by_value_problem.values():
        for subject_id, sessions_dict in subjects_dict.items():
            if subject_id not in counts_by_mouse:
                counts_by_mouse[subject_id] = _MouseCounts.empty()
            counts = counts_by_mouse[subject_id]
            _count_mouse_sessions(counts, sessions_dict, use_believed=use_believed)
    return counts_by_mouse


def _probabilities(numerator, denom):
    """Numerator / denominator per cell; NaN where a cell has no trials."""
    return numerator.divide(denom.where(denom > 0))


def _nanmean_matrices(prob_dict):
    """Cell-wise nanmean across mice, so every mouse contributes equally per cell."""
    if not prob_dict:
        return init_matrix(fill=np.nan)
    stack = np.stack([df.to_numpy(dtype=float) for df in prob_dict.values()], axis=0)
    ref = next(iter(prob_dict.values()))
    return pd.DataFrame(np.nanmean(stack, axis=0), index=ref.index, columns=ref.columns)


def accumulate_choice_probs(subject_trials_by_value_problem, use_believed=False):
    """
    subject_trials_by_value_problem[problem][subject_id][session_id] = session_data

    session_data["trial_variables"] is a list with a single element: a dict of
    parallel lists, one value per trial. Keys include:
      - "reward_magnitudes" : list[dict]  — arm->magnitude per trial
      - "chosen_rank"       : list[str]   — "best"/"second"/"third" per trial
      - "choice"            : list[str]   — chosen arm label per trial

    Parameters
    ----------
    use_believed : bool
        False (default) — bin by true reward magnitudes and true chosen rank.
        True            — bin by believed magnitudes (last observed value per arm
                          before this trial), with arms re-ranked by believed
                          value to determine best/second and the chosen rank.
                          Trials where fewer than 2 arms have been visited, or
                          where the chosen arm has no believed value, are skipped.
                          Tied trials split contributions evenly across best/second.

    Returns
    -------
    overall_best_prob_df   : 3x3 DataFrame  (nanmean across mice)
    overall_second_prob_df : 3x3 DataFrame
    per_mouse_best_prob    : dict[subject_id -> 3x3 DataFrame]
    per_mouse_second_prob  : dict[subject_id -> 3x3 DataFrame]
    """
    counts_by_mouse = _count_mice(subject_trials_by_value_problem, use_believed=use_believed)
    per_mouse_best_prob = {
        sid: _probabilities(c.num_best, c.denom) for sid, c in counts_by_mouse.items()
    }
    per_mouse_second_prob = {
        sid: _probabilities(c.num_second, c.denom) for sid, c in counts_by_mouse.items()
    }
    overall_best_prob = _nanmean_matrices(per_mouse_best_prob)
    overall_second_prob = _nanmean_matrices(per_mouse_second_prob)
    return overall_best_prob, overall_second_prob, per_mouse_best_prob, per_mouse_second_prob
