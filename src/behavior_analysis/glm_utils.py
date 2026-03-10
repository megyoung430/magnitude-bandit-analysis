"""Utilities for building GLM (multinomial logistic regression) inputs.

Functions here transform raw trial data (choices and rewards per tower) into
design matrices suitable for multinomial logistic regression.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def make_arm_mapping(tower_keys: Any) -> dict[str, str]:
    """Create a deterministic tower-key -> arm-name mapping.

    Sorts tower keys lexicographically and assigns ``"Arm1"``, ``"Arm2"``,
    ``"Arm3"``, ...

    Args:
        tower_keys: An iterable of tower identifiers.

    Returns:
        ``{tower_key: "ArmN"}`` for each key.
    """
    keys_sorted = sorted(list(tower_keys))
    return {k: f"Arm{i + 1}" for i, k in enumerate(keys_sorted)}


def remap_tower_dict_to_arms(d: dict, mapping: dict[str, str]) -> dict[str, Any]:
    """Remap a ``{tower: data}`` dict to ``{ArmN: data}`` using a mapping.

    Args:
        d: Dict keyed by tower identifiers.
        mapping: ``{tower: arm_name}`` as produced by :func:`make_arm_mapping`.

    Returns:
        New dict keyed by arm names; towers absent from *mapping* are skipped.
    """
    return {mapping[tower]: arr for tower, arr in d.items() if tower in mapping}


def get_chosen_reward_per_trial(
    choices_by_tower: dict,
    reward_magnitudes_by_tower: dict,
) -> list[float]:
    """Extract the reward magnitude of the chosen tower at each trial.

    Exactly one tower must be chosen (``choices_by_tower[tower][t] == 1``) per
    trial.

    Args:
        choices_by_tower: ``{tower: list[0|1]}`` - one-hot choice indicator.
        reward_magnitudes_by_tower: ``{tower: list[float]}`` - reward magnitudes.

    Returns:
        List of reward magnitudes length ``n_trials``.

    Raises:
        KeyError: If *choices_by_tower* is missing a tower present in
            *reward_magnitudes_by_tower*.
        ValueError: If a trial does not have exactly one chosen tower.
    """
    towers = list(reward_magnitudes_by_tower.keys())
    if not towers:
        return []

    n_trials = min(len(reward_magnitudes_by_tower[t]) for t in towers)

    missing = [t for t in towers if t not in choices_by_tower]
    if missing:
        raise KeyError(f"choices_by_tower missing towers: {missing}")

    n_trials = min(n_trials, *(len(choices_by_tower[t]) for t in towers))

    rewards: list[float] = []
    for i in range(n_trials):
        chosen = [t for t in towers if choices_by_tower[t][i] == 1]
        if len(chosen) != 1:
            raise ValueError(
                f"Trial {i}: expected exactly one chosen tower, got {chosen}"
            )
        rewards.append(reward_magnitudes_by_tower[chosen[0]][i])

    return rewards

def format_glm_input(
    data: dict,
    n_lags: int = 4,
    drop_redundant_cols: bool = True,
) -> tuple[dict, dict, list[str]]:
    """Build GLM design matrices and true-choice arrays for all subjects.

    Constructs a regressor matrix per subject containing a bias term,
    choice history (n_lags x 3 arms), reward history (n_lags), and
    choice x reward interaction terms (n_lags x 3 arms).

    Arms are assigned lexicographically by tower key (Arm1, Arm2, Arm3).
    For rank-ordered arms use :func:`format_glm_input_by_rank`.

    Args:
        data: Raw subjects data as returned by ``import_data()``.
        n_lags: Number of history lags to include (default: 4).
        drop_redundant_cols: If ``True``, removes the ``Arm3`` columns for
            choice and interaction terms to make the design matrix full rank
            (default: ``True``).

    Returns:
        A three-tuple ``(regressors_across_subjects, true_choices_across_subjects,
        col_labels)`` where:

        - *regressors_across_subjects* is ``{subject: ndarray(n_trials, n_cols)}``,
        - *true_choices_across_subjects* is ``{subject: ndarray(n_trials, 3)}``,
        - *col_labels* is a list of column label strings.
    """
    from src.behavior_analysis.get_variables_across_sessions import (
        get_vars_across_all_sessions,
    )

    merged_data, _ = get_vars_across_all_sessions(data)

    # Build full column labels
    _full_labels = (
        ["Bias"] +
        [f"Choice Arm{a} t-{l}" for l in range(1, n_lags + 1) for a in range(1, 4)] +
        [f"Reward t-{l}"        for l in range(1, n_lags + 1)] +
        [f"Chx Rew Arm{a} t-{l}" for l in range(1, n_lags + 1) for a in range(1, 4)]
    )
    input_dimension = len(_full_labels)

    # Drop Arm3 columns by label matching — robust to any n_lags
    drop_set   = {i for i, l in enumerate(_full_labels) if "Arm3" in l}
    keep_cols  = [i for i in range(input_dimension) if i not in drop_set]
    col_labels = [l for i, l in enumerate(_full_labels) if i not in drop_set]

    regressors_across_subjects: dict = {}
    true_choices_across_subjects: dict = {}

    for subject, subject_data in merged_data.items():
        choices_by_tower = subject_data["choices_by_tower"]
        chosen_rewards = get_chosen_reward_per_trial(
            choices_by_tower,
            subject_data["reward_magnitudes_by_tower"],
        )

        tower_to_arm   = make_arm_mapping(choices_by_tower.keys())
        choices_by_arm = remap_tower_dict_to_arms(choices_by_tower, tower_to_arm)

        num_trials   = len(subject_data["trial"])
        regressors   = np.zeros((num_trials, input_dimension))
        true_choices = np.zeros((num_trials, len(choices_by_tower)))

        for t in range(1, num_trials):
            true_choices[t] = [
                choices_by_arm["Arm1"][t],
                choices_by_arm["Arm2"][t],
                choices_by_arm["Arm3"][t],
            ]
            regressors[t, 0] = 1  # bias

            for lag in range(1, n_lags + 1):
                if t < lag:
                    continue
                tk = t - lag

                # choice columns: 1 + (lag-1)*3 .. 1 + (lag-1)*3 + 2
                base_c = 1 + (lag - 1) * 3
                regressors[t, base_c]     = choices_by_arm["Arm1"][tk]
                regressors[t, base_c + 1] = choices_by_arm["Arm2"][tk]
                regressors[t, base_c + 2] = choices_by_arm["Arm3"][tk]

                # reward column: 1 + n_lags*3 + (lag-1)
                base_r = 1 + n_lags * 3 + (lag - 1)
                regressors[t, base_r] = chosen_rewards[tk]

                # interaction columns: 1 + n_lags*3 + n_lags + (lag-1)*3 ..+2
                base_i = 1 + n_lags * 3 + n_lags + (lag - 1) * 3
                regressors[t, base_i]     = choices_by_arm["Arm1"][tk] * chosen_rewards[tk]
                regressors[t, base_i + 1] = choices_by_arm["Arm2"][tk] * chosen_rewards[tk]
                regressors[t, base_i + 2] = choices_by_arm["Arm3"][tk] * chosen_rewards[tk]

        # Drop first trial (no history)
        regressors   = regressors[1:]
        true_choices = true_choices[1:]

        if drop_redundant_cols:
            regressors = regressors[:, keep_cols]

        regressors_across_subjects[subject]   = regressors
        true_choices_across_subjects[subject] = true_choices

    return regressors_across_subjects, true_choices_across_subjects, col_labels

# ---------------------------------------------------------------------------
# Rank-based GLM
# ---------------------------------------------------------------------------

def _get_true_rank_at_trial(reward_magnitudes_list: list[dict]) -> list[dict]:
    """
    For each trial, return a dict mapping tower -> rank label
    ("best", "second", "third") based on the true reward magnitudes
    at that trial.

    Ties are broken deterministically by tower key (lexicographic).
    """
    rank_names = ["best", "second", "third"]
    result = []
    for rm in reward_magnitudes_list:
        sorted_towers = sorted(rm.items(), key=lambda kv: (-kv[1], str(kv[0])))
        tower_to_rank = {tower: rank_names[min(i, 2)]
                         for i, (tower, _) in enumerate(sorted_towers)}
        result.append(tower_to_rank)
    return result


def _get_believed_rank_at_trial(
    choices: list,
    reward_magnitudes_list: list[dict],
) -> list[dict | None]:
    """
    For each trial, return a dict mapping tower -> believed rank label
    ("best", "second", "third") based on last-seen magnitudes, or None
    if any arm has never been visited yet.

    The believed rank at trial t reflects what the mouse believed
    BEFORE making its choice on trial t (i.e. last_seen is updated
    AFTER recording the rank for trial t).
    """
    rank_names = ["best", "second", "third"]
    last_seen  = {}
    result     = []

    for i, rm in enumerate(reward_magnitudes_list):
        choice = choices[i]
        arms   = list(rm.keys())

        if all(arm in last_seen for arm in arms):
            sorted_towers = sorted(last_seen.items(),
                                   key=lambda kv: (-kv[1], str(kv[0])))
            tower_to_rank = {tower: rank_names[min(i, 2)]
                             for i, (tower, _) in enumerate(sorted_towers)}
            result.append(tower_to_rank)
        else:
            result.append(None)

        # update AFTER recording
        if choice and choice in rm:
            last_seen[choice] = rm[choice]

    return result


def format_glm_input_by_rank(
    data: dict,
    use_believed: bool = False,
    n_lags: int = 4,
    drop_redundant_cols: bool = True,
) -> tuple[dict, dict, list[str]]:
    """Build rank-ordered GLM design matrices for all subjects.

    Like :func:`format_glm_input` but arms are ordered by rank (best,
    second, third) rather than lexicographically by tower key.

    At each lag t-k, the rank of each arm is determined at trial t-k
    (i.e. what the mouse knew at that moment). With ``use_believed=True``
    the rank is based on last-seen magnitudes (believed values); with
    ``use_believed=False`` the true reward magnitudes are used.

    The reference category (dropped arm) is always "third", so all
    coefficients are log-odds of choosing best (or second) vs third.

    Design matrix columns (before dropping, 4 lags):
        [Bias,
         Chose-Best t-1, Chose-Second t-1, Chose-Third t-1,
         ...repeated for lags 2-4...,
         Reward t-1, Reward t-2, Reward t-3, Reward t-4,
         Best*Reward t-1, Second*Reward t-1, Third*Reward t-1,
         ...repeated for lags 2-4...]

    After dropping the redundant "Chose-Third" and "Third*Reward" columns
    (which are linearly dependent on the other choice columns), the design
    matrix has 1 + 2*n_lags + n_lags + 2*n_lags = 1 + 5*n_lags columns.

    Args:
        data:                Raw subjects data.
        use_believed:        If True, use believed rank; if False, use true rank.
        n_lags:              Number of history lags (default 4).
        drop_redundant_cols: Drop "Chose-Third" and "Third*Reward" columns
                             (default True).

    Returns:
        A three-tuple ``(regressors, true_choices, col_labels)`` where:

        - *regressors* is ``{subject: ndarray(n_trials, n_cols)}``,
        - *true_choices* is ``{subject: ndarray(n_trials, 3)}``
          columns = [chose_best, chose_second, chose_third],
        - *col_labels* is a list of strings describing each column.
    """
    from src.behavior_analysis.get_variables_across_sessions import (
        get_vars_across_all_sessions,
    )

    merged_data, _ = get_vars_across_all_sessions(data)

    rank_names = ["best", "second", "third"]

    # Build full column labels
    full_labels = ["Bias"]
    for lag in range(1, n_lags + 1):
        for r in rank_names:
            full_labels.append(f"Chose {r.capitalize()} t-{lag}")
    for lag in range(1, n_lags + 1):
        full_labels.append(f"Reward t-{lag}")
    for lag in range(1, n_lags + 1):
        for r in rank_names:
            full_labels.append(f"{r.capitalize()}*Reward t-{lag}")

    # Redundant cols: "Chose Third t-k" and "Third*Reward t-k"
    drop_label_patterns = ["Chose Third", "Third*Reward"]
    if drop_redundant_cols:
        keep_mask   = [not any(p in l for p in drop_label_patterns)
                       for l in full_labels]
        col_labels  = [l for l, k in zip(full_labels, keep_mask) if k]
    else:
        keep_mask  = [True] * len(full_labels)
        col_labels = full_labels

    n_full_cols = len(full_labels)

    regressors_across_subjects:    dict = {}
    true_choices_across_subjects:  dict = {}

    for subject, subject_data in merged_data.items():
        choices_by_tower        = subject_data["choices_by_tower"]
        reward_magnitudes_by_tower = subject_data["reward_magnitudes_by_tower"]
        towers = list(choices_by_tower.keys())

        n_trials = min(len(choices_by_tower[t]) for t in towers)
        n_trials = min(n_trials,
                       min(len(reward_magnitudes_by_tower[t]) for t in towers))

        # Per-trial chosen tower
        chosen_tower_per_trial = []
        for i in range(n_trials):
            chosen = [t for t in towers if choices_by_tower[t][i] == 1]
            chosen_tower_per_trial.append(chosen[0] if len(chosen) == 1 else None)

        # Per-trial reward magnitude of chosen arm
        reward_magnitudes_list = [
            {t: reward_magnitudes_by_tower[t][i] for t in towers}
            for i in range(n_trials)
        ]
        chosen_rewards = [
            reward_magnitudes_by_tower[chosen_tower_per_trial[i]][i]
            if chosen_tower_per_trial[i] is not None else np.nan
            for i in range(n_trials)
        ]

        # Per-trial rank mapping: tower -> "best"/"second"/"third"
        if use_believed:
            rank_at_trial = _get_believed_rank_at_trial(
                chosen_tower_per_trial, reward_magnitudes_list
            )
        else:
            rank_at_trial = _get_true_rank_at_trial(reward_magnitudes_list)

        # Per-trial chose_rank indicator: {rank: 0/1}
        chose_rank = []
        for i in range(n_trials):
            tower   = chosen_tower_per_trial[i]
            rm      = rank_at_trial[i]
            if tower is not None and rm is not None:
                r = rm.get(tower)
                chose_rank.append({rk: (1 if rk == r else 0) for rk in rank_names})
            else:
                chose_rank.append({rk: np.nan for rk in rank_names})

        regressors   = np.full((n_trials, n_full_cols), np.nan)
        true_choices = np.full((n_trials, 3), np.nan)

        for t in range(1, n_trials):
            # true choice at t (by rank at trial t)
            rm_t = rank_at_trial[t]
            tower_t = chosen_tower_per_trial[t]
            if rm_t is not None and tower_t is not None:
                chosen_rank_t = rm_t.get(tower_t)
                true_choices[t] = [
                    1.0 if chosen_rank_t == "best"   else 0.0,
                    1.0 if chosen_rank_t == "second" else 0.0,
                    1.0 if chosen_rank_t == "third"  else 0.0,
                ]

            regressors[t, 0] = 1.0  # bias

            for lag in range(1, n_lags + 1):
                tk = t - lag
                if tk < 0:
                    continue

                # chose-rank indicators at t-lag (using rank at t-lag)
                cr = chose_rank[tk]
                base_choice_col = 1 + (lag - 1) * 3
                regressors[t, base_choice_col]     = cr.get("best",   np.nan)
                regressors[t, base_choice_col + 1] = cr.get("second", np.nan)
                regressors[t, base_choice_col + 2] = cr.get("third",  np.nan)

                # reward at t-lag
                base_reward_col = 1 + n_lags * 3 + (lag - 1)
                rwd = chosen_rewards[tk]
                regressors[t, base_reward_col] = rwd

                # rank * reward interactions at t-lag
                base_inter_col = 1 + n_lags * 3 + n_lags + (lag - 1) * 3
                regressors[t, base_inter_col]     = cr.get("best",   np.nan) * rwd
                regressors[t, base_inter_col + 1] = cr.get("second", np.nan) * rwd
                regressors[t, base_inter_col + 2] = cr.get("third",  np.nan) * rwd

        # Drop first trial (no history) and any trial with NaN rank
        valid = np.arange(1, n_trials)
        # also drop trials where believed rank was unavailable (NaN in regressors)
        nan_rows = np.any(np.isnan(regressors[valid]), axis=1)
        valid    = valid[~nan_rows]

        regressors   = regressors[valid]
        true_choices = true_choices[valid]

        if drop_redundant_cols:
            regressors = regressors[:, keep_mask]

        regressors_across_subjects[subject]   = regressors
        true_choices_across_subjects[subject] = true_choices

    return regressors_across_subjects, true_choices_across_subjects, col_labels
