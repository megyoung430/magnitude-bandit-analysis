"""Utilities for building GLM (multinomial logistic regression) inputs.

Functions here transform raw trial data (choices and rewards per tower) into
design matrices suitable for multinomial logistic regression.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def make_arm_mapping(tower_keys: Any) -> dict[str, str]:
    """Create a deterministic tower-key → arm-name mapping.

    Sorts tower keys lexicographically and assigns ``"Arm1"``, ``"Arm2"``,
    ``"Arm3"``, …

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
        choices_by_tower: ``{tower: list[0|1]}`` — one-hot choice indicator.
        reward_magnitudes_by_tower: ``{tower: list[float]}`` — reward magnitudes.

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
    input_dimension: int = 29,
    drop_redundant_cols: bool = True,
) -> tuple[dict, dict, list[int]]:
    """Build GLM design matrices and true-choice arrays for all subjects.

    Constructs a 29-column regressor matrix per subject containing a bias
    term, choice history (4 lags × 3 arms), reward history (4 lags), and
    choice × reward interaction terms (4 lags × 3 arms).

    Args:
        data: Raw subjects data as returned by ``import_data()``.
        input_dimension: Total number of regressor columns before column
            dropping (default: 29).
        drop_redundant_cols: If ``True``, removes the ``Arm3`` columns for
            choice and interaction terms to make the design matrix full rank
            (default: ``True``).

    Returns:
        A three-tuple ``(regressors_across_subjects, true_choices_across_subjects,
        keep_cols)`` where:

        - *regressors_across_subjects* is ``{subject: ndarray(n_trials, n_cols)}``,
        - *true_choices_across_subjects* is ``{subject: ndarray(n_trials, 3)}``,
        - *keep_cols* is the list of column indices retained after dropping.
    """
    from src.behavior_analysis.get_variables_across_sessions import (
        get_vars_across_all_sessions,
    )

    merged_data, _ = get_vars_across_all_sessions(data)

    # Columns that are redundant given the one-hot choice structure
    drop_cols = [3, 6, 9, 12, 19, 22, 25, 28]
    keep_cols = [i for i in range(input_dimension) if i not in drop_cols]

    regressors_across_subjects: dict = {}
    true_choices_across_subjects: dict = {}

    for subject, subject_data in merged_data.items():
        choices_by_tower = subject_data["choices_by_tower"]
        chosen_rewards = get_chosen_reward_per_trial(
            choices_by_tower,
            subject_data["reward_magnitudes_by_tower"],
        )

        tower_to_arm = make_arm_mapping(choices_by_tower.keys())
        choices_by_arm = remap_tower_dict_to_arms(choices_by_tower, tower_to_arm)

        num_trials = len(subject_data["trial"])
        regressors = np.zeros((num_trials, input_dimension))
        true_choices = np.zeros((num_trials, len(choices_by_tower)))

        for t in range(1, num_trials):
            true_choices[t] = [
                choices_by_arm["Arm1"][t],
                choices_by_arm["Arm2"][t],
                choices_by_arm["Arm3"][t],
            ]

            regressors[t, 0] = 1  # bias term

            # lag 1
            regressors[t, 1] = choices_by_arm["Arm1"][t - 1]
            regressors[t, 2] = choices_by_arm["Arm2"][t - 1]
            regressors[t, 3] = choices_by_arm["Arm3"][t - 1]
            regressors[t, 13] = chosen_rewards[t - 1]
            regressors[t, 17] = choices_by_arm["Arm1"][t - 1] * chosen_rewards[t - 1]
            regressors[t, 18] = choices_by_arm["Arm2"][t - 1] * chosen_rewards[t - 1]
            regressors[t, 19] = choices_by_arm["Arm3"][t - 1] * chosen_rewards[t - 1]

            if t >= 2:
                regressors[t, 4] = choices_by_arm["Arm1"][t - 2]
                regressors[t, 5] = choices_by_arm["Arm2"][t - 2]
                regressors[t, 6] = choices_by_arm["Arm3"][t - 2]
                regressors[t, 14] = chosen_rewards[t - 2]
                regressors[t, 20] = choices_by_arm["Arm1"][t - 2] * chosen_rewards[t - 2]
                regressors[t, 21] = choices_by_arm["Arm2"][t - 2] * chosen_rewards[t - 2]
                regressors[t, 22] = choices_by_arm["Arm3"][t - 2] * chosen_rewards[t - 2]

            if t >= 3:
                regressors[t, 7] = choices_by_arm["Arm1"][t - 3]
                regressors[t, 8] = choices_by_arm["Arm2"][t - 3]
                regressors[t, 9] = choices_by_arm["Arm3"][t - 3]
                regressors[t, 15] = chosen_rewards[t - 3]
                regressors[t, 23] = choices_by_arm["Arm1"][t - 3] * chosen_rewards[t - 3]
                regressors[t, 24] = choices_by_arm["Arm2"][t - 3] * chosen_rewards[t - 3]
                regressors[t, 25] = choices_by_arm["Arm3"][t - 3] * chosen_rewards[t - 3]

            if t >= 4:
                regressors[t, 10] = choices_by_arm["Arm1"][t - 4]
                regressors[t, 11] = choices_by_arm["Arm2"][t - 4]
                regressors[t, 12] = choices_by_arm["Arm3"][t - 4]
                regressors[t, 16] = chosen_rewards[t - 4]
                regressors[t, 26] = choices_by_arm["Arm1"][t - 4] * chosen_rewards[t - 4]
                regressors[t, 27] = choices_by_arm["Arm2"][t - 4] * chosen_rewards[t - 4]
                regressors[t, 28] = choices_by_arm["Arm3"][t - 4] * chosen_rewards[t - 4]

        # Drop first trial (no history)
        regressors = regressors[1:]
        true_choices = true_choices[1:]

        if drop_redundant_cols:
            regressors = regressors[:, keep_cols]

        regressors_across_subjects[subject] = regressors
        true_choices_across_subjects[subject] = true_choices

    return regressors_across_subjects, true_choices_across_subjects, keep_cols
