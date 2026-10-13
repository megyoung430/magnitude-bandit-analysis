"""Utilities for building GLM (multinomial logistic regression) inputs.

Functions here transform raw trial data (choices and rewards per tower) into
design matrices suitable for multinomial logistic regression.
"""

from __future__ import annotations

from functools import partial
from itertools import compress
from typing import Any

import numpy as np

from src.behavior_analysis.get_variables_across_sessions import get_vars_across_all_sessions


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
            raise ValueError(f"Trial {i}: expected exactly one chosen tower, got {chosen}")
        rewards.append(reward_magnitudes_by_tower[chosen[0]][i])

    return rewards


# ---------------------------------------------------------------------------
# Design-matrix building blocks shared by both formatters
# ---------------------------------------------------------------------------

_N_ARMS = 3
_RANK_NAMES = ("best", "second", "third")
_ARM_REDUNDANT = ("Arm3",)
_RANK_REDUNDANT = ("Chose Third", "Third*Reward")


def _design_labels(choice_terms: list[str], interaction_terms: list[str], n_lags: int) -> list[str]:
    """Column labels in design-matrix order: bias, choices, rewards, interactions.

    Choice and interaction terms repeat once per lag (lag-major), the reward
    term has a single column per lag.
    """
    lags = range(1, n_lags + 1)
    return (
        ["Bias"]
        + [f"{term} t-{lag}" for lag in lags for term in choice_terms]
        + [f"Reward t-{lag}" for lag in lags]
        + [f"{term} t-{lag}" for lag in lags for term in interaction_terms]
    )


def _keep_mask(labels: list[str], redundant_patterns: tuple[str, ...]) -> np.ndarray:
    """Boolean mask of the columns whose label contains none of the patterns."""
    return np.array([not any(p in label for p in redundant_patterns) for label in labels])


def _lagged(values: np.ndarray, lag: int, fill: float) -> np.ndarray:
    """``values`` shifted down by ``lag`` trials; the first ``lag`` rows are ``fill``."""
    out = np.full(values.shape, fill, dtype=float)
    if lag < len(values):
        out[lag:] = values[: len(values) - lag]
    return out


def _design_matrix(
    choices: np.ndarray, rewards: np.ndarray, n_lags: int, fill: float
) -> np.ndarray:
    """Full ``(n_trials, 1 + 7 * n_lags)`` regressor matrix, one row per trial.

    Args:
        choices: ``(n_trials, 3)`` choice indicators.
        rewards: ``(n_trials,)`` reward magnitude of the chosen arm.
        n_lags: Number of history lags.
        fill: Value for history that reaches before the first trial.

    Row ``t`` holds the bias, then the choices, reward and choice x reward
    interactions at trials ``t-1 .. t-n_lags``.
    """
    interactions = choices * rewards[:, None]
    lags = range(1, n_lags + 1)
    blocks = (
        [np.ones((len(rewards), 1))]
        + [_lagged(choices, lag, fill) for lag in lags]
        + [_lagged(rewards, lag, fill)[:, None] for lag in lags]
        + [_lagged(interactions, lag, fill) for lag in lags]
    )
    return np.hstack(blocks)


def _designs_by_subject(merged_data: dict, design_fn) -> tuple[dict, dict]:
    """Apply ``design_fn(subject_data) -> (regressors, true_choices)`` to every subject.

    Returns ``({subject: regressors}, {subject: true_choices})``.
    """
    per_subject = {subject: design_fn(data) for subject, data in merged_data.items()}
    regressors = {subject: x for subject, (x, _) in per_subject.items()}
    true_choices = {subject: y for subject, (_, y) in per_subject.items()}
    return regressors, true_choices


def _arm_labels(n_lags: int) -> list[str]:
    arms = [f"Arm{a}" for a in range(1, _N_ARMS + 1)]
    return _design_labels([f"Choice {a}" for a in arms], [f"Chx Rew {a}" for a in arms], n_lags)


def _rank_labels(n_lags: int) -> list[str]:
    ranks = [r.capitalize() for r in _RANK_NAMES]
    return _design_labels([f"Chose {r}" for r in ranks], [f"{r}*Reward" for r in ranks], n_lags)


# ---------------------------------------------------------------------------
# Arm-ordered GLM
# ---------------------------------------------------------------------------


def _first_n(values: list, n_trials: int, what: str) -> np.ndarray:
    """First ``n_trials`` entries as floats; fail if the series is shorter."""
    if len(values) < n_trials:
        raise ValueError(f"{what} has {len(values)} entries for {n_trials} trials")
    return np.asarray(values[:n_trials], dtype=float)


def _arm_design_for_subject(
    subject_data: dict, n_lags: int, n_cols: int
) -> tuple[np.ndarray, np.ndarray]:
    """Full-width regressors and true choices (first trial dropped) for one subject."""
    choices_by_tower = subject_data["choices_by_tower"]
    chosen_rewards = get_chosen_reward_per_trial(
        choices_by_tower, subject_data["reward_magnitudes_by_tower"]
    )
    n_trials = len(subject_data["trial"])
    if n_trials < 2:  # the first trial has no history, so no row survives
        return np.zeros((0, n_cols)), np.zeros((0, len(choices_by_tower)))

    choices_by_arm = remap_tower_dict_to_arms(choices_by_tower, make_arm_mapping(choices_by_tower))
    arm_series = [choices_by_arm[f"Arm{a}"] for a in range(1, _N_ARMS + 1)]
    if len(choices_by_tower) != _N_ARMS:
        raise ValueError(f"Expected {_N_ARMS} towers, got {sorted(choices_by_tower)}")
    choices = np.column_stack([_first_n(s, n_trials, "choices") for s in arm_series])
    rewards = _first_n(chosen_rewards, n_trials, "chosen rewards")
    return _design_matrix(choices, rewards, n_lags, fill=0.0)[1:], choices[1:]


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
    merged_data, _ = get_vars_across_all_sessions(data)

    full_labels = _arm_labels(n_lags)
    keep = _keep_mask(full_labels, _ARM_REDUNDANT)
    regressors, true_choices = _designs_by_subject(
        merged_data, partial(_arm_design_for_subject, n_lags=n_lags, n_cols=len(full_labels))
    )
    if drop_redundant_cols:
        regressors = {subject: x[:, keep] for subject, x in regressors.items()}
    # Known issue: the labels omit the Arm3 columns even when
    # drop_redundant_cols=False keeps them, so labels and columns misalign.
    return regressors, true_choices, list(compress(full_labels, keep))


# ---------------------------------------------------------------------------
# Rank-based GLM
# ---------------------------------------------------------------------------


def _ranks_from_magnitudes(magnitudes: dict) -> dict:
    """Map tower -> "best"/"second"/"third" by descending magnitude.

    Ties are broken deterministically by tower key (lexicographic); any tower
    beyond the third is labelled "third".
    """
    ordered = sorted(magnitudes.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return {tower: _RANK_NAMES[min(i, 2)] for i, (tower, _) in enumerate(ordered)}


def _get_true_rank_at_trial(reward_magnitudes_list: list[dict]) -> list[dict]:
    """
    For each trial, return a dict mapping tower -> rank label
    ("best", "second", "third") based on the true reward magnitudes
    at that trial.

    Ties are broken deterministically by tower key (lexicographic).
    """
    return [_ranks_from_magnitudes(rm) for rm in reward_magnitudes_list]


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
    last_seen = {}
    result = []

    for choice, rm in zip(choices, reward_magnitudes_list, strict=True):
        if all(arm in last_seen for arm in rm):
            result.append(_ranks_from_magnitudes(last_seen))
        else:
            result.append(None)

        # update AFTER recording
        if choice and choice in rm:
            last_seen[choice] = rm[choice]

    return result


def _chosen_tower_per_trial(choices_by_tower: dict, towers: list, n_trials: int) -> list:
    """The chosen tower at each trial, or ``None`` unless exactly one was chosen."""
    chosen = [[t for t in towers if choices_by_tower[t][i] == 1] for i in range(n_trials)]
    return [c[0] if len(c) == 1 else None for c in chosen]


def _reward_magnitudes_per_trial(
    reward_magnitudes_by_tower: dict, towers: list, n_trials: int
) -> list[dict]:
    """``{tower: magnitude}`` for each trial."""
    return [{t: reward_magnitudes_by_tower[t][i] for t in towers} for i in range(n_trials)]


def _chosen_rewards(reward_magnitudes_by_tower: dict, chosen_towers: list) -> np.ndarray:
    """Reward magnitude of the chosen tower at each trial (NaN when none was chosen)."""
    return np.array(
        [
            reward_magnitudes_by_tower[tower][i] if tower is not None else np.nan
            for i, tower in enumerate(chosen_towers)
        ],
        dtype=float,
    )


def _chose_rank_indicators(chosen_towers: list, ranks_per_trial: list) -> np.ndarray:
    """``(n_trials, 3)`` one-hot of the chosen rank [best, second, third].

    Rows are NaN where no single tower was chosen or no rank is available.
    """

    def one_hot(tower, ranks):
        if tower is None or ranks is None:
            return [np.nan] * len(_RANK_NAMES)
        return [1.0 if ranks.get(tower) == name else 0.0 for name in _RANK_NAMES]

    rows = [
        one_hot(tower, ranks) for tower, ranks in zip(chosen_towers, ranks_per_trial, strict=True)
    ]
    return np.array(rows, dtype=float).reshape(-1, len(_RANK_NAMES))


def _complete_history_rows(regressors: np.ndarray) -> np.ndarray:
    """Indices of rows after the first trial that contain no NaN."""
    rows = np.arange(1, len(regressors))
    return rows[~np.isnan(regressors[rows]).any(axis=1)]


def _rank_design_for_subject(
    subject_data: dict, use_believed: bool, n_lags: int
) -> tuple[np.ndarray, np.ndarray]:
    """Full-width regressors and rank-ordered true choices for one subject.

    Only trials with a complete history (no NaN anywhere in the row) are kept.
    """
    choices_by_tower = subject_data["choices_by_tower"]
    reward_magnitudes_by_tower = subject_data["reward_magnitudes_by_tower"]
    towers = list(choices_by_tower)

    n_trials = min(
        min(len(choices_by_tower[t]) for t in towers),
        min(len(reward_magnitudes_by_tower[t]) for t in towers),
    )
    chosen = _chosen_tower_per_trial(choices_by_tower, towers, n_trials)
    magnitudes = _reward_magnitudes_per_trial(reward_magnitudes_by_tower, towers, n_trials)
    rank_at_trial = (
        _get_believed_rank_at_trial(chosen, magnitudes)
        if use_believed
        else _get_true_rank_at_trial(magnitudes)
    )
    # The rank indicator at trial t is both the outcome at t and the choice
    # regressor at t+lag, so one array serves as true choices and history.
    indicators = _chose_rank_indicators(chosen, rank_at_trial)
    rewards = _chosen_rewards(reward_magnitudes_by_tower, chosen)
    regressors = _design_matrix(indicators, rewards, n_lags, fill=np.nan)
    rows = _complete_history_rows(regressors)
    return regressors[rows], indicators[rows]


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
    merged_data, _ = get_vars_across_all_sessions(data)

    full_labels = _rank_labels(n_lags)
    regressors, true_choices = _designs_by_subject(
        merged_data, partial(_rank_design_for_subject, use_believed=use_believed, n_lags=n_lags)
    )
    if not drop_redundant_cols:
        return regressors, true_choices, full_labels

    keep = _keep_mask(full_labels, _RANK_REDUNDANT)
    regressors = {subject: x[:, keep] for subject, x in regressors.items()}
    return regressors, true_choices, list(compress(full_labels, keep))
