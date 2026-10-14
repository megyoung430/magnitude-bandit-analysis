"""Utilities for building GLM (multinomial logistic regression) inputs.

Functions here transform raw trial data (choices and rewards per tower) into
design matrices suitable for multinomial logistic regression.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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

_N_ARMS = 3  # also the number of rank categories (best, second, third)
_RANK_NAMES = ("best", "second", "third")
_ARM_REDUNDANT = ("Arm3",)
_RANK_REDUNDANT = ("Chose Third", "Third*Reward")


def _n_design_cols(n_lags: int) -> int:
    """Width of the full design matrix: bias, then 3 choices + 1 reward + 3 interactions per lag.

    A negative ``n_lags`` has no lags, like zero: the matrix is the bias alone.
    """
    return 1 + (2 * _N_ARMS + 1) * max(n_lags, 0)


def _design_labels(
    choice_terms: list[str], interaction_terms: list[str], *, n_lags: int
) -> list[str]:
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


def _keep_mask(labels: list[str], *, redundant_patterns: tuple[str, ...]) -> np.ndarray:
    """Boolean mask of the columns whose label contains none of the patterns."""
    return np.array([not any(p in label for p in redundant_patterns) for label in labels])


def _lagged(values: np.ndarray, *, lag: int, fill: float) -> np.ndarray:
    """``values`` shifted down by ``lag`` trials; the first ``lag`` rows are ``fill``."""
    out = np.full(values.shape, fill, dtype=float)
    if lag < len(values):
        out[lag:] = values[: len(values) - lag]
    return out


def _history_terms(
    choice_rows: Sequence[Sequence[float]], rewards: Sequence, *, n_lags: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-trial reward and choice x reward products, as ``(n,)`` and ``(n, 3)`` arrays.

    The last trial is never part of any row's history, so its reward is neither
    read nor multiplied (a missing last reward is harmless) and its entries are
    NaN padding. Without lags (``n_lags <= 0``) no history is read at all.

    The products are formed element-wise on the original objects, not on a float
    array: a missing reward (``None``) must raise ``TypeError`` here rather than
    become a NaN regressor that silently flows into a regression. A reward list
    too short for the trials raises ``IndexError``.
    """
    n_trials = len(choice_rows)
    reward_col = np.full(n_trials, np.nan)
    interactions = np.full((n_trials, _N_ARMS), np.nan)
    if n_lags <= 0:
        return reward_col, interactions

    n_history = max(n_trials - 1, 0)
    history = [rewards[i] for i in range(n_history)]
    reward_col[:n_history] = np.asarray(history, dtype=float)
    products = [[c * r for c in row] for row, r in zip(choice_rows, history, strict=False)]
    interactions[:n_history] = np.array(products, dtype=float).reshape(n_history, _N_ARMS)
    return reward_col, interactions


def _design_matrix(
    choices: np.ndarray,
    rewards: np.ndarray,
    interactions: np.ndarray,
    *,
    n_lags: int,
    fill: float,
) -> np.ndarray:
    """Full ``(n_trials, 1 + 7 * n_lags)`` regressor matrix, one row per trial.

    Args:
        choices: ``(n_trials, 3)`` choice indicators.
        rewards: ``(n_trials,)`` reward magnitude of the chosen arm.
        interactions: ``(n_trials, 3)`` choice x reward products.
        n_lags: Number of history lags.
        fill: Value for history that reaches before the first trial.

    Row ``t`` holds the bias, then the choices, reward and choice x reward
    interactions at trials ``t-1 .. t-n_lags``.
    """
    lags = range(1, n_lags + 1)
    blocks = (
        [np.ones((len(rewards), 1))]
        + [_lagged(choices, lag=lag, fill=fill) for lag in lags]
        + [_lagged(rewards, lag=lag, fill=fill)[:, None] for lag in lags]
        + [_lagged(interactions, lag=lag, fill=fill) for lag in lags]
    )
    return np.hstack(blocks)


def _designs_by_subject(
    merged_data: dict,
    *,
    design_fn: Callable[[dict], tuple[np.ndarray, np.ndarray]],
) -> tuple[dict, dict]:
    """Apply ``design_fn(subject_data) -> (regressors, true_choices)`` to every subject.

    Returns ``({subject: regressors}, {subject: true_choices})``.
    """
    per_subject = {subject: design_fn(data) for subject, data in merged_data.items()}
    regressors = {subject: x for subject, (x, _) in per_subject.items()}
    true_choices = {subject: y for subject, (_, y) in per_subject.items()}
    return regressors, true_choices


def _arm_labels(n_lags: int) -> list[str]:
    arms = [f"Arm{a}" for a in range(1, _N_ARMS + 1)]
    return _design_labels(
        [f"Choice {a}" for a in arms], [f"Chx Rew {a}" for a in arms], n_lags=n_lags
    )


def _rank_labels(n_lags: int) -> list[str]:
    ranks = [r.capitalize() for r in _RANK_NAMES]
    return _design_labels(
        [f"Chose {r}" for r in ranks], [f"{r}*Reward" for r in ranks], n_lags=n_lags
    )


# ---------------------------------------------------------------------------
# Arm-ordered GLM
# ---------------------------------------------------------------------------


def _check_three_arms(choices_by_arm: dict, n_towers: int) -> None:
    """Read the first kept trial's choices, failing for a tower count other than three.

    Known issue: only exactly three towers are supported and this is not
    checked up front. Fewer towers fail with ``KeyError('Arm3')`` (or
    ``'Arm2'``), more with numpy's "could not broadcast input array" error,
    both raised by the row assignment below.
    """
    first_row = np.zeros(n_towers)
    first_row[:] = [choices_by_arm[f"Arm{a}"][1] for a in range(1, _N_ARMS + 1)]


def _choice_rows(series: list[list], n_trials: int) -> list[list]:
    """The first ``n_trials`` choices of every arm series, one row per trial.

    Raises ``IndexError`` if a series is shorter than ``n_trials``.
    """
    return [[s[t] for s in series] for t in range(n_trials)]


def _arm_design_for_subject(subject_data: dict, *, n_lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Full-width regressors and true choices (first trial dropped) for one subject.

    Known issue: a choice series shorter than ``trial`` raises ``IndexError``,
    and a reward list one shorter than ``trial`` is accepted (the last reward
    is never used) but, when it belongs to a session that is not last, shifts
    every later session's rewards one trial earlier after merging.
    """
    choices_by_tower = subject_data["choices_by_tower"]
    chosen_rewards = get_chosen_reward_per_trial(
        choices_by_tower, subject_data["reward_magnitudes_by_tower"]
    )
    n_trials = len(subject_data["trial"])
    n_towers = len(choices_by_tower)
    if n_trials < 2:  # the first trial has no history, so no row survives
        return np.zeros((0, _n_design_cols(n_lags))), np.zeros((0, n_towers))

    choices_by_arm = remap_tower_dict_to_arms(choices_by_tower, make_arm_mapping(choices_by_tower))
    _check_three_arms(choices_by_arm, n_towers)
    series = [choices_by_arm[f"Arm{a}"] for a in range(1, _N_ARMS + 1)]

    # Rewards are checked for the trials that have choices before a short choice
    # series is reported, so errors surface in trial order.
    readable = min(n_trials, *map(len, series))
    rewards, interactions = _history_terms(
        _choice_rows(series, readable), chosen_rewards, n_lags=n_lags
    )
    choices = np.array(_choice_rows(series, n_trials), dtype=float)
    x = _design_matrix(choices, rewards, interactions, n_lags=n_lags, fill=0.0)
    return x[1:], choices[1:]


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
        - *col_labels* is a list of column label strings. Known issue: the
          labels always omit the ``Arm3`` columns, so with
          ``drop_redundant_cols=False`` they do not match the returned columns.

    Raises:
        KeyError, ValueError, IndexError: For malformed subjects (a tower count
            other than three, a trial without exactly one chosen tower, a
            choice series shorter than ``trial``).
        TypeError: If the reward of a chosen tower that enters a lag is missing
            (``None``) and ``n_lags >= 1``.

    Known issues:
        - Sessions are concatenated before the design is built, so the first
          trials of a session use the end of the previous session as lag
          history.
        - Only subjects with exactly three towers are supported.
    """
    merged_data, _ = get_vars_across_all_sessions(data)

    full_labels = _arm_labels(n_lags)
    keep = _keep_mask(full_labels, redundant_patterns=_ARM_REDUNDANT)
    regressors, true_choices = _designs_by_subject(
        merged_data, design_fn=partial(_arm_design_for_subject, n_lags=n_lags)
    )
    if drop_redundant_cols:
        regressors = {subject: x[:, keep] for subject, x in regressors.items()}
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
    return {
        tower: _RANK_NAMES[min(i, len(_RANK_NAMES) - 1)] for i, (tower, _) in enumerate(ordered)
    }


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


def _chosen_tower_per_trial(choices_by_tower: dict, towers: list, *, n_trials: int) -> list:
    """The chosen tower at each trial, or ``None`` unless exactly one was chosen."""
    chosen = [[t for t in towers if choices_by_tower[t][i] == 1] for i in range(n_trials)]
    return [c[0] if len(c) == 1 else None for c in chosen]


def _reward_magnitudes_per_trial(
    reward_magnitudes_by_tower: dict, towers: list, *, n_trials: int
) -> list[dict]:
    """``{tower: magnitude}`` for each trial."""
    return [{t: reward_magnitudes_by_tower[t][i] for t in towers} for i in range(n_trials)]


def _chosen_rewards(reward_magnitudes_by_tower: dict, chosen_towers: list) -> list:
    """Reward magnitude of the chosen tower at each trial (NaN when none was chosen).

    A list of the original values: a missing (``None``) reward is left for
    :func:`_history_terms` to reject instead of being coerced to NaN here.
    """
    return [
        reward_magnitudes_by_tower[tower][i] if tower is not None else np.nan
        for i, tower in enumerate(chosen_towers)
    ]


def _chose_rank_indicators(chosen_towers: list, ranks_per_trial: list) -> list[list]:
    """One row per trial: one-hot of the chosen rank [best, second, third].

    Rows are all NaN where no single tower was chosen or no rank is available.
    """

    def one_hot(tower, ranks):
        if tower is None or ranks is None:
            return [np.nan] * len(_RANK_NAMES)
        return [1 if ranks.get(tower) == name else 0 for name in _RANK_NAMES]

    return [
        one_hot(tower, ranks) for tower, ranks in zip(chosen_towers, ranks_per_trial, strict=True)
    ]


def _complete_history_rows(regressors: np.ndarray) -> np.ndarray:
    """Indices of rows after the first trial that contain no NaN."""
    rows = np.arange(1, len(regressors))
    return rows[~np.isnan(regressors[rows]).any(axis=1)]


def _rank_design_for_subject(
    subject_data: dict, *, use_believed: bool, n_lags: int
) -> tuple[np.ndarray, np.ndarray]:
    """Full-width regressors and rank-ordered true choices for one subject.

    Only trials with a complete history (no NaN anywhere in the row) are kept.
    A kept row's own outcome (true choice) can still be NaN, since the outcome
    is not part of the row's regressors.
    """
    choices_by_tower = subject_data["choices_by_tower"]
    reward_magnitudes_by_tower = subject_data["reward_magnitudes_by_tower"]
    towers = list(choices_by_tower)

    n_trials = min(
        min(len(choices_by_tower[t]) for t in towers),
        min(len(reward_magnitudes_by_tower[t]) for t in towers),
    )
    chosen = _chosen_tower_per_trial(choices_by_tower, towers, n_trials=n_trials)
    magnitudes = _reward_magnitudes_per_trial(reward_magnitudes_by_tower, towers, n_trials=n_trials)
    rank_at_trial = (
        _get_believed_rank_at_trial(chosen, magnitudes)
        if use_believed
        else [_ranks_from_magnitudes(rm) for rm in magnitudes]
    )
    # The rank indicator at trial t is both the outcome at t and the choice
    # regressor at t+lag, so one array serves as true choices and history.
    indicator_rows = _chose_rank_indicators(chosen, rank_at_trial)
    indicators = np.array(indicator_rows, dtype=float).reshape(-1, len(_RANK_NAMES))
    rewards, interactions = _history_terms(
        indicator_rows, _chosen_rewards(reward_magnitudes_by_tower, chosen), n_lags=n_lags
    )
    regressors = _design_matrix(indicators, rewards, interactions, n_lags=n_lags, fill=np.nan)
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
         Chose Best t-1, Chose Second t-1, Chose Third t-1,
         ...repeated for lags 2-4...,
         Reward t-1, Reward t-2, Reward t-3, Reward t-4,
         Best*Reward t-1, Second*Reward t-1, Third*Reward t-1,
         ...repeated for lags 2-4...]

    After dropping the redundant "Chose Third" and "Third*Reward" columns
    (which are linearly dependent on the other choice columns), the design
    matrix has 1 + 2*n_lags + n_lags + 2*n_lags = 1 + 5*n_lags columns.

    Args:
        data:                Raw subjects data.
        use_believed:        If True, use believed rank; if False, use true rank.
        n_lags:              Number of history lags (default 4).
        drop_redundant_cols: Drop "Chose Third" and "Third*Reward" columns
                             (default True).

    Returns:
        A three-tuple ``(regressors, true_choices, col_labels)`` where:

        - *regressors* is ``{subject: ndarray(n_trials, n_cols)}``,
        - *true_choices* is ``{subject: ndarray(n_trials, 3)}``
          columns = [chose_best, chose_second, chose_third],
        - *col_labels* is a list of strings describing each column.

    Raises:
        TypeError: If a magnitude that has to be sorted or multiplied into the
            history is missing (``None``).

    Known issues:
        - Rows are kept when their *history* is complete, so a row whose own
          outcome is NaN (no single chosen tower, or, with ``use_believed``,
          no believed rank yet) is returned with a NaN row in *true_choices*.
          With ``n_lags=0`` nothing is NaN, so every row after the first is
          kept.
        - Sessions are concatenated before the design is built: lags, and the
          believed-rank state (last-seen magnitudes), carry over from the end
          of one session into the start of the next.
    """
    merged_data, _ = get_vars_across_all_sessions(data)

    full_labels = _rank_labels(n_lags)
    regressors, true_choices = _designs_by_subject(
        merged_data,
        design_fn=partial(_rank_design_for_subject, use_believed=use_believed, n_lags=n_lags),
    )
    if not drop_redundant_cols:
        return regressors, true_choices, full_labels

    keep = _keep_mask(full_labels, redundant_patterns=_RANK_REDUNDANT)
    regressors = {subject: x[:, keep] for subject, x in regressors.items()}
    return regressors, true_choices, list(compress(full_labels, keep))
