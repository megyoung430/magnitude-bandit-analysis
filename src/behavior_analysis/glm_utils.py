"""Utilities for building GLM (multinomial logistic regression) inputs.

Functions here transform raw trial data (choices and rewards per tower) into
design matrices suitable for multinomial logistic regression.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import compress, pairwise
from typing import Any

import numpy as np

from src.behavior_analysis.get_variables_across_sessions import get_vars_across_all_sessions

logger = logging.getLogger(__name__)


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


def remap_tower_dict_to_arms(d: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    """Remap a ``{tower: data}`` dict to ``{ArmN: data}`` using a mapping.

    Args:
        d: Dict keyed by tower identifiers.
        mapping: ``{tower: arm_name}`` as produced by :func:`make_arm_mapping`.

    Returns:
        New dict keyed by arm names; towers absent from *mapping* are skipped.
    """
    return {mapping[tower]: arr for tower, arr in d.items() if tower in mapping}


def get_chosen_reward_per_trial(
    choices_by_tower: dict[str, list[int]],
    reward_magnitudes_by_tower: dict[str, list[float]],
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


@dataclass(frozen=True)
class RowCounts:
    """Where one subject's trials went on the way to the fitted rows.

    ``n_trials = n_dropped_no_history + n_dropped_nan + n_rows``.

    Attributes:
        n_trials: Trials of the subject across all sessions (``len(trial)``).
        n_dropped_no_history: Trials set aside because a lag has no real trial
            behind it. With ``reset_history_per_session=True``: the first
            ``n_lags`` trials of every session. With ``False``: the arm design
            drops only trial 0 (the next ``n_lags - 1`` trials are kept, their
            lags reaching before the start filled with zeros); the rank design
            drops the first ``max(n_lags, 1)`` trials.
        n_dropped_nan: Remaining trials with a NaN response (true choice) or a
            NaN in any regressor.
        n_rows: Rows returned.
    """

    n_trials: int
    n_dropped_no_history: int
    n_dropped_nan: int
    n_rows: int


@dataclass(frozen=True)
class GlmDesign:
    """Design matrices of all subjects together with how they were built.

    Attributes:
        regressors: ``{subject: ndarray(n_rows, n_cols)}``.
        true_choices: ``{subject: ndarray(n_rows, 3)}``.
        col_labels: One label per column of every regressor matrix.
        row_counts: ``{subject: RowCounts}``.
        n_lags: Number of history lags.
        reset_history_per_session: Whether lags were restricted to the trial's
            own session.
        reset_beliefs_per_session: Rank design only (``None`` for the arm
            design, which has no beliefs): whether the believed ranks
            (last-seen magnitudes) started empty in every session. Only
            affects the result with ``use_believed=True``.
    """

    regressors: dict[str, np.ndarray]
    true_choices: dict[str, np.ndarray]
    col_labels: list[str]
    row_counts: dict[str, RowCounts]
    n_lags: int
    reset_history_per_session: bool
    reset_beliefs_per_session: bool | None = None


@dataclass(frozen=True)
class _Subject:
    """One subject's merged trials plus the length of each merged session."""

    name: str
    data: dict[str, Any]
    session_lengths: list[int]


def _design_labels(
    choice_terms: list[str], *, interaction_terms: list[str], n_lags: int
) -> list[str]:
    """Column labels in design-matrix order: bias, choices, rewards, interactions.

    ``interaction_terms`` is keyword-only: the two term lists have the same
    shape, and swapping them would silently mislabel the columns.

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


def _arm_labels(n_lags: int) -> list[str]:
    arms = [f"Arm{a}" for a in range(1, _N_ARMS + 1)]
    return _design_labels(
        [f"Choice {a}" for a in arms],
        interaction_terms=[f"Chx Rew {a}" for a in arms],
        n_lags=n_lags,
    )


def _rank_labels(n_lags: int) -> list[str]:
    ranks = [r.capitalize() for r in _RANK_NAMES]
    return _design_labels(
        [f"Chose {r}" for r in ranks],
        interaction_terms=[f"{r}*Reward" for r in ranks],
        n_lags=n_lags,
    )


# ---------------------------------------------------------------------------
# Sessions, history rows and NaN rows
# ---------------------------------------------------------------------------


def _tower_set_text(towers: set[str]) -> str:
    return "{" + ", ".join(repr(t) for t in sorted(towers)) + "}"


def _check_session_towers(subject: str, unmerged: dict[str, list]) -> None:
    """Raise unless every session has the tower set of the first session.

    Sessions are merged tower by tower, so a session lacking a tower would make
    that tower's list shorter than the trial list.
    """
    kinds = (("choices", "choices_by_tower"), ("rewards", "reward_magnitudes_by_tower"))
    n_sessions = len(unmerged["trial"])
    expected = set(dict(unmerged["choices_by_tower"][0]))
    for i in range(n_sessions):
        for kind, key in kinds:
            towers = set(dict(unmerged[key][i]))
            if towers != expected:
                raise ValueError(
                    f"subject {subject!r}: session {i + 1} of {n_sessions} has towers "
                    f"{_tower_set_text(towers)}, expected {_tower_set_text(expected)} ({kind})"
                )


def _check_sessions_aligned(
    subject: str, unmerged: dict[str, list], *, allow_missing_final_reward: bool
) -> None:
    """Raise unless every session has exactly one choice and one reward per trial.

    Sessions are merged by concatenating their lists, so a list that is too
    short or too long in one session moves every later session's entries onto
    the wrong trials, and in the last session it silently changes which trials
    are used. The one exception, with ``allow_missing_final_reward``: the last
    session's reward list may lack the final trial, whose reward is never part
    of any history.
    """
    n_trials = [len(trials) for trials in unmerged["trial"]]
    if not n_trials:
        return
    _check_session_towers(subject, unmerged)
    last = len(n_trials) - 1
    for kind, key in (("choices", "choices_by_tower"), ("rewards", "reward_magnitudes_by_tower")):
        for i, (n, series_by_tower) in enumerate(zip(n_trials, unmerged[key], strict=True)):
            shortest_ok = (
                n - 1 if (kind == "rewards" and i == last and allow_missing_final_reward) else n
            )
            for tower, values in dict(series_by_tower).items():
                if not shortest_ok <= len(values or []) <= n:
                    raise ValueError(
                        f"subject {subject!r}: session {i + 1} of {len(n_trials)} has "
                        f"{len(values or [])} {kind} for tower {tower!r} but {n} trials; "
                        "every session needs one entry per trial"
                        + (
                            " (the last session may lack the final reward)"
                            if shortest_ok < n
                            else ""
                        )
                    )


def _subjects(data: dict[str, Any], *, allow_missing_final_reward: bool) -> list[_Subject]:
    """Merge every subject's sessions and keep the session lengths for history resets."""
    merged, unmerged = get_vars_across_all_sessions(data)
    for name, parts in unmerged.items():
        _check_sessions_aligned(name, parts, allow_missing_final_reward=allow_missing_final_reward)
    return [
        _Subject(name, merged[name], [len(trials) for trials in unmerged[name]["trial"]])
        for name in merged
    ]


def _positions_in_session(session_lengths: list[int], n_trials: int) -> np.ndarray:
    """0-based position of each of the first ``n_trials`` merged trials within its session.

    Trials beyond the summed session lengths count on in the last session.
    """
    index = np.arange(n_trials)
    if not session_lengths:
        return index
    starts = np.cumsum([0, *session_lengths[:-1]])
    return index - starts[np.searchsorted(starts, index, side="right") - 1]


def _history_ready(
    positions: np.ndarray, *, n_lags: int, reset: bool, first_row: int
) -> np.ndarray:
    """Boolean mask of the trials whose ``n_lags`` lags all have a real trial behind them.

    With ``reset`` a lag may not leave the trial's own session, so the first
    ``n_lags`` trials of every session are not ready. Without it, history runs
    across sessions and only the trials before ``first_row`` of the merged data
    are not ready.
    """
    if reset:
        return positions >= n_lags
    return np.arange(len(positions)) >= first_row


def _select_complete_rows(
    regressors: np.ndarray, true_choices: np.ndarray, ready: np.ndarray
) -> tuple[np.ndarray, np.ndarray, RowCounts]:
    """Rows that are ready and free of NaN in the regressors and the true choice.

    The full-width regressors are checked, including the columns that may be
    dropped later: a redundant column is NaN only where another used column is.
    """
    complete = ~(np.isnan(regressors).any(axis=1) | np.isnan(true_choices).any(axis=1))
    keep = ready & complete
    counts = RowCounts(
        n_trials=len(regressors),
        n_dropped_no_history=int((~ready).sum()),
        n_dropped_nan=int((ready & ~complete).sum()),
        n_rows=int(keep.sum()),
    )
    return regressors[keep], true_choices[keep], counts


def _build_design(
    data: dict[str, Any],
    *,
    design_fn: Callable[[_Subject], tuple[np.ndarray, np.ndarray, RowCounts]],
    full_labels: list[str],
    redundant_patterns: tuple[str, ...],
    drop_redundant_cols: bool,
    n_lags: int,
    reset: bool,
    reset_beliefs: bool | None = None,
    allow_missing_final_reward: bool,
) -> GlmDesign:
    """Apply ``design_fn`` to every subject and drop the redundant columns of all of them.

    The returned labels are those of the returned columns.
    """
    subjects = _subjects(data, allow_missing_final_reward=allow_missing_final_reward)
    per_subject = {s.name: design_fn(s) for s in subjects}
    keep = (
        _keep_mask(full_labels, redundant_patterns=redundant_patterns)
        if drop_redundant_cols
        else np.ones(len(full_labels), dtype=bool)
    )
    for name, (_, _, counts) in per_subject.items():
        if counts.n_dropped_nan:
            logger.warning(
                "subject %s: dropped %d of %d trials with a NaN response or regressor",
                name,
                counts.n_dropped_nan,
                counts.n_trials,
            )
    return GlmDesign(
        regressors={name: x[:, keep] for name, (x, _, _) in per_subject.items()},
        true_choices={name: y for name, (_, y, _) in per_subject.items()},
        col_labels=list(compress(full_labels, keep)),
        row_counts={name: counts for name, (_, _, counts) in per_subject.items()},
        n_lags=n_lags,
        reset_history_per_session=reset,
        reset_beliefs_per_session=reset_beliefs,
    )


# ---------------------------------------------------------------------------
# Arm-ordered GLM
# ---------------------------------------------------------------------------


def _check_three_towers(subject: str, towers: Any) -> None:
    """Raise unless the subject has exactly the three towers the arm design is built for."""
    if len(towers) != _N_ARMS:
        raise ValueError(
            f"expected exactly {_N_ARMS} towers, got {len(towers)} ({sorted(towers)}) "
            f"for subject {subject!r}"
        )


def _check_one_choice_per_trial(subject: _Subject) -> None:
    """Raise, naming the session and its own trial number, at the first trial without one choice."""
    choices_by_tower = subject.data["choices_by_tower"]
    towers = list(choices_by_tower)
    chosen = np.array([[v == 1 for v in choices_by_tower[t]] for t in towers])
    bad = np.flatnonzero(chosen.sum(axis=0) != 1)
    if bad.size:
        first = int(bad[0])
        starts = np.cumsum([0, *subject.session_lengths])
        session = int(np.searchsorted(starts, first, side="right"))
        picked = [t for t, row in zip(towers, chosen, strict=True) if row[first]]
        raise ValueError(
            f"subject {subject.name!r}: session {session} of {len(subject.session_lengths)}, "
            f"trial {first - starts[session - 1] + 1} (1-based): expected exactly one chosen "
            f"tower, got {picked}"
        )


def _choice_rows(series: list[list[int]], *, n_trials: int) -> list[list[int]]:
    """The first ``n_trials`` choices of every arm series, one row per trial."""
    return [[s[t] for s in series] for t in range(n_trials)]


def _arm_design_for_subject(
    subject: _Subject, *, n_lags: int, reset: bool
) -> tuple[np.ndarray, np.ndarray, RowCounts]:
    """Full-width regressors and true choices of the usable trials of one subject.

    A reward list one shorter than ``trial`` is accepted for the last session
    (its final reward is never used).
    """
    choices_by_tower = subject.data["choices_by_tower"]
    n_trials = len(subject.data["trial"])
    _check_three_towers(subject.name, choices_by_tower)
    _check_one_choice_per_trial(subject)
    chosen_rewards = get_chosen_reward_per_trial(
        choices_by_tower, subject.data["reward_magnitudes_by_tower"]
    )

    choices_by_arm = remap_tower_dict_to_arms(choices_by_tower, make_arm_mapping(choices_by_tower))
    series = [choices_by_arm[f"Arm{a}"] for a in range(1, _N_ARMS + 1)]
    rows = _choice_rows(series, n_trials=n_trials)
    rewards, interactions = _history_terms(rows, chosen_rewards, n_lags=n_lags)
    choices = np.array(rows, dtype=float).reshape(-1, _N_ARMS)
    x = _design_matrix(choices, rewards, interactions, n_lags=n_lags, fill=0.0)
    # Without a reset, history before the first trial is zero-filled: only trial 0 lacks history.
    ready = _history_ready(
        _positions_in_session(subject.session_lengths, n_trials),
        n_lags=n_lags,
        reset=reset,
        first_row=1,
    )
    return _select_complete_rows(x, choices, ready)


def format_glm_input_with_info(
    data: dict[str, Any],
    n_lags: int = 4,
    drop_redundant_cols: bool = True,
    *,
    reset_history_per_session: bool = True,
) -> GlmDesign:
    """Build arm-ordered GLM design matrices and report how they were built.

    Same as :func:`format_glm_input`, but returns a :class:`GlmDesign` that also
    records ``n_lags``, ``reset_history_per_session`` and, per subject, how many
    trials were dropped and why (:class:`RowCounts`).
    """
    return _build_design(
        data,
        design_fn=partial(_arm_design_for_subject, n_lags=n_lags, reset=reset_history_per_session),
        full_labels=_arm_labels(n_lags),
        redundant_patterns=_ARM_REDUNDANT,
        drop_redundant_cols=drop_redundant_cols,
        n_lags=n_lags,
        reset=reset_history_per_session,
        allow_missing_final_reward=True,
    )


def format_glm_input(
    data: dict[str, Any],
    n_lags: int = 4,
    drop_redundant_cols: bool = True,
    *,
    reset_history_per_session: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
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
        reset_history_per_session: How lags treat session boundaries
            (keyword-only, default ``True``).

            - ``True``: a lag never crosses a session boundary. The first
              ``n_lags`` trials of every session have no complete history and
              are dropped (with ``n_lags=0`` nothing is dropped).
            - ``False``: sessions are concatenated and history runs across
              their boundaries; only the first trial of the whole data is
              dropped, and lags reaching before it are zero-filled.

    Rows with a NaN regressor (for example a NaN reward in the history) or a
    NaN true choice are dropped whatever ``reset_history_per_session`` is; use
    :func:`format_glm_input_with_info` to get the number of dropped rows and the
    settings used.

    Returns:
        A three-tuple ``(regressors_across_subjects, true_choices_across_subjects,
        col_labels)`` where:

        - *regressors_across_subjects* is ``{subject: ndarray(n_rows, n_cols)}``,
        - *true_choices_across_subjects* is ``{subject: ndarray(n_rows, 3)}``,
        - *col_labels* is the list of the column labels of the returned
          matrices (``Arm3`` columns included only with
          ``drop_redundant_cols=False``).

    Raises:
        ValueError: For malformed subjects: a tower count other than three,
            sessions with different tower sets, a trial without exactly one
            chosen tower (naming session and trial), or a choice or reward list
            that does not have one entry per trial of its session. Only the
            last session's reward list may lack its final entry.
        TypeError: If the reward of a chosen tower that enters a lag is missing
            (``None``) and ``n_lags >= 1``.
    """
    design = format_glm_input_with_info(
        data,
        n_lags,
        drop_redundant_cols,
        reset_history_per_session=reset_history_per_session,
    )
    return design.regressors, design.true_choices, design.col_labels


# ---------------------------------------------------------------------------
# Rank-based GLM
# ---------------------------------------------------------------------------


def _ranks_from_magnitudes(magnitudes: dict[str, float]) -> dict[str, str]:
    """Map tower -> "best"/"second"/"third" by descending magnitude.

    Ties are broken deterministically by tower key (lexicographic); any tower
    beyond the third is labelled "third".
    """
    ordered = sorted(magnitudes.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return {
        tower: _RANK_NAMES[min(i, len(_RANK_NAMES) - 1)] for i, (tower, _) in enumerate(ordered)
    }


def _rank_or_none(magnitudes: dict[str, float]) -> dict[str, str] | None:
    """:func:`_ranks_from_magnitudes`, or ``None`` if a magnitude is NaN (the order is unknown)."""
    if any(m != m for m in magnitudes.values()):
        return None
    return _ranks_from_magnitudes(magnitudes)


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
            result.append(_rank_or_none(last_seen))
        else:
            result.append(None)

        # update AFTER recording
        if choice and choice in rm:
            last_seen[choice] = rm[choice]

    return result


def _believed_ranks_by_session(
    chosen: list, magnitudes: list[dict], positions: np.ndarray
) -> list[dict | None]:
    """Believed ranks computed afresh at every session start (``positions == 0``)."""
    starts = [*np.flatnonzero(positions == 0).tolist(), len(chosen)]
    return [
        rank
        for first, last in pairwise(starts)
        for rank in _get_believed_rank_at_trial(chosen[first:last], magnitudes[first:last])
    ]


def _chosen_tower_per_trial(
    choices_by_tower: dict[str, list[int]], towers: list[str], *, n_trials: int
) -> list[str | None]:
    """The chosen tower at each trial, or ``None`` unless exactly one was chosen."""
    chosen = [[t for t in towers if choices_by_tower[t][i] == 1] for i in range(n_trials)]
    return [c[0] if len(c) == 1 else None for c in chosen]


def _reward_magnitudes_per_trial(
    reward_magnitudes_by_tower: dict[str, list[float]], towers: list[str], *, n_trials: int
) -> list[dict[str, float]]:
    """``{tower: magnitude}`` for each trial."""
    return [{t: reward_magnitudes_by_tower[t][i] for t in towers} for i in range(n_trials)]


def _chosen_rewards(
    reward_magnitudes_by_tower: dict[str, list[float]], chosen_towers: list[str | None]
) -> list[float | None]:
    """Reward magnitude of the chosen tower at each trial (NaN when none was chosen).

    A list of the original values: a missing (``None``) reward is left for
    :func:`_history_terms` to reject instead of being coerced to NaN here.
    """
    return [
        reward_magnitudes_by_tower[tower][i] if tower is not None else np.nan
        for i, tower in enumerate(chosen_towers)
    ]


def _chose_rank_indicators(
    chosen_towers: list[str | None], ranks_per_trial: list[dict[str, str] | None]
) -> list[list[float]]:
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


def _rank_design_for_subject(
    subject: _Subject, *, use_believed: bool, n_lags: int, reset: bool, reset_beliefs: bool
) -> tuple[np.ndarray, np.ndarray, RowCounts]:
    """Full-width regressors and rank-ordered true choices of the usable trials of one subject.

    Only trials with a complete history (no NaN anywhere in the row) and a
    known own outcome (true choice) are kept.
    """
    choices_by_tower = subject.data["choices_by_tower"]
    reward_magnitudes_by_tower = subject.data["reward_magnitudes_by_tower"]
    _check_three_towers(subject.name, choices_by_tower)
    towers = list(choices_by_tower)

    n_trials = len(subject.data["trial"])
    positions = _positions_in_session(subject.session_lengths, n_trials)
    chosen = _chosen_tower_per_trial(choices_by_tower, towers, n_trials=n_trials)
    magnitudes = _reward_magnitudes_per_trial(reward_magnitudes_by_tower, towers, n_trials=n_trials)
    if not use_believed:
        rank_at_trial = [_rank_or_none(rm) for rm in magnitudes]
    elif reset_beliefs:
        rank_at_trial = _believed_ranks_by_session(chosen, magnitudes, positions)
    else:
        rank_at_trial = _get_believed_rank_at_trial(chosen, magnitudes)
    # The rank indicator at trial t is both the outcome at t and the choice
    # regressor at t+lag, so one array serves as true choices and history.
    indicator_rows = _chose_rank_indicators(chosen, rank_at_trial)
    indicators = np.array(indicator_rows, dtype=float).reshape(-1, len(_RANK_NAMES))
    rewards, interactions = _history_terms(
        indicator_rows, _chosen_rewards(reward_magnitudes_by_tower, chosen), n_lags=n_lags
    )
    regressors = _design_matrix(indicators, rewards, interactions, n_lags=n_lags, fill=np.nan)
    # Without a reset, lags that reach before the first trial are NaN, so the
    # first max(n_lags, 1) trials of the merged data lack history.
    ready = _history_ready(positions, n_lags=n_lags, reset=reset, first_row=max(n_lags, 1))
    return _select_complete_rows(regressors, indicators, ready)


def format_glm_input_by_rank_with_info(
    data: dict[str, Any],
    use_believed: bool = False,
    n_lags: int = 4,
    drop_redundant_cols: bool = True,
    *,
    reset_history_per_session: bool = True,
    reset_beliefs_per_session: bool = True,
) -> GlmDesign:
    """Build rank-ordered GLM design matrices and report how they were built.

    Same as :func:`format_glm_input_by_rank`, but returns a :class:`GlmDesign`
    that also records ``n_lags``, ``reset_history_per_session``,
    ``reset_beliefs_per_session`` and, per subject, how many trials were dropped
    and why (:class:`RowCounts`).
    """
    return _build_design(
        data,
        design_fn=partial(
            _rank_design_for_subject,
            use_believed=use_believed,
            n_lags=n_lags,
            reset=reset_history_per_session,
            reset_beliefs=reset_beliefs_per_session,
        ),
        full_labels=_rank_labels(n_lags),
        redundant_patterns=_RANK_REDUNDANT,
        drop_redundant_cols=drop_redundant_cols,
        n_lags=n_lags,
        reset=reset_history_per_session,
        reset_beliefs=reset_beliefs_per_session,
        allow_missing_final_reward=False,
    )


def format_glm_input_by_rank(
    data: dict[str, Any],
    use_believed: bool = False,
    n_lags: int = 4,
    drop_redundant_cols: bool = True,
    *,
    reset_history_per_session: bool = True,
    reset_beliefs_per_session: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
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
        reset_history_per_session: How lags treat session boundaries
                             (keyword-only, default True).

                             - True: a lag never crosses a session boundary.
                               The first ``n_lags`` trials of every session are
                               dropped (with ``n_lags=0`` none).
                             - False: sessions are concatenated and lags run
                               across their boundaries; the first
                               ``max(n_lags, 1)`` trials of the merged data
                               are dropped.
        reset_beliefs_per_session: Independent of the history option, and only
                             relevant with ``use_believed=True`` (keyword-only,
                             default True).

                             - True: the believed ranks (last-seen magnitudes)
                               start empty in every session, so a session's
                               first trials have no believed rank until every
                               arm was visited in that session.
                             - False: the last-seen magnitudes carry over from
                               the previous session, so a session's first
                               trials are ranked by the previous session's
                               last-seen magnitudes.

                             All four combinations are valid. With both options
                             False the result equals concatenating the sessions
                             and building one design. ``False`` is meant for
                             sessions of one problem (the same towers and
                             magnitude schedule); the caller's grouping by
                             problem must ensure this, because last-seen
                             magnitudes of another problem are meaningless.

    Rows are kept only when the history and the row's own outcome (true choice)
    are known: a row with no single chosen tower, or with ``use_believed`` no
    believed rank yet, is dropped whatever the two reset options are, as is a
    row with a NaN in any regressor or a NaN magnitude that makes the ranking
    of its trial (or of a trial in its history) unknown. Use
    :func:`format_glm_input_by_rank_with_info` to get the number of dropped rows
    and the settings used.

    Returns:
        A three-tuple ``(regressors, true_choices, col_labels)`` where:

        - *regressors* is ``{subject: ndarray(n_rows, n_cols)}``,
        - *true_choices* is ``{subject: ndarray(n_rows, 3)}``
          columns = [chose_best, chose_second, chose_third],
        - *col_labels* is a list of strings describing each column.

    Raises:
        ValueError: If a subject does not have exactly three towers, sessions
            have different tower sets, or a choice or reward list does not have
            one entry per trial of its session (unlike the arm design, the last
            session's rewards must be complete: the own rank of each trial needs
            every tower's magnitude).
        TypeError: If a magnitude that has to be sorted or multiplied into the
            history is missing (``None``).
    """
    design = format_glm_input_by_rank_with_info(
        data,
        use_believed,
        n_lags,
        drop_redundant_cols,
        reset_history_per_session=reset_history_per_session,
        reset_beliefs_per_session=reset_beliefs_per_session,
    )
    return design.regressors, design.true_choices, design.col_labels
