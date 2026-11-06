"""Psychometric curves against a continuous value: ranks, quantile bins and stay/switch pairs.

The value psychometrics of :mod:`src.behavior_analysis.value_psychometrics` count trials in a
grid of integer values (best 2-4, second 1-3). A model value (the reward part of the choice
logit, :mod:`src.behavior_analysis.choice_logit`) or a learned Q-value is continuous and in
units of its own, so the curves here are drawn against *quantile bins* of the value instead:

- Each mouse's values are cut into ``n_bins`` bins holding equal numbers of its trials
  (:func:`quantile_bins`), separately for each x axis. Value scales differ between mice (a
  weight in logits per pulse is not the same number for every mouse), and a bin of equal
  numbers of trials has the same evidence behind it for every mouse and every x. The cost is
  that a bin number is a rank, not a value; the curves show the shape of the dependence, not
  its units. Tied values (e.g. several zeros) fall in the lowest bin they reach.
- Within a mouse, trials are pooled in a bin (``pool_trials=True`` of the other modules);
  mice are the replicates of every mean and standard error.

Arms are ranked by value on each trial. Tied values share the rank positions they cover evenly
(the "split" tie mode of the other modules), so ranks never depend on port names.

Stay/switch pairs follow :mod:`src.behavior_analysis.stay_switch_psychometrics`: on trial ``t``
whose predecessor ``t-1`` lies in the same session segment, the *chosen* arm is the one chosen
on ``t-1``, ``v_chosen`` its value on ``t`` and ``v_alt`` the higher value of the other two
arms. A continuous value has no "chose the 0-value option" row, so the rows are whether the
choice on ``t-1`` was rewarded (a non-zero magnitude) or not.

Everything here is pure: arrays and tables in, :class:`Curve` objects and tables out.
"""

import numpy as np
import pandas as pd

from src.behavior_analysis.stats import nan_mean, nan_standard_error
from src.behavior_analysis.stay_switch_psychometrics import (
    stay_switch_curves,
    switch_target_curves,
)
from src.behavior_analysis.value_psychometrics import Curve

DEFAULT_BINS = 5
CHOICES = ("best", "second")
X_AXES = ("best_val", "second_val", "diff")
REWARDED, UNREWARDED = "rewarded", "unrewarded"
REWARD_ROWS = (REWARDED, UNREWARDED)
N_ARMS = 3


# ---------------------------------------------------------------------------
# Ranks and bins
# ---------------------------------------------------------------------------


def rank_table(mouse, values: np.ndarray, chosen: np.ndarray) -> pd.DataFrame:
    """Each trial's values in rank order and the chosen arm's share of the top two ranks.

    Args:
        mouse (Sequence[str]): Mouse of each trial, ``(n_trials,)``.
        values (np.ndarray): ``(n_trials, 3)`` value of each arm.
        chosen (np.ndarray): ``(n_trials,)`` index of the chosen arm.

    Returns:
        pd.DataFrame: ``mouse, v_best, v_second, v_third`` (the sorted values) and
        ``best_share``, ``second_share``: the share of the best and second rank positions held
        by the chosen arm, 1 or 0 unless it ties with another arm (equal split of the tied
        positions).

    Raises:
        ValueError: If the shapes disagree or *values* is not finite.
    """
    values = np.asarray(values, dtype=float)
    chosen = np.asarray(chosen)
    if values.ndim != 2 or values.shape[1] != N_ARMS or chosen.shape != (len(values),):
        raise ValueError(f"values must be (trials, 3) and chosen (trials,), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("values must be finite")
    own = values[np.arange(len(values)), chosen]
    above = (values > own[:, None]).sum(axis=1)
    tied = (values == own[:, None]).sum(axis=1)
    # the tied arms hold the rank positions above .. above + tied - 1
    share = [((above <= position) & (position < above + tied)) / tied for position in (0, 1)]
    ranked = -np.sort(-values, axis=1)
    return pd.DataFrame(
        dict(
            mouse=np.asarray(mouse),
            v_best=ranked[:, 0],
            v_second=ranked[:, 1],
            v_third=ranked[:, 2],
            best_share=share[0].astype(float),
            second_share=share[1].astype(float),
        )
    )


def quantile_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin number (1 to *n_bins*) of each value, the bins holding equal numbers of values.

    Bin edges are the quantiles of *x*; a value equal to an edge falls in the lower bin, so a
    run of tied values stays in one bin and the bins above it may stay empty.

    Args:
        x (np.ndarray): Values of one mouse, ``(n,)``.
        n_bins (int): Number of bins, at least 2.

    Returns:
        np.ndarray: ``(n,)`` integers.

    Raises:
        ValueError: If ``n_bins < 2``.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins}")
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(edges, x, side="left") + 1


def _binned_by_mouse(table: pd.DataFrame, column: str, n_bins: int) -> pd.Series:
    """*column* as bin numbers, cut separately for each mouse."""
    return table.groupby("mouse")[column].transform(lambda s: quantile_bins(s.to_numpy(), n_bins))


def with_bins(table: pd.DataFrame, n_bins: int = DEFAULT_BINS) -> pd.DataFrame:
    """*table* of :func:`rank_table` with the bin numbers of the three x axes added.

    The axes are the best value (``best_val``), the second value (``second_val``) and the
    best minus the second value (``diff``), each binned per mouse.
    """
    gap = table.assign(gap=table["v_best"] - table["v_second"])
    return table.assign(
        best_val=_binned_by_mouse(table, "v_best", n_bins),
        second_val=_binned_by_mouse(table, "v_second", n_bins),
        diff=_binned_by_mouse(gap, "gap", n_bins),
    )


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------


def _curve(frame: pd.DataFrame, share: str, axis: str, n_bins: int) -> Curve:
    """P(chosen arm holds the rank *share*) per mouse against the bins of *axis*."""
    x = np.arange(1, n_bins + 1)
    mice = sorted(frame["mouse"].unique())
    per_mouse = (
        frame.groupby(["mouse", axis])[share]
        .mean()
        .unstack(axis)
        .reindex(index=mice, columns=x)
        .to_numpy(dtype=float)
    )
    n_trials = frame.groupby(axis).size().reindex(x, fill_value=0).to_numpy()
    return Curve(
        x=x,
        mice=mice,
        per_mouse=per_mouse,
        mean=nan_mean(per_mouse, axis=0),
        se=nan_standard_error(per_mouse),
        n_mice=np.sum(~np.isnan(per_mouse), axis=0),
        n_trials=n_trials,
    )


def binned_psychometric_curves(
    table: pd.DataFrame, n_bins: int = DEFAULT_BINS
) -> dict[str, dict[str, Curve]]:
    """P(best) and P(second) against the quantile bins of the best, second and best - second value.

    Args:
        table (pd.DataFrame): :func:`rank_table` of one or more mice (all trials of a mouse).
        n_bins (int): Bins per axis and mouse. Default ``DEFAULT_BINS``.

    Returns:
        dict[str, dict[str, Curve]]: ``curves[choice][axis]`` for the choices ``"best"`` and
        ``"second"`` and the axes ``"best_val"``, ``"second_val"``, ``"diff"``. A ``Curve.x`` is
        the bin number, 1 to *n_bins*.

    Raises:
        ValueError: If *table* is empty or ``n_bins < 2``.
    """
    if table.empty:
        raise ValueError("no trials to make psychometric curves from")
    binned = with_bins(table, n_bins)
    return {
        choice: {axis: _curve(binned, f"{choice}_share", axis, n_bins) for axis in X_AXES}
        for choice in CHOICES
    }


# ---------------------------------------------------------------------------
# Stay/switch pairs
# ---------------------------------------------------------------------------

PAIR_META = ("mouse", "problem", "trial", "trial_in_session", "after_gap", "reward")


def _pair_mask(meta: pd.DataFrame) -> np.ndarray:
    """Trials whose predecessor is the previous row of the same mouse, problem and segment."""
    problem = pd.factorize(meta["problem"])[0]
    mouse = pd.factorize(meta["mouse"])[0]
    same_problem = (mouse[1:] == mouse[:-1]) & (problem[1:] == problem[:-1])
    starts = (meta["trial_in_session"].to_numpy() == 0) | meta["after_gap"].to_numpy()
    mask = np.zeros(len(meta), dtype=bool)
    mask[1:] = same_problem & ~starts[1:]
    return mask


def stay_switch_pairs_from_values(
    meta: pd.DataFrame, values: np.ndarray, chosen: np.ndarray, n_bins: int = DEFAULT_BINS
) -> pd.DataFrame:
    """Trial pairs ``(t-1, t)`` with the values of trial ``t`` and the stay/switch outcome.

    Args:
        meta (pd.DataFrame): One row per trial, in trial order within each mouse's problem,
            with the ``PAIR_META`` columns (``LogitDesign.meta`` of the mice, concatenated).
        values (np.ndarray): ``(n_trials, 3)`` value of each arm on each trial.
        chosen (np.ndarray): ``(n_trials,)`` index of the chosen arm.
        n_bins (int): Bins of the value axes, cut per mouse over its pairs. Default
            ``DEFAULT_BINS``.

    Returns:
        pd.DataFrame: One row per pair with ``mouse, problem, trial``; the raw values
        ``raw_chosen, raw_alt, v_other``; the bin numbers ``v_chosen, v_alt`` and ``diff`` (the
        bin of ``raw_chosen - raw_alt``); the outcomes ``stay, switch_best_alt, switch_other``
        (a switch between two alternatives of equal value counts half to each); and
        ``rewarded``: whether trial ``t-1`` paid a non-zero magnitude.

    Raises:
        ValueError: If a column is missing or the arrays do not match *meta*.
    """
    missing = [c for c in PAIR_META if c not in meta]
    if missing:
        raise ValueError(f"meta lacks the columns {missing}")
    values, chosen = np.asarray(values, dtype=float), np.asarray(chosen)
    if values.shape != (len(meta), N_ARMS) or chosen.shape != (len(meta),):
        raise ValueError(f"values {values.shape} and chosen {chosen.shape} must match meta")
    meta = meta.reset_index(drop=True)
    now = np.flatnonzero(_pair_mask(meta))
    previous_arm, current_arm = chosen[now - 1], chosen[now]
    others = (previous_arm[:, None] + np.array([1, 2])) % N_ARMS
    at = values[now]
    other_values = np.take_along_axis(at, others, axis=1)
    alt_arm = np.take_along_axis(others, other_values.argmax(axis=1)[:, None], axis=1)[:, 0]
    raw_alt, v_other = other_values.max(axis=1), other_values.min(axis=1)
    raw_chosen = at[np.arange(len(now)), previous_arm]
    stay = (current_arm == previous_arm).astype(float)
    switch = 1.0 - stay
    best_alt = np.where(raw_alt == v_other, switch / 2, (current_arm == alt_arm).astype(float))
    pairs = pd.DataFrame(
        dict(
            mouse=meta["mouse"].to_numpy()[now],
            problem=meta["problem"].to_numpy()[now],
            trial=meta["trial"].to_numpy()[now],
            raw_chosen=raw_chosen,
            raw_alt=raw_alt,
            v_other=v_other,
            stay=stay,
            switch_best_alt=best_alt,
            switch_other=switch - best_alt,
            rewarded=meta["reward"].to_numpy()[now - 1] > 0,
        )
    )
    pairs["gap"] = pairs["raw_chosen"] - pairs["raw_alt"]
    for column, source in (("v_chosen", "raw_chosen"), ("v_alt", "raw_alt"), ("diff", "gap")):
        pairs[column] = _binned_by_mouse(pairs, source, n_bins)
    return pairs.drop(columns="gap")


def split_by_previous_reward(pairs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The pairs whose previous choice was rewarded, and those whose previous choice was not.

    Returns:
        dict[str, pd.DataFrame]: ``{REWARDED: ..., UNREWARDED: ...}``, either may be empty.
    """
    return {
        REWARDED: pairs.loc[pairs["rewarded"]].reset_index(drop=True),
        UNREWARDED: pairs.loc[~pairs["rewarded"]].reset_index(drop=True),
    }


def stay_switch_curves_by_reward(
    pairs: pd.DataFrame, *, pool_trials: bool
) -> dict[str, dict[str, dict[str, Curve]] | None]:
    """:func:`stay_switch_curves` for each row of :func:`split_by_previous_reward`.

    Returns:
        dict: ``{row: {outcome: {axis: Curve}}}``; ``None`` for a row without pairs.
    """
    return {
        row: stay_switch_curves(subset, pool_trials=pool_trials) if not subset.empty else None
        for row, subset in split_by_previous_reward(pairs).items()
    }


def switch_target_curves_by_reward(
    pairs: pd.DataFrame, *, pool_trials: bool
) -> dict[str, dict[str, dict[str, Curve]] | None]:
    """:func:`switch_target_curves` for each row of :func:`split_by_previous_reward`.

    Returns:
        dict: ``{row: {target: {axis: Curve}}}``; ``None`` for a row without switches.
    """
    return {
        row: switch_target_curves(subset, pool_trials=pool_trials)
        if (subset["stay"] == 0).any()
        else None
        for row, subset in split_by_previous_reward(pairs).items()
    }
