"""Psychometric curves against a continuous value, with a nuisance baseline.

The value psychometrics of :mod:`src.behavior_analysis.value_psychometrics` count trials in a
grid of integer values (best 2-4, second 1-3). A model value (the reward part of the choice
logit, :mod:`src.behavior_analysis.choice_logit`) or a learned Q-value is continuous and in
units of its own, so the curves here are drawn against *quantile bins* of the value instead.

Bins
----
Each mouse's values are cut into ``n_bins`` bins of equal numbers of its trials
(:func:`quantile_bins`), separately for each x axis, because value scales differ between mice.
The bins are **rank-based**: values are ranked and the ranks cut into equal parts, tied values
being ordered at random (a seeded permutation). Cutting at value quantiles instead would leave
bins empty whenever many trials share a value, as exact zeros do. A bin therefore always holds
the same number of a mouse's trials (to within one), and a run of tied values is split between
adjacent bins. A curve's x is the mean value in the bin (mean over mice of each mouse's bin
mean), so it is in the units of the value, not a bin number; two bins made of the same value
have the same x. Values that are discrete by nature (task magnitudes, believed magnitudes) are
not binned: ``n_bins=None`` uses the exact values as x.

Mice are the replicates of every mean and standard error. A mean drawn from fewer than
``MIN_MICE`` mice is not drawn (with fewer mice than that in total, all are).

Baseline
--------
A value built from recent rewards is non-zero mainly for recently chosen arms, and a mouse that
repeats its choices picks them. Ranking arms by such a value therefore predicts choice even if
the mouse ignores rewards. Every curve can come with a *baseline*: the probability that a
model with choice history and tower bias but no reward term (fitted on the other problems,
:mod:`src.behavior_analysis.choice_logit_cv`) assigns to the same outcome on the same trials,
averaged in the same bins in the same way. The gap between observed and baseline is what the
value adds beyond perseveration and tower bias.

Arms are ranked by value on each trial. Tied values share the rank positions they cover evenly
(the "split" tie mode of the other modules), so ranks never depend on port names.

Stay/switch pairs follow :mod:`src.behavior_analysis.stay_switch_psychometrics`: on trial ``t``
whose predecessor ``t-1`` lies in the same session segment, the *chosen* arm is the one chosen
on ``t-1``, ``v_chosen`` its value on ``t`` and ``v_alt`` the higher value of the other two
arms. A continuous value has no "chose the 0-value option" row, so the rows are whether the
choice on ``t-1`` was rewarded (a non-zero magnitude) or not; bins are cut within each row.

Everything here is pure: arrays and tables in, :class:`Curve` objects and tables out.
"""

from dataclasses import replace

import numpy as np
import pandas as pd

from src.behavior_analysis.stats import nan_mean, nan_standard_error
from src.behavior_analysis.stay_switch_psychometrics import stay_switch_curves
from src.behavior_analysis.value_psychometrics import Curve

DEFAULT_BINS = 5
MIN_MICE = 3
CHOICES = ("best", "second")
X_AXES = ("best_val", "second_val", "diff")
REWARDED, UNREWARDED = "rewarded", "unrewarded"
REWARD_ROWS = (REWARDED, UNREWARDED)
N_ARMS = 3
# Seed of the random order of tied values within a bin cut.
TIE_SEED = 0
# Pair axes: x column of the curves, and the raw value it is a function of.
PAIR_AXES = {"chosen": "raw_chosen", "alt": "raw_alt", "diff": "gap"}
PAIR_BIN_COLUMNS = {"chosen": "v_chosen", "alt": "v_alt", "diff": "diff"}


def zero_fraction(values: np.ndarray) -> float:
    """Fraction of the (trial, arm) values that are exactly 0."""
    return float(np.mean(np.asarray(values) == 0))


# ---------------------------------------------------------------------------
# Ranks and bins
# ---------------------------------------------------------------------------


def _rank_shares(values: np.ndarray, position: int) -> np.ndarray:
    """``(n_trials, 3)`` share of rank *position* (0 best) held by each arm; ties split evenly."""
    above = (values[:, None, :] > values[:, :, None]).sum(axis=2)
    tied = (values[:, None, :] == values[:, :, None]).sum(axis=2)
    # the tied arms hold the rank positions above .. above + tied - 1
    return ((above <= position) & (position < above + tied)) / tied


def rank_table(
    mouse, values: np.ndarray, chosen: np.ndarray, probabilities: np.ndarray | None = None
) -> pd.DataFrame:
    """Each trial's values in rank order and the chosen arm's share of the top two ranks.

    Args:
        mouse (Sequence[str]): Mouse of each trial, ``(n_trials,)``.
        values (np.ndarray): ``(n_trials, 3)`` value of each arm.
        chosen (np.ndarray): ``(n_trials,)`` index of the chosen arm.
        probabilities (np.ndarray | None): ``(n_trials, 3)`` baseline choice probabilities of
            each arm (rows sum to 1). Default ``None``: no baseline columns.

    Returns:
        pd.DataFrame: ``mouse, v_best, v_second, v_third`` (the sorted values), ``gap``
        (best minus second) and ``best_share``, ``second_share``: the share of the best and
        second rank positions held by the chosen arm, 1 or 0 unless it ties with another arm.
        With *probabilities*, also ``best_baseline`` and ``second_baseline``: the probability
        the baseline gives to choosing the arm that holds each rank (shared likewise).

    Raises:
        ValueError: If the shapes disagree or *values* is not finite.
    """
    values = np.asarray(values, dtype=float)
    chosen = np.asarray(chosen)
    if values.ndim != 2 or values.shape[1] != N_ARMS or chosen.shape != (len(values),):
        raise ValueError(f"values must be (trials, 3) and chosen (trials,), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("values must be finite")
    rows = np.arange(len(values))
    best, second = _rank_shares(values, 0), _rank_shares(values, 1)
    ranked = -np.sort(-values, axis=1)
    table = pd.DataFrame(
        dict(
            mouse=np.asarray(mouse),
            v_best=ranked[:, 0],
            v_second=ranked[:, 1],
            v_third=ranked[:, 2],
            gap=ranked[:, 0] - ranked[:, 1],
            best_share=best[rows, chosen],
            second_share=second[rows, chosen],
        )
    )
    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.shape != values.shape:
            raise ValueError(f"probabilities {probabilities.shape} must match values")
        table["best_baseline"] = (probabilities * best).sum(axis=1)
        table["second_baseline"] = (probabilities * second).sum(axis=1)
    return table


def quantile_bins(x: np.ndarray, n_bins: int, seed: int = TIE_SEED) -> np.ndarray:
    """Bin number (1 to *n_bins*) of each value, the bins holding equal numbers of values.

    Values are ranked and the ranks cut into *n_bins* parts of equal size (to within one);
    equal values are ordered by a seeded random permutation, so a run of ties is split between
    adjacent bins and no bin is ever empty.

    Args:
        x (np.ndarray): Values of one mouse, ``(n,)``, at least *n_bins* of them.
        n_bins (int): Number of bins, at least 2.
        seed (int): Seed of the order of tied values. Default ``TIE_SEED``.

    Returns:
        np.ndarray: ``(n,)`` integers.

    Raises:
        ValueError: If ``n_bins < 2`` or there are fewer values than bins.
    """
    x = np.asarray(x, dtype=float)
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins}")
    if len(x) < n_bins:
        raise ValueError(f"{len(x)} values cannot fill {n_bins} bins")
    tie_break = np.random.default_rng(seed).permutation(len(x))
    order = np.lexsort((tie_break, x))
    rank = np.empty(len(x), dtype=int)
    rank[order] = np.arange(len(x))
    return rank * n_bins // len(x) + 1


def _binned_by_mouse(frame: pd.DataFrame, column: str, n_bins: int, seed: int) -> pd.Series:
    """*column* as bin numbers, cut separately for each mouse.

    A mouse with fewer values than bins gets NaN: it cannot fill the bins, and its rows are
    dropped by the caller.
    """

    def cut(values: pd.Series) -> pd.Series:
        if len(values) < n_bins:
            return pd.Series(np.nan, index=values.index)
        return pd.Series(quantile_bins(values.to_numpy(), n_bins, seed), index=values.index)

    return frame.groupby("mouse")[column].transform(cut)


def _axis_frame(table: pd.DataFrame, n_bins: int | None, seed: int) -> pd.DataFrame:
    """*table* with the x column of each axis: bin numbers, or the exact values if no bins.

    Rows of a mouse too short to fill the bins are dropped.
    """
    raw = {"best_val": "v_best", "second_val": "v_second", "diff": "gap"}
    if n_bins is None:
        return table.assign(**{axis: table[col].round(6) for axis, col in raw.items()})
    binned = table.assign(
        **{axis: _binned_by_mouse(table, col, n_bins, seed) for axis, col in raw.items()}
    )
    return binned.dropna(subset=list(raw))


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------


def _curve(frame: pd.DataFrame, outcome: str, axis: str) -> Curve:
    """Mean of *outcome* per mouse against the x values (or bins) of *axis*, pooled in a mouse."""
    x = np.sort(frame[axis].unique())
    mice = sorted(frame["mouse"].unique())
    per_mouse = (
        frame.groupby(["mouse", axis])[outcome]
        .mean()
        .unstack(axis)
        .reindex(index=mice, columns=x)
        .to_numpy(dtype=float)
    )
    return Curve(
        x=x,
        mice=mice,
        per_mouse=per_mouse,
        mean=nan_mean(per_mouse, axis=0),
        se=nan_standard_error(per_mouse),
        n_mice=np.sum(~np.isnan(per_mouse), axis=0),
        n_trials=frame.groupby(axis).size().reindex(x).to_numpy(),
    )


def _bin_means(frame: pd.DataFrame, bin_column: str, raw_column: str) -> pd.Series:
    """Mean over mice of each mouse's mean raw value in each bin, indexed by bin."""
    per_mouse = frame.groupby(["mouse", bin_column])[raw_column].mean()
    return per_mouse.groupby(level=bin_column).mean()


def _positioned(curve: Curve, means: pd.Series | None) -> Curve:
    """*curve* with its x at the bin means (unchanged if there are none, e.g. exact values)."""
    if means is None:
        return curve
    return replace(curve, x=means.reindex(curve.x).to_numpy(dtype=float))


def _masked(curve: Curve, keep: np.ndarray) -> Curve:
    """*curve* with the mean and se of the x values outside *keep* set to NaN."""
    return replace(
        curve, mean=np.where(keep, curve.mean, np.nan), se=np.where(keep, curve.se, np.nan)
    )


def _enough_mice(observed: Curve, min_mice: int) -> np.ndarray:
    """Where *observed* rests on at least *min_mice* mice (or on all mice, if fewer)."""
    return observed.n_mice >= min(min_mice, len(observed.mice))


def psychometric_curves_from_table(
    table: pd.DataFrame,
    n_bins: int | None = DEFAULT_BINS,
    min_mice: int = MIN_MICE,
    seed: int = TIE_SEED,
) -> tuple[dict, dict | None]:
    """P(best) and P(second) against the best, second and best - second value, and the baseline.

    Args:
        table (pd.DataFrame): :func:`rank_table` of one or more mice (all trials of a mouse).
        n_bins (int | None): Quantile bins per axis and mouse; ``None`` uses the exact values
            (for discrete values such as magnitudes). Default ``DEFAULT_BINS``.
        min_mice (int): Means from fewer mice are not drawn. Default ``MIN_MICE``.
        seed (int): Seed of the order of tied values in the bin cuts.

    Returns:
        tuple[dict, dict | None]: ``(curves, baselines)``, each ``[choice][axis]`` for the
        choices ``"best"``, ``"second"`` and axes ``"best_val"``, ``"second_val"``, ``"diff"``;
        ``baselines`` has the same curves of the baseline probabilities (same trials, bins and
        weighting) and is ``None`` if *table* has no baseline columns. ``Curve.x`` is the mean
        value in each bin (the exact value with ``n_bins=None``).

    Raises:
        ValueError: If *table* is empty or ``n_bins`` is invalid.
    """
    if table.empty:
        raise ValueError("no trials to make psychometric curves from")
    frame = _axis_frame(table, n_bins, seed)
    raw = {"best_val": "v_best", "second_val": "v_second", "diff": "gap"}
    means = {a: _bin_means(frame, a, raw[a]) if n_bins else None for a in X_AXES}
    with_baseline = "best_baseline" in table
    curves: dict = {c: {} for c in CHOICES}
    baselines: dict = {c: {} for c in CHOICES}
    for choice in CHOICES:
        for axis in X_AXES:
            observed = _curve(frame, f"{choice}_share", axis)
            keep = _enough_mice(observed, min_mice)
            curves[choice][axis] = _positioned(_masked(observed, keep), means[axis])
            if with_baseline:
                base = _curve(frame, f"{choice}_baseline", axis)
                baselines[choice][axis] = _positioned(_masked(base, keep), means[axis])
    return curves, (baselines if with_baseline else None)


def value_effect(curves: dict, baselines: dict, axis: str = "best_val") -> float:
    """What the value adds beyond the baseline: the rise of P(best) over observed - baseline.

    For each mouse, the observed minus baseline P(best) in the highest bin minus the same in the
    lowest bin of *axis*; the mean over mice. A value that only mirrors the nuisance terms gives
    about 0; a value that guides choice gives a positive number (P(best) rises more than the
    baseline says).

    Args:
        curves (dict): Observed curves of :func:`psychometric_curves_from_table`.
        baselines (dict): Its baseline curves.
        axis (str): X axis. Default ``"best_val"``.

    Returns:
        float: The mean over mice (NaN if no mouse has both end bins).
    """
    gap = curves["best"][axis].per_mouse - baselines["best"][axis].per_mouse
    rise = gap[:, -1] - gap[:, 0]
    return float(np.nanmean(rise)) if np.isfinite(rise).any() else float("nan")


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
    meta: pd.DataFrame,
    values: np.ndarray,
    chosen: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> pd.DataFrame:
    """Trial pairs ``(t-1, t)`` with the values of trial ``t`` and the stay/switch outcome.

    Args:
        meta (pd.DataFrame): One row per trial, in trial order within each mouse's problem,
            with the ``PAIR_META`` columns (``LogitDesign.meta`` of the mice, concatenated).
        values (np.ndarray): ``(n_trials, 3)`` value of each arm on each trial; a pair whose
            trial ``t`` has a non-finite value (an arm without a value yet) is left out.
        chosen (np.ndarray): ``(n_trials,)`` index of the chosen arm.
        probabilities (np.ndarray | None): ``(n_trials, 3)`` baseline choice probabilities.
            Default ``None``: no baseline columns.

    Returns:
        pd.DataFrame: One row per pair with ``mouse, problem, trial``; the raw values
        ``raw_chosen, raw_alt, v_other`` and ``gap`` (``raw_chosen - raw_alt``); the outcomes
        ``stay, switch_best_alt, switch_other`` (a switch between two alternatives of equal value
        counts half to each); and ``rewarded``: whether trial ``t-1`` paid a non-zero magnitude.
        With *probabilities*: ``base_stay``, and ``base_best_alt`` / ``base_other``, the
        baseline's probability of each switch target *given a switch*.

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
    now = np.flatnonzero(_pair_mask(meta) & np.isfinite(values).all(axis=1))
    previous_arm, current_arm = chosen[now - 1], chosen[now]
    others = (previous_arm[:, None] + np.array([1, 2])) % N_ARMS
    pick = np.take_along_axis(values[now], others, axis=1).argmax(axis=1)[:, None]
    alt_arm = np.take_along_axis(others, pick, axis=1)[:, 0]
    other_values = np.take_along_axis(values[now], others, axis=1)
    raw_alt, v_other = other_values.max(axis=1), other_values.min(axis=1)
    raw_chosen = values[now, previous_arm]
    stay = (current_arm == previous_arm).astype(float)
    switch = 1.0 - stay
    tie = raw_alt == v_other
    best_alt = np.where(tie, switch / 2, (current_arm == alt_arm).astype(float))
    pairs = pd.DataFrame(
        dict(
            mouse=meta["mouse"].to_numpy()[now],
            problem=meta["problem"].to_numpy()[now],
            trial=meta["trial"].to_numpy()[now],
            raw_chosen=raw_chosen,
            raw_alt=raw_alt,
            v_other=v_other,
            gap=raw_chosen - raw_alt,
            stay=stay,
            switch_best_alt=best_alt,
            switch_other=switch - best_alt,
            rewarded=meta["reward"].to_numpy()[now - 1] > 0,
        )
    )
    if probabilities is not None:
        p = np.asarray(probabilities, dtype=float)[now]
        p_alt = p[np.arange(len(now)), alt_arm]
        p_rest = 1.0 - p[np.arange(len(now)), previous_arm] - p_alt
        share_alt = np.where(tie, 0.5, p_alt / (p_alt + p_rest))
        pairs["base_stay"] = p[np.arange(len(now)), previous_arm]
        pairs["base_best_alt"] = share_alt
        pairs["base_other"] = 1.0 - share_alt
    return pairs


def split_by_previous_reward(pairs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The pairs whose previous choice was rewarded, and those whose previous choice was not.

    Returns:
        dict[str, pd.DataFrame]: ``{REWARDED: ..., UNREWARDED: ...}``, either may be empty.
    """
    return {
        REWARDED: pairs.loc[pairs["rewarded"]].reset_index(drop=True),
        UNREWARDED: pairs.loc[~pairs["rewarded"]].reset_index(drop=True),
    }


def _axis_columns(pairs: pd.DataFrame, n_bins: int | None, seed: int) -> pd.DataFrame:
    """*pairs* with ``v_chosen``, ``v_alt`` and ``diff`` as bin numbers, or the exact values."""
    if n_bins is None:
        return pairs.assign(
            **{PAIR_BIN_COLUMNS[a]: pairs[raw].round(6) for a, raw in PAIR_AXES.items()}
        )
    binned = pairs.assign(
        **{
            PAIR_BIN_COLUMNS[a]: _binned_by_mouse(pairs, raw, n_bins, seed)
            for a, raw in PAIR_AXES.items()
        }
    )
    return binned.dropna(subset=list(PAIR_BIN_COLUMNS.values()))


def _row_curves(subset, baseline_outcomes, n_bins, pool, min_mice, seed):
    """``(curves, baselines)`` of the stay/switch outcomes of one row of pairs."""
    frame = _axis_columns(subset, n_bins, seed)
    observed = stay_switch_curves(frame, pool_trials=pool)
    baseline = (
        stay_switch_curves(frame.assign(**baseline_outcomes), pool_trials=pool)
        if baseline_outcomes
        else None
    )
    means = {
        axis: _bin_means(frame, PAIR_BIN_COLUMNS[axis], raw) if n_bins else None
        for axis, raw in PAIR_AXES.items()
    }
    curves, baselines = {}, ({} if baseline else None)
    for outcome, by_axis in observed.items():
        curves[outcome] = {}
        if baseline:
            baselines[outcome] = {}
        for axis, curve in by_axis.items():
            keep = _enough_mice(curve, min_mice)
            curves[outcome][axis] = _positioned(_masked(curve, keep), means[axis])
            if baseline:
                baselines[outcome][axis] = _positioned(
                    _masked(baseline[outcome][axis], keep), means[axis]
                )
    return curves, baselines


def stay_switch_curves_by_reward(
    pairs: pd.DataFrame,
    *,
    pool_trials: bool,
    n_bins: int | None = DEFAULT_BINS,
    min_mice: int = MIN_MICE,
    seed: int = TIE_SEED,
) -> tuple[dict, dict | None]:
    """P(stay) and P(switch) for each row of :func:`split_by_previous_reward`, and the baseline.

    Bins are cut within each row, so every row's bins hold equal numbers of that row's pairs.

    Args:
        pairs (pd.DataFrame): :func:`stay_switch_pairs_from_values` output.
        pool_trials (bool): As in ``stay_switch_curves``. Keyword-only.
        n_bins (int | None): Quantile bins per axis, mouse and row; ``None`` uses exact values.
            Default ``DEFAULT_BINS``. Keyword-only.
        min_mice (int): Means from fewer mice are not drawn. Keyword-only.
        seed (int): Seed of the order of tied values. Keyword-only.

    Returns:
        tuple[dict, dict | None]: ``(curves_by_row, baselines_by_row)``, each ``{row: {outcome:
        {axis: Curve}}}``; ``None`` for a row without pairs, and ``baselines_by_row`` is ``None``
        without baseline columns. ``Curve.x`` is the mean value in the bin.
    """
    with_baseline = "base_stay" in pairs
    base = (lambda subset: {"stay": subset["base_stay"]}) if with_baseline else (lambda _: None)
    curves_by_row, baselines_by_row = {}, {}
    for row, subset in split_by_previous_reward(pairs).items():
        if subset.empty:
            curves_by_row[row], baselines_by_row[row] = None, None
            continue
        curves_by_row[row], baselines_by_row[row] = _row_curves(
            subset, base(subset), n_bins, pool_trials, min_mice, seed
        )
    return curves_by_row, (baselines_by_row if with_baseline else None)


# ---------------------------------------------------------------------------
# Switch targets
# ---------------------------------------------------------------------------

TARGET_GAP = "alt_gap"


def _untied_switches(subset: pd.DataFrame) -> pd.DataFrame:
    """The switches of *subset* whose two unchosen arms differ in value (a best alternative)."""
    return subset.loc[(subset["stay"] == 0) & (subset["raw_alt"] > subset["v_other"])]


def tied_switches_by_reward(pairs: pd.DataFrame) -> dict[str, int]:
    """Switches between two unchosen arms of equal value, in each row of the reward split.

    These have no best alternative and are left out of the switch-target curve.

    Returns:
        dict[str, int]: ``{row: count}`` for each of ``REWARD_ROWS``.
    """
    return {
        row: int(((s["stay"] == 0) & (s["raw_alt"] == s["v_other"])).sum())
        for row, s in split_by_previous_reward(pairs).items()
    }


def _target_frame(switches: pd.DataFrame, n_bins: int | None, seed: int) -> pd.DataFrame:
    """*switches* with ``alt_gap`` as bin numbers (or exact gaps) and the raw gap in ``raw_gap``.

    A mouse with fewer switches than bins is left out.
    """
    frame = switches.assign(raw_gap=switches["raw_alt"] - switches["v_other"])
    if n_bins is None:
        return frame.assign(**{TARGET_GAP: frame["raw_gap"].round(6)})
    binned = frame.assign(**{TARGET_GAP: _binned_by_mouse(frame, "raw_gap", n_bins, seed)})
    return binned.dropna(subset=[TARGET_GAP])


def switch_target_curves_by_reward(
    pairs: pd.DataFrame,
    *,
    n_bins: int | None = DEFAULT_BINS,
    min_mice: int = MIN_MICE,
    seed: int = TIE_SEED,
) -> tuple[dict, dict | None]:
    """P(best alternative | switch) against the gap between the unchosen arms, per reward row.

    Only the switches count, and only those whose two unchosen arms differ in value (ties have
    no best alternative; :func:`tied_switches_by_reward` counts them). The x is
    ``V(best alternative) - V(other arm)``, cut into rank bins within each mouse and row and
    placed at the bin-mean gap. Switches are pooled within a mouse. The baseline is the
    nuisance model's probability of the best alternative given a switch, averaged in the same
    bins.

    Args:
        pairs (pd.DataFrame): :func:`stay_switch_pairs_from_values` output.
        n_bins (int | None): Bins per mouse and row; ``None`` uses the exact gaps (for discrete
            values). Default ``DEFAULT_BINS``. Keyword-only.
        min_mice (int): Means from fewer mice are not drawn. Keyword-only.
        seed (int): Seed of the order of tied gaps in the bin cuts. Keyword-only.

    Returns:
        tuple[dict, dict | None]: ``(curves_by_row, baselines_by_row)``, each ``{row: Curve |
        None}`` for the rows of ``REWARD_ROWS``; ``None`` for a row without untied switches,
        and ``baselines_by_row`` is ``None`` without the ``base_best_alt`` column.
        ``Curve.n_trials`` counts switches.
    """
    with_baseline = "base_best_alt" in pairs
    curves, baselines = {}, {}
    for row, subset in split_by_previous_reward(pairs).items():
        frame = _target_frame(_untied_switches(subset), n_bins, seed)
        if frame.empty:
            curves[row], baselines[row] = None, None
            continue
        observed = _curve(frame, "switch_best_alt", TARGET_GAP)
        keep = _enough_mice(observed, min_mice)
        means = _bin_means(frame, TARGET_GAP, "raw_gap") if n_bins else None
        curves[row] = _positioned(_masked(observed, keep), means)
        if with_baseline:
            base = _curve(frame, "base_best_alt", TARGET_GAP)
            baselines[row] = _positioned(_masked(base, keep), means)
    return curves, (baselines if with_baseline else None)
