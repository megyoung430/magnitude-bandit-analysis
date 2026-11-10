"""Where does a mouse go when it leaves an arm: to the better alternative, or to the second best?

The switch trials, their choice sets (the two arms other than the previous choice), the believed
values and the after-zero / after-non-zero split are those of
:mod:`src.behavior_analysis.switch_regression`; this module only ranks the two alternatives and
asks whether the mouse chose the higher one. Three measures, each a proportion of switches:

- ``second_best``: *leaving the best.* Switches whose previous choice was the arm with the
  highest true magnitude on the previous trial (a strict maximum). The outcome is whether the
  mouse chose the better of the two alternatives, ranked on that same trial, i.e. the second
  best rather than the third. A tied top (the previous choice shares the maximum) and tied
  alternatives are excluded and counted. The ranking is that of the previous trial, the one the
  mouse's last payout came from; the magnitudes of the switch trial can differ by a drift step or
  a block change. In this task the reward of a choice is the arm's magnitude and exactly one arm
  pays 0, so leaving the best never follows a zero payout and the after-zero cell is empty.
- ``believed_better``: switches on which both alternatives have been chosen before in the
  problem and their believed values (the magnitude each last paid) differ; the outcome is whether
  the mouse chose the believed-better one. Switches with a never-chosen alternative, and ties,
  are excluded and counted.
- ``true_better``: switches on which the alternatives' magnitudes on the switch trial differ; the
  outcome is whether the mouse chose the truly better one (the true-value convention of the
  switch regression). Ties are excluded and counted.

Each is reported after zero, after non-zero and pooled. 0.5 is not the null of any of them: the
alternative chosen most recently tends to be the better one, so each proportion is read against
the same statistic of the mouse's fitted win-stay/lose-shift agents (:func:`calibrate`).

The graded version bins the used switches by the magnitude difference of the two alternatives
(1 to 4 pulses) and gives the proportion that chose the higher one in each bin.

Everything here is pure: trial tables in, tables out.
"""

import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.behavior_analysis import choice_logit as cl
from src.behavior_analysis import choice_logit_extensions as ext
from src.behavior_analysis import switch_regression as sr
from src.behavior_analysis.stats import nan_mean, nan_standard_error

SECOND_BEST, BELIEVED_BETTER, TRUE_BETTER = "second_best", "believed_better", "true_better"
MEASURES = (SECOND_BEST, BELIEVED_BETTER, TRUE_BETTER)
GRADED_MEASURES = (BELIEVED_BETTER, TRUE_BETTER)
POOLED = "pooled"
ROW_CONDITIONS = (*sr.CONDITIONS, POOLED)
# Magnitude differences between the two alternatives: magnitudes are 0-4 pulses.
DIFFERENCES = (1, 2, 3, 4)
# Used switches a mouse needs in a cell to be reported: the binomial se of a proportion from 30
# switches is up to 0.09, which is the least that still says something about a mouse.
MIN_USED = 30

PROPORTION_COLUMNS = ["mouse", "measure", "condition", "n_switches", "n_used", "n_success",
                      "n_tied", "n_unvisited", "p"]  # fmt: skip
GRADED_COLUMNS = ["mouse", "measure", "condition", "difference", "n_used", "n_success", "p"]


class SwitchContext(NamedTuple):
    """The switch trials of one mouse with what the ranking needs.

    Attributes:
        data (sr.SwitchData): The switch trials (``trace_decay`` plays no part here).
        previous_magnitudes (np.ndarray): ``(n_switches, 3)`` magnitudes of the three arms on
            the trial before each switch, in sorted port order.
        previous_choice (np.ndarray): ``(n_switches,)`` arm chosen on the trial before.
    """

    data: sr.SwitchData
    previous_magnitudes: np.ndarray
    previous_choice: np.ndarray


class Outcome(NamedTuple):
    """Per switch trial, how a measure treats it; all arrays are ``(n_switches,)``.

    Attributes:
        used (np.ndarray): Bool; the switch counts in the proportion.
        success (np.ndarray): Bool; used, and the better alternative was chosen.
        tied (np.ndarray): Bool; excluded for a tie.
        unvisited (np.ndarray): Bool; excluded because an alternative was never chosen before.
        difference (np.ndarray): Absolute magnitude difference of the alternatives for used
            switches, 0 elsewhere.
    """

    used: np.ndarray
    success: np.ndarray
    tied: np.ndarray
    unvisited: np.ndarray
    difference: np.ndarray


class MouseSecondBest(NamedTuple):
    """The three measures of one mouse.

    Attributes:
        proportions (pd.DataFrame): ``PROPORTION_COLUMNS``, one row per measure and condition
            (after zero, after non-zero, pooled); ``p`` is NaN without used switches.
        graded (pd.DataFrame): ``GRADED_COLUMNS``, one row per graded measure, condition and
            magnitude difference.
    """

    proportions: pd.DataFrame
    graded: pd.DataFrame


# ---------------------------------------------------------------------------
# Switch trials and rankings
# ---------------------------------------------------------------------------


def _magnitudes(design: cl.LogitDesign) -> np.ndarray:
    """``(n_trials, 3)`` magnitudes offered on every trial, in the design's port order."""
    return np.array(
        [[m[p] for p in ports] for m, ports in zip(design.meta["magnitudes"], design.ports,
                                                    strict=True)],
        dtype=float,
    )  # fmt: skip


def switch_context(trials: pd.DataFrame, n_lags: int = cl.DEFAULT_LAGS) -> SwitchContext:
    """The switch trials of one mouse with the previous trial's magnitudes and choice.

    Args:
        trials (pd.DataFrame): One mouse's rows of the drifting trial table.
        n_lags (int): Lags of the switch regression's design. Default
            ``choice_logit.DEFAULT_LAGS``.

    Returns:
        SwitchContext: The switches of :func:`switch_regression.switch_data` and, for each, the
        magnitudes and the choice of the trial before.

    Raises:
        TooFewSwitchesError: If the mouse never switches.
        ValueError: As :func:`switch_regression.switch_data`.
    """
    data = sr.switch_data(trials, n_lags, trace_decay=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # switch_data raised the design's warnings already
        design = cl.design_matrix(trials, 1)
    if not np.array_equal(design.meta["problem"].to_numpy()[data.rows], data.problem):
        raise RuntimeError("the switch trials and the design have different rows")
    # a switch is never a segment's first trial, so the row before it is the previous trial
    return SwitchContext(
        data=data,
        previous_magnitudes=_magnitudes(design)[data.rows - 1],
        previous_choice=design.chosen[data.rows - 1],
    )


def _compare_alternatives(values: np.ndarray, chosen: np.ndarray, available: np.ndarray):
    """``(used, success, tied, difference)`` of the higher of two alternatives' *values*.

    *available* marks the switches that may count; of those, equal values are ties.
    """
    tied = available & (values[:, 0] == values[:, 1])
    used = available & ~tied
    success = used & (chosen == np.argmax(values, axis=1))
    difference = np.where(used, np.abs(values[:, 0] - values[:, 1]), 0).astype(int)
    return used, success, tied, difference


def leave_best_outcome(context: SwitchContext) -> Outcome:
    """Switches whose previous choice was the strict best, and whether the second best followed."""
    mags, rows = context.previous_magnitudes, np.arange(len(context.previous_choice))
    top = mags.max(axis=1)
    at_top = mags[rows, context.previous_choice] == top
    n_top = (mags == top[:, None]).sum(axis=1)
    alternatives = np.take_along_axis(mags, context.data.alternatives, axis=1)
    used, success, tied, difference = _compare_alternatives(
        alternatives, context.data.chosen, at_top & (n_top == 1)
    )
    return Outcome(used, success, tied | (at_top & (n_top > 1)), np.zeros_like(used), difference)


def believed_outcome(context: SwitchContext) -> Outcome:
    """Switches with two visited, believed-different alternatives, and whether the better won."""
    data = context.data
    never = data.X[:, :, data.feature_names.index(sr.NEVER)] > 0
    unvisited = never.any(axis=1)
    used, success, tied, difference = _compare_alternatives(
        data.values[sr.BELIEVED], data.chosen, ~unvisited
    )
    return Outcome(used, success, tied, unvisited, difference)


def true_outcome(context: SwitchContext) -> Outcome:
    """Switches with two alternatives of different magnitude, and whether the better won."""
    data = context.data
    used, success, tied, difference = _compare_alternatives(
        data.values[sr.TRUE], data.chosen, np.ones(len(data.chosen), dtype=bool)
    )
    return Outcome(used, success, tied, np.zeros_like(used), difference)


def outcomes(context: SwitchContext) -> dict[str, Outcome]:
    """``{measure: Outcome}`` for every measure in ``MEASURES``."""
    return {
        SECOND_BEST: leave_best_outcome(context),
        BELIEVED_BETTER: believed_outcome(context),
        TRUE_BETTER: true_outcome(context),
    }


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _condition_masks(after_zero: np.ndarray) -> dict[str, np.ndarray]:
    return {
        sr.AFTER_ZERO: after_zero,
        sr.AFTER_NONZERO: ~after_zero,
        POOLED: np.ones_like(after_zero),
    }


def _proportion(n_success: int, n_used: int) -> float:
    return n_success / n_used if n_used else np.nan


def _proportion_rows(mouse: str, outcome_by_measure: dict, masks: dict) -> list[dict]:
    rows = []
    for measure, outcome in outcome_by_measure.items():
        for condition, mask in masks.items():
            n_used, n_success = (
                int((outcome.used & mask).sum()),
                int((outcome.success & mask).sum()),
            )
            rows.append(
                dict(mouse=mouse, measure=measure, condition=condition,
                     n_switches=int(mask.sum()), n_used=n_used, n_success=n_success,
                     n_tied=int((outcome.tied & mask).sum()),
                     n_unvisited=int((outcome.unvisited & mask).sum()),
                     p=_proportion(n_success, n_used))
            )  # fmt: skip
    return rows


def _graded_rows(mouse: str, outcome_by_measure: dict, masks: dict) -> list[dict]:
    rows = []
    for measure in GRADED_MEASURES:
        outcome = outcome_by_measure[measure]
        largest = int(outcome.difference.max(initial=0))
        if largest > DIFFERENCES[-1]:
            raise ValueError(
                f"mouse {mouse}: {measure} magnitude difference of {largest} pulses between the "
                f"alternatives; magnitudes are expected in 0-{DIFFERENCES[-1]} pulses"
            )
        for condition, mask in masks.items():
            for difference in DIFFERENCES:
                here = outcome.used & mask & (outcome.difference == difference)
                n_used, n_success = int(here.sum()), int((outcome.success & here).sum())
                rows.append(
                    dict(mouse=mouse, measure=measure, condition=condition,
                         difference=difference, n_used=n_used, n_success=n_success,
                         p=_proportion(n_success, n_used))
                )  # fmt: skip
    return rows


def analyze_mouse(trials: pd.DataFrame, n_lags: int = cl.DEFAULT_LAGS) -> MouseSecondBest:
    """The three measures and the graded curves of one mouse.

    Args:
        trials (pd.DataFrame): One mouse's rows of the drifting trial table.
        n_lags (int): Lags of the switch regression's design. Default
            ``choice_logit.DEFAULT_LAGS``.

    Returns:
        MouseSecondBest: Proportions and graded curves for after zero, after non-zero and pooled.

    Raises:
        TooFewSwitchesError: If the mouse never switches.
        ValueError: If the alternatives differ by more than 4 pulses, or as :func:`switch_context`.
    """
    context = switch_context(trials, n_lags)
    mouse = context.data.mouse
    by_measure = outcomes(context)
    masks = _condition_masks(context.data.after_zero)
    return MouseSecondBest(
        proportions=pd.DataFrame(_proportion_rows(mouse, by_measure, masks))[PROPORTION_COLUMNS],
        graded=pd.DataFrame(_graded_rows(mouse, by_measure, masks))[GRADED_COLUMNS],
    )


def apply_minimum(
    table: pd.DataFrame, min_used: int = MIN_USED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a proportions table into the cells with at least *min_used* used switches and the rest.

    Args:
        table (pd.DataFrame): ``MouseSecondBest.proportions`` of one or more mice.
        min_used (int): Used switches a cell needs. Default ``MIN_USED``.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The cells kept, and the cells dropped (to be named in
        a warning and in the figure).
    """
    enough = table["n_used"] >= min_used
    return table[enough].reset_index(drop=True), table[~enough].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Across mice and against a null
# ---------------------------------------------------------------------------


def group_summary(table: pd.DataFrame, min_mice: int = ext.MIN_TEST_MICE) -> pd.DataFrame:
    """Across mice: mean ± se of each proportion, from *min_mice* mice.

    No test: the proportion is not centered on 0.5 under the null, so the test of a source is the
    one of the calibrated z (:func:`calibrated_summary`).

    Args:
        table (pd.DataFrame): Proportions of one source, one row per mouse, measure and condition.
        min_mice (int): Fewer mice give NaN mean and se. Default ``MIN_TEST_MICE``.

    Returns:
        pd.DataFrame: ``measure, condition, n, mean, se``.
    """
    rows = []
    for (measure, condition), group in table.groupby(["measure", "condition"], sort=False):
        values = group["p"].dropna().to_numpy(dtype=float)
        enough = len(values) >= min_mice
        rows.append(
            dict(measure=measure, condition=condition, n=len(values),
                 mean=float(nan_mean(values[:, None])[0]) if enough else np.nan,
                 se=float(nan_standard_error(values[:, None])[0]) if enough else np.nan)
        )  # fmt: skip
    return pd.DataFrame(rows, columns=["measure", "condition", "n", "mean", "se"])


def _as_value_table(table: pd.DataFrame) -> pd.DataFrame:
    """A proportions table in the schema of ``switch_regression.value_table``.

    The null calibration there is written for a value weight (``beta``) and a gain; the
    proportion is passed as the weight and the gain is a copy that is dropped afterwards.
    """
    return table.rename(columns={"measure": "value"}).assign(beta=table["p"], gain=table["p"])


def calibrate(observed: pd.DataFrame, null: pd.DataFrame) -> pd.DataFrame:
    """Each observed proportion against its mouse's own null distribution.

    Args:
        observed (pd.DataFrame): ``mouse, measure, condition, p``, one row per cell.
        null (pd.DataFrame): The same columns plus ``replicate``, one row per replicate of the
            mouse's null agent; replicates without a proportion (no used switch) are ignored.

    Returns:
        pd.DataFrame: ``mouse, measure, condition, observed, n_null, null_mean, null_sd,
        null_low, null_high, z, percentile, p_empirical`` as
        :func:`switch_regression.null_calibration`.

    Raises:
        ValueError: If an observation has no null replicate.
    """
    keep = ["mouse", "condition", "value", "beta", "gain"]
    calibration = sr.null_calibration(
        _as_value_table(observed)[keep], _as_value_table(null.dropna(subset=["p"]))
    )
    calibration = calibration[calibration["term"] == "beta"].drop(columns="term")
    return calibration.rename(columns={"value": "measure"}).reset_index(drop=True)


def calibrated_summary(calibration: pd.DataFrame, min_mice: int = ext.MIN_TEST_MICE):
    """Across mice: mean ± se and the Wilcoxon and sign tests of the calibrated z.

    Args:
        calibration (pd.DataFrame): :func:`calibrate` of the mice.
        min_mice (int): Fewer mice give NaN mean, se and tests. Default ``MIN_TEST_MICE``.

    Returns:
        pd.DataFrame: ``measure, condition, n, mean, se, n_positive, p_wilcoxon, p_sign``.
    """
    summary = sr.calibrated_summary(
        calibration.rename(columns={"measure": "value"}).assign(term="p"), min_mice
    )
    columns = ["measure", "condition", "n", "mean", "se", "n_positive", "p_wilcoxon", "p_sign"]
    return summary.rename(columns={"value": "measure"})[columns]
