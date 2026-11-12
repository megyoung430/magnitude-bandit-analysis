"""Each mouse's readout against replays of a fitted agent on the mouse's own schedules.

A readout of a mouse (a weight, a held-out gain, a slope) is not interpretable alone: the task's
schedule, the mouse's trial counts and the analysis itself shape it. The comparator is the same
analysis of an agent fitted to that mouse and played on that mouse's problems, ``n`` times. For
each mouse and readout:

- ``z = (observed - null mean) / null sd`` and the two-sided empirical
  ``p = (1 + #{|null - null mean| >= |observed - null mean|}) / (n + 1)``, as the switch
  regression's calibration (:func:`src.behavior_analysis.switch_regression.null_calibration`);
- across mice, the mean ± se of z and its Wilcoxon test
  (:func:`src.behavior_analysis.choice_logit_extensions.gain_summary`);
- a mouse *passes* a readout when its p is below ``ALPHA`` and its z lies in the direction in which
  a rank algorithm differs from a softmax learner (:data:`rank_signatures.PREDICTED`).

Pure: tables in, tables out.
"""

import numpy as np
import pandas as pd

from src.behavior_analysis import choice_logit_extensions as ext

ALPHA = 0.05
KEY = ["mouse"]


def calibrate(observed: pd.DataFrame, null: pd.DataFrame) -> pd.DataFrame:
    """Every observed readout against its mouse's null draws.

    Args:
        observed (pd.DataFrame): ``mouse`` and one column per readout, one row per mouse.
        null (pd.DataFrame): ``mouse, replicate`` and the same readout columns, one row per
            replicate.

    Returns:
        pd.DataFrame: ``mouse, readout, observed, n_null, null_mean, null_sd, null_low, null_high``
        (2.5th and 97.5th percentiles), ``diff`` (``observed - null_mean``), ``z`` (NaN for fewer
        than two draws or a null without spread), ``percentile`` (ties half) and ``p_empirical``.
        An observation or a draw that is NaN is left out of its row's count; a readout with no
        draw gives NaN statistics.

    Raises:
        ValueError: If a mouse of *observed* has no replicate at all.
    """
    readouts = [c for c in observed.columns if c not in KEY]
    groups = dict(tuple(null.groupby("mouse", sort=False)))
    rows = []
    for _, mouse_row in observed.iterrows():
        mouse = mouse_row["mouse"]
        if mouse not in groups:
            raise ValueError(f"no null replicates for mouse {mouse}")
        mine = groups[mouse]
        for name in readouts:
            rows.append(_one(mouse, name, float(mouse_row[name]), mine[name].to_numpy(float)))
    return pd.DataFrame(rows)


def _one(mouse: str, name: str, value: float, draws: np.ndarray) -> dict:
    draws = draws[~np.isnan(draws)]
    row = dict(mouse=mouse, readout=name, observed=value, n_null=len(draws))
    if len(draws) < 2 or np.isnan(value):
        return row | dict.fromkeys(
            ["null_mean", "null_sd", "null_low", "null_high", "diff", "z", "percentile",
             "p_empirical"], np.nan)  # fmt: skip
    center, spread = draws.mean(), draws.std(ddof=1)
    extreme = np.abs(draws - center) >= abs(value - center)
    return row | dict(
        null_mean=center,
        null_sd=spread,
        null_low=float(np.percentile(draws, 2.5)),
        null_high=float(np.percentile(draws, 97.5)),
        diff=value - center,
        # a null that never varies gives no scale for a z (the p still holds)
        z=(value - center) / spread if spread > 0 else np.nan,
        percentile=float((draws < value).mean() + 0.5 * (draws == value).mean()),
        p_empirical=(1 + int(extreme.sum())) / (len(draws) + 1),
    )


def passes(calibration: pd.DataFrame, predicted: dict) -> pd.DataFrame:
    """Add ``predicted`` (+1 / -1 / NaN) and ``passes`` (p < ``ALPHA`` in the predicted direction).

    Args:
        calibration (pd.DataFrame): :func:`calibrate` output, possibly stacked over agents.
        predicted (dict): ``{readout: +1 or -1}``, the direction in which a rank algorithm lies
            above (+1) or below (-1) a softmax learner.
    """
    direction = calibration["readout"].map(predicted)
    in_direction = np.sign(calibration["diff"]) == direction
    return calibration.assign(
        predicted=direction, passes=in_direction & (calibration["p_empirical"] < ALPHA)
    )


def summarize(calibration: pd.DataFrame, predicted: dict, min_mice: int = ext.MIN_TEST_MICE):
    """Across mice, per readout: mean ± se of z, its Wilcoxon p, and which mice pass.

    Args:
        calibration (pd.DataFrame): :func:`calibrate` output of one agent.
        predicted (dict): As :func:`passes`.
        min_mice (int): With fewer mice the mean, se and tests are NaN. Default
            ``choice_logit_extensions.MIN_TEST_MICE``.

    Returns:
        pd.DataFrame: ``readout, n, mean, se, n_positive, p_wilcoxon, p_sign`` (of the z, as
        ``gain_summary``), ``n_pass`` (mice passing), ``passing`` (their names, comma-joined) and
        ``predicted``.
    """
    table = passes(calibration, predicted)
    rows = []
    for name, group in table.groupby("readout", sort=False):
        summary = ext.gain_summary(group, ["z"], min_mice).drop(columns="gain")
        summary.insert(0, "readout", name)
        summary["n_pass"] = int(group["passes"].sum())
        summary["passing"] = ", ".join(group.loc[group["passes"], "mouse"])
        summary["predicted"] = group["predicted"].iloc[0]
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)
