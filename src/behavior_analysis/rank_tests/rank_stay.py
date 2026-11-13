"""Stay models: which alternative does staying depend on, and do ranks or values carry it?

All three analyses fit P(stay) on the decisions of :func:`rank_decisions.decision_table` with the
same history covariates (run length of the own arm, its choice trace, the own arm's earlier
choices, the plus-port bias) and a free intercept, so they differ only in the value terms.

1. :func:`stay_weights` (**stay depends on the second best, not the third**): own payout (pulses
   and a zero indicator), the value of the best alternative and of the worst alternative. A rank
   algorithm compares the own arm with the best alternative only: ``w_best < 0`` and
   ``w_worst`` about 0. A softmax over arm values lowers staying with both, the worst less.
2. :func:`value_or_rank` (**ranks matter more than values**): leave-one-problem-out held-out fit
   of the value terms entered as continuous values, as rank indicators of the own arm (first and
   second of three), or both.
3. :func:`scale_invariance`: P(stay) at a given rank of the own payout across the magnitudes it
   occurs at. A rank algorithm is invariant to the scale; a value learner stays more at 4 than at 2.
"""

import numpy as np
import pandas as pd

from src.behavior_analysis.rank_tests import rank_logit as rl
from src.behavior_analysis.rank_tests.rank_result import RankResult, curve_rows, empty_curves

HISTORY = ("run_log", "trace", "lag2", "lag3", "lag4", "plus_own")
VALUE_TERMS = ("payout", "own_zero", "v_best", "v_worst")
RANK_TERMS = ("own_first", "own_second")
# Magnitudes are centred here so that a rank indicator is the stay logit at a typical payout.
CENTER = 2.0


def stay_design(table: pd.DataFrame, terms) -> tuple[np.ndarray, tuple]:
    """``(X, names)``: intercept, the value *terms*, then the history covariates."""
    columns = [*terms, *HISTORY]
    X = np.column_stack([np.ones(len(table)), *[table[c].to_numpy(float) for c in columns]])
    return X, ("intercept", *columns)


def stay_outcome(table: pd.DataFrame) -> np.ndarray:
    """1 where the mouse stayed."""
    return 1.0 - table["leave"].to_numpy(float)


def stay_weights(table: pd.DataFrame, prefix: str = "") -> RankResult:
    """Own payout, best-alternative and worst-alternative weights of P(stay).

    Args:
        table (pd.DataFrame): ``rank_decisions.decision_table`` of one mouse.
        prefix (str): Put before every readout name (e.g. ``"q_"`` for another value source).

    Returns:
        RankResult: ``w_best, w_worst`` and their ``se_best, se_worst``, ``w_diff = w_worst -
        w_best`` with ``se_diff`` (Laplace, from the joint covariance) and ``w_ratio =
        w_worst / w_best`` (unstable when ``w_best`` is near 0: listed, never calibrated), and
        ``w_own`` (per pulse).

    Raises:
        ValueError: As :func:`rank_logit.fit_logit` (e.g. a mouse that never leaves).
    """
    X, names = stay_design(table, VALUE_TERMS)
    fit = rl.fit_logit(X, stay_outcome(table), names)
    best, worst = fit.weight("v_best"), fit.weight("v_worst")
    readouts = {
        "w_best": best,
        "w_worst": worst,
        "se_best": fit.weight_se("v_best"),
        "se_worst": fit.weight_se("v_worst"),
        "w_diff": worst - best,
        "se_diff": rl.contrast_se(fit, {"v_worst": 1.0, "v_best": -1.0}),
        "w_ratio": worst / best if best != 0 else np.nan,
        "w_own": fit.weight("payout"),
    }
    return RankResult({prefix + k: v for k, v in readouts.items()}, empty_curves())


def _nats(table: pd.DataFrame, terms) -> float:
    X, names = stay_design(table, terms)
    return rl.cross_validate(
        X, stay_outcome(table), table["problem"].to_numpy(), names
    ).nats_per_trial


def value_or_rank(table: pd.DataFrame) -> RankResult:
    """Held-out nats per decision of the value, rank and both models of P(stay), and their gains.

    Returns:
        RankResult: ``nats_value, nats_rank, nats_both`` (leave-one-problem-out), ``rank_over_value
        = nats_value - nats_both`` (what rank adds to value, positive if it helps),
        ``value_over_rank = nats_rank - nats_both`` and ``rank_minus_value = nats_value -
        nats_rank`` (positive if the rank model alone predicts better than the value model).

    Raises:
        ValueError: If the decisions come from fewer than two problems.
    """
    value, rank = _nats(table, VALUE_TERMS), _nats(table, RANK_TERMS)
    both = _nats(table, (*VALUE_TERMS, *RANK_TERMS))
    return RankResult(
        {
            "nats_value": value,
            "nats_rank": rank,
            "nats_both": both,
            "rank_over_value": value - both,
            "value_over_rank": rank - both,
            "rank_minus_value": value - rank,
        },
        empty_curves(),
    )


def scale_invariance(table: pd.DataFrame) -> RankResult:
    """P(stay) against the magnitude of the own payout, within the own arm's rank.

    Model: rank indicators of the own arm (first, second) and, within each, the magnitude of the
    payout (centred at ``CENTER`` pulses) with the history covariates. The rank slopes are the
    change of the stay logit per pulse at a fixed rank: 0 if only the rank matters. The curve is
    P(stay) by payout for the own arm ranked first and second.

    Returns:
        RankResult: ``scale_slope_first``, ``scale_slope_second`` (logit per pulse, NaN if the
        rank occurs at a single magnitude) with ``se_`` of each, and the stay curves.
    """
    first, second = table["own_first"].to_numpy(float), table["own_second"].to_numpy(float)
    magnitude = table["payout"].to_numpy(float) - CENTER
    extra = {
        "mag_first": first * magnitude,
        "mag_second": second * magnitude,
    }
    frame = table.assign(**extra)
    terms = ("own_first", "own_second", "mag_first", "mag_second")
    readouts = dict.fromkeys(
        ("scale_slope_first", "scale_slope_second", "se_scale_first", "se_scale_second"), np.nan
    )
    X, names = stay_design(frame, terms)
    fit = rl.fit_logit(X, stay_outcome(frame), names)
    for rank, name in (("first", "mag_first"), ("second", "mag_second")):
        varied = frame.loc[frame[f"own_{rank}"] == 1, "payout"].nunique() > 1
        if varied:
            readouts[f"scale_slope_{rank}"] = fit.weight(name)
            readouts[f"se_scale_{rank}"] = fit.weight_se(name)
    return RankResult(readouts, _scale_curves(table))


def _scale_curves(table: pd.DataFrame) -> pd.DataFrame:
    curves = []
    for rank in ("first", "second"):
        pure = table[table[f"own_{rank}"] == 1]
        grouped = pure.groupby("payout")["leave"].agg(["mean", "size"])
        if len(grouped):
            curves.append(
                curve_rows(
                    "scale", grouped.index.to_numpy(), 1 - grouped["mean"], grouped["size"], rank
                )
            )
    return pd.concat(curves, ignore_index=True) if curves else empty_curves()
