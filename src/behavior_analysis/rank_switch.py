"""Ranks or values in the switch regression: where does a switch go?

The switch regression (:mod:`src.behavior_analysis.switch_regression`) asks which of the two
alternatives a leaving mouse chooses, from choice history and one value per alternative. Here the
value term is entered three ways, on the switches after a non-zero payout whose state is defined
(all three arms visited), and compared by leave-one-problem-out held-out likelihood:

- **value**: the believed value of each alternative (pulses);
- **rank**: the weights of each alternative on being first and second of the three arms by
  believed value (:func:`src.behavior_analysis.rank_state.rank_weights`);
- **both**.

A value learner should prefer the value, a rank algorithm the ranks.
"""

import numpy as np

from src.behavior_analysis import rank_state as rs
from src.behavior_analysis import switch_regression as sr
from src.behavior_analysis.rank_result import RankResult, empty_curves

BELIEVED = sr.BELIEVED
RANK_FEATURES = ("rank_first", "rank_second")


def _nats(folds) -> float:
    return float(-folds["held_out_log_likelihood"].sum() / folds["n_switches"].sum())


def _check_alignment(data: sr.SwitchData, state: rs.MouseState) -> None:
    """The switch data and the state describe the same trials in the same order."""
    chosen = data.alternatives[np.arange(len(data.chosen)), data.chosen]
    if data.rows.max() >= len(state.chosen) or not np.array_equal(chosen, state.chosen[data.rows]):
        raise RuntimeError("the switch data and the believed state are not aligned")


def _defined_switches(data: sr.SwitchData, state: rs.MouseState) -> sr.SwitchData:
    """The after-non-zero switches whose state before the switch is defined."""
    data = sr.select_condition(data, sr.AFTER_NONZERO)
    keep = state.defined[data.rows - 1]
    return data._replace(
        X=data.X[keep],
        chosen=data.chosen[keep],
        values={k: v[keep] for k, v in data.values.items()},
        after_zero=data.after_zero[keep],
        problem=data.problem[keep],
        alternatives=data.alternatives[keep],
        rows=data.rows[keep],
    )


def _with_ranks(data: sr.SwitchData, state: rs.MouseState) -> sr.SwitchData:
    """*data* with each alternative's first and second rank weights as extra features."""
    weights = rs.rank_weights(rs.average_rank(state.believed[data.rows - 1]))
    alternatives = np.take_along_axis(weights, data.alternatives[:, :, None], axis=1)
    return data._replace(
        X=np.concatenate([data.X, alternatives[:, :, :2]], axis=2),
        feature_names=(*data.feature_names, *RANK_FEATURES),
    )


def switch_rank_or_value(trials, trace_decay: float = 0.9) -> RankResult:
    """Held-out nats per switch of the value, rank and both models of the switch regression.

    Args:
        trials (pd.DataFrame): One mouse's rows of the drifting trial table.
        trace_decay (float): Decay of the choice trace feature. Default 0.9, the null agents'.

    Returns:
        RankResult: ``sw_nats_history`` (no value term), ``sw_nats_value``, ``sw_nats_rank``,
        ``sw_nats_both``, ``sw_rank_over_value`` (``nats_value - nats_both``),
        ``sw_value_over_rank`` (``nats_rank - nats_both``) and ``sw_rank_minus_value``
        (``nats_value - nats_rank``; positive if ranks alone predict better) and ``sw_n``.

    Raises:
        switch_regression.TooFewSwitchesError: If the switches come from fewer than two problems.
        RuntimeError: If the switch data and the state disagree, or a fit does not converge.
    """
    state = rs.mouse_state(trials)
    data = sr.switch_data(trials, trace_decay=trace_decay)
    _check_alignment(data, state)
    data = _defined_switches(data, state)
    ranked = _with_ranks(data, state)
    nats = {
        "history": _nats(sr.cross_validate(data, None)),
        "value": _nats(sr.cross_validate(data, BELIEVED)),
        "rank": _nats(sr.cross_validate(ranked, None)),
        "both": _nats(sr.cross_validate(ranked, BELIEVED)),
    }
    readouts = {f"sw_nats_{k}": v for k, v in nats.items()}
    readouts |= {
        "sw_rank_over_value": nats["value"] - nats["both"],
        "sw_value_over_rank": nats["rank"] - nats["both"],
        "sw_rank_minus_value": nats["value"] - nats["rank"],
        "sw_n": float(len(data.chosen)),
    }
    return RankResult(readouts, empty_curves())
