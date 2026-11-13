"""The third option: is the arm believed worst treated as a category, or as a low value?

Rank algorithms compare adjacent ranks, so the arm believed worst is chosen at a floor rate
whatever its gap to the second best (it is promoted only when the arm above it falls below it). A
softmax over values chooses it less the further it lies below the second best.

- **Choice of the worst** (:func:`worst_choice`): P(choose the believed-worst arm next) against the
  gap ``V(second) - V(worst)``, over the decisions with a unique worst arm (the gap is at least 1).
  A logistic on the gap, the own arm being the worst arm and the run length of the own arm; the
  slope is the readout. Rank: flat; softmax: negative.
- **Independence of irrelevant alternatives** (:func:`top_two_choice`): among the decisions in
  which the next choice is the believed best or second best (strict order of all three), does the
  value of the worst arm change which of the two is chosen? A softmax obeys independence of
  irrelevant alternatives, so its worst-arm weight is 0 as well; this is a check of the setup,
  not a test that separates the two.
"""

import numpy as np
import pandas as pd

from src.behavior_analysis.rank_tests import rank_logit as rl
from src.behavior_analysis.rank_tests import rank_state as rs
from src.behavior_analysis.rank_tests.rank_result import RankResult, curve_rows, empty_curves


def sorted_arms(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(arms, values)`` of each row from the highest to the lowest value (stable on ties)."""
    order = np.argsort(-values, axis=1, kind="stable")
    return order, np.take_along_axis(values, order, axis=1)


def _fit_columns(frame: pd.DataFrame, outcome: str, columns: tuple) -> rl.LogitFit:
    X = np.column_stack([np.ones(len(frame)), *[frame[c].to_numpy(float) for c in columns]])
    return rl.fit_logit(X, frame[outcome].to_numpy(float), ("intercept", *columns))


def worst_choice(state: rs.MouseState, decisions: pd.DataFrame) -> RankResult:
    """P(choose the believed-worst arm) against its gap to the second best.

    Args:
        state (rs.MouseState): The mouse's state.
        decisions (pd.DataFrame): ``rank_decisions.decision_table`` of the mouse.

    Returns:
        RankResult: ``worst_slope`` (logit per pulse of the gap) with ``se_worst_slope``, the
        pooled ``p_worst`` and ``n_worst`` (decisions used), and the curve ``third_option`` (P by
        gap).

    Raises:
        rank_logit.DegenerateOutcomeError: If the worst arm is never or always chosen.
    """
    rows = decisions["t"].to_numpy()
    arms, values = sorted_arms(state.believed[rows])
    unique = values[:, 2] < values[:, 1]
    frame = pd.DataFrame(
        dict(
            gap=(values[:, 1] - values[:, 2])[unique],
            chose_worst=(decisions["next"].to_numpy() == arms[:, 2])[unique].astype(float),
            worst_is_own=(decisions["own"].to_numpy() == arms[:, 2])[unique].astype(float),
            run_log=decisions["run_log"].to_numpy()[unique],
        )
    )
    fit = _fit_columns(frame, "chose_worst", ("gap", "worst_is_own", "run_log"))
    grouped = frame.groupby("gap")["chose_worst"].agg(["mean", "size"])
    readouts = {
        "worst_slope": fit.weight("gap"),
        "se_worst_slope": fit.weight_se("gap"),
        "p_worst": float(frame["chose_worst"].mean()),
        "n_worst": float(len(frame)),
    }
    curve = curve_rows("third_option", grouped.index.to_numpy(), grouped["mean"], grouped["size"])
    return RankResult(readouts, curve)


def top_two_choice(state: rs.MouseState, decisions: pd.DataFrame) -> RankResult:
    """Best against second best: does the worst arm's value matter?

    Over decisions with a strict order ``V(best) > V(second) > V(worst)`` whose next choice is
    the best or the second best, a logistic of choosing the best on ``V(best) - V(second)``,
    ``V(worst)``, whether the best / the second best is the arm just chosen, and the run length.

    Returns:
        RankResult: ``top2_w_worst`` and ``se_top2_worst`` (logit per pulse of the worst arm's
        value), ``top2_w_gap`` (per pulse of the best's lead over the second) and ``n_top2``.

    Raises:
        rank_logit.DegenerateOutcomeError: If the best (or second best) is always chosen.
    """
    rows = decisions["t"].to_numpy()
    arms, values = sorted_arms(state.believed[rows])
    next_arm = decisions["next"].to_numpy()
    ordered = (values[:, 0] > values[:, 1]) & (values[:, 1] > values[:, 2])
    top = ordered & ((next_arm == arms[:, 0]) | (next_arm == arms[:, 1]))
    own = decisions["own"].to_numpy()
    frame = pd.DataFrame(
        dict(
            chose_best=(next_arm == arms[:, 0])[top].astype(float),
            lead=(values[:, 0] - values[:, 1])[top],
            v_worst=values[:, 2][top],
            best_is_own=(own == arms[:, 0])[top].astype(float),
            second_is_own=(own == arms[:, 1])[top].astype(float),
            run_log=decisions["run_log"].to_numpy()[top],
        )
    )
    columns = ("lead", "v_worst", "best_is_own", "second_is_own", "run_log")
    fit = _fit_columns(frame, "chose_best", columns)
    readouts = {
        "top2_w_worst": fit.weight("v_worst"),
        "se_top2_worst": fit.weight_se("v_worst"),
        "top2_w_gap": fit.weight("lead"),
        "n_top2": float(len(frame)),
    }
    return RankResult(readouts, empty_curves())
