"""The stay/leave decisions of one mouse with the believed state each was made in.

A **decision** is a pair of consecutive trials ``t, t + 1`` of one segment (no problem change,
session start or dropped trial in between) whose state at ``t`` is defined (all three arms
visited, :mod:`src.behavior_analysis.rank_state`). The mouse *leaves* if it chooses another arm on
``t + 1``. The state includes the payout of trial ``t``.

Columns of :func:`decision_table`, one row per decision:

- ``t`` (row of trial ``t`` in the mouse's state), ``problem`` (fold label), ``own`` (arm chosen
  on ``t``), ``next`` (arm chosen on ``t + 1``), ``leave``;
- ``payout``, ``own_zero`` (payout is 0), ``own_value`` (value of the own arm), ``v_best`` and
  ``v_worst`` (higher and lower value of the two other arms) and ``d = own_value - v_best``;
- ``own_first``, ``own_second``: weight of the own arm on being first and second of the three
  (average ranks, :func:`src.behavior_analysis.rank_state.rank_weights`);
- history: ``run_log`` (log of the run length of the own arm), ``trace`` (the own arm's decayed
  choice count before ``t``, as the switch regression's trace), ``lag2`` to ``lag4`` (the own arm
  was also chosen ``k - 1`` trials before ``t``, within the segment) and ``plus_own`` (the own
  arm is in the row or column of the initiation port).

*Value* is the believed value by default; pass *values* to use another per-arm value (e.g. a
fitted Q-learner's) on the same trials.
"""

import numpy as np
import pandas as pd

from src.agents import kernel_q_agent as kq
from src.behavior_analysis import rank_state as rs
from src.behavior_analysis import switch_regression as sr

TRACE_DECAY = 0.9
LAGS = (2, 3, 4)


def position_in_segment(start: np.ndarray) -> np.ndarray:
    """Trials since the segment started (0 at a start)."""
    position = np.arange(len(start))
    return position - np.maximum.accumulate(np.where(start, position, 0))


def _lag_indicators(chosen: np.ndarray, start: np.ndarray) -> dict:
    """``{"lag<k>": the arm chosen on t was also chosen k - 1 trials earlier}`` in the segment."""
    inside = position_in_segment(start)
    out = {}
    for k in LAGS:
        earlier = np.r_[np.full(k - 1, -1), chosen[: len(chosen) - (k - 1)]]
        out[f"lag{k}"] = ((earlier == chosen) & (inside >= k - 1)).astype(float)
    return out


def decision_table(state: rs.MouseState, values: np.ndarray | None = None) -> pd.DataFrame:
    """The decisions of a mouse with their state (see the module docstring).

    Args:
        state (rs.MouseState): :func:`rank_state.mouse_state` of the mouse.
        values (np.ndarray | None): ``(n, 3)`` per-arm values replacing the believed values;
            the defined-state mask stays the believed values'. Default ``None``.

    Returns:
        pd.DataFrame: One row per decision, trials in order.

    Raises:
        ValueError: If *values* does not have one row per trial.
    """
    if values is None:
        values = state.believed
    if values.shape != state.believed.shape:
        raise ValueError(f"values must be {state.believed.shape}, got {values.shape}")
    rows = np.flatnonzero(state.defined & state.next_ok)
    own = state.chosen[rows]
    masked = state._replace(believed=values)
    best, worst = rs.alternative_values(masked)
    own_value = values[rows, own]
    ranks = rs.rank_weights(rs.average_rank(values[rows]))
    trace = sr.choice_trace(state.chosen, state.start, TRACE_DECAY)
    lags = _lag_indicators(state.chosen, state.start)
    return pd.DataFrame(
        dict(
            t=rows,
            problem=state.problem_code[rows],
            own=own,
            next=state.chosen[rows + 1],
            leave=(state.chosen[rows + 1] != own).astype(float),
            payout=state.payout[rows],
            own_zero=(state.payout[rows] == 0).astype(float),
            own_value=own_value,
            v_best=best[rows],
            v_worst=worst[rows],
            d=own_value - best[rows],
            own_first=ranks[np.arange(len(rows)), own, 0],
            own_second=ranks[np.arange(len(rows)), own, 1],
            run_log=np.log(state.run_length[rows]),
            trace=trace[rows, own],
            plus_own=state.plus[rows, own],
            **{name: lag[rows] for name, lag in lags.items()},
        )
    )


def kernel_q_values(state: rs.MouseState, params) -> np.ndarray:
    """``(n, 3)`` values of a fitted kernel Q-learner after each trial's update.

    The learner's values start at 0 at each problem and carry over its sessions; the choice
    kernel plays no part in the values, so it is not tracked.

    Args:
        state (rs.MouseState): The mouse's state.
        params (Sequence[float]): ``kernel_q_agent.PARAMETER_NAMES`` order.

    Returns:
        np.ndarray: The Q values the learner holds after the payout of each trial.
    """
    out = np.zeros((len(state.chosen), rs.N_ARMS))
    q, k = np.zeros(rs.N_ARMS), np.zeros(rs.N_ARMS)
    problem = -1
    for t, (arm, payout, code) in enumerate(
        zip(state.chosen, state.payout, state.problem_code, strict=True)
    ):
        if code != problem:
            q, problem = np.zeros(rs.N_ARMS), code
        q, k = kq.update_state(q, k, int(arm), float(payout), params)
        out[t] = q
    return out
