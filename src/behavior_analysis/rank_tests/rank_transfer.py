"""Transfer across problems and inference at block changes.

**Learning across problems** (:func:`transfer`). A mouse that has learned the structure of the task
(the ranks matter, one arm is always at zero) should get better at each new problem; a learner
that starts every problem from nothing should not. Per problem: P(choose the true best arm) over
the first 50 and 100 trials, and the trials until a sliding 20-trial window of that probability
first reaches ``THRESHOLD`` (the problem's length if it never does: censored). Against the
problem's number within its cohort, the per-mouse slope. A replay of a fitted agent does not
transfer, so its slope is the schedule's own trend; the calibration compares the two.

**Inference at block changes** (:func:`block_inference`). In the task a block change zeroes the
arm chosen most in the block and gives its value to the arm that was at zero. A mouse that knows
this can infer the new best arm without sampling it: when its favorite arm suddenly pays 0, the
arm it had been avoiding (the old zero arm) now holds that favorite's value. A learner that goes by
last payouts believes that arm is worth 0 and goes to the third arm instead. The readout is the
share of first leaves after such a change that go to the old zero arm.
"""

import numpy as np
import pandas as pd

from src.behavior_analysis.rank_tests import rank_state as rs
from src.behavior_analysis.rank_tests.rank_result import RankResult, curve_rows, empty_curves

FIRST_TRIALS = (50, 100)
WINDOW = 20
THRESHOLD = 0.6  # chance is 1/3
MIN_PROBLEMS = 3  # a slope needs this many problems with the metric
MAX_WAIT = 30  # trials after a block change within which the first leave must come
MIN_BLOCK_EVENTS = 5


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    keep = ~np.isnan(y)
    if keep.sum() < MIN_PROBLEMS or np.unique(x[keep]).size < 2:
        return np.nan
    return float(np.polyfit(x[keep], y[keep], 1)[0])


def _first_reaching(hit: np.ndarray) -> float:
    """Trials until a ``WINDOW``-trial mean of *hit* reaches ``THRESHOLD``; the length if never."""
    if len(hit) < WINDOW:
        return np.nan
    means = np.convolve(hit, np.ones(WINDOW) / WINDOW, mode="valid")
    reached = np.flatnonzero(means >= THRESHOLD - 1e-12)
    return float(reached[0] + WINDOW) if len(reached) else float(len(hit))


def problem_metrics(state: rs.MouseState) -> pd.DataFrame:
    """Per problem: ``problem_index, n_trials``, ``p50``, ``p100`` and ``trials_to_threshold``.

    ``p50`` / ``p100`` are NaN for a problem with fewer trials than that, ``trials_to_threshold``
    for one shorter than ``WINDOW``. P(best) is P(choosing the arm with the highest magnitude).
    """
    best = state.true.argmax(axis=1)
    hit = (state.chosen == best).astype(float)
    rows = []
    for code in np.unique(state.problem_code):
        mine = hit[state.problem_code == code]
        row = dict(
            problem_index=int(state.problem_index[state.problem_code == code][0]),
            n_trials=len(mine),
            trials_to_threshold=_first_reaching(mine),
        )
        for n in FIRST_TRIALS:
            row[f"p{n}"] = float(mine[:n].mean()) if len(mine) >= n else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def transfer(state: rs.MouseState) -> RankResult:
    """Learning across problems: per-problem metrics, their slope and the half-to-half change.

    Returns:
        RankResult: ``slope_p50``, ``slope_p100``, ``slope_threshold`` (per problem; NaN with
        fewer than ``MIN_PROBLEMS`` problems), ``half_p100`` (mean ``p100`` of the later half of
        the mouse's problems minus the earlier half) and ``n_problems``; curves
        ``transfer_p100`` and ``transfer_threshold`` by problem number.
    """
    table = problem_metrics(state).sort_values("problem_index")
    x = table["problem_index"].to_numpy(float)
    p100 = table["p100"].to_numpy()
    half = len(table) // 2
    early, late = p100[:half], p100[len(table) - half :]
    readouts = {
        "slope_p50": _slope(x, table["p50"].to_numpy()),
        "slope_p100": _slope(x, p100),
        "slope_threshold": _slope(x, table["trials_to_threshold"].to_numpy()),
        "half_p100": float(np.nanmean(late) - np.nanmean(early))
        if half and not (np.isnan(early).all() or np.isnan(late).all())
        else np.nan,
        "n_problems": float(len(table)),
    }
    curves = [
        curve_rows(f"transfer_{name}", x, table[column].to_numpy(), table["n_trials"].to_numpy())
        for name, column in (("p100", "p100"), ("threshold", "trials_to_threshold"))
    ]
    return RankResult(readouts, pd.concat(curves, ignore_index=True))


def block_events(state: rs.MouseState) -> pd.DataFrame:
    """The block changes after which the mouse leaves the zeroed arm, and where it goes.

    A block change is a trial on which the arm paying 0 changes. An event needs the mouse to
    choose the zeroed arm on that trial (it receives the 0), a first leave within ``MAX_WAIT``
    trials in the same problem, and the believed values of all three arms just before the leave.

    Returns:
        pd.DataFrame: ``t`` (change), ``leave`` (first trial on another arm), ``zeroed``,
        ``inherit`` (the old zero arm, which now holds the zeroed arm's value), ``target`` (arm
        chosen at the leave), ``v_inherit`` and ``v_other`` (believed values of the old zero arm
        and the third arm before the leave) and ``to_inherit`` (the leave went to the old zero arm).
    """
    zero = state.true.argmin(axis=1)
    same_problem = np.r_[False, state.problem_code[1:] == state.problem_code[:-1]]
    changes = np.flatnonzero(same_problem & (zero != np.r_[-1, zero[:-1]]))
    rows = []
    for t in changes:
        zeroed, inherit = zero[t], zero[t - 1]
        if state.chosen[t] != zeroed:
            continue
        after = np.arange(t + 1, min(t + 1 + MAX_WAIT, len(zero)))
        after = after[state.problem_code[after] == state.problem_code[t]]
        left = after[state.chosen[after] != zeroed]
        if len(left) == 0 or not state.defined[left[0] - 1]:
            continue
        s = left[0]
        other = rs.N_ARMS * (rs.N_ARMS - 1) // 2 - zeroed - inherit
        rows.append(
            dict(
                t=t, leave=s, zeroed=zeroed, inherit=inherit, target=state.chosen[s],
                v_inherit=state.believed[s - 1, inherit], v_other=state.believed[s - 1, other],
                to_inherit=float(state.chosen[s] == inherit),
            )
        )  # fmt: skip
    columns = ["t", "leave", "zeroed", "inherit", "target", "v_inherit", "v_other", "to_inherit"]
    return pd.DataFrame(rows, columns=columns)


def block_inference(state: rs.MouseState) -> RankResult:
    """Share of first leaves after a block change that go to the old zero arm.

    Returns:
        RankResult: ``block_inherit`` (all events), ``block_inherit_below`` (events in which the
        last payouts say the old zero arm is worth less than the third arm, so a value learner
        goes to the third arm) and ``n_block_events``, ``n_block_below``. The shares are NaN with
        fewer than ``MIN_BLOCK_EVENTS`` events.
    """
    events = block_events(state)
    below = events[events["v_inherit"] < events["v_other"]]
    readouts = {
        "n_block_events": float(len(events)),
        "n_block_below": float(len(below)),
        "block_inherit": events["to_inherit"].mean() if len(events) >= MIN_BLOCK_EVENTS else np.nan,
        "block_inherit_below": (
            below["to_inherit"].mean() if len(below) >= MIN_BLOCK_EVENTS else np.nan
        ),
    }
    return RankResult(readouts, empty_curves())
