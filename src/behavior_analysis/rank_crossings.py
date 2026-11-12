"""Abrupt switching at rank crossings: a step against a graded change of leaving.

**Believed difference** ``D = own value - best alternative's value`` after the payout of the trial
(pulses of last payouts): ``D < 0`` means the mouse believes an alternative is now better than the
arm it is on, a rank crossing. A rank algorithm leaves at the crossing, a step at ``D = 0``
(with lapses); a softmax learner leaves more gradually the more negative ``D`` is.

- :func:`crossing_step`: P(leave) by ``D``; per mouse a smooth logistic in ``D`` against a step at
  ``D = 0`` and against the logistic plus the indicator ``[D < 0]``, held out by problem.
- :func:`believed_crossings` and :func:`true_crossings` find events; :func:`aligned_choices`
  gives the choice probabilities from ``WINDOW[0]`` to ``WINDOW[1]`` trials around them, and
  :func:`alignment_readouts` summarizes the first trial after and the trials to adopt the new best.

A *believed crossing* is the trial on which an arm the mouse has been choosing pays below the best
alternative's believed value, having been at or above it the trial before. A *true crossing* is a
trial on which the arm with the highest (or second highest) magnitude changes, the schedule's own
best / second-best reversals. Events are kept only where the roles are unambiguous (no tie).
"""

import numpy as np
import pandas as pd

from src.behavior_analysis import rank_logit as rl
from src.behavior_analysis import rank_state as rs
from src.behavior_analysis.rank_result import RankResult, curve_rows, empty_curves

WINDOW = (-5, 10)
D_CLIP = 3  # leaving is binned and plotted for D from -D_CLIP to D_CLIP; beyond it is pooled
MIN_EVENTS = 10  # events needed for the summary readouts of an alignment
NOT_ADOPTED = WINDOW[1] + 1  # trials to adoption when the new best is never at 50% in the window
LAPSE_FLOOR = 1e-3  # a step model's leaving rates stay inside (floor, 1 - floor)
ADOPTION = 0.5


# ---------------------------------------------------------------------------
# Step against graded
# ---------------------------------------------------------------------------


def _step_nats(table: pd.DataFrame) -> float:
    """Held-out nats per decision of a step at ``D = 0``: a leaving rate on each side."""
    below = (table["d"].to_numpy() < 0).astype(int)
    leave, problem = table["leave"].to_numpy(), table["problem"].to_numpy()
    total = 0.0
    labels = np.unique(problem)
    if len(labels) < 2:
        raise rl.TooFewGroupsError("cross-validation needs at least two groups")
    for label in labels:
        test = problem == label
        for side in (0, 1):
            train = ~test & (below == side)
            rate = np.clip(
                leave[train].mean() if train.any() else 0.5, LAPSE_FLOOR, 1 - LAPSE_FLOOR
            )
            mine = test & (below == side)
            total += float(
                (leave[mine] * np.log(rate) + (1 - leave[mine]) * np.log(1 - rate)).sum()
            )
    return -total / len(leave)


def _leave_nats(table: pd.DataFrame, columns: tuple) -> float:
    X = np.column_stack([np.ones(len(table)), *[table[c].to_numpy(float) for c in columns]])
    names = ("intercept", *columns)
    return rl.cross_validate(
        X, table["leave"].to_numpy(float), table["problem"].to_numpy(), names
    ).nats_per_trial


def crossing_step(table: pd.DataFrame) -> RankResult:
    """P(leave) by the believed difference, and the step against the graded fit.

    Args:
        table (pd.DataFrame): ``rank_decisions.decision_table`` of one mouse.

    Returns:
        RankResult: ``step_weight`` (the weight of ``[D < 0]`` in a logistic of leaving on ``D``,
        the indicator and the run length of the own arm) with ``se_step``; ``gain_indicator``
        (held-out nats per decision of that model without the indicator minus with it, positive if
        the indicator helps) and ``gain_step_vs_smooth`` (a logistic on ``D`` alone minus a step
        at ``D = 0`` with a leaving rate on each side, positive if the step predicts better). The
        curve ``crossing_step`` is P(leave) by ``D`` clipped to +-``D_CLIP``.

    Raises:
        ValueError: If the decisions come from fewer than two problems, or one class only.
    """
    frame = table.assign(below=(table["d"] < 0).astype(float))
    X = np.column_stack(
        [np.ones(len(frame)), *[frame[c].to_numpy(float) for c in ("d", "below", "run_log")]]
    )
    fit = rl.fit_logit(X, frame["leave"].to_numpy(float), ("intercept", "d", "below", "run_log"))
    smooth_history = _leave_nats(frame, ("d", "run_log"))
    with_indicator = _leave_nats(frame, ("d", "below", "run_log"))
    readouts = {
        "step_weight": fit.weight("below"),
        "se_step": fit.weight_se("below"),
        "gain_indicator": smooth_history - with_indicator,
        "gain_step_vs_smooth": _leave_nats(frame, ("d",)) - _step_nats(frame),
    }
    clipped = frame["d"].clip(-D_CLIP, D_CLIP)
    grouped = frame.groupby(clipped)["leave"].agg(["mean", "size"])
    curve = curve_rows("crossing_step", grouped.index.to_numpy(), grouped["mean"], grouped["size"])
    return RankResult(readouts, curve)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def _best_alternative(state: rs.MouseState) -> tuple[np.ndarray, np.ndarray]:
    """``(arm, value)`` of the best alternative on every trial; arm is -1 on a tie or no state."""
    best, _ = rs.alternative_values(state)
    unchosen = np.arange(rs.N_ARMS)[None, :] != state.chosen[:, None]
    at_best = unchosen & (state.believed == best[:, None])
    unique = at_best.sum(axis=1) == 1
    return np.where(unique, at_best.argmax(axis=1), -1), best


def believed_crossings(state: rs.MouseState) -> pd.DataFrame:
    """The believed crossings of a mouse.

    Returns:
        pd.DataFrame: ``t`` (row of the crossing trial), ``own`` (the arm it was on) and
        ``new_best`` (the alternative it now believes best), in trial order. A crossing needs
        the trial before in the same segment on the same arm with ``D >= 0``, a defined state on
        both trials and no tie for the best alternative.
    """
    arm, best = _best_alternative(state)
    d = state.payout - best
    previous_ok = np.r_[False, ~state.start[1:] & (state.chosen[1:] == state.chosen[:-1])]
    d_before = np.r_[np.nan, d[:-1]]
    event = previous_ok & (arm >= 0) & (d < 0) & (d_before >= 0)
    rows = np.flatnonzero(event)
    return pd.DataFrame(dict(t=rows, own=state.chosen[rows], new_best=arm[rows]))


def _top_two(true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(best, second)`` arm of each row by magnitude; the task never offers equal magnitudes."""
    order = np.argsort(-true, axis=1, kind="stable")
    return order[:, 0], order[:, 1]


def true_crossings(state: rs.MouseState, rank: int) -> pd.DataFrame:
    """The trials on which the true best (``rank = 1``) or second best (``rank = 2``) changes.

    A second-best change counts only while the best arm is unchanged, so the two sets are
    distinct.

    Returns:
        pd.DataFrame: ``t``, and the arm that held the rank before (``prev``) and holds it now
        (``next``).

    Raises:
        ValueError: For a rank other than 1 or 2.
    """
    if rank not in (1, 2):
        raise ValueError(f"rank must be 1 or 2, got {rank}")
    best, second = _top_two(state.true)
    same_problem = np.r_[False, state.problem_code[1:] == state.problem_code[:-1]]
    if rank == 1:
        holder = best
        event = same_problem & (best != np.r_[-1, best[:-1]])
    else:
        holder = second
        event = same_problem & (second != np.r_[-1, second[:-1]]) & (best == np.r_[-1, best[:-1]])
    rows = np.flatnonzero(event)
    return pd.DataFrame(dict(t=rows, prev=holder[rows - 1], next=holder[rows]))


def aligned_choices(state: rs.MouseState, rows: np.ndarray, roles: dict) -> pd.DataFrame:
    """P(choosing each role's arm) from ``WINDOW[0]`` to ``WINDOW[1]`` trials around events.

    Args:
        state (rs.MouseState): The mouse's state.
        rows (np.ndarray): ``(E,)`` event rows (trial 0 of the window).
        roles (dict): ``{role name: (E,) arm of the role at each event}``.

    Returns:
        pd.DataFrame: ``role, k, p, n`` (events with a trial of the same problem at ``k``); ``p`` is
        NaN where no event has one.
    """
    ks = np.arange(WINDOW[0], WINDOW[1] + 1)
    at = rows[:, None] + ks[None, :]
    inside = (at >= 0) & (at < len(state.chosen))
    clipped = np.clip(at, 0, len(state.chosen) - 1)
    valid = inside & (state.problem_code[clipped] == state.problem_code[rows][:, None])
    choice = state.chosen[clipped]
    out = []
    for role, arms in roles.items():
        hit = (choice == np.asarray(arms)[:, None]) & valid
        n = valid.sum(axis=0)
        p = np.divide(hit.sum(axis=0), n, out=np.full(len(ks), np.nan), where=n > 0)
        out.append(pd.DataFrame(dict(role=role, k=ks, p=p, n=n)))
    return pd.concat(out, ignore_index=True)


def _first_adoption(profile: pd.DataFrame, role: str, first: int) -> float:
    """First k >= *first* at which the role's probability reaches ``ADOPTION``."""
    mine = profile[(profile["role"] == role) & (profile["k"] >= first)]
    reached = mine.loc[mine["p"] >= ADOPTION, "k"]
    return float(reached.min()) if len(reached) else float(NOT_ADOPTED)


def _p_at(profile: pd.DataFrame, role: str, k: int) -> float:
    return float(profile.loc[(profile["role"] == role) & (profile["k"] == k), "p"].iloc[0])


def alignment_readouts(state: rs.MouseState) -> RankResult:
    """Choices around believed and true crossings, with their summaries.

    Returns:
        RankResult: Curves ``cross_believed`` (series ``own`` / ``new_best`` / ``other``),
        ``cross_true_best`` and ``cross_true_second`` (series ``prev`` / ``next`` / ``other``) and
        readouts ``cross_leave_1`` (P(leave) the trial after a believed crossing),
        ``cross_adopt_1`` (P(choose the new believed best) then), ``cross_t50`` (first trial after
        the crossing at which the new believed best is chosen at least half the time; 11 if never),
        ``true_best_next_1`` and ``true_best_t50``, ``true_second_next_1`` and ``true_second_t50``
        likewise for the true crossings, and ``n_cross_believed``, ``n_cross_true_best``,
        ``n_cross_true_second`` (event counts). Summary readouts are NaN with fewer than
        ``MIN_EVENTS`` events.
    """
    readouts: dict = {}
    curves = []
    events = believed_crossings(state)
    readouts["n_cross_believed"] = float(len(events))
    plans = [
        ("cross_believed", events, {"own": "own", "new_best": "new_best"}, "cross", "new_best")
    ]
    for rank, name, tag in (
        (1, "cross_true_best", "true_best"),
        (2, "cross_true_second", "true_second"),
    ):
        plans.append(
            (name, true_crossings(state, rank), {"prev": "prev", "next": "next"}, tag, "next")
        )
    for curve, found, role_columns, tag, adopted in plans:
        n_name = {"cross": "n_cross_believed"}.get(tag, f"n_cross_{tag}")
        readouts[n_name] = float(len(found))
        names = (
            [f"{tag}_{s}" for s in ("leave_1", "adopt_1", "t50")]
            if tag == "cross"
            else [f"{tag}_next_1", f"{tag}_t50"]
        )
        if len(found) < MIN_EVENTS:
            readouts |= dict.fromkeys(names, np.nan)
            continue
        roles = {role: found[col].to_numpy() for role, col in role_columns.items()}
        roles["other"] = _remaining_arm(found, list(role_columns.values()))
        profile = aligned_choices(state, found["t"].to_numpy(), roles)
        for role in roles:
            part = profile[profile["role"] == role]
            curves.append(curve_rows(curve, part["k"], part["p"], part["n"], role))
        if tag == "cross":
            readouts["cross_leave_1"] = 1.0 - _p_at(profile, "own", 1)
            readouts["cross_adopt_1"] = _p_at(profile, adopted, 1)
            readouts["cross_t50"] = _first_adoption(profile, adopted, 1)
        else:
            readouts[f"{tag}_next_1"] = _p_at(profile, adopted, 1)
            readouts[f"{tag}_t50"] = _first_adoption(profile, adopted, 1)
    return RankResult(readouts, pd.concat(curves, ignore_index=True) if curves else empty_curves())


def _remaining_arm(found: pd.DataFrame, columns: list) -> np.ndarray:
    """The arm that is neither of the two arms in *columns* at each event (arms are 0, 1, 2)."""
    first, second = (found[c].to_numpy() for c in columns)
    return rs.N_ARMS * (rs.N_ARMS - 1) // 2 - first - second
