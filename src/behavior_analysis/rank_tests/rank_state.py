"""What a mouse believes at each trial: believed values, their ranks and segment bookkeeping.

The ranked value algorithm of the thesis ranks the three arms and gates choices on comparisons
between adjacent ranks. Every rank test starts from the same per-trial state, built here once
from one mouse's rows of the drifting trial table (:class:`MouseState`).

**Believed value** of an arm is the last payout the mouse received at that port in the current
problem, carried across the problem's sessions. It is undefined (NaN) before the arm's first
visit, and a trial's **state is defined** only once all three arms have been visited, so a rank
never rests on a guessed value. The believed values at row ``t`` include the payout of trial
``t``: the state the mouse holds when it decides trial ``t + 1``.

**Ranks** are average ranks (1 = highest believed value): tied arms share the mean of the ranks
they span, so two arms tied for the top are both 1.5. The indicators of :func:`rank_weights`
interpolate linearly, so a tie gives each arm half of each of the two ranks it spans and the
three indicators of an arm always sum to 1. Nothing depends on the order of the ports.

Everything here is pure: a trial table in, arrays out.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd

from src.agents import drifting_simulation as sim

N_ARMS = 3


class MouseState(NamedTuple):
    """Per-trial arrays of one mouse, problems in order of first appearance, trials in order.

    Attributes:
        mouse (str): The mouse.
        chosen (np.ndarray): ``(n,)`` chosen arm (0-2, sorted port order of its problem).
        payout (np.ndarray): ``(n,)`` pulses paid to the chosen arm.
        true (np.ndarray): ``(n, 3)`` magnitudes offered before the choice.
        believed (np.ndarray): ``(n, 3)`` last payout of each arm in the problem up to and
            including this trial; NaN before an arm's first visit.
        defined (np.ndarray): ``(n,)`` bool, all three arms visited (the state has no NaN).
        start (np.ndarray): ``(n,)`` bool, the trial starts a segment: a problem's first trial, a
            session's first trial, or the first trial after a dropped trial.
        next_ok (np.ndarray): ``(n,)`` bool, the next row is the next trial of the same segment.
        run_length (np.ndarray): ``(n,)`` consecutive choices of the chosen arm ending here,
            within the segment (at least 1).
        plus (np.ndarray): ``(n, 3)`` 1 for an arm in the row or column of the initiation port.
        problem_code (np.ndarray): ``(n,)`` 0, 1, ... in order of first appearance.
        problem_index (np.ndarray): ``(n,)`` problem number within its cohort's drifting
            problems (the second element of a ``(cohort, k)`` key, or the key itself).
        trial (np.ndarray): ``(n,)`` trial within the problem, across its sessions.
    """

    mouse: str
    chosen: np.ndarray
    payout: np.ndarray
    true: np.ndarray
    believed: np.ndarray
    defined: np.ndarray
    start: np.ndarray
    next_ok: np.ndarray
    run_length: np.ndarray
    plus: np.ndarray
    problem_code: np.ndarray
    problem_index: np.ndarray
    trial: np.ndarray


def _forward_fill(values: np.ndarray) -> np.ndarray:
    """Each column's last non-NaN value up to every row (NaN before its first)."""
    rows = np.arange(len(values))[:, None]
    last = np.maximum.accumulate(np.where(np.isnan(values), 0, rows), axis=0)
    return values[last, np.arange(values.shape[1])]


def _run_length(chosen: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Length of the run of equal choices ending at each trial, restarting at a segment start."""
    repeat = np.r_[False, chosen[1:] == chosen[:-1]] & ~start
    position = np.arange(len(chosen))
    run_start = np.maximum.accumulate(np.where(~repeat, position, 0))
    return position - run_start + 1


def _problem_arrays(rows: pd.DataFrame) -> dict:
    ports = sorted(rows["magnitudes"].iloc[0])
    if any(sorted(m) != ports for m in rows["magnitudes"]):
        raise ValueError(f"problem {rows['problem'].iloc[0]!r}: the offered ports change")
    chosen = np.array([ports.index(c) for c in rows["choice"]], dtype=int)
    payout = rows["reward"].to_numpy(dtype=float)
    start = (rows["trial_in_session"].to_numpy() == 0) | rows["after_gap"].to_numpy(dtype=bool)
    start[0] = True
    seen = np.full((len(rows), N_ARMS), np.nan)
    seen[np.arange(len(rows)), chosen] = payout
    believed = _forward_fill(seen)
    next_ok = np.r_[~start[1:], False]
    initiation = rows["initiation"].iloc[0]
    return dict(
        chosen=chosen,
        payout=payout,
        true=np.array([[m[p] for p in ports] for m in rows["magnitudes"]], dtype=float),
        believed=believed,
        defined=~np.isnan(believed).any(axis=1),
        start=start,
        next_ok=next_ok,
        run_length=_run_length(chosen, start),
        plus=np.tile(sim.plus_ports(ports, initiation).astype(float), (len(rows), 1)),
        trial=rows["trial"].to_numpy(),
    )


def _problem_index(key) -> int:
    return key[1] if isinstance(key, tuple) else key


def mouse_state(trials: pd.DataFrame) -> MouseState:
    """The per-trial state of one mouse.

    Args:
        trials (pd.DataFrame): One mouse's rows of the drifting trial table.

    Returns:
        MouseState: Problems in order of first appearance, each in trial order.

    Raises:
        ValueError: If the table is empty or holds more than one mouse, or a problem does not
            offer the same ports on every trial.
    """
    if trials.empty:
        raise ValueError("the trial table is empty")
    if trials["mouse"].nunique() != 1:
        raise ValueError(f"mouse_state takes one mouse, got {trials['mouse'].nunique()}")
    parts, codes, indices = [], [], []
    for code, (key, rows) in enumerate(trials.groupby("problem", sort=False)):
        arrays = _problem_arrays(rows.sort_values("trial"))
        parts.append(arrays)
        codes.append(np.full(len(rows), code))
        indices.append(np.full(len(rows), _problem_index(key)))
    joined = {name: np.concatenate([p[name] for p in parts]) for name in parts[0]}
    return MouseState(
        mouse=trials["mouse"].iloc[0],
        problem_code=np.concatenate(codes),
        problem_index=np.concatenate(indices),
        **joined,
    )


def average_rank(values: np.ndarray) -> np.ndarray:
    """``(n, 3)`` average rank of each arm (1 = highest value, ties share the mean rank).

    Args:
        values (np.ndarray): ``(n, 3)`` values; a row with a NaN gives NaN ranks.
    """
    greater = (values[:, None, :] > values[:, :, None]).sum(axis=2)
    equal = (values[:, None, :] == values[:, :, None]).sum(axis=2)  # includes the arm itself
    ranks = 1.0 + greater + (equal - 1) / 2.0
    return np.where(np.isnan(values).any(axis=1)[:, None], np.nan, ranks)


def rank_weights(ranks: np.ndarray) -> np.ndarray:
    """``(n, 3 arms, 3)`` weight of each arm on being first, second and third.

    Piecewise linear in the average rank, so tied arms split the ranks they span and the
    weights of an arm sum to 1.
    """
    first = np.clip(2.0 - ranks, 0, 1)
    second = np.clip(1.0 - np.abs(ranks - 2.0), 0, 1)
    third = np.clip(ranks - 2.0, 0, 1)
    return np.stack([first, second, third], axis=2)


def alternative_values(state: MouseState) -> tuple[np.ndarray, np.ndarray]:
    """``(best, worst)``: the higher and lower believed value of the two unchosen arms.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(n,)`` each, NaN where the state is undefined.
    """
    values = np.where(state.defined[:, None], state.believed, np.nan)
    unchosen = np.arange(N_ARMS)[None, :] != state.chosen[:, None]
    pair = values[unchosen].reshape(-1, N_ARMS - 1)  # row-major: each row's two unchosen arms
    return pair.max(axis=1), pair.min(axis=1)
