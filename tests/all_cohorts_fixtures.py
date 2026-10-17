"""Synthetic imported cohorts for the all-cohorts drifting pipeline.

Each cohort is ``{overall problem: {mouse: {session_key: session}}}``: one blocked problem first,
then drifting problems. Sessions are built from pyControl-like frames (a ``run_start`` row, one
``print`` row per trial, a ``run_end`` row) and imported by ``extract_session_trials``, so the
value, rank and port analyses read the same fields as on real data.

A drifting problem has two blocks per session. In a block one arm is at 0 and the other two
drift by a step now and then; at the block change the most chosen arm drops to 0 and the old zero
arm is rewarded again. Mice mostly choose the arm with the highest magnitude, so the best arm
changes at some block changes and the second-best arm at others.
"""

import json

import numpy as np
import pandas as pd

from src.behavior_import.extract_trials import extract_session_trials
from tests.poke_fixtures import session_record

BLOCKED_NAME = "Meg\\3x3_field_blocked_value_bandit"
DRIFTING_NAME = "Meg\\3x3_field_drifting_value_bandit"
# Initiation port B2: A3 and C3 are X ports, B3 a plus port. Problem 3 mixes X and plus ports.
PORTS_BY_PROBLEM = {
    1: ["A1", "B1", "C1"],
    2: ["A3", "B3", "C3"],
    3: ["A1", "B3", "C2"],
}
TRIALS_PER_SESSION = 30
SESSIONS_PER_PROBLEM = 2


def _drift(mags, zero_arm, rng):
    """One drift step: each non-zero arm moves by at most 1 within 1-5."""
    return {
        arm: 0 if arm == zero_arm else int(np.clip(value + rng.integers(-1, 2), 1, 5))
        for arm, value in mags.items()
    }


def _drifting_trials(ports, rng, mags, n_trials, change_at):
    """``(print payloads, mags at the end)``; the block changes before trial *change_at*."""
    rows, blocks = [], 1
    for i in range(n_trials):
        if i == change_at:
            blocks = 2
            old_zero = next(arm for arm, v in mags.items() if v == 0)
            top = max(mags, key=lambda arm: mags[arm])  # the most chosen arm drops to 0
            mags = {**mags, top: 0, old_zero: int(rng.integers(2, 5))}
        elif rng.random() < 0.3:
            mags = _drift(mags, next(arm for arm, v in mags.items() if v == 0), rng)
        order = sorted(ports, key=lambda arm: -mags[arm])
        pick = order[0] if rng.random() < 0.75 else order[1]
        rows.append(
            {
                "num_t": i + 1,
                "num_blocks": blocks,
                "curr_rew_mag": dict(mags),
                "chosen_tow": pick,
                "c_tows": list(ports),
                "chosen_rank": ("best", "second", "third")[order.index(pick)],
                "rank_counts": {"best": 1, "second": 0, "third": 0},
            }
        )
    return rows, mags


def _frame(start, trials, end):
    rows = [(0.0, "variable", "run_start", json.dumps(start))]
    rows += [(1.0 + i, "variable", "print", json.dumps(t)) for i, t in enumerate(trials)]
    rows.append((2.0 + len(trials), "variable", "run_end", json.dumps(end)))
    return pd.DataFrame(rows, columns=["time", "type", "subtype", "content"])


def drifting_problem_sessions(overall_problem, seed, *, first_session=1):
    """``{session_key: raw frame}`` for one mouse's drifting problem (blocks carry over)."""
    ports = PORTS_BY_PROBLEM[overall_problem]
    rng = np.random.default_rng(seed)
    mags = {ports[0]: 4, ports[1]: 3, ports[2]: 0}
    total_blocks, frames = 0, {}
    for n in range(first_session, first_session + SESSIONS_PER_PROBLEM):
        trials, end_mags = _drifting_trials(ports, rng, mags, TRIALS_PER_SESSION, 15)
        # the block open at the end of the previous session continues
        total_blocks = max(total_blocks, 1)
        start = {"num_problem": overall_problem, "curr_rew_mag": dict(mags),
                 "num_blocks": 1, "tot_num_blocks": total_blocks}  # fmt: skip
        total_blocks += 1
        end = {"num_problem": overall_problem, "curr_rew_mag": dict(end_mags),
               "num_blocks": 2, "tot_num_blocks": total_blocks}  # fmt: skip
        frames[session_key(n)] = _frame(start, trials, end)
        mags = end_mags
    return frames


def blocked_problem_sessions(overall_problem, seed, *, first_session=1):
    """``{session_key: raw frame}`` for a blocked problem (a good reversal mid-session)."""
    ports = PORTS_BY_PROBLEM[overall_problem]
    rng = np.random.default_rng(seed)
    frames = {}
    for n in range(first_session, first_session + SESSIONS_PER_PROBLEM):
        trials = []
        for i in range(TRIALS_PER_SESSION):
            good = int(i >= 15)
            best = ports[good]
            mags = {p: (4 if p == best else 1 if p == ports[2] else 0) for p in ports}
            pick = best if rng.random() < 0.8 else ports[2]
            trials.append(
                {
                    "num_t": i + 1,
                    "num_good_rev": good,
                    "num_bad_rev": 0,
                    "curr_rew_mag": mags,
                    "chosen_tow": pick,
                    "c_tows": list(ports),
                    "chosen_rank": "best" if pick == best else "third",
                    "rank_counts": {"best": 1, "second": 0, "third": 0},
                }
            )
        start = {"num_problem": overall_problem, "num_good_rev": 0, "tot_num_good_rev": 0}
        end = {"num_problem": overall_problem, "num_good_rev": 1, "tot_num_good_rev": 1}
        frames[session_key(n)] = _frame(start, trials, end)
    return frames


def session_key(number):
    return f"ses-{number}_date-202601{number:02d}"


def imported(frames, mouse, overall_problem, task_name):
    """Import each raw frame as a session of *mouse* in *overall_problem*."""
    ports = PORTS_BY_PROBLEM[overall_problem]
    sessions = {}
    for key, frame in frames.items():
        session = session_record(
            [frame], problem=overall_problem, task_name=task_name, choice_towers=set(ports)
        )
        sessions[key] = extract_session_trials(session, mouse, key)
    return sessions


def synthetic_cohort(mice, *, drifting_problems=(2, 3), seed=0):
    """A blocked problem 1 then drifting problems, each run by every mouse of *mice*.

    Args:
        mice: Mouse ids; every id gets its own random drift, so mice differ.
        drifting_problems: Overall problem numbers of the drifting problems (keys of
            ``PORTS_BY_PROBLEM``); sessions are numbered on from the problem before.
        seed: Added to every mouse's seed, so two cohorts do not share values.
    """
    cohort = {}
    for m, mouse in enumerate(mice):
        for p in (1, *drifting_problems):
            first = 1 + SESSIONS_PER_PROBLEM * (p - 1)
            offset = 1000 * seed + 100 * m + p
            if p == 1:
                frames, name = (
                    blocked_problem_sessions(p, offset, first_session=first),
                    BLOCKED_NAME,
                )
            else:
                frames, name = (
                    drifting_problem_sessions(p, offset, first_session=first),
                    DRIFTING_NAME,
                )
            cohort.setdefault(p, {})[mouse] = imported(frames, mouse, p, name)
    return cohort
