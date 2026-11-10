"""Synthetic mice for the switch-regression tests: hand-built sequences and a known switch rule."""

import numpy as np
import pandas as pd
from scipy.special import expit

from src.pipelines.choice_logit_values import SIMULATED_PORTS

DECAY = 0.9


def table_from_rows(rows: list[tuple], mouse: str = "m1") -> pd.DataFrame:
    """A drifting trial table from ``(problem, session, trial_in_session, after_gap, choice,
    reward, magnitudes)`` tuples, with ``trial`` counted within each problem."""
    records, trial_of = [], {}
    for problem, session, in_session, after_gap, choice, reward, magnitudes in rows:
        trial = trial_of.get(problem, 0)
        trial_of[problem] = trial + 1
        records.append(
            dict(
                mouse=mouse,
                problem=problem,
                session=f"{problem}-{session}",
                session_index=session,
                trial=trial,
                trial_in_session=in_session,
                choice=choice,
                magnitudes=dict(magnitudes),
                reward=reward,
                initiation="B2",
                block=None,
                after_gap=after_gap,
            )
        )
    return pd.DataFrame(records)


def simulate_switch_mouse(
    mouse: str,
    seed: int,
    *,
    value_weight: float,
    trace_weight: float = 0.0,
    p_stay: float = 0.5,
    n_problems: int = 6,
    sessions_per_problem: int = 2,
    trials_per_session: int = 400,
) -> pd.DataFrame:
    """One mouse whose switches follow the switch regression with known weights.

    Every arm offers 0-4 pulses at random on every trial, independently of everything else. The
    first trial of a session is uniform. On later trials the mouse repeats its previous choice
    with probability *p_stay*; otherwise it goes to the first of the two other arms (sorted port
    order) with probability ``sigmoid(value_weight * (m_first - m_second) + trace_weight *
    (h_first - h_second))``, where ``m`` are the magnitudes offered on that trial and ``h`` the
    choice trace (decay ``DECAY``, cleared at a session start).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_problems):
        ports = sorted(SIMULATED_PORTS[k % len(SIMULATED_PORTS)])
        for session in range(sessions_per_problem):
            previous, trace = None, np.zeros(3)
            for in_session in range(trials_per_session):
                magnitudes = rng.integers(0, 5, size=3)
                if previous is None:
                    chosen = int(rng.integers(3))
                elif rng.random() < p_stay:
                    chosen = previous
                else:
                    first, second = (a for a in range(3) if a != previous)
                    utility = value_weight * (magnitudes[first] - magnitudes[second])
                    utility += trace_weight * (trace[first] - trace[second])
                    chosen = first if rng.random() < expit(utility) else second
                trace = DECAY * trace
                trace[chosen] += 1.0
                previous = chosen
                rows.append(
                    (
                        k + 1,
                        session,
                        in_session,
                        False,
                        ports[chosen],
                        int(magnitudes[chosen]),
                        dict(zip(ports, (int(m) for m in magnitudes), strict=True)),
                    )
                )
    return table_from_rows(rows, mouse)
