"""Synthetic mice with a known rule for the rank tests: a toy rank agent and a softmax learner."""

import numpy as np
import pandas as pd

from src.agents import drifting_simulation as sim
from src.agents.rank_agent import RankPolicy
from src.behavior_analysis.drifting_trial_table import drifting_trial_table
from tests.strategy_fixtures import RULES, layouts

# Learning rate 1 makes the learner's value the last payout, as the believed value is; beta sets
# how sharply the softmax follows it.
SOFTMAX = (1.0, 1.2, 0.0, 0.0)  # learning rate, beta, kappa, plus bias


def simulate(make_policy, seed: int, mouse: str, n_problems=6, sessions=3, trials=200):
    """One mouse's trial table played by ``make_policy(layout)``."""
    played = sim.simulate_policy_mouse(
        make_policy,
        layouts(n_problems, sessions, trials, seed),
        RULES,
        np.random.default_rng([seed, 7]),
        mouse,
    )
    return drifting_trial_table(played)


def rank_mouse(seed=0, mouse="rank", **size) -> pd.DataFrame:
    return simulate(lambda layout: RankPolicy(layout), seed, mouse, **size)


def softmax_mouse(seed=0, mouse="softmax", **size) -> pd.DataFrame:
    return simulate(
        lambda layout: sim.ValuePolicy(SOFTMAX, "magnitude", layout), seed, mouse, **size
    )
