"""The choice-history kernel within the trials a GLM-HMM assigns to each state."""

import numpy as np
import pytest

from src.agents import drifting_simulation as sim
from src.agents import glm_hmm_agent as gha
from src.behavior_analysis.data.drifting_trial_table import drifting_trial_table
from src.behavior_analysis.glm_hmm import glm_hmm_features as feats
from src.behavior_analysis.glm_hmm import glm_hmm_kernel as ker
from tests.strategy_fixtures import KERNEL_Q, RULES, layouts


def habit_and_random_mouse(seed=0):
    """Sessions that are entirely a habit state or entirely a random one: table and states."""
    weights = np.stack([gha.habit_weights(4.0, 0.0), np.zeros(len(gha.FEATURES))])
    holder = []

    def factory(layout):
        policy = gha.FeaturePolicy(
            weights,
            np.eye(2),
            np.array([0.5, 0.5]),
            KERNEL_Q,
            layout,
            np.random.default_rng([seed, len(holder)]),
        )
        holder.append(policy)
        return policy

    played = sim.simulate_policy_mouse(
        factory, layouts(6, 6, 100, seed), RULES, np.random.default_rng(seed), "m"
    )
    table = drifting_trial_table(played)
    return table, np.concatenate([p.true_states() for p in holder])


def one_hot_posterior(data, states, n_states=2):
    gamma = np.zeros((*data.sequences.mask.shape, n_states))
    gamma[data.trials["session"], data.trials["step"], states] = 1.0
    return gamma


def test_the_kernel_is_sharp_in_a_habit_state_and_zero_in_a_random_one():
    table, states = habit_and_random_mouse()
    data = feats.session_arrays(table, KERNEL_Q)
    result = ker.kernel_by_state(table, data, one_hot_posterior(data, states), min_trials=200)
    weights = result.pivot(index="state", columns="lag", values="weight")
    assert set(weights.index) == {"all", 0, 1}
    assert weights.loc[0, 1] > 1.0  # the habit state repeats its last choice
    assert weights.loc[1].abs().max() < 0.5  # a random state has no kernel
    counts = result.groupby("state")["n_trials"].first()
    assert counts["all"] == len(table)
    assert counts[0] + counts[1] == len(table)


def test_threshold_leaves_out_uncertain_trials():
    table, states = habit_and_random_mouse(1)
    data = feats.session_arrays(table, KERNEL_Q)
    gamma = one_hot_posterior(data, states)
    gamma[0, 0] = [0.5, 0.5]
    rows = ker.state_rows(data, gamma, 0.8)
    assert rows.shape == (len(table), 2)
    assert rows[0].sum() == 0
    assert rows[1:].sum(axis=1).min() == 1


def test_a_state_with_too_few_trials_gives_nan_not_a_fit():
    table, states = habit_and_random_mouse(2)
    data = feats.session_arrays(table, KERNEL_Q)
    result = ker.kernel_by_state(table, data, one_hot_posterior(data, states), min_trials=10**6)
    state_rows = result[result["state"] != "all"]
    assert state_rows["weight"].isna().all()
    assert state_rows["n_trials"].gt(0).all()


def test_flatness_is_late_over_first_lag():
    assert ker.flatness(np.array([1.0, 1.0, 1.0, 1.0, 1.0])) == pytest.approx(1.0)
    assert ker.flatness(np.array([2.0, 0.5, 0.0, 0.0, 0.0])) == pytest.approx(0.0)
    assert ker.flatness(np.array([1.0, 0.5, 0.4, 0.2, 0.0])) == pytest.approx(0.2)
    assert np.isnan(ker.flatness(np.array([-0.1, 0.2, 0.2, 0.2, 0.2])))


def test_rows_must_line_up_with_the_design():
    table, states = habit_and_random_mouse(3)
    data = feats.session_arrays(table, KERNEL_Q)
    with pytest.raises(ValueError, match="align"):
        ker.kernel_by_state(table.iloc[:-5], data, one_hot_posterior(data, states))
