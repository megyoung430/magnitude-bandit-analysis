"""The GLM-HMM pipeline: one mouse, simulated recovery, and the replays that calibrate it."""

import numpy as np
import pandas as pd
import pytest

from src.agents import drifting_simulation as sim
from src.agents import glm_hmm_agent as gha
from src.behavior_analysis.data.drifting_trial_table import drifting_trial_table
from src.behavior_analysis.glm_hmm import glm_hmm as hmm
from src.behavior_analysis.glm_hmm import glm_hmm_features as feats
from src.pipelines.glm_hmm import glm_hmm as pipe
from tests.strategy_fixtures import HABIT, KERNEL_Q, RULES, layouts

AGENTS = pipe.Agents(kq=KERNEL_Q, habit=HABIT)
# small and fast: the structure of the real analysis at toy size
TINY = pipe.Config(
    n_states=(1, 2),
    replay_states=(1, 2),
    n_restarts=3,
    screen_iterations=None,
    max_iterations=60,
    n_folds=3,
    agent_starts=1,
)


def switching_mouse(seed=0, mouse="m1", n_problems=4, sessions=4, trials=60) -> pd.DataFrame:
    """A mouse that is a value learner in some sessions and a habit/biased one in others."""
    weights = np.stack([gha.value_weights(KERNEL_Q), gha.habit_weights(3.0, 1.0)])
    holder = []

    def factory(layout):
        policy = gha.FeaturePolicy(
            weights, np.eye(2), np.array([0.5, 0.5]), KERNEL_Q, layout,
            np.random.default_rng([seed, len(holder)]),
        )  # fmt: skip
        holder.append(policy)
        return policy

    played = sim.simulate_policy_mouse(
        factory, layouts(n_problems, sessions, trials, seed), RULES,
        np.random.default_rng(seed), mouse,
    )  # fmt: skip
    return drifting_trial_table(played)


@pytest.fixture(scope="module")
def mouse_table():
    return switching_mouse()


@pytest.fixture(scope="module")
def result(mouse_table):
    return pipe.analyze_mouse(mouse_table, AGENTS, TINY, seed=1)


def test_config_is_validated():
    with pytest.raises(ValueError, match="one state"):
        pipe.Config(n_states=(2, 3))
    with pytest.raises(ValueError, match="replay"):
        pipe.Config(n_states=(1, 2, 3), replay_states=(1, 4))
    with pytest.raises(ValueError, match="restarts"):
        pipe.Config(n_restarts=0)


def test_a_mouse_is_summarized_by_selection_weights_occupancy_and_kernel(result):
    assert result.selected_k == 2
    assert set(result.scores["n_states"]) == {1, 2}
    assert result.readouts["gain_k2"] > 0
    assert result.readouts["best_gain"] == result.readouts["gain_k2"]
    assert result.summary_k == 2
    weights = result.weights
    assert set(weights["n_states"]) == {1, 2}
    two = weights[weights["n_states"] == 2].pivot(index="state", columns="feature", values="weight")
    assert list(two.columns) == list(gha.FEATURES) or set(two.columns) == set(gha.FEATURES)
    assert two.loc[0, "q"] > two.loc[1, "q"]  # states are sorted by the value weight
    assert result.occupancy["occupancy"].sum() == pytest.approx(1.0)
    assert (result.occupancy["dwell"] > 1).all()
    assert set(result.by_trial_in_session["state"]) == {0, 1}
    assert (
        result.by_trial_in_session.groupby("trial_in_session")["probability"]
        .sum()
        .round(6)
        .eq(1.0)
        .all()
    )
    assert set(result.by_problem["state"]) == {0, 1}
    assert {"all", 0, 1} <= set(result.kernel["state"])
    assert set(result.static_levels["model"]) == {"kernel_q", "wsls_habit"}
    assert np.isfinite(result.static_levels["nats"]).all()
    assert result.n_trials == 4 * 4 * 60


def test_static_levels_use_the_scored_trials_of_the_same_folds(result):
    levels = result.static_levels
    scored = result.scores.groupby("n_states")["n_scored"].sum().iloc[0]
    assert (levels["n_scored"] == scored).all()


def test_analysis_is_repeatable(mouse_table):
    first = pipe.analyze_mouse(mouse_table, AGENTS, TINY, seed=1)
    again = pipe.analyze_mouse(mouse_table, AGENTS, TINY, seed=1)
    pd.testing.assert_frame_equal(first.scores, again.scores)


def test_gain_readouts_come_from_the_scores():
    scores = pd.DataFrame(
        [
            dict(fold=f, n_states=k, n_scored=100, log_likelihood=-100 * (0.7 - gain))
            for f in range(3)
            for k, gain in ((1, 0.0), (2, 0.02), (3, 0.05), (4, 0.01))
        ]
    )
    readouts = pipe.gain_readouts(scores, (1, 2, 3))
    assert readouts == {
        "gain_k2": pytest.approx(0.02), "gain_k3": pytest.approx(0.05),
        "best_gain": pytest.approx(0.05),
    }  # fmt: skip
    assert pipe.gain_readouts(scores, (1, 2, 3, 4))["gain_k4"] == pytest.approx(0.01)


def test_switching_scenario_has_a_true_state_for_every_trial(mouse_table):
    rules = RULES
    simulated = pipe.simulate_scenario(mouse_table, AGENTS, "switch_fast", rules, seed=0)
    assert len(simulated.true_states) == len(simulated.table)
    assert set(simulated.true_states) == {0, 1, 2}
    assert simulated.weights.shape == (3, len(gha.FEATURES))
    assert hmm.dwell_times(simulated.transition) == pytest.approx([pipe.FAST_DWELL] * 3)
    slow = pipe.simulate_scenario(mouse_table, AGENTS, "switch_slow", rules, seed=0)
    assert hmm.dwell_times(slow.transition) == pytest.approx([pipe.SLOW_DWELL] * 3)


def test_static_scenarios_have_no_states(mouse_table):
    for scenario in ("static_kernel_q", "static_wsls_habit", "static_blend"):
        simulated = pipe.simulate_scenario(mouse_table, AGENTS, scenario, RULES, seed=0)
        assert simulated.true_states is None and simulated.weights is None
        assert len(simulated.table) == len(mouse_table)


def test_simulation_is_repeatable_and_unknown_scenarios_fail(mouse_table):
    a = pipe.simulate_scenario(mouse_table, AGENTS, "switch_fast", RULES, seed=3)
    b = pipe.simulate_scenario(mouse_table, AGENTS, "switch_fast", RULES, seed=3)
    pd.testing.assert_frame_equal(a.table, b.table)
    with pytest.raises(ValueError, match="scenario"):
        pipe.simulate_scenario(mouse_table, AGENTS, "nope", RULES, seed=0)


def test_recovery_of_a_switching_scenario_reports_truth_and_estimates(mouse_table):
    config = pipe.Config(
        n_states=(1, 2, 3), replay_states=(1, 2, 3), n_restarts=3, screen_iterations=None,
        max_iterations=60, n_folds=3, agent_starts=1,
    )  # fmt: skip
    recovered = pipe.recover_scenario(mouse_table, AGENTS, "switch_fast", RULES, config, seed=2)
    assert recovered.scenario == "switch_fast"
    assert recovered.readouts["selected_k"] >= 1
    assert 0.0 <= recovered.accuracy <= 1.0
    assert {"state", "feature", "true", "recovered"} <= set(recovered.weights.columns)
    assert len(recovered.dwell) == 3
    assert recovered.dwell["true"].to_numpy() == pytest.approx(pipe.FAST_DWELL)


def test_recovery_of_a_static_scenario_has_no_truth_to_match(mouse_table):
    recovered = pipe.recover_scenario(mouse_table, AGENTS, "static_kernel_q", RULES, TINY, seed=2)
    assert recovered.weights.empty and recovered.dwell.empty and np.isnan(recovered.accuracy)
    assert recovered.readouts["best_gain"] < 0.05


def test_a_replay_gives_gain_readouts_of_a_static_agent(mouse_table):
    readouts = pipe.replay_one(mouse_table, AGENTS, RULES, TINY, replicate=0, seed=5, version="x")
    assert set(readouts) >= {"replicate", "gain_k2", "best_gain", "selected_k"}
    assert readouts["replicate"] == 0


def test_replays_and_calibration_across_mice(mouse_table):
    table = pd.concat([mouse_table, switching_mouse(1, "m2")], ignore_index=True)
    agents = {"m1": AGENTS, "m2": AGENTS}
    null = pipe.replay_source(table, agents, TINY, replicates=2, rules=RULES, seed=0)
    assert sorted(null["mouse"].unique()) == ["m1", "m2"]
    assert null.groupby("mouse")["replicate"].apply(list).tolist() == [[0, 1], [0, 1]]
    observed = pd.DataFrame(
        {"mouse": ["m1", "m2"], "gain_k2": [0.05, 0.05], "best_gain": [0.05, 0.05]}
    )
    calibration = pipe.calibrate_gains(observed, null)
    assert set(calibration.table["readout"]) == {"gain_k2", "best_gain"}
    assert (calibration.table["n_null"] == 2).all()
    assert set(calibration.table["predicted"]) == {1}
    assert {"readout", "n", "mean", "se", "p_wilcoxon"} <= set(calibration.summary.columns)


def test_session_data_and_folds_are_consistent(mouse_table):
    data = feats.session_arrays(mouse_table, KERNEL_Q)
    assert len(feats.assign_folds(data, 3)) == len(data.sessions)
