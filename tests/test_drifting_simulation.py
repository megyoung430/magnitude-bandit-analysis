"""The simulated drifting task and the agents that play it."""

import numpy as np
import pytest

from src.agents import drifting_simulation as sim
from src.behavior_analysis.drifting_task_rules import block_changes, complete_block_lengths
from src.behavior_analysis.drifting_trial_table import drifting_trial_table
from src.behavior_analysis.get_variables_across_sessions import get_vars_across_all_sessions

LAYOUT = sim.MouseLayout(
    ports=["A1", "B3", "C2"],
    initiation="B2",
    start_magnitudes={"A1": 4, "B3": 1, "C2": 0},
    session_lengths=[80, 0, 120, 60],
)
RULES = sim.TaskRules(drift_rate=0.15, block_lengths=np.array([20, 25]))
PARAMS = (0.5, 1.0, 1.0, 0.5)


def test_agent_rewards():
    rewards = np.array([0, 1, 4])
    assert sim.agent_rewards(rewards, "magnitude").tolist() == [0, 1, 4]
    assert sim.agent_rewards(rewards, "binary").tolist() == [0, 1, 1]
    with pytest.raises(ValueError, match="unknown agent kind"):
        sim.agent_rewards(rewards, "other")


def test_drift_keeps_values_distinct_and_in_range():
    rng = np.random.default_rng(0)
    mags = {"a": 4, "b": 1, "c": 0}
    for _ in range(2000):
        mags = sim.drift_step(mags, 0.5, rng)
        assert mags["c"] == 0
        assert {mags["a"], mags["b"]} <= set(sim.REWARD_VALUES)
        assert mags["a"] != mags["b"]
    still = sim.drift_step({"a": 4, "b": 1, "c": 0}, 0.0, rng)
    assert still == {"a": 4, "b": 1, "c": 0}


def test_block_change_zeroes_the_most_chosen_arm_which_hands_over_its_value():
    rng = np.random.default_rng(0)
    new = sim.block_change({"a": 4, "b": 2, "c": 0}, {"a": 3, "b": 7}, rng)
    assert new == {"a": 4, "b": 0, "c": 2}
    tied = {
        tuple(sorted(sim.block_change({"a": 4, "b": 2, "c": 0}, {"a": 1, "b": 1}, rng).items()))
        for _ in range(50)
    }
    assert len(tied) == 2


def test_calibrated_attempt_rate_reproduces_the_observed_rate():
    attempt = sim.calibrate_drift_rate(0.1, seed=1)
    assert attempt > 0.1
    observed = sim.observed_drift_rate(attempt, np.random.default_rng(2), n_trials=40000)
    assert observed == pytest.approx(0.1, abs=0.006)
    with pytest.raises(ValueError):
        sim.calibrate_drift_rate(0.0)


def test_simulated_problem_round_trips_through_the_importer_and_follows_the_rules():
    rng = np.random.default_rng(5)
    sessions = sim.simulate_problem(PARAMS, "magnitude", LAYOUT, RULES, rng)
    assert [len(s.trials) for s in sessions] == LAYOUT.session_lengths
    imported = sim.simulated_sessions("agent", 1, LAYOUT, sessions)
    assert len(imported) == 3  # the empty session is skipped
    table = drifting_trial_table({1: {"agent": imported}})
    assert len(table) == sum(LAYOUT.session_lengths)
    assert table["magnitudes"].iloc[0] == LAYOUT.start_magnitudes
    changes = block_changes(table)
    assert len(changes) > 5
    assert set(changes["zeroed_choice_rank"]) <= {"most", "tie"}
    assert changes["inherits_value"].all()
    assert set(complete_block_lengths(changes)) <= {20, 25}


def test_simulation_is_reproducible_from_the_generator():
    a = sim.simulate_problem(PARAMS, "binary", LAYOUT, RULES, np.random.default_rng(9))
    b = sim.simulate_problem(PARAMS, "binary", LAYOUT, RULES, np.random.default_rng(9))
    assert a == b


def test_fit_and_layout_from_a_simulated_mouse():
    sessions = sim.simulate_problem(PARAMS, "magnitude", LAYOUT, RULES, np.random.default_rng(4))
    table = drifting_trial_table({1: {"m": sim.simulated_sessions("m", 1, LAYOUT, sessions)}})
    layout = sim.mouse_layouts(table)[1]
    assert layout.ports == LAYOUT.ports
    assert layout.session_lengths == [80, 120, 60]
    fits = [sim.fit_agent("m", table, kind, n_starts=2, seed=0) for kind in sim.AGENT_KINDS]
    fits_table = sim.fits_table(fits).set_index("kind")
    assert fits_table.loc["magnitude", "delta_nll_vs_magnitude"] == 0
    assert np.isfinite(fits_table.loc["binary", "delta_nll_vs_magnitude"])
    assert fits_table.loc["magnitude", "n_trials"] == 260


def test_block_change_at_a_session_boundary_survives_the_importer():
    # every block ends on a session's last trial, so every block change is at a boundary
    layout = LAYOUT._replace(session_lengths=[30] * 5)
    rules = sim.TaskRules(drift_rate=0.1, block_lengths=np.array([30]))
    sessions = sim.simulate_problem(PARAMS, "magnitude", layout, rules, np.random.default_rng(0))
    imported = sim.simulated_sessions("a", 1, layout, sessions)
    table = drifting_trial_table({1: {"a": imported}})
    changes = block_changes(table)
    assert len(changes) == 4
    assert set(table.set_index("trial").loc[changes["trial"], "trial_in_session"]) == {0}
    merged, _ = get_vars_across_all_sessions({"a": imported})
    blocks = np.asarray(merged["a"]["blocks"])
    assert np.flatnonzero(np.diff(blocks)).tolist() == [t - 1 for t in changes["trial"]]


def test_rank_counts_accumulate_within_a_session():
    sessions = sim.simulate_problem(PARAMS, "magnitude", LAYOUT, RULES, np.random.default_rng(1))
    trials = sessions[0].trials
    assert sum(trials[-1]["rank_counts"].values()) == len(trials)
    assert sessions[2].trials[0]["rank_counts"] == {
        r: int(r == sessions[2].trials[0]["chosen_rank"]) for r in sim.RANKS
    }


def test_parameters_recovered_through_the_importer():
    layout = LAYOUT._replace(session_lengths=[150] * 12)
    for kind, true in (("magnitude", (0.3, 1.2, 1.0, 0.4)), ("binary", (0.3, 3.0, 1.0, 0.4))):
        sessions = sim.simulate_problem(true, kind, layout, RULES, np.random.default_rng(7))
        table = drifting_trial_table({1: {"m": sim.simulated_sessions("m", 1, layout, sessions)}})
        fit = sim.fit_agent("m", table, kind, n_starts=3, seed=0)
        assert np.allclose(fit.parameters, true, atol=[0.12, 0.6, 0.25, 0.25]), (kind, fit)


def test_problem_sequences_reset_previous_choice_after_a_gap():
    sessions = sim.simulate_problem(PARAMS, "magnitude", LAYOUT, RULES, np.random.default_rng(2))
    table = drifting_trial_table({1: {"m": sim.simulated_sessions("m", 1, LAYOUT, sessions)}})
    table.loc[5, "after_gap"] = True
    seqs = sim.problem_sequences(table, "magnitude")
    starts = np.flatnonzero(seqs.session_start[0])
    assert 5 in starts
    assert set(starts) == {0, 5, 80, 200}


def test_fits_table_flags_bounds_and_needs_the_magnitude_fit():
    fits = [
        sim.AgentFit("m", "magnitude", (1.0, 1.0, 0.5, 0.2), -100.0, 100),
        sim.AgentFit("m", "binary", (0.5, 1.0, 0.5, 0.2), -110.0, 100),
    ]
    table = sim.fits_table(fits).set_index("kind")
    assert table.loc["magnitude", "at_bounds"] == "learning_rate"
    assert table.loc["binary", "at_bounds"] == ""
    assert table.loc["binary", "delta_nll_vs_magnitude"] == 10.0
    with pytest.raises(ValueError, match="no magnitude-agent fit"):
        sim.fits_table(fits[1:])
