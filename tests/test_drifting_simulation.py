"""The simulated drifting task and the agents that play it."""

import hashlib
import json

import numpy as np
import pytest

from src.agents import drifting_simulation as sim
from src.behavior_analysis.data.drifting_task_rules import block_changes, complete_block_lengths
from src.behavior_analysis.data.drifting_trial_table import drifting_trial_table
from src.behavior_analysis.data.get_variables_across_sessions import get_vars_across_all_sessions

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


class _ScriptedPolicy:
    """Always picks arm *arm*; records the sessions it saw and what it was paid."""

    def __init__(self, arm: int):
        self.arm, self.paid, self.session_starts = arm, [], 0

    def start_session(self) -> None:
        self.session_starts += 1

    def probabilities(self) -> np.ndarray:
        return np.eye(3)[self.arm]

    def update(self, action: int, magnitude: float) -> None:
        self.paid.append((action, magnitude))


def test_play_problem_runs_any_policy_through_the_task():
    policy = _ScriptedPolicy(arm=1)
    sessions = sim.play_problem(policy, LAYOUT, RULES, np.random.default_rng(0))
    assert policy.session_starts == len(LAYOUT.session_lengths)
    assert [len(s.trials) for s in sessions] == LAYOUT.session_lengths
    assert {t["chosen_tow"] for s in sessions for t in s.trials} == {"B3"}
    first = sessions[0].trials[0]
    assert policy.paid[0] == (1, first["curr_rew_mag"]["B3"])


# Digests of the value learner's simulations as produced by ``simulate_problem`` and
# ``simulate_mouse`` before agents were played through ``play_problem``; a change in what the
# value learner does (or in its random stream) changes them.
GOLDEN = {
    "problem/magnitude/0": "9bf9126ca12d30ae845689e6b11b7a60",
    "mouse/magnitude/0": "1081a1d5fd6a6525c22ef38fd16e7926",
    "problem/magnitude/1": "c4bf567a84b07f0e48283510491e706b",
    "mouse/magnitude/1": "e504bdcfa339c928b36d5048abd9b652",
    "problem/binary/0": "ba5e26c51e572daa980e94a4f3e06a36",
    "mouse/binary/0": "908e54690cf7283b2b755e3bc7698969",
    "problem/binary/1": "25d893e61744869a21a701b794561ef7",
    "mouse/binary/1": "f7569426dbc7c8a147c317cb3563e6cd",
}
GOLDEN_LAYOUTS = {
    1: LAYOUT,
    2: LAYOUT._replace(start_magnitudes={"A1": 0, "B3": 2, "C2": 3}),
}


def _digest(payload) -> str:
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize("kind", ["magnitude", "binary"])
@pytest.mark.parametrize("seed", [0, 1])
def test_value_learner_simulations_match_the_recorded_output(kind, seed):
    sessions = sim.simulate_problem(PARAMS, kind, LAYOUT, RULES, np.random.default_rng(seed))
    payload = [[s.trials, s.start, s.end] for s in sessions]
    assert _digest(payload) == GOLDEN[f"problem/{kind}/{seed}"]
    fit = sim.AgentFit("m", kind, PARAMS, 0.0, 0)
    played = sim.simulate_mouse(fit, GOLDEN_LAYOUTS, RULES, np.random.default_rng(seed), "m")
    table = drifting_trial_table(played)
    columns = [table["choice"].tolist(), table["reward"].tolist(), table["after_gap"].tolist()]
    assert _digest(columns) == GOLDEN[f"mouse/{kind}/{seed}"]


def test_simulate_policy_mouse_labels_and_numbers_the_problems():
    layouts = {"p1": LAYOUT, "p2": LAYOUT}
    played = sim.simulate_policy_mouse(
        lambda layout: _ScriptedPolicy(0), layouts, RULES, np.random.default_rng(0), "agent"
    )
    assert list(played) == ["p1", "p2"]
    assert list(played["p2"]) == ["agent"]
    assert {s["problem"] for s in played["p2"]["agent"].values()} == {2}
