"""The GLM-HMM's agent signals, the switching agent that generates them, and the session folds."""

import numpy as np
import pandas as pd
import pytest

from src.agents import drifting_simulation as sim
from src.agents import glm_hmm_agent as gha
from src.agents import kernel_q_agent as kq
from src.agents import null_agents as na
from src.behavior_analysis.data.drifting_trial_table import drifting_trial_table
from src.behavior_analysis.glm_hmm import glm_hmm_features as feats
from tests.strategy_fixtures import HABIT, KERNEL_Q, RULES, WSLS, layouts
from tests.switch_regression_fixtures import table_from_rows

PORTS = ("A1", "B3", "C2")
MAGS = {"A1": 2, "B3": 1, "C2": 0}


def small_table() -> pd.DataFrame:
    """Two sessions of one problem: choices A A B A | B B."""
    spec = [
        (0, 0, False, "A1", 2),
        (0, 1, False, "A1", 0),
        (0, 2, False, "B3", 3),
        (0, 3, False, "A1", 1),
        (1, 0, False, "B3", 1),
        (1, 1, False, "B3", 0),
    ]
    rows = [((1, 1), s, t, gap, c, r, MAGS) for s, t, gap, c, r in spec]
    return table_from_rows(rows)


def played_table(policy_factory, seed=0, n_problems=3, sessions=3, trials=60):
    played = sim.simulate_policy_mouse(
        policy_factory, layouts(n_problems, sessions, trials, seed), RULES,
        np.random.default_rng(seed), "sim",
    )  # fmt: skip
    return drifting_trial_table(played)


def test_signals_on_a_hand_built_table():
    data = feats.session_arrays(small_table(), KERNEL_Q)
    X = data.sequences.X
    f = {name: i for i, name in enumerate(feats.FEATURES)}
    assert X.shape == (2, 4, 3, len(feats.FEATURES))
    # session start: no previous choice, so no stay or trace signal
    for name in ("stay_win", "stay_magnitude", "stay_zero", "switch_trace", "habit"):
        assert np.all(X[0, 0, :, f[name]] == 0)
    # trial 1: previous choice A paid 2
    assert X[0, 1, :, f["stay_win"]].tolist() == [1, 0, 0]
    assert X[0, 1, :, f["stay_magnitude"]].tolist() == [1, 0, 0]
    assert X[0, 1, :, f["stay_zero"]].tolist() == [0, 0, 0]
    # trial 2: previous choice A paid 0; the other arms have no trace to compare
    assert X[0, 2, :, f["stay_zero"]].tolist() == [1, 0, 0]
    assert X[0, 2, :, f["switch_trace"]].tolist() == [0, 0, 0]
    # trial 3: previous choice B paid 3; A was chosen on trials 0 and 1, so its trace is
    # 0.9 * (0.9 * 1 + 1) = 1.71 against 0 for C, centered on their mean; the stay arm B has none
    assert X[0, 3, :, f["stay_magnitude"]].tolist() == [0, 2, 0]
    assert X[0, 3, :, f["switch_trace"]] == pytest.approx([0.9 * 1.9 / 2, 0, -0.9 * 1.9 / 2])
    # the choice kernel is cleared at the next session start, the value carries over
    assert np.all(X[1, 0, :, f["habit"]] == 0)
    assert X[1, 0, 0, f["q"]] > 0
    np.testing.assert_array_equal(data.sequences.chosen[0, :4], [0, 0, 1, 0])
    np.testing.assert_array_equal(data.sequences.scored, [[0, 1, 1, 1], [0, 1, 0, 0]])
    np.testing.assert_array_equal(data.sequences.mask, [[1, 1, 1, 1], [1, 1, 0, 0]])


def test_a_gap_starts_a_segment_but_not_a_session():
    rows = [((1, 1), 0, t, t == 2, c, 1, MAGS) for t, c in enumerate(["A1", "B3", "B3", "B3"])]
    data = feats.session_arrays(table_from_rows(rows), KERNEL_Q)
    assert data.sequences.X.shape[0] == 1
    np.testing.assert_array_equal(data.sequences.scored[0], [0, 1, 0, 1])
    f = feats.FEATURES.index("stay_win")
    assert data.sequences.X[0, 2, :, f].tolist() == [0, 0, 0]


def test_value_weights_reproduce_the_kernel_q_likelihood():
    table = played_table(lambda layout: sim.KernelQPolicy(KERNEL_Q, "magnitude", layout))
    data = feats.session_arrays(table, KERNEL_Q)
    weights = gha.value_weights(KERNEL_Q, habit=KERNEL_Q[kq.INDEX["beta_k"]])
    X, y = data.sequences.X[data.sequences.mask], data.sequences.chosen[data.sequences.mask]
    utility = X @ weights
    log_p = utility[np.arange(len(y)), y] - np.log(np.exp(utility).sum(axis=1))
    expected = kq.log_likelihood(KERNEL_Q, sim.problem_sequences(table, "magnitude"))
    assert log_p.sum() == pytest.approx(expected)


def test_wsls_weights_reproduce_the_wsls_likelihood_with_uniform_switches():
    uniform_switches = na.WslsMagnitudeParams(WSLS.a_win, WSLS.a_lose, WSLS.b, w=0.0)
    table = played_table(lambda layout: na.NullPolicy(uniform_switches))
    data = feats.session_arrays(table, KERNEL_Q)
    weights = gha.wsls_weights(uniform_switches)
    X, y = data.sequences.X[data.sequences.mask], data.sequences.chosen[data.sequences.mask]
    utility = X @ weights
    log_p = utility[np.arange(len(y)), y] - np.log(np.exp(utility).sum(axis=1))
    assert log_p.sum() == pytest.approx(na.log_likelihood(table, uniform_switches))


def test_policy_features_equal_the_batch_features():
    weights = np.stack([gha.value_weights(KERNEL_Q), gha.wsls_weights(WSLS)])
    holder = []

    def factory(layout):
        policy = gha.FeaturePolicy(
            weights, np.full((2, 2), 0.5), np.full(2, 0.5), KERNEL_Q, layout,
            np.random.default_rng(len(holder)),
        )  # fmt: skip
        holder.append(policy)
        return policy

    table = played_table(factory)
    data = feats.session_arrays(table, KERNEL_Q)
    logged = np.concatenate([p.logged_features() for p in holder])
    np.testing.assert_allclose(logged, data.sequences.X[data.sequences.mask])
    states = np.concatenate([p.true_states() for p in holder])
    assert len(states) == len(table) == data.sequences.mask.sum()


def test_the_switching_agent_resets_its_state_at_a_session_start():
    weights = np.stack([gha.value_weights(KERNEL_Q), gha.wsls_weights(WSLS)])
    transition = np.array([[1.0, 0.0], [0.5, 0.5]])
    holder = []

    def factory(layout):
        policy = gha.FeaturePolicy(
            weights, transition, np.array([0.0, 1.0]), KERNEL_Q, layout,
            np.random.default_rng(len(holder)),
        )  # fmt: skip
        holder.append(policy)
        return policy

    table = played_table(factory, trials=80)
    for policy in holder:
        for session_states in policy.sessions_true_states():
            assert session_states[0] == 1  # always drawn from `initial`, never carried over
    first_rows = table["trial_in_session"] == 0
    assert first_rows.sum() == sum(len(p.sessions_true_states()) for p in holder)


def test_the_switching_agent_dwells_as_long_as_its_transition_matrix_says():
    weights = np.stack([gha.value_weights(KERNEL_Q), gha.wsls_weights(WSLS)])
    stay = 1 - 1 / 20
    transition = np.array([[stay, 1 - stay], [1 - stay, stay]])
    holder = []

    def factory(layout):
        policy = gha.FeaturePolicy(
            weights,
            transition,
            np.full(2, 0.5),
            KERNEL_Q,
            layout,
            np.random.default_rng(len(holder)),
        )
        holder.append(policy)
        return policy

    played_table(factory, n_problems=4, sessions=4, trials=200)
    stays = np.concatenate([(s[1:] == s[:-1]) for p in holder for s in p.sessions_true_states()])
    assert stays.mean() == pytest.approx(stay, abs=0.01)


def test_wsls_and_value_agents_choose_differently():
    p_stay = []
    for weights in (gha.value_weights(KERNEL_Q), gha.wsls_weights(HABIT)):
        table = played_table(
            lambda layout, w=weights: gha.FeaturePolicy(
                w[None, :], np.ones((1, 1)), np.ones(1), KERNEL_Q, layout, np.random.default_rng(0)
            ),
            trials=100,
        )
        chosen = table["choice"].to_numpy()
        same = (chosen[1:] == chosen[:-1]) & (table["trial_in_session"].to_numpy()[1:] > 0)
        p_stay.append(same.mean())
    assert abs(p_stay[0] - p_stay[1]) > 0.02


def test_stationary_blend_is_the_mean_of_the_states():
    states = np.array([[1.0, 0, 3], [2.0, 4, 0]])
    np.testing.assert_allclose(gha.blend_weights(states), [1.5, 2, 1.5])


def test_folds_spread_every_problem_over_the_folds():
    table = pd.concat(
        [
            table_from_rows(
                [((1, p), s, t, False, "A1", 1, MAGS) for s in range(sessions) for t in range(3)]
            )
            for p, sessions in ((1, 7), (2, 3), (3, 6))
        ],
        ignore_index=True,
    )
    data = feats.session_arrays(table, KERNEL_Q)
    folds = feats.assign_folds(data, 5)
    keys = data.sessions
    assert set(folds) == set(range(5))
    for _, group in keys.groupby("problem"):
        assert len(set(folds[group.index])) == min(len(group), 5)
    counts = np.bincount(folds)
    assert counts.max() - counts.min() <= 1
    np.testing.assert_array_equal(folds, feats.assign_folds(data, 5))


def test_folds_need_at_least_as_many_sessions_as_folds():
    data = feats.session_arrays(small_table(), KERNEL_Q)
    with pytest.raises(ValueError, match="sessions"):
        feats.assign_folds(data, 5)


def test_subset_keeps_the_chosen_sessions_and_trims_padding():
    data = feats.session_arrays(small_table(), KERNEL_Q)
    sub = feats.subset(data.sequences, np.array([1]))
    assert sub.X.shape[:2] == (1, 2)
    np.testing.assert_array_equal(sub.chosen[0], data.sequences.chosen[1, :2])


def test_one_mouse_and_known_ports_are_required():
    mixed = pd.concat([small_table(), small_table().assign(mouse="m2")], ignore_index=True)
    with pytest.raises(ValueError, match="one mouse"):
        feats.session_arrays(mixed, KERNEL_Q)
    odd = small_table().assign(choice="Z9")
    with pytest.raises(ValueError, match="offered"):
        feats.session_arrays(odd, KERNEL_Q)
    with pytest.raises(ValueError, match="seven"):
        feats.session_arrays(small_table(), KERNEL_Q[:4])


def test_trial_index_maps_posteriors_back_to_rows():
    data = feats.session_arrays(small_table(), KERNEL_Q)
    index = data.trials
    assert len(index) == 6
    assert index["session"].tolist() == [0, 0, 0, 0, 1, 1]
    assert index["step"].tolist() == [0, 1, 2, 3, 0, 1]
    assert index["trial"].tolist() == [0, 1, 2, 3, 4, 5]


def test_wsls_weights_keep_the_switch_target_rule_with_a_trace_weight():
    # the split between the two other arms is exactly expit(w * trace difference)
    table = played_table(lambda layout: na.NullPolicy(WSLS), trials=80)
    data = feats.session_arrays(table, KERNEL_Q)
    f = feats.FEATURES.index("switch_trace")
    seqs = data.sequences
    X, mask = seqs.X, seqs.mask
    live = mask & (np.abs(X[..., f]).sum(axis=-1) > 0)
    weights = gha.wsls_weights(WSLS)
    utility = X[live] @ weights
    others = X[live][..., f] != 0
    first = [np.flatnonzero(o)[0] for o in others]
    second = [np.flatnonzero(o)[1] for o in others]
    rows = np.arange(len(first))
    to_first = 1 / (1 + np.exp(-(utility[rows, first] - utility[rows, second])))
    delta = X[live][rows, first, f] - X[live][rows, second, f]
    np.testing.assert_allclose(to_first, 1 / (1 + np.exp(-WSLS.w * delta)))
