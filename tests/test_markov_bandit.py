"""Unit tests for the pure helpers behind the Markov bandit simulation.

All inputs are tiny synthetic arrays, so nothing depends on lab data.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.tasks.markov_bandit import (
    ChainSpec,
    advance_arms,
    best_run_lengths,
    build_session,
    build_vals3,
    draw_block_boundaries,
    draw_initial_states,
    example_session_title,
    gap_series,
    plot_session_values,
    resolve_collision,
    sample_direction,
    sample_two_markov_bandits,
    session_metrics,
    simulate_3armed_blockswaps_markov,
    simulate_example_session,
    step_in_direction,
    summarise_sweep_cell,
    swap_arm_roles,
    sweep_markov_metrics,
    update_hold,
    validate_chain_args,
)


class FixedRandom:
    """Stand-in generator whose ``random()`` returns a preset value."""

    def __init__(self, value: float):
        self.value = value

    def random(self) -> float:
        return self.value


def spec(n_states=4, pud=0.2, min_hold=1, enforce_distinct=True) -> ChainSpec:
    return ChainSpec(n_states, pud, 1 - 2 * pud, min_hold, enforce_distinct)


# --- chain set-up ------------------------------------------------------------


def test_validate_returns_stay_probability():
    assert validate_chain_args(4, 0.2, True) == pytest.approx(0.6)


def test_validate_rejects_large_pud():
    with pytest.raises(ValueError, match="pud"):
        validate_chain_args(4, 0.6, True)


def test_validate_rejects_single_state_only_when_distinct():
    with pytest.raises(ValueError, match="2 states"):
        validate_chain_args(1, 0.1, True)
    validate_chain_args(1, 0.1, False)


@pytest.mark.parametrize("distinct", [True, False])
def test_initial_states_are_valid_indices(distinct):
    states = draw_initial_states(np.random.default_rng(0), 3, distinct)
    assert states.shape == (2,)
    assert ((0 <= states) & (states < 3)).all()
    if distinct:
        assert states[0] != states[1]


# --- single-step moves -------------------------------------------------------


@pytest.mark.parametrize("r, expected", [(0.1, -1), (0.5, 0), (0.9, 1)])
def test_sample_direction_follows_transition_probabilities(r, expected):
    assert sample_direction(FixedRandom(r), pud=0.2, pstay=0.6) == expected


def test_step_in_direction_stay_and_simple_move():
    assert step_in_direction(1, 0, 2, spec()) == 1
    assert step_in_direction(1, 1, 0, spec()) == 2


def test_step_skips_the_other_arms_state():
    assert step_in_direction(1, 1, 2, spec()) == 3


def test_step_reverses_at_the_edge():
    assert step_in_direction(3, 1, 0, spec()) == 2


def test_step_ignores_other_arm_when_not_distinct():
    assert step_in_direction(1, 1, 2, spec(enforce_distinct=False)) == 2


def test_step_raises_when_both_directions_blocked():
    with pytest.raises(RuntimeError, match="collision"):
        step_in_direction(0, 1, 1, spec(n_states=2))


def test_resolve_collision_moves_arm_zero_up_or_down():
    np.testing.assert_array_equal(resolve_collision(np.array([1, 1]), 4), [2, 1])
    np.testing.assert_array_equal(resolve_collision(np.array([3, 3]), 4), [2, 3])


def test_resolve_collision_does_not_mutate_input():
    nxt = np.array([1, 1])
    resolve_collision(nxt, 4)
    np.testing.assert_array_equal(nxt, [1, 1])


def test_resolve_collision_raises_with_one_state():
    with pytest.raises(RuntimeError, match="Distinctness"):
        resolve_collision(np.array([0, 0]), 1)


def test_update_hold_resets_changed_arms_and_increments_others():
    hold = update_hold(np.array([3, 4]), prev=np.array([1, 2]), new=np.array([1, 3]))
    np.testing.assert_array_equal(hold, [4, 1])


def test_advance_arms_respects_min_hold():
    cur = np.array([0, 3])
    nxt = advance_arms(np.random.default_rng(0), cur, np.array([1, 1]), spec(pud=0.5, min_hold=5))
    np.testing.assert_array_equal(nxt, cur)


def test_advance_arms_moves_when_hold_is_satisfied():
    cur = np.array([1, 2])
    nxt = advance_arms(np.random.default_rng(0), cur, np.array([9, 9]), spec(pud=0.5))
    assert (nxt != cur).all()
    assert ((0 <= nxt) & (nxt < 4)).all()


# --- sample_two_markov_bandits -----------------------------------------------


def test_sampled_streams_properties():
    vals = (10, 20, 30, 40)
    out = sample_two_markov_bandits(200, 0.25, vals, np.random.default_rng(1), min_hold=4)
    assert out.shape == (200, 2)
    assert set(np.unique(out)) <= set(vals)
    assert (out[:, 0] != out[:, 1]).all()
    for arm in range(2):
        change_at = np.flatnonzero(np.diff(out[:, arm])) + 1
        dwell = np.diff(np.r_[0, change_at])
        assert (dwell >= 4).all()


def test_sampling_is_reproducible_given_a_seed():
    a = sample_two_markov_bandits(50, 0.1, (1, 2, 3), np.random.default_rng(7))
    b = sample_two_markov_bandits(50, 0.1, (1, 2, 3), np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_sampling_rejects_bad_arguments():
    with pytest.raises(ValueError):
        sample_two_markov_bandits(10, 0.7, (1, 2, 3), np.random.default_rng(0))
    with pytest.raises(ValueError):
        sample_two_markov_bandits(10, 0.1, (1,), np.random.default_rng(0))


# --- simulate_3armed_blockswaps_markov ----------------------------------------


def test_block_boundaries_are_increasing_and_inside_session():
    b = draw_block_boundaries(np.random.default_rng(0), 100, 10, 20, 1000)
    assert (np.diff(b) > 0).all()
    assert b.max() < 100
    assert b.min() >= 10


def test_swap_arm_roles_hands_best_arm_stream_to_old_inactive_arm():
    inactive, active, stream = swap_arm_roles(
        np.array([1.0, 3.0, 0.0]), inactive_arm=2, active_arms=[0, 1], arm_to_stream={0: 0, 1: 1}
    )
    assert inactive == 1
    assert active == [2, 0]
    assert stream == {0: 0, 2: 1}


def test_build_vals3_swaps_roles_after_each_boundary():
    streams = np.array([[1, 2], [1, 2], [1, 3], [1, 3], [1, 4], [1, 4]])
    vals3 = build_vals3(streams, np.array([3]))
    expected = np.array(
        [[1, 2, 0], [1, 2, 0], [1, 3, 0], [1, 0, 3], [1, 0, 4], [1, 0, 4]], dtype=float
    )
    np.testing.assert_array_equal(vals3, expected)


def test_build_vals3_without_boundaries_leaves_third_arm_inactive():
    streams = np.array([[1, 2], [2, 1], [3, 1]])
    vals3 = build_vals3(streams, np.array([], dtype=int))
    np.testing.assert_array_equal(vals3, np.c_[streams, np.zeros(3)])


def test_gap_series_is_best_minus_second():
    vals3 = np.array([[1.0, 4.0, 0.0], [2.0, 2.0, 0.0]])
    np.testing.assert_array_equal(gap_series(vals3), [3.0, 0.0])


def test_best_run_lengths_cover_the_session():
    np.testing.assert_array_equal(best_run_lengths(np.array([0, 0, 1, 1, 1, 0])), [2, 3, 1])
    np.testing.assert_array_equal(best_run_lengths(np.array([2])), [1])


def test_session_metrics_summarise_gap_and_runs():
    gap = np.array([1.0, 2.0, 3.0, 0.0])
    best = np.array([0, 0, 1, 1])
    assert session_metrics(gap, best) == (1.5, 0.5, 2.0, 0.0)


def test_build_session_has_expected_keys_and_block_counts():
    vals3 = np.array([[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [1.0, 2.0, 0.0]])
    session = build_session(vals3, np.array([2]), np.argmax(vals3, axis=1), gap_series(vals3))
    assert set(session) == {
        "vals3",
        "boundaries",
        "best_arm_series",
        "best_minus_second",
        "best_second_switches_per_block",
    }
    np.testing.assert_array_equal(session["best_second_switches_per_block"], [1, 1])


def test_simulation_returns_session_consistent_with_metrics():
    *metrics, session = simulate_3armed_blockswaps_markov(
        80, 0.1, 20, rng=np.random.default_rng(3), return_session=True
    )
    assert metrics[0] == pytest.approx(session["best_minus_second"].mean())
    assert session["vals3"].shape == (80, 3)
    assert len(session["best_second_switches_per_block"]) == len(session["boundaries"]) + 1


def test_simulation_is_reproducible_given_a_seed():
    kwargs = dict(T=60, pud=0.2, max_block_len=20)
    a = simulate_3armed_blockswaps_markov(rng=np.random.default_rng(5), **kwargs)
    b = simulate_3armed_blockswaps_markov(rng=np.random.default_rng(5), **kwargs)
    assert a == b


# --- sweep ---------------------------------------------------------------------


def fake_result(gap, ge2, run_mean, run_std, switches):
    return gap, ge2, run_mean, run_std, {"best_second_switches_per_block": np.array(switches)}


def test_summarise_sweep_cell_averages_over_simulations():
    results = [fake_result(1, 0.5, 10, 2, [1, 2]), fake_result(3, 0.5, 20, 4, [3])]
    assert summarise_sweep_cell(results, True) == (2.0, 0.5, 15.0, 3.0, 2.0)


def test_summarise_sweep_cell_skips_switches_when_not_requested():
    results = [(1.0, 0.0, 5.0, 1.0), (3.0, 1.0, 7.0, 1.0)]
    assert summarise_sweep_cell(results, False) == (2.0, 0.5, 6.0, 1.0, 0.0)


def test_sweep_output_shapes_and_switch_option():
    kwargs = dict(
        pud_values=np.array([0.1, 0.2]), max_block_lens=np.array([20, 25, 30]), T=40, n_sims=2
    )
    four = sweep_markov_metrics(**kwargs)
    five = sweep_markov_metrics(include_switches=True, **kwargs)
    assert len(four) == 4 and len(five) == 5
    assert all(h.shape == (3, 2) for h in five)
    np.testing.assert_array_equal(four[0], five[0])


# --- example session -----------------------------------------------------------


def test_example_session_is_seeded():
    m1, s1 = simulate_example_session(0.1, 30, (1, 2, 3, 4), 60, seed=2)
    m2, s2 = simulate_example_session(0.1, 30, (1, 2, 3, 4), 60, seed=2)
    assert m1 == m2
    np.testing.assert_array_equal(s1["vals3"], s2["vals3"])


def test_example_title_uses_custom_title_or_metrics():
    metrics = (1.0, 0.25, 12.0, 3.0)
    assert example_session_title(0.1, 30, metrics, "mine") == "mine"
    default = example_session_title(0.1, 30, metrics, None)
    assert "pud=0.100" in default and "run_std=3.00" in default


def test_plot_session_values_draws_into_given_axes():
    vals3 = np.tile(np.array([1.0, 2.0, 0.0]), (10, 1))
    fig, ax = plt.subplots()
    n_figs = len(plt.get_fignums())
    out = plot_session_values(vals3, np.array([4, 7]), 4, "t", choices=np.zeros(10, int), ax=ax)
    assert out is ax
    assert len(plt.get_fignums()) == n_figs
    assert len(ax.lines) == 3 + 2
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        "Arm 0",
        "Arm 1",
        "Arm 2",
        "Choice",
        "Block",
    ]
    plt.close(fig)


def test_plot_session_values_creates_figure_only_without_axes():
    plt.close("all")
    ax = plot_session_values(np.ones((5, 3)), np.array([], dtype=int), 4, "t")
    assert len(plt.get_fignums()) == 1
    assert ax.figure is plt.gcf()
    plt.close("all")
