import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.tasks.gp_bandit import (
    ArmState,
    best_run_lengths,
    build_arm_values,
    draw_block_boundaries,
    format_session_title,
    gap_best_minus_second,
    initial_arm_state,
    plot_example_session_gp,
    plot_session_panel,
    simulate_3armed_blockswaps_gp,
    summarize_session,
    swap_arms,
    trial_arm_values,
)

# ---------------------------------------------------------------------------
# Block boundaries
# ---------------------------------------------------------------------------


def test_draw_block_boundaries_are_increasing_and_inside_session():
    b = draw_block_boundaries(np.random.default_rng(0), 5, 8, 1000, T=100)
    assert np.all(np.diff(b) >= 5)
    assert np.all(np.diff(b) <= 8)
    assert b[0] >= 5
    assert b.max() < 100


def test_draw_block_boundaries_empty_when_first_block_exceeds_session():
    b = draw_block_boundaries(np.random.default_rng(0), 50, 60, 10, T=30)
    assert b.size == 0


def test_draw_block_boundaries_consumes_one_integers_draw():
    rng_a, rng_b = np.random.default_rng(3), np.random.default_rng(3)
    draw_block_boundaries(rng_a, 2, 4, 7, T=50)
    rng_b.integers(2, 5, size=7)
    assert rng_a.random() == rng_b.random()


def test_draw_block_boundaries_rejects_empty_length_range():
    with pytest.raises(ValueError):
        draw_block_boundaries(np.random.default_rng(0), 5, 3, 10, T=50)


# ---------------------------------------------------------------------------
# Arm state
# ---------------------------------------------------------------------------


def test_initial_arm_state_has_arm_two_inactive():
    s = initial_arm_state()
    assert s.inactive_arm == 2
    assert s.active_arms == (0, 1)
    assert s.arm_to_gp == {0: 0, 1: 1}


def test_trial_arm_values_places_streams_on_active_arms_only():
    row = trial_arm_values(initial_arm_state(), np.array([3.0, 4.0]))
    np.testing.assert_array_equal(row, [3.0, 4.0, 0.0])


def test_trial_arm_values_follows_gp_assignment_after_swap():
    state = ArmState(inactive_arm=0, active_arms=(2, 1), arm_to_gp={1: 1, 2: 0})
    row = trial_arm_values(state, np.array([3.0, 4.0]))
    np.testing.assert_array_equal(row, [0.0, 4.0, 3.0])


def test_swap_arms_best_arm_goes_inactive_and_old_inactive_inherits_its_stream():
    # arm 1 (gp 1) is best, so it leaves; arm 2 re-enters with gp 1.
    state = swap_arms(initial_arm_state(), np.array([3.0, 5.0, 0.0]))
    assert state.inactive_arm == 1
    assert state.active_arms == (2, 0)
    assert state.arm_to_gp == {0: 0, 2: 1}


def test_swap_arms_ties_pick_lowest_arm_index():
    state = swap_arms(initial_arm_state(), np.array([4.0, 4.0, 0.0]))
    assert state.inactive_arm == 0
    assert state.active_arms == (2, 1)
    assert state.arm_to_gp == {1: 1, 2: 0}


def test_swap_arms_does_not_mutate_input_state():
    s = initial_arm_state()
    swap_arms(s, np.array([3.0, 5.0, 0.0]))
    assert s == initial_arm_state()


def test_swap_arms_when_inactive_arm_is_best_keeps_it_inactive_and_reactivates_it():
    # All active values are negative (raw GP), so the inactive arm's 0 wins the
    # argmax; the inactive arm then appears in both the inactive and active slots.
    state = swap_arms(initial_arm_state(), np.array([-1.0, -2.0, 0.0]))
    assert state.inactive_arm == 2
    assert state.active_arms == (2, 0)
    assert state.arm_to_gp == {0: 0, 2: 0}


def test_swap_arms_fallback_arm_without_gp_assignment_raises_key_error():
    # No active arm differs from the best arm, so the fallback picks arm 0, which
    # has no GP stream assigned in this state.
    state = ArmState(inactive_arm=2, active_arms=(2, 2), arm_to_gp={2: 0})
    with pytest.raises(KeyError):
        swap_arms(state, np.array([-1.0, -2.0, 0.0]))


# ---------------------------------------------------------------------------
# Session values
# ---------------------------------------------------------------------------


def test_build_arm_values_without_boundaries_leaves_third_arm_at_zero():
    streams = np.arange(10, dtype=float).reshape(5, 2)
    vals3 = build_arm_values(streams, np.array([], dtype=int))
    np.testing.assert_array_equal(vals3[:, :2], streams)
    np.testing.assert_array_equal(vals3[:, 2], np.zeros(5))


def test_build_arm_values_swaps_after_boundary_trial():
    # arm 1 is best at trial 1 (boundary 2 -> swap after trial index 1).
    streams = np.array([[1.0, 2.0], [1.0, 5.0], [7.0, 8.0], [7.0, 8.0]])
    vals3 = build_arm_values(streams, np.array([2]))
    np.testing.assert_array_equal(vals3[:2], [[1, 2, 0], [1, 5, 0]])
    # arm 1 left; arm 2 now carries gp 1, arm 0 keeps gp 0.
    np.testing.assert_array_equal(vals3[2:], [[7, 0, 8], [7, 0, 8]])


def test_build_arm_values_boundary_at_or_beyond_T_is_ignored():
    streams = np.ones((3, 2))
    vals3 = build_arm_values(streams, np.array([3, 10]))
    np.testing.assert_array_equal(vals3[:, 2], np.zeros(3))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_gap_best_minus_second_on_known_values():
    vals3 = np.array([[5.0, 3.0, 0.0], [2.0, 2.0, 0.0], [1.0, 4.0, 3.0], [1.0, 1.0, 5.0]])
    np.testing.assert_array_equal(gap_best_minus_second(vals3), [2.0, 0.0, 1.0, 4.0])


def test_gap_best_minus_second_all_equal_has_zero_gap():
    np.testing.assert_array_equal(gap_best_minus_second(np.full((3, 3), 2.0)), np.zeros(3))


def test_best_run_lengths_splits_at_changes():
    np.testing.assert_array_equal(best_run_lengths(np.array([0, 0, 1, 1, 1, 0])), [2, 3, 1])


def test_best_run_lengths_constant_series_is_one_run():
    np.testing.assert_array_equal(best_run_lengths(np.zeros(6, dtype=int)), [6])


def test_best_run_lengths_single_trial():
    np.testing.assert_array_equal(best_run_lengths(np.array([2])), [1])


def test_best_run_lengths_alternating():
    np.testing.assert_array_equal(best_run_lengths(np.array([0, 1, 0, 1])), [1, 1, 1, 1])


# ---------------------------------------------------------------------------
# Simulation (end to end, synthetic)
# ---------------------------------------------------------------------------


def test_simulation_is_reproducible_and_returns_session_on_request():
    kwargs = dict(T=80, sig=6.0, max_block_len=20)
    a = simulate_3armed_blockswaps_gp(rng=np.random.default_rng(1), **kwargs)
    b = simulate_3armed_blockswaps_gp(rng=np.random.default_rng(1), return_session=True, **kwargs)
    assert len(a) == 4
    assert b[:4] == a
    assert set(b[4]) == {
        "vals3",
        "boundaries",
        "best_arm_series",
        "best_minus_second",
        "gp_int",
        "gp_raw",
    }
    assert b[4]["vals3"].shape == (80, 3)


def test_simulation_default_min_block_len_is_ten_below_max():
    kwargs = dict(T=100, sig=5.0, max_block_len=30)
    a = simulate_3armed_blockswaps_gp(rng=np.random.default_rng(2), **kwargs)
    b = simulate_3armed_blockswaps_gp(rng=np.random.default_rng(2), min_block_len=20, **kwargs)
    assert a == b


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def test_format_session_title_reports_parameters_and_metrics():
    t = format_session_title(10.0, 40, (1.234, 0.5, 12.0, 3.456))
    assert t == (
        "Example GP session | sig=10.00, max_block_len=40 "
        "| gap_mean=1.23, frac≥2=0.50, run_mean=12.00, run_std=3.46"
    )


def test_plot_session_panel_draws_into_given_axes_without_new_figure():
    fig, ax = plt.subplots()
    vals3 = np.tile(np.array([1.0, 2.0, 3.0]), (10, 1))
    out = plot_session_panel(ax, vals3, np.array([4, 8]), clip=(1, 5), use_raw=False)
    assert out is ax
    assert len(plt.get_fignums()) == 1
    assert [ln.get_label() for ln in ax.lines[:3]] == ["Arm 0", "Arm 1", "Arm 2"]
    # one legend entry for all block lines, drawn at b - 0.5
    block_lines = [ln for ln in ax.lines if ln.get_label() == "Block"]
    assert len(block_lines) == 1
    assert len(ax.lines) == 5
    assert ax.get_xlim() == (0, 9)
    assert ax.get_ylabel() == "Reward Magnitude"
    plt.close(fig)


def test_plot_session_panel_marks_choices_on_chosen_arm_value():
    fig, ax = plt.subplots()
    vals3 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 1.0]])
    plot_session_panel(
        ax, vals3, np.array([], dtype=int), clip=(1, 5), use_raw=False, choices=[2, 1]
    )
    np.testing.assert_array_equal(ax.collections[0].get_offsets(), [[0, 3.0], [1, 5.0]])
    plt.close(fig)


def test_plot_session_panel_raw_axis_follows_data_max():
    fig, ax = plt.subplots()
    vals3 = np.array([[1.0, 7.5, 0.0], [2.0, 1.0, 0.0]])
    plot_session_panel(ax, vals3, np.array([], dtype=int), clip=(1, 5), use_raw=True)
    assert ax.get_ylim() == (-0.5, 8.0)
    assert ax.get_ylabel() == "Latent value"
    plt.close(fig)


def test_plot_example_session_gp_returns_session_and_shows_one_figure(monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.get_fignums()))
    session = plot_example_session_gp(sig=5.0, max_block_len=20, T=60, seed=1)
    assert session["vals3"].shape == (60, 3)
    assert len(shown) == 1 and len(shown[0]) == 1
    plt.close("all")


def test_summarize_session_metrics_on_known_values():
    # Arm 0 is best on trials 0-1 and arm 1 on trials 2-3: runs of 2 and 2.
    vals3 = np.array([[5.0, 1.0, 0.0], [4.0, 1.0, 0.0], [1.0, 3.0, 0.0], [1.0, 2.0, 0.0]])
    (gap, frac, run_mean, run_std), session = summarize_session(
        vals3, np.array([2]), np.zeros((4, 2)), np.zeros((4, 2))
    )
    assert gap == (4 + 3 + 2 + 1) / 4
    assert frac == 0.75
    assert (run_mean, run_std) == (2.0, 0.0)
    np.testing.assert_array_equal(session["best_arm_series"], [0, 0, 1, 1])
    np.testing.assert_array_equal(session["boundaries"], [2])


def test_summarize_session_single_trial_has_one_run():
    (_, _, run_mean, run_std), _ = summarize_session(
        np.array([[3.0, 1.0, 0.0]]), np.array([], dtype=int), np.zeros((1, 2)), np.zeros((1, 2))
    )
    assert (run_mean, run_std) == (1.0, 0.0)
