import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.tasks.gp_bandit import (
    best_run_lengths,
    build_arm_values,
    build_session,
    draw_block_boundaries,
    first_other_active_arm,
    format_session_title,
    gap_best_minus_second,
    mean_left_to_right,
    plot_example_session_gp,
    plot_session_panel,
    session_metrics,
    simulate_3armed_blockswaps_gp,
    simulate_example_session,
    summarize_sweep_cell,
    swap_arm_roles,
    y_axis_max,
)

# ---------------------------------------------------------------------------
# Block boundaries
# ---------------------------------------------------------------------------


def draw(rng, *, T, lo, hi, n):
    return draw_block_boundaries(rng, T=T, min_block_len=lo, max_block_len=hi, n_blocks_draw=n)


def test_draw_block_boundaries_are_increasing_and_inside_session():
    b = draw(np.random.default_rng(0), T=100, lo=5, hi=8, n=1000)
    assert np.all(np.diff(b) >= 5)
    assert np.all(np.diff(b) <= 8)
    assert b[0] >= 5
    assert b.max() < 100


def test_draw_block_boundaries_empty_when_first_block_exceeds_session():
    b = draw(np.random.default_rng(0), T=30, lo=50, hi=60, n=10)
    assert b.size == 0


def test_draw_block_boundaries_consumes_one_integers_draw():
    rng_a, rng_b = np.random.default_rng(3), np.random.default_rng(3)
    draw(rng_a, T=50, lo=2, hi=4, n=7)
    rng_b.integers(2, 5, size=7)
    assert rng_a.random() == rng_b.random()


def test_draw_block_boundaries_rejects_empty_length_range():
    with pytest.raises(ValueError):
        draw(np.random.default_rng(0), T=50, lo=5, hi=3, n=10)


def test_draw_block_boundaries_requires_keyword_arguments():
    with pytest.raises(TypeError):
        draw_block_boundaries(np.random.default_rng(0), 5, 8, 1000, 100)


# ---------------------------------------------------------------------------
# Arm roles
# ---------------------------------------------------------------------------

START = (2, [0, 1], {0: 0, 1: 1})


def test_first_other_active_arm_skips_the_best_arm():
    assert first_other_active_arm(0, active_arms=[0, 1], old_inactive=2) == 1
    assert first_other_active_arm(1, active_arms=[0, 1], old_inactive=2) == 0


def test_first_other_active_arm_falls_back_to_the_arm_that_is_neither_best_nor_inactive():
    assert first_other_active_arm(2, active_arms=[2, 2], old_inactive=2) == 0
    assert first_other_active_arm(0, active_arms=[0, 0], old_inactive=1) == 2


def test_swap_arm_roles_best_arm_goes_inactive_and_old_inactive_inherits_its_stream():
    # arm 1 (stream 1) is best, so it leaves; arm 2 re-enters with stream 1.
    out = swap_arm_roles(np.array([3.0, 5.0, 0.0]), *START)
    assert out == (1, [2, 0], {0: 0, 2: 1})


def test_swap_arm_roles_ties_pick_lowest_arm_index():
    out = swap_arm_roles(np.array([4.0, 4.0, 0.0]), *START)
    assert out == (0, [2, 1], {1: 1, 2: 0})


def test_swap_arm_roles_does_not_mutate_its_inputs():
    active, mapping = [0, 1], {0: 0, 1: 1}
    swap_arm_roles(np.array([3.0, 5.0, 0.0]), 2, active, mapping)
    assert active == [0, 1] and mapping == {0: 0, 1: 1}


def test_swap_arm_roles_when_inactive_arm_is_best_keeps_it_inactive_and_reactivates_it():
    # All active values are negative (raw GP), so the inactive arm's 0 wins the
    # argmax; the inactive arm then appears in both the inactive and active slots.
    out = swap_arm_roles(np.array([-1.0, -2.0, 0.0]), *START)
    assert out == (2, [2, 0], {0: 0, 2: 0})


def test_swap_arm_roles_fallback_arm_without_stream_raises_key_error():
    # No active arm differs from the best arm, so the fallback picks arm 0, which
    # has no stream assigned in this state.
    with pytest.raises(KeyError) as excinfo:
        swap_arm_roles(np.array([-1.0, -2.0, 0.0]), 2, [2, 2], {2: 0})
    assert excinfo.value.args == (0,)


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
    # arm 1 left; arm 2 now carries stream 1, arm 0 keeps stream 0.
    np.testing.assert_array_equal(vals3[2:], [[7, 0, 8], [7, 0, 8]])


def test_build_arm_values_boundary_at_or_beyond_T_is_ignored():
    streams = np.ones((3, 2))
    vals3 = build_arm_values(streams, np.array([3, 10]))
    np.testing.assert_array_equal(vals3[:, 2], np.zeros(3))


def test_build_arm_values_ignores_non_positive_and_repeated_boundaries():
    streams = np.array([[1.0, 2.0], [1.0, 5.0], [7.0, 8.0], [7.0, 8.0]])
    expected = build_arm_values(streams, np.array([2]))
    np.testing.assert_array_equal(build_arm_values(streams, np.array([0, -3, 2, 2])), expected)


def test_build_arm_values_swaps_twice_in_order():
    # Swap after trial 0 (arm 1 best), then after trial 1 (arm 0, carrying stream 0, best).
    streams = np.array([[1.0, 2.0], [9.0, 3.0], [4.0, 5.0]])
    vals3 = build_arm_values(streams, np.array([1, 2]))
    np.testing.assert_array_equal(vals3, [[1, 2, 0], [9, 0, 3], [0, 4, 5]])


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


def test_session_metrics_on_known_values():
    # Arm 0 is best on trials 0-1 and arm 1 on trials 2-3: runs of 2 and 2.
    vals3 = np.array([[5.0, 1.0, 0.0], [4.0, 1.0, 0.0], [1.0, 3.0, 0.0], [1.0, 2.0, 0.0]])
    gap, frac, run_mean, run_std = session_metrics(
        best_minus_second=gap_best_minus_second(vals3), best_arm_series=np.argmax(vals3, axis=1)
    )
    assert gap == (4 + 3 + 2 + 1) / 4
    assert frac == 0.75
    assert (run_mean, run_std) == (2.0, 0.0)


def test_session_metrics_single_trial_has_one_run():
    _, _, run_mean, run_std = session_metrics(
        best_minus_second=np.array([2.0]), best_arm_series=np.array([0])
    )
    assert (run_mean, run_std) == (1.0, 0.0)


def test_build_session_keeps_the_key_order_and_passes_arrays_through():
    arrays = {k: np.zeros(2) for k in ("vals3", "boundaries", "best_arm_series")}
    session = build_session(
        **arrays,
        best_minus_second=np.ones(2),
        gp_int=np.zeros((2, 2), dtype=int),
        gp_raw=np.zeros((2, 2)),
    )
    assert list(session) == [
        "vals3",
        "boundaries",
        "best_arm_series",
        "best_minus_second",
        "gp_int",
        "gp_raw",
    ]
    assert session["vals3"] is arrays["vals3"]


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
# Sweep summary
# ---------------------------------------------------------------------------


def test_mean_left_to_right_adds_in_order_without_compensation():
    # Plain left-to-right addition rounds 1 + 1e-16 + 1e-16 down to 1.0; a
    # compensated sum (built-in sum on Python >= 3.12) would keep the 2e-16.
    assert mean_left_to_right([1.0, 1e-16, 1e-16]) == 1.0 / 3
    assert mean_left_to_right(iter([2.0, 4.0])) == 3.0


def test_mean_left_to_right_count_overrides_the_divisor():
    assert np.signbit(mean_left_to_right([], count=-2))
    with pytest.raises(ZeroDivisionError):
        mean_left_to_right([], count=0)


def test_summarize_sweep_cell_averages_each_metric():
    results = [(1.0, 0.0, 10.0, 2.0), (3.0, 1.0, 20.0, 4.0)]
    assert summarize_sweep_cell(results, n_sims=2) == (2.0, 0.5, 15.0, 3.0)


# ---------------------------------------------------------------------------
# Example session
# ---------------------------------------------------------------------------

EXAMPLE = dict(
    sig=5.0,
    max_block_len=20,
    T=40,
    seed=1,
    scale=2.0,
    shift=2.5,
    clip=(1, 5),
    jitter=1e-6,
    use_raw=False,
)


def test_simulate_example_session_matches_a_seeded_simulation():
    metrics, session = simulate_example_session(**EXAMPLE)
    *expected, expected_session = simulate_3armed_blockswaps_gp(
        T=40,
        sig=5.0,
        max_block_len=20,
        min_block_len=10,
        rng=np.random.default_rng(1),
        return_session=True,
    )
    assert metrics == tuple(expected)
    np.testing.assert_array_equal(session["vals3"], expected_session["vals3"])


def test_simulate_example_session_coerces_numeric_parameters():
    kwargs = {**EXAMPLE, "sig": np.float64(5.0), "max_block_len": np.int64(20)}
    assert simulate_example_session(**kwargs)[0] == simulate_example_session(**EXAMPLE)[0]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

METRICS = (1.234, 0.5, 12.0, 3.456)


def test_format_session_title_reports_parameters_and_metrics():
    t = format_session_title(METRICS, sig=10.0, max_block_len=40)
    assert t == (
        "Example GP session | sig=10.00, max_block_len=40 "
        "| gap_mean=1.23, frac≥2=0.50, run_mean=12.00, run_std=3.46"
    )


def test_format_session_title_prefers_a_given_title():
    assert format_session_title(METRICS, sig="not a number", max_block_len=40, title="Mine") == (
        "Mine"
    )


def test_format_session_title_empty_title_falls_back_to_the_generated_one():
    t = format_session_title(METRICS, sig=10.0, max_block_len=40, title="")
    assert t.startswith("Example GP session")


def test_y_axis_max_is_the_clip_maximum_or_the_data_maximum():
    vals3 = np.array([[1.0, 7.5, 0.0], [2.0, np.nan, 0.0]])
    assert y_axis_max(vals3, clip=(1, 5), use_raw=False) == 5
    assert y_axis_max(vals3, clip=(1, 5), use_raw=True) == 7.5


def test_plot_session_panel_draws_into_given_axes_without_new_figure():
    fig, ax = plt.subplots()
    vals3 = np.tile(np.array([1.0, 2.0, 3.0]), (10, 1))
    out = plot_session_panel(vals3, np.array([4, 8]), y_max=5, ax=ax)
    assert out is ax
    assert len(plt.get_fignums()) == 1
    assert [ln.get_label() for ln in ax.lines[:3]] == ["Arm 0", "Arm 1", "Arm 2"]
    # one legend entry for all block lines, drawn at b - 0.5
    block_lines = [ln for ln in ax.lines if ln.get_label() == "Block"]
    assert len(block_lines) == 1
    assert len(ax.lines) == 5
    assert ax.get_xlim() == (0, 9)
    assert ax.get_ylim() == (-0.5, 5.5)
    assert ax.get_ylabel() == "Reward Magnitude"
    assert ax.get_legend() is not None
    plt.close(fig)


def test_plot_session_panel_creates_a_figure_only_without_axes():
    plt.close("all")
    ax = plot_session_panel(np.ones((4, 3)), np.array([], dtype=int), y_max=5)
    assert len(plt.get_fignums()) == 1
    assert ax.figure.get_size_inches().tolist() == [12, 6]
    plt.close("all")


def test_plot_session_panel_can_leave_the_legend_to_the_caller():
    fig, ax = plt.subplots()
    plot_session_panel(np.ones((4, 3)), np.array([], dtype=int), y_max=5, ax=ax, legend=False)
    assert ax.get_legend() is None
    plt.close(fig)


def test_plot_session_panel_marks_choices_on_chosen_arm_value():
    fig, ax = plt.subplots()
    vals3 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 1.0]])
    plot_session_panel(vals3, np.array([], dtype=int), y_max=5, choices=[2, 1], ax=ax)
    np.testing.assert_array_equal(ax.collections[0].get_offsets(), [[0, 3.0], [1, 5.0]])
    plt.close(fig)


def test_plot_example_session_gp_returns_session_and_shows_one_figure(monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.get_fignums()))
    session = plot_example_session_gp(sig=5.0, max_block_len=20, T=60, seed=1)
    assert session["vals3"].shape == (60, 3)
    assert len(shown) == 1 and len(shown[0]) == 1
    plt.close("all")


def test_plot_example_session_gp_custom_title_wins(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    plot_example_session_gp(sig=5.0, max_block_len=20, T=60, title="Mine")
    assert plt.gcf().axes[0].get_title() == "Mine"
    plt.close("all")
