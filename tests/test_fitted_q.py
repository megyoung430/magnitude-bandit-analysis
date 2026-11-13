"""The fitted Q-learner positive control: fit, simulate on a mouse's task, and its power check."""

import pandas as pd
import pytest

from src.agents import drifting_simulation as sim
from src.agents import kernel_q_agent as kq
from src.behavior_analysis.strategy_models import strategy_models as sm
from src.behavior_analysis.switching import switch_regression as sr
from src.pipelines.strategy_models import fitted_q as fq
from src.pipelines.switching import switch_regression as switch_pipe
from tests.strategy_fixtures import Q_LEARNER, RULES, simulate_mouse

SIZE = dict(n_problems=4, sessions=2, trials=250)


@pytest.fixture(scope="module")
def mice():
    return pd.concat(
        [simulate_mouse(sm.Q_LEARNER, seed=3 + i, mouse=f"m{i}", **SIZE) for i in (1, 2)],
        ignore_index=True,
    )


@pytest.fixture(scope="module")
def fits(mice):
    return fq.fit_q_learners(mice, n_starts=3, seed=0)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def test_known_parameters_are_recovered_from_simulated_choices(fits):
    # generating (learning rate, beta, kappa, plus bias) = (0.25, 2.5, 0.8, 0.3), 2000 trials
    for mouse in ("m1", "m2"):
        learning_rate, beta, kappa, plus_bias = fits[mouse]
        assert learning_rate == pytest.approx(Q_LEARNER[0], abs=0.15)
        assert beta == pytest.approx(Q_LEARNER[1], abs=1.0)
        assert kappa == pytest.approx(Q_LEARNER[2], abs=0.6)
        assert plus_bias == pytest.approx(Q_LEARNER[3], abs=0.5)


def test_the_parameters_table_flags_parameters_at_a_bound(fits):
    table = fq.parameters_table(fits)
    assert list(table["mouse"]) == ["m1", "m2"]
    assert {"learning_rate", "beta", "kappa", "plus_bias", "at_bound"} <= set(table.columns)
    edge = {"m1": (1.0, 2.0, 0.0, 0.0), "m2": (0.3, 2.0, 0.0, 0.0)}
    flagged = fq.parameters_table(edge).set_index("mouse")["at_bound"]
    assert flagged["m1"] and not flagged["m2"]


def test_the_fits_of_a_comparison_are_reused(fits):
    source = type(
        "S", (), {"fits": {"m1": {sm.Q_LEARNER: fits["m1"]}, "m2": {sm.Q_LEARNER: (0.2,) * 4}}}
    )
    assert fq.fits_of(source) == {"m1": fits["m1"], "m2": (0.2,) * 4}


def test_a_mouse_that_cannot_be_fitted_raises(mice):
    with pytest.raises(ValueError, match="no trials"):
        fq.fit_q_learners(mice.iloc[0:0], n_starts=1, seed=0)


# ---------------------------------------------------------------------------
# Simulation on the mice's own tasks
# ---------------------------------------------------------------------------


def test_the_cohort_plays_each_mouses_layouts_and_keeps_its_name(mice, fits):
    cohort = fq.simulate_fitted_q_cohort(mice, fits, seed=1, rules=RULES)
    assert cohort.kind == fq.KIND == "q_fitted"
    assert set(cohort.trials["mouse"]) == {"m1", "m2"}
    for mouse in ("m1", "m2"):
        real = sim.mouse_layouts(mice[mice["mouse"] == mouse])
        played = sim.mouse_layouts(cohort.trials[cohort.trials["mouse"] == mouse])
        assert list(real) == list(played)
        for problem, layout in real.items():
            assert played[problem].ports == layout.ports
            assert played[problem].session_lengths == layout.session_lengths
    assert set(cohort.parameters["mouse"]) == {"m1", "m2"}
    assert set(cohort.problems) == set(mice["problem"])


def test_the_task_rules_are_those_of_the_cohorts(mice, fits):
    # one rule set for every mouse, or measured per cohort; a cohort without rules is an error
    table_rules = switch_pipe.rules_by_mouse(mice, ["m1", "m2"], RULES)
    assert set(table_rules) == {"m1", "m2"}
    with pytest.raises(ValueError, match="no task rules"):
        fq.simulate_fitted_q_cohort(mice, fits, seed=1, rules={"cohort 9": RULES})


def test_the_simulation_is_seeded_and_independent_of_other_mice(mice, fits):
    first = fq.simulate_fitted_q_cohort(mice, fits, seed=1, rules=RULES).trials
    again = fq.simulate_fitted_q_cohort(mice, fits, seed=1, rules=RULES).trials
    other = fq.simulate_fitted_q_cohort(mice, fits, seed=2, rules=RULES).trials
    alone = fq.simulate_fitted_q_cohort(mice, {"m1": fits["m1"]}, seed=1, rules=RULES).trials
    pd.testing.assert_frame_equal(first, again)
    assert not first["choice"].equals(other["choice"])
    one = first[first["mouse"] == "m1"].reset_index(drop=True)
    pd.testing.assert_frame_equal(one, alone.reset_index(drop=True))


def test_the_simulation_checks_its_inputs(mice, fits):
    with pytest.raises(ValueError, match="no mice"):
        fq.simulate_fitted_q_cohort(mice, {}, seed=0, rules=RULES)
    with pytest.raises(ValueError, match="not in the trial table"):
        fq.simulate_fitted_q_cohort(mice, {"ghost": fits["m1"]}, seed=0, rules=RULES)


def test_the_fitted_learner_chooses_like_its_mouse(mice, fits):
    # the fitted Q-learner repeats and learns as its (Q-learner) mouse does: stay rates agree
    cohort = fq.simulate_fitted_q_cohort(mice, fits, seed=5, rules=RULES).trials

    def stay(trials):
        same = trials.groupby(["mouse", "problem"])["choice"].transform(lambda c: c == c.shift())
        return float(same.mean())

    assert stay(cohort) == pytest.approx(stay(mice), abs=0.08)


# ---------------------------------------------------------------------------
# Power check
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def switch_source(mice):
    return switch_pipe.analyze_switch_source(mice, trace_grid=(0.9,), min_switches=30)


def test_the_power_check_reports_the_share_of_replicates_above_the_mouse(mice, fits, switch_source):
    table = fq.power_check(mice, switch_source, fits, replicates=3, rules=RULES, seed=0)
    assert {"mouse", "condition", "value", "observed", "n_replicates", "fraction_above"} <= set(
        table.columns
    )
    assert set(table["value"]) == {sr.TRUE, sr.BELIEVED}
    assert set(table["mouse"]) == {"m1", "m2"}
    assert table["fraction_above"].between(0, 1).all() and (table["n_replicates"] <= 3).all()


def test_the_power_check_is_cached_and_reproducible(mice, fits, switch_source, tmp_path):
    kwargs = dict(replicates=2, rules=RULES, seed=0, cache_dir=tmp_path)
    first = fq.power_check(mice, switch_source, fits, **kwargs)
    assert any(tmp_path.rglob("*.pkl"))
    again = fq.power_check(mice, switch_source, fits, n_jobs=2, **kwargs)
    pd.testing.assert_frame_equal(first, again)


def test_the_power_check_leaves_out_mice_without_a_fit_and_warns(mice, fits, switch_source):
    with pytest.warns(UserWarning, match="mouse m2 left out"):
        table = fq.power_check(mice, switch_source, {"m1": fits["m1"]}, replicates=2, rules=RULES)
    assert set(table["mouse"]) == {"m1"}
    with pytest.raises(ValueError, match="no fitted Q-learner for any"):
        fq.power_check(mice, switch_source, {}, replicates=2, rules=RULES)
    with pytest.raises(ValueError, match="at least 1"):
        fq.power_check(mice, switch_source, fits, replicates=0, rules=RULES)


def test_the_code_version_follows_the_module_sources(monkeypatch):
    import inspect

    before = fq.code_version()
    real = inspect.getsource
    monkeypatch.setattr(inspect, "getsource", lambda module: real(module) + "# edit\n")
    assert fq.code_version() != before


# ---------------------------------------------------------------------------
# The Q-learner with kernel and forgetting as the fitted positive control
# ---------------------------------------------------------------------------

KERNEL_SIZE = dict(n_problems=3, sessions=2, trials=200)


@pytest.fixture(scope="module")
def kernel_mice():
    return pd.concat(
        [simulate_mouse(sm.KERNEL_Q, seed=7 + i, mouse=f"k{i}", **KERNEL_SIZE) for i in (1, 2)],
        ignore_index=True,
    )


@pytest.fixture(scope="module")
def kernel_fits(kernel_mice):
    return fq.fit_q_learners(kernel_mice, n_starts=2, seed=0, model=sm.KERNEL_Q)


def test_the_two_controls_have_their_own_kind_label_and_stream():
    kernel = fq.CONTROLS[sm.KERNEL_Q]
    assert fq.CONTROLS[sm.Q_LEARNER].label == fq.LABEL == "Q-Learner (Fitted)"
    assert kernel.label == "Q-Learner (Kernel + Forgetting, Fitted)"
    assert kernel.kind == "kernel_q_fitted"
    assert len({c.stream for c in fq.CONTROLS.values()}) == len(fq.CONTROLS)


def test_kernel_fits_have_seven_parameters_each(kernel_fits):
    assert set(kernel_fits) == {"k1", "k2"}
    assert all(len(p) == 7 for p in kernel_fits.values())
    table = fq.parameters_table(kernel_fits, model=sm.KERNEL_Q)
    assert {"alpha_pos", "alpha_neg", "zeta", "alpha_k", "beta", "beta_k", "plus_bias"} <= set(
        table.columns
    )
    assert table["at_bound"].dtype == bool


def test_the_kernel_fits_of_a_comparison_are_reused(kernel_fits):
    source = type("S", (), {"fits": {m: {sm.KERNEL_Q: p} for m, p in kernel_fits.items()}})
    assert fq.fits_of(source, model=sm.KERNEL_Q) == kernel_fits


def test_the_kernel_cohort_plays_each_mouses_layouts_and_keeps_its_name(kernel_mice, kernel_fits):
    cohort = fq.simulate_fitted_q_cohort(
        kernel_mice, kernel_fits, seed=1, rules=RULES, model=sm.KERNEL_Q
    )
    assert cohort.kind == "kernel_q_fitted"
    assert set(cohort.trials["mouse"]) == {"k1", "k2"}
    assert list(cohort.parameters.columns) == ["mouse", *kq.PARAMETER_NAMES]
    for mouse in ("k1", "k2"):
        real = sim.mouse_layouts(kernel_mice[kernel_mice["mouse"] == mouse])
        played = sim.mouse_layouts(cohort.trials[cohort.trials["mouse"] == mouse])
        assert list(real) == list(played)


def test_the_two_controls_do_not_share_a_random_stream(mice, fits, kernel_fits):
    both = {m: kernel_fits["k1"] for m in ("m1", "m2")}
    q = fq.simulate_fitted_q_cohort(mice, fits, seed=1, rules=RULES).trials
    kernel = fq.simulate_fitted_q_cohort(mice, both, seed=1, rules=RULES, model=sm.KERNEL_Q).trials
    assert not q["choice"].reset_index(drop=True).equals(kernel["choice"].reset_index(drop=True))


def test_a_model_without_a_control_is_refused(mice, fits):
    with pytest.raises(ValueError, match="no fitted control"):
        fq.simulate_fitted_q_cohort(mice, fits, seed=0, rules=RULES, model=sm.WSLS_MAGNITUDE)


def test_the_kernel_power_check_replays_the_kernel_learner(kernel_mice, kernel_fits):
    source = switch_pipe.analyze_switch_source(kernel_mice, trace_grid=(0.9,), min_switches=30)
    table = fq.power_check(
        kernel_mice, source, kernel_fits, replicates=2, rules=RULES, seed=0, model=sm.KERNEL_Q
    )
    assert not table.empty and set(table["mouse"]) <= {"k1", "k2"}  # a mouse may lack switches
    assert table["fraction_above"].between(0, 1).all()
