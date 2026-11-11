"""Null agents fitted to a cohort and simulated on each mouse's own task."""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from src.agents import drifting_simulation as sim
from src.agents import null_agents as na
from src.behavior_analysis.drifting_task_rules import block_changes, complete_block_lengths
from src.behavior_analysis.drifting_trial_table import drifting_trial_table
from src.pipelines import choice_logit_values as logit_pipe
from src.pipelines import null_agents as pipe

RULES = sim.TaskRules(drift_rate=0.1, block_lengths=np.arange(30, 36))
RECENCY = na.WslsRecencyParams(a_win=1.0, a_lose=-0.6, w=1.0)


@pytest.fixture(scope="module")
def real_like():
    """Four Q-learning mice standing in for real ones: six problems, uneven session lengths."""
    cohort = logit_pipe.simulate_cohort(
        [(0.3, 1.2, 0.0, 0.0), (0.3, 1.2, 0.8, 0.6)],
        mice_per_set=2,
        n_problems=3,
        sessions_per_problem=3,
        trials_per_session=300,
        seed=1,
    )
    return cohort.trials


@pytest.fixture(scope="module")
def fits(real_like):
    return {kind: na.fit_null_agents(real_like, kind) for kind in na.KINDS}


@pytest.fixture(scope="module")
def cohort(real_like, fits):
    return pipe.simulate_null_cohort(real_like, fits["wsls_recency"], "wsls_recency", 3, RULES)


def test_each_mouse_is_simulated_on_its_own_layouts(real_like, cohort):
    assert sorted(cohort.trials["mouse"].unique()) == sorted(real_like["mouse"].unique())
    for mouse, rows in real_like.groupby("mouse"):
        mine = cohort.trials[cohort.trials["mouse"] == mouse]
        assert len(mine) == len(rows)
        real_layouts = sim.mouse_layouts(rows)
        for number, layout in enumerate(real_layouts.values(), start=1):
            simulated = mine[mine["problem"] == number]
            assert simulated["magnitudes"].iloc[0] == layout.start_magnitudes
            assert simulated.groupby("session_index").size().tolist() == layout.session_lengths
            assert set(simulated["choice"]) <= set(layout.ports)
            assert simulated["initiation"].iloc[0] == layout.initiation


def test_cohort_has_the_synthetic_cohort_interface(cohort, fits):
    assert isinstance(cohort, logit_pipe.SyntheticCohort)
    assert list(cohort.parameters.columns) == ["mouse", "a_win", "a_lose", "w", "decay"]
    mouse = cohort.parameters["mouse"].iloc[0]
    assert cohort.parameters.set_index("mouse").loc[mouse, "w"] == fits["wsls_recency"][mouse].w
    assert sorted(next(iter(cohort.problems.values()))) == sorted(cohort.trials["mouse"].unique())


def test_simulation_is_reproducible_and_independent_of_the_other_mice(real_like, fits, cohort):
    again = pipe.simulate_null_cohort(real_like, fits["wsls_recency"], "wsls_recency", 3, RULES)
    pd.testing.assert_frame_equal(cohort.trials, again.trials)
    one = sorted(fits["wsls_recency"])[1]
    alone = pipe.simulate_null_cohort(
        real_like, {one: fits["wsls_recency"][one]}, "wsls_recency", 3, RULES
    )
    kept = cohort.trials[cohort.trials["mouse"] == one].reset_index(drop=True)
    pd.testing.assert_frame_equal(alone.trials.reset_index(drop=True), kept)


def test_another_agent_or_seed_gives_another_stream(real_like, fits, cohort):
    other_seed = pipe.simulate_null_cohort(
        real_like, fits["wsls_recency"], "wsls_recency", 4, RULES
    )
    assert not other_seed.trials["choice"].equals(cohort.trials["choice"])


@pytest.mark.filterwarnings("ignore:the problems of:UserWarning")  # three disjoint tower triples
def test_simulated_cohort_goes_through_the_logit_pipeline(cohort):
    analysis = logit_pipe.analyse_trial_table(cohort.trials, n_lags=3, tail_decay=0.9)
    assert sorted(analysis.designs) == sorted(cohort.trials["mouse"].unique())


def test_fits_to_the_simulated_cohort_recover_the_agents(real_like, fits, cohort):
    refit = na.fit_wsls_recency(cohort.trials)
    for mouse, truth in fits["wsls_recency"].items():
        assert refit[mouse].a_win == pytest.approx(truth.a_win, abs=0.5)
        assert refit[mouse].a_lose == pytest.approx(truth.a_lose, abs=0.5)


def test_rules_are_measured_from_the_trial_table_when_not_given():
    layout = sim.MouseLayout(["A1", "B3", "C2"], "B2", {"A1": 4, "B3": 2, "C2": 0}, [250] * 8)
    sessions = sim.play_problem(na.NullPolicy(RECENCY), layout, RULES, np.random.default_rng(0))
    table = drifting_trial_table({1: {"m": sim.simulated_sessions("m", 1, layout, sessions)}})
    cohort = pipe.simulate_null_cohort(table, {"m": RECENCY}, "wsls_recency", seed=0)
    assert len(cohort.trials) == len(table)


def test_parameters_must_match_the_agent_and_the_mice(real_like, fits):
    with pytest.raises(ValueError, match="wsls_magnitude"):
        pipe.simulate_null_cohort(real_like, fits["wsls"], "wsls_magnitude", 0, RULES)
    with pytest.raises(ValueError, match="unknown agent"):
        pipe.simulate_null_cohort(real_like, fits["wsls"], "other", 0, RULES)
    with pytest.raises(ValueError, match="not in the trial table"):
        pipe.simulate_null_cohort(real_like, {"ghost": fits["wsls"]["sim01"]}, "wsls", 0, RULES)


def test_parameters_table_has_one_row_per_mouse(fits):
    table = pipe.parameters_table(fits["wsls_magnitude"])
    assert list(table.columns) == ["mouse", "a_win", "a_lose", "b", "w", "decay"]
    assert table["mouse"].tolist() == sorted(fits["wsls_magnitude"])


def test_scores_table_is_long_with_nats_per_choice(real_like, fits):
    table = pipe.scores_table(real_like, fits)
    assert set(table["model"]) == set(na.KINDS)
    assert len(table) == len(na.KINDS) * real_like["mouse"].nunique()
    row = table[(table["model"] == "wsls") & (table["mouse"] == "sim01")].iloc[0]
    mine = real_like[real_like["mouse"] == "sim01"]
    assert row["n_choices"] == len(mine)
    assert row["nats_per_choice"] == pytest.approx(na.nats_per_choice(mine, fits["wsls"]["sim01"]))
    assert row["log_likelihood"] == pytest.approx(-row["nats_per_choice"] * len(mine))


def test_nested_agents_never_fit_worse_than_plain_wsls(real_like, fits):
    table = pipe.scores_table(real_like, fits).pivot(
        index="mouse", columns="model", values="nats_per_choice"
    )
    # same stay rule, plus a fitted switch weight / slope: maximum likelihood cannot lose
    assert (table["wsls_recency"] <= table["wsls"] + 1e-9).all()
    assert (table["wsls_magnitude"] <= table["wsls_recency"] + 1e-9).all()


def test_observed_stay_by_previous_magnitude(real_like):
    table = pipe.stay_by_magnitude(real_like[real_like["mouse"] == "sim01"])
    assert list(table.columns) == ["mouse", "magnitude", "n_pairs", "p_stay"]
    assert sorted(table["magnitude"]) == sorted(set(table["magnitude"]))
    assert table["n_pairs"].sum() == (
        len(real_like[real_like["mouse"] == "sim01"])
        - real_like[real_like["mouse"] == "sim01"]["trial_in_session"].eq(0).sum()
    )
    assert table["p_stay"].between(0, 1).all()


def test_implied_stay_curves_follow_the_fitted_stay_logit(fits):
    table = pipe.implied_stay_curves(fits)
    row = table[(table["model"] == "wsls_magnitude") & (table["mouse"] == "sim01")]
    params = fits["wsls_magnitude"]["sim01"]
    assert row["magnitude"].tolist() == [0, 1, 2, 3, 4]
    assert row["p_stay"].to_numpy() == pytest.approx(expit(params.stay_logit(np.arange(5))))


@pytest.mark.filterwarnings("ignore:the problems of:UserWarning")  # three disjoint tower triples
def test_null_controls_fit_simulate_and_analyse_every_agent(real_like):
    controls = pipe.null_controls(real_like, seed=1, n_lags=3, tail_decay=0.9, l2=1.0, rules=RULES)
    mice = sorted(real_like["mouse"].unique())
    assert list(controls.fits) == list(na.KINDS)
    assert list(controls.cohorts) == list(na.KINDS)
    assert list(controls.analyses) == list(na.KINDS)
    for kind in na.KINDS:
        assert sorted(controls.fits[kind]) == mice
        assert sorted(controls.cohorts[kind].trials["mouse"].unique()) == mice
        assert sorted(controls.analyses[kind].designs) == mice
        assert controls.analyses[kind].tail_decay == 0.9
    # the agents differ in what they do, so the streams differ
    assert (
        not controls.cohorts["wsls"]
        .trials["choice"]
        .equals(controls.cohorts["wsls_magnitude"].trials["choice"])
    )


def test_cohorts_carry_the_kind_of_agent_that_made_them(cohort):
    assert cohort.kind == "wsls_recency"
    value_cohort = logit_pipe.simulate_cohort(
        [(0.3, 1.2, 0.0, 0.0)], mice_per_set=1, n_problems=2, sessions_per_problem=1,
        trials_per_session=60, seed=0,
    )  # fmt: skip
    assert value_cohort.kind == "value"


@pytest.mark.filterwarnings("ignore:the problems of:UserWarning")  # three disjoint tower triples
def test_value_learner_summaries_refuse_a_null_cohort(cohort):
    analysis = logit_pipe.analyse_trial_table(cohort.trials, n_lags=3, tail_decay=0.9)
    for summary in (
        logit_pipe.generating_values,
        logit_pipe.reference_weights,
        logit_pipe.value_agreement,
        logit_pipe.kernel_comparison,
    ):
        with pytest.raises(ValueError, match="wsls_recency.*value learner"):
            summary(analysis, cohort)


def _two_cohort_table():
    """Mouse ``a`` in cohort 1 (blocks of 20-25 trials), mouse ``b`` in cohort 2 (40-45)."""
    layout = sim.MouseLayout(["A1", "B3", "C2"], "B2", {"A1": 4, "B3": 2, "C2": 0}, [250] * 8)
    problems = {}
    for cohort, mouse, lengths in ((1, "a", [20, 25]), (2, "b", [40, 45])):
        rules = sim.TaskRules(0.1, np.array(lengths))
        sessions = sim.play_problem(
            na.NullPolicy(RECENCY), layout, rules, np.random.default_rng(cohort)
        )
        problems[(cohort, 1)] = {mouse: sim.simulated_sessions(mouse, 1, layout, sessions)}
    return drifting_trial_table(problems)


@pytest.fixture(scope="module")
def two_cohorts():
    return _two_cohort_table()


def test_task_rules_are_measured_per_cohort(two_cohorts):
    rules = pipe.task_rules_by_cohort(two_cohorts)
    assert sorted(rules) == [1, 2]
    assert set(rules[1].block_lengths) <= {20, 25}
    assert set(rules[2].block_lengths) <= {40, 45}


def test_each_mouse_is_simulated_with_its_own_cohorts_block_lengths(two_cohorts):
    fits = na.fit_wsls_recency(two_cohorts, min_switches=10)
    cohort = pipe.simulate_null_cohort(two_cohorts, fits, "wsls_recency", seed=0)
    for mouse, allowed in (("a", {20, 25}), ("b", {40, 45})):
        mine = cohort.trials[cohort.trials["mouse"] == mouse]
        assert set(complete_block_lengths(block_changes(mine))) <= allowed


def test_a_cohort_whose_rules_cannot_be_estimated_fails_loudly(two_cohorts):
    short = two_cohorts[two_cohorts["mouse"] == "a"].iloc[:15].copy()
    short["mouse"] = "c"
    short["problem"] = [(3, 1)] * len(short)
    table = pd.concat([two_cohorts, short], ignore_index=True)
    with pytest.raises(ValueError, match="cohort 3"):
        pipe.task_rules_by_cohort(table)


def test_a_mouse_in_two_cohorts_is_refused(two_cohorts):
    both = two_cohorts.copy()
    both["mouse"] = "a"
    with pytest.raises(ValueError, match="more than one cohort"):
        pipe.task_rules_by_cohort(both)


def test_explicit_rules_may_be_one_for_all_or_one_per_cohort(two_cohorts):
    fits = na.fit_wsls(two_cohorts)
    one = pipe.simulate_null_cohort(two_cohorts, fits, "wsls", 0, RULES)
    many = pipe.simulate_null_cohort(two_cohorts, fits, "wsls", 0, {1: RULES, 2: RULES})
    pd.testing.assert_frame_equal(one.trials, many.trials)
    with pytest.raises(ValueError, match="cohort 2"):
        pipe.simulate_null_cohort(two_cohorts, fits, "wsls", 0, {1: RULES})


# ---------------------------------------------------------------------------
# Switch targets and the choice baseline
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def layouts_table():
    """Four mice on six problems of three sessions of 300 trials, as targets to play."""
    cohort = logit_pipe.simulate_cohort(
        [(0.3, 1.2, 0.0, 0.0)],
        mice_per_set=4,
        n_problems=6,
        sessions_per_problem=3,
        trials_per_session=300,
        seed=2,
    )
    return cohort.trials


def _switch_target_excess(layouts_table, decay):
    params = {
        m: na.WslsRecencyParams(a_win=1.0, a_lose=-0.5, w=1.5, decay=decay)
        for m in layouts_table["mouse"].unique()
    }
    cohort = pipe.simulate_null_cohort(layouts_table, params, "wsls_recency", 5, RULES)
    analysis = logit_pipe.analyse_trial_table(cohort.trials, n_lags=5, tail_decay=0.9)
    return pipe.switch_target_excess(analysis, n_bins=4)


def test_recency_beyond_the_baseline_choice_lags_gives_a_switch_target_excess(layouts_table):
    # A trace that decays slowly (0.9) mostly lies beyond the logit's five choice lags, so the
    # choice-history baseline cannot absorb where recency sends the switches; a fast trace
    # (0.5) fits in five lags and the switch targets sit on the baseline.
    slow = _switch_target_excess(layouts_table, decay=0.9)
    fast = _switch_target_excess(layouts_table, decay=0.5)
    assert slow["mean_excess"].min() > 0.05
    assert fast["mean_excess"].abs().max() < 0.04
    assert list(slow["row"]) == ["rewarded", "unrewarded"]
    assert {"observed_top", "baseline_top", "mean_excess"} <= set(slow.columns)


# ---------------------------------------------------------------------------
# Comparison with a reference, and its figures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def comparison(real_like):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "the problems of")  # three disjoint tower triples
        reference = logit_pipe.analyse_trial_table(real_like, n_lags=3, tail_decay=0.9)
        return pipe.compare_with_controls(
            real_like, reference, "Q-Learner", seed=1, n_bins=4, min_mice=1, rules=RULES
        )


def test_comparison_tabulates_the_reference_and_every_agent(comparison, real_like):
    labels = ["Q-Learner", "WSLS", "WSLS (Recency)", "WSLS (Magnitude)", "WSLS (Graded)"]
    assert list(comparison.analyses) == labels
    assert list(comparison.psychometrics) == labels
    assert list(comparison.stay_switch) == labels
    assert list(comparison.switch_targets) == labels
    assert list(comparison.effects["source"]) == labels
    assert set(comparison.switch_excess["source"]) == set(labels)
    mice = real_like["mouse"].nunique()
    # four held-out agents plus the logit's held-out score, one row per mouse
    assert set(comparison.scores["model"]) == {*na.KINDS, "choice_logit"}
    assert len(comparison.scores) == 5 * mice
    assert comparison.analyses["WSLS"].tail_decay == 0.9  # the reference's logit settings
    assert comparison.analyses["WSLS"].n_lags == 3
    assert comparison.observed_stay["mouse"].nunique() == mice


@pytest.fixture(scope="module")
def comparison_with_fitted_q(real_like):
    """The same comparison with a fitted Q-learner (a positive control) as an extra source."""
    from src.pipelines import fitted_q

    fitted = {m: (0.3, 1.2, 0.5, 0.3) for m in sorted(real_like["mouse"].unique())}
    cohort = fitted_q.simulate_fitted_q_cohort(real_like, fitted, seed=2, rules=RULES)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "the problems of")
        reference = logit_pipe.analyse_trial_table(real_like, n_lags=3, tail_decay=0.9)
        return pipe.compare_with_controls(
            real_like, reference, "Mice", seed=1, n_bins=4, min_mice=1, rules=RULES,
            extra_cohorts={fitted_q.LABEL: cohort},
        )  # fmt: skip


def test_an_extra_cohort_is_analyzed_like_the_agents_and_listed_after_them(
    comparison_with_fitted_q,
):
    labels = [
        "Mice",
        "WSLS",
        "WSLS (Recency)",
        "WSLS (Magnitude)",
        "WSLS (Graded)",
        "Q-Learner (Fitted)",
    ]
    result = comparison_with_fitted_q
    assert list(result.analyses) == labels
    assert list(result.psychometrics) == labels and list(result.effects["source"]) == labels
    assert set(result.stay_switch) == set(result.switch_targets) == set(labels)
    assert set(result.switch_excess["source"]) == set(labels)
    fitted = result.analyses["Q-Learner (Fitted)"]
    assert fitted.n_lags == 3 and fitted.tail_decay == 0.9  # the reference's logit settings
    assert sorted(fitted.designs) == sorted(result.analyses["Mice"].designs)  # the mice's names
    assert set(result.scores["model"]) == {*na.KINDS, "choice_logit"}


def test_an_extra_label_may_not_repeat_a_source(real_like, comparison_with_fitted_q):
    cohort = comparison_with_fitted_q.extra_cohorts["Q-Learner (Fitted)"]
    reference = comparison_with_fitted_q.analyses["Mice"]
    with pytest.raises(ValueError, match="already sources"):
        pipe.compare_with_controls(
            real_like, reference, "Mice", seed=1, rules=RULES, extra_cohorts={"WSLS": cohort}
        )


def test_the_fitted_q_learner_appears_in_the_figures_with_its_label_and_a_mouse_color(
    comparison_with_fitted_q, tmp_path, monkeypatch
):
    from src.behavior_visualization import plot_helpers
    from src.behavior_visualization import plot_null_agents as plots

    legends = {}
    real = plot_helpers._write_figure_files

    def spy(fig, base, formats, dpi):
        result = real(fig, base, formats, dpi)
        found = [*fig.legends, *(ax.get_legend() for ax in fig.axes if ax.get_legend())]
        texts = [t.get_text() for legend in found for t in legend.get_texts()]
        legends[str(base).rsplit("/", 1)[-1]] = texts
        return result

    monkeypatch.setattr(plot_helpers, "_write_figure_files", spy)
    plots.plot_comparison(comparison_with_fitted_q, tmp_path)
    assert any("Q-Learner (Fitted)" in t for t in legends["Kernels by Source"])
    for kind in ("Stay-Switch Curves", "Switch Target Curves"):
        for extension in ("pdf", "png", "svg", "caption.txt"):
            assert (tmp_path / f"Q-Learner (Fitted) {kind}.{extension}").exists()
    caption = (tmp_path / "Psychometric Curves.caption.txt").read_text()
    assert "positive control" in caption and "Q-Learner (Fitted)" in caption
    # its simulated mice carry the real mice's names, so the mouse-colored figures use their colors
    names = set(comparison_with_fitted_q.extra_cohorts["Q-Learner (Fitted)"].trials["mouse"])
    assert names == set(comparison_with_fitted_q.analyses["Mice"].designs)


def test_comparison_plots_write_every_figure_with_a_caption(comparison, tmp_path):
    from src.behavior_visualization import plot_null_agents as plots

    plots.plot_comparison(comparison, tmp_path)
    names = {p.name for p in tmp_path.iterdir()}
    expected = ["Null Agent Fits", "Kernels by Source", "Psychometric Curves"] + [
        f"{label} {kind}"
        for label in ("WSLS", "WSLS (Recency)", "WSLS (Magnitude)", "WSLS (Graded)")
        for kind in ("Stay-Switch Curves", "Switch Target Curves")
    ]
    for name in expected:
        for extension in ("pdf", "png", "svg", "caption.txt"):
            assert f"{name}.{extension}" in names, (name, extension)
    assert "leave-one-problem-out" in (tmp_path / "Null Agent Fits.caption.txt").read_text().lower()


def test_magnitude_agents_excess_comes_from_the_recency_trace_not_from_graded_staying(
    layouts_table,
):
    def rewarded_excess(w, decay):
        params = {
            m: na.WslsMagnitudeParams(0.2, -0.5, 0.5, w, decay)
            for m in layouts_table["mouse"].unique()
        }
        cohort = pipe.simulate_null_cohort(layouts_table, params, "wsls_magnitude", 5, RULES)
        analysis = logit_pipe.analyse_trial_table(cohort.trials, n_lags=5, tail_decay=0.9)
        table = pipe.switch_target_excess(analysis, n_bins=4).set_index("row")
        return table.loc["rewarded", "mean_excess"]

    assert rewarded_excess(w=1.5, decay=0.9) > 0.1  # slow trace beyond the five baseline lags
    assert abs(rewarded_excess(w=1.5, decay=0.5)) < 0.05  # fast trace: on the baseline
    assert abs(rewarded_excess(w=0.0, decay=0.9)) < 0.05  # graded staying alone: on the baseline
