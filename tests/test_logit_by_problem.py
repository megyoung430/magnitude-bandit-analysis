"""The choice logit fitted per problem through structured and drifting training."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis.choice_logit import choice_logit as cl
from src.behavior_analysis.choice_logit import logit_by_problem as lbp
from src.behavior_analysis.data import training_stages as stages
from src.behavior_visualization.choice_logit import plot_logit_by_problem as plots
from src.pipelines.choice_logit import logit_by_problem as pipe

matplotlib.use("Agg")

PORTS = ("A1", "B1", "C1")
BLOCKED_NAME = "Meg\\3x3_field_blocked_value_bandit"
DRIFTING_NAME = "Meg\\3x3_field_drifting_value_bandit"


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def problem_rows(mouse, cohort, task, k, overall, n_trials, rng, stickiness=0.0):
    """Trials of a learner that prefers the port that paid most last time (and repeats choices)."""
    values = dict.fromkeys(PORTS, 0.0)
    magnitudes = dict(zip(PORTS, rng.permutation([0, 1, 4]).tolist(), strict=True))
    rows, last = [], None
    for t in range(n_trials):
        if t and t % 60 == 0:  # a block change: the magnitudes move between ports
            magnitudes = dict(zip(PORTS, rng.permutation([0, 1, 4]).tolist(), strict=True))
        utility = np.array([2.0 * values[p] + stickiness * (p == last) for p in PORTS])
        p = np.exp(utility - utility.max())
        choice = PORTS[rng.choice(3, p=p / p.sum())]
        reward = magnitudes[choice]
        values[choice] = reward / 4
        last = choice
        rows.append(dict(mouse=mouse, problem=(cohort, task, k), session=f"s{k}",
                         session_index=0, trial=t, trial_in_session=t, choice=choice,
                         magnitudes=dict(magnitudes), reward=reward, initiation="B2",
                         block=None, after_gap=False, cohort=cohort, task=task, k=k,
                         overall_problem=overall))  # fmt: skip
    return rows


def mouse_trials(mouse, cohort, sequence, n_trials=200, seed=0):
    """*sequence*: ``[(task, k), ...]`` in training order."""
    rng = np.random.default_rng(seed)
    rows = []
    for overall, (task, k) in enumerate(sequence, start=1):
        rows += problem_rows(mouse, cohort, task, k, overall, n_trials, rng)
    return pd.DataFrame(rows)


SEQUENCE = [("blocked", 1), ("blocked", 2), ("drifting", 1), ("drifting", 2), ("drifting", 3)]


# ---------------------------------------------------------------------------
# training_trials
# ---------------------------------------------------------------------------


def problems_of(table):
    return sorted(set(zip(table["mouse"], table["phase"], table["phase_problem"], strict=True)))


def test_structured_problems_are_the_blocked_problems_before_the_first_drifting_one():
    late_blocked = [*SEQUENCE[:3], ("blocked", 3), *SEQUENCE[3:]]
    trials = mouse_trials("m1", 1, late_blocked, n_trials=10)
    kept = stages.training_trials(trials, {1: None})
    assert problems_of(kept) == [
        ("m1", "drifting", 1), ("m1", "drifting", 2), ("m1", "drifting", 3),
        ("m1", "structured", 1), ("m1", "structured", 2),
    ]  # fmt: skip


def test_drifting_problems_after_the_cutoff_are_dropped_per_cohort():
    trials = pd.concat([mouse_trials("m1", 1, SEQUENCE, n_trials=10),
                        mouse_trials("m3", 3, SEQUENCE, n_trials=10)])  # fmt: skip
    kept = stages.training_trials(trials, {1: 2, 3: None})
    drifting = kept[kept["phase"] == lbp.DRIFTING].groupby("mouse")["phase_problem"].max()
    assert drifting.to_dict() == {"m1": 2, "m3": 3}
    assert set(kept.loc[kept["phase"] == lbp.STRUCTURED, "phase_problem"]) == {1, 2}


def test_a_mouse_without_drifting_problems_keeps_its_structured_ones():
    trials = mouse_trials("m1", 1, SEQUENCE[:2], n_trials=10)
    kept = stages.training_trials(trials, {1: None})
    assert set(kept["phase"]) == {lbp.STRUCTURED}


def test_a_cohort_without_a_cutoff_entry_raises():
    trials = mouse_trials("m1", 1, SEQUENCE, n_trials=10)
    with pytest.raises(ValueError, match="cohort 1"):
        stages.training_trials(trials, {3: None})


def test_an_unknown_task_raises():
    trials = mouse_trials("m1", 1, SEQUENCE, n_trials=10).assign(task="pretraining")
    with pytest.raises(ValueError, match="pretraining"):
        stages.training_trials(trials, {1: None})


# ---------------------------------------------------------------------------
# fits and summaries
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted():
    trials = pd.concat([mouse_trials(f"m{i}", 1, SEQUENCE, n_trials=300, seed=i)
                        for i in range(3)])  # fmt: skip
    return lbp.fit_by_problem(stages.training_trials(trials, {1: None}))


def test_every_problem_of_every_mouse_gets_one_fit(fitted):
    fits = fitted.groupby(["mouse", "phase", "phase_problem"])["n_trials"].first()
    assert len(fits) == 3 * len(SEQUENCE)
    assert set(fits) == {300}
    # reward and choice lags, and the two towers that are not the reference
    n_features = len(cl.feature_names(cl.DEFAULT_LAGS, ["B1", "C1"]))
    per_fit = fitted.groupby(["mouse", "phase", "phase_problem"]).size()
    assert (per_fit == n_features).all()


def test_a_learner_that_follows_the_last_payout_has_a_positive_lag_one_reward_weight(fitted):
    lag_one = fitted[fitted["feature"] == "reward_lag1"]["weight"]
    assert (lag_one > 0).all()


def test_problems_with_too_few_trials_get_no_fit():
    trials = mouse_trials("m1", 1, SEQUENCE[:3], n_trials=lbp.MIN_TRIALS - 1)
    assert lbp.fit_by_problem(stages.training_trials(trials, {1: None})).empty


def test_lag_one_and_the_mean_of_the_older_lags_are_summarized():
    weights = pd.DataFrame(
        dict(mouse="m", cohort=1, phase=lbp.DRIFTING, phase_problem=1, n_trials=200,
             feature=["reward_lag1", "reward_lag2", "reward_lag3", "choice_lag1", "choice_lag2",
                      "tower_B1"],
             group=["reward", "reward", "reward", "choice", "choice", "tower"],
             lag=[1, 2, 3, 1, 2, np.nan], weight=[0.5, 0.2, 0.4, -0.1, 0.3, 9.0])
    )  # fmt: skip
    summaries = lbp.weight_summaries(weights).set_index(["group", "summary"])["weight"]
    assert summaries.to_dict() == pytest.approx({
        ("choice", lbp.LAG_ONE): -0.1, ("choice", lbp.OLDER): 0.3,
        ("reward", lbp.LAG_ONE): 0.5, ("reward", lbp.OLDER): 0.3,
    })  # fmt: skip


# ---------------------------------------------------------------------------
# trends and tests
# ---------------------------------------------------------------------------


def summary_rows(mouse, structured, drifting):
    rows = [(lbp.STRUCTURED, k, w) for k, w in enumerate(structured, start=1)]
    rows += [(lbp.DRIFTING, k, w) for k, w in enumerate(drifting, start=1)]
    return pd.DataFrame([dict(mouse=mouse, cohort=1, phase=p, phase_problem=k, group="reward",
                              summary=lbp.LAG_ONE, weight=w) for p, k, w in rows])  # fmt: skip


def test_slopes_within_each_phase_and_the_step_at_the_switch():
    trends = lbp.phase_trends(summary_rows("m", [0.0, 0.1, 0.2], [0.5, 0.4, 0.3, 0.2]))
    row = trends.iloc[0]
    assert row["slope_structured"] == pytest.approx(0.1)
    assert row["slope_drifting"] == pytest.approx(-0.1)
    assert row["step"] == pytest.approx(0.5 - 0.2)


def test_a_phase_with_one_problem_has_no_slope_and_one_phase_has_no_step():
    trends = lbp.phase_trends(summary_rows("m", [0.3], []))
    row = trends.iloc[0]
    assert np.isnan(row["slope_structured"]) and np.isnan(row["slope_drifting"])
    assert np.isnan(row["step"])


def test_trend_tests_summarize_across_mice_and_need_two_mice_for_a_test():
    summaries = pd.concat([summary_rows("a", [0.0, 0.1], [0.5, 0.6]),
                           summary_rows("b", [0.0, 0.3], [0.2, 0.2])])  # fmt: skip
    tests = lbp.trend_tests(lbp.phase_trends(summaries)).set_index("measure")
    assert tests.loc["slope_structured", "mean"] == pytest.approx(0.2)
    assert tests.loc["slope_structured", "n_mice"] == 2
    assert np.isfinite(tests.loc["slope_structured", "p"])
    single = lbp.trend_tests(lbp.phase_trends(summary_rows("a", [0.0, 0.1], [0.5, 0.6])))
    assert single["p"].isna().all()


# ---------------------------------------------------------------------------
# loading and the pipeline
# ---------------------------------------------------------------------------


def session(task_name, problem, n_trials=5):
    frame = pd.DataFrame({"type": []})
    frame.attrs["source_path"] = f"/raw/sub-01_id-A/ses-{problem}_date-2026010{problem}/A.tsv"
    return {"task_name": task_name, "df": frame, "problem": problem,
            "choice": ["A1"] * n_trials, "reward_magnitudes": [dict(A1=1, B1=0, C1=4)] * n_trials,
            "initiation_tower": "B2"}  # fmt: skip


def test_a_cohort_table_numbers_each_problem_within_its_type_and_overall():
    names = [BLOCKED_NAME, DRIFTING_NAME, BLOCKED_NAME]
    imported = {
        p: {"A": {f"ses-{p}_date-2026010{p}": session(name, p)}}
        for p, name in enumerate(names, start=1)
    }
    table = pipe.cohort_trial_table(4, imported)
    problems = table.groupby(["task", "k"])["overall_problem"].first().to_dict()
    assert problems == {("blocked", 1): 1, ("drifting", 1): 2, ("blocked", 2): 3}
    assert set(table["cohort"]) == {4}


def test_the_pipeline_writes_the_tables_and_both_figures_of_each_coding(tmp_path):
    # 300 trials: an early part of 150 and a late part of 150
    trials = pd.concat([mouse_trials(f"m{i}", 1, SEQUENCE, n_trials=300, seed=i)
                        for i in range(3)])  # fmt: skip
    results = pipe.run_logit_by_problem(trials, {1: None}, tmp_path)
    assert all(not r.early.empty and not r.late.empty for r in results.values())
    assert set(results) == set(lbp.CODINGS)
    assert set(results[lbp.OUTCOME].tests["group"]) == {lbp.OUTCOME_X_CHOICE, cl.CHOICE}
    assert all(len(r.tests) == 12 for r in results.values())
    written = {p.name for p in tmp_path.rglob("*") if p.is_file()}
    for coding in lbp.CODINGS:
        for base in plots.FIGURE_NAMES:
            name = plots.figure_name(base, coding)
            assert {f"{name}.{ext}" for ext in ("pdf", "png", "svg", "caption.txt")} <= written
        for table in ("weights", "summaries", "trends", "tests"):
            assert f"logit_by_problem_{coding}_{table}.csv" in written


# ---------------------------------------------------------------------------
# the paper's coding
# ---------------------------------------------------------------------------


def test_outcome_x_choice_is_plus_one_after_a_paid_choice_minus_one_after_a_zero_else_zero():
    rows = pd.DataFrame(problem_rows("m", 1, "drifting", 1, 1, 40, np.random.default_rng(0)))
    design = lbp.outcome_design(cl.design_matrix(rows, n_lags=2, extras=(cl.BINARY,)))
    names = design.feature_names
    interaction = design.X[:, :, names.index("binary_lag1")]
    choice = design.X[:, :, names.index("choice_lag1")]
    previous = rows["reward"].shift(1).to_numpy()
    for t in range(1, len(rows)):
        expected = choice[t] * (1 if previous[t] > 0 else -1)
        assert interaction[t] == pytest.approx(expected)
    assert not any(n.startswith("reward_") for n in names)


def test_the_outcome_coding_names_its_weights_outcome_x_choice():
    rows = pd.DataFrame(problem_rows("m", 1, "drifting", 1, 1, 300, np.random.default_rng(1)))
    weights = lbp.fit_problem(rows, n_lags=3, coding=lbp.OUTCOME)
    lagged = weights[weights["lag"].notna()]
    assert set(lagged["group"]) == {lbp.OUTCOME_X_CHOICE, cl.CHOICE}
    assert "outcome_x_choice_lag1" in set(weights["feature"])
    # a learner that follows the last payout stays after a paid choice and leaves after a zero
    assert weights.set_index("feature").loc["outcome_x_choice_lag1", "weight"] > 0


def test_an_unknown_coding_raises():
    rows = pd.DataFrame(problem_rows("m", 1, "drifting", 1, 1, 50, np.random.default_rng(0)))
    with pytest.raises(ValueError, match="coding"):
        lbp.fit_problem(rows, coding="binary")


def test_drifting_problems_are_placed_after_the_structured_ones_with_a_gap():
    table = pd.DataFrame(dict(phase=[lbp.STRUCTURED] * 2 + [lbp.DRIFTING] * 2,
                              phase_problem=[1, 2, 1, 2]))  # fmt: skip
    assert plots.problem_positions(table) == {
        (lbp.STRUCTURED, 1): 1.0, (lbp.STRUCTURED, 2): 2.0,
        (lbp.DRIFTING, 1): 4.0, (lbp.DRIFTING, 2): 5.0,
    }  # fmt: skip
