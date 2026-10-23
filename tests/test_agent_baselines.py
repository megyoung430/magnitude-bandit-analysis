"""The agent-baselines pipeline end to end, on a small synthetic cohort."""

import numpy as np
import pandas as pd
import pytest

from src.agents import drifting_simulation as sim
from src.behavior_analysis.drifting_trial_table import drifting_trial_table
from src.pipelines.agent_baselines import (
    SOURCES,
    check_block_change_rule,
    cohorts_folder_name,
    compute_agent_baselines,
    simulate_agents,
    write_agent_baselines,
)
from src.pipelines.cohort_analysis import DRIFTING_SELECTION

LAYOUTS = (
    sim.MouseLayout(["A1", "B3", "C2"], "B2", {"A1": 4, "B3": 1, "C2": 0}, [90, 110, 100]),
    sim.MouseLayout(["A3", "B1", "C3"], "B2", {"A3": 0, "B1": 3, "C3": 2}, [120, 80]),
)
RULES = sim.TaskRules(drift_rate=0.12, block_lengths=np.array([28, 32, 36]))


def _pooled_problems():
    """Three synthetic mice, two drifting problems each, played by a magnitude learner.

    The problems are keyed ``(cohort, k)`` as in ``pool_cohorts``, and the sessions follow the
    real task's block-change rule, which the pipeline checks.
    """
    problems: dict = {}
    for m in range(3):
        rng = np.random.default_rng(m)
        for k, layout in enumerate(LAYOUTS, start=1):
            sessions = sim.simulate_problem((0.4, 1.0, 1.0, 0.5), "magnitude", layout, RULES, rng)
            mouse = f"m{m + 1}"
            problems.setdefault((1, k), {})[mouse] = sim.simulated_sessions(
                mouse, k, layout, sessions
            )
    return problems


def test_compute_and_write(tmp_path):
    result = compute_agent_baselines(_pooled_problems(), n_starts=1)
    assert sorted(result.fits["kind"].unique()) == ["binary", "magnitude"]
    assert result.fits["mouse"].nunique() == 3
    assert set(result.weights["source"]) == set(SOURCES)
    assert set(result.summary["source"]) == set(SOURCES)
    for kind in ("magnitude", "binary"):
        animals = {a for subjects in result.simulated[kind].values() for a in subjects}
        assert animals == {f"m{i}-{kind}" for i in (1, 2, 3)}
    overall = result.task.summary.set_index("mouse").loc["all"]
    assert overall["n_block_changes"] > 20
    assert overall["frac_inherits_value"] == 1.0

    figures, tables = tmp_path / "figures", tmp_path / "tables"
    write_agent_baselines(result, figures, tables, DRIFTING_SELECTION, "test")
    for ext in ("pdf", "png", "svg", "caption.txt"):
        assert (figures / f"summary.{ext}").exists(), ext
    for name in ("agent_fits", "choice_regression_weights", "task_rules_summary"):
        assert (tables / f"{name}.csv").exists()
    for kind in ("magnitude", "binary"):
        assert any((figures / f"{kind}-agent" / "choice-stats").glob("*.png"))
        assert any((figures / f"{kind}-agent" / "value-psychometrics").glob("*.png"))


def test_cohorts_folder_name():
    assert cohorts_folder_name([3, 1]) == "cohorts-1-3"


def _summary(frac_most, frac_inherits):
    return pd.DataFrame([dict(mouse="all", frac_most=frac_most, frac_inherits_value=frac_inherits)])


def test_block_change_rule_check():
    check_block_change_rule(_summary(0.94, 0.998))
    with pytest.raises(ValueError, match="do not follow the simulated block-change rule"):
        check_block_change_rule(_summary(0.5, 0.998))
    with pytest.raises(ValueError, match="do not follow"):
        check_block_change_rule(_summary(0.94, 0.5))


def test_a_mouse_simulation_does_not_depend_on_the_other_mice():
    problems = _pooled_problems()
    table = drifting_trial_table(problems)
    rules = sim.TaskRules(0.1, np.array([30]))
    fits = [sim.AgentFit(m, k, (0.4, 1.0, 1.0, 0.5), -1.0, 1) for m in ("m1", "m2", "m3")
            for k in sim.AGENT_KINDS]  # fmt: skip
    alone = simulate_agents(table[table["mouse"] == "m2"], fits[2:4], rules, seed=0)
    together = simulate_agents(table, fits, rules, seed=0)
    for kind in sim.AGENT_KINDS:
        label = f"m2-{kind}"
        for problem, animals in alone[kind].items():
            a = drifting_trial_table({problem: {label: animals[label]}})
            b = drifting_trial_table({problem: {label: together[kind][problem][label]}})
            assert a["choice"].tolist() == b["choice"].tolist()


def test_the_regression_panel_states_the_mean_and_the_number_of_animals():
    import pandas as pd

    from src.behavior_analysis.choice_regression import FEATURES
    from src.behavior_visualization.plot_agent_baselines import plot_regression_weights

    weights = pd.DataFrame(
        [
            {"source": source, "mouse": f"m{i}", "feature": feature, "weight": float(i)}
            for source in ("mice", "magnitude")
            for i in range(3)
            for feature in FEATURES
        ]
    )
    ax = plot_regression_weights(weights)
    assert ax.get_title() == (
        "Choice regression: what drives port choice\n"
        "mean ± se across animals of each source, n=3 animals"
    )
