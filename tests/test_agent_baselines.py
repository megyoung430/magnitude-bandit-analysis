"""The agent-baselines pipeline end to end, on a small synthetic cohort."""

from src.pipelines.agent_baselines import (
    SOURCES,
    cohorts_folder_name,
    compute_agent_baselines,
    write_agent_baselines,
)
from src.pipelines.all_cohorts_drifting import drifting_problems, pool_cohorts
from src.pipelines.cohort_analysis import DRIFTING_SELECTION
from tests.all_cohorts_fixtures import synthetic_cohort


def _pooled_problems():
    cohort = synthetic_cohort(["m1", "m2", "m3"])
    return pool_cohorts({1: drifting_problems(cohort)}).problems


def test_compute_and_write(tmp_path):
    result = compute_agent_baselines(_pooled_problems(), n_starts=1, min_trials=50)
    assert sorted(result.fits["kind"].unique()) == ["binary", "magnitude"]
    assert result.fits["mouse"].nunique() == 3
    assert set(result.weights["source"]) == set(SOURCES)
    assert set(result.summary["source"]) == set(SOURCES)
    for kind in ("magnitude", "binary"):
        animals = {a for subjects in result.simulated[kind].values() for a in subjects}
        assert animals == {f"m{i}-{kind}" for i in (1, 2, 3)}
    # the synthetic task zeroes the arm with the highest value, so this is a real check
    assert result.task.summary.set_index("mouse").loc["all", "n_block_changes"] > 0

    figures, tables = tmp_path / "figures", tmp_path / "tables"
    write_agent_baselines(result, figures, tables, DRIFTING_SELECTION, "test")
    assert (figures / "summary.png").exists()
    for name in ("agent_fits", "choice_regression_weights", "task_rules_summary"):
        assert (tables / f"{name}.csv").exists()
    for kind in ("magnitude", "binary"):
        assert any((figures / f"{kind}-agent" / "choice-stats").glob("*.png"))
        assert any((figures / f"{kind}-agent" / "value-psychometrics").glob("*.png"))


def test_cohorts_folder_name():
    assert cohorts_folder_name([3, 1]) == "cohorts-1-3"
