"""Re-run only the value-change plots for some cohorts, into the usual results folders."""

from src.behavior_import.load_trials import load_trials_with_log
from src.behavior_import.task_types import DRIFTING, number_within_task_type, regroup_by_task_type
from src.pipelines.cohort_analysis import (
    DEFAULT_RESULTS_ROOT,
    POOLED_DRIFTING_FOLDER,
    cohort_name,
    cohort_raw_dir,
)
from src.pipelines.open_field import VALUE_CHANGE_FOLDER
from src.pipelines.value_change_analysis import run_value_change_analysis
from src.settings import get_settings

COHORTS = [1, 2, 3, 4]

for cohort_id in COHORTS:
    cohort = cohort_name(cohort_id)
    raw = cohort_raw_dir(get_settings().data_root, cohort_id)
    trials = load_trials_with_log(raw, use_cache=True, classify_task_types=True).trials_by_problem
    grouped = regroup_by_task_type(trials, number_within_task_type(trials))
    drifting = {k: subjects for (kind, k), subjects in grouped.items() if kind == DRIFTING}
    figures = DEFAULT_RESULTS_ROOT / "figures" / "open-field" / cohort / DRIFTING
    for k, subjects in drifting.items():
        print(f"{cohort} drifting problem {k}")
        run_value_change_analysis({k: subjects}, figures / f"problem-{k:02d}" / VALUE_CHANGE_FOLDER)
    if drifting:
        print(f"{cohort} all drifting problems")
        run_value_change_analysis(drifting, figures / POOLED_DRIFTING_FOLDER / VALUE_CHANGE_FOLDER)
