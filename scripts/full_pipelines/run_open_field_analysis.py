"""Make every standard open-field figure for one cohort.

Edit the settings below, then run:

    uv run python scripts/full_pipelines/run_open_field_analysis.py

Figures are written under results/figures/<task>/<cohort>/problem-NN/.

This script applies one analysis selection (the default: choice probabilities, first leave and
the other sections) to every problem of one cohort. That suits the blocked-task cohort in
FOLDER_NAME, whose problems are all blocked. For cohorts 1-4, or any cohort that has drifting
problems, use scripts/full_pipelines/run_cohorts.py: it reads the task type of each session from
its task name and runs the analyses that apply to each type.
"""

from src.behavior_import.load_trials import load_trials_by_problem
from src.pipelines.open_field import ChoiceProbabilitySettings, run_open_field_analysis
from src.settings import REPO_ROOT, get_settings

TASK = "open-field"
COHORT = "cohort-01"
FOLDER_NAME = "3x3_field_blocked_reward_bandit"

CHOICE = ChoiceProbabilitySettings(
    run_all=True,
    skip=True,
    moving_avg=True,
    remove_bad=True,
    split_by_best_change=False,
    split_by_best_change_and_half=False,
    split_by_first_two=True,
    split_by_half=True,
    pre=10,
    post=30,
    skip_n_trials_after_reversal=15,
    moving_avg_window=4,
)


def main() -> None:
    print(f"Running analysis for {TASK} task, {COHORT}")
    print("Choice probability settings:")
    for name, value in CHOICE.model_dump().items():
        print(f"  {name}: {value}")

    raw_data = get_settings().data_root / FOLDER_NAME / COHORT / "rawdata"
    subjects_trials_by_problem = load_trials_by_problem(raw_data)
    run_open_field_analysis(
        subjects_trials_by_problem,
        task=TASK,
        cohort=COHORT,
        output_root=REPO_ROOT / "results" / "figures",
        choice=CHOICE,
    )


if __name__ == "__main__":
    main()
