from pathlib import Path

from src.behavior_import.fix_grid_maze_cohort_02_problems import fix_grid_maze_cohort_02_problems
from src.behavior_import.load_trials import load_trials_by_problem
from src.behavior_visualization.plot_num_reversals import (
    plot_moving_avg_reversals_over_time,
    plot_num_reversals,
    plot_num_reversals_over_time,
)
from src.settings import get_settings

# task = "grid-maze"
task = "open-field"

folder_name = None
cohort = None
if task == "grid-maze":
    cohort = "cohort-02"
    folder_name = "3x3_maze_blocked_reward_bandit"
elif task == "open-field":
    cohort = "cohort-02"
    folder_name = "3x3_field_blocked_reward_bandit"
root = get_settings().data_root / folder_name / cohort / "rawdata"

subjects_trials_by_problem = load_trials_by_problem(root)
if task == "grid-maze" and cohort == "cohort-02":
    subjects_trials_by_problem = fix_grid_maze_cohort_02_problems(subjects_trials_by_problem)

for problem_number in subjects_trials_by_problem.keys():
    print(problem_number)
    problem = f"problem-{problem_number:02d}"

    subjects_trials = subjects_trials_by_problem[problem_number]

    curr_save_path = Path(f"../../results/figures/{task}/{cohort}/{problem}/reversal-stats/Total Reversals")
    plot_num_reversals(subjects_trials, save_path=curr_save_path)

    curr_save_path = Path(f"../../results/figures/{task}/{cohort}/{problem}/reversal-stats/Cumulative Reversals Over Time")
    plot_num_reversals_over_time(subjects_trials, threshold=10, save_path=curr_save_path)

    curr_save_path = Path(f"../../results/figures/{task}/{cohort}/{problem}/reversal-stats/Moving Average Reversals Over Time")
    plot_moving_avg_reversals_over_time(subjects_trials, save_path=curr_save_path)
