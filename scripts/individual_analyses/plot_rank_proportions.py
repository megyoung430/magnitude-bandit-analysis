from src.behavior_analysis.get_diagnostic_p_value import pvalue_paired_t_best_vs_second_vs_third
from src.behavior_analysis.get_good_reversal_info import get_good_reversal_info
from src.behavior_analysis.get_rank_counts_by_good_reversal import get_rank_counts_by_good_reversal
from src.behavior_import.fix_grid_maze_cohort_02_problems import fix_grid_maze_cohort_02_problems
from src.behavior_import.load_trials import load_trials_by_problem
from src.behavior_visualization.plot_rank_proportions import plot_rank_proportions
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

    reversal_windows = get_good_reversal_info(subjects_trials, include_first_block=True)
    rank_counts_by_good_reversal = get_rank_counts_by_good_reversal(reversal_windows)
    p_values = pvalue_paired_t_best_vs_second_vs_third(rank_counts_by_good_reversal)

    save_path = f"../../results/figures/{task}/{cohort}/{problem}/choice-stats/Rank Proportions"
    plot_rank_proportions(rank_counts_by_good_reversal, average_across_mice_pvalues=p_values, save_path=save_path)
