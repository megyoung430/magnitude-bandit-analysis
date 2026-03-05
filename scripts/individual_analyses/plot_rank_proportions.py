import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
from src.behavior_import.import_data import *
from src.behavior_import.extract_trials import *
from src.behavior_analysis.get_good_reversal_info import *
from src.behavior_analysis.get_rank_counts_by_good_reversal import *
from src.behavior_analysis.get_diagnostic_p_value import *
from src.behavior_visualization.plot_rank_proportions import *
sys.path.append(str(Path(__file__).resolve().parent.parent))
from fix_grid_maze_cohort_02_problems import *

task = "grid-maze"
# task = "open-field"

folder_name = None
cohort = None
if task == "grid-maze":
    cohort = "cohort-02"
    folder_name = "3x3_maze_blocked_reward_bandit"
elif task == "open-field":
    cohort = "cohort-01"
    folder_name = "3x3_field_blocked_reward_bandit"
root = f"/Volumes/behrens/meg/{folder_name}/{cohort}/rawdata/"

subjects_data = import_data(root)
subjects_trials_by_problem = extract_trials_grouped_by_problem(subjects_data)
if task == "grid-maze" and cohort == "cohort-02":
    subjects_trials_by_problem = fix_grid_maze_cohort_02_problems(subjects_trials_by_problem)

for problem_number in subjects_trials_by_problem.keys():
    print(problem_number)
    problem = f"problem-{problem_number:02d}"
    
    subjects_trials = subjects_trials_by_problem[problem_number]

    reversal_windows = get_good_reversal_info(subjects_trials, include_first_block=True)
    rank_counts_by_good_reversal = get_rank_counts_by_good_reversal(reversal_windows)
    p_values = pvalue_paired_t_best_vs_second_vs_third(rank_counts_by_good_reversal)

    save_path = f"../results/figures/{task}/{cohort}/{problem}/choice-stats/Rank Proportions"
    plot_rank_proportions(rank_counts_by_good_reversal, average_across_mice_pvalues=p_values, save_path=save_path)