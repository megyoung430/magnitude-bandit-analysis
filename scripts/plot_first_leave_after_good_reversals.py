import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
from src.behavior_import.import_data import *
from src.behavior_import.extract_trials import *
from src.behavior_analysis.get_good_reversal_info import *
from src.behavior_analysis.get_first_leave_after_good_reversals import *
from src.behavior_analysis.split_early_late_good_reversals import *
from src.behavior_analysis.get_diagnostic_p_value import *
from src.behavior_visualization.plot_first_leave_after_good_reversals import *

cohort = "cohort-02"
problem_number = 2
problem = f"problem-{problem_number:02d}"
root = f"../data/{cohort}/{problem}/rawdata/"

subjects_data = import_data(root)
subjects_trials = extract_trials(subjects_data)

reversal_windows = get_good_reversal_info(subjects_trials, include_first_block=False)
first_leave_per_subject = get_first_leave_after_good_reversals(reversal_windows)
mean, se, n_subjects = average_first_leave_across_subjects(first_leave_per_subject)
p_value = pvalue_paired_t_new_vs_third(first_leave_per_subject, alternative="greater")
curr_save_path = Path(f"../results/figures/{cohort}/{problem}/reversal-stats/First Leave After Good Reversals")
plot_first_leave_after_good_reversals(mean, se, first_leave_per_subject, p_value, save_path=curr_save_path)

early, late = split_good_reversals_early_late(reversal_windows, first_n=2)
first_leave_per_subject_early = get_first_leave_after_good_reversals(early)
mean_early, se_early, n_subjects_early = average_first_leave_across_subjects(first_leave_per_subject_early)
p_value_early = pvalue_paired_t_new_vs_third(first_leave_per_subject_early, alternative="greater")
curr_save_path = Path(f"../results/figures/{cohort}/{problem}/reversal-stats/First Leave After Good Reversals (Early)")
plot_first_leave_after_good_reversals(mean_early, se_early, first_leave_per_subject_early, p_value_early, save_path=curr_save_path)

first_leave_per_subject_late = get_first_leave_after_good_reversals(late)
mean_late, se_late, n_subjects_late = average_first_leave_across_subjects(first_leave_per_subject_late)
p_value_late = pvalue_paired_t_new_vs_third(first_leave_per_subject_late, alternative="greater")
curr_save_path = Path(f"../results/figures/{cohort}/{problem}/reversal-stats/First Leave After Good Reversals (Late)")
plot_first_leave_after_good_reversals(mean_late, se_late, first_leave_per_subject_late, p_value_late, save_path=curr_save_path)