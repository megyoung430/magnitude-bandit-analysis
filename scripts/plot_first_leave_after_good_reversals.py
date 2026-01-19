import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
from src.behavior_import.import_data import *
from src.behavior_import.extract_trials import *
from src.behavior_analysis.get_good_reversal_info import *
from src.behavior_analysis.get_first_leave_after_good_reversals import *
from src.behavior_analysis.split_early_late_good_reversals import *
from src.behavior_visualization.plot_first_leave_after_good_reversals import *

cohort = "cohort-02"
root = f"../data/{cohort}/rawdata/"
pre=10
post=40

subjects_data = import_data(root)
subjects_trials = extract_trials(subjects_data)

reversal_windows = get_good_reversal_info(subjects_trials, pre=pre, post=post, include_first_block=False)
first_leave_per_subject = get_first_leave_after_good_reversals(reversal_windows)
mean, std, n_subjects = average_first_leave_across_subjects(first_leave_per_subject)
curr_save_path = Path(f"../results/figures/{cohort}/reversal-stats/First Leave After Good Reversals")
plot_first_leave_after_good_reversals(mean, std, first_leave_per_subject, save_path=curr_save_path)

early, late = split_good_reversals_early_late(reversal_windows)
first_leave_per_subject_early = get_first_leave_after_good_reversals(early)
mean_early, std_early, n_subjects_early = average_first_leave_across_subjects(first_leave_per_subject_early)
curr_save_path = Path(f"../results/figures/{cohort}/reversal-stats/First Leave After Good Reversals (Early)")
plot_first_leave_after_good_reversals(mean_early, std_early, first_leave_per_subject_early, save_path=curr_save_path)

first_leave_per_subject_late = get_first_leave_after_good_reversals(late)
mean_late, std_late, n_subjects_late = average_first_leave_across_subjects(first_leave_per_subject_late)
curr_save_path = Path(f"../results/figures/{cohort}/reversal-stats/First Leave After Good Reversals (Late)")
plot_first_leave_after_good_reversals(mean_late, std_late, first_leave_per_subject_late, save_path=curr_save_path)