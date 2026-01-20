import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
from src.behavior_import.import_data import *
from src.behavior_import.extract_trials import *
from src.behavior_analysis.get_good_reversal_info import *
from src.behavior_analysis.get_choice_probs_around_good_reversals import *
from src.behavior_analysis.split_early_late_good_reversals import *
from src.behavior_visualization.plot_choice_probs_around_good_reversals import *

# With all trials after reversal
cohort = "cohort-02"
root = f"../data/{cohort}/rawdata/"
pre = 10
post = 30

subjects_data = import_data(root)
subjects_trials = extract_trials(subjects_data)
reversal_windows = get_good_reversal_info(subjects_trials, pre=pre, post=post, include_first_block=False)
x, per_subject, across = get_choice_probs_around_good_reversals(reversal_windows, pre=pre, post=post)
curr_save_path = Path(f"../results/figures/{cohort}/reversal-stats/Choice Probabilities Around Good Reversals")
plot_reversal_probs_around_good_reversals(x, across, show_chance=True, skip_n_trials_after_reversal=0, save_path=curr_save_path)

early, late = split_good_reversals_early_late(reversal_windows)
x, per_subject, across = get_choice_probs_around_good_reversals(early, pre=pre, post=post)
curr_save_path = Path(f"../results/figures/{cohort}/reversal-stats/Choice Probabilities Around Good Reversals (Early)")
plot_reversal_probs_around_good_reversals(x, across, show_chance=True, skip_n_trials_after_reversal=0, save_path=curr_save_path)

x, per_subject, across = get_choice_probs_around_good_reversals(late, pre=pre, post=post)
curr_save_path = Path(f"../results/figures/{cohort}/reversal-stats/Choice Probabilities Around Good Reversals (Late)")
plot_reversal_probs_around_good_reversals(x, across, show_chance=True, skip_n_trials_after_reversal=0, save_path=curr_save_path)

# Skipping trials after a reversal
pre = 10
post = 30
skip_n_trials_after_reversal = 10

subjects_data = import_data(root)
subjects_trials = extract_trials(subjects_data)
reversal_windows = get_good_reversal_info(subjects_trials, pre=pre, post=post, include_first_block=False)
x, per_subject, across = get_choice_probs_around_good_reversals(reversal_windows, pre=pre, post=post, skip_n_trials_after_reversal=skip_n_trials_after_reversal)
curr_save_path = Path(f"../results/figures/{cohort}/reversal-stats/Choice Probabilities Around Good Reversals, Skipping Initial Trials")
plot_reversal_probs_around_good_reversals(x, across, show_chance=True, skip_n_trials_after_reversal=skip_n_trials_after_reversal, save_path=curr_save_path)

early, late = split_good_reversals_early_late(reversal_windows)
x, per_subject, across = get_choice_probs_around_good_reversals(early, pre=pre, post=post, skip_n_trials_after_reversal=skip_n_trials_after_reversal)
curr_save_path = Path(f"../results/figures/{cohort}/reversal-stats/Choice Probabilities Around Good Reversals, Skipping Initial Trials (Early)")
plot_reversal_probs_around_good_reversals(x, across, show_chance=True, skip_n_trials_after_reversal=skip_n_trials_after_reversal, save_path=curr_save_path)

x, per_subject, across = get_choice_probs_around_good_reversals(late, pre=pre, post=post, skip_n_trials_after_reversal=skip_n_trials_after_reversal)
curr_save_path = Path(f"../results/figures/{cohort}/reversal-stats/Choice Probabilities Around Good Reversals, Skipping Initial Trials (Late)")
plot_reversal_probs_around_good_reversals(x, across, show_chance=True, skip_n_trials_after_reversal=skip_n_trials_after_reversal, save_path=curr_save_path)