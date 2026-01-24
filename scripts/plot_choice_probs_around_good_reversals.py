import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
from src.behavior_import.import_data import *
from src.behavior_import.extract_trials import *
from src.behavior_analysis.get_good_reversal_info import *
from src.behavior_analysis.get_choice_probs_around_good_reversals import *
from src.behavior_analysis.split_early_late_good_reversals import *
from src.behavior_visualization.plot_choice_probs_around_good_reversals import *

all = True
skip = True
moving_avg = True

cohort = "cohort-02"
root = f"../data/{cohort}/rawdata/"

if all:
    # With all trials after reversal
    pre = 25
    post = 50

    subjects_data = import_data(root)
    subjects_trials = extract_trials(subjects_data)
    reversal_windows = get_good_reversal_info(subjects_trials, pre=pre, post=post, include_first_block=False)
    x, per_subject, across = get_choice_probs_around_good_reversals(reversal_windows, pre=pre, post=post)
    curr_save_path = Path(f"../results/figures/{cohort}/reversal-stats/Choice Probabilities Around Good Reversals")
    plot_choice_probs_around_good_reversals(x, across, show_chance=True, skip_n_trials_after_reversal=0, save_path=curr_save_path)

    early, late = split_good_reversals_early_late(reversal_windows, first_n=2)
    x, per_subject, across = get_choice_probs_around_good_reversals(early, pre=pre, post=post)
    curr_save_path = Path(f"../results/figures/{cohort}/reversal-stats/Choice Probabilities Around Good Reversals (Early)")
    plot_choice_probs_around_good_reversals(x, across, show_chance=True, skip_n_trials_after_reversal=0, save_path=curr_save_path)

    x, per_subject, across = get_choice_probs_around_good_reversals(late, pre=pre, post=post)
    curr_save_path = Path(f"../results/figures/{cohort}/reversal-stats/Choice Probabilities Around Good Reversals (Late)")
    plot_choice_probs_around_good_reversals(x, across, show_chance=True, skip_n_trials_after_reversal=0, save_path=curr_save_path)

if skip:
    # Skipping trials after a reversal
    pre = 10
    post = 50
    skip_n_trials_after_reversal = 15

    subjects_data = import_data(root)
    subjects_trials = extract_trials(subjects_data)
    reversal_windows = get_good_reversal_info(subjects_trials, pre=pre, post=post, include_first_block=False)
    x, per_subject, across = get_choice_probs_around_good_reversals(reversal_windows, pre=pre, post=post, skip_n_trials_after_reversal=skip_n_trials_after_reversal)
    curr_save_path = Path(f"../results/figures/{cohort}/reversal-stats/Choice Probabilities Around Good Reversals, Skipping Initial Trials")
    plot_choice_probs_around_good_reversals(x, across, show_chance=True, skip_n_trials_after_reversal=skip_n_trials_after_reversal, save_path=curr_save_path)

    early, late = split_good_reversals_early_late(reversal_windows, first_n=2)
    x, per_subject, across = get_choice_probs_around_good_reversals(early, pre=pre, post=post, skip_n_trials_after_reversal=skip_n_trials_after_reversal)
    curr_save_path = Path(f"../results/figures/{cohort}/reversal-stats/Choice Probabilities Around Good Reversals, Skipping Initial Trials (Early)")
    plot_choice_probs_around_good_reversals(x, across, show_chance=True, skip_n_trials_after_reversal=skip_n_trials_after_reversal, save_path=curr_save_path)

    x, per_subject, across = get_choice_probs_around_good_reversals(late, pre=pre, post=post, skip_n_trials_after_reversal=skip_n_trials_after_reversal)
    curr_save_path = Path(f"../results/figures/{cohort}/reversal-stats/Choice Probabilities Around Good Reversals, Skipping Initial Trials (Late)")
    plot_choice_probs_around_good_reversals(x, across, show_chance=True, skip_n_trials_after_reversal=skip_n_trials_after_reversal, save_path=curr_save_path)

if moving_avg:
    # With moving average
    pre = 25
    post = 50
    moving_avg_window = 4

    subjects_data = import_data(root)
    subjects_trials = extract_trials(subjects_data)
    reversal_windows = get_good_reversal_info(subjects_trials, pre=pre, post=post, include_first_block=False)
    x, per_subject, across = get_choice_probs_around_good_reversals(reversal_windows, pre=pre, post=post)
    x, per_subject_moving_avg, across_moving_avg = apply_moving_average_to_choice_probs(x, per_subject, moving_avg_window=moving_avg_window, mode="centered")

    curr_save_path = Path(f"../results/figures/{cohort}/reversal-stats/Choice Probabilities Around Good Reversals (Moving Average)")
    plot_choice_probs_around_good_reversals(x, across_moving_avg, show_chance=True, save_path=curr_save_path)

    early, late = split_good_reversals_early_late(reversal_windows, first_n=2)
    x, per_subject, across = get_choice_probs_around_good_reversals(early, pre=pre, post=post)
    x, per_subject_moving_avg, across_moving_avg = apply_moving_average_to_choice_probs(x, per_subject, moving_avg_window=moving_avg_window, mode="centered")
    curr_save_path = Path(f"../results/figures/{cohort}/reversal-stats/Choice Probabilities Around Good Reversals (Moving Average) (Early)")
    plot_choice_probs_around_good_reversals(x, across_moving_avg, show_chance=True, save_path=curr_save_path)

    x, per_subject, across = get_choice_probs_around_good_reversals(late, pre=pre, post=post)
    x, per_subject_moving_avg, across_moving_avg = apply_moving_average_to_choice_probs(x, per_subject, moving_avg_window=moving_avg_window, mode="centered")
    curr_save_path = Path(f"../results/figures/{cohort}/reversal-stats/Choice Probabilities Around Good Reversals (Moving Average) (Late)")
    plot_choice_probs_around_good_reversals(x, across_moving_avg, show_chance=True, save_path=curr_save_path)