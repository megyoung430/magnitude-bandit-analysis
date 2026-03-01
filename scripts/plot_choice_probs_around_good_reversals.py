import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
from collections import defaultdict
from src.behavior_import.import_data import *
from src.behavior_import.extract_trials import *
from src.behavior_analysis.get_total_reversals import *
from src.behavior_analysis.get_good_reversal_info import *
from src.behavior_analysis.get_choice_probs_around_good_reversals import *
from src.behavior_analysis.split_good_reversals_by_best_change import *
from src.behavior_analysis.split_early_late_good_reversals import *
from src.behavior_visualization.plot_choice_probs_around_good_reversals import *
from fix_grid_maze_cohort_02_problems import *

task = "grid-maze"
# task = "open-field"

all = True
skip = True
moving_avg = True
remove_bad = True
split_by_best_change = False
split_by_best_change_and_half = False
split_by_first_two = True
split_by_half = True

pre = 10
post = 30
skip_n_trials_after_reversal = 15
moving_avg_window = 4

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
    any_good = any(bool(sess.get("has_good", False)) for sess in subjects_trials.values())
    if problem_number < 7:
        use_total = False
    else:
        use_total = True

    if all:
        subjects_trials = subjects_trials_by_problem[problem_number]

        reversal_windows = get_good_reversal_info(subjects_trials, pre=pre, post=post, include_first_block=False)
        all_good_idx, all_bad_idx, all_blocks = get_all_reversal_indices(subjects_trials)

        x, per_subject, across = get_choice_probs_around_good_reversals(reversal_windows, pre=pre, post=post)
        curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/reversal-stats/Choice Probabilities Around Good Reversals")
        plot_choice_probs_around_good_reversals(x, across, add_cumulative_axis=True, only_good=True, use_total=use_total, windows_for_cumulative_axis=reversal_windows, 
                                                all_good_idx=all_good_idx, all_bad_idx=all_bad_idx, save_path=curr_save_path)

        early, late = split_good_reversals_early_late(reversal_windows, first_n=2)
        x, per_subject, across = get_choice_probs_around_good_reversals(early, pre=pre, post=post)
        curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/reversal-stats/Choice Probabilities Around Good Reversals (Early)")
        plot_choice_probs_around_good_reversals(x, across, add_cumulative_axis=True, only_good=True, use_total=use_total, windows_for_cumulative_axis=early, 
                                                all_good_idx=all_good_idx, all_bad_idx=all_bad_idx, save_path=curr_save_path)

        x, per_subject, across = get_choice_probs_around_good_reversals(late, pre=pre, post=post)
        curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/reversal-stats/Choice Probabilities Around Good Reversals (Late)")
        plot_choice_probs_around_good_reversals(x, across, add_cumulative_axis=True, only_good=True, use_total=use_total,windows_for_cumulative_axis=late, 
                                                all_good_idx=all_good_idx, all_bad_idx=all_bad_idx, save_path=curr_save_path)

    if skip:
        subjects_trials = subjects_trials_by_problem[problem_number]

        reversal_windows = get_good_reversal_info(subjects_trials, pre=pre, post=post, include_first_block=False)

        x, per_subject, across = get_choice_probs_around_good_reversals(reversal_windows, pre=pre, post=post, skip_n_trials_after_reversal=skip_n_trials_after_reversal)
        curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/reversal-stats/Choice Probabilities Around Good Reversals, Skipping Initial Trials")
        plot_choice_probs_around_good_reversals(x, across, add_cumulative_axis=False,
                                                skip_n_trials_after_reversal=skip_n_trials_after_reversal, save_path=curr_save_path)

        early, late = split_good_reversals_early_late(reversal_windows, first_n=2)
        x, per_subject, across = get_choice_probs_around_good_reversals(early, pre=pre, post=post, skip_n_trials_after_reversal=skip_n_trials_after_reversal)
        curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/reversal-stats/Choice Probabilities Around Good Reversals, Skipping Initial Trials (Early)")
        plot_choice_probs_around_good_reversals(x, across, add_cumulative_axis=False,
                                                skip_n_trials_after_reversal=skip_n_trials_after_reversal, save_path=curr_save_path)

        x, per_subject, across = get_choice_probs_around_good_reversals(late, pre=pre, post=post, skip_n_trials_after_reversal=skip_n_trials_after_reversal)
        curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/reversal-stats/Choice Probabilities Around Good Reversals, Skipping Initial Trials (Late)")
        plot_choice_probs_around_good_reversals(x, across, add_cumulative_axis=False, 
                                                skip_n_trials_after_reversal=skip_n_trials_after_reversal, save_path=curr_save_path)

    if moving_avg:
        subjects_trials = subjects_trials_by_problem[problem_number]

        reversal_windows = get_good_reversal_info(subjects_trials, pre=pre, post=post, include_first_block=False)
        all_good_idx, all_bad_idx, all_blocks = get_all_reversal_indices(subjects_trials)

        x, per_subject, across = get_choice_probs_around_good_reversals(reversal_windows, pre=pre, post=post)
        x, per_subject_moving_avg, across_moving_avg = apply_moving_average_to_choice_probs(x, per_subject, moving_avg_window=moving_avg_window, mode="centered")

        curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/reversal-stats/Choice Probabilities Around Good Reversals (Moving Average)")
        plot_choice_probs_around_good_reversals(x, across_moving_avg, add_cumulative_axis=True, only_good=True, use_total=use_total, windows_for_cumulative_axis=reversal_windows, 
                                                all_good_idx=all_good_idx, all_bad_idx=all_bad_idx, save_path=curr_save_path)

        if split_by_half:
            early, late = split_good_reversals_early_late(reversal_windows)
            
            x, per_subject, across = get_choice_probs_around_good_reversals(early, pre=pre, post=post)
            x, per_subject_moving_avg, across_moving_avg = apply_moving_average_to_choice_probs(x, per_subject, moving_avg_window=moving_avg_window, mode="centered")
            curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/reversal-stats/Choice Probabilities Around Good Reversals (Moving Average) (First Half)")
            plot_choice_probs_around_good_reversals(x, across_moving_avg, add_cumulative_axis=True, only_good=True, use_total=use_total, windows_for_cumulative_axis=early, 
                                                    all_good_idx=all_good_idx, all_bad_idx=all_bad_idx, save_path=curr_save_path)
            
            x, per_subject, across = get_choice_probs_around_good_reversals(late, pre=pre, post=post)
            x, per_subject_moving_avg, across_moving_avg = apply_moving_average_to_choice_probs(x, per_subject, moving_avg_window=moving_avg_window, mode="centered")
            curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/reversal-stats/Choice Probabilities Around Good Reversals (Moving Average) (Second Half)")
            plot_choice_probs_around_good_reversals(x, across_moving_avg, add_cumulative_axis=True, only_good=True, use_total=use_total, windows_for_cumulative_axis=late, 
                                                    all_good_idx=all_good_idx, all_bad_idx=all_bad_idx, save_path=curr_save_path)
        if split_by_first_two:
            early, late = split_good_reversals_early_late(reversal_windows, first_n=2)
            
            x, per_subject, across = get_choice_probs_around_good_reversals(early, pre=pre, post=post)
            x, per_subject_moving_avg, across_moving_avg = apply_moving_average_to_choice_probs(x, per_subject, moving_avg_window=moving_avg_window, mode="centered")
            curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/reversal-stats/Choice Probabilities Around Good Reversals (Moving Average) (Early)")
            plot_choice_probs_around_good_reversals(x, across_moving_avg, add_cumulative_axis=True, only_good=True, use_total=use_total, windows_for_cumulative_axis=early, 
                                                    all_good_idx=all_good_idx, all_bad_idx=all_bad_idx, save_path=curr_save_path)

            x, per_subject, across = get_choice_probs_around_good_reversals(late, pre=pre, post=post)
            x, per_subject_moving_avg, across_moving_avg = apply_moving_average_to_choice_probs(x, per_subject, moving_avg_window=moving_avg_window, mode="centered")
            curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/reversal-stats/Choice Probabilities Around Good Reversals (Moving Average) (Late)")
            plot_choice_probs_around_good_reversals(x, across_moving_avg, add_cumulative_axis=True, only_good=True, use_total=use_total, windows_for_cumulative_axis=late, 
                                                    all_good_idx=all_good_idx, all_bad_idx=all_bad_idx, save_path=curr_save_path)

    if remove_bad and not use_total:
        subjects_trials = subjects_trials_by_problem[problem_number]

        reversal_windows = get_good_reversal_info(subjects_trials, pre=pre, post=post, include_first_block=False)
        all_good_idx, all_bad_idx, all_blocks = get_all_reversal_indices(subjects_trials)
        reversal_windows = remove_trials_after_bad_rev(reversal_windows, all_good_idx, all_bad_idx, include_bad_trial=True)

        x, per_subject, across = get_choice_probs_around_good_reversals(reversal_windows, pre=pre, post=post)
        x, per_subject_moving_avg, across_moving_avg = apply_moving_average_to_choice_probs(x, per_subject, moving_avg_window=moving_avg_window, mode="centered")

        curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/reversal-stats/Choice Probabilities Around Good Reversals (Remove Bad)")
        plot_choice_probs_around_good_reversals(x, across_moving_avg, add_cumulative_axis=True, only_good=False, windows_for_cumulative_axis=reversal_windows, 
                                                all_good_idx=all_good_idx, all_bad_idx=all_bad_idx, save_path=curr_save_path)

        early, late = split_good_reversals_early_late(reversal_windows, first_n=2)

        early = remove_trials_after_bad_rev(early, all_good_idx, all_bad_idx, include_bad_trial=True)
        x, per_subject, across = get_choice_probs_around_good_reversals(early, pre=pre, post=post)
        x, per_subject_moving_avg, across_moving_avg = apply_moving_average_to_choice_probs(x, per_subject, moving_avg_window=moving_avg_window, mode="centered")
        curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/reversal-stats/Choice Probabilities Around Good Reversals (Remove Bad) (Early)")
        plot_choice_probs_around_good_reversals(x, across_moving_avg, add_cumulative_axis=True, only_good=False, windows_for_cumulative_axis=early, 
                                                all_good_idx=all_good_idx, all_bad_idx=all_bad_idx, save_path=curr_save_path)

        late = remove_trials_after_bad_rev(late, all_good_idx, all_bad_idx, include_bad_trial=True)
        x, per_subject, across = get_choice_probs_around_good_reversals(late, pre=pre, post=post)
        x, per_subject_moving_avg, across_moving_avg = apply_moving_average_to_choice_probs(x, per_subject, moving_avg_window=moving_avg_window, mode="centered")
        curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/reversal-stats/Choice Probabilities Around Good Reversals (Remove Bad) (Late)")
        plot_choice_probs_around_good_reversals(x, across_moving_avg, add_cumulative_axis=True, only_good=False, windows_for_cumulative_axis=late, 
                                                all_good_idx=all_good_idx, all_bad_idx=all_bad_idx, save_path=curr_save_path)

    if split_by_best_change and moving_avg:
        subjects_trials = subjects_trials_by_problem[problem_number]

        reversal_windows = get_good_reversal_info(subjects_trials, pre=pre, post=post, include_first_block=False)
        all_good_idx, all_bad_idx, all_blocks = get_all_reversal_indices(subjects_trials)
        best2, best3 = split_good_reversals_by_best_change(reversal_windows)

        # plot best->second
        x, per_subject, across = get_choice_probs_around_good_reversals(best2, pre=pre, post=post)
        x, _, across_sm = apply_moving_average_to_choice_probs(x, per_subject, moving_avg_window=moving_avg_window, mode="centered")
        curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/reversal-stats/Choice Probs Good Reversals (Best->Second)")
        plot_choice_probs_around_good_reversals(x, across_sm, add_cumulative_axis=True, only_good=True, use_total=use_total, windows_for_cumulative_axis=best2,
                                                all_good_idx=all_good_idx, all_bad_idx=all_bad_idx, save_path=curr_save_path)

        # plot best->third
        x, per_subject, across = get_choice_probs_around_good_reversals(best3, pre=pre, post=post)
        x, _, across_sm = apply_moving_average_to_choice_probs(x, per_subject, moving_avg_window=moving_avg_window, mode="centered")
        curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/reversal-stats/Choice Probs Good Reversals (Best->Third)")
        plot_choice_probs_around_good_reversals(x, across_sm, add_cumulative_axis=True, only_good=True, use_total=use_total, windows_for_cumulative_axis=best3,
                                                all_good_idx=all_good_idx, all_bad_idx=all_bad_idx, save_path=curr_save_path)
        
        early, late = split_good_reversals_early_late(reversal_windows, first_n=2)
        best2_early, best3_early = split_good_reversals_by_best_change(early)
        best2_late, best3_late = split_good_reversals_by_best_change(late)

        if split_by_best_change_and_half:
            # plot best->second early
            x, per_subject, across = get_choice_probs_around_good_reversals(best2_early, pre=pre, post=post)
            x, _, across_sm = apply_moving_average_to_choice_probs(x, per_subject, moving_avg_window=moving_avg_window, mode="centered")
            curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/reversal-stats/Choice Probs Good Reversals (Best->Second) (Early)")
            plot_choice_probs_around_good_reversals(x, across_sm, add_cumulative_axis=True, only_good=True, use_total=use_total, windows_for_cumulative_axis=best2_early,
                                                    all_good_idx=all_good_idx, all_bad_idx=all_bad_idx, save_path=curr_save_path)

            # plot best->third early
            x, per_subject, across = get_choice_probs_around_good_reversals(best3_early, pre=pre, post=post)
            x, _, across_sm = apply_moving_average_to_choice_probs(x, per_subject, moving_avg_window=moving_avg_window, mode="centered")
            curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/reversal-stats/Choice Probs Good Reversals (Best->Third) (Early)")
            plot_choice_probs_around_good_reversals(x, across_sm, add_cumulative_axis=True, only_good=True, use_total=use_total, windows_for_cumulative_axis=best3_early,
                                                    all_good_idx=all_good_idx, all_bad_idx=all_bad_idx, save_path=curr_save_path)

            # plot best->second late
            x, per_subject, across = get_choice_probs_around_good_reversals(best2_late, pre=pre, post=post)
            x, _, across_sm = apply_moving_average_to_choice_probs(x, per_subject, moving_avg_window=moving_avg_window, mode="centered")
            curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/reversal-stats/Choice Probs Good Reversals (Best->Second) (Late)")
            plot_choice_probs_around_good_reversals(x, across_sm, add_cumulative_axis=True, only_good=True, use_total=use_total, windows_for_cumulative_axis=best2_late,
                                                    all_good_idx=all_good_idx, all_bad_idx=all_bad_idx, save_path=curr_save_path)

            # plot best->third late
            x, per_subject, across = get_choice_probs_around_good_reversals(best3_late, pre=pre, post=post)
            x, _, across_sm = apply_moving_average_to_choice_probs(x, per_subject, moving_avg_window=moving_avg_window, mode="centered")
            curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/reversal-stats/Choice Probs Good Reversals (Best->Third) (Late)")
            plot_choice_probs_around_good_reversals(x, across_sm, add_cumulative_axis=True, only_good=True, use_total=use_total, windows_for_cumulative_axis=best3_late,
                                                    all_good_idx=all_good_idx, all_bad_idx=all_bad_idx, save_path=curr_save_path)

    if split_by_best_change and remove_bad and moving_avg and not use_total:
        subjects_trials = subjects_trials_by_problem[problem_number]

        reversal_windows = get_good_reversal_info(subjects_trials, pre=pre, post=post, include_first_block=False)
        all_good_idx, all_bad_idx, all_blocks = get_all_reversal_indices(subjects_trials)
        
        early, late = split_good_reversals_early_late(reversal_windows, first_n=2)
        best2_early, best3_early = split_good_reversals_by_best_change(early)
        best2_early = remove_trials_after_bad_rev(best2_early, all_good_idx, all_bad_idx, include_bad_trial=True)
        best3_early = remove_trials_after_bad_rev(best3_early, all_good_idx, all_bad_idx, include_bad_trial=True)
        best2_late, best3_late = split_good_reversals_by_best_change(late)
        best2_late = remove_trials_after_bad_rev(best2_late, all_good_idx, all_bad_idx, include_bad_trial=True)
        best3_late = remove_trials_after_bad_rev(best3_late, all_good_idx, all_bad_idx, include_bad_trial=True)

        # plot best->second
        x, per_subject, across = get_choice_probs_around_good_reversals(best2, pre=pre, post=post)
        x, _, across_sm = apply_moving_average_to_choice_probs(x, per_subject, moving_avg_window=moving_avg_window, mode="centered")
        curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/reversal-stats/Choice Probs Good Reversals (Best->Second) (Remove Bad)")    
        plot_choice_probs_around_good_reversals(x, across_sm, add_cumulative_axis=True, only_good=False, windows_for_cumulative_axis=best2,
            all_good_idx=all_good_idx, all_bad_idx=all_bad_idx, save_path=curr_save_path)

        # plot best->third
        x, per_subject, across = get_choice_probs_around_good_reversals(best3, pre=pre, post=post)
        x, _, across_sm = apply_moving_average_to_choice_probs(x, per_subject, moving_avg_window=moving_avg_window, mode="centered")
        curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/reversal-stats/Choice Probs Good Reversals (Best->Third) (Remove Bad)")
        plot_choice_probs_around_good_reversals(x, across_sm, add_cumulative_axis=True, only_good=False, windows_for_cumulative_axis=best3, 
                                                all_good_idx=all_good_idx, all_bad_idx=all_bad_idx, save_path=curr_save_path)
        
        if split_by_best_change_and_half:
            # plot best->second early
            x, per_subject, across = get_choice_probs_around_good_reversals(best2_early, pre=pre, post=post)
            x, _, across_sm = apply_moving_average_to_choice_probs(x, per_subject, moving_avg_window=moving_avg_window, mode="centered")
            curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/reversal-stats/Choice Probs Good Reversals (Best->Second) (Remove Bad) (Early)")
            plot_choice_probs_around_good_reversals(x, across_sm, add_cumulative_axis=True, only_good=False, windows_for_cumulative_axis=best2_early,
                                                    all_good_idx=all_good_idx, all_bad_idx=all_bad_idx, save_path=curr_save_path)

            # plot best->third early
            x, per_subject, across = get_choice_probs_around_good_reversals(best3_early, pre=pre, post=post)
            x, _, across_sm = apply_moving_average_to_choice_probs(x, per_subject, moving_avg_window=moving_avg_window, mode="centered")
            curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/reversal-stats/Choice Probs Good Reversals (Best->Third) (Remove Bad) (Early)")
            plot_choice_probs_around_good_reversals(x, across_sm, add_cumulative_axis=True, only_good=False, windows_for_cumulative_axis=best3_early,
                                                    all_good_idx=all_good_idx, all_bad_idx=all_bad_idx, save_path=curr_save_path)

            # plot best->second late
            x, per_subject, across = get_choice_probs_around_good_reversals(best2_late, pre=pre, post=post)
            x, _, across_sm = apply_moving_average_to_choice_probs(x, per_subject, moving_avg_window=moving_avg_window, mode="centered")
            curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/reversal-stats/Choice Probs Good Reversals (Best->Second) (Remove Bad) (Late)")
            plot_choice_probs_around_good_reversals(x, across_sm, add_cumulative_axis=True, only_good=False, windows_for_cumulative_axis=best2_late,
                                                    all_good_idx=all_good_idx, all_bad_idx=all_bad_idx, save_path=curr_save_path)

            # plot best->third late
            x, per_subject, across = get_choice_probs_around_good_reversals(best3_late, pre=pre, post=post)
            x, _, across_sm = apply_moving_average_to_choice_probs(x, per_subject, moving_avg_window=moving_avg_window, mode="centered")
            curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/reversal-stats/Choice Probs Good Reversals (Best->Third) (Remove Bad) (Late)")
            plot_choice_probs_around_good_reversals(x, across_sm, add_cumulative_axis=True, only_good=False, windows_for_cumulative_axis=best3_late,
                                                    all_good_idx=all_good_idx, all_bad_idx=all_bad_idx, save_path=curr_save_path)