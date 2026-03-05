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
from src.behavior_analysis.get_first_leave_after_good_reversals import *
from src.behavior_analysis.split_early_late_good_reversals import *
from src.behavior_analysis.get_diagnostic_p_value import *
from src.behavior_visualization.plot_first_leave_after_good_reversals import *
from src.behavior_visualization.plot_num_reversals import *
from src.behavior_analysis.get_rank_counts_by_good_reversal import *
from src.behavior_visualization.plot_rank_proportions import *
from src.behavior_visualization.plot_single_session import *
from fix_grid_maze_cohort_02_problems import *

task = "grid-maze"
cohort = "cohort-02"
folder_name = "3x3_maze_blocked_reward_bandit"

# Choice Probability Analysis

run_all = True
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

print(f"Running analysis for {task} task, {cohort}")
print("Running choice probability analysis around good reversals, with the following settings:")
print(f"  All: {all}")
print(f"  Skip: {skip}")
print(f"  Moving Average: {moving_avg}")
print(f"  Remove Bad: {remove_bad}")
print(f"  Split by Best Change: {split_by_best_change}")
print(f"  Split by Best Change and Half: {split_by_best_change_and_half}")
print(f"  Split by First Two: {split_by_first_two}")
print(f"  Split by Half: {split_by_half}")
print(f"  Pre: {pre}")
print(f"  Post: {post}")
print(f"  Skip N Trials After Reversal: {skip_n_trials_after_reversal}")
print(f"  Moving Avg Window: {moving_avg_window}")

root = f"/Volumes/behrens/meg/{folder_name}/{cohort}/rawdata/"
subjects_data = import_data(root)
subjects_trials_by_problem = extract_trials_grouped_by_problem(subjects_data)
subjects_trials_by_problem = fix_grid_maze_cohort_02_problems(subjects_trials_by_problem)

for problem_number in subjects_trials_by_problem.keys():
    print(problem_number)
    problem = f"problem-{problem_number:02d}"

    subjects_trials = subjects_trials_by_problem[problem_number]
    any_good = any(bool(sess.get("has_good", False)) for sess in subjects_trials.values())
    if (task == "grid-maze" and problem_number < 7) or (task == "open-field" and problem_number < 5):
        use_total = False
    else:
        print("Using total reversals for this problem since all reversals are good")
        use_total = True

    if run_all:
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

# First Leave Analysis

print("Running first leave analysis")

for problem_number in subjects_trials_by_problem.keys():
    print(problem_number)
    problem = f"problem-{problem_number:02d}"

    subjects_trials = subjects_trials_by_problem[problem_number]
    reversal_windows = get_good_reversal_info(subjects_trials, include_first_block=False)
    
    first_leave_per_subject = get_first_leave_after_good_reversals(reversal_windows)
    mean, se, n_subjects = average_first_leave_across_subjects(first_leave_per_subject)
    p_value = pvalue_paired_t_new_vs_third(first_leave_per_subject, alternative="greater")
    curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/reversal-stats/First Leave After Good Reversals")
    plot_first_leave_after_good_reversals(mean, se, first_leave_per_subject, p_value, save_path=curr_save_path)

    early, late = split_good_reversals_early_late(reversal_windows, first_n=2)
    first_leave_per_subject_early = get_first_leave_after_good_reversals(early)
    mean_early, se_early, n_subjects_early = average_first_leave_across_subjects(first_leave_per_subject_early)
    p_value_early = pvalue_paired_t_new_vs_third(first_leave_per_subject_early, alternative="greater")
    curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/reversal-stats/First Leave After Good Reversals (Early)")
    plot_first_leave_after_good_reversals(mean_early, se_early, first_leave_per_subject_early, p_value_early, save_path=curr_save_path)

    first_leave_per_subject_late = get_first_leave_after_good_reversals(late)
    mean_late, se_late, n_subjects_late = average_first_leave_across_subjects(first_leave_per_subject_late)
    p_value_late = pvalue_paired_t_new_vs_third(first_leave_per_subject_late, alternative="greater")
    curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/reversal-stats/First Leave After Good Reversals (Late)")
    plot_first_leave_after_good_reversals(mean_late, se_late, first_leave_per_subject_late, p_value_late, save_path=curr_save_path)

# Number of Reversals Analysis

print("Running number of reversals analysis")

for problem_number in subjects_trials_by_problem.keys():
    print(problem_number)
    problem = f"problem-{problem_number:02d}"

    subjects_trials = subjects_trials_by_problem[problem_number]

    curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/reversal-stats/Total Reversals")
    plot_num_reversals(subjects_trials, save_path=curr_save_path)

    curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/reversal-stats/Cumulative Reversals Over Time")
    plot_num_reversals_over_time(subjects_trials, threshold=10, save_path=curr_save_path)

    curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/reversal-stats/Moving Average Reversals Over Time")
    plot_moving_avg_reversals_over_time(subjects_trials, save_path=curr_save_path)

# Rank Proportion Analysis

print("Running rank proportion analysis")

for problem_number in subjects_trials_by_problem.keys():
    print(problem_number)
    problem = f"problem-{problem_number:02d}"
    
    subjects_trials = subjects_trials_by_problem[problem_number]

    reversal_windows = get_good_reversal_info(subjects_trials, include_first_block=True)
    rank_counts_by_good_reversal = get_rank_counts_by_good_reversal(reversal_windows)
    p_values = pvalue_paired_t_best_vs_second_vs_third(rank_counts_by_good_reversal)

    save_path = f"../results/figures/{task}/{cohort}/{problem}/choice-stats/Rank Proportions"
    plot_rank_proportions(rank_counts_by_good_reversal, average_across_mice_pvalues=p_values, save_path=save_path)

# Single Session Analysis

print("Running single session analysis")

for problem_number in subjects_trials_by_problem.keys():
    print(problem_number)
    problem = f"problem-{problem_number:02d}"

    subjects_trials = subjects_trials_by_problem[problem_number]

    all_subjects = list(subjects_trials.keys())
    for current_subject in all_subjects:
        subject_sessions = subjects_trials[current_subject]
        all_sessions = list(subject_sessions.keys())
        for current_session in all_sessions:
            if not subject_sessions[current_session]["trial_info"]:
                print(f"Skipping {current_subject} | {current_session}. There are no trials.")
                continue
            print(f"Plotting {current_subject} | {current_session}")
            curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/single-sessions/{current_subject}/{current_session}/Session Plot")
            plot_single_session(subject_sessions[current_session], title=f"{current_subject} | {current_session}", save_path=curr_save_path)