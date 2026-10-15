---
type: feature
complexity: complex
status: in review
---

# Reversal-aligned arm choice probability with reward magnitudes (prev and new labels)

## Description

Choice probability of the three arms aligned to a block change, with the mean reward magnitude of the same three arms on a secondary axis. Two labelling schemes, each its own figure set:

- `prev`: arms labelled by their rank before the reversal (best blue `#5DA5DA`, second green `#60BD68`, third grey `#7f7f7f`).
- `new`: arms labelled by their rank after the reversal (best blue, second green, third black).

Each scheme for reversal types `all`, `best_to_third`, `second_to_third`. The type is defined by the prev ranking for both schemes (stated in titles and docs). Windows are computed per problem (`include_first_block=False`), then concatenated per subject for the pooled figures.

## Entry point

```python
run_reversal_aligned_arm_analysis(
    windows_or_trials_by_problem, out_dir, *, pre=10, post=30, moving_avg_window=4,
    label_schemes=("prev", "new"), reversal_types=("all", "best_to_third", "second_to_third"),
    moving_avg_mode="centered", skip_n_trials_after_reversal=0,
) -> {"results": {scope: {scheme: {type: ReversalAlignedArms}}}, "counts": DataFrame,
      "figure_paths": [Path], "windows_by_problem": {...}}
```

`src/pipelines/reversal_aligned_arms.py`. Input is `{problem: {subject: {session: dict}}}` (one problem = one entry) or `{problem: {subject: [reversal]}}` built with `build_windows_with_magnitudes`. Scopes: `pooled`, plus one per problem when there are several. Files: `reversal_aligned_arms_<scope>_<scheme>_<type>.png/.pdf` and `reversal_aligned_arms_counts.csv`. `out_dir` None or `""` shows the figures. A combination with no reversal left makes no figure and warns (`EmptySelectionWarning`). Not wired into the runner.

Modules: `src/behavior_analysis/reversal_arm_ranks.py` (rankings, labels, types, selection), `src/behavior_analysis/reversal_aligned_arms.py` (windows, alignment, curves), `src/behavior_visualization/plot_reversal_aligned_arms.py` (panel with `ax=None`, saving wrapper).

## Acceptance criteria

- Both schemes and all three types produce figures per scope; magnitudes are those of the same labelled arms.
- Each notebook bug below is fixed with a test that fails on the old behaviour (checked by mutating the code back).
- Works for one problem and pooled.

## Bugs found in the notebook

1. Short pre windows were stacked left-aligned (`pre_seq + post_seq` padded at the end), so every point sat at the wrong offset. Fixed: pre right-aligned to offset -1, post from 0 (`align_window_values`). Tests: `test_a_short_pre_window_is_right_aligned_to_the_reversal_trial`, `test_curves_of_a_truncated_pre_window_sit_at_the_right_offsets`, and the magnitude twin. The repo's `get_choice_probs_around_good_reversals` (`_reversal_choice_rows`) has the same left alignment; it is left unchanged here (outside this item), PI to decide.
2. `keep_zero_then_skip` dropped k-1 trials. Fixed: the repo's `_keep_zero_then_skip` (keeps trial 0, drops 1..k); `trial_offsets` give true offsets. Tests: `test_skip_k_keeps_post_trial_zero...`, `test_curves_with_a_skip...`.
3. SE used `nanstd/sqrt(n_rows)` and gave 0 for one unit. Fixed: `stats.nan_standard_error`. Tests: `test_se_is_over_the_subjects_with_data_in_the_bin`, `test_se_is_nan_not_zero_for_one_subject`. Plots omit the band where SE is NaN.
4. Ties were broken by arm name. Fixed: a tie leaves labels undefined (reversal left out of that scheme, in every type) and a type that changes with the order of tied arms is `ambiguous` (left out of type-specific sets). Counts reported per scope/scheme/type. Tests in `test_reversal_arm_ranks.py`, `test_tied_reversals_are_left_out_and_counted`.
5. `assert` for the sum-to-1 data check. Fixed: `ValueError` (`test_choice_probabilities_that_do_not_sum_to_one_raise_value_error`).
6. Magnitude traces were built by re-merging sessions with no check that they matched the windows, and padded left-aligned. Fixed: `add_magnitude_windows` slices the series from `get_vars_across_all_sessions` (same session order and merge as the windows) over `trial_window_idx`, and raises if the magnitudes at the reversal and the trial before differ from those stored in the reversal. Tests: `test_magnitude_windows_*`.
7. Magnitude SE used `n` rows. Fixed: per-subject mean, then `nan_standard_error` across subjects. `test_magnitude_se_is_across_subjects_of_their_per_subject_means`.
8. Moving-average mode: per-problem used "centered", pooled "trailing". Fixed: one explicit `moving_avg_mode` (default centered, validated) for all figures, shown in titles. `test_moving_average_mode_is_centered_by_default_and_selectable`.
9. Also: a window longer than `pre`/`post` was silently truncated, now `ValueError`; a tower count other than three or a missing magnitude raises instead of returning "other".

## Decisions to check

- A tied label ranking drops the reversal from every type of that scheme, including `all`, so `n` of `all` can differ between `prev` and `new`. (`all` would otherwise label tied arms arbitrarily.)
- Magnitude curves are not smoothed; SE is across subjects, not reversals.
- Magnitude trace colours equal the choice colours (dashed steps).
- Reuses private helpers `_keep_zero_then_skip`, `_trial_offsets`, `_smooth_pre_post` from `get_choice_probs_around_good_reversals` (its public smoother is hard-wired to prev/next/third roles and asserts).

## Results change

New outputs only; no existing output changes. Sample session (one subject, blocks only, reversals at trials 22/53/86, pre=10, post=30, window 4 centered): all three are best_to_third (3 of 3, none tied); second_to_third is empty (warning, no figure). Prev scheme at offsets -1 / 0 / +10: prev_best choice 0.78 / 0.50 / 0.38, prev_third 0.11 / 0.50 / 0.38; magnitude prev_best 3.0 / 0.0 / 0.0, prev_third 0.0 / 3.0 / 2.5. New scheme: new_best equals the prev_third curves and new_third the prev_best curves (best_to_third swaps them); new_second = prev_second (choice 0.11 / 0.0 / 0.25, magnitude 2.0 / 2.0 / 1.5). SE is NaN (one subject), so no bands.
