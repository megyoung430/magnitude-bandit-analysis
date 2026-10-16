---
type: feature
complexity: complex
status: in review
---

# Choice probabilities aligned to changes in the true best / second-best value

## Description

Ports the notebook's value-change analysis (`get_value_change_windows`, `_aggregate_windows`,
`plot_value_change_choice_probs`, `assert_truncation_invariant`,
`diagnose_mid_window_value_changes`, `run_all_value_change_plots`, `flatten_subjects_trials`)
into the codebase for the drifting task, with the bugs fixed.

Drifting problems only: the drifting task has block changes, no good or bad reversals. The entry point raises a `ValueError` naming the mouse and session if it is given a counter-format session (good/bad reversal counters).

New files:
- `src/behavior_analysis/value_change_windows.py`: event detection, windows, truncation,
  aggregation, contamination summary (pure).
- `src/behavior_visualization/plot_value_change_choice_probs.py`: `plot_value_change_panel(curves, ax=None, ...)` and the saving wrapper `plot_value_change_choice_probs(curves, save_path, ...)`.
- `src/pipelines/value_change_analysis.py`: the entry point (not yet wired into the runner).

Entry point:

```python
run_value_change_analysis(subjects_trials_by_problem: dict, out_dir: Path, *, pre=5, post=15,
                          truncate_modes=(False, True), show_individual_mice=False) -> dict
```

Writes `out_dir/value_change_<rank>_<direction>_<all_events|rank_preserving|rank_swap>_<untruncated|truncated_at_rank_swap>.png/.pdf`
for the 10 configurations (rank best/second x direction increase/decrease x filter all/preserving/swap,
minus best/increase/swap and second/decrease/swap), `value_change_contamination.csv` (one row per
configuration: contamination, rank-swap counts, event accounting) and `value_change_bin_counts.csv`
(per configuration, mode and time point t: events and mice with a trial there). Configurations
with no events get no figure but are recorded (`plotted: False`). Returns
`{"summaries": {config: {mode: {n_events, n_subjects, n_truncated, counts, x, mean, se, n_events_per_bin, n_subjects_per_bin, plotted, files}}}, "contamination": {...}, "contamination_csv": Path, "bin_counts_csv": Path, "files": [Path]}`.
Works for one problem (`{1: ...}`) and pooled: events are computed per problem and merged per mouse.

## Acceptance criteria

- All 10 configurations, with and without truncation, plus the diagnostic CSV.
- Every bug below fixed, each with a test that fails on the old behaviour.
- Tests: `tests/test_value_change_windows.py`, `tests/test_plot_value_change.py`.

## Decisions

- **Believed-value events removed.** The `use_believed` / `use_believed_rank` paths and
  `_compute_believed_ranks` are dropped. The PI did not ask for believed events, and the
  notebook's believed events were defined from the TRUE best arm (target arm = best by true
  magnitude, compared with the last-seen value), which mixes true and believed definitions.
- `rank_preserving_only` / `rank_swap_only` booleans replaced by `rank_filter in {"all","rank_preserving","rank_swap"}` (mutually exclusive by construction).
- **The third arm stays fixed (PI decision).** The third arm is the one with the lowest
  magnitude (the unrewarded arm); it changes only at block changes, which the reversal analyses
  cover. An event whose third arm at t = 0 differs from t = -1 is excluded and counted
  (`n_third_arm_changed`). Each window is cut so that every trial shares the event's block AND
  third arm: going back from t = 0 (and forward) it extends over the trials that do, and stops at
  the first that does not; trials beyond are `None`. A trial with missing magnitudes or a tied
  lowest magnitude breaks the window. A tied lowest magnitude at t-1 or t makes the third arm
  undetermined: the event is skipped and counted (`n_tie_skipped`). Exactly three arms per trial
  are required (anything else raises).
- **One rank-swap definition.** With the third arm fixed, a swap is the best and second-best arms
  exchanging places (their relative order flips). The filter (`_passes_filter`: the arm that held
  the rank moved in the direction AND the order flipped) and the truncation (first later trial
  where the order of the t = 0 best/second arms is flipped) both use `_exchanged`. "Old best falls
  to last" is a third-arm change, so it is excluded and counted, not left as neither. Preserving
  and swap now partition the determinable events (sample: best/decrease 5 + 2 = 7).
- **Window:** t = 0 is the first trial with the new value; the window is `-pre..+post` inclusive
  (`pre + post + 1` points; `post` counts trials after t = 0). Every event is kept; trials missing
  because of the session edge, another block, or another third arm are `None`. The number of
  events and mice contributing at each t is reported (`n_events_per_bin`, `value_change_bin_counts.csv`)
  so thinning is visible. Windows can span the recordings of a session (the task continues the
  same drifting values across a restart, so a change at a restart is a real event) but never
  sessions. Changes at a block boundary are reversals, not events. Caveat: the importer keeps
  the block number across a restart, so a restart is recognised only by being an ordinary step
  inside a block; a restart that landed exactly on a real block change with the same third arm
  would be counted as an event. The sample never shows this (test:
  `test_a_recording_restart_inside_a_block_keeps_its_value_change_as_an_event`).
- Detection is on the value of the rank position (larger / middle magnitude), so it is tie-free.
- **Ties:** arm names never break ties. Events whose verdict depends on a tie are skipped and
  counted (`n_tie_skipped`; `"all"` only needs the third arm determinable). On a window trial where
  best and second tie, a choice of either tied arm counts half for "best" and half for "second".
  The sample has no tied trials (0 of 93).
- **Rank labels come from the magnitudes** of each trial (with the even tie share above), not from
  the raw `chosen_rank` of the trial variables, which is not filled from `rank_counts` the way the
  session-level field is. The analysis therefore does not need `chosen_rank`.
- **Truncation renamed** `truncate_at_value_change` -> `truncate_at_rank_swap` and redefined (see
  above), applied within the third-arm-bounded window. The notebook only truncated when a later
  swap reverted to the PRE-event ordering, which matched neither its docstring nor its name. A
  swap event is not truncated by its own exchange (reference = t = 0). Which events exist never
  depends on truncation (tested over random data). An event with tied best/second at t = 0 has no
  reference ordering: left untruncated and flagged (`n_reference_tied`).
- **Contamination diagnostic** is computed from each event's own session: contaminated = another
  magnitude change later in the event's window (same block and third arm) within `post`; rank swap =
  the truncation rule fires. The notebook only classified the FIRST later change.
- Pooling across problems: events per problem, merged per mouse (`merge_value_change_windows`); no session-key prefixing.
- Sessions with no trial variables are counted (`n_sessions_without_trials`); trials with missing magnitudes are counted (`n_missing_magnitude_trials`).
- The CSV takes `n_events` from the event accounting and raises if the contamination summary disagrees.

## Bugs found in the notebook

1. `trial_variables[0]` only -> all recordings merged with `merge_trial_variables` (`test_all_recordings_of_a_session_are_used`).
2. Sessions in dict order -> `ordered_session_keys` (`test_sessions_are_visited_in_session_order_not_dict_order`); windows stay within a session (`test_windows_do_not_cross_sessions`).
3. Label set inferred from the first non-None label (if "third", prev_best/prev_second became 0) -> explicit from the filter; mismatching labels raise (`test_aggregation_uses_the_filter_label_set_even_when_the_first_label_is_third`, `test_aggregation_rejects_labels_outside_the_label_set`, `test_swap_labels_name_the_arms_by_their_rank_before_the_event`).
4. Diagnostic indexed a concatenation of all sessions with session-local `trial_idx` -> contamination computed per session when events are built (`test_contamination_is_checked_at_session_local_trials_in_every_session`, which includes the old algorithm and shows it finds 0 where there is 1).
5. Truncation semantics mismatch -> see Decisions (`test_truncation_*`, `test_a_swap_event_is_not_truncated_by_its_own_exchange`, `test_the_swap_filter_and_the_truncation_use_the_same_exchange`). `assert_truncation_invariant` is now `test_truncation_never_changes_which_events_are_found`.
6. Window bounds: events near the end silently dropped -> kept with `None`, per-bin n reported (`test_events_near_the_session_edges_are_kept_with_missing_trials_as_none`, `test_every_detected_change_is_accounted_for`, `test_entry_writes_per_bin_counts_that_show_the_thinning`).
7. Ties broken by arm name -> skip and count (`test_ties_are_counted_and_skipped_when_the_filter_depends_on_them`, `test_a_tied_lowest_magnitude_leaves_the_third_arm_undetermined`, `test_tied_best_and_second_trials_split_the_choice_evenly`).
8. `_same_block` true when blocks missing -> missing or misaligned `blocks` raise (`test_missing_or_misaligned_blocks_raise`).
9. SE: `nanstd/sqrt(n)` with n counting NaN subjects, 0 for n == 1 -> `stats.nan_standard_error`; plots omit the band (`test_se_counts_only_subjects_with_data_and_is_nan_for_one`, `test_single_subject_draws_the_mean_without_an_error_band`).
10. Title: `events_str` overwritten so the truncated count never showed -> shown (`test_title_shows_the_truncated_event_count`). `num_truncated` counted any `None` in the post window, including block cuts; now counts only rank-swap truncation.
11. `flatten_subjects_trials` prefixed session keys (breaks session-key parsing) -> removed (`test_problems_are_pooled_per_mouse_without_session_key_collisions`).
12. Others: `n = min(lengths)` silently hid unequal list lengths (now raises); "Mean of empty slice" warnings (`test_time_points_without_data_are_nan_and_do_not_warn`); rcParams set in the module; no `ax` argument.
13. The notebook's swap and truncation rules differed (exact positions vs relative order); now one definition. The notebook's swap filter also left "old best falls to last" in neither preserving nor swap.

## Results change

New outputs only (no existing figure changes): 20 figures (10 configurations x 2 modes), `value_change_contamination.csv` and `value_change_bin_counts.csv` per call of `run_value_change_analysis`. Not wired into `cohort_analysis.py` yet.

Sample session (one mouse, 93 trials), events per configuration before -> after the third-arm and edge-handling changes: best/increase all 4 -> 5, preserving 2 -> 3; best/decrease all 5 -> 7, preserving 3 -> 5, swap 2 -> 2; second/increase all 3 -> 4, preserving 2 -> 3, swap 1 -> 1; second/decrease all 5 -> 6, preserving 2 -> 3. The extra events are those that were dropped for lack of room near the session end (now kept). On the sample the third arm always has magnitude 0, never ties, and changes only at the block changes (trials 22, 53, 86); no event is excluded for a third-arm change or a tie.
