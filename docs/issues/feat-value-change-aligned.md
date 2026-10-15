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
minus best/increase/swap and second/decrease/swap) and `value_change_contamination.csv`
(one row per configuration: contamination, rank-swap counts, event accounting). Configurations
with no events get no figure but are recorded (`plotted: False`). Returns
`{"summaries": {config: {mode: {n_events, n_subjects, n_truncated, counts, x, mean, se, plotted, files}}}, "contamination": {...}, "contamination_csv": Path, "files": [Path]}`.
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
- Window: t = 0 is the first trial with the new value; window is `-pre..+post` inclusive
  (`pre + post + 1` points; `post` counts trials after t = 0). Only complete windows kept
  (comparable averages); dropped events are counted (`n_dropped_no_pre_room`, `n_dropped_no_post_room`) in the returned summaries and the CSV. Windows never cross sessions, and trials outside the event's block are `None`. Changes at a block boundary are reversals, not events.
- Detection is on the value of the rank position (largest / middle magnitude), so it is tie-free. Rank swap = the arm that held the rank moved in the direction AND the best and second-best arms exchanged places (exact exchange; the notebook accepted any arrangement with the old best below the top).
- **Ties:** arm names never break ties. An event whose preserving/swap verdict depends on a tie is skipped and counted (`n_tie_skipped`). `"all"` is unaffected. Events whose t = 0 best/second-best are tied have no truncation reference: left untruncated, flagged, counted (`n_reference_tied` in the CSV). Values here are small integers, so ties do occur.
- **Truncation renamed** `truncate_at_value_change` -> `truncate_at_rank_swap` and redefined: the post window is blanked from the first later trial where the arms that were best/second at t = 0 have strictly exchanged places. The notebook only truncated when a later swap reverted to the PRE-event ordering, which matched neither its docstring (any mid-window swap) nor its name (any value change). Truncation at any value change was not chosen: it would cut about a quarter of all windows at changes of the third arm that do not affect the best/second contrast. Which events exist never depends on truncation (tested over random data).
- **Contamination diagnostic** is computed from each event's own session: contaminated = another magnitude change later in the event's block within the post window; rank swap = the truncation rule fires. The notebook only classified the FIRST later change, so a swap after a non-swap change was missed.
- Pooling across problems: events per problem, merged per mouse (`merge_value_change_windows`); no session-key prefixing.
- Sessions with no trial variables are counted (`n_sessions_without_trials`); trials with missing magnitudes are counted (`n_missing_magnitude_trials`) and changes across them are not detected.

## Bugs found in the notebook

1. `trial_variables[0]` only -> all recordings merged with `merge_trial_variables` (`test_all_recordings_of_a_session_are_used`).
2. Sessions in dict order -> `ordered_session_keys` (`test_sessions_are_visited_in_session_order_not_dict_order`); windows stay within a session (`test_windows_do_not_cross_sessions`).
3. Label set inferred from the first non-None label (if "third", prev_best/prev_second became 0) -> explicit from the filter; mismatching labels raise (`test_aggregation_uses_the_filter_label_set_even_when_the_first_label_is_third`, `test_aggregation_rejects_labels_outside_the_label_set`).
4. Diagnostic indexed a concatenation of all sessions with session-local `trial_idx` -> contamination computed per session when events are built (`test_contamination_is_checked_at_session_local_trials_in_every_session`, which includes the old algorithm and shows it finds 0 where there is 1).
5. Truncation semantics mismatch -> see Decisions (`test_truncation_*`). `assert_truncation_invariant` is now `test_truncation_never_changes_which_events_are_found`.
6. Window bounds: events near the end silently dropped -> kept as complete-window rule, documented, counted (`test_window_bounds_keep_full_windows_and_count_the_dropped_ones`, `test_every_detected_change_is_accounted_for`).
7. Ties broken by arm name -> skip and count (`test_ties_are_counted_and_skipped_when_the_filter_depends_on_them`, `test_a_tie_between_the_other_arms_does_not_block_the_classification`).
8. `_same_block` true when blocks missing -> missing or misaligned `blocks` raise (`test_missing_or_misaligned_blocks_raise`).
9. SE: `nanstd/sqrt(n)` with n counting NaN subjects, 0 for n == 1 -> `stats.nan_standard_error` (non-NaN count, NaN below two); plots omit the band (`test_se_counts_only_subjects_with_data_and_is_nan_for_one`, `test_single_subject_draws_the_mean_without_an_error_band`).
10. Title: `events_str` overwritten so the truncated count never showed -> shown (`test_title_shows_the_truncated_event_count`). Also `num_truncated` counted any `None` in the post window, including block cuts; now counts only events truncated at a rank swap.
11. `flatten_subjects_trials` prefixed session keys (breaks session-key parsing) -> removed (`test_problems_are_pooled_per_mouse_without_session_key_collisions`).
12. Others: `n = min(lengths)` silently hid unequal list lengths (now raises); `np.nanmean` "Mean of empty slice" warnings (counts-based mean, `test_time_points_without_data_are_nan_and_do_not_warn`); rcParams set in the module (now only `plot_style`); `plt.show()` without save and no `ax` (now `ax=None`/`axes_or_new` and `save_or_show`).

## Results change

New outputs only (no existing figure changes): 20 figures (10 configurations x 2 modes) plus `value_change_contamination.csv` per call of `run_value_change_analysis`. Not wired into `cohort_analysis.py` yet. Numbers differ from the notebook's where the old bugs bit: later recordings now included, later sessions' contamination now correct, SE no longer 0/overcounted, swap curves use the correct label set, truncation rule differs, tie events are skipped for preserving/swap. On the sample session, event counts per configuration equal the notebook function's counts (run on the merged trial variables).
