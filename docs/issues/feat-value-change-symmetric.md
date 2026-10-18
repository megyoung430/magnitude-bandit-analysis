# Symmetric, complete-window value-change figures

## Goal

The regular value-change windows keep every event and cut a window where it meets a block or
third-arm change, a session edge or a best-arm change, so the curve at large |t| averages
fewer and different events than at t = 0. Meg asked for a version with a symmetric window
(for example 6 trials before and after) that only includes events with no block change in any
of those trials, with "symmetric" in the title for the mouse-weighted and event-weighted plots.

## Change

- `select_symmetric_windows` (`src/behavior_analysis/value_change_symmetric.py`) keeps the
  events whose `-N .. +N` window is complete: every trial exists in the event's block with its
  third arm (no block or third-arm change, tied or missing magnitudes, session edge), the best
  arm does not change in the pre window (such an event is excluded, not truncated), and every
  trial has a choice. A best-arm change after the event is allowed. It returns the kept events
  and counts of exclusions by reason (`n_excluded_*`: an event with several causes is counted
  once, in the order block/third arm, session edge, best-arm change, other; "other" is a trial
  without a choice) and the events having each cause (`n_with_*`, overlapping).
  In symmetric windows `counts["n_events"]` is the number kept and
  `counts["n_events_before_exclusion"]` the number found.
- Events record every reason their full window is incomplete (`incomplete_causes`), found by
  scanning the whole window, so a cause behind a nearer one on the same side is still seen.
  A block or best arm starting exactly at t = -N keeps the event; a best/second tie in the pre
  window counts as a best-arm change (only a tie at t = -1 continues a run of ties).
- `run_value_change_analysis(..., symmetric_window=6)` also makes, for every configuration, the
  symmetric figures in `mouse-weighted/` and `event-weighted/`, named and titled
  "Symmetric ±N Trials", with "complete windows only" and the number of events after
  exclusion in the title. There is no rank-swap truncation in the symmetric version.
  `None` switches it off; anything but an integer >= 1 raises.
- Tables: the symmetric mode (`symmetric_N`) is in the bin-count table, and the new
  `symmetric_exclusions_table` (events before, kept, excluded by reason per configuration) is
  saved as `value_change_symmetric_exclusions.csv` next to the other diagnostics (cohort
  tables folder, and the all-cohorts tables folder).

## Results change

Existing figures and tables are unchanged. New: ten symmetric figures per weighting for each
problem, pooled drifting problems and the all-cohorts runner; one new table,
`value_change_symmetric_exclusions.csv`; extra `symmetric_6` rows in
`value_change_bin_counts.csv`. On cohort 3, mouse MY_175_L (pooled drifting problems), 28 to
111 of 70 to 222 events per configuration survive.

## Merging with PR #75 (incremental cohort runs)

On `origin/feat-incremental-cohort-runs` the table merge (`incremental.merge_diagnostic_tables`)
loops over whatever `stack_diagnostics` returns, which is `DIAGNOSTIC_TABLES`, so
`value_change_symmetric_exclusions` is picked up without a code change. Hard-coded lists to
update by whoever merges second: `TABLE_NAMES` in `src/pipelines/all_cohorts_drifting.py` (same on
both branches), the table lists in `tests/test_incremental_run.py`,
`tests/test_cohort_review_tables.py`, `tests/test_all_cohorts_drifting.py`,
`tests/test_review_diagnostic_tables.py`, and the README sentences that name the diagnostic
tables (README lines 207, 319 and 341 of that branch).
