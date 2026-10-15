---
type: bug
complexity: complex
status: in review
---

# Fix rank-proportion blocks, NaN-aware SEs and pooled SE

## Description

1. `rank_proportions.py` dropped each subject's last row as a "summary" entry
   (`block_rows_for_subject`, `rows_in_block`, `proportions_by_block`,
   `n_blocks_excluding_last`), but `get_rank_counts_by_good_reversal` appends no summary
   row, so a real block was lost in the by-mouse, by-block and average-across-blocks views.
   `proportions_by_block` also raised `ValueError` from `max()` when every list was empty.
2. Standard errors divided by the number of units including NaN ones
   (`standard_error`, `rank_mean_and_se`), and the problem-summary SEs returned 0.0
   for a single value. A standard error from fewer than two values is NaN, never 0.
3. With `pool_trials_within_subject=True` the mouse panel drew the pooled (trial-weighted)
   mean with the SE across unpooled block proportions: different units.
4. Plots hid the mean when the SE was NaN.

## Decisions

- Every row is a real block. `block_rows_for_subject` is removed (it became the identity;
  the only caller was `plot_rank_proportions_by_mouse`), and `n_blocks_excluding_last`
  is renamed `n_blocks`. No notebook or script uses either, so no deprecated alias is kept.
- `nan_standard_error` (per-column SE over non-NaN rows, NaN below two) and `mean_se_n` live in
  `src/behavior_analysis/stats.py`; every module imports them from there. The function body is
  identical to the helper on
  `fix-choice-prob-windows`; the two are meant to be merged into one.
- Pooled mean and SE: the pooled proportion averages trials, so its SE is across the same
  trials (`rank_proportions.pooled_standard_errors`, `sqrt(p(1-p)/(n-1))`). Trials are treated
  as independent although they come in blocks, so this is a lower bound if blocks differ.
  The across-mice view (`plot_average_across_mice`) already used the same units for mean and SE
  (each mouse's pooled proportion, SE across mice) and is unchanged.
- Problem-summary path: `compute_problem_length_summary`, `compute_problem_blocklen_summary`
  (`mean_se_n`) and `compute_problem_goodrev_summary` now share `mean_se_n`: NaN values are
  left out of the mean, SE and n, and one value gives a NaN SE instead of 0.0.
- Plotting: `plot_problem_summary._annotation_ceiling` treated a NaN SE as a NaN bar top
  (hiding the label height and ylim); it now adds no error for NaN. In `plot_rank_proportions`
  matplotlib already draws the bar without an error bar for a NaN SE.

## Acceptance criteria

- A subject's last row is a block in every by-block and by-mouse view; the by-block figure has
  one panel per block of the longest subject; lines and bars of a block panel cover the same mice.
- `proportions_by_block({"a": []})` returns `[]`.
- SEs count only non-NaN units; fewer than two gives NaN, never 0.
- The pooled mouse panel's mean and SE are both across trials.
- A NaN SE leaves the mean drawn (and labelled) with only the error bar omitted.
- Outside these, results are unchanged (checked on synthetic no-NaN inputs, main vs branch).

## Results change

Sample data (`cohort-01`): one mouse (MY_72_L), one problem, 4 blocks. Main treated the 4th
block as a summary row. The sample has a single mouse, so the across-mice SE is NaN on both
sides; "pseudo-mice" are the same blocks dealt round-robin to 3 mice (rows 0,3 / 1 / 2), which
exercises the across-mouse paths. Each value is (best, second, third).

| Quantity | main | fixed |
|---|---|---|
| Blocks per mouse used by by-mouse / by-block / average-across-blocks | 3 of 4 | 4 of 4 |
| By-mouse bars, MY_72_L (unpooled) | 0.652, 0.239, 0.109 | 0.560, 0.322, 0.118 |
| By-mouse SE across blocks, MY_72_L | 0.067, 0.087, 0.034 | 0.103, 0.104, 0.025 |
| By-mouse panel lines | 3 | 4 |
| By-mouse bars, MY_72_L (pooled trials) | 0.624, 0.258, 0.118 (unchanged) | 0.624, 0.258, 0.118 |
| By-mouse SE, MY_72_L (pooled trials) | 0.103, 0.104, 0.025 (across unpooled blocks) | 0.051, 0.046, 0.034 (across trials) |
| By-block figure panels | 3 (Block 4 missing) | 4 (Block 4: 0.286, 0.571, 0.143) |
| Average across blocks (bars) | 0.652, 0.239, 0.109 | 0.560, 0.322, 0.118 |
| Average across blocks (SE) | 0.067, 0.087, 0.034 | 0.103, 0.104, 0.025 |
| Average across mice (bars, SE) | 0.560, 0.322, 0.118; NaN | unchanged (mean drawn, no error bar) |
| Pseudo-mice by-mouse: pseudo0 | 1 block (0.636, 0.318, 0.045), pseudo1/2 not drawn | 2 blocks (0.461, 0.445, 0.094); pseudo1, pseudo2 drawn |
| Pseudo-mice by-block panels | 1 (Block 1 without the 3 last rows) | 2 (Block 1: 0.652, 0.239, 0.109; SE 0.067, 0.087, 0.034) |
| Pseudo-mice average across blocks | 0.636, 0.318, 0.045; SE NaN | 0.469, 0.405, 0.126; SE 0.183, 0.166, 0.017 |
| Pseudo-mice average across mice | 0.594, 0.281, 0.126; SE 0.094, 0.113, 0.020 | unchanged |
| Pseudo-mice pooled by-mouse SE, pseudo0 | 0.175, 0.127, 0.049 | 0.094, 0.092, 0.048 |
| Problem-summary SE with one mouse (length, block length, good reversals) | 0.0 | NaN |

Problem-summary and cohort-comparison code paths on the sample data: the sample has no reversal
boundaries (`get_block_lengths` raises), so the block-length rows above are from a synthetic
one-mouse input.

Checked unchanged on synthetic no-NaN inputs with at least two units (main vs branch, identical
values): `proportions_by_mouse` (pooled and unpooled), `mean_by_rank`, `se_by_rank`,
`standard_error`, `mean_rank_proportions_by_subject`, `rank_mean_and_se`,
`compute_problem_{length,goodrev,blocklen}_summary`.

## NaN everywhere: other SE computations

Rule: an SE over fewer than two non-NaN units is NaN, never 0, and the count excludes NaN.
`mean_se_n` (in `stats.py`, built on `nan_standard_error`) is shared.

Searched all of `src/` for `std`/`nanstd` over `sqrt(len)`, `.sem`, and 0 for a single unit.

| Location | Finding | Action |
|---|---|---|
| `cohort_comparison_stats.mean_and_sem` | SEM 0 for one mouse (pinned) | fixed: `nan_standard_error`; test updated |
| `get_task_statistics._standard_error` (block-length plot) | 0.0 below two values | fixed: `mean_se_n`; `plot_task_statistics._tallest_bar` no longer drops a bar whose SE is NaN |
| `problem_utils.summarize_by_problem`, `summarize_by_problem_safe` | 0.0 for one value, NaN counted | fixed: `mean_se_n` |
| `get_first_leave_after_good_reversals.average_first_leave_across_subjects` | 0.0 for one subject | fixed |
| `long_pokes_by_session._mean_and_se` | 0.0 for one value | fixed; bar test updated (NaN SE: bar drawn, no error bar) |
| `get_choice_probs_around_good_reversals._mean_and_se`, `average_choice_probs._mean_and_se_over_problems` | NaN counted in n; 0 for one problem | not touched: `fix-choice-prob-windows` rewrites both |
| `trial_count_summaries` (pandas `sem`) | already NaN below two, NaN skipped | correct |
| `plot_problem_summary` descriptive `sd` text | SD of a sample, not an SE | left |
| `plot_cohort_comparison` / `plot_trial_block_summary` `fillna(0)` on pandas SEM | zero-length error bar with a cap, which reads as SE 0 | fixed: NaN passed through, error bar omitted |

Plots: cohort comparison (`_label_height` already ignores a NaN SEM), long-poke bars and first-leave bars
draw the bar/marker with matplotlib omitting a NaN error bar; the task-statistics block-length plot
needed the `_tallest_bar` fix above so labels and y-limits still use the median.

Effect (sample data is one mouse, so every across-mouse SE of one unit changes from 0 to NaN):

| Quantity | main | fixed |
|---|---|---|
| `mean_and_sem` of one mouse | 0, 0, 0 | NaN, NaN, NaN |
| Block-length SE of a block seen in one mouse | 0.0 | NaN |
| Problem SE (`summarize_by_problem*`), first-leave SE, long-poke SE with one unit | 0.0 | NaN |

Main vs branch on synthetic no-NaN inputs with at least two units: `mean_and_sem`, `_standard_error`,
`average_first_leave_across_subjects`, long-poke `_mean_and_se`, `summarize_by_problem*` are identical.

## Review follow-ups

**Merge note: `fix-choice-prob-windows` must import `nan_standard_error` from
`src.behavior_analysis.stats` and delete its own copy when the second branch merges.** The body is
identical; only the docstring differs (it now says a 1-D input is one column, giving one 0-d value
over all elements, not one value per element).

**Blocks across sessions (PI decision: a block open at a session's end continues into the next
session).** Traced `get_vars_across_all_sessions` -> `get_good_reversal_info` ->
`get_rank_counts_by_good_reversal`; nothing splits a block at a session boundary, so no code change
was needed there (and `get_good_reversal_info.py` is untouched, so no conflict with
`fix-reversal-classification`):

- Counter sessions: `good_reversals` is made cumulative across sessions by adding each session's
  total (`reversal_totals`, which includes a reversal after the session's last trial). A session
  start therefore increments only for a real reversal.
- Legacy blocks-only sessions: each session's block ids are offset so the first block of the next
  session carries the last id of the previous one (plus any carried terminal reversal events).
- Windows are cut on the merged trial list, so a block's window spans sessions.
- `tests/test_blocks_across_sessions.py` pins this: a block over two sessions is one row with all its
  trials (legacy and counter), a reversal carried to a session's first trial is still a boundary,
  and sessions are ordered by number.
- New field: every rank-count row has `n_trials` (its window length; for believed ranks the number of
  trials in the window, while `total` stays the eligible choices). Only a subject's final block can
  be cut short by the end of the data. Sample data (one session, 93 trials): block lengths
  22, 31, 33 and 7; the last block has 7 trials.

**Merge assumptions.**

- The PI confirmed that the rig's counters (good/bad reversals, block number) restart at each
  session; the merger relies on this when it offsets them.
- Merging counter-format sessions with blocks-only sessions misaligns trials (a blocks-only session
  contributes no counter values). Only the `has_good` part of the problem grouping keeps them apart
  today; `fix-reversal-classification` is adding a length check that raises.
- With recurring problems, a block open at the end of the first A stretch would join the second A
  stretch. `fix-import-data` has removed `merge_recurring_problems`, so recurring layouts are now
  separate problems and this no longer happens.
- `n_trials` marks partial blocks: the final block with `post=None`, the first block too if the data
  starts mid-problem, and every block when an integer `post` caps the windows.

**Other review items.** `se_by_rank(guard_single=True)` now counts non-NaN units per rank.
`fillna(0)` is gone from the cohort-comparison and trial-block problem-mean plots, so a single-mouse
problem draws its mean with no error bar (no zero-length cap). The pooled-SE docstring now states
that within-block serial correlation also understates it, that pooled and across-block bars are not
comparable, and that fractional tie weights (believed ranks) break the 0/1 indicator assumption
(`p(1-p)` is then an upper bound of the variance). The `plot_rank_proportions_average` docstring now
describes the current warnings.

## Found, not fixed

- The descriptive-statistics text in `plot_problem_summary` uses `sd = 0.0` for one value (a descriptive SD, not an SE).
- `_annotate_pvalues` / `_draw_rank_bars` still misplace p-values when a mean is NaN (separate known issues).
- The two choice-probability SE helpers above, pending `fix-choice-prob-windows`.
