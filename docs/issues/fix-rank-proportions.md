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
- `nan_standard_error` (per-column SE over non-NaN rows, NaN below two) is defined in
  `rank_proportions_summary.py`, with the same name and signature as the helper on
  `fix-choice-prob-windows`; the two are meant to be merged into one.
- Pooled mean and SE: the pooled proportion averages trials, so its SE is across the same
  trials (`rank_proportions.pooled_standard_errors`, `sqrt(p(1-p)/(n-1))`). Trials are treated
  as independent although they come in blocks, so this is a lower bound if blocks differ.
  The across-mice view (`plot_average_across_mice`) already used the same units for mean and SE
  (each mouse's pooled proportion, SE across mice) and is unchanged.
- Problem-summary path: `compute_problem_length_summary`, `compute_problem_blocklen_summary`
  (`_mean_se_n`) and `compute_problem_goodrev_summary` now share `_mean_se_n`: NaN values are
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

## Found, not fixed

- `cohort_comparison_stats.mean_and_sem` returns SEM 0 for a single mouse (pinned by a test), and
  `get_task_statistics._standard_error` returns 0.0 for fewer than two values; both contradict
  "fewer than two values gives NaN" but are outside the rank-proportion / problem-summary path.
- The descriptive-statistics text in `plot_problem_summary` uses `sd = 0.0` for one value (a descriptive SD, not an SE).
- `_annotate_pvalues` / `_draw_rank_bars` still misplace p-values when a mean is NaN (separate known issues).
