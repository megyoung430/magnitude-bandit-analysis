---
type: feature
complexity: complex
status: in review
---

# Value heatmaps and psychometric curves (true and believed values)

## Description

For the drifting task, P(best) and P(second) against option value, for true and believed values:
overall and per-mouse heatmaps on the best x second grid (best 2-4, second 1-3), and three-panel
psychometric curves (vs best value, second value, best - second) in two weightings, each made
with ties excluded and with ties split. Builds on `get_choice_probs_by_value` and
`get_believed_value`.

## Entry point

```python
from src.pipelines.value_psychometrics import run_value_psychometric_analysis

run_value_psychometric_analysis(
    subjects_trials_by_problem: dict,   # {problem: {subject: {session: session_data}}}
    out_dir: Path,
    *,
    value_bases=("true", "believed"),
    pooling_modes=(True, False),
    tie_modes=("exclude", "split"),
) -> dict  # {basis: {tie_mode: {"counts": {mouse: MouseCounts}, "heatmaps": HeatmapTables,
           #                    "psychometric": {pool_trials: {"best"|"second": {axis: Curve}}}}}}
```

Files (PNG and PDF) in `out_dir`, per value basis and tie mode: `Heatmaps (<Value> Value) (Ties
<Excluded|Split>) Overall`, `... Per Mouse`, and `Psychometric Curves (<Value> Value) (Ties
<Excluded|Split>) (Pool Trials True|False)`. One problem or many: counts are made within each
problem (believed values reset per problem), then added per mouse. Unknown `value_bases`,
`pooling_modes` (anything but real bools) or `tie_modes` raise `ValueError` before anything is
written.

New modules: `src/behavior_analysis/value_psychometrics.py` (pure tables),
`src/behavior_visualization/plot_value_psychometrics.py` (drawing),
`src/pipelines/value_psychometrics.py` (entry point, not wired into the runners).
`count_choice_probs(..., use_believed, tie_mode)` / `choice_probs_from_counts` are new in
`get_choice_probs_by_value`; `accumulate_choice_probs` keeps its return value and gains
`tie_mode="split"`.

## Acceptance criteria

- Heatmaps: fixed 0-1 colour range, every cell annotated with probability and `n=` trial count,
  grey for empty cells. Overall = mean across mice; per-mouse grid.
- Curves: thin per-mouse lines, thick mean, SE band (omitted where SE is NaN), mean annotated
  inside 0-1 and clear of the other curve's label; `pool_trials` True (pool trials over cells
  sharing x) and False (equal-weight cell mean); mice are the replicates.
- Two versions of every figure: ties excluded and ties split. Every figure prints the number of
  trials with best/second, second/third and all-three ties, in both modes.
- Ties split evenly among tied arms (true and believed); out-of-grid trials counted, returned and
  warned per mouse; no empty-slice warnings; SE via `stats.nan_standard_error`.

## Decisions

PI decisions:
- A believed trial counts only once all three arms have been sampled in the problem (reason
  `fewer_than_three_arms_seen`). Beliefs carry across sessions, so this affects only the start of
  a problem.
- Tie modes: `"exclude"` drops every trial with a tie among the top three values (best/second,
  second/third, all three); `"split"` divides each tied rank position evenly among the tied arms
  (e.g. 3.5 trials to best and 3.5 to second for 7 trials of a best/second tie). Ties are counted
  per category per mouse in `MouseCounts.ties` and printed on every figure.
- Tied cells at the grid edge (4,4) and (1,1) have no grid cell, so in split mode they remain
  out-of-grid (counted, warned); in exclude mode they are excluded as ties instead.
- No minimum n for equal weighting; the cell n stays visible.

Implementation choices:
- Trial counts are printed in each heatmap cell ("0.62 / n=134") rather than as a companion
  heatmap: probability and evidence are read together and the per-mouse figure stays two panels
  wide.
- Heatmap cells do not depend on the pooling mode, so one heatmap set per value basis and tie
  mode; only the psychometric figures carry "(Pool Trials ...)". The overall heatmap stays the
  equal-mouse mean of cell probabilities (as in the notebook).
- The difference axis includes best - second = 0 (ties). The notebook kept only cells with
  best > second, so its three panels used different trials. Cell (best 2, second 3) cannot occur;
  trials there raise `ValueError`.
- Beliefs carry across a problem's sessions (in `session_order` order) and reset at each
  problem, as the rank-proportion analysis does. Previously they reset every session.
- `believed_chosen_rank` still breaks ties by arm name (other code uses it); the new
  `believed_best_share` / `believed_second_share` / `believed_tie` are what binning uses.
- A warning is logged by every `count_choice_probs` call, so a run over two value bases and two
  tie modes logs it up to four times per mouse.
- The out-of-grid warning's "N of M" has M = counted trials + out-of-grid trials (trials skipped
  for other reasons, `MouseCounts.dropped`, are not in M); the message says so.
- The label rule on psychometric panels: the higher mean's label goes above its point and the
  lower one's below; if that collides at the axis edge, the labels go beside the points.
- Both tie modes call `rank_position_shares`/`rank_weights`, also used now by
  `get_rank_counts_by_value`, so the two believed paths cannot drift apart.

## Bugs found in the notebook (and src)

1. `trial_variables[0]` (notebook): only the first recording was read. Already fixed in src
   (merged with `merge_trial_variables`); confirmed, covered by existing tests.
2. Believed ties between second and third, or all three, were broken by arm name, so P(second)
   depended on tower labels (notebook and src). Fixed: `rank_weights` splits each rank position
   evenly among tied arms. Tests: `tests/test_believed_ties.py`, `test_value_counts.py`
   (`test_believed_*_tie_*`; the choices are uneven between the tied arms so the old arm-name
   rule fails them).
3. True-value ties used the recorded rank label, which is arbitrary among equal magnitudes. Fixed
   using the chosen arm's magnitude; a split trial whose choice is not an arm raises. Tests:
   `test_true_*_tie_*` in `tests/test_value_counts.py`.
4. Out-of-grid trials were silently dropped. Now counted per mouse by (best, second) values
   (`MouseCounts.out_of_grid`), other skips by reason (`MouseCounts.dropped`), and a warning is
   logged per mouse. Tests: `test_out_of_grid_*`, `test_other_skipped_*`,
   `test_the_warning_says_what_the_total_counts`.
5. `nanmean` over all-NaN cells warned "Mean of empty slice". New `stats.nan_mean` is silent.
6. Notebook SE used its own formula; now `stats.nan_standard_error` (NaN with fewer than two
   mice).
7. Annotations at y +/- 0.05 could leave 0-1 and collided with the other curve's label.
   `annotation_position` and `label_placements` keep labels in range and apart (tests sweep every
   mean pair on a 0.01 grid for gaps 0 to 0.1).
8. The notebook dropped NaN points before drawing (`x[mask]`), joining a line across a missing x;
   now the line breaks at NaN. It also used global pyplot state; panels now take `ax`.
9. Believed values reset per session instead of per problem (src). See Decisions.
10. The session-order test passed with dict order too (dict order and date order gave the same
    count). It now uses a sequence where they differ; the mutation `for key in sessions_dict:`
    fails `test_believed_values_carry_across_sessions_of_a_problem_in_session_order`.
11. The per-mouse heatmap title overlapped the first row, and "1 mice". Titles and the tie line
    are now placed in fixed-inch margins around `tight_layout`; `_mice` pluralises.
12. `pooling_modes` was not validated (`"yes"` wrote "Pool Trials yes"). Now only real bools.

## Results change

- `accumulate_choice_probs` (default `tie_mode="split"`): unchanged for data without ties and with
  one session per problem, except that believed trials now wait for all three arms. Tied cells
  (second/third or three-way ties, true or believed) shift to the even split; believed values
  carry across sessions in a problem.
- Sample session (`MY_72_L`, one session, 93 trials): the three-arm rule removes 0 further trials
  (the 3 trials that were dropped before for fewer than two seen arms or an unseen chosen arm are
  the same 3, now `fewer_than_three_arms_seen`). Believed overall heatmaps in split mode are
  identical to before; true heatmaps identical (no true ties). The "Mean of empty slice" warning
  is gone.
- Ties on the sample: true values none. Believed: best/second 10, second/third 3, all three 0.
  Of these, in split mode (4,4) x4 (best/second), and (4,0) x2 and (2,0) x1 (second/third) are
  out of grid; (3,3) x6 is the tied cell in the grid. Split mode: 83 trials in the grid, 7
  out of grid. Exclude mode: 77 trials in the grid, 13 excluded as ties, nothing out of grid.
- Believed heatmaps, ties excluded (P(best) / P(second), rows best 2,3,4; columns second 1,2,3):
  best 2: .1765 / .6471 at second 1; best 3: 0, .8095 / 1.0, .0476 (cell (3,3) now empty); best 4:
  NA, .9091, .0625 / NA, 0, .9375. Split mode adds (3,3): .25 / .25.
- Psychometric points (sample; one mouse so SE is NaN), mean at x = 2,3,4 (best value), 1,2,3
  (second value), 0,1,2,3 (best - second); NaN where no trials:
  - true (identical in both tie modes), pooled, P(best): .1875, .7073, .7222 | .6346, .7419, .2 |
    NaN, .4423, .8, .9048; equal-weight: .1875, .7128, .7016 | .6085, .8462, .2 | NaN, .3599,
    .8667, .9048
  - true, pooled, P(second): .625, .1707, .1944 | .25, .129, .7 | NaN, .4038, .15, 0;
    equal-weight: .625, .1769, .2333 | .275, .0769, .7 | NaN, .4929, .1, 0
  - believed, split, pooled, P(best): .1765, .6607, .5526 | .1667, .8605, .1136 | .25, .3889,
    .8696, NaN; P(second): .6471, .125, .3947 | .6667, .0233, .75 | .25, .5, .0435, NaN
  - believed, exclude, pooled, P(best): .1765, .7727, .5526 | .1667, .8605, .0625 | NaN, .3889,
    .8696, NaN; P(second): .6471, .0909, .3947 | .6667, .0233, .9375 | NaN, .5, .0435, NaN
