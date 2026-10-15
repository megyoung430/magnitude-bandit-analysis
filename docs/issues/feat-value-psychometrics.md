---
type: feature
complexity: complex
status: in review
---

# Value heatmaps and psychometric curves (true and believed values)

## Description

For the drifting task, P(best) and P(second) against option value, for true and believed values:
overall and per-mouse heatmaps on the best x second grid (best 2-4, second 1-3), and three-panel
psychometric curves (vs best value, second value, best - second) in two weightings. Builds on
`get_choice_probs_by_value` and `get_believed_value`.

## Entry point

```python
from src.pipelines.value_psychometrics import run_value_psychometric_analysis

run_value_psychometric_analysis(
    subjects_trials_by_problem: dict,   # {problem: {subject: {session: session_data}}}
    out_dir: Path,
    *,
    value_bases=("true", "believed"),
    pooling_modes=(True, False),
) -> dict  # {basis: {"counts": {mouse: MouseCounts}, "heatmaps": HeatmapTables,
           #          "psychometric": {pool_trials: {"best"|"second": {axis: Curve}}}}}
```

Files (PNG and PDF) in `out_dir`: `Heatmaps (<Value> Value) Overall`, `Heatmaps (<Value> Value)
Per Mouse`, `Psychometric Curves (<Value> Value) (Pool Trials True|False)`. One problem or many:
counts are made within each problem (believed values reset per problem), then added per mouse.

New modules: `src/behavior_analysis/value_psychometrics.py` (pure tables),
`src/behavior_visualization/plot_value_psychometrics.py` (drawing),
`src/pipelines/value_psychometrics.py` (entry point, not wired into the runners).
`count_choice_probs` / `choice_probs_from_counts` are new in `get_choice_probs_by_value`;
`accumulate_choice_probs` keeps its signature and return value.

## Acceptance criteria

- Heatmaps: fixed 0-1 colour range, every cell annotated with probability and `n=` trial count,
  grey for empty cells. Overall = mean across mice; per-mouse grid.
- Curves: thin per-mouse lines, thick mean, SE band (omitted where SE is NaN), mean annotated
  inside 0-1; `pool_trials` True (pool trials over cells sharing x) and False (equal-weight cell
  mean); mice are the replicates.
- Ties split evenly among tied arms (true and believed); out-of-grid trials counted, returned and
  warned per mouse; no empty-slice warnings; SE via `stats.nan_standard_error`.

## Decisions

- Trial counts are printed in each heatmap cell ("0.62 / n=134") rather than as a companion
  heatmap: probability and evidence are read together and the per-mouse figure stays two panels
  wide.
- Heatmap cells do not depend on the pooling mode, so there is one heatmap set per value basis;
  only the psychometric figures carry "(Pool Trials ...)". Overall heatmap stays the equal-mouse
  mean of cell probabilities (as in the notebook).
- The difference axis includes best - second = 0 (ties). The notebook kept only cells with
  best > second, so its three panels used different trials. Cell (best 2, second 3) cannot occur;
  trials there raise `ValueError`.
- Believed values carry across a problem's sessions (in `session_order` order) and reset at each
  problem, as the rank-proportion analysis does. Previously they reset every session.
- Beliefs rank only seen arms (existing `compute_believed_value` behaviour, kept): a trial with
  two seen arms is binned by those two. The notebook instead excluded trials until all arms were
  seen. The PI may want to check which is intended.
- `believed_chosen_rank` still breaks ties by arm name (other code uses it); the new
  `believed_best_share` / `believed_second_share` are what binning uses.
- Per-mouse heatmaps for a mouse with no counted trials are all-grey rather than omitted.

## Bugs found in the notebook (and src)

1. `trial_variables[0]` (notebook): only the first recording was read. Already fixed in src
   (merged with `merge_trial_variables`); confirmed, covered by existing tests.
2. Believed ties between second and third, or all three, were broken by arm name, so P(second)
   depended on tower labels (notebook and src). Fixed: `rank_position_shares` splits each rank
   position evenly among tied arms. Tests: `tests/test_believed_ties.py`,
   `test_believed_second_third_tie_does_not_depend_on_arm_names`,
   `test_believed_three_way_tie_gives_each_arm_a_third`.
3. True-value ties (best = second, second = third, all three) used the recorded rank label, which
   is arbitrary among equal magnitudes. Fixed using the chosen arm's magnitude; a tied trial whose
   choice is not an arm raises. Tests: `test_true_*_tie_*` in `tests/test_value_counts.py`.
4. Out-of-grid trials were silently dropped. Now counted per mouse by (best, second) values
   (`MouseCounts.out_of_grid`), other skips by reason (`MouseCounts.dropped`), and a warning is
   logged per mouse. Tests: `test_out_of_grid_*`, `test_other_skipped_*`.
5. `nanmean` over all-NaN cells warned "Mean of empty slice". New `stats.nan_mean` is silent.
   Tests: `test_nan_mean_*`, `test_no_warnings_when_every_mouse_is_empty_at_some_x`.
6. Notebook SE used its own formula; now `stats.nan_standard_error` (NaN with fewer than two
   mice). Test: `test_se_is_nan_with_a_single_mouse_and_mean_is_still_drawn`.
7. Annotations at y +/- 0.05 could leave 0-1; `annotation_position` flips the label to the other
   side when there is no room. Test: `test_annotation_stays_inside_the_unit_interval`,
   `test_mean_annotations_are_inside_the_axes_even_at_the_extremes`.
8. Notebook joined mouse points across NaN gaps by dropping them (`x[mask]`), drawing a line over
   a missing x; now the line breaks at NaN. Notebook also used `plt.show()` and `plt.colorbar`
   on the global state; panels now take `ax`.
9. Believed values reset per session instead of per problem (src). See Decisions.

## Results change

- `accumulate_choice_probs`: unchanged for data without ties and with one session per problem.
  Changes: tied cells (second/third or three-way ties, true or believed) shift to the even split;
  believed values now carry across sessions in a problem (more binned trials after the first
  session).
- On the sample session (`MY_72_L`, one session) the true and believed overall heatmaps are
  identical before and after (no second/third tie; the best/second tie cell (3,3) was already
  0.25/0.25). True P(best) rows (second = 1/2/3): best 2: .1875; best 3: .7333, .6923; best 4:
  .9048, 1.0, .2. True P(second): 2: .625; 3: .2, .1538; 4: 0, 0, .7. Believed P(best): 2: .1765;
  3: 0, .8095, .25; 4: NA, .9091, .0625. Believed P(second): 2: .6471; 3: 1, .0476, .25; 4: NA, 0,
  .9375. The "Mean of empty slice" warning no longer appears.
- New: 7 of 90 believed trials on the sample fall outside the grid ((4,4) x4, (4,0) x2, (2,0) x1;
  true values: none), reported by the warning.
- Psychometric points (sample, one mouse so SE is NaN), mean at x = 2,3,4 (best value), 1,2,3
  (second value), 0,1,2,3 (best - second); NaN where no trials:
  - true, pooled, P(best): .1875, .7073, .7222 | .6346, .7419, .2 | NaN, .4423, .8, .9048
  - true, equal-weight, P(best): .1875, .7128, .7016 | .6085, .8462, .2 | NaN, .3599, .8667, .9048
  - true, pooled, P(second): .625, .1707, .1944 | .25, .129, .7 | NaN, .4038, .15, 0
  - true, equal-weight, P(second): .625, .1769, .2333 | .275, .0769, .7 | NaN, .4929, .1, 0
  - believed, pooled, P(best): .1765, .6607, .5526 | .1667, .8605, .1136 | .25, .3889, .8696, NaN
  - believed, equal-weight, P(best): .1765, .3532, .4858 | .0882, .8593, .1562 | .25, .3495, .4545, NaN
  - believed, pooled, P(second): .6471, .125, .3947 | .6667, .0233, .75 | .25, .5, .0435, NaN
  - believed, equal-weight, P(second): .6471, .4325, .4688 | .8235, .0238, .5938 | .25, .5441, .5, NaN
