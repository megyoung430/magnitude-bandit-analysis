# fix-psychometric-diff-axis

## Problem

The value psychometric figures have three panels: value of best (x = 2, 3, 4), value of second
(x = 1, 2, 3) and value of best - second. With ties excluded, best - second = 0 has no trials,
but the third panel still had x = 0 on its axis, so it looked unlike the other two.

## Change

- `psychometric_curves` takes a required keyword `tie_mode`. With `"exclude"` the difference
  curve has x = (1, 2, 3); with `"split"` it keeps x = (0, 1, 2, 3). The fix is in the curves,
  not the plotting, so every reader of the curves (figures, returned `psychometric` results)
  sees the same axis.
- With `"exclude"`, trials in a cell with best equal to second raise `ValueError` (counts made
  with another tie mode). An unknown `tie_mode` raises too.
- `plot_psychometric_panel` sets x limits explicitly: 10% of the tick range of empty space
  beyond the first and last tick in every panel. Autoscaling ignores NaN points, so limits
  used to move with whichever end points had data.
- `run_value_psychometric_analysis` passes its tie mode to `psychometric_curves`.

Heatmaps, tie tables and the all-cohorts runner do not read the difference curve and are
unchanged.

## Tests

In exclude mode the difference x is (1, 2, 3), all three panels have three ticks and equal
relative padding; in split mode x includes 0; limits do not depend on which points have data;
tied-cell trials under exclude raise.

## Results change

Every `Psychometric Curves (...) (Ties Excluded) (Pool Trials ...)` PNG and PDF, in every
folder the runners write them to (`run_value_psychometric_analysis`, the cohort runner and
the all-cohorts drifting runner): the third panel has ticks 1, 2, 3 instead of 0-3, and all
panels have 10% instead of matplotlib's 5% margin of the tick range. The `Ties Split` curve
figures change only in that margin (about 0.05 of a tick range). Heatmaps are unchanged. The
returned `Curve` for `"diff"` in exclude mode has three x values instead of four.
