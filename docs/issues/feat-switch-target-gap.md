# feat-switch-target-gap

## Request
Switch target plots: condition on the earlier choice being the 0 option (top) or a non-0 option
(bottom) and the next choice being a switch, and plot P(next choice = best alternative) against
the value gap between the best and second-best of the options not chosen on the earlier trial.

## Change
- `switch_target_curves` (`src/behavior_analysis/stay_switch_psychometrics.py`) returns one
  `Curve`: P(switch to the best alternative | switch) against `alt_gap = v_alt - v_other`.
  Switches between tied alternatives (gap 0) are left out; they have no best alternative.
  `switch_target_curves_by_row` returns `{row: Curve | None}`.
- `Switch Target Curves (...)` (same file names) has one panel per row (chose 0 / chose non-0),
  y "P(Best Alternative | Switch)", x "Value of Best Alternative - Other Arm", a dotted 0.5
  line, n=mice and n=switches per panel; the caption defines the conditioning and the gap.

## Results change
Switch Target Curves only: new x axis and one panel per row. Rerun the pooled sets.
