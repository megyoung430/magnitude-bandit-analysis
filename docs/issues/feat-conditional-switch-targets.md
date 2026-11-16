# feat-conditional-switch-targets

## Request
Switch Target Curves as probabilities conditional on switching:
P(best alternative | switch after choosing 0), P(other arm | switch after choosing 0),
P(best alternative | switch after choosing non-0), P(other arm | switch after choosing non-0).

## Change
- `switch_target_curves` / `switch_target_curves_by_row` in
  `src/behavior_analysis/value/stay_switch_psychometrics.py`: the switch targets computed over the
  switches only, so the two targets add up to 1. A mouse without switches in a row is not one of
  that row's mice; `Curve.n_trials` counts switches. `stay_switch_curves` now gives only P(stay)
  and P(switch).
- `Switch Target Curves (...)` (same file names) draws them: y axis "Fraction of Switches",
  panel titles "mean ± se across mice, n=… mice, n=… switches", "No switches" for a row
  without switches; caption updated.
- `run_stay_switch_analysis` results gain `"switch_target_curves"`.

## Results change
Switch Target Curves figures: the curves are now fractions of switches, not of all pairs, and
sum to 1. Stay-Switch Curves and the Stay Heatmap are unchanged.
