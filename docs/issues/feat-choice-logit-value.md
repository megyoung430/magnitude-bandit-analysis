# feat-choice-logit-value

## Request
Fit a logistic regression of choice on port location, reward magnitude and previous choice
history, and use the resulting logits as internal value estimates to replot the psychometrics;
show it on synthetic data (as in the Q-learner simulations) and then on the dataset, as a
notebook.

## Change
- `src/behavior_analysis/choice_logit/choice_logit.py`: conditional logit over the three offered ports with
  reward-magnitude lags, choice lags and one bias per tower (alphabetically first tower fixed at
  0); analytic gradient, L2 penalty (default 1.0), L-BFGS-B, `RuntimeError` on non-convergence.
  Reward history carries over the sessions of a problem; choice history resets at a session start
  and after a dropped trial, as the value learner's stickiness does.
- `src/behavior_analysis/choice_logit/choice_logit_cv.py`: leave-one-problem-out scores of three nested models
  (reward; + choice history; + tower bias) and the held-out model value, the reward part of the
  full model's fold fit that excluded the problem.
  A reward tail (decayed sum of rewards older than the lags, decay chosen by held-out likelihood)
  gives long-unvisited arms a value. The held-out choice+tower model without reward terms is the
  *nuisance baseline*: a value built from recent rewards is non-zero mostly for recently chosen
  arms, so sticky mice look value-driven; every value curve is drawn with its baseline (dashed).
- `src/behavior_analysis/value/model_value_psychometrics.py`: rank-based per-mouse quantile bins
  (default 5, never empty, x at the bin-mean value, means from fewer than 3 mice not drawn),
  P(best)/P(second) curves and stay/switch pairs for a continuous value, each with its baseline;
  rows of the stay/switch figures are whether the previous choice was rewarded. The switch-target
  figure follows `feat-switch-target-gap`: only switches count, y = P(best alternative | switch),
  x = V(best alternative) - V(other arm) in rank bins within each reward row (at the bin-mean
  gap), exact ties dropped and counted in the caption, the nuisance model's P(best alternative |
  switch) drawn as the baseline, dotted 0.5 line.
- `stay_switch_curves` accepts a precomputed `diff` column; `plot_stay_switch.py` gains
  `plot_stay_switch_grid` (the 2 x 3 stay/switch grid with row headings, panels and baselines as
  arguments; `plot_stay_switch_curves` calls it with its old defaults), a `baselines` argument on
  `plot_stay_switch_panel`, and `baseline`/`xlabel` arguments on `plot_switch_target_panel`.
  Existing figures are unchanged.
- `src/agents/choice_logit_truth.py`, `src/pipelines/choice_logit/choice_logit_values.py`: generating utilities
  and the soft-target logit fit to their choice probabilities (reference weights) for simulated
  agents, cache-first loading, simulated cohorts (six tower triples, random start magnitudes),
  value-blind control.
- `src/behavior_visualization/choice_logit/plot_choice_logit.py`: weights, model comparison, psychometric rows,
  stay/switch figures.
- `scripts/choice_logit_values.ipynb`: the model, synthetic mice with known values, real data.

## Results change
None: new notebook and modules. Existing figures and tables are unchanged.
