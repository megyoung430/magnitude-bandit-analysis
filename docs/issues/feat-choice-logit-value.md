# feat-choice-logit-value

## Request
Fit a logistic regression of choice on port location, reward magnitude and previous choice
history, and use the resulting logits as internal value estimates to replot the psychometrics;
show it on synthetic data (as in the Q-learner simulations) and then on the dataset, as a
notebook.

## Change
- `src/behavior_analysis/choice_logit.py`: conditional logit over the three offered ports with
  reward-magnitude lags, choice lags and one bias per tower (alphabetically first tower fixed at
  0); analytic gradient, L2 penalty (default 1.0), L-BFGS-B, `RuntimeError` on non-convergence.
  Reward history carries over the sessions of a problem; choice history resets at a session start
  and after a dropped trial, as the value learner's stickiness does.
- `src/behavior_analysis/choice_logit_cv.py`: leave-one-problem-out scores of three nested models
  (reward; + choice history; + tower bias) and the held-out model value, the reward part of the
  full model's fold fit that excluded the problem.
- `src/behavior_analysis/model_value_psychometrics.py`: rank shares, per-mouse quantile bins
  (default 5), binned P(best)/P(second) curves and stay/switch pairs for a continuous value; rows
  of the stay/switch figures are whether the previous choice was rewarded.
- `stay_switch_curves` accepts a precomputed `diff` column and `plot_outcome_curves` takes its
  row headings and panels as arguments (defaults unchanged).
- `src/agents/choice_logit_truth.py`, `src/pipelines/choice_logit_values.py`: generating utilities
  and their best linear summary for simulated agents, cache-first loading, simulated cohorts.
- `src/behavior_visualization/plot_choice_logit.py`: weights, model comparison, psychometric rows,
  stay/switch figures.
- `scripts/choice_logit_values.ipynb`: the model, synthetic mice with known values, real data.

## Results change
None: new notebook and modules. Existing figures and tables are unchanged.
