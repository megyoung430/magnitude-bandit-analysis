# feat-switch-regression

## Request
Meeting to-do: "switching regression controlling for choice history, looking at the loading on value
of best alternative; separate models after zero and after non-zero". On trials where the mouse leaves
its previous arm, does the value of the two alternatives predict which one it goes to, beyond choice
history, recency and tower bias? The choice-logit baseline has five choice lags and no tail, so a
recency-only agent (WSLS (Recency), decay 0.9) lands above it on switch targets; this regression has
to absorb long-range recency so that the null agents give no value weight.

## Change
- `src/behavior_analysis/switch_regression.py` (pure): `switch_data` (switch trials, choice sets,
  history features and values of both alternatives), `choice_trace`, `trials_since_chosen`,
  `fit_switch_logit` (two-alternative conditional logit with Laplace standard errors),
  `cross_validate` (leave-one-problem-out), `analyse_mouse` (trace decay chosen by held-out
  likelihood, every model in both conditions), `value_table`, `history_weight_table`,
  `random_effects` (DerSimonian-Laird) and `group_summary`.
- `src/pipelines/switch_regression.py`: `analyse_switch_source` (per source, with the lags, penalty
  and held-out model values of that source's logit analysis; parallel over mice, cached per mouse;
  the cache key holds the settings and a hash of the design-matrix, null-agent and switch-regression
  sources; worker warnings re-emitted in the parent, also on a cache hit), `analyse_switch_sources`,
  `source_summaries`, `per_mouse_table`.
- `src/behavior_visualization/plot_switch_regression.py`: `Switch Regression Value Weights`,
  `Switch Regression Model Comparison`, `Switch Regression History Weights` (pdf, png, svg and a
  caption file each), and `plot_switch_regression` for all three.
- `scripts/choice_logit_values.ipynb`: sections 2.9 (synthetic: Q-learner, the three null agents
  fitted to it, and two stress agents) and 3.5 (mice next to the same controls); figures in
  `switch-regression/` subfolders of the synthetic and real output folders.

## Definitions
- Switch trial: a stay/switch pair of `null_agents.decisions` (consecutive trials of one segment; a
  segment starts at a problem's start, at `trial_in_session == 0` and after `after_gap`) whose choice
  differs from the previous one. After zero: the previous choice paid 0; after non-zero: 1 or more.
- Choice set: the two arms other than the previous choice, ordered by sorted port. A problem always
  offers the same three arms (`design_matrix` raises otherwise) and an arm paying 0 is still
  available, so no arm is ever dropped.
- Model: `P(first) = sigmoid(U(first) - U(second))` with
  `U(a) = b_V V(a) + sum_{k=2..K} b_k c_a(t-k) + b_tr h_a + b_rec log(1 + min(since_a, 100)) +
  b_nev never_a + tower(a)`. No intercept: it would be a bias towards the lower-sorted arm, an
  artifact of the ordering; port preferences are the tower terms. Swapping the labels then leaves
  weights and likelihood unchanged (tested).
- Lags `c_a(t-k)`: the logit's choice lags, reset at segment starts; K = the logit's lags (5). Lag 1
  is the previous choice, never in the choice set, so it is 0 for both alternatives and dropped.
- Trace `h_a`: decayed count of past choices within the segment, the null agents' own trace
  (tested equal to theirs). Decay chosen per mouse from {0.5, 0.7, 0.8, 0.9, 0.95} by the
  held-out likelihood of the history-only models, both conditions pooled, and used for every value
  variant of the mouse. The value plays no part in the choice; choosing on the same folds flatters
  the history-only model slightly, which can only shrink the value's gain.
- Recency: trials since the arm was last chosen, within the problem and across sessions (trials,
  not time), capped at 100 (about three blocks), log(1 + x). Never chosen in the problem: indicator 1
  and recency at the cap. The indicator is the believed value's never-visited column, so the
  believed variant adds only `believed_value`.
- Towers: the logit's tower columns (first tower as reference) and its warning when the problems'
  tower sets share no tower.
- Values: true = the magnitude offered on the switch trial (`task_values` convention); believed =
  the logit's `believed_value` column (last magnitude received in the problem, 0 before the first
  visit); model = the logit's held-out reward part (`MouseCV.values`), a consistency check: same
  reward history as the believed value, and the fold fit behind a training problem's model value saw
  the held-out problem.
- Fit: L2 = the logit's (1.0, a Gaussian prior with SD 1); nothing standardized, so `b_V` is logit per
  pulse (per logit for the model value). Standard errors from the inverse Hessian of the penalized
  likelihood. Held-out gain: leave-one-problem-out nats per switch of the history-only model minus
  that of the model with the value, same folds; a problem without switches of a condition is no fold.
- Minimum: 100 switches per condition and mouse, in at least two problems (about 15 weights);
  `analyse_mouse` raises below it, the pipeline skips such mice with a warning.
- Group statistics per source: per-mouse dots; from 6 mice mean ± se, Wilcoxon and sign tests
  (`MIN_TEST_MICE`), and a DerSimonian-Laird random-effects mean of the per-mouse weights (tables and
  captions only).

## Validation
- Tests (`tests/test_switch_regression_validation.py`, about 50 s): six Q-learners, the three null
  agents fitted to them, and WSLS (Recency) and WSLS (Magnitude) with w = 1.5, decay 0.9. For every
  null source and value (true, believed) the pooled z of the value weight over mice and conditions
  is below 3.5 (Bonferroni over the 10 checks, family-wise 1%), both signs occur, the mean weight is
  below 0.2 and the mean gain below 0.003 nats per switch; the stress agents get decay 0.9; the
  Q-learners have a positive believed-value weight in every mouse and a positive gain.
- Seed checks outside the suite: over 10 seeds x 6 mice per stress agent the pooled z of every
  condition and value is within ±1.7 (z SD 0.8-1.06, so the standard errors are calibrated) and the
  mean gains are negative; for the fitted WSLS (Recency) four seeds give a believed-value weight
  after non-zero of 0.039, -0.007, 0.016, -0.009, and fixing the decay at the agents' 0.9 changes
  nothing, so the controls needed no remedy (second trace, spline) beyond the trace and recency
  terms above.

## Results change
None: new analysis. No existing function, figure or file name changes.
