# feat-switch-regression

## Request
Meeting to-do: "switching regression controlling for choice history, looking at the loading on value
of best alternative; separate models after zero and after non-zero". On trials where the mouse leaves
its previous arm, does the value of the two alternatives predict which one it goes to, beyond choice
history, recency and tower bias? The choice-logit baseline has five choice lags and no tail, so a
recency-only agent (WSLS (Recency), decay 0.9) lands above it on switch targets; this regression has
to absorb long-range recency, and a mouse's value weight has to be read against what agents that
ignore value produce.

## Change
- `src/behavior_analysis/switch_regression.py` (pure): `switch_data` and `switch_data_by_decay`
  (switch trials, choice sets, history features and values of both alternatives; the design is built
  once and only the trace column changes across decays), `choice_trace`, `trials_since_chosen`,
  `fit_switch_logit` (two-alternative conditional logit, L2 and optional Firth, Laplace standard
  errors), `cross_validate` (leave-one-problem-out), `analyze_mouse`, `value_table`,
  `history_weight_table`, `random_effects` (DerSimonian-Laird), `group_summary`, `null_calibration`,
  `calibrated_summary`, and `TooFewSwitchesError`.
- `src/pipelines/switch_regression.py`: `analyze_switch_source(s)` (per source, with the lags, penalty
  and held-out model values of that source's logit analysis; parallel over mice, cached per mouse;
  only `TooFewSwitchesError` skips a mouse, with a warning naming its switch counts; worker warnings
  re-emitted, also on a cache hit), `source_summaries`, `per_mouse_table`, and the null calibration
  `calibrate_source` (`simulate_mouse_null`, cached per mouse and null agent).
- `src/behavior_visualization/plot_switch_regression.py`: `Switch Regression Value Weights`,
  `Switch Regression Model Comparison`, `Switch Regression History Weights` (pdf, png, svg and a
  caption file each), and `plot_switch_regression` for all three.
- `src/behavior_visualization/plot_regression_extensions.py`: the drawing helpers both figure modules
  share are public (`colors_of`, `draw_column`, `source_stats_line`, `source_ticks`,
  `counts_caption`, `style_source_axis`; they were underscore-prefixed), and `draw_column` takes
  optional per-mouse `bands` and `summary=False`. The extension figures draw as before.
- `scripts/choice_logit_values.ipynb`: sections 2.9 (synthetic: Q-learner, the three null agents
  fitted to it, two stress agents, and the check of the calibration) and 3.5 (mice next to the same
  controls, each mouse calibrated); figures in `switch-regression/` subfolders.

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
  artifact of the ordering; port preferences are the tower terms. Swapping the labels leaves
  weights and likelihood unchanged (tested).
- Lags `c_a(t-k)`: the logit's choice lags, reset at segment starts; K = the logit's lags (5). Lag 1
  is the previous choice, never in the choice set, so it is 0 for both alternatives and dropped.
- Trace `h_a`: decayed count of past choices within the segment, the null agents' own trace (tested
  equal). Decay chosen per mouse from {0.5, 0.7, 0.8, 0.9, 0.95} by the held-out likelihood of the
  history-only models (both conditions pooled), used for every value variant. Consequences: the
  decay is chosen on the folds that then score the gain, so the gain of the value is likely biased
  down; and the decay is chosen without the value, so for a mouse whose switches follow value the
  trace can take up part of it and a real value weight is biased towards 0. Neither protects
  against a bias of the null; that is what the calibration is for.
- Recency: trials since the arm was last chosen, within the problem and across sessions (trials,
  not time), capped at 100 (about three blocks), log(1 + x). Never chosen in the problem: indicator 1
  and recency at the cap. The indicator is the believed value's never-visited column, so the
  believed variant adds only `believed_value`.
- Values: true = the magnitude offered on the switch trial (`task_values` convention); believed =
  the logit's `believed_value` column; model = the logit's held-out reward part (`MouseCV.values`),
  a consistency check (same reward history as the believed value; the fold fit behind a training
  problem's model value saw the held-out problem), not calibrated.
- Fit: L2 = the logit's (1.0, a Gaussian prior with SD 1; it keeps tower columns that a fold never
  varies at 0). Nothing standardized, so `b_V` is logit per pulse (per logit for the model value).
  Standard errors from the inverse Hessian of the penalized likelihood: model-based, not clustered
  by problem, so they can be too small when switches within a problem are dependent, and the
  random-effects se built on them can be too small for real mice; a problem-level bootstrap would
  multiply the cost by its replicates and is not done.
- Firth (`fit_switch_logit(firth=True)`, notebook `SWITCH_FIRTH`): Jeffreys-prior penalty
  `-1/2 log det(I + l2)` on top of L2. Tested against the null agents below: it narrows the stress
  agents' null spread by 9-40% (mostly 20-35%) and that of agents with realistic fitted weights by
  4-11%, and does not
  remove the near-minimum bias (-0.116 to -0.098, pooled z -3.1 either way), at 3.3 times the cost
  per fit. Off by default; the calibration absorbs the bias whichever fit is used.
- Minimum: 100 switches per condition and mouse, in at least two problems; `analyze_mouse` raises
  `TooFewSwitchesError` below it and the pipeline skips such mice with a warning; the figures name
  them with the reason and the switch counts.
- Group statistics per source: per-mouse dots; from 6 mice mean ± se, Wilcoxon and sign tests
  (`MIN_TEST_MICE`) and a DerSimonian-Laird random-effects mean (tables and captions only).

## Null calibration (the inferential output for the mice)
A value-blind agent's raw value weight is not reliably 0 at a mouse's switch counts. With a strong,
nearly deterministic recency rule, and block changes that zero the arm chosen most, the true value
correlates with the trace; the null weights then spread widely and are biased at low counts. So each
mouse's fitted WSLS (Magnitude) and WSLS (Recency) agents (`NullComparison.controls.fits`) are
replayed on the mouse's own task (its layouts, sessions and trials; drift and block lengths measured
per cohort) `replicates` times (default 100, seeded per mouse, agent and replicate, cached), and every
replicate goes through the identical analysis, decay selection included. The mouse's weight and gain
are reported as `z = (observed - null mean) / null sd`, the percentile, and a two-sided empirical
p `(1 + #{|null - mean| >= |observed - mean|}) / (R + 1)`; across mice the calibrated z are tested
with a Wilcoxon test from 6 mice. The Value Weights and Model Comparison figures draw each calibrated
mouse's 2.5th-97.5th percentile of its WSLS (Magnitude) null as a gray band. Cost: about 0.6 s per
replicate and agent on 4,800 trials, so about 2 minutes per mouse for 100 replicates of both agents;
the notebook uses 100 for the mice and 20 (WSLS (Magnitude) only) for the synthetic check.

## Readouts
The believed value and the true value after non-zero, each against the mouse's own null. The true
value after zero is weak even for the simulated Q-learner (0.07 ± 0.03 logit per pulse, gain 0.0006)
and the null spread of a strong recency agent there (SD 0.2-0.3) is larger, so it cannot separate a
value learner from the null agents at these switch counts.

## Validation
Null value weights at realistic switch counts (median 370-710 per condition) and near the minimum
(120-250), 30 simulated mice per cell (26 for plain WSLS near the minimum, 4 had too few switches):
mean / SD across mice / pooled z, maximum likelihood (Firth in brackets where it differs).

| Agent | Size | True after zero | True after non-zero | Believed after zero | Believed after non-zero |
|---|---|---|---|---|---|
| WSLS (Magnitude), fitted | realistic | -0.001 / 0.056 / 0.0 | -0.003 / 0.034 / -0.5 | +0.014 / 0.073 / +1.1 | +0.002 / 0.039 / +0.3 |
| WSLS (Magnitude), fitted | near min. | +0.002 / 0.132 / 0.0 (0.118) | -0.022 / 0.073 / -1.7 | +0.022 / 0.107 / +0.9 | -0.015 / 0.072 / -1.0 |
| WSLS (Recency), w = 1.5 | realistic | +0.017 / 0.188 / +0.5 (0.173) | +0.031 / 0.129 / +1.4 | +0.051 / 0.251 / +1.0 (0.203) | +0.010 / 0.141 / +0.2 |
| WSLS (Recency), w = 1.5 | near min. | +0.014 / 0.315 / +0.5 (0.223) | -0.116 / 0.200 / -3.1 (-0.098 / 0.163) | +0.111 / 0.303 / +1.3 (0.181) | -0.010 / 0.320 / -1.0 (0.236) |
| WSLS (Magnitude), w = 1.5 | realistic | +0.031 / 0.244 / +0.6 (0.197) | +0.029 / 0.127 / +1.2 | +0.064 / 0.282 / +1.0 (0.228) | -0.023 / 0.137 / -0.9 |
| WSLS (Magnitude), w = 1.5 | near min. | -0.088 / 0.326 / -0.9 (0.214) | +0.049 / 0.200 / +1.1 | +0.107 / 0.339 / +0.9 (0.257) | +0.091 / 0.307 / +1.2 (0.251) |

Plain WSLS and fitted WSLS (Recency) behave like fitted WSLS (Magnitude) (SD 0.02-0.07 realistic,
0.06-0.12 near the minimum, |pooled z| at most 1.3).

Tests: `tests/test_switch_regression_validation.py` checks sign and pooled z per condition and value
variant for every null agent at realistic counts and for the fitted agents near the minimum
(Bonferroni over 20 checks, 12 mice: a bias of one per-mouse se is detected with about 67% power),
asserts the wide null spread of the stress agents near the minimum, and checks the Q-learner.
`tests/test_switch_null_calibration.py` validates the calibration: Q-learners' believed value lies
beyond all of 19 null replicates, and strong-recency null "mice" get calibrated p spread over 0-1.

## Results change
None: new analysis. The extension figures' shared drawing helpers lost their underscore (and
`draw_column` gained optional arguments); no existing figure, table or file name changes.
