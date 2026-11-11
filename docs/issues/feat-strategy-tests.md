# feat-strategy-tests

## Request
Meeting to-dos: (1) the *second-best test*: when the mouse leaves the best arm, does it go to the
second best rather than the third? (2) *heuristic or value learner*: does a heuristic that switches
towards what the alternatives last paid and towards never-visited arms predict held-out choices as
well as a Q-learner? Each mouse is read against its own null (fitted win-stay/lose-shift agents),
and against a positive control, a Q-learner fitted to the mouse. Added later: the fitted Q-learner
as a source everywhere the WSLS agents appear, longer schedules for the comparison columns, a
fourth null agent (WSLS (Graded)), and leave-one-problem-out scoring of every null agent.

## Change
- `src/behavior_analysis/second_best.py` (pure): the switch context, `leave_best_outcome`,
  `believed_outcome`, `true_outcome`, `analyze_mouse` (proportions and graded curves by magnitude
  difference), `apply_minimum`, `group_summary`, `calibrate`, `calibrated_summary`.
- `src/pipelines/second_best.py`: `analyze_second_best_source(s)`, `source_summaries`,
  `per_mouse_table`, and `calibrate_second_best` (each mouse against its own fitted WSLS (Magnitude)
  and WSLS (Recency) agents replayed on its task, cached per mouse and agent, keyed on
  `code_version()`). `switch_regression._rules_by_mouse` is public as `rules_by_mouse`.
- `src/agents/heuristic_agent.py`: Heuristic+ (choice probabilities, policy, fit, likelihood).
  `src/agents/value_agent.py`: `trial_log_probabilities` (held-out scoring).
- `src/agents/null_agents.py`: WSLS (Graded) (`wsls_graded`): P(stay) = sigma(w0 + w_m m), m the
  previous magnitude (0 if unrewarded); a switch goes to the more recently chosen of the two
  alternatives in the problem with probability `p_recent`, otherwise to the other (50/50 if neither
  was chosen yet or on a tie). `w0`, `w_m` by a two-parameter logistic regression (`w0` can be fixed
  to 0), `p_recent` in closed form from the informative switches (clipped); too few pairs or switches
  raise. It is the fourth comparison agent and the third calibration agent (WSLS (Magnitude) stays
  the agent whose band is drawn).
- `src/behavior_analysis/strategy_models.py`: the models behind one registry (WSLS (Magnitude),
  Heuristic+, Q-learner, WSLS (Graded); all four WSLS agents are also scorable), the
  leave-one-problem-out cross-validation, the paired differences, winners and the recovery tables.
  `src/pipelines/null_agents.py`: `heldout_scores` scores every null agent with that machinery, on the
  folds of the choice logit.
- `src/pipelines/strategy_comparison.py`: `analyze_comparison_source` (parallel over mice, cached per
  mouse on `code_version()`), the summaries, and `recover_models` (model and parameter recovery on
  the mice's own schedules).
- `src/pipelines/fitted_q.py`: the fitted Q-learner positive control: `fits_of` (reuses the
  comparison's fits), `fit_q_learners`, `parameters_table` (flags parameters at a bound),
  `simulate_fitted_q_cohort` (a `SyntheticCohort` with `kind="q_fitted"`, simulated on each mouse's
  own layouts under the cohort's task rules, carrying the mouse's name) and `power_check`.
- `src/pipelines/null_agents.py`: `compare_with_controls(extra_cohorts=...)` analyzes further
  simulated sources (the fitted Q-learner) like the agents; `lengthen_layouts` and the
  `length_factor` argument of `simulate_null_cohort`, `null_controls`, `compare_with_controls`.
  `regression_extensions.comparison_inputs` lists the extra cohorts, so 3.4 and 3.5 include them.
- `src/behavior_visualization/plot_second_best.py`: `Second Best Choices`, `Better Alternative Curves`.
  `plot_strategy_models.py`: `Strategy Model Comparison`, `Strategy Model Recovery`. Each with pdf,
  png, svg and a caption file, in a `strategy-tests/` subfolder.
- `scripts/choice_logit_values.ipynb`: sections 2.10 (synthetic: Q-learner, the four WSLS agents,
  recovery) and 3.6 (mice), and a "Fitted Q-learner" block before 3.3.

## Definitions and decisions
- Leave-best: the previous choice had the strict maximum magnitude; the outcome is whether the better
  of the two alternatives (ranked on the same trial) was chosen. Ties are excluded and counted.
  **Leaving the best never follows a zero payout** (the payout is the chosen arm's magnitude, and
  exactly one arm pays 0, never the best), so the after-zero cell is empty by construction; the figure
  says "none". A cell needs 30 used switches (`MIN_USED`); a mouse below it is left out of that cell
  with a warning and named in the figure and caption.
- Believed better: both alternatives chosen before in the problem, believed values differ; true
  better: the magnitudes of the alternatives differ. The 0.5 line is for orientation only; mice are
  read against their own null.
- **Trace decay is fixed at 0.9** (the null agents' default), not fitted: Heuristic+ then nests WSLS
  (Magnitude) and a difference is attributable to `v` and `u`; a decay fitted next to `w` is weakly
  identified. `u` (never visited) carries the switch regression's L2 prior, because the mouse nearly
  always takes the unvisited arm and an unpenalized fit would diverge.
- **Null agents are held out too.** The agents' negative log likelihood in the null comparison is
  leave-one-problem-out, on the folds of the choice logit (one per problem, in order of appearance;
  segment starts credited log(1/3) so nats per trial equal the logit's convention). The in-sample
  table stays on `NullComparison.in_sample`, labeled, and is not compared with the logit. A mouse
  with one problem is skipped with a warning. Every model comparison (choice logit, regression
  extensions, switch regression, strategy comparison, null agents) uses these folds.
- WSLS (Graded) does not nest WSLS (Magnitude) (a different stay predictor and a recency rule instead
  of a decayed trace), so its differences are not limited by parsimony in either direction.
- All four models are scored on the same choices (segment starts excluded; the Q-learner still
  learns from them), in held-out nats per choice pooled over the mouse's folds. Differences are first
  minus second (negative: the first fits better). The choice logit's held-out nats are a descriptive
  reference (all trials), not part of the comparison.
- The Q-learner is fitted from `Q_STARTS` (5) random starts; non-convergence and a Heuristic+ or
  WSLS parameter at a bound raise. A Q-learner parameter at a bound is flagged in the parameter table.
- Model recovery plays each model, fitted to each mouse, on that mouse's own layouts and runs the
  identical comparison. Recovery is called poor when fewer than 80% of a model's simulated mice are
  won by it; a parameter when its correlation with the generating value is below 0.7. The captions
  say so plainly. Heuristic+ nests WSLS (Magnitude): when the simpler model generates, the two differ
  by about 0.001 nats and either can win by chance.
- A source with fewer than 6 mice shows its mice only (no mean, se or test).

## Controls
- **Negative controls and calibration**: each mouse is calibrated against its own fitted WSLS
  (Magnitude), WSLS (Recency) and WSLS (Graded) agents (z, percentile, empirical p; Wilcoxon on z from n >= 6).
- **Positive control, the fitted Q-learner** ("Q-Learner (Fitted)"): the Q-learner fitted to each
  mouse on all its problems (the comparison's fits, reused), played on that mouse's own task. It
  appears wherever the WSLS agents do (3.3 kernels, psychometrics, stay/switch and switch targets, 3.4,
  3.5, 3.6); the synthetic "Q-Learner" source stays. It is not a null: calibration stays against the
  WSLS agents. `power_check` replays each fitted Q-learner `POWER_REPLICATES` (50) times through the
  switch regression and reports the share whose value weight exceeds the mouse's own (about 1 s per
  replicate).
- **Schedule length**: the comparison columns (the four WSLS agents and the fitted Q-learner) play
  schedules `CONTROL_LENGTH_FACTOR` (2.0) times as long as the mouse's (`lengthen_layouts` repeats
  each problem's sessions), so agents of mice with few trials do not fall under the switch
  regression's 100-switch minimum, and are less noisy. **The calibration nulls (switch regression and
  second best) and the power check always use the mouse's own length** (factor 1); tests assert this.
  Nothing else pairs trial counts across sources.

## Cost (notebook)
Synthetic section 2.10: about 4 minutes uncached (second best on 4 sources of 10 mice with 20
calibration replicates; comparison on 3 mice per source and recovery on 2, one Q start). Real data per
mouse: the comparison costs about 1.5 minutes per mouse and the recovery about three times that, with
5 starts (cached per mouse, `STRATEGY_JOBS` workers); second-best calibration about 1 minute per mouse
at 100 replicates for two agents; power check under 1 minute per mouse. Meg runs cohorts 1, 3 and 4.

## Results change
None for the existing figures' data except one deliberate change: the null agents' negative log
likelihood table in 2.7 and 3.3 is now leave-one-problem-out (it was in-sample), so those numbers
move up by the (small) optimism and the table title says so. Everything else is a new analysis;
`compare_with_controls`, `null_controls` and `simulate_null_cohort` gained optional arguments with
defaults that reproduce their previous output exactly.
