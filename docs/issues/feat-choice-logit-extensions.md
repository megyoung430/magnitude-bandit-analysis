# feat-choice-logit-extensions

## Request
Four analyses from the supervisor meeting, added to the per-mouse choice logit and run on the mice,
a simulated Q-learner (positive control) and the three win-stay/lose-shift null agents (negative
controls): a believed-value regressor, a binary-against-pulse test of magnitude sensitivity, unique
and shared variance of the regressor groups, and a calibration figure whose y axis is the actual
choices.

## Change
- `src/behavior_analysis/choice_logit.py`: `design_matrix(..., extras=...)` appends optional
  column groups after the standard columns (their names and order are unchanged).
  - `BINARY`: `binary_lag<k>` (1 if the arm paid on trial `t-k`, else 0) and `binary_tail`, the
    same structure as the pulse lags.
  - `BELIEVED`: `believed_value` (the magnitude each arm last paid in the current problem, 0 before
    its first visit) and `believed_unvisited` (1 before the first visit). The history carries
    across sessions and dropped trials and resets at a problem's start, like the reward lags.
  - `select_features` and `weights_table` know the new groups; an unknown extra raises.
- `src/behavior_analysis/choice_logit_extensions.py` (pure): `cross_validate_variants` scores any
  set of feature groups leave-one-problem-out and returns the held-out probabilities;
  `VARIANTS` names the models; `believed_value_table`, `binary_pulse_table`, `gain_summary`
  (mean, se, Wilcoxon and sign tests across mice, tests only from 5 mice),
  `variance_partition`, and `calibration_bins` / `calibration_table` / `calibration_summary`.
  `choice_logit_cv.model_comparison` takes an optional model order.
- `src/pipelines/regression_extensions.py`: `analyse_extensions` (all variants per mouse, parallel
  over mice with `n_jobs`, each mouse's result cached under `cache_dir`), `extension_tables`,
  `comparison_inputs` (reuses the null-agent cohorts of a `NullComparison`), `analyse_sources`,
  `source_summaries`.
- `src/behavior_visualization/plot_regression_extensions.py`: `plot_believed_value`,
  `plot_binary_vs_pulse`, `plot_variance_partition`, `plot_calibration`, and `plot_extensions` for
  all of them (pdf, png, svg and a caption file each).
- `scripts/choice_logit_values.ipynb`: sections 2.8 (synthetic: Q-learner and null agents,
  reusing the cohorts of 2.7) and 3.4 (mice next to the same controls); figures in
  `regression-extensions/` subfolders of the synthetic and real output folders.

## Definitions
All losses are held-out negative log likelihood per choice (nats), chance is log 3.
- Believed-value variants: (a) full, (b) full + believed value and never-visited, (c) choice history
  + tower + believed value (no reward lags). Gain of (b) over (a) is `loss(a) - loss(b)`.
- Binary variants: binary (binary lags + choice + tower) and binary + pulse; gain over the pulse
  (standard) model is `loss(pulse) - loss(model)`.
- Unique variance of a group: `loss(full minus the group) - loss(full)`.
- Shared reward/choice: `[gain(reward + tower) + gain(choice + tower)] - gain(full)` with
  `gain(model) = loss(tower only) - loss(model)`. Unique reward + unique choice + shared equals
  the full model's gain over the tower-only model exactly. The tower bias stays in every model of
  the shared term: with single-group models (no tower) against a tower-only baseline the "gains"
  can be negative for a mouse with a strong port preference. Single-group gains over chance are in
  the tables.
- Calibration: every (trial, arm) pair is a prediction (the arm's held-out probability, outcome 1
  if chosen), cut per mouse into ten equal-count (quantile) bins; observed fraction chosen against
  mean predicted probability. Summary: mean over bins of |observed - predicted|, and the same
  weighted by bin size.

## Design choices
- One figure with a panel per source for the variance partition (common y axis, so the mice are
  read against the controls directly); one calibration figure per source with a full-model and a
  choice-history-only panel.
- Fewer than 5 mice in a source: the binary-against-pulse figure shows the mice individually,
  with no mean, error bar or test, and says so in the title and caption.
- Cost: every variant is refitted on each of a mouse's problems (11 variants), about 7 s per mouse
  with 5 lags and the tail on 4,800 trials. The mice are independent, so the work runs in
  parallel over mice and is cached per mouse (keyed on the trials, the settings and
  `CACHE_VERSION`).

## Results change
None: new regressors, model variants and figures. The standard design matrix, fits, cross
validation, figures and file names are unchanged.
