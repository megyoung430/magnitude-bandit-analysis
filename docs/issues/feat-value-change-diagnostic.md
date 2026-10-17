---
type: feature
complexity: simple
status: in review
---

# Value-change diagnostic: one session figure per drifting session

To check the "Choice Probabilities Around Best-Value Decreases (Rank Swap, Truncated at Swap)"
figure by eye. For every drifting session it draws the single-session magnitude/choice panel and,
for each best-decrease rank-swap event, the pre and post bins the analysis counts and the arm each
label (`prev_best`, `prev_second`, `third`) names.

## Acceptance criteria
- Top panel is `plot_magnitudes_and_choices` (session-local trial index = `ValueChangeEvent.trial_idx`).
- Shaded bins are exactly the non-`None` bins of `event.pre` / `event.post`, one horizontal band per
  event in a strip below the panel (overlapping events stay readable), with a line and "t=0" at t = 0.
- Choices in a window are coloured by label (prev_best blue, prev_second green, third grey, as in
  `plot_value_change_choice_probs`); legend for shading and labels.
- Text below the strip: `#n t=0 at trial T: prev best = X, prev second = Y, third = Z | pre u of 5
  trials, post v of 15 trials, truncated at swap (t=+k)` / `not truncated`.
- Sessions without events still get a figure, marked "no best-decrease rank-swap events".
- Script `scripts/individual_analyses/plot_value_change_diagnostic.py --cohorts N [N ...] [--no-cache]`
  writes `results/figures/value-change-diagnostic/cohort-XX/problem-KK/<mouse>/<session>.png` (PNG
  only) and a per-cohort `index.txt`; drifting problems only.

## Entry points
- `src.behavior_analysis.value_change_diagnostic.diagnose_session(subject, session_key, session)`
  returns one `EventDiagnostic` per event (window trial indices, label per trial, arm per label).
- `src.behavior_analysis.value_change_windows.swap_label_arms(magnitudes)` is the public form of the
  analysis's label-to-arm rule.
- `src.behavior_visualization.plot_value_change_diagnostic.plot_value_change_diagnostic`.
- `src.pipelines.value_change_diagnostic.run_cohort_diagnostic`.

## Decisions
- Events and windows come from `get_value_change_windows(..., pre=5, post=15,
  truncate_at_rank_swap=True)` itself, so the figure cannot drift from the analysis.
- Where events' windows overlap, a trial's dot on the session panel takes the label of the event whose
  t = 0 is nearest (earlier on a tie); each event's own labels are the dots in its strip band.
- The plotted per-tower series must have as many trials as the merged trial variables, else a
  `ValueError` (the event indices would not match the plot).
- A session without trial information (the importer skips it) has no figure.
- `--results-root` is an optional extra argument (default: the runner's `DEFAULT_RESULTS_ROOT`).

## Results change
New figures and `index.txt` under `results/figures/value-change-diagnostic/`. No existing output
changes; the cohort runner is untouched.
