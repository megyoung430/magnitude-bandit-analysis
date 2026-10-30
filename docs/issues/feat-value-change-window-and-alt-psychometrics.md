---
type: data-science
complexity: complex
status: in review
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Complete value-change windows with separate pre and post, and stay/switch psychometrics

Meg asked for two things:

1. "For the symmetric 6 trials plots, can we edit it to be 2 trials before and 4 trials after a
   true value change (or at least edit the code to not require it to be symmetric but allow
   specifying how many trials before and after the true value change)."
2. Psychometrics of the value of the chosen option against the next-best alternative, then
   clarified as "you can do the p(stay) vs p(switch)".

## 1. Complete-window value-change figures

- `src/behavior_analysis/value_change_complete_window.py` (was `value_change_symmetric.py`):
  `select_complete_windows` keeps the events whose whole `-pre .. +post` window is intact, with
  the exclusion rules of the symmetric version (block or third-arm change, session edge, best-arm
  change before the event, missing choice). `pre` and `post` need not be equal; (6, 6) gives the
  same events, exclusion counts and counts as the former symmetric selection.
- `COMPLETE_WINDOW = (2, 4)` in `src/pipelines/value_change_analysis.py` is the default;
  `run_value_change_analysis(..., complete_window=(pre, post) | None)` changes it per call.
- Names: `... (<filter>, -2 to +4 Trials)` (ASCII hyphen, as in the other file names); equal
  sides keep `Symmetric ±N Trials`. Mode `complete_pre2_post4` (`symmetric_N` when equal).
  Caption: "Window: -2 to +4 trials, complete windows only."
- `value_change_symmetric_exclusions.csv` keeps its name, so a partial run still merges into the
  earlier table, and has `pre` and `post` columns instead of `symmetric_window`.
- `run_value_change_analysis` deletes complete-window figures of any size from its two
  weighting folders before writing, so a changed window leaves no stale figure (the cohort runner
  does not clear the value-change folders; the all-cohorts runner clears them anyway).

## 2. Stay/switch psychometrics

`src/behavior_analysis/stay_switch_psychometrics.py`, `src/behavior_visualization/
plot_stay_switch.py`, `src/pipelines/stay_switch_psychometrics.py`; figures in
`value-psychometrics-stay-switch/`, made wherever `value-psychometrics/` is (each drifting
problem, a cohort's pooled drifting problems, the all-cohorts pool).

- Pair: trial t and t-1 of the same session. The chosen option is the arm chosen on t-1.
  Values are those of trial t: true magnitudes, or believed ones (what the arm paid the last
  time it was chosen before t's choice, reset per problem, counted once all three arms are
  seen). V(chosen) is that arm's value, V(alt) the higher of the two other arms.
- Outcome: stay, or switch (1 - stay) to the best alternative or to the other arm.
- Pairs are kept across block and value changes, never across sessions or a missed trial.
- Ties, relative to the chosen arm and mutually exclusive: V(chosen) = V(alt); the two
  alternatives tie; all three tie. `exclude` drops them all (not quite the value
  psychometrics' exclude set: a tie of the chosen arm with the arm below V(alt) is kept here); `split` keeps them, V(chosen) = V(alt) at difference 0 and
  a switch between tied alternatives counted half to each target. Counts are in the captions.
- Figures per value basis and tie mode: `Stay-Switch Curves (...) (Pool Trials True|False)`
  (top row P(stay) and P(switch), bottom row the two switch targets, against V(chosen) - V(alt),
  V(chosen), V(alt); per-mouse lines, mean ± se across mice) and `Stay Heatmap (...)` (P(stay)
  over V(chosen) x V(alt), mean across mice, pairs per cell). Pooling: a mouse's pairs at an x
  pooled, or its (V(chosen), V(alt)) cells averaged equally.
- Results are returned (pairs, heatmap tables, curves); no CSV, like the value psychometrics.
- The descriptive chosen-vs-best-unchosen figure is not made (lower priority).

## Acceptance criteria

- (2, 4) keeps exactly the events with a full -2 .. +4 window; (6, 6) reproduces the symmetric
  result; names, captions, modes and table columns as above; stale complete-window figures go.
- Hand-built trials give the expected pairs, values, outcomes, ties and drops; curves and
  heatmap match hand computations for both weightings; the pipeline writes every figure into
  the new folder and the runners make it.

## Results change

- Value change: the ±6 symmetric figures (`... Symmetric ±6 Trials`, mode `symmetric_6`) are
  replaced by the -2 .. +4 set (`... -2 to +4 Trials`, mode `complete_pre2_post4`) in every
  `value-change/{mouse,event}-weighted/` folder; old ±6 figures are deleted on the next run of
  that folder. `value_change_symmetric_exclusions.csv` has `pre`, `post` instead of
  `symmetric_window`, and `value_change_bin_counts.csv` the new mode. Other figures unchanged.
- Run each cohort once in full (not `--since` / `--last-sessions`) after this change: a partial
  run only rewrites the problems it reruns, so the stacked exclusions table would mix ±6 rows
  (`symmetric_window`) with -2 .. +4 rows (`pre`, `post`) and the other problems keep their ±6
  figures.
- New folder `value-psychometrics-stay-switch/` next to every `value-psychometrics/`.
