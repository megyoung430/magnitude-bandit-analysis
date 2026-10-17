---
type: infrastructure
complexity: simple
status: done
---

# Per-mouse curves CSV for the value-change diagnostic

Show exactly what the mouse-weighted value-change plot averages: for every mouse its total number of events, its `prev_best` / `prev_second` / `third` choice-probability arrays and how many of its events have a trial at each t.

- **Files:** `src/behavior_analysis/value_change_windows.py`, `src/behavior_analysis/value_change_mouse_curves.py` (new), `src/pipelines/value_change_diagnostic.py`, `tests/test_value_change_mouse_curves.py` (new), `tests/test_value_change_diagnostic.py`

## What changes
- `ValueChangeCurves` gains `per_subject_n_events` (`{subject: events with a trial at each t}`) and `per_subject_total_events` (`{subject: int}`), at the end with a `None` default. No existing number changes.
- `mouse_curves_table(curves, scope=...)` (pure) returns one row per mouse and label (`prev_best`, `prev_second`, `third`, `n_events_at_t`), then the same rows for `mean_across_mice`, which equals `curves.mean` (what the mouse-weighted figure plots). Columns: `scope, config, mode, mouse, n_events, label, t-5 ... t+15`. A probability is empty (NaN) where no event has a trial at that t.
- `run_cohort_diagnostic` writes `mouse_curves_<scope>.csv` per cohort for each drifting problem (`problem-KK`) and for the pooled drifting problems (`all-drifting-problems`), both truncation modes in one file (`mode` column: `untruncated`, `truncated_at_rank_swap`), and lists them in `index.txt`. Events, window (pre 5, post 15) and configuration (best/decrease/rank_swap) are those of the value-change figures.

## Results change
- New files only: `results/figures/value-change-diagnostic/cohort-XX/mouse_curves_problem-KK.csv` and `mouse_curves_all-drifting-problems.csv`.
- `results/figures/value-change-diagnostic/cohort-XX/index.txt` gains a trailing section listing those CSVs.
- No figure, no existing table and no number in any existing output changes.

## Verification
- Synthetic two-mouse tests (unequal event counts) pin the per-mouse arrays to hand-computed values and to `_subject_means`, `mean_across_mice` to `curves.mean`, `n_events_at_t` and the column names.
- Cohort-03 MY_175_L: the pooled CSV's `mean_across_mice` rows equal the curves `run_value_change_analysis` plots (both modes, to 1e-12) and equal the single mouse's rows.
