---
type: infrastructure
complexity: simple
status: done
---

# Split the remaining long analysis and import functions

Refactor the long functions below into small, single-purpose functions (principle 4) with identical behaviour.

- **Files:** `src/behavior_analysis/{get_task_statistics,get_believed_value,split_early_late_good_reversals,problem_summary,matched_trial_budget}.py`, `src/behavior_import/{import_data,fix_grid_maze_cohort_02_problems}.py`
- **Functions over 60 lines:** get_block_lengths (91), compute_believed_value (73), split_good_reversals_early_late (73), compute_problem_blocklen_summary (64), match_trial_budget (63), import_data (74), fix_grid_maze_cohort_02_problems (65)
- **Notes:** import_data sits on the import path every analysis uses; src/behavior_import/load_trials.py caches its output keyed on a hash of src/behavior_import/*.py, so editing import_data invalidates users' caches (that is expected). Check the full sample-session import fingerprint.

## Acceptance criteria
- Every function in scope is at most ~60 lines of code; public names, signatures, defaults, return values unchanged.
- An EXACT before/after comparison against main is identical, including the full output of `extract_trials_grouped_by_problem(import_data(root))` for the sample session with merge False/True.
- New behaviour is pinned by tests through the PUBLIC functions with synthetic fixtures; deliberate one-line mutations make tests fail.
- `uv run pytest -q` passes; ruff check + format --check pass on touched files.
- Pre-existing bugs are documented with a present-tense `Known issue:` comment, not fixed.
