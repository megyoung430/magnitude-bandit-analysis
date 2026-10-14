---
type: infrastructure
complexity: simple
status: in progress
---

# Split the long agent-fitting functions and the pipeline choice section

Refactor the long functions below into small, single-purpose functions (principle 4) with identical behaviour.

- **Files:** `src/agents/{cohort_fitting,parameter_recovery,trajectory_viewer}.py`, `src/pipelines/open_field.py`
- **Functions over 60 lines:** run_problem_cross_validation (126), run_recovery (86), trajectory_viewer (71), run_choice_probability_analysis (63)

## Acceptance criteria
- Every function in scope is at most ~60 lines of code; public names, signatures, defaults, return values and on-disk output layout unchanged.
- An exact before/after comparison against main is identical (float.hex / raw bytes, RNG state, printed output, exceptions, every file written, input mutation).
- New behaviour is pinned by tests through the public functions with synthetic fixtures; deliberate one-line mutations make tests fail.
- `uv run pytest -q` passes; ruff check + format --check pass on touched files.
- Pre-existing bugs are documented with a present-tense `Known issue:` comment, not fixed.
