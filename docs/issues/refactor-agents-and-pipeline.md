---
type: infrastructure
complexity: simple
status: done
---

# Split the long agent-fitting functions and the pipeline choice section

> **WARNING: merging this changes every recovery and cohort cross-validation directory name.**
> `run_problem_cross_validation` hashes the full text of `cohort_fitting.py` and
> `run_recovery` hashes the text of `parameter_recovery.py` (plus the runner, design, task,
> mouse, fit and loss sources) into the manifest; the directory name is a hash of the
> manifest. Any edit to those files, including this refactor, moves a run to a new folder.
> Partially-run sweeps and cross-validations started on main will **restart from zero in a
> new folder rather than resume**; finished results stay readable in the old folder. Every
> other output file is unchanged. The cache key is deliberately not narrowed here; that is a
> follow-up (shared `run_store` module plus an explicit `ALGORITHM_VERSION`).
>
> Superseded by `infra-run-store`: that follow-up is done; source text no longer enters the hash.

Refactor the long functions below into small, single-purpose functions (principle 4) with identical behaviour.

- **Files:** `src/agents/{cohort_fitting,parameter_recovery,trajectory_viewer}.py`, `src/pipelines/open_field.py`
- **Functions over 60 lines:** run_problem_cross_validation (126), run_recovery (86), trajectory_viewer (71), run_choice_probability_analysis (63)

## Acceptance criteria
- Every function in scope is at most ~60 lines of code; public names, signatures, defaults, return values and on-disk output layout unchanged.
- An exact before/after comparison against main is identical (float.hex / raw bytes, RNG state, printed output, exceptions, every file written, input mutation).
- New behaviour is pinned by tests through the public functions with synthetic fixtures; deliberate one-line mutations make tests fail.
- `uv run pytest -q` passes; ruff check + format --check pass on touched files.
- Pre-existing bugs are documented with a present-tense `Known issue:` comment, not fixed.
