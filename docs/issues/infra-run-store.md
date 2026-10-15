---
type: infrastructure
complexity: complex
status: in review
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Name resumable run directories by algorithm version and configuration, not source text

`cohort_fitting.py`, `parameter_recovery.py` and `paper_rl_fitting.py` named their
resumable output directories by hashing source text, so any edit (even a comment or a
reformat) restarted in-progress runs from zero. The three modules also duplicated four
helpers.

## What changed

- New `src/agents/run_store.py` with `write_atomically` (plus `write_json_atomically`),
  `manifest_dir` (identity hash names the directory; provenance is written but not
  hashed), `fingerprint_sequences` and `at_bound`. All three modules use it.
- `at_bound` treats a `None` limit as unbounded. `parameter_recovery` used to raise
  `TypeError` on it; the `Known issue:` comment there is gone.
- Each module has `ALGORITHM_VERSION = 1` with the comment "Bump when a change alters
  results; comments and refactors don't need a bump."
- The directory hash covers the version and the configuration: data fingerprint,
  settings, model names, bounds, the at-bound tolerance, and (paper-style sweeps) jax,
  optax and numpy versions. In `parameter_recovery` the fit and loss function names are
  part of the configuration. Source text stays in `manifest.json` (`implementation`,
  `model_implementation`, `value_update_implementation`, `neural_implementation`,
  `rl_implementations`, `code`) as provenance, outside the hash.
- Docs: README section "Resumable cohort cross-validation and recovery runs" rewritten;
  `docs/published_model_alignment.md` and the two earlier items' warnings updated.

## Acceptance criteria

- Editing source text (the `edit_module_text` fixture) no longer changes a directory, and
  a rerun resumes without refitting (all three modules).
- Changing `ALGORITHM_VERSION`, a setting, or a module constant (the at-bound tolerance)
  changes the directory.
- `at_bound` handles a `None` limit as unbounded in both modules (test in
  `tests/test_run_store.py` and a recovery run with an open bound).
- Numerical results and output file bytes are identical to main; only directory names, the
  manifest's hash fields and the `at_bound` `None` case differ.

## Migration

This change moves every existing directory one last time.

- Runs started on main (cohort cross-validation, recovery, paper-style sweeps and the
  `problem-cv-*` aggregate folder, whose name hashes its fold directories) restart from
  zero in a new folder after merging. Finished results stay readable in the old folders;
  nothing is deleted or rewritten.
- To avoid losing progress, finish or pause interrupted runs on main before merging, or
  accept the restart. A finished old folder can be kept as the record of that run.
- After this change only a version bump or a configuration change moves a directory.
  Bump `ALGORITHM_VERSION` in the module whose results changed. For `cohort_fitting` that
  includes edits to `best_rl.py` and `released_rl.py`; for `paper_rl_fitting`, edits to
  `neural_bandit.py`, `best_rl.py` and `released_rl.py`; for `parameter_recovery`, edits to
  the task or mouse. These modules are no longer in the hash, so forgetting a bump after
  such an edit resumes stale results.
- Descriptive strings in the manifest (for example the paper sweep's `objective` and
  `initialization` text) are still hashed: editing them moves directories.

## Verification

Synthetic runs of all three modules (cohort: sticky, best, released; recovery; paper
sweep with sticky, best and a neural model; pooled-problem sweep) on main and on this
branch, compared by output file bytes (timing columns removed) and `float.hex` of every
numeric field: all identical. The manifests differ only by the added hash fields
(`algorithm_version`, `at_bound_atol`, and in recovery `fit_function`/`loss_function`) and
the edited module's own source snapshot.
