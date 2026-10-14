---
type: infrastructure
complexity: complex
status: in review
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Split run_paper_sweep and make_backend in paper_rl_fitting.py

> **WARNING: merging this changes every sweep's output-directory hash.**
> The manifest embeds the full source of `paper_rl_fitting.py` (`implementation`),
> and the directory name is a hash of the manifest, so any edit to that file —
> including this refactor — moves a sweep to a new folder. Partially-run sweeps
> started before the merge will **restart from zero in a new folder rather than
> resume**; finished results stay readable in the old folder. This is the existing
> design and is deliberately not changed here. Follow-up item: a narrower cache key
> (settings + data fingerprint + library versions + an explicit
> `FIT_ALGORITHM_VERSION`) so cosmetic edits stop invalidating caches.
>
> Known quirks kept as-is and now documented in the code: `_weight_decay` prefers
> `weight_decay` over the legacy `l2_weight` and silently falls back to 0.0, and the
> RL objective's `l2_weight` argument is unused (the trainer passes 0.0).

Refactor the long functions below into small, single-purpose functions
(principle 4), separating analysis from plotting (principles 2–3), with
identical behaviour.

- **Files:** `src/agents/paper_rl_fitting.py`
- **Functions over 60 lines:** run_paper_sweep (158), make_backend (75)
- **Notes:** Covered partly by tests/test_paper_rl_fitting.py and test_neural_bandit.py (JAX extra installed). Keep RNG draw order and the on-disk output layout (manifest/results files) identical.

## Acceptance criteria

- Every function in scope is at most ~60 lines; the public API (names,
  signatures, defaults, return values, saved file names) is unchanged.
- A before/after golden fingerprint of all public functions in scope is
  identical (hashes reported in your final report).
- New pure functions have synthetic-fixture unit tests, including edge cases.
- `uv run pytest -q` passes; ruff check + format pass on touched files.
- Anything that looks like a pre-existing bug is reported, not silently changed.

## How to do it (applies to every refactor item)

Type: `infrastructure` — behavior-preserving refactor, test-first.

1. **Golden snapshot before touching code.** Write a throwaway harness *outside the repo* (scratch dir) that calls every public function in scope on synthetic inputs and saves a canonical fingerprint of the outputs (for run_paper_sweep: a tiny seeded sweep into a temp dir; fingerprint every file it writes plus the return value). Numbers rounded to 1e-10.
2. **Refactor** following the `delab-coding-practices` principles, using `src/behavior_visualization/plot_num_reversals.py` + `src/behavior_analysis/reversals_by_session.py` as the template for style: computation in small, pure, named functions; no function over ~60 lines; loops with index bookkeeping become comprehensions where it reads better; no broad `except`, no silent fallbacks (principle 9); comments explain the present (principle 10).
3. **Public names, signatures, defaults, return values and saved file names stay exactly the same** — scripts and notebooks call them. Do not change behaviour, even where it looks like a bug: flag it in your report instead (and leave a short present-tense comment if it is non-obvious).
4. **Prove equivalence:** re-run the harness; the fingerprint must be identical. Report the before/after hashes and what the harness covered.
5. **Unit tests** (`tests/`, synthetic fixtures only, no real-data paths) for the new pure functions, including edge cases.
6. `uv run pytest -q` passes; `uv run ruff check` and `uv run ruff format --check` pass on every file you touched (don't reformat untouched files).
7. Small commits whose messages explain what and why (principle 10), each ending with:
   `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
