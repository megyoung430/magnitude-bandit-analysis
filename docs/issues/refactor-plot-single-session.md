---
type: infrastructure
complexity: simple
status: in progress
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Split plot_single_session into ax-based panels

Refactor the long functions below into small, single-purpose functions
(principle 4), separating analysis from plotting (principles 2–3), with
identical behaviour.

- **Files:** `src/behavior_visualization/plot_single_session.py`
- **Functions over 60 lines:** plot_single_session (191)
- **Notes:** Fingerprint on the sample session and a synthetic session; keep the saved file names.

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

1. **Golden snapshot before touching code.** Write a throwaway harness *outside the repo* (scratch dir) that calls every public function in scope on synthetic inputs (and the sample session in `data/3x3_field_blocked_reward_bandit/cohort-01/rawdata`) and saves a canonical fingerprint of the outputs. For plotting functions use the Agg backend, patch `plt.show`, and fingerprint every figure: line x/y data, scatter offsets, bar heights, colours, alpha, linewidths, markers, labels, titles, tick positions and labels, axis limits, legend texts, suptitles, figure size, and the set of saved file names when `save_path` is given. Numbers rounded to 1e-10.
2. **Refactor** following the `delab-coding-practices` principles, using `src/behavior_visualization/plot_num_reversals.py` + `src/behavior_analysis/reversals_by_session.py` as the template:
   - computation moves into small, pure, named functions (in `src/behavior_analysis/` when it is analysis that plotting code was doing inline);
   - plotting functions accept `ax=None` (or `axes=None` for grids) and only create a figure when none is given; saving/showing stays in the public wrapper;
   - no function over ~60 lines; loops with index bookkeeping become comprehensions / vectorised pandas-numpy where it reads better;
   - no broad `except`, no silent fallbacks (principle 9); comments explain the present (principle 10).
3. **Public names, signatures, defaults, return values and saved file names stay exactly the same** — scripts and notebooks call them. Do not change behaviour, even where it looks like a bug: flag it in your report instead (and leave a short present-tense comment if it is non-obvious).
4. **Prove equivalence:** re-run the harness; the fingerprint must be identical. Report the before/after hashes and what the harness covered.
5. **Unit tests** (`tests/`, synthetic fixtures only, no real-data paths) for the new pure functions, including edge cases.
6. `uv run pytest -q` passes; `uv run ruff check` and `uv run ruff format --check` pass on every file you touched (don't reformat untouched files).
7. Small commits whose messages explain what and why (principle 10), each ending with:
   `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
