---
type: infrastructure
complexity: simple
status: in review
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Split the Gaussian-process bandit simulation and plot

Refactor the long functions below into small, single-purpose functions
(principle 4), separating analysis from plotting (principles 2–3), with
identical behaviour.

- **Files:** `src/tasks/gp_bandit.py`
- **Functions over 60 lines:** simulate_3armed_blockswaps_gp (107), plot_example_session_gp (75)
- **Notes:** Simulations are seeded: equivalence means identical outputs for the same seeds/RNG state (keep the exact order of RNG draws). Plotting split into ax-based panels.

## Acceptance criteria

- Every function in scope is at most ~60 lines; the public API (names,
  signatures, defaults, return values, saved file names) is unchanged.
- A before/after golden fingerprint of all public functions in scope is
  identical (hashes reported in your final report).
- New pure functions have synthetic-fixture unit tests, including edge cases.
- `uv run pytest -q` passes; ruff check + format pass on touched files.
- Anything that looks like a pre-existing bug is reported, not silently changed.

## Alignment with `markov_bandit`

The helpers use the same names and shapes as `src/tasks/markov_bandit.py` so the two
task modules read the same way: `draw_block_boundaries(rng, *, T, min_block_len,
max_block_len, n_blocks_draw)`, `swap_arm_roles` / `first_other_active_arm`,
`build_arm_values`, `gap_best_minus_second`, `best_run_lengths`, `session_metrics`,
`build_session`, `summarize_sweep_cell`, `mean_left_to_right`,
`simulate_example_session`, `format_session_title(metrics, *, ...)`,
`plot_session_panel(vals3, boundaries, *, y_max, choices, ax, legend)`, and an
`__all__` plus module docstring listing the entry points. The two modules are not
shared yet; that is a later follow-up. `tests/test_gp_bandit_golden.py` pins values
captured from main (outputs, generator state, a tiny sweep, the `KeyError` path, the
example plot).

Differences from the Markov helpers: `mean_left_to_right` and `summarize_sweep_cell`
take the sweep's `n_sims` as the divisor, so a negative `n_sims` yields `-0.0` and `0`
raises `ZeroDivisionError` (both pinned by tests).

## Remaining differences from main (error paths only)

Return values, generator and global RNG state, printed lines, warnings and exception
types and messages match main on every compared input, including inputs with several
invalid arguments (the seed is validated before `sig` and `max_block_len`, as on
main). One difference remains in the matplotlib state left behind when
`plot_example_session_gp` raises:

- `T=0` with `use_raw=True`: both versions raise the same `ValueError` from
  `np.nanmax` of the empty session, but main has already drawn the three (empty) arm
  lines and set the x-limits on a figure that stays open, while the refactor raises
  while computing `y_max`, before any figure is created.

Behaviour kept from main (see the comments in `swap_arm_roles` and
`sweep_gp_metrics`): non-positive reward values can let the inactive arm win the
argmax at a block boundary (a later `KeyError`), and `max_block_len <= 10` gives
non-positive block lengths without validation.

## How to do it (applies to every refactor item)

Type: `infrastructure` — behavior-preserving refactor, test-first.

1. **Golden snapshot before touching code.** Write a throwaway harness *outside the repo* (scratch dir) that calls every public function in scope on several seeds and parameter settings and saves a canonical fingerprint of the outputs. For plotting functions use the Agg backend, patch `plt.show`, and fingerprint every figure: line x/y data, bar heights, colours, alpha, linewidths, labels, titles, tick labels, legend texts, suptitles, figure size, and the set of saved file names when `save_path` is given. Numbers rounded to 1e-10.
2. **Refactor** following the `delab-coding-practices` principles, using `src/behavior_visualization/plot_num_reversals.py` + `src/behavior_analysis/reversals_by_session.py` as the template:
   - computation moves into small, pure, named functions;
   - plotting functions accept `ax=None` (or `axes=None` for grids) and only create a figure when none is given; saving/showing stays in the public wrapper;
   - no function over ~60 lines; loops with index bookkeeping become comprehensions / vectorised numpy where it reads better (but never change the order of RNG draws);
   - no broad `except`, no silent fallbacks (principle 9); comments explain the present (principle 10).
3. **Public names, signatures, defaults, return values and saved file names stay exactly the same** — scripts and notebooks call them. Do not change behaviour, even where it looks like a bug: flag it in your report instead (and leave a short present-tense comment if it is non-obvious).
4. **Prove equivalence:** re-run the harness; the fingerprint must be identical. Report the before/after hashes and what the harness covered.
5. **Unit tests** (`tests/`, synthetic fixtures only, no real-data paths) for the new pure functions, including edge cases.
6. `uv run pytest -q` passes; `uv run ruff check` and `uv run ruff format --check` pass on every file you touched (don't reformat untouched files).
7. Small commits whose messages explain what and why (principle 10), each ending with:
   `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
