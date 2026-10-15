---
type: infrastructure
complexity: complex
status: done
---

<!-- status: todo | in progress | in review | done | blocked | awaiting your confirmation -->

# Split plot_problem_summary.py into analysis + ax panels

Refactor the long functions in `src/behavior_visualization/plot_problem_summary.py` to follow the delab coding principles without changing behaviour.

Targets: plot_problem_lengths (103 lines), plot_rank_proportions_average_without_blocks (102), plot_metric_by_problem (95), plot_distributions_per_problem_all_mice (87), plot_distributions_per_mouse (84).

Template: plot_num_reversals.py. Find the call sites in scripts/ notebooks (read-only) to learn realistic inputs; golden-test on synthetic multi-problem, multi-mouse inputs.

## Method (applies to every refactor item)
Behaviour-preserving refactor, test-first (principle 7):
1. BEFORE touching code, write a capture script in `.golden/` inside your worktree (git-ignored, not committed) using
   /home/claude/golden/goldkit.py (import read-only; `canon`, `capture_figures`, `save`, `compare`). Record return values of every
   target function, and for plotting functions every figure drawn/saved (capture_figures), over inputs that cover
   every branch: synthetic inputs you construct, plus the sample session in
   data/3x3_field_blocked_reward_bandit/cohort-01/rawdata (load with
   `src.behavior_import.load_trials.load_trials_by_problem(root, use_cache=False)`). Save before.json.
2. Refactor. After: the same script must report IDENTICAL via compare().
3. Add pytest unit tests with small synthetic fixtures for each new pure helper (tests/test_<module>.py).

## Acceptance criteria
- Every target function is <= 60 lines (aim for <= 40); helpers are small, named for what they do (principle 4).
- Computation is separated from plotting; panel-drawing functions accept `ax=None` and create a figure only if
  none is given (principle 3). Saving/showing stays in the public function.
- Public function names, signatures, defaults, return values and saved file names are unchanged, so scripts and
  notebooks keep working. Do not edit notebooks or other modules except for import lines.
- Golden capture reports IDENTICAL; include the case count in the report.
- Prefer comprehensions/vectorised operations over index loops where it doesn't change results (principle 5);
  keep explicit loops where state is sequential.
- No new broad `except`; no silent fallbacks (principle 9). Comments explain the present (principle 10).
- If you find a bug, DO NOT fix it: preserve behaviour, add a short comment stating the issue, and report it.
- `uv run --no-sync ruff check <touched and new files>` clean; `uv run --no-sync ruff format` only touched/new files.
- `uv run --no-sync pytest -q` passes in your worktree.
- Commit on your branch in your worktree only, small commits, messages per principle 10. Set
  `status: in review` in this file in your final commit.


## Review follow-up

- The pure analysis helpers moved to `src/behavior_analysis/blocklen_distributions.py`
  (`collect_blocklens`, `drop_last_problem_per_mouse_raw`, `pooled_blocklens`) and
  `src/behavior_analysis/rank_proportions_summary.py` (`mean_rank_proportions_by_subject`,
  `rank_mean_and_se`). `collect_blocklens` and `drop_last_problem_per_mouse_raw` are still
  importable from `plot_problem_summary`, which `scripts/to_integrate/testing_code.ipynb` uses.
- Tests go through the public figure functions; the private helpers are not called directly.

## Known issues (behaviour kept, not fixed)

- `rank_mean_and_se` divides the standard deviation (which skips NaN mice) by `sqrt(len(per_mouse))`
  (which counts them), so the SE is understated when any mouse is NaN, and NaN for a single mouse.
- With `drop_last_problem=True` leaving nothing, `plot_distributions_per_problem_all_mice` raises
  matplotlib's "Number of rows must be a positive integer, not 0", while
  `plot_distributions_per_mouse` silently draws nothing.
- Bar charts fix the y-axis bottom at 0, so bars with negative means are clipped.
- The rank p-value labels anchor on the i-th finite bar, so with an earlier NaN mean a label sits on
  a different rank than its text names.
- `testing_code.ipynb` calls the distribution plots with `drop_last_problem_per_mouse=`, which is not
  a parameter of either function (it is `drop_last_problem`); unchanged from before the refactor.

## Differences from the original on malformed input

Equal to the original, including the random state, warnings, exceptions, files and open figures,
for everything the repo's producers generate. The malformed-input differences that remain:

- `plot_metric_by_problem` and `plot_problem_lengths` with a summary that lacks `"ns"` or `"mice"`
  (or `"per_mouse_blocklens"`): still a `KeyError` before any figure exists, but its message is now
  `summary is missing 'ns'` instead of the bare `'ns'`. `"ns"` is still required although nothing
  reads it, because the original did.

- `plot_problem_lengths`: the original wrote value labels for `zip(x, means, ns)`, so a `"ns"` shorter
  than `"means"` produced fewer labels; the labels now follow `"means"` only. The producers always
  return `ns` as long as `blocks`.
- `plot_rank_proportions_average_without_blocks` with `average_across_mice_pvalues` that is not a
  dict (e.g. a list): the same `AttributeError`, but the figure left open by the failure has more of
  the panel drawn than before (the original failed straight after the bars).
- `plot_rank_proportions_average_without_blocks` with `rank_counts_by_good_reversal` that is not a
  dict (e.g. a list): the `AttributeError` message names `items` instead of `keys`. No figure is left
  open, as before.
- `plot_rank_proportions_average_without_blocks` with rows that are not dicts (e.g. `{"a": [1]}`):
  the same `AttributeError`, but no figure is left open; the original created the figure before
  reading the rows and left it open.
