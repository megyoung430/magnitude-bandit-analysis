---
type: infrastructure
complexity: simple
status: in review
---

# Split the remaining long plotting functions

Refactor the long functions below into small, single-purpose functions (principle 4), separating analysis from plotting (principles 2-3), with identical behaviour.

- **Files:** `src/behavior_visualization/{plot_averaged_choice_probs,plot_num_long_pokes,plot_cohort_comparison,plot_task_statistics,plot_trial_block_summary,plot_first_leave_after_good_reversals}.py`
- **Functions over 60 lines:** plot_choice_probs_with_fraction_removed_axis (84), average_across_problem_curves (67), plot_num_pokes_across_mice (131), plot_cohort_comparison (130), plot_block_lengths (94), plot_trial_block_summaries (85), plot_first_leave_after_good_reversals (74)

## Acceptance criteria
- Every function in scope is at most ~60 lines of code; public names, signatures, defaults, return values and saved file names unchanged.
- An EXACT before/after comparison against main is identical (see below).
- Each public figure function has at least one characterization test through the public function (stub plt.show / capture the figure) that pins appearance: figure size, axes count, titles/labels, tick labels, line/bar data, colours, legend texts, saved file names and dpi; deliberate one-line mutations (colours, offsets, ylims, labels, SE denominator) make tests fail.
- `uv run pytest -q` passes; ruff check + format --check pass on touched files.
- Pre-existing bugs are documented with a present-tense `Known issue:` comment, not fixed.

## How
1. Exact comparison FIRST: archive main into scratch and compare main vs branch (PYTHONHASHSEED=0, Agg, SOURCE_DATE_EPOCH=0, `plt.show` patched, no float rounding): every artist, titles, labels, ticks, limits, legends, spines, axes positions after tight_layout, figure size, rendered pixels hash, savefig calls and file bytes, show calls, figures left open (also after exceptions), warnings, exception type and message, return values, input mutation.
2. Refactor following the delab principles and the template (`plot_num_reversals.py` + `reversals_by_session.py`): pure computation moves into `src/behavior_analysis/`, panels accept `ax=None` (keyword-only), saving/showing only in the public wrapper, keyword-only arguments for easily-swapped parameters, `_`-prefixed helpers and styling constants, type hints, no broad except/silent fallbacks, present-tense comments. `plot_averaged_choice_probs.py` keeps importing `cumulative_total_events_over_post` from `plot_choice_probs_around_good_reversals.py`.
3. Identical behaviour, even where it looks like a bug; document with `Known issue:`.
4. Characterization tests + mutation check (~15 one-line mutations, each reverted).
5. Small commits explaining what and why.
