---
type: feature
complexity: simple
status: in review
---

# Rank-proportion plots: grids off by default, two single-panel averages

## Description

`plot_rank_proportions` made four-plus figures per call by default. Now it makes one by default: the across-mice average (with p-values and pooling). The two panel grids and the across-blocks average are opt-in, and the old two-panel average figure is no longer made by the dispatcher.

## Acceptance criteria

- `plot_rank_proportions(by_mouse=False, by_block=False, average_by_mouse=True, average_by_block=False)` are the defaults; `average` is gone.
- New public wrappers `plot_rank_proportions_average_by_mouse` and `plot_rank_proportions_average_by_block` draw `plot_average_across_mice` / `plot_average_across_blocks` into a 6.5 x 5.5 figure and save with `plot_helpers.save_or_show` (str or `Path`, parent folder created).
- Saved names: `<save_path> by Average Across Mice` and `<save_path> by Average Across Blocks`.
- The new figures draw the same content as the matching panel of `plot_rank_proportions_average` (bars, error bars, lines, mean labels, p-values, title, labels, legend).

## Decisions

- `average` was used by callers (two notebooks, five calls), so it was removed and the callers were updated; no compatibility shim. Calls that passed `average=True`/`RANK_PLOT_AVERAGE` asked for the combined figure, so they now pass both `average_by_mouse` and `average_by_block` and still get both panels (as two files, or two shown figures).
- `plot_rank_proportions_average` stays public and unchanged (still `str`-only, no folder creation).
- The names `by Average Across Mice/Blocks` follow the dispatcher's `<save_path> by <view>` pattern.
- A `Path` `save_path` is now accepted by the dispatcher too (it builds names with an f-string); before it raised `TypeError` for a `Path`.
- Callers using only defaults (`src/pipelines/open_field.py`, `scripts/full_pipelines/*.py`, `scripts/individual_analyses/plot_rank_proportions.py`, cohort-1/cohort-3 notebooks, `to_integrate` notebooks) get the new defaults. `plot_cohort_comparison` does not call the dispatcher.
- Callers that pass `by_mouse=`/`by_block=` explicitly (funcimg notebook) keep them.

## Results change

Per dispatcher call with default flags:

- No longer produced: `<name> by Mouse`, `<name> by Block`, `<name> by Average` (.pdf and .png each).
- New: `<name> by Average Across Mice` (the old left panel, 6.5 x 5.5 instead of half of a 13 x 5.5 figure).
- Opt-in: `<name> by Average Across Blocks`.
- Unchanged: calls in `scripts/session_behav_funcimg_trial_summary.ipynb` and `scripts/to_integrate/value_bandit_analysis_2.ipynb` that asked for `average=True` still get both average panels (now as two figures instead of one) and keep their explicit grid flags.
