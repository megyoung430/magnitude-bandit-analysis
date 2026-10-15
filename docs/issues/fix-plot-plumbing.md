---
type: bug
complexity: complex
status: in review
---

# Fix figure plumbing: stray figure, leaked figures, one set of save/show helpers

The figure modules in `src/behavior_visualization/` each carried their own save/show, despine and colour code, and several leaked open figures or mishandled paths. This item fixes five bugs and folds the duplicates into `plot_helpers.py`.

## Bugs

1. `plot_choice_probs_around_good_reversals` (with a cumulative axis) built a plain figure that was never saved or closed, so a real `plt.show()` displayed it as a duplicate, and saving leaked one open figure per call.
2. A figure that a function created itself stayed open when drawing or saving raised (`save_or_show`, `plot_cohort_comparison`, `plot_trial_block_summaries`, and the panel functions of `plot_averaged_choice_probs`, `plot_first_leave_after_good_reversals`, `plot_num_long_pokes`, `plot_task_statistics`). Figures passed in via `ax=` belong to the caller and must stay open.
3. Duplicated helpers: `plot_helpers.save_or_show` and `write_outputs`; module-private `_save_or_show` in `plot_problem_summary`, `plot_rank_proportions` (`create_folders`), `plot_num_reversals`, `plot_choice_probs_around_good_reversals`; inline copies in `plot_single_session` and `plot_reversals_by_trial`; `_despine` in two modules; `_mouse_color_map` and `MOUSE_COLORS[i % len(...)]`; the global-stream `_jittered`.
4. `plot_num_pokes_across_mice` did not create the parent directory of `save_path`.
5. `plot_first_leave_after_good_reversals` indexed the palette (`MOUSE_COLORS[index]`), so a 17th subject raised `IndexError`.

## Design

All in `src/behavior_visualization/plot_helpers.py`:

- `save_or_show(figures, save_path, *, formats, dpi, make_parent)`: the one function for single-figure wrappers. `figures` may also be `{suffix: figure}` for wrappers that save several figures under one base path (`plot_num_reversals*`, `plot_reversals_by_trial`); they are then shown together in one `plt.show()`. Always closes the figures, also when saving raises.
- `write_outputs(outputs, directory, *, show)`: unchanged API, for the multi-output wrappers (`plot_cohort_comparison`, `plot_trial_block_summaries`). On any exception it closes every figure opened while consuming the lazy builders (the unfinished figure of a builder that raises is not reachable otherwise) and leaves figures that were open beforehand alone.
- `axes_or_new(ax, **subplots_kwargs)`: context manager for panel functions; yields `ax`, or a new figure's axes that is closed if the body raises.
- `palette_color` / `mouse_color_map` (the one colour helper), `despine`, and `jittered(xs, half_width)` (global `np.random` stream, as before).

`save_or_show(make_parent=False)` is general (the path is used as given, any type). `plot_rank_proportions_average` keeps its documented quirk (a `str` path in an existing folder) itself: it closes its figure and raises `TypeError` for a non-`str` path.

`palette_color(index, palette=None)` takes an optional palette. `plot_rank_proportions` passes its module-level `MOUSE_COLORS`, looked up at call time, because `scripts/session_behav_funcimg_trial_summary.ipynb` replaces it through `plot_rank_proportions.__globals__["MOUSE_COLORS"]` to extend the palette (the only notebook that patches a module palette). Names that notebooks star-import (`MOUSE_COLORS`, `Path`) stay importable from the figure modules.

## Acceptance criteria

- Each bug has a regression test that fails on `main` and passes here (see the final report for names).
- Saved file names, dpi, bbox, formats order and figure content are identical to `main`; the only differences are the bug fixes above.
- `uv run pytest -q` passes; `ruff check` and `ruff format --check` pass on every touched file.

## Verification of "nothing else changed"

Compared `main` with this branch, `PYTHONHASHSEED=0`, `SOURCE_DATE_EPOCH=0`, `MPLBACKEND=Agg`:

- A pytest plugin recorded, for every `savefig`, `plt.show` and `plt.close` during the 288 plot tests of `main`: a fingerprint of the figure (every artist's data, colours, labels, limits, ticks, spines, legends, axes positions, size, dpi and a pixels hash), the `savefig` path and keyword arguments, the order of calls, and the figures still open at the end. Run against `main` twice (identical except the unseeded global-RNG jitter in one first-leave test) and against this branch's `src/` with `main`'s tests: 271 of 288 identical; the other 17 are exactly the fixed bugs (the stray figure in 8 choice-probability tests; figures no longer left open after a failure in 3; figures leaked by those that showed up open in 5 later tests) plus the unseeded jitter test.
- 22 scenarios for the wrappers the suite does not save end to end (`plot_num_reversals`, `plot_num_reversals_over_time`, `plot_moving_avg_reversals_over_time`, `plot_reversals_by_trial`, each with no path, an empty path, an existing folder and a nested missing folder): events, fingerprints, sha1 of every written file and open figures are identical.

## Results change

No numbers change. Figure content, file names, dpi and bbox are identical. Differences:

| Quantity | main | fixed |
|---|---|---|
| Figures open when `plt.show()` runs, `plot_choice_probs_around_good_reversals` with a cumulative axis | 2 (plain + cumulative) | 1 |
| Figures left open after saving it | 1 per call | 0 |
| Figures left open after drawing or saving raises (created in the function) | 1 or more | 0 |
| `plot_num_pokes_across_mice(save_path=<missing folder>/x)` | `FileNotFoundError` | folder created, files saved |
| First-leave plot with 17 subjects | `IndexError` | palette cycles |
| `save_path` with a trailing slash, e.g. `plot_num_pokes_across_mice(..., "out/")` | wrote `out/.pdf` and `out/.png` | writes `out.pdf` and `out.png` (the path is normalised through `Path`) |
| `save_path=""` in `plot_metric_by_problem`, `plot_problem_lengths`, `plot_distributions_per_problem_all_mice`, the problem-summary rank figures and `plot_choice_probs_with_fraction_removed_axis` | wrote `..pdf` and `..png` into the working directory | shows the figure, as every other wrapper already did |

Sample data (`data/3x3_field_blocked_reward_bandit/cohort-01/rawdata`, one mouse, one session): `plot_num_reversals`, `plot_num_reversals_over_time`, `plot_moving_avg_reversals_over_time` (with their by-trial companions) and `plot_single_session` wrote 20 files; all 20 are byte-identical on `main` and here. The sample does not exercise the bugs (it saves, never raises, and uses no cumulative-axis choice-probability figure), so only the identical-output claim is shown on it.

## Follow-ups (not changed)

- The jitter in `plot_first_leave_after_good_reversals`, `plot_task_statistics`, `plot_problem_summary` and `plot_rank_proportions` draws from the global NumPy random state (the `Known issue` comments stay), so figures differ between runs unless the caller seeds it. Switching to a passed-in generator would change figures.
- `plot_rank_proportions_average` still does not create its folder and needs a `str` path.
- Figures created by a drawing function that raises mid-draw stay open in `plot_single_session`, `plot_rank_proportions` (`max()` `ValueError`), `plot_num_reversals_over_time` (`fig1` when the by-mouse figure raises) and the problem-summary wrappers.
- `plot_style.py` has an unsorted import (pre-existing).
