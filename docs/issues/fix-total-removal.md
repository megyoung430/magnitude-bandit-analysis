---
type: bug
complexity: simple
status: in review
---

# Total cumulative-removal curve ignores bad reversals

`total_removal_curves` only saw reversals that were anchors in the windows, and windows are built from good reversals (or, without counters, block changes), so bad reversals never cut a window and total < good + bad. PI decision: a window is cut by the next reversal of any kind, so total = good + bad (+ unclassified) at every offset.

Related bugs in `src/behavior_analysis/cumulative_reversal_removal.py`:

- with 0 anchors the curves reported 0% removed instead of "no data";
- a subject absent from `good_idx` / `bad_idx` was silently treated as having no reversals.

## Blocks-only sessions (PI decision)

A session that logs no good/bad counters (`has_good` and `has_bad` both false) cannot say what kind of reversal a block change is. As on main, every block change in such a session is a reversal for removal purposes. They are tracked as a third kind, **unclassified**:

- `get_unclassified_reversal_indices` gives them per subject, decided per session (a subject can mix counters and blocks-only sessions), on the same merged-trial axis as `get_all_reversal_indices`. `get_reversal_indices_by_kind` bundles good, bad and unclassified.
- Total mode counts them, so total = good + bad + unclassified at every offset.
- Good/bad mode keeps `good_bad_removal_curves` returning good and bad only (anchors first cut by an unclassified reversal are in neither, so they are not mis-attributed to a later good or bad one). `removal_curves_by_kind` returns all three, and `cumulative_removal_series` adds an `"unclassified"` line (grey in the plot) whenever `unclassified_idx` holds any reversal, so they never silently vanish.
- `unclassified_idx=None` (the default) means no session lacks counters. Callers with blocks-only sessions must pass it, otherwise the total reads flat 0 there.

## Acceptance criteria

- `total_removal_curves(..., *, good_idx, bad_idx, unclassified_idx=None)` uses the same reversals as `good_bad_removal_curves`; total equals good + bad + unclassified at every offset (tested on synthetic mixes). `good_idx` / `bad_idx` are required keyword-only arguments.
- `cumulative_removal_series` raises one `ValueError` ("Cumulative removal needs both all_good_idx and all_bad_idx.") in either mode if an index dict is `None`.
- `cumulative_total_events_over_post`, `plot_choice_probs_around_good_reversals` and `add_cumulative_rev_axis` thread the indices through; `fraction_removed_by_problem` takes a required keyword-only `reversals_by_problem` (`{problem: get_reversal_indices_by_kind(...)}`).
- `fraction_removed_by_problem` does not count a problem without anchors in `probs_used`, and names a problem missing from `reversals_by_problem` in a `KeyError`.
- With 0 anchors the fractions are `nan`.
- A subject in `windows` but missing from `good_idx`, `bad_idx` or `unclassified_idx` raises a `KeyError` naming the subject and the dict.
- Tests pin the `_check_fraction` guard (directly and through the public curves).
- `scripts/to_integrate/perf_indep_choice_probs_around_good_reversals.ipynb` builds `reversals_by_problem` in the window cell and passes it to all 10 `fraction_removed_by_problem` calls (cell sources only, not executed).
- Docstrings in the touched modules describe the new behaviour.

## Results change

Sample data (`cohort-01`, 1 subject, 1 session ses-91, blocks only: 0 good, 0 bad, block-change anchors at 22, 53, 86), total fraction removed:

| Offset | main | fixed |
| --- | --- | --- |
| 30 | 0.000 | 0.000 |
| 35 | 0.667 | 0.667 |
| 40 | 0.667 | 0.667 |

Identical to main. (The first version of this fix read a flat 0 here; the unclassified kind restores main's numbers.)

Blocks-only synthetic cohort (6 subjects, 12 block changes each, gaps of 20-49 trials), total fraction removed:

| Offset | 10 | 20 | 30 | 40 | 60 |
| --- | --- | --- | --- | --- | --- |
| main | 0.000 | 0.042 | 0.375 | 0.681 | 0.917 |
| fixed | 0.000 | 0.042 | 0.375 | 0.681 | 0.917 |

Mixed synthetic cohort (3 subjects with counters, 65% good reversals; 3 blocks-only; gaps of 25-69 trials):

| Offset | 10 | 20 | 30 | 40 | 60 |
| --- | --- | --- | --- | --- | --- |
| main total | 0.000 | 0.000 | 0.119 | 0.239 | 0.657 |
| fixed total | 0.000 | 0.000 | 0.149 | 0.284 | 0.746 |

Realistic synthetic example with counters (6 subjects, 14 reversals each, 65% good, gaps of 25-69 trials; windows from good reversals), fraction of anchors removed:

| Offset | main total | fixed total | good | bad |
| --- | --- | --- | --- | --- |
| 20 | 0.000 | 0.000 | 0.000 | 0.000 |
| 40 | 0.104 | 0.333 | 0.104 | 0.229 |
| 60 | 0.396 | 0.729 | 0.354 | 0.375 |

On main the total at offset 60 is below good + bad (0.729); the fixed total equals it.

Curves that change: total-mode removal axes (`use_total=True`) and `fraction_removed_by_problem` on data with bad reversals. Good/bad curves and blocks-only totals are unchanged.
