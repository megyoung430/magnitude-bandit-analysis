---
type: bug
complexity: simple
status: in review
---

# Total cumulative-removal curve ignores bad reversals

`total_removal_curves` only saw reversals that were anchors in the windows, and windows are built from good reversals only, so bad reversals never cut a window and total < good + bad. PI decision: a window is cut by the next reversal of either kind, so total = good + bad at every offset.

Related bugs in `src/behavior_analysis/cumulative_reversal_removal.py`:

- with 0 anchors the curves reported 0% removed instead of "no data";
- a subject absent from `good_idx` / `bad_idx` was silently treated as having no reversals.

## Acceptance criteria

- `total_removal_curves(..., *, good_idx, bad_idx)` uses the same reversal indices as `good_bad_removal_curves`; total equals good + bad at every offset (tested on a synthetic mix). Passing no `good_idx` / `bad_idx` raises `ValueError`.
- `cumulative_removal_series` total mode, `cumulative_total_events_over_post` and `fraction_removed_by_problem` thread the indices through (the last takes keyword-only `good_idx_by_problem` / `bad_idx_by_problem`).
- With 0 anchors the fractions are `nan`.
- A subject in `windows` but missing from `good_idx` or `bad_idx` raises a `KeyError` naming the subject and the dict.
- Docstrings in both modules describe the new behaviour.

## Results change

| Quantity | main | fixed |
| --- | --- | --- |
| Sample data (`cohort-01`, 1 subject, 1 session): max good / bad / total removal | 0 / 0 / 0 | 0 / 0 / 0 |

The sample session has no good or bad reversals in the index lists (3 window anchors from the block-change fallback), so nothing changes there.

Realistic synthetic example (6 subjects, 14 reversals each, 65% good, gaps of 25-69 trials; windows from good reversals), fraction of anchors removed:

| Offset | main total | fixed total | good | bad |
| --- | --- | --- | --- | --- |
| 20 | 0.000 | 0.000 | 0.000 | 0.000 |
| 40 | 0.104 | 0.333 | 0.104 | 0.229 |
| 60 | 0.396 | 0.729 | 0.354 | 0.375 |

On main the total at offset 60 is below the good curve plus the bad curve (0.729); the fixed total equals it.

Curves that change: any total-mode removal axis (`use_total=True`) and `fraction_removed_by_problem`. Good/bad curves are unchanged.
