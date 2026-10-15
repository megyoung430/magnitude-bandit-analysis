---
type: bug
complexity: simple
status: in review
---

# Align choice-probability rows to the reversal when the pre window is short

`get_choice_probs_around_good_reversals` built each row as `pre + post` concatenated from the left. A reversal whose pre window held fewer than `pre` trials (a reversal near the start of the data) therefore had its pre values in the leftmost bins and its post values shifted into the pre bins: every point sat at the wrong offset. A pre window longer than `pre` spilled into the post bins in the same way.

## Acceptance criteria

- Pre values are right-aligned: the last pre trial is at offset -1. A pre window longer than `pre` keeps its last `pre` trials.
- Post values start at offset 0; skip semantics are unchanged (trial 0 kept, trials `1..skip` dropped), and the row keeps `post - skip` post bins.
- Positions a window does not cover are NaN and ignored by the means.
- Other row-building code in `src/behavior_analysis` and `src/behavior_visualization` is checked for the same bug.

## Decisions

- Only `_reversal_choice_rows` changes. `_pad_to_length` is replaced by `_right_aligned` (pre) and `_left_aligned` (post); the row is the concatenation of two fixed-length segments, so one segment can never leak into the other.
- The pre window is clipped only at the start of the data (`_window_indices` does not stop at the previous boundary), so a short pre window happens only for a reversal within `pre` trials of the data start. A short post window (cut by the next boundary or the data end) was already positioned correctly and keeps its NaN tail.

## Audit of other row builders

| Code | Builds fixed-length rows from windows? | Verdict |
|---|---|---|
| `get_first_leave_after_good_reversals` | No: scans the post list from index 0 (the reversal trial) | Unaffected; pinned by test |
| `get_rank_counts_by_good_reversal` (true and believed) | No: sums the post window | Unaffected; pinned by test |
| `cumulative_reversal_removal` | No: works on the `x` axis and removal delays | Unaffected |
| `average_choice_probs` | Delegates to `get_choice_probs_around_good_reversals` | Fixed through it |
| `apply_moving_average_to_choice_probs`, `remove_trials_after_bad_rev` | Operate on finished rows / cut post lists | Unaffected |
| `get_choice_probs_by_value`, `reversals_by_trial`, `split_*`, plotting modules | Do not read pre/post lists | Unaffected |

## Tests

- `tests/test_get_choice_probs_around_good_reversals.py`: `test_get_choice_probs_right_aligns_a_short_pre_window` (3-trial pre, `pre=10`), `..._short_pre_does_not_move_the_post_values`, `..._reversal_at_data_start_has_empty_pre_bins`, `..._keeps_the_last_trials_of_a_pre_window_longer_than_the_axis`, `..._short_pre_and_short_post_with_skip`. All five fail on main.
- `tests/test_short_window_independence.py`: first-leave and rank counts give the same result for pre lengths 0, 3 and 10.

## Results change

- Any figure or table that uses `get_choice_probs_around_good_reversals` (also through `average_choice_probs`) changes only if some reversal lies within `pre` trials of the start of the data it was cut from; for those reversals the pre and post points move to the right offsets. Reversals with a full pre window are bit-identical.
- Sample session (MY_72_L, reversals at 22/53/86, all with 10 pre trials): outputs are bit-identical to main for `(pre, post) = (10, 30)` and `(10, 40)`.
- Synthetic sessions (gap 15-40, `pre=10`, `post=30`, 8 mice x 10 sessions): with the first boundary also 15-40 trials in, no pre window is short (0%) and curves are bit-identical; 61% of post windows are short (cut by the next boundary) and were already correct. If sessions may start mid-block (first boundary 1-40), 1.7% of reversals have a short pre window and the mean curves move by at most 0.010 (prev-best, at offset +2) and 0.006 (third).
