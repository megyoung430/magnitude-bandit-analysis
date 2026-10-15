---
type: bug
complexity: complex
status: in review
---

# Fix choice-probability windows, error bars and multi-recording sessions

Six bugs in the choice-probability analysis, each previously marked with a `Known issue:` comment and pinned by a test.

1. `skip_n_trials_after_reversal=k` removed `k - 1` post-reversal trials and the over-long row then lost its last post trial to padding.
2. The across-subject standard error divided by `sqrt(n_rows)` even where some rows are NaN in a bin.
3. A centred moving-average window longer than a pre/post segment made `np.convolve(mode="same")` return the wrong length (IndexError).
4. `get_choice_probs_by_value` read only `trial_variables[0]`, dropping later recordings of a multi-file session.
5. `get_choice_probs_by_value` silently truncated the three trial lists to the shortest.
6. `average_choice_probs` divided the SE by `sqrt(n_problems)` even where some problems are NaN, and `num_problems` counted problems that yielded no curve.

## Acceptance criteria

- `skip=k` keeps post trial 0 and drops exactly post trials `1..k`; rows have `pre + post - k` bins and keep the last post trial. Docstrings say so.
- SE is the sample SD over non-NaN values per bin divided by the square root of their count; NaN where fewer than 2 values exist. A single row (subject or problem) keeps its documented SE of zero.
- Smoothed output always has the input length; a centred window longer than a segment averages the segment's finite bins.
- All recordings of a session are merged with `merge_trial_variables` (the importer's rule), in recording order; a single recording is used as is.
- Reward-magnitude, chosen-rank and choice lists of different lengths raise `ValueError`, for true and believed binning.
- `num_problems` and `problems_used` count only the problems that supplied at least one curve.
- Nothing else changes: outputs on inputs that trigger none of the bugs are bit-identical to main.

## Results change

Sample data: `data/3x3_field_blocked_reward_bandit/cohort-01/rawdata` (one mouse, MY_72_L, one problem, ses-91 with recordings of 31 and 62 trials). Quantities are cell values of the best / second choice-probability matrices (best value, second value) and offsets of the reversal-aligned curves (pre=3, post=6, 3 good reversals).

| Quantity | main | fixed |
|---|---|---|
| P(best), true binning, cell (4,1) | NaN | 0.905 |
| P(best), true binning, cell (3,1) | 1.000 | 0.733 |
| P(best), true binning, cell (3,2) | 0.727 | 0.692 |
| P(best), true binning, cell (2,1) | NaN | 0.188 |
| P(best), believed binning, cell (3,2) | 1.000 | 0.810 |
| Prev-best P at offset +3, skip=2 | 0.000 | 0.333 |
| Next-best P at offset +1, skip=2 | 0.333 | 0.667 |
| Third P at offset +1, skip=2 | 0.667 | 0.333 |
| Third P at offset +3, skip=2 | 0.333 | 0.000 |
| Smoothed (window 50) curve, centred | IndexError | defined (0.111 pre, 0.556 post) |
| skip=0 curves, window-2 smoothed curve | unchanged | unchanged |

The sample data has one mouse and one problem, so it does not exercise the SE or `num_problems` fixes (SE is zero, `num_problems` is 1 on both). Synthetic illustration: with per-bin values 1, 0 and one NaN, the SE goes from 0.408 (`sqrt(0.5)/sqrt(3)`) to 0.5 (`sqrt(0.5)/sqrt(2)`). The averaged curve changes only through the skip fix (above).
