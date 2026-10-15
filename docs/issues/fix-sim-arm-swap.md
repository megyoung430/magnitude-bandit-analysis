---
type: bug
complexity: simple
status: in review
---

# Fix arm-role swap with non-positive values and unvalidated block lengths

Defects in the seeded task simulations `src/tasks/gp_bandit.py` and
`src/tasks/markov_bandit.py`.

1. `swap_arm_roles` took the argmax over all three arms. The inactive arm pays 0,
   so with non-positive reward values it could win, be listed as active again,
   share a stream with another arm, and cause a `KeyError` at a later swap.
   Roles are now chosen strictly among the active arms (`max` over
   `sorted(active_arms)`, so ties go to the lowest index even when `-inf` values
   are involved or `active_arms` is out of order).
2. The session metrics (`best_minus_second`, `best_arm_series`, run lengths and
   the Markov switch counts) took best and second-best over all three arms, so
   the inactive arm's 0 counted whenever an active value was non-positive. They
   are now computed over the two active arms only (`build_arm_roles` returns each
   trial's inactive arm; `active_arm_values` masks it to `-inf`).
3. `min_block_len` defaults to `max_block_len - 10`, which is non-positive for
   `max_block_len <= 10`. `simulate_3armed_blockswaps_*` now raises `ValueError`
   unless `1 <= min_block_len <= max_block_len`, before any random draw, and the
   sweeps validate every `max_block_lens` entry (must exceed 10) before running
   or printing anything.
4. `first_other_active_arm` is renamed `remaining_active_arm` and lost its
   `old_inactive` argument and fallback, which only the defect could reach.

## Why metrics over active arms (not a `ValueError` for non-positive values)

Raw GP values (`use_raw=True`) go negative with the default `shift=2.5`,
`scale=2.0`, so raising on non-positive values would make the default raw
configuration unusable. Comparing the two active arms is the quantity the metrics
are defined for and equals the old definition whenever active values are positive
(the inactive 0 is then always lowest), so positive-value results are unchanged.

## Acceptance criteria

- With negative reward values the simulation runs; at every boundary the best
  active arm becomes inactive and the metrics never use the inactive arm
  (`test_non_positive_values_assign_roles_among_active_arms` in both golden test
  files; unit tests in `tests/test_gp_bandit.py` and `tests/test_markov_bandit.py`,
  including all-`-inf` active values and out-of-order `active_arms`).
- Bad `min_block_len` and bad sweep `max_block_lens` raise `ValueError` before any
  draw or print.
- For positive values, outputs and generator state are bit-identical to `main`.
  The existing golden tests pass unchanged, except the three raw-GP goldens
  (below), whose expected values were regenerated.

## Results change

Simulations only; no real-data path uses these modules (only the tests and a
notebook in `scripts/to_integrate`), so the sample data does not exercise them and
nothing in the real-data analysis changes.

**Default `use_raw=True` GP runs change.** Raw values are frequently negative with
the default shift, so on `main`:

- some runs crashed with `KeyError` (roles collapsed after the inactive arm won
  the argmax), so earlier raw-GP sweeps could crash outright; and
- nearly every run that did complete used the inactive arm's 0 as best or
  second-best, so `gap_mean` and `frac_gap_ge2` were biased low.

Measured on 300 seeded sessions per cell (T=150, `min_block_len = max_block_len
- 10`, `use_raw=True`, default shift and scale), main versus fixed:

| max_block_len, sig | main KeyError | runs with different metrics | gap_mean main / fixed | frac gap>=2 main / fixed |
|---|---|---|---|---|
| 20, 4 | 4.7% | 100% | 2.050 / 2.252 | 0.447 / 0.481 |
| 20, 9 | 6.7% | 96% | 2.054 / 2.243 | 0.449 / 0.480 |
| 50, 4 | 0.7% | 100% | 2.071 / 2.257 | 0.451 / 0.481 |
| 50, 9 | 2.3% | 96% | 2.063 / 2.233 | 0.449 / 0.476 |
| 80, 4 | 0.0% | 100% | 2.075 / 2.258 | 0.452 / 0.481 |
| 80, 9 | 0.0% | 96% | 2.041 / 2.234 | 0.445 / 0.476 |

"Runs with different metrics" counts crashed runs and completed runs whose four
metrics differ; mean columns use only runs that completed on main. The gap metrics shift by about +0.2 (mean) and +0.03 (fraction >= 2) in these cells.
In the small golden raw
sweep (`n_sims=3`) the gap-mean heatmap goes from `[[1.906, 1.582], [2.719, 2.547]]`
to `[[2.020, 1.764], [2.787, 3.250]]`, with the run-length heatmaps identical.

**Markov runs whose `val_list` contains 0 also change when
`enforce_distinct=False`.** The two active arms can then share a value, and an
active 0 used to tie with the inactive arm's 0 (counted as a non-unique maximum in
the switch counts, and as a tied best arm in the series). Only active arms are
compared now, so such sessions can give different gap, run-length and switch
metrics. With `enforce_distinct=True` and a positive `val_list` nothing changes.

Non-finite inputs (`val_list` for Markov; `sig`, `scale`, `shift`, `jitter` for GP)
now raise `ValueError` before any draw instead of returning `gap_mean = nan`.

Integer-valued GP runs (`use_raw=False`, clip `(1, 5)`) and Markov runs with
positive `val_list` are unaffected: 40 seeds x `max_block_len` in {12, 20, 40},
GP (integer and raw with a positive shift) and Markov, including sessions and the
final generator state, are byte-identical to `main`. Runs with
`max_block_len <= 10` and the default minimum, previously accepted with
meaningless block lengths, now raise `ValueError`.
