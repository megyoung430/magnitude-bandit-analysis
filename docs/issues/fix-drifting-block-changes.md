---
type: feature
complexity: simple
status: in review
---

# Drifting-only analyses: block changes, and refuse counter-format data

## Description

The value heatmaps and psychometrics, the block-change-aligned arm analysis and the value-change
analysis are for the drifting task, which has block changes and no good or bad reversals. The
cohort and open-field runners already run them on drifting problems only. The entry points now
also refuse counter-format data themselves, and their figure text speaks of block changes.

## Acceptance criteria

- `run_value_psychometric_analysis`, `run_reversal_aligned_arm_analysis` and
  `run_value_change_analysis` raise a `ValueError` naming the mouse and session, and saying the
  drifting task has only block changes, if any session has good/bad reversal counters
  (`is_counter_session`).
- `run_reversal_aligned_arm_analysis` checks prebuilt windows too: each record's
  `boundary_source` must be `"blocks"`.
- One shared helper per check, in `src/behavior_analysis/reversal_format.py`:
  `check_blocks_only_sessions` and `check_block_change_windows`.
- `build_windows_with_magnitudes` yields `boundary_source == "blocks"` on blocks-only data.
- Figure titles, axis labels, legends and docstrings of the arm analysis (analysis, ranks, plot,
  pipeline) and the value-change modules say "block change": "All Block Changes", "Best to Third
  Block Changes", "Trials from Block Change". Public names, arguments, file names and CSV columns
  are unchanged.
- README and the item docs say these analyses use block changes and run only on drifting problems.

## Decisions

- The session check reads the data in the order given and stops at the first counter-format
  session; it does not need valid session keys, so it cannot fail for a reason unrelated to the
  format.
- Prebuilt windows without a `boundary_source` are refused as well (the builder always writes it),
  rather than treated as unverifiable. A window bounded by `num_blocks` is refused too, since the
  item asks for `"blocks"` exactly.
- Wording was changed only in prose (docstrings, comments, strings, titles); identifiers such as
  `reversal_type`, `num_reversals`, `ReversalAlignedArms` and the `n_reversals` CSV column keep
  their names.
- `reversal_alignment.py` and `get_good_reversal_info` are shared with the blocked-task analyses
  and keep their wording.
- Test fixtures: `make_session`/`make_mouse` take `counters=` (default True, as before); the
  arm pipeline tests use blocks-only sessions.

## Results change

No numbers change. On the sample data (a drifting, blocks-only task) the pooled arm results,
value-change summaries and contamination/bin-count CSVs, and the psychometric heatmap tables, are
byte-identical to the base. Figure text changes: the arm figures say "Block Changes" in the title
and "Trials from Block Change" on the x axis. Counter-format input now raises instead of running.
