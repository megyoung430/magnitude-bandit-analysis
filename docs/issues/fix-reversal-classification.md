---
type: bug
complexity: simple
status: in review
---

# Fix reversal classification bugs

Bugs in `get_good_reversal_info.py`, `get_bad_reversal_info.py` and the first-leave analysis.

1. `classify_towers_at_good_reversals`: when `prev_best == next_best` the third tower came from
   a set difference, so its identity depended on `PYTHONHASHSEED`. The same copy lived in
   `get_first_leave_after_good_reversals.py`.
2. `get_good_reversal_info` treated every block change as a good reversal for a session that logs
   bad reversals but no good ones, so bad reversals were counted as good.
3. `get_bad_reversal_info`: all reversal records of a subject shared one `towers` list object.
4. `get_bad_reversal_info`: an explicit `post` could run past the next reversal boundary.

## Decisions

- **Best tower unchanged (PI decision): drop everywhere.** A good reversal should change the
  best tower, so a reversal where it does not is a suspected bug to investigate, and no "third"
  tower is invented for it.
  - `get_good_reversal_info` drops such reversals once, at the source, so rank proportions, the
    early/late split (which therefore splits only valid reversals), the best-change split, the
    first-leave counts, the choice probabilities and every other consumer of the windows see
    only valid reversals.
  - One `ReversalWarning` per call lists every dropped reversal (subject, session, session-local
    trial, recording run, merged trial, reason) and gives their count.
  - A tie for best (before or after) is also dropped and is reported as "tie for best", so it is
    not mistaken for an off-by-one.
  - **Dropped reversals still end their neighbours' windows.** They are real block or counter
    changes in the data even if mislabelled, and this matches how a reversal skipped by the
    zero-reward gate already behaves. Without this, the window of the reversal before a dropped
    one would run into the next block.
  - `classify_towers_at_good_reversals` returns `(prev_best, next_best, None)` when the best
    tower is unchanged; the earlier "first tower in session order" rule is removed. The two
    classifying analyses (`get_first_leave_after_good_reversals`,
    `get_choice_probs_around_good_reversals`) also exclude such records and warn with the count
    (`exclude_best_unchanged`), which only matters for hand-built windows and the first-block
    record.
  - The record prepended by `include_first_block=True` always has `prev_best == next_best`
    (trial 0 supplies both "before" and "after"). The first block is not a reversal, so it has
    `first_block=True` and is never dropped or warned about. The two classifying analyses drop it
    silently (no third tower); no caller combines `include_first_block=True` with them.
  - Records now also carry `subject`, `session` and `session_trial`.
  - **Missing magnitudes are dropped, not errors.** The importer represents a missing magnitude
    as `None` in three ways: `{}` for a trial with no `curr_rew_mag` print, a tower listed in
    `c_tows` but absent from the magnitude dict (`None` throughout), and arrays that end before
    the trial. A reversal with a missing magnitude on either side is dropped with the same
    `ReversalWarning` and the reason "missing magnitude", and still ends its neighbours'
    windows. Each representation has a test. The classifier, given a record with a `None`, still
    raises `ValueError` naming subject, session and trial (records from `get_good_reversal_info`
    never contain one, except the first-block record).
  - **Logged totals stay as the task logged them; suspect counts sit beside them.**
    `get_total_reversals` returns `suspect_good_reversals` and `uncheckable_good_reversals` next
    to the unchanged `good_reversals`/`bad_reversals`/`total_reversals`; they reach
    `reversal_counts_by_session`, `reversal_totals_by_subject` and the cohort-comparison CSV rows
    as new columns, and `compute_suspect_good_reversals_per_session` (problem_summary) gives
    per-subject averages beside the unchanged `compute_avg_good_reversals_per_session`. A
    within-session increment of the good counter is suspect when `best_change_problem` (the
    rule `get_good_reversal_info` uses) finds a problem between the previous trial and the
    increment trial. Reversals between runs (`between_runs`) and terminal events (`run_end`)
    have no trial to compare and are counted as uncheckable, as are the increments of a session
    with no reward magnitudes. The rule lives in `best_change.py`, the counting in
    `suspect_reversals.py`.
  - **Early/late split:** stays "the first N valid reversals" (PI), because the suspect ones are
    dropped before the split. Documented in the split's docstring.
  - **Length check:** `get_good_reversal_info` raises `ValueError` naming the subject when
    `good_reversals`, `blocks` or a reward array is not as long as the merged `trial` list
    (empty means "not logged"). Merging by position silently shifts or cuts trials when sessions
    have different formats or tower sets. This also covers the case of reward arrays shorter than
    the trials: through the public function that is now an error, not a dropped reversal.
  - **Counters restart at each session** (PI, matching the rig). The merge in
    `get_vars_across_all_sessions` relies on it; documented there and in
    `get_good_reversal_info`.
  - Importing `get_good_reversal_info` sets the `"always"` filter for `ReversalWarning`;
    stated in the module docstring.
- **Warnings.** `ReversalWarning` (a `UserWarning`) is registered as `"always"` so a rerun
  shows it again. The warnings are raised in the public functions, so `stacklevel` points at the
  caller. The existing `print("[WARN]")` lines are untouched.
- **Bad-only sessions (per session).** A session is bad-only when `has_bad` is set and
  `has_good` is not. In the block-change fallback (no good counter anywhere in the subject) a
  bad-only session's block changes get no record, but they stay in the boundary list and still
  end the window of the reversal before them (a legacy reversal followed by a bad-only session
  has post-window 5..10, as on main, not 5..15). A legacy session of the same subject keeps its
  block changes. A `ReversalWarning` names the subject and session of every bad-only session.
  When the subject has a good counter on some session, the good counter is used as before and
  sessions without it contribute nothing.
  - The block-change fallback is deliberate for legacy sessions that have neither counter
    (public docstring; `test_good_reversals_counter_takes_priority_over_block_changes`).
  - `get_total_reversals` (~30) and `extract_trials` (~191, ~219) were checked and are correct:
    `get_total_reversals` only falls back to `max(blocks) - 1` when no session logged either
    counter, and then returns no good/bad split; `drop_unlogged_reversal_vars` removes counters a
    session never logged.
- **`include_first_block=True` for bad-only sessions: no record.** When trial 0 belongs to a
  bad-only session and the subject has no good counter, no first-block record is made. The
  alternative, a record whose post window is clipped at the first bad reversal, would need bad
  reversals as boundaries inside the good-reversal code; with no record the behaviour matches
  "a bad-only session has no good reversals" and the docstring. A legacy first session keeps its
  first-block record.
- **Bug 4 was a real inconsistency.** `get_good_reversal_info` never lets the post-window cross
  the next boundary (explicit or default, tested), and the bad docstring calls the window "to
  the next bad reversal". Fixed to match.
- `session_order` (the `ses-<n>` sort key) is a module-level function of
  `get_variables_across_sessions` so the session spans use exactly the merge order.

## Open investigation for the PI: why does the best tower not change?

Question: on a good-reversal trial, could the reward dict still hold the old values, so that
"after" repeats "before"?

What the sample raw files show (`cohort-01/rawdata/sub-05_id-MY_72_L/ses-91_date-20260330/`;
both files have `task_file_hash` 2651649461). This is a drifting-value task: magnitudes drift
within blocks. There are no good/bad counters, so no good reversals.

- Each trial has one `variable print` line, written in state `poked_in`, immediately before the
  `choice` state in which the reward pulses are delivered. It holds `num_t`, `num_blocks`,
  `curr_rew_mag`, `num_rew_pulses` and the chosen tower.
- There are three block changes. `MY_72_L-2026-03-30-120049.tsv` has two: trials 22 to 23
  (`num_blocks` 1 to 2) and 55 to 56 (2 to 3). `MY_72_L-2026-03-30-115657.tsv` has one: 22 to 23.
- At each, the print of the first trial of the new block already has the new `curr_rew_mag`, and
  `num_rew_pulses` for that trial matches the new magnitude of the chosen tower. The best and
  worst towers swap and the middle one stays: for example `{B1: 0, A1: 2, C1: 3}` becomes
  `{B1: 3, A1: 2, C1: 0}` (120049, trial 22 to 23). So `reward_magnitudes_by_tower[t]` on the
  first trial of a block is the state after the swap, in step with the block counter, and the
  code's `before = idx - 1`, `after = idx` is correct for block-based boundaries. No off-by-one
  was found, and nothing was changed.
- Not settled: the order for the `num_good_reversals` counter itself. The sample has no good
  counter, so it cannot show whether the counter increments on the same trial as the magnitude
  swap or one trial later (which would make `after` at the counter increment still show the old
  values). Other possible causes not ruled out: reversals that swap only the second and third
  tower, and counter increments caused by something other than a swap of the best tower.

Three steps that would settle the timing of good reversals:

1. A performance-dependent raw file, looked at around the good-counter increments (print order
   of `num_good_reversals` against `curr_rew_mag`).
2. The pyControl task source matching `task_file_hash`, to read the order in which the counter
   and the magnitudes are updated.
3. Run this branch on the real cohorts and list the dropped reversals. The
   `ReversalWarning` gives subject, session, session-local trial and recording run for each, to
   compare with the raw print lines.

## Acceptance criteria

- `classify_towers_at_good_reversals` gives the same result under any `PYTHONHASHSEED`
  (subprocess test, seeds 0-7) and returns `third=None` when the best tower is unchanged.
- `get_good_reversal_info` drops reversals with an unchanged best tower (or a tie for best)
  with one warning listing them; the dropped reversals still end neighbouring windows; the
  first-block record is never dropped.
- Rank counts, the early/late split and the best-change split cover only valid reversals
  (`tests/test_best_unchanged_exclusion.py`).
- A bad-only session yields no good reversals and warns; a legacy session in the same subject
  keeps its block-change boundaries, first-block record and full window.
- A reversal with a missing magnitude is dropped (three representations tested); the classifier
  raises `ValueError` naming the location for a record that has one.
- Suspect and uncheckable counts appear beside the logged totals and leave them unchanged.
- A merged series whose length differs from the trial count raises `ValueError` naming the subject.
- Each bad-reversal record owns its `towers` list; an explicit `post` in
  `get_bad_reversal_info` never extends past the next boundary.
- Outputs on inputs untouched by these changes are identical to main.

## Results change

The sample cohort (`cohort-01/rawdata`, one subject, one session) has neither `has_good` nor
`has_bad` set and no reversals, so none of the bugs is exercised. Output of both reversal
functions (default, `post=8`, `include_first_block=True`) and of the first-leave analysis is
byte-identical on main and on this branch.

| quantity | main | fixed |
| --- | --- | --- |
| good reversals, sample cohort | 0 | 0 |
| bad reversals, sample cohort | 0 | 0 |
| first-leave counts, sample cohort | new_best 0, third 0, total 0 | same |
| reversals dropped for unchanged best tower, sample cohort | n/a | 0 |
| `suspect_good_reversals` / `uncheckable_good_reversals`, sample cohort | n/a | not present (no good/bad counters) |

On synthetic subjects (best tower changing at every reversal, so nothing is dropped), the
good-reversal windows, rank counts, early/late split, first-leave counts, choice probabilities and
bad-reversal windows are identical to main, with these exceptions:

| case | main | fixed |
| --- | --- | --- |
| bad-only subject (8 trials, one block change) | 1 good reversal | 0, one warning |
| legacy session + bad-only session, same subject | block changes of both sessions (5 and 11) | block change of the legacy session only (5), post-window 5..10 as on main |
| bad-only subject, `include_first_block=True` | 1 first-block record spanning the bad reversal | no record |
| first-leave counts with `include_first_block=True` | first-block record counted with an arbitrary third tower | first-block record not counted (new_best 2, third 2, total 4 becomes 2, 1, 3 for one 20-trial session) |
| explicit `post=6` bad window at reversal@3 | 6 trials | 4 trials |
| best tower unchanged (A best before and after) | third tower depends on the hash seed (T2, T3 or T4); record kept in every analysis | dropped at the source, with a warning listing it |
