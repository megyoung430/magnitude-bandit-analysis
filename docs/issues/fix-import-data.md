---
type: bug
complexity: simple
status: in review
---

# fix-import-data

Fixes known issues in `behavior_import` and puts problem numbering behind one shared rule.

## Acceptance criteria

1. **One problem rule** (`src/behavior_import/problem_numbering.py`), used by `import_data` (`problem`) and `group_sessions_by_problem`, decided by the PI:
   - A problem is the full tower layout: choice towers plus initiation tower (default `B2`). Tower order and case do not matter. Changing only the initiation tower starts a new problem. `has_good` no longer splits problems.
   - A session whose layout differs from the layout-bearing session before it is a suspected settings error if (a) its layout lasts only that one session before changing again, or (b) it has no `num_t_found`, or (c) it does not start with `found_ports` false.
   - Flagged sessions get `problem = None`, `problem_flag = "suspected_settings_error"` and a `warnings.warn` naming subject and session. `group_sessions_by_problem` leaves them out (neither a new problem nor pooled); they stay in the data for single-session analyses.
   - A, flagged B, A: both A sessions are problem 1. A, B, A with a real B: 1, 2, 3.
   - The first layout-bearing session is never flagged. Evidence is only checked on sessions that change the layout.
   - Sessions that are not bandit problems are skipped (PI decision), never break pooling (A, skipped, A is problem 1, 1) and stay in the data. Pretraining: `problem_flag: "pretraining"`, no warning. Port finding: `"port_finding"` plus a warning. Any other session without choice towers: `"no_choice_towers"` plus a warning.
   - `merge_recurring_problems` is removed: A, B, A to 1, 2, 1 contradicts the rule.
2. `import_data` returns subjects sorted.
3. A path with no valid `ses-<N>_date-<YYYYMMDD>` folder, or one whose date is not a calendar date (`ses-1_date-20261340`), raises a `ValueError` naming the path.
4. A missing, empty or corrupt first `run_start` row raises a `ValueError` naming subject, session and file, in `import_data` and in `split_recording_configurations`. `reconcile_reversals` raises a `ValueError` naming subject, session and run index for an unparseable `run_start`.
5. An unparseable `run_end` is still skipped (pinned by a test).
6. `testing_code.ipynb`: the `plot_distributions_per_mouse` / `plot_distributions_per_problem_all_mice` calls pass `drop_last_problem`, their real parameter. (The `compute_problem_blocklen_summary(..., drop_last_problem_per_mouse=False)` call in cell 24 is correct and unchanged.)

## What is checked, exactly

- `num_t_found` (real representation): `run_start` of the session's earliest recording is a JSON dict that carries a `num_t_found` counter (0 on the sample data); port-finding trials are printed dicts that carry `num_t_found` (`recording_trial_info` drops them). Present means the first `run_start` has the key, or any recording of the session prints a dict with it.
- `found_ports`: the first `run_start` has `"found_ports": false` (JSON false). Missing or null counts as not false. The sample session is mid-problem (`found_ports` true), which is why evidence is only checked on sessions that change layout.
- Layout in `import_data` comes from `run_start`; in `group_sessions_by_problem` from the trial-level `choice_towers` and the session's `initiation_tower`. Both feed the same function.

## How each kind is recognised

- **Task name** is the pyControl header row `info task_name`, e.g. `Meg\3x3_field_drifting_value_bandit` in the sample files. `import_data` now stores it as `task_name` on each session (and in `settings_evidence`).
- **Pretraining** (`PRETRAINING_TASK_NAMES = {"mag"}`, `PRETRAINING_TASK_NAME_PARTS = ("pretraining",)` in `problem_numbering.py`): the last path component of the task name, case-insensitive, equals `mag` or contains `pretraining` (covers `initiation_pretraining`). Chosen from the lab's cohort-fitting notebook, which excludes task names containing "mag" or "pretraining"; "mag" is matched as a whole name so a `..._magnitude_bandit` task is not caught. The sample data has only a bandit task, so this rests on the notebook and the PI's examples. Pretraining wins even if the run_start lists choice towers.
- **Port finding**: no choice towers, not pretraining, and a printed trial dict carries `num_t_found` (the same port-finding trials `recording_trial_info` drops). `run_start` always carries a `num_t_found` counter (0 on the sample), so only printed dicts count as evidence. No task name for port finding is known from the data, so none is guessed.
- **No choice towers, neither of the above**: `no_choice_towers`.
- In `group_sessions_by_problem` an empty trial-level tower list now counts as no layout.

## Decisions

- Later sessions of a flagged layout are not "changes" (their previous session has the same layout) so they are not checked; they are numbered against the last unflagged layout. A flagged B followed by B, B gives A, None, 2, 2.
- Skipped sessions are removed from the sequence before layouts are compared, so they are not "the session before" a change and do not count towards "lasts one session".
- A no-choice-towers session that is neither pretraining nor port finding is flagged, not guessed (was: joined the current problem).
- `group_sessions_by_problem` still skips sessions without `trial_info` before numbering, so its numbers can differ from `import_data`'s when such sessions exist.
- Sessions that did not come from `import_data` (no `settings_evidence`) are rejected by `group_sessions_by_problem` with a `ValueError`, not given a default.
- `run_end`: an unparseable one is skipped, unlike `run_start`. A crashed recording can end with a truncated `run_end`, and the end snapshot only adds events after the last trial, which the trial counters and `run_start` totals still bound. A missing baseline (`run_start`) double counts; a missing end does not.
- A session with no `run_start` row at all now also raises a `ValueError` (was an `IndexError`).

## Results change

Sample data (`cohort-01/rawdata`): one subject, one session (`ses-91`, mid-problem), so no layout change occurs and no flag fires.

| quantity | main | fixed |
| --- | --- | --- |
| subjects / problem numbers | MY_72_L, [1] | MY_72_L, [1] |
| `group_sessions_by_problem` sizes | {1: 1} | {1: 1} |
| `reconcile_reversals` totals (md5) | 593f4783... | 593f4783... |

The sample's `task_name` is `Meg\3x3_field_drifting_value_bandit`: not pretraining.

Elsewhere, numbers change as follows (synthetic cases in the tests): `import_data` `problem` now increments (was always 1); `group_sessions_by_problem` splits on initiation tower, no longer splits on `has_good`, drops flagged sessions, and A, none, A is one problem (was 1, 1, 2 there). Re-run any saved per-problem results; cohort data with one-session layouts will lose those sessions from problem analyses.
