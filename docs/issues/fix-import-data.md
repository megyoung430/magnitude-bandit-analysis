---
type: bug
complexity: simple
status: in review
---

# fix-import-data

Fixes known issues in `behavior_import` and puts problem numbering behind one shared rule.

## Acceptance criteria

1. **One problem rule** (`src/behavior_import/problem_numbering.py`), used by `import_data` (`problem`, `problem_flag`) and `group_sessions_by_problem`, decided by the PI:
   - A problem is the full layout: choice towers plus initiation tower (default `B2`); order and case do not matter. `has_good` no longer splits problems.
   - Sessions are taken per subject in one shared order, `session_sort_key`: date, then session number (then the key, so `_part-NN` segments keep recording order).
   - Skipped sessions (below) are removed first. The rest are cut into runs of consecutive equal layouts; each run is compared with the **last unflagged layout**, not the session before it.
   - Same as the last unflagged layout: continues that problem, with no evidence check. A, flagged B, resumed A is 1, None, 1.
   - A new layout lasting exactly one session before the layout changes again is a suspected settings error: `problem` None, `problem_flag: "suspected_settings_error"`, a warning. It is not a new problem and not pooled; it stays in the data. It does not become the last unflagged layout.
   - A new layout lasting two or more sessions, or the final run of the data (duration unknown), is a real problem numbered from its first session, also on a return to an earlier layout (A, B, B, A is 1, 2, 2, 3). If its first session fails the evidence checks it keeps its number but gets `problem_flag: "review_settings"` and a warning (A, B failing, B, B is 1, 2 flagged, 2, 2).
   - Evidence checks on that first session: (c) `found_ports` is `false` in the first `run_start`; (b) port finding happened, meaning a printed trial dict carries `num_t_found`. The `num_t_found` key of `run_start` is always present (0) and is ignored. When (c) fails, (b) is not reported.
   - A first run lasting one session before the layout changes (B, A, A) is a settings error like any other one-session layout, and A is problem 1. A first run that is the only layout in the data, or the last one, is problem 1. The first problem of a subject is not evidence-checked (the sample session is mid-problem).
   - The number of a final run of one session is provisional: a later session can turn it into a settings error. The warning (when one fires) says so, and `ProblemAssignment.provisional` records it.
   - Not bandit problems, skipped with problem None, ignored when comparing neighbours (A, skipped, A is 1, None, 1), kept in the data: pretraining (`"pretraining"`, no warning); no choice towers with printed port-finding trials (`"port_finding"`, warning); any other session without choice towers (`"no_choice_towers"`, warning).
   - `merge_recurring_problems` is removed: A, B, A to 1, 2, 1 contradicts the rule.
2. `import_data` returns subjects sorted.
3. A path with no valid `ses-<N>_date-<YYYYMMDD>` folder, or an invalid calendar date, raises a `ValueError` naming the path.
4. A missing, empty or corrupt first `run_start` row raises a `ValueError` naming subject, session and file, in `import_data` and `split_recording_configurations`; `reconcile_reversals` names subject, session and run index.
5. An unparseable `run_end` is still skipped (pinned by a test).
6. `testing_code.ipynb`: the `plot_distributions_*` calls pass `drop_last_problem`, their real parameter. (Cell 24's `compute_problem_blocklen_summary(..., drop_last_problem_per_mouse=False)` is correct.)
7. Both entry points agree, including through the real `extract_trials` path with session numbers that run opposite to the dates.
8. Split sessions (a session whose recordings have different layouts becomes `_part-NN` segments in `extract_trials`): `import_data` never sees the segments and numbers the unsplit session by its first recording. Segments do not inherit the parent's `problem` / `problem_flag`. `group_sessions_by_problem` numbers all sessions including segments and writes `problem` / `problem_flag` back onto each of them (and onto every other numbered session), so after it runs the session dicts hold the numbering to use. Tests: S2 = [A then B] with S3 = A (the B segment is a flagged one-session layout) and with S3 = S4 = B (the B segment starts problem 2), through the real `extract_trials` path.
9. Each flagged session warns once, with `ProblemNumberingWarning` (shown always, so a notebook rerun warns again; pinned by a fresh-interpreter test). `import_data` warns at its call site. `group_sessions_by_problem` warns only about flags that differ from the `problem_flag` already recorded on the session, i.e. flags on split segments and flags on data that did not come through `import_data`.
10. The RL notebook metadata records `"problem_numbering": "layout_v2"` and no longer lists `merge_recurring_problems`.
11. One session order everywhere numbering or fitting orders sessions: `problem_utils.session_sort_key` (used by `cohort_fitting` to order sessions inside each problem) is now the shared key, date then session number, where it used to be session number then date. Order changes only when session numbers and dates disagree; on such data cohort fits change. (`sort_by_session.sort_sessions_chronologically`, a plotting helper ordering by session number alone, is untouched.)

## How each kind is recognised

- **Task name**: the pyControl header row `info task_name`, e.g. `Meg\3x3_field_drifting_value_bandit` in the sample files. `import_data` stores it as `task_name` on each session (and in `settings_evidence`); split segments get their own.
- **Pretraining** (`PRETRAINING_TASK_NAMES = {"mag"}`, `PRETRAINING_TASK_NAME_PARTS = ("pretraining",)`): the last path component of the task name, case-insensitive, equals `mag` or contains `pretraining`. Taken from the lab's cohort-fitting notebook, which excludes task names containing "mag" or "pretraining"; `mag` is matched as a whole name so a `..._magnitude_bandit` task is not caught. The sample data has only a bandit task, so this rests on the notebook and the PI's examples. It wins even if `run_start` lists choice towers.
- **Port finding**: no choice towers, not pretraining, and a printed trial dict carries `num_t_found` (the dicts `recording_trial_info` drops). No port-finding task name is known from the data, so none is guessed.
- **No choice towers, neither**: `no_choice_towers`. An empty trial-level tower list counts as no layout in `group_sessions_by_problem`.

## Decisions

- A single-session layout at the very end of the data is treated as a real problem (it has not "changed again"); it gets the evidence checks. Its number is provisional (see above).
- `group_sessions_by_problem` still skips sessions without `trial_info` before numbering, so its numbers can differ from `import_data`'s when such sessions exist. Sessions without `settings_evidence` are rejected with a `ValueError`.
- `run_end`: an unparseable one is skipped, unlike `run_start`. A crashed recording can end with a truncated `run_end`, and the end snapshot only adds events after the last trial, which the trial counters and `run_start` totals still bound. A missing baseline (`run_start`) double counts; a missing end does not.
- A session with no `run_start` row at all raises a `ValueError` (was an `IndexError`).
- `import_data.sort_session_key` is now an alias of the shared `session_sort_key`.

## Results change

Sample data (`cohort-01/rawdata`): one subject, one session (`ses-91`, mid-problem, task `Meg\3x3_field_drifting_value_bandit`, not pretraining), so no layout change occurs and no flag fires.

| quantity | main | fixed |
| --- | --- | --- |
| subjects / problem numbers | MY_72_L, [1] | MY_72_L, [1] |
| `group_sessions_by_problem` sizes | {1: 1} | {1: 1} |
| `reconcile_reversals` totals (md5) | 593f4783... | 593f4783... |

Elsewhere (synthetic cases in the tests): `import_data` `problem` now increments (was always 1); `group_sessions_by_problem` splits on initiation tower, no longer splits on `has_good`, orders by date then session number (was session number then date), leaves out settings-error, pretraining, port-finding and tower-less sessions, and keeps review-flagged sessions. Re-run saved per-problem results; one-session layouts drop out of problem analyses.

## TODO (not in code)

- TODO: when infra-run-store is merged, bump `ALGORITHM_VERSION` so cohort fits (`cohort_fitting`) rerun: problem numbers, the per-problem session order and the set of sessions in each problem all changed.
