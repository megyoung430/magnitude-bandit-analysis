---
type: feature
complexity: complex
status: in review
---

# Rerun only the problems touched by recent sessions

After a couple of new recording days the cohort runner regenerated everything. `run_cohorts.py`
now takes `--last-sessions N` or `--since YYYY-MM-DD` and regenerates only the problems that
contain those sessions, plus the pooled all-drifting-problems figures when a drifting problem is
among them.

## Acceptance criteria

- `select_recent_sessions` (`src/pipelines/incremental.py`, pure) returns the `(task_type, k)`
  problems containing a selected session and the selected `(mouse, session)` pairs, for each
  mouse's last N sessions or the sessions dated on or after a day. N below 1, a date that is not
  `YYYY-MM-DD`, both or neither option, and a selection matching no session raise `ValueError`
  (the last as `NoSessionsSelectedError`, listing the mice and dates that exist).
- `run_cohort(..., only_problems=...)` runs `run_problem` for those problems only; the pooled
  drifting analyses run only if one of them is drifting (and still pool every drifting problem);
  the review tables, suspect-reversal table and trial/block summaries are always rewritten.
- The four stacked diagnostic tables get the new rows of the rerun problems in place of their old
  rows and keep the others. A missing table, or one without `task_type` / `problem_in_type`
  columns, gets a warning ("the table is partial") and only the new rows.
- `run_cohort_incremental` / `run_cohorts(last_sessions=, since=, dry_run=)` and the CLI options
  `--last-sessions`, `--since` (mutually exclusive) and `--dry-run` (plan only, nothing written).
  The plan (problems, pooled scopes, sessions behind each) is printed first and a summary of what
  was written last.
- A run without these options is the full run.

## Decisions

- A session split into `_part-NN` segments counts once, and sessions are ordered by
  `session_order.ordered_session_keys`, the order problems are numbered in. A mouse's last N
  sessions are taken across all its problems, so they can span a blocked and a drifting one.
- Problems are selected on the whole cohort's numbering (the cached import is loaded once and
  used for both the selection and the run), so `problem-KK` means what it does in a full run.
- With several cohorts, a cohort with no matching session is skipped with a `[SKIP]` line,
  because a date that is recent for one cohort may not reach another. The run raises only if no
  cohort matches.
- `warnings_log.csv` lists the warnings of the latest run only, so after a partial run it does not
  cover the problems that were not rerun.
- Rows are matched on the problem columns the tables already carry; stale rows of a problem that
  no longer exists under that number are not detected, so renumbered cohorts need a full run.

## Results change

None for full runs (no option given): same figures, tables and file names. A partial run writes
the same files as a full run for the problems it regenerates.
