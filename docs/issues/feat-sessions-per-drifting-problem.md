---
type: feat
complexity: moderate
status: in review
---

# Sessions per drifting problem, per mouse

Requested: for the drifting task, a figure in `all-drifting-problems` of the number of sessions it
took to complete each drifting problem: x = drifting problem, y = number of sessions, one line per
mouse (like the long-pokes-per-relative-session plot).

## Acceptance criteria
- `sessions_per_problem_table` counts, per mouse and drifting problem, the sessions with bandit
  trials (`_part-NN` parts once), the assigned port-finding-only sessions in a separate column,
  and the total; a mouse's latest problem of any task type is `complete=False`.
- The figure has integer ticks, a line per mouse, the running problem hollow and dashed, and the
  mean ± SE of the completed problems with n per problem on the x ticks.
- Both runners write `sessions-per-problem/` (cohort run: under
  `drifting/all-drifting-problems/`; all-cohorts: under `all-cohorts/`, with a cohort-mean figure
  when several cohorts are pooled) and `sessions_per_problem.csv`.

## Decisions (PI to check)
- Plotted count is the bandit sessions; `include_port_finding=True` on
  `write_sessions_per_problem_figures` / `summarize_sessions_per_problem` plots the total. The
  title says which. The switch is not exposed on `run_cohort` or the scripts.
- The table is rewritten from the whole cohort on every run (also partial ones), not merged by
  problem; the figure is redrawn on every run of a cohort with drifting problems (also a
  blocked-only partial run), because any new session can complete a problem.
- The all-cohorts figures do not name mice (as its other figures); the cohort-mean figure is
  skipped for a single cohort. All-cohorts colours mice by position in the sorted mouse list.
- `run_all_cohorts_drifting` takes `import_logs={cohort: ImportLog}` so port-finding-only sessions
  of passed-in trials are counted; loaded cohorts use their own logs.
- A provisional latest problem (`problem_provisional`) gets `provisional=True` and the problem
  before it is not yet complete (the final session may be a settings error).
- A newest port-finding-only session that no problem took leaves the previous problem running
  (conservative; documented).
- A session split into parts across two problems counts once in each.
- `n_sessions` agrees with `problem_map.csv`'s `n_sessions` (checked on the staged data).
- Single-mouse readability: the mean is drawn under the mouse lines. The by-cohort figure labels
  cohorts without a completed problem and shows n per point.
- A mouse with a running problem and one with no completed problem at a number leave that mean
  point empty; an SE needs two mice.
- `session_without_part` lives in `behavior_analysis`, because the import cache key hashes
  `src/behavior_import`.

## Results change
- New figures `sessions-per-problem/Sessions per Drifting Problem.{png,pdf}` (and `... by
  Cohort` in the all-cohorts folder) and new table `sessions_per_problem.csv` (cohort tables and
  all-cohorts tables). Nothing else changes.
