---
type: feature
complexity: complex
status: in review
---

# One command for the cohort analysis suite (cohorts 1-4)

The suite in `scripts/full_pipelines/cohort-3-analysis.ipynb` ran one cohort at a time and cut
blocked from drifting problems at a fixed problem number (4). The number of blocked problems
differs by mouse, so that cut is wrong for some mice. The PI wants all four cohorts from one
command, with the task type taken from the task name.

## Acceptance criteria
- `uv run python scripts/full_pipelines/run_cohorts.py --cohorts 1 2 3 4` runs the suite for
  each cohort (`src/pipelines/cohort_analysis.py`: `run_cohort`, `run_cohorts`); the notebook
  `cohort-analysis.ipynb` does the same with `cohort_ids = [1, 2, 3, 4]`.
- Task type is classified from the session's `task_name` (`classify_task_name`): last path
  component, case-insensitive, `blocked`/`structured` is blocked, `drifting` is drifting;
  pretraining stays excluded by problem numbering; an unknown name or a name matching both raises
  a `ValueError` naming subject, session, file and task name.
- A change of task type starts a new problem, exactly like a layout change: the type is part of
  the signature `assign_problems` compares, so the one-session settings-error rule applies to
  type changes too (`problem_numbering.py` documents it). `import_data(classify_task_types=True)`
  reads the type of each session before numbering (pretraining skipped first); a missing,
  unrecognised or ambiguous task name raises, naming subject, session, file and task name, and a
  missing name says the run stops. `group_sessions_by_problem` reads the stored `"task_type"`, so
  both entry points agree. The cohort runner loads with it on.
- A problem whose sessions have different types raises, naming the sessions (a guard for data
  imported without task types).
- Each mouse's problems are renumbered within type; `{problem: {subject: sessions}}` is
  regrouped as `{(task_type, k): {subject: sessions}}`; each mouse's overall problem number stays
  on its sessions (`"problem"`) and is written to `problem_map.csv`.
- Blocked: choice probability, first leave, number of reversals, true-value rank proportions
  (pooling modes True and False), single sessions. Drifting: number of reversals, true- and
  believed-value rank proportions, single sessions.
- Drifting rank proportions are also pooled over all of a mouse's drifting problems, ranks
  computed within each problem first, both value bases and both pooling modes, with p-values.
- Trial/block summaries over all problems, and a suspect-reversal CSV with the task type and
  the within-type problem number as columns, are written per cohort.
- Outputs: `results/figures/open-field/cohort-0N/{blocked,drifting}/problem-KK/...`,
  `.../drifting/all-drifting-problems/choice-stats`, `.../trial-block-summaries/all-problems`,
  `results/tables/open-field/cohort-0N/{problem_map,suspect_reversals}.csv`. `results/` is
  already in `.gitignore`.

## Decisions
- `AnalysisSelection.last_reversal_window_problem` is removed; the selection alone decides, and
  the runner picks `BLOCKED_SELECTION` or `DRIFTING_SELECTION` per type. `run_problem` gains an
  optional `task_type` that adds a folder between the cohort and the problem.
- The task-type code lives in `src/behavior_import/task_types.py`; `session_source_paths`
  (in `import_data.py`) reads the file paths from the `"df"` frames' `source_path` attribute, so
  the subject numbers for the summaries need no re-read of the raw files.
- `run_cohort(results_root=...)` is the folder that holds `figures/` and `tables/` (named
  `results_root`, not `output_root`, because `run_problem`'s `output_root` is the figures folder).
- The cohort folder is `cohort-{id:02d}`.
- Trial/block summaries take the overall problem numbers (all problems, as in the notebook).
- `run_cohorts` stops at the first failing cohort with a `RuntimeError` naming it (the original
  error is chained). With `continue_on_error=True` the remaining cohorts run and the failures are
  raised together at the end, so the exit code is still non-zero.
- Task types are opt-in at import (`classify_task_types`, default off) because other tasks, such
  as the grid maze, and existing tests have other or no task names; unclassified data is
  numbered by layout alone, as before.
- `number_within_task_type` is computed once per cohort and passed to `regroup_by_task_type` and
  `problem_map_table`. Each distinct `ReversalWarning` text is shown once per cohort run
  (`each_reversal_warning_once`), including the repeats from the pooled drifting step.
- Suspect-reversal table: new `num_t` (the 1-based, run-local trial number printed in the run's
  file) and `print_time` (that row's time) columns, so a reviewer can search the raw file;
  `session_trial` stays 0-based over the session. Checked on the sample (runs of 31 and 62
  trials): merged trial 53 is run 1, `num_t` 23, as in the file. A reversal logged between runs
  keeps its real check result as `reason` and is flagged by a new `logged_between_runs` column
  (the old "uncheckable between-run or terminal" label is gone); the module docstring now says
  such a reversal is dropped only when the best-change check fails or a magnitude is missing.
- `write_suspect_reversal_table` also accepts the leading columns `task_type`,
  `problem_in_type`, `overall_problem`. The CSV's `overall_problem` is looked up per mouse.
- Rank-proportion plots use the dispatcher defaults (average across mice only).
- `problem_map.csv` columns: `mouse, task_type, problem_in_type, overall_problem, sessions`
  (session keys joined by `;`).
- The import is cached on disk by default (`use_cache=True`, keyed on the raw files and code).

## Results change
- Drifting analyses now run on every problem the task name calls drifting, for each mouse,
  instead of problems >= 4; blocked analyses (including choice probabilities and first leave)
  run on the problems the task name calls blocked, instead of problems <= 3 (cohort notebook) or
  <= 4 (`run_problem` default).
- Figure folders change from `.../cohort-0N/problem-NN/` to
  `.../cohort-0N/{blocked,drifting}/problem-KK/`, with KK counted within the type. The pooled
  folder `all-drifting-problems` moves under `drifting/`.
- `run_open_field_analysis` (and so `run_open_field_analysis.py`) no longer limits choice
  probabilities and first leave to problems <= 4; its selection now applies to every problem it
  is given. Pass a selection with those two off if only the early problems should have them.
- New files: the two CSVs per cohort.
- A task-type change with an unchanged layout now splits a problem in two, so problem numbers
  (and the within-type numbers) can differ from the layout-only numbering in such data.
- The suspect-reversal table gains `num_t`, `print_time` and `logged_between_runs`, and its
  between-runs rows change `reason` from "uncheckable between-run or terminal" to the real
  check result.
- `run_open_field_analysis.py` keeps its behaviour (one selection for every problem of its
  blocked-task cohort); its docstring points to `run_cohorts.py`.
- `cohort-3-analysis.ipynb` is unchanged apart from a pointer cell at its top.
