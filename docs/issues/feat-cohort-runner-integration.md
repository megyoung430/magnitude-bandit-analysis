---
type: feature
complexity: complex
status: in review
---

# Wire the value, reversal-aligned and poke analyses into the cohort runner

`run_cohort` (`src/pipelines/cohort_analysis.py`) and `run_problem` / `run_single_session_analysis`
(`src/pipelines/open_field.py`) did not call the analyses added on the branches
`feat-value-psychometrics`, `feat-reversal-aligned-arms`, `feat-value-change-aligned` and
`feat-poke-heatmaps`. Running a cohort now produces all of them.

## Acceptance criteria
- Drifting problems only: `run_value_psychometric_analysis`, `run_reversal_aligned_arm_analysis`
  and `run_value_change_analysis`, per drifting problem (`drifting/problem-KK/<subfolder>`) and
  pooled over a mouse's drifting problems (`drifting/all-drifting-problems/<subfolder>`).
  Problems stay separate where the analysis needs it (beliefs reset per problem, windows never
  cross a problem boundary). Three switches on `AnalysisSelection`, on in `DRIFTING_SELECTION`.
  A blocked problem with any of them on raises.
- Every problem, blocked and drifting: `run_poke_analysis` into `<problem>/long-pokes/`
  (mean non-choice long pokes per session and the problem-level maze heatmaps). The runner had no
  long-pokes output before, so nothing is duplicated.
- Red outlines show the choice ports of the problem immediately before, whatever its task type:
  `previous_problem_choice_towers` once per cohort from the overall problem order, then
  `previous_choice_towers_for_problem` per (task type, k) problem.
- Single-session figures get the red outlines; each session is parsed once and the parsed
  `poke_counts` go to both `plot_single_session` and `run_poke_analysis`. A `PokeParseError`
  warns and draws without the maps (already handled); a `PokeAlignmentError` stops the run.
- Existing figures, file names and folder layout are unchanged; problems are numbered within
  task type.
- README documents how to run and the output folder tree.

## Decisions
- New `AnalysisSelection` fields (`run_poke_analysis`, `run_value_psychometric_analysis`,
  `run_reversal_aligned_arm_analysis`, `run_value_change_analysis`) default to off, so
  `run_open_field_analysis` and any caller with a bare `AnalysisSelection()` behave as before.
  `BLOCKED_SELECTION` turns on only the poke analysis; `DRIFTING_SELECTION` turns on all four.
- `run_problem` raises `ValueError` if a drifting-only switch is on for `task_type="blocked"`;
  `selection_for_task_type` raises the same, and `run_cohort` checks both selections before it
  writes anything.
- Subfolders: `value-psychometrics`, `reversal-aligned-arms`, `value-change`, `long-pokes`, in the
  kebab-case style of `reversal-stats`, `choice-stats` and `single-sessions`.
- Per-problem and pooled folders use the same function (`run_drifting_value_analyses`), called
  with `{k: sessions}` for one problem or `{k: sessions, ...}` for the pool. The value-psychometric
  and value-change entry points pool across the problems they are given. The reversal-aligned
  entry point would also write one extra figure set per problem (its scope `problem-KK`) next to
  the pooled one, duplicating the per-problem folders. So the runner builds the windows per
  problem (`build_windows_with_magnitudes`), merges them per mouse (`merge_windows_by_subject`)
  and hands the entry point one pooled entry; windows therefore never cross a problem and the
  pooled folder holds only the pooled figures. Window settings are shared constants
  (`pre=10`, `post=30`) so the builder and the entry point cannot disagree.
- `run_reversal_aligned_arm_analysis` gains `scope_label` (default `None`: unchanged), the title
  text of the pooled scope. The runner passes `problem-KK` for one problem and
  `all drifting problems` for the pool, so no per-problem figure says `[pooled]`. File names
  and the counts table keep the scope name `pooled`.
- Parsing once: when both the poke analysis and the single-session figures are on, `run_problem`
  parses every mouse's sessions with `mouse_poke_counts` and passes the result to both.
- `run_problem(..., previous_choice_towers=None)` and
  `run_single_session_analysis(..., previous_choice_towers=None, poke_counts=None)` are new
  optional arguments; `None` draws no red outline as before.

## Results change
- New folders per problem: `long-pokes/` (blocked and drifting). Per drifting problem:
  `value-psychometrics/`, `reversal-aligned-arms/`, `value-change/`. Pooled:
  `drifting/all-drifting-problems/{value-psychometrics,reversal-aligned-arms,value-change}/`.
- Single-session figures gain red outlines of the previous problem's ports on the maze row.
- Nothing else changes.
