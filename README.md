# magnitude-bandit-analysis

![Python 3.13](https://img.shields.io/badge/python-3.13-blue)
![Status](https://img.shields.io/badge/status-work--in--progress-yellow)

Analysis code for a 3x3 spatial drifting-value bandit task in mice, where
reward magnitude at each arm drifts over time and periodically reverses.

> 🚧 **Work in progress.** This repo is under active development — code
> organisation and documentation will keep evolving.

## What's in this repo

- Import pipeline for raw pyControl session logs into trial-level DataFrames
- Behavioral analyses: reversal detection, choice probabilities, task
  statistics, GLMs
- Matching plotting functions for every analysis
- Synthetic bandit task simulations (Gaussian Process and Markov-chain
  reward dynamics) for comparison against real behavior

## Project organisation

```
magnitude-bandit-analysis/
├── data/              # local/sample data (main dataset lives on ceph — see below)
├── pyproject.toml     # dependencies (uv); uv.lock pins exact versions
├── results/
│   └── figures/       # generated plots, organized by task/cohort/problem
├── scripts/            # entry points that use src/
│   ├── bandit_behavior_tutorial.ipynb
│   ├── individual_analyses/
│   ├── full_pipelines/
│   └── to_integrate/
└── src/                 # the actual analysis library
    ├── behavior_import/
    ├── behavior_analysis/
    ├── behavior_visualization/
    └── task_simulations/
```

## Setup

```bash
git clone https://github.com/megyoung430/magnitude-bandit-analysis.git
cd magnitude-bandit-analysis
uv sync                              # core analysis environment
uv sync --extra notebooks            # + Jupyter kernel and notebook-only libraries
uv sync --extra paper-rl             # + JAX/optax for paper-style RL and RNN fitting
uv sync --all-extras                 # everything, incl. the Haiku reference tests
```

`uv sync` creates `.venv/` from `uv.lock`, so every machine gets the same
versions. Run code with `uv run python ...` / `uv run jupyter lab`, or select
`.venv` as the notebook kernel. Install uv with
`curl -LsSf https://astral.sh/uv/install.sh | sh`.

To add a dependency: `uv add <package>` (or `uv add --optional notebooks <package>`),
then commit `pyproject.toml` and `uv.lock` together.

The project follows the [Good Research Code Handbook](https://goodresearch.dev/).

## Data organisation

Data follows the [NeuroBlueprint](https://neuroblueprint.neuroinformatics.dev/latest/index.html)
specification, with one addition: a `cohort-0X` folder sits above `sub-0X`,
since animals are run in cohorts.

```
3x3_field_value_bandit/
└── rawdata/
    └── cohort-01/
        └── sub-01/
            └── ses-01_date-20260115/
                └── behav/
                    └── sub-01_ses-01_task-bandit.tsv
└── derivatives/
    └── cohort-01/
        └── sub-01/
```

The main dataset lives on the SWC's ceph storage, not in this repo. For
instructions on accessing it locally (SMB mount) or on the HPC cluster
(VS Code + SLURM), see the **Setting Up magnitude-bandit-analysis** page
on the project Notion.

Scripts and notebooks find the data through `BANDIT_DATA_ROOT`, the directory
that contains the experiment folders. Set it once per machine:

```bash
cp .env.example .env      # .env is git-ignored
# then edit BANDIT_DATA_ROOT, e.g. /Volumes/behrens/meg on a Mac with ceph mounted
```

If the variable is unset or the directory isn't mounted, `get_settings()` raises
immediately with a clear message instead of failing later inside `import_data`.

### Cached imports

Scripts load data with `load_trials_by_problem(root)`
(`src/behavior_import/load_trials.py`), which caches the result of
`extract_trials_grouped_by_problem(import_data(root))` in `.cache/`
(git-ignored; override with `BANDIT_CACHE_DIR`). The cache key includes every
raw `.tsv` file's path, size and modification time and a hash of
`src/behavior_import/`, so new sessions or import-code changes trigger a fresh
import automatically. Delete `.cache/` at any time; it only costs a re-import.

### Resumable cohort cross-validation and recovery runs

`run_problem_cross_validation` (`src/agents/cohort_fitting.py`), `run_recovery`
(`src/agents/parameter_recovery.py`) and `run_paper_sweep` / `run_problem_sweep`
(`src/agents/paper_rl_fitting.py`) save into a folder named by a hash of what
determines their results, and an identical rerun resumes from the finished folds, runs
or checkpoints in that folder. The shared helpers live in `src/agents/run_store.py`.

The hash covers:

- the module's `ALGORITHM_VERSION` (an integer, currently `1`), and the `MODEL_VERSION` of
  each model module the run uses (`best_rl`, `released_rl`, `neural_bandit`,
  `mouse_agents`, `drifting_bandit`);
- the configuration: a fingerprint of the input sequences, the settings (seed, number
  of starts, grid, models, problems), the bounds, constants such as the at-bound
  tolerance, and the installed library versions (numpy and scipy for cohort
  cross-validation and recovery; jax, jaxlib, optax, numpy and the JAX backend and
  platform for the paper-style sweeps), so environments that resolve to different
  versions never mix saved results;
- for `run_recovery`, the source text of the `fit_function` and `loss_function` you pass
  in, and the values of their closure variables when those are simple immutable
  values (numbers, strings, booleans, `None`, and tuples of those). Redefining one in a notebook, or
  building it from a factory with a different number (`make(0.2)` vs `make(0.7)`), starts
  a new experiment. Only the function's own source is hashed: not the functions it calls,
  the globals it reads, its default arguments, or closure variables holding arrays, lists,
  dicts or objects. Change one of those and the directory does not move, so use a new output root
  or edit the function's own text. A callable without retrievable source raises an error.

It does **not** cover the source text of the modules themselves. Editing a comment,
reformatting or refactoring keeps the folder, so an interrupted run resumes.
`manifest.json` records the source text of the code that created the folder, as
provenance only. It is written once and never overwritten; if you resume after the source
has changed, a `ProvenanceMismatchWarning` (shown every time) names the modules that differ from the stored ones.

**When to bump a version:** bump `ALGORITHM_VERSION` in the module whose results changed,
and `MODEL_VERSION` in a model module when what it computes changes. A bump starts every
affected run in a new folder; comments and refactors don't need one. Which version covers
which code:

| Run | Bump `ALGORITHM_VERSION` of | Bump `MODEL_VERSION` of |
|---|---|---|
| `run_problem_cross_validation` | `cohort_fitting.py` (the sticky model and the cross-validation) | `best_rl.py` for `best_rl` and `released_rl`; `released_rl.py` for `released_rl` |
| `run_recovery` | `parameter_recovery.py` | `mouse_agents.py`, `drifting_bandit.py` |
| `run_paper_sweep`, `run_problem_sweep` | `paper_rl_fitting.py` (includes its own sticky, Best RL and released CogMod models) | `neural_bandit.py` (neural models only) |

A configuration change (a different seed, bounds, data or library version) moves the
folder without any bump. Finished results always stay readable in the old folder. See
`docs/issues/infra-run-store.md` for the one-time migration to this scheme.

## Code organisation

| Module | What it does |
|---|---|
| `src/behavior_import/` | Parses raw pyControl `.tsv` logs into trial-level structures. `import_data.py` is the main entry point; handles field-name aliasing across protocol versions. |
| `src/behavior_analysis/` | Computes derived metrics from imported trials — reversals, choice probabilities, task statistics, GLM utilities, cross-problem summaries. |
| `src/behavior_visualization/` | One plotting function per analysis in `behavior_analysis/`. |
| `src/pipelines/` | End-to-end analysis pipelines as functions with typed settings (`open_field.py`: every standard open-field figure, per problem; `cohort_analysis.py`: the whole suite for a cohort, split by blocked/drifting task type). Scripts and notebooks call these rather than copying or exec-ing script code. |
| `src/task_simulations/` | Synthetic bandit sessions (GP-driven and Markov-chain reward dynamics) for comparing real behavior against simulated agents. |
| `scripts/individual_analyses/` | Standalone scripts, one per analysis/plot type — thin wrappers around `src/`. |
| `scripts/full_pipelines/` | Entry points for full cohort runs (`run_cohorts.py` calls `src/pipelines/cohort_analysis.py`; `run_all_drifting_problems.py` calls `src/pipelines/all_cohorts_drifting.py`; `run_open_field_analysis.py` calls `src/pipelines/open_field.py`; `run_grid_maze_analysis.py`) and the cohort notebooks. |
| `scripts/to_integrate/` | Exploratory notebooks not yet folded into `src/`. |

## Getting started

1. Complete the [Setup](#setup) steps above.
2. Open `scripts/bandit_behavior_tutorial.ipynb` — this walks through loading
   a session, building a trial-level DataFrame, and reproducing the core
   plots.
3. For a specific analysis, check `scripts/individual_analyses/` for a
   standalone example, or `scripts/full_pipelines/` for a full pipeline.

### How to run cohorts 1-4

```bash
uv run python scripts/full_pipelines/run_cohorts.py --cohorts 1 2 3 4
```

(or `scripts/full_pipelines/cohort-analysis.ipynb`, with `cohort_ids = [1, 2, 3, 4]`). Raw data
is read from `<BANDIT_DATA_ROOT>/3x3_field_magnitude_bandit/rawdata/cohort-0N`. The run stops at
the first cohort that fails and names it; add `--continue-on-error` to run the others and
report all failures at the end.

- **Task type** comes from each session's pyControl task name (last path component, case
  insensitive): `blocked` or `structured` is blocked, `drifting` is drifting. Pretraining is
  skipped first; a missing name, any other name, or a name matching both stops the run with an
  error naming the mouse, session, file and task name.
- A **change of task type starts a new problem**, like a change of tower layout, even when the
  layout is unchanged; a task type that lasts one session before changing again is a suspected
  settings error, left out and pooled around like a one-session layout change.
- **Numbering** is per mouse within the type (blocked 1, 2, ...; drifting 1, 2, ...), so
  "drifting problem 1" lines up across mice even if their blocked problems differ in number.
  All sessions of one problem must have the same type.
- **Blocked** problems: choice probabilities, first leave, number of reversals, true-value rank
  proportions, long pokes and maze heatmaps, single sessions. **Drifting** problems: number of
  reversals, true- and believed-value rank proportions, long pokes and maze heatmaps, the three
  drifting-only analyses below, and single sessions, plus the rank proportions and the three
  drifting-only analyses pooled over all of a mouse's drifting problems. Ranks, beliefs and
  reversal windows are computed within each problem first, so nothing crosses a problem boundary.
- **Drifting-only analyses** (they use block changes, not good or bad reversals, so they run only on
  drifting problems and raise a `ValueError` naming the mouse and session if given a session with
  good/bad reversal counters; never run on blocked problems; switches on the drifting
  `AnalysisSelection`, all on in `DRIFTING_SELECTION`): `run_value_psychometric_analysis`
  (`value-psychometrics/`: heatmaps and psychometric curves of P(best) and P(second) by true and
  believed value), `run_reversal_aligned_arm_analysis` (`reversal-aligned-arms/`: arm choice
  probability and reward magnitude around block changes) and `run_value_change_analysis`
  (`value-change/`: choice probabilities around changes in the best or second value; the pre
  window starts no earlier than the last change of the best arm). Turn one off
  by passing `drifting_selection=DRIFTING_SELECTION.model_copy(update={"run_value_change_analysis":
  False})` to `run_cohort`. The reversal-aligned figure titles name their scope: `[problem-KK]`
  in a problem folder and `[all drifting problems]` in the pooled one (file names keep `pooled`).
  The reversal-aligned-arms and value-change figures each come in two versions, in the
  subfolders `mouse-weighted/` (each mouse's events pooled into one curve, mean ± SE across mice)
  and `event-weighted/` (every event counts once, mean ± SE across events; the SE treats events
  as independent, which events of one mouse are not, so it is narrower than the real
  uncertainty). Nothing is written at the top level of those two folders. The
  `reversal_aligned_arms_counts` table has a `weighting` column (`subject` or `event`).
- **Value-change diagnostic** (`scripts/individual_analyses/plot_value_change_diagnostic.py --cohorts 1 3
  [--no-cache]`): one PNG per drifting session in `figures/value-change-diagnostic/cohort-XX/problem-KK/<mouse>/`,
  showing the session's magnitudes and choices with each best-decrease rank-swap event's counted pre and post
  windows shaded, the choices coloured by label and the arm behind each label written underneath; `index.txt`
  lists every session's events. It also writes `mouse_curves_<scope>.csv` (`problem-KK`, `all-drifting-problems`):
  each mouse's event count, `prev_best`/`prev_second`/`third` choice probabilities and events per t (both truncation
  modes), with a `mean_across_mice` row equal to the mouse-weighted value-change curve.
- **Port preference** (`run_port_preference_analysis`, drifting problems, `port-preference/`, also
  pooled over all drifting problems): `P(chosen) / P(best)` per port (one maze per mouse) and for
  the X and plus ports (mean ± SE across mice). Relative to the problem's initiation port, plus
  ports share its row or column and X ports lie on its diagonals (from B2 these are the eight
  neighbours). A mouse offered ports from one category only has that category's ratio of 1 by
  construction; it is left out of the X vs plus figure, which says how many mice were excluded.
  The X vs plus figure is made only for the pool of all drifting problems: one problem's three
  choice ports usually all lie in one category.
  In a problem's folder the per-mouse mazes use the same outlines as the poke heatmaps (green:
  the problem's choice ports; red: the previous problem's; blue: initiation). The pooled mazes
  outline only the initiation port, since the choice ports differ between problems.
- **Long pokes and maze heatmaps** (`run_poke_analysis`, every problem, `long-pokes/`): mean
  non-choice long pokes per trial per relative session, and per-mouse maze maps of the port visits
  after the choice and the long pokes after initiation. After the choice (from the ITI to the next
  initiation, the initiating B2 poke included), consecutive beam breaks at one port count as one
  visit, and visits to the chosen tower count only after a visit somewhere else. Green outlines are the problem's choice ports and
  red outlines the choice ports of the problem the mouse ran immediately before, whatever its task
  type (so drifting problem 1 is outlined with the last blocked problem's ports); blue is the
  initiation port. The single-session figures carry the same maps for their session, with the red
  outlines. Two more figures, `Maze Heatmaps - Pokes After Choice (Port Finding)` and `... After
  Initiation (Port Finding)`, count the same two kinds of pokes in the port-finding trials, summed
  over every session of the problem that has some; a mouse without port-finding trials in the
  problem has no panel, and the figure is not drawn when no mouse has any (the single-session
  figures do not show them). Port-finding trials are never in the task-trial maps. Every session's raw events are parsed once for both. A session whose raw events cannot
  be read as trials warns and is drawn without maps (and noted in the `parse_error` column of the returned `summary`); parsed
  trials that disagree with the imported trials stop the run with a `PokeAlignmentError`.
- **Outputs** (under `results/`, which is git-ignored), for cohort `cohort-0N`:

  | Folder under `results/figures/open-field/cohort-0N/` | Contents |
  | --- | --- |
  | `blocked/problem-KK/reversal-stats/` | choice probabilities around good reversals, first leave, reversal counts |
  | `blocked/problem-KK/choice-stats/` | true-value rank proportions |
  | `blocked/problem-KK/long-pokes/` | long pokes per relative session, maze heatmaps |
  | `blocked/problem-KK/single-sessions/<mouse>/<session>/` | one overview figure per session |
  | `drifting/problem-KK/reversal-stats/` | reversal counts |
  | `drifting/problem-KK/choice-stats/` | true- and believed-value rank proportions |
  | `drifting/problem-KK/value-psychometrics/` | value heatmaps and psychometric curves |
  | `drifting/problem-KK/reversal-aligned-arms/` | arm choice probability and magnitude around block changes |
  | `drifting/problem-KK/value-change/` | choice probabilities around value changes |
  | `drifting/problem-KK/port-preference/` | port preference per port (X vs plus only in the pooled folder) |
  | `drifting/problem-KK/long-pokes/`, `.../single-sessions/` | as for blocked |
  | `drifting/all-drifting-problems/choice-stats/` | rank proportions pooled over the drifting problems |
  | `drifting/all-drifting-problems/{value-psychometrics,reversal-aligned-arms,value-change,port-preference}/` | the four drifting-only analyses pooled over the drifting problems |
  | `trial-block-summaries/all-problems/` | trial and block-length summaries |

  Tables go to `tables/open-field/cohort-0N/`, for looking at by hand. Each row names the mouse,
  the session (and the file where it has one), the task type and the problem (number within the
  type, and overall). The long-pokes, reversal-aligned-arms and value-change CSVs are in this
  folder, no longer in the figure folders; `trial-block-summaries/all-problems/` keeps its six
  summary CSVs.

  | Table | One row per |
  | --- | --- |
  | `problem_map.csv` | mouse and problem: initiation tower, choice towers, `sessions`, `n_sessions`, `session_range` (e.g. "1-10" or "3-8, 20") |
  | `suspect_reversals.csv` | good reversal or block change the analyses drop |
  | `suspect_sessions.csv` | problem-numbering flag (`suspected_settings_error`, `review_settings`, `provisional`, `port_finding`, `pretraining`, `no_choice_towers`) with a `reason`; or configuration setting that changed within a session (`run_start vs run_end`) or between recordings (`run_end vs next run_start`); keys are `suspect_sessions.SETTINGS_KEYS`; `found_ports` False to True in a problem's first session is not listed; a recording without `run_start` or `run_end` gets one "not checked: no run_end" row instead |
  | `suspect_starting_reward_mags.csv` | session (within a problem) whose starting magnitudes differ from the previous session's end, compared with the problem's previous session that has trials (last trial against first trial), with `likely_cause`, most to least likely (`starting_magnitudes.CAUSES`); a total that goes down is a task restart (`RESET`); a missing tower on either side is "cannot compare" |
  | `no_trial_sessions.csv` | file or recording the importer skipped, or session with no trials (one row per session, `files` lists its files), with `kind` and `reason` |
  | `no_second_best.csv` | block with more than one arm at 0, with the only rewarded arm |
  | `poke_trial_summary.csv`, `reversal_aligned_arms_counts.csv`, `value_change_events.csv`, `value_change_bin_counts.csv` | diagnostics of the analyses, stacked over problems with `task_type`, `problem_in_type` (and `overall_problem` where a mouse is named) |
  | `warnings_log.csv` | distinct Python warning of the run that the console shows (`n_times`), with the mouse, session, task type and problem its text names; written even when the run fails, and always with every table (header only when empty) |

  In `suspect_reversals.csv`, `session_trial` counts from 0 over the whole
  session; to find a row in the raw file, search `file` for `"num_t": <num_t>` (the run-local,
  1-based trial number pyControl printed) or for the row at `print_time`.
- **Before you compare with older runs**: a `results/` folder from before this runner keeps the
  previous folder layout (`cohort-0N/problem-NN/`) and the old `... by Average` rank-proportion
  files next to the new ones, so start from a clean `results/figures` folder. The
  across-blocks rank-proportion figure now has its own y tick labels.

### Pooling drifting problems across cohorts

```bash
uv run python scripts/full_pipelines/run_all_drifting_problems.py --cohorts 1 3 4
```

makes the figures of `drifting/all-drifting-problems/` (`choice-stats`, `value-psychometrics`,
`reversal-aligned-arms`, `value-change`, `port-preference`) from the drifting problems of the
listed cohorts together (`--no-cache` re-imports the raw files). They go to
`results/figures/open-field/all-cohorts/<same subfolders>/`, with `cohorts.txt` there naming the
cohorts pooled, the mice per cohort and the time of the run; the diagnostic tables
(`reversal_aligned_arms_counts.csv`, `value_change_events.csv`, `value_change_bin_counts.csv`) go
to `results/tables/open-field/all-cohorts/`. A rerun replaces those five folders and tables, so
the files always match `cohorts.txt`; the titles say "all drifting problems, cohorts 1, 3, 4".

- Problems stay separate: each is keyed `(cohort, drifting problem number)`, beliefs reset at each
  problem, and windows and events never cross a problem boundary.
- Mouse ids must be unique across cohorts; a mouse in two cohorts stops the run with an error
  naming it. A cohort without drifting problems also stops it.
- There are no per-mouse figures (the "Per Mouse" value heatmaps and port-preference mazes are in
  the cohort folders) and no legends naming mice; the per-mouse lines are still drawn, and the
  legends for curve types (P(best), P(second), ...) stay. The code is
  `src/pipelines/all_cohorts_drifting.py`.

### Reversals at session boundaries

Trial extraction reconciles good/bad counters and block numbers with raw `run_start` and
`run_end` records, including both long and abbreviated variable names. The recordings of a
session form one continuous session. The per-recording counters (`num_blocks`, `num_good_rev`,
`num_bad_rev`) restart in every recording; the session totals (`tot_num_blocks`,
`tot_num_good_rev`, `tot_num_bad_rev`) do not, so a change that the task logged while nothing was
recording (in a `run_end` after the last trial, or in the next `run_start`) is read from them.
The next recording's first trial is then the first trial of the new block (drifting task: a
block change; blocked task: a good or bad reversal), and a recording that restarts at 1 while
the totals show no change continues the block. Files without totals use the reward magnitudes
instead: the zero arm moved (drifting) or the magnitudes permuted (blocked).
`reversal_boundary_events` lists these events with kind `good`, `bad` or `block` (a block
change of unknown kind in a blocked session without totals). `run_boundaries` describes every
recording boundary (magnitudes before and after, counter and total deltas, and a verdict) with
`src.behavior_import.run_boundaries.describe_boundary`.

Such a reversal is checked like any other (the best tower changes at a good reversal), on the
last trial before it and the first trial after it. A change logged after a session's final
trial contributes to that session's `reversal_totals`. It starts a new block at the next
session's first trial (same problem) only if the magnitudes there show it (`session_check`
`consistent`); otherwise it is not carried into the next session, is recorded in
`reversal_diagnostics` as `session_end_change_unmatched`, and counts as uncheckable. A reversal is
uncheckable only when there is no trial to check against. Recording restarts subtract their
starting counter baseline to avoid counting persisted events twice. Magnitude-only changes and
counter decreases are retained in `reversal_diagnostics`, without assigning a good/bad
reversal type.

All reversal analyses use these corrected counters and totals. Re-extract raw
sessions before rerunning analyses on previously loaded data; existing notebook
outputs and saved figures are not automatically refreshed by a code change.
Run regression checks with `uv run pytest`; GitHub Actions runs the same suite on every push and pull request (`.github/workflows/tests.yml`). Tests build their own synthetic inputs, so they pass without access to ceph.

Reversal-over-time plotting also saves trial-based companions. Cumulative
curves use completed trials and include reversals after the final trial.
Moving-rate curves use a trailing 100-trial window by default and report
reversals per 100 trials; early windows use their actual observed exposure.
Across-mouse means include only mice observed at each trial count. Set
`reversal_trial_window` in the cohort-03 notebook, or pass `trial_window` to
`plot_moving_avg_reversals_over_time`, to change the window. Files contain
`(By Trial)` or `(By Trial, 100-Trial Window)` to distinguish them from
session-based plots.
