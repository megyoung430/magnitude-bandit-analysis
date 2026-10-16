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
| `scripts/full_pipelines/` | Entry points for full cohort runs (`run_cohorts.py` calls `src/pipelines/cohort_analysis.py`; `run_open_field_analysis.py` calls `src/pipelines/open_field.py`; `run_grid_maze_analysis.py`) and the cohort notebooks. |
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
  (`value-change/`: choice probabilities around changes in the best or second value). Turn one off
  by passing `drifting_selection=DRIFTING_SELECTION.model_copy(update={"run_value_change_analysis":
  False})` to `run_cohort`. The reversal-aligned figure titles name their scope: `[problem-KK]`
  in a problem folder and `[all drifting problems]` in the pooled one (file names keep `pooled`).
- **Long pokes and maze heatmaps** (`run_poke_analysis`, every problem, `long-pokes/`): mean
  non-choice long pokes per trial per relative session, and per-mouse maze maps of the pokes after
  the choice and the long pokes after initiation. Green outlines are the problem's choice ports and
  red outlines the choice ports of the problem the mouse ran immediately before, whatever its task
  type (so drifting problem 1 is outlined with the last blocked problem's ports); blue is the
  initiation port. The single-session figures carry the same maps for their session, with the red
  outlines. Every session's raw events are parsed once for both. A session whose raw events cannot
  be read as trials warns and is drawn without maps (and noted in `Poke Trial Summary.csv`); parsed
  trials that disagree with the imported trials stop the run with a `PokeAlignmentError`.
- **Outputs** (under `results/`, which is git-ignored), for cohort `cohort-0N`:

  | Folder under `results/figures/open-field/cohort-0N/` | Contents |
  | --- | --- |
  | `blocked/problem-KK/reversal-stats/` | choice probabilities around good reversals, first leave, reversal counts |
  | `blocked/problem-KK/choice-stats/` | true-value rank proportions |
  | `blocked/problem-KK/long-pokes/` | long pokes per relative session, maze heatmaps, four CSVs |
  | `blocked/problem-KK/single-sessions/<mouse>/<session>/` | one overview figure per session |
  | `drifting/problem-KK/reversal-stats/` | reversal counts |
  | `drifting/problem-KK/choice-stats/` | true- and believed-value rank proportions |
  | `drifting/problem-KK/value-psychometrics/` | value heatmaps and psychometric curves |
  | `drifting/problem-KK/reversal-aligned-arms/` | arm choice probability and magnitude around block changes, counts CSV |
  | `drifting/problem-KK/value-change/` | choice probabilities around value changes, contamination and bin-count CSVs |
  | `drifting/problem-KK/long-pokes/`, `.../single-sessions/` | as for blocked |
  | `drifting/all-drifting-problems/choice-stats/` | rank proportions pooled over the drifting problems |
  | `drifting/all-drifting-problems/{value-psychometrics,reversal-aligned-arms,value-change}/` | the three drifting-only analyses pooled over the drifting problems |
  | `trial-block-summaries/all-problems/` | trial and block-length summaries |

  Tables go to `tables/open-field/cohort-0N/`: `problem_map.csv` (mouse, task type, number within
  type, overall problem, sessions) and `suspect_reversals.csv` (the good reversals the analyses drop, for
  review by hand). In `suspect_reversals.csv`, `session_trial` counts from 0 over the whole
  session; to find a row in the raw file, search `file` for `"num_t": <num_t>` (the run-local,
  1-based trial number pyControl printed) or for the row at `print_time`.
- **Before you compare with older runs**: a `results/` folder from before this runner keeps the
  previous folder layout (`cohort-0N/problem-NN/`) and the old `... by Average` rank-proportion
  files next to the new ones, so start from a clean `results/figures` folder. The
  across-blocks rank-proportion figure now has its own y tick labels.

### Reversals at session boundaries

Trial extraction reconciles good/bad counters with raw `run_start` and `run_end`
records, including both long and abbreviated variable names. A reversal after
its session's final trial contributes to that session's `reversal_totals`; its
post-reversal window starts at the next available trial. Positive total-counter
changes between runs are included in the receiving session. Recording restarts
subtract their starting counter baseline to avoid counting persisted events twice.
Magnitude-only changes and counter decreases are retained in
`reversal_diagnostics`, without assigning a good/bad reversal type.

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
