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

- the module's `ALGORITHM_VERSION` (an integer, currently `1`);
- the configuration: a fingerprint of the input sequences, the settings (seed, number
  of starts, grid, models, problems), the bounds, constants such as the at-bound
  tolerance, and for the paper-style sweeps the jax, optax and numpy versions.

It does **not** cover source text. Editing a comment, reformatting or refactoring keeps
the folder, so an interrupted run resumes. `manifest.json` still records the source text
of the code that wrote the folder, as provenance only.

**When to bump `ALGORITHM_VERSION`:** when a change alters results (the model, the loss,
the optimiser, the simulated task or mouse, the scoring), including in code a module
calls, such as `best_rl.py`, `released_rl.py` or `neural_bandit.py`. A bump starts every
run in a new folder; comments and refactors don't need one. A configuration change (a
different seed, bounds, data or library version) moves the folder without any bump.
Finished results always stay readable in the old folder.

**Migration.** The change that introduced the version scheme moved every folder one last
time: runs started before it restart from zero in a new folder (finished results stay in
the old one). From then on, only a version bump or a configuration change moves them.

## Code organisation

| Module | What it does |
|---|---|
| `src/behavior_import/` | Parses raw pyControl `.tsv` logs into trial-level structures. `import_data.py` is the main entry point; handles field-name aliasing across protocol versions. |
| `src/behavior_analysis/` | Computes derived metrics from imported trials — reversals, choice probabilities, task statistics, GLM utilities, cross-problem summaries. |
| `src/behavior_visualization/` | One plotting function per analysis in `behavior_analysis/`. |
| `src/pipelines/` | End-to-end analysis pipelines as functions with typed settings (`open_field.py`: every standard open-field figure, per problem). Scripts and notebooks call these rather than copying or exec-ing script code. |
| `src/task_simulations/` | Synthetic bandit sessions (GP-driven and Markov-chain reward dynamics) for comparing real behavior against simulated agents. |
| `scripts/individual_analyses/` | Standalone scripts, one per analysis/plot type — thin wrappers around `src/`. |
| `scripts/full_pipelines/` | Entry points for full cohort runs (`run_open_field_analysis.py` calls `src/pipelines/open_field.py`; `run_grid_maze_analysis.py`) and the cohort notebooks. |
| `scripts/to_integrate/` | Exploratory notebooks not yet folded into `src/`. |

## Getting started

1. Complete the [Setup](#setup) steps above.
2. Open `scripts/bandit_behavior_tutorial.ipynb` — this walks through loading
   a session, building a trial-level DataFrame, and reproducing the core
   plots.
3. For a specific analysis, check `scripts/individual_analyses/` for a
   standalone example, or `scripts/full_pipelines/` for a full pipeline.

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
