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

### Import cache

Parsing every raw `.tsv` is the slow part of an import, so imports go through a cache in
`.cache/` (git-ignored; override with `BANDIT_CACHE_DIR`; see
`src/behavior_import/load_trials.py` and `cache_store.py`):

- each file's parse is kept, keyed on its path, size, modification time and the parsing code,
  so a refresh parses only new or changed files and reuses the rest;
- the assembled cohort (sessions, problem numbers, trials, import log) is stored as a
  snapshot, with a small `manifest.json` listing the files it was made from.

**Runs read the cache by default.** `run_cohorts.py` and `run_all_drifting_problems.py` load
each cohort from its newest snapshot without touching the server, and print one line per
cohort with the snapshot's date and file count. Sessions added on the server since are
missing: a `StaleCacheWarning` (also in `warnings_log.csv`) and a `from-cache.txt` in every
output folder say so. A cohort with only an old-style cache (from before snapshots) uses its
newest one the same way; a cohort with no cache at all is imported from the server and cached.

```bash
uv run python scripts/full_pipelines/import_cache.py check --cohorts 1 3 4    # what is missing?
uv run python scripts/full_pipelines/import_cache.py refresh --cohorts 3      # include new data
uv run python scripts/full_pipelines/import_cache.py prune --dry-run          # what can go?
uv run python scripts/full_pipelines/run_cohorts.py --cohorts 3 --refresh-cache  # refresh, then run
```

- `check` lists the server (sizes and dates only) and says, per cohort, when its cache was made,
  whether the import or parsing code (or pandas) changed since, and which subjects, sessions
  and files are new, changed, removed or were skipped as unreadable. It exits 0 when a refresh
  would change nothing, 1 otherwise, and 2 when the server or the settings are missing;
  `--json` prints the same as JSON.
- `refresh` parses the new and changed files, writes a new snapshot and then prunes that raw
  folder's older snapshots (`--keep-old` to skip). If the server stops answering part-way it
  stops with an error and the previous snapshot stays in use; a single unreadable file is
  skipped, as a plain import skips it, and tried again by the next refresh. Snapshots are
  written to a temporary folder and renamed into place.
- `prune` removes snapshots older than the newest of the same raw folder (a partial copy at
  another path keeps its own), old-style caches that a snapshot of their folder has replaced,
  per-file parses no kept snapshot uses, and leftovers of interrupted writes, printing the
  space freed. A cohort's only cache is never removed.

`--refresh-cache` on either run script refreshes the listed cohorts first; `--no-cache`
(`run_all_drifting_problems.py`) re-imports everything without the cache. `--from-cache` is
still accepted and changes nothing.

In code (notebooks), `load_trials_with_log(root)` refreshes **and prunes**: the first call on a
cohort deletes its old-style caches once its snapshot exists, and later calls remove its older
snapshots. `load_trials_with_log(root, from_cache=True)` is cache-first, and `use_cache=False`
leaves the cache alone. Nothing is ever written under the data root: a refresh or prune stops
with an error if `BANDIT_CACHE_DIR` points inside `BANDIT_DATA_ROOT`. Delete `.cache/` at any
time; it only costs a re-import.

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

- **Pooled drifting analyses only, and a pooled set without some mice.** To redo just
  `drifting/all-drifting-problems/` (no problem folders) and also get the same analyses without
  chosen mice:

  ```bash
  uv run python scripts/full_pipelines/run_cohorts.py --cohorts 3 --pooled-only --exclude-mice MY_175_N MY_66_LL
  uv run python scripts/full_pipelines/run_cohorts.py --cohorts 3 --pooled-only --exclude-mice MY_175_N MY_66_LL --dry-run
  ```

  `--pooled-only` writes the pooled figures (with `sessions-per-problem/`) and the review tables
  and leaves the problem folders, `trial-block-summaries/` and the stacked diagnostic tables as
  they are; `warnings_log.csv` is rewritten with this run's warnings only; it cannot be combined
  with `--last-sessions`/`--since`. `--exclude-mice` writes a
  second set, `drifting/all-drifting-problems-without-MY_175_N-MY_66_LL/` (sorted ids, same
  subfolders), next to the normal one, which is unchanged. Every figure's caption file in it
  starts with "without MY_175_N, MY_66_LL"; its tables are in
  `tables/open-field/cohort-03/all-drifting-problems-without-MY_175_N-MY_66_LL/` (not in the
  stacked diagnostic tables), and `excluded_mice.txt` in both folders lists the excluded mice and
  the run time. Both folders are emptied before the analyses are rewritten; the extra set's
  `sessions-per-problem/` figure and table are redrawn on every run of a cohort with drifting
  problems, like the normal one. The option also works with a full run and with `--last-sessions`/`--since` (the
  extra set is regenerated whenever the pooled set is). Names are checked against the cohort's
  drifting mice before anything is written: a mouse that is not there, or a list that leaves no
  mouse, stops the run with a one-line error listing the mice that exist. With several cohorts
  every cohort is checked first and must hold all the named mice, so pass the cohort that has
  them.
  `run_all_drifting_problems.py --exclude-mice ...` does the same across cohorts into
  `all-cohorts-without-<mice>/`.

- **Redoing only what recent sessions changed.** After a few new sessions the whole suite need
  not be rerun:

  ```bash
  uv run python scripts/full_pipelines/run_cohorts.py --cohorts 4 --last-sessions 4
  uv run python scripts/full_pipelines/run_cohorts.py --cohorts 3 4 --since 2026-10-16
  uv run python scripts/full_pipelines/run_cohorts.py --cohorts 4 --last-sessions 4 --dry-run
  ```

  `--last-sessions N` takes each mouse's N most recent sessions (by session number and date; a
  mouse with fewer contributes all it has), `--since YYYY-MM-DD` the sessions dated on or after
  that day; give one of them. `--last-sessions` takes each mouse's last N sessions however old
  they are, so a mouse that stopped weeks ago has its final problem regenerated every time; use
  `--since` for "what is new since I last ran". A problem folder holds every mouse that ran the problem, so every
  problem containing a selected session is regenerated whole; the other problem folders are left
  untouched. `drifting/all-drifting-problems/` is regenerated (from all drifting problems) only
  if a selected problem is drifting. The review tables and `trial-block-summaries/` are always
  rewritten. In `poke_trial_summary.csv`, `port_finding_check.csv`, `reversal_aligned_arms_counts.csv`,
  `value_change_events.csv`, `value_change_bin_counts.csv` and `value_change_symmetric_exclusions.csv` the rows of the regenerated problems
  replace the earlier ones and the rows of other problems are kept (matched on `task_type` and
  `problem_in_type`); if such a table is missing, a warning says that only the new rows were
  written and the table is partial. `warnings_log.csv` covers the latest run only. The run
  prints the plan first (problems, pooled scopes and the sessions behind each) and a summary of
  what it wrote at the end; `--dry-run` prints the plan and writes no results or tables (the import cache may be filled). A bad N, a date that
  is not `YYYY-MM-DD`, or a selection matching no session stops with an error (which lists the
  mice and dates that exist); with several cohorts, one without a matching session is skipped
  with a `[SKIP]` line, and the run stops with an error only if none matches. Problem numbers
  come from the full cohort, so a session that starts a new problem makes that problem's number
  new, while folders and rows of problems that no longer exist under that number stay on disk:
  do a full run after renumbering. A run without either option is the full run, unchanged.
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
  subfolders `mouse-weighted/` (each mouse's events pooled into one curve, mean ± se across mice)
  and `event-weighted/` (every event counts once, mean ± se across events; the se treats events
  as independent, which events of one mouse are not, so it is narrower than the real
  uncertainty). Nothing is written at the top level of those two folders. The
  `reversal_aligned_arms_counts` table has a `weighting` column (`subject` or `event`).
  Each value-change configuration also gets a **complete-window** version in the same two
  folders (`... (All Events, -2 to +4 Trials).png`): windows of 2 trials before to 4 after the
  change, keeping only events whose whole window is intact (no block or third-arm change, session
  edge, best-arm change before the event, or missing choice), so every time point averages the
  same events. A block or best arm that starts exactly at t = -pre still counts as complete, and a
  best/second tie in the pre window counts as a best-arm change. It has no rank-swap truncation.
  `COMPLETE_WINDOW = (2, 4)` in `src/pipelines/value_change_analysis.py` sets the default; change
  it or switch it off per call with `run_value_change_analysis(..., complete_window=(pre, post) |
  None)`. Equal sides keep the old names (`Symmetric ±6 Trials`, mode `symmetric_6`); complete-window
  figures of another size left from an earlier run are deleted when the folder is rewritten. The
  events kept and dropped per configuration are in `value_change_symmetric_exclusions.csv` (name
  kept; columns `pre` and `post` give the window): `n_excluded_*` count each event once, by its
  first cause (block/third arm, session edge, best arm, other), and `n_with_*` count the events
  having each cause, which overlap; the complete-window mode (`complete_pre2_post4`) is also in
  `value_change_bin_counts.csv`.
- **Stay/switch psychometrics** (`value-psychometrics-stay-switch/`, made wherever
  `value-psychometrics/` is): on each trial t whose previous trial is in the same session, the
  arm chosen on t-1 is the chosen option; with the true or believed values of trial t, V(chosen)
  is its value and V(alt) the higher value of the two other arms. `Stay-Switch Curves (<Value>
  Value) (Ties <Excluded|Split>) (Pool Trials <True|False>)` plots P(stay) and P(switch), then
  P(switch to the best alternative) and P(switch to the other arm), against V(chosen) - V(alt),
  V(chosen) and V(alt) (each mouse thin, mean ± se across mice); `Stay Heatmap (<Value> Value)
  (Ties ...)` shows P(stay) over V(chosen) x V(alt) with the pairs per cell. Pairs are kept
  across block changes and never span sessions or a missed trial. Ties are named relative to the
  chosen arm (V(chosen) = V(alt), the two alternatives tie, all three): excluded, or kept with a
  switch between tied alternatives split half and half (see
  `src/behavior_analysis/stay_switch_psychometrics.py`).
- **Value-change diagnostic** (`scripts/individual_analyses/plot_value_change_diagnostic.py --cohorts 1 3
  [--no-cache]`): one PNG per drifting session in `figures/value-change-diagnostic/cohort-XX/problem-KK/<mouse>/`,
  showing the session's magnitudes and choices with each best-decrease rank-swap event's counted pre and post
  windows shaded, the choices coloured by label and the arm behind each label written underneath; `index.txt`
  lists every session's events. It also writes `mouse_curves_<scope>.csv` (`problem-KK`, `all-drifting-problems`):
  each mouse's event count, `prev_best`/`prev_second`/`third` choice probabilities and events per t (both truncation
  modes), with a `mean_across_mice` row equal to the mouse-weighted value-change curve.
- **Port preference** (`run_port_preference_analysis`, drifting problems, `port-preference/`, also
  pooled over all drifting problems): `P(chosen) / P(best)` per port (one maze per mouse) and for
  the X and plus ports (mean ± se across mice). Relative to the problem's initiation port, plus
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
  figures do not show them). Port-finding trials are never in the task-trial maps.
  **Port finding can span sessions**: a mouse that does not finish starts the next session in
  port finding again, and the task remembers the ports already found. A session that holds only
  port finding has no bandit trials, so the importer skips it (kind `only_port_finding` in
  `no_trial_sessions.csv`; it stays out of `trials_by_problem` and every task analysis) but keeps
  its raw recordings in `ImportLog.port_finding_sessions` (`{mouse: {session: record}}`, cached
  with the import). `assign_port_finding_sessions` gives each one to the problem of the mouse's
  next session with bandit trials if that session has the same choice towers (the `c_tows` of
  the port-finding prints) and initiation tower; several in a row before one problem all go to
  it. One with other towers, or without a later session, is not used: a warning (also in
  `warnings_log.csv`) names the mouse, session and towers. The port-finding maps and their
  panel titles of the problem count the assigned sessions' trials; the task maps, titles and
  long-pokes means do not change. `poke_trial_summary.csv` has a row for each such session
  (`port_finding_only` true, `n_trials` 0, the `n_port_finding_*` columns filled), so every
  counted trial is traceable.
  **Port-finding check** (`port_finding_check.csv`, one row per problem and mouse with
  port-finding trials): `n_port_finding_trials` (printed `num_t_found` trials over the problem's
  sessions, port-finding-only ones included), `n_ports`, `found_t_thresh` (the latest value
  set by hand before port finding ended, else the first session's `run_start`) and
  `expected_min_port_finding_trials` = `n_ports x found_t_thresh`. `below_expected_min` marks a
  row under the minimum. `flagged` is true (with a warning, never an error) whatever the total
  when the first session with port finding starts with non-zero `found_port_counts` (listed in
  `carried_in_counts`: port finding began in a session that is missing from the data or not
  assigned, see the warnings) or when a `user_set` sets `found_port_counts`, `found_t_thresh` or
  `found_ports` by hand; `flag_reason` names the causes. Every session's raw events are parsed once for both. A session whose raw events cannot
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
  | `drifting/problem-KK/value-psychometrics-stay-switch/` | stay/switch curves and P(stay) heatmaps |
  | `drifting/problem-KK/reversal-aligned-arms/` | arm choice probability and magnitude around block changes |
  | `drifting/problem-KK/value-change/` | choice probabilities around value changes |
  | `drifting/problem-KK/port-preference/` | port preference per port (X vs plus only in the pooled folder) |
  | `drifting/problem-KK/long-pokes/`, `.../single-sessions/` | as for blocked |
  | `drifting/all-drifting-problems/choice-stats/` | rank proportions pooled over the drifting problems |
  | `drifting/all-drifting-problems/{value-psychometrics,value-psychometrics-stay-switch,reversal-aligned-arms,value-change,port-preference}/` | the drifting-only analyses pooled over the drifting problems |
  | `drifting/all-drifting-problems/sessions-per-problem/` | sessions each mouse ran per drifting problem: a line per mouse (a problem still running is hollow and dashed) and the mean ± SE of the completed problems; redrawn on every run (also a blocked-only partial one), because any new session can complete a problem |
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
  | `no_trial_sessions.csv` | file or recording the importer skipped, or session with no trials (one row per session, `files` lists its files), with `kind` and `reason`; a session of kind `only_port_finding` is skipped for the task analyses but its port-finding trials are used in the port-finding poke maps |
  | `no_second_best.csv` | block with more than one arm at 0, with the only rewarded arm |
  | `port_finding_check.csv` | mouse and problem with port-finding trials: trials against `n_ports x found_t_thresh`, `below_expected_min` and `flag_reason` (see the long-pokes bullet) |
  | `poke_trial_summary.csv`, `reversal_aligned_arms_counts.csv`, `value_change_events.csv`, `value_change_bin_counts.csv`, `value_change_symmetric_exclusions.csv` | diagnostics of the analyses, stacked over problems with `task_type`, `problem_in_type` (and `overall_problem` where a mouse is named); `poke_trial_summary.csv` has one row per mouse and session, port-finding-only sessions included (`port_finding_only`) |
  | `sessions_per_problem.csv` | mouse and drifting problem: `n_sessions` (sessions with bandit trials; `_part-NN` parts count once), `n_port_finding_only_sessions` (assigned port-finding-only sessions), `n_sessions_total`, `first_session`, `last_session`, and `complete` and `provisional`; always rewritten from the whole cohort. A mouse's latest problem of any task type may still be running (`complete` False); if it is provisional (a final one-session run on a new layout, which a later session can turn into a settings error) the problem before it is not yet complete either. A newest port-finding-only session that no problem took is not a problem, so the problem before it counts as still running (conservative). A session split into `_part-NN` parts counts once per problem, so one split across two problems counts in both. `n_sessions` equals `problem_map.csv`'s |
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
`value-psychometrics-stay-switch`, `reversal-aligned-arms`, `value-change`, `port-preference`, `sessions-per-problem`, `port-finding-summary`) from the drifting problems of the
listed cohorts together (`--no-cache` re-imports the raw files). They go to
`results/figures/open-field/all-cohorts/<same subfolders>/`, with `cohorts.txt` there naming the
cohorts pooled, the mice per cohort and the time of the run; the diagnostic tables
(`reversal_aligned_arms_counts.csv`, `value_change_events.csv`, `value_change_bin_counts.csv`, `value_change_symmetric_exclusions.csv`) go
to `results/tables/open-field/all-cohorts/`, with `sessions_per_problem.csv` (a `cohort` column in front)
as a fifth table. `sessions-per-problem/` also has a figure with one mean line per cohort (n mice at each point)
when several cohorts are pooled. A rerun replaces those seven folders and five tables, so
the files always match `cohorts.txt`; the captions say "all drifting problems, cohorts 1, 3, 4".

- Problems stay separate: each is keyed `(cohort, drifting problem number)`, beliefs reset at each
  problem, and windows and events never cross a problem boundary.
- Mouse ids must be unique across cohorts; a mouse in two cohorts stops the run with an error
  naming it. A cohort without drifting problems also stops it.
- There are no per-mouse figures (the "Per Mouse" value heatmaps and port-preference mazes are in
  the cohort folders) and no legends naming mice; the per-mouse lines are still drawn, and the
  legends for curve types (P(best), P(second), ...) stay. The code is
  `src/pipelines/all_cohorts_drifting.py`.

### Port-finding summary

Where mice poke while they find a new problem's ports, and just after. For each mouse and problem
the pokes are counted per port in three windows:

- **Port finding**: every port-finding trial of the problem, port-finding-only sessions included
  (as in the per-mouse port-finding maps);
- **First 50 Trials After Finding**: the problem's first 50 task trials, over as many sessions as
  it takes;
- **First 3 Blocks After Finding**: the task trials of the first three blocks. Block numbers restart
  in every session, so each session starts a new block.

Both poke kinds of the long-pokes figures are counted with the same rules (long pokes after
initiation; port visits after the choice). A mouse is left out of a problem unless it has
port-finding trials, at least 50 task trials and a complete third block (one still in its first
problem sessions, for example); every caption lists who was left out and why.

- **Heatmaps**: each mouse's pokes become percentages of its own pokes in the window and are
  averaged across mice, so mice weigh equally. One row of three maps per port layout (mice of one
  problem can have different current or previous ports), with the usual green/red/blue outlines.
  Per problem, in `<blocked|drifting>/problem-KK/long-pokes/Port Finding Summary - <kind>`; across
  cohorts, pooled by layout (the same layout is often a different problem number in each cohort),
  in `all-cohorts/port-finding-summary/Port Finding Summary - <ports> After <previous ports> - <kind>`.
- **Port categories** (`Port Finding Summary - Port Categories`): the fraction of pokes at the
  current problem's ports, the previous problem's ports, the initiation port and the other ports,
  as grouped bars per window (mean ± se across mice, a dot per mouse). A port of both problems
  counts as current; each mouse is averaged over its drifting problems first. Per cohort in
  `drifting/all-drifting-problems/port-finding-summary/`, across cohorts in
  `all-cohorts/port-finding-summary/`.

They run with the poke analysis (`run_poke_analysis`). The code is
`src/behavior_analysis/port_finding_summary.py` (counts),
`src/behavior_visualization/plot_port_finding_summary.py` (figures) and
`src/pipelines/port_finding_summary.py` (files).

### Agent baselines and the choice regression

```bash
uv run python scripts/full_pipelines/run_agent_baselines.py --cohorts 1 3 --jobs 4
```

checks what the pooled drifting figures can and cannot show about magnitude learning. It pools
the drifting problems of the listed cohorts (as above) and:

- **Measures the task rules** from the trials (`src/behavior_analysis/drifting_task_rules.py`):
  at each block change, whether the arm set to 0 was the one the mouse chose most in that block,
  and whether the arm that was at 0 took its value; how often non-zero arms drift; the block
  lengths (`task_rules_summary.csv`). The simulated task zeroes the most chosen arm and gives
  the old zero arm its value, so a block change removes the mouse's current favourite and
  moves its value to the arm it was avoiding; the run stops with an error if the data follow
  that rule in fewer than 80% (most chosen) or 95% (value inherited) of block changes.
- **Fits a choice regression** per mouse (`src/behavior_analysis/choice_regression.py`): a
  conditional logit over the three ports with the value the mouse last received at each port,
  whether that visit paid 0, whether the port is still unvisited, stay, and plus-port bias.
  The `last_value` weight is magnitude sensitivity beyond avoiding the empty port, repeating
  the last choice and the spatial bias.
- **Fits two agents** to each mouse (`src/agents/value_agent.py`, a four-parameter value learner
  with stickiness and a plus-port bias): one learns from the reward magnitude, the other only
  from whether there was a reward (1 and 4 both count as 1). Same number of parameters, so
  their likelihoods compare directly; the binary agent is the null model that can avoid the
  empty port, perseverate and prefer ports, but cannot see magnitude.
- **Simulates each fitted agent** (`src/agents/drifting_simulation.py`) through the mouse's own
  problems (ports, initiation port, starting values, trials per session) under the measured
  task rules, and runs the simulated animals through the same pooled drifting figures and the
  same regression as the mice.

Figures go to `results/figures/open-field/agent-baselines/cohorts-<ids>/`: `summary.png`
(regression weights of mice and both agents; per-mouse likelihood difference) and
`magnitude-agent/` and `binary-agent/` with the subfolders of `all-cohorts/`, to be compared
with the mice's own figures there. Tables (`task_rules_*.csv`, `agent_fits.csv`,
`choice_regression_*.csv`, and each agent's diagnostic tables) go to the matching `tables/`
folder. Fits are cached in `.cache/agent_baselines/`, keyed on each mouse's trials, the
settings and `value_agent.MODEL_VERSION` (bump it when the model changes); `--no-cache`
refits. Drift rate and block lengths are pooled over the cohorts given, so pool cohorts that
ran with the same task settings. The code is `src/pipelines/agent_baselines.py`.

### Figure files, titles and captions

Every figure is saved as `.pdf`, `.png` (300 dpi; 180 dpi for the review summaries) and `.svg`,
with `<name>.caption.txt` next to it; the file names are the same as before.

- **Titles** are short: the heading (e.g. "Choice Probabilities Around Block Changes (All)") and,
  for a panel that shows a mean with an error bar or band, one line naming the statistic, what it
  is taken across and the n, e.g. "mean ± se across mice, n=13 mice" or
  "mean ± se across events, n=120 events, n=13 mice". "± se" appears only when an error is
  drawn (one mouse gives "mean").
- **Captions** hold everything else that used to be in titles, subtitles and notes on the figure:
  the scope (cohort, problem, "without <mice>" for an exclusion set), labelling, smoothing,
  window mode, weighting, tie handling, left-out events and caveats. The file is written for
  every figure, empty when there is nothing to add. The code is `Title`, `set_title` and
  `add_caption` in `src/behavior_visualization/plot_helpers.py`.
- **Legends and labels** are tidied when a figure is saved
  (`src/behavior_visualization/plot_legends.py`): no frame, short handles, moved out of the data
  (below the axes when short, beside them for long mouse lists), drawn once when repeated on
  several panels, and a legend entry for a single mouse goes to the caption as
  "line: MY_175_L". Legend, axis and colour-bar labels are in title case (articles,
  prepositions and the statistics "se"/"std" lower case; units in parentheses as written).
- **Font**: Arial (then Helvetica, Liberation Sans, DejaVu Sans where Arial is missing), with
  editable text in the PDF and SVG files. It is set in `plot_style.FONT_RC`, applied when the
  plotting package is imported; font sizes are those of each plot.
- **Mouse colours** (`plot_style.MOUSE_COLORS`): the Okabe-Ito colours (orange, sky blue,
  vermillion, reddish purple, yellow, blue, bluish green), then a darker shade of each in the
  same order, so up to 14 mice in one figure get different colours.

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
