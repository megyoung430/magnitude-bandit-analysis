---
type: feature
complexity: simple
status: in review
---

# Pool the all-drifting-problems analyses across cohorts

`scripts/full_pipelines/run_all_drifting_problems.py --cohorts 1 3 4 [--no-cache]` runs the
analyses of `drifting/all-drifting-problems/` over the drifting problems of the chosen cohorts
(`src/pipelines/all_cohorts_drifting.py`).

## Acceptance criteria

- Figures in `results/figures/open-field/all-cohorts/{choice-stats,value-psychometrics,
  reversal-aligned-arms,value-change,port-preference}/`, plus `cohorts.txt` (cohorts, mice per
  cohort, run time); tables in `results/tables/open-field/all-cohorts/`.
- Blocked problems ignored; problems of different cohorts stay separate.
- No per-mouse figures, no legend entries naming mice; curve-type legends stay.
- The cohort runner's output is unchanged (new switches default to the old behaviour).

## Decisions

- Mouse ids are not prefixed: a mouse id in two cohorts raises (it would be pooled as one animal).
- Problem keys are `(cohort, k)` tuples; every entry point accepts them (the pooled arm windows
  merge into one `"pooled"` entry, so no folder is named from a key).
- A rerun deletes the five figure folders and three tables it owns, so no file of an earlier
  cohort set survives next to the new `cohorts.txt`.
- New keyword switches, all defaulting to the old behaviour: `per_mouse_heatmaps`
  (`run_value_psychometric_analysis`), `per_mouse_heatmap` and `mouse_legend`
  (`run_port_preference_analysis`), `per_mouse_figures`, `mouse_legends`, `x_vs_plus`
  (`run_drifting_value_analyses`), `mouse_legend` (rank-proportion plots,
  `plot_pooled_drifting_rank`). Value-change already drew no per-mouse curves by default.
- `cohort_analysis._load` is now the public `load_cohort`.

## Results change

None for existing outputs. New outputs under `all-cohorts/` only.
