# refactor-subpackages: one subfolder per analysis

## What
`src/behavior_analysis`, `src/behavior_visualization` and `src/pipelines` were flat folders of
95, 37 and 27 modules. Each now has one subfolder per analysis, with the same names in all three,
so an analysis, its figures and its pipeline sit in folders of the same name:

| Subfolder | Analysis | Figures | Pipelines |
|---|---|---|---|
| `data/` (analysis) / `common/` (figures, pipelines) | trial table, believed values, problem selection, shared statistics | style, save step, legends, maze drawings | cohort runners, cache provenance, resumable runs |
| `reversals/` | reversal windows, choice probabilities and first leaves around good reversals, arm ranks around block changes, reversal counts | yes | yes |
| `rank_proportions/` | best/second/third choice proportions | yes | |
| `task_statistics/` | blocks, sessions, trials, cohort comparisons, single sessions | yes | yes |
| `pokes/` | pokes, port finding, port preference | yes | yes |
| `value/` | value psychometrics, stay/switch, stay by run length, value changes | yes | yes |
| `choice_logit/` | choice logit, its CV and extensions, choice regression, kernel timescale | yes | yes |
| `switching/` | switch regression, second best | yes | yes |
| `strategy_models/` | strategy-model comparison | yes (with null agents, kernel Q, agent baselines) | yes |
| `rank_tests/` | ranked value algorithm tests | yes | yes |
| `glm_hmm/` | GLM-HMM | yes | yes |
| `review/` | tables for checking the data | | |

## How
File names are unchanged; only folders changed. Every reference was rewritten: imports
(`from src.behavior_analysis import x` becomes `from src.behavior_analysis.<group> import x`),
dotted names (including `mock.patch` targets and `importlib` strings in notebooks), and file paths in
docs, the code-map notes and the README. Notebook outputs are untouched. Docstring and comment lines
that grew past 100 characters were rewrapped. The code map now lists files under their own folder.

## Results change
None for results. joblib caches are keyed on each function's module, so the first run after this
change recomputes cached analysis steps (the import cache of `behavior_import` is not affected).
