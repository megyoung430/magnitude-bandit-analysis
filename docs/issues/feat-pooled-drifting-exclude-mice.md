---
type: feat
complexity: moderate
status: in review
---

# Pooled drifting set without chosen mice, and a pooled-only rerun

Requested: for cohort 3 drifting, another set of all-drifting-problems plots without MY_175_N and
MY_66_LL, without regenerating every problem folder.

## Acceptance criteria
- `exclude_mice_from_problems` drops the named mice, raises (listing the mice that exist) for a
  mouse that is not there, drops problems left empty, and raises when no mouse is left.
- `run_cohort(exclude_mice=...)` / `run_cohort_incremental` / `run_cohorts` write
  `drifting/all-drifting-problems-without-<sorted mice>/` (same subfolders, sessions-per-problem
  included) whenever the plain pooled set is written; the plain set and the stacked diagnostic
  tables are unchanged. Tables go to `tables/open-field/<cohort>/all-drifting-problems-without-…/`;
  `excluded_mice.txt` in both folders lists the mice and the run time.
- Every figure of the extra set is marked "without MY_175_N, MY_66_LL".
- `run_cohorts.py --pooled-only` runs only the pooled analyses (plain and extra) and the review
  tables; `--exclude-mice` works with it, with a full run and with `--last-sessions`/`--since`;
  `--dry-run` prints the plan including the extra set. Names are validated before any figure or
  table is written.
- `run_all_drifting_problems.py --exclude-mice` writes `all-cohorts-without-<mice>/`.

## Decisions (PI to check)
- The mark is a bold line above the top-left of every figure, added where figures are written
  (`figure_banner` in `plot_helpers`), because rank, psychometric and value-change figures have no
  scope label in their titles. The figures that do (reversal-aligned arms, port preference,
  sessions per problem) also carry the text in their own titles.
- A pooled-only run skips `trial-block-summaries/` and leaves the stacked diagnostic tables
  alone (it has no per-problem rows to add); it does write the review tables, as asked.
- With several cohorts every listed cohort is checked (imported, from the cache if present)
  before the first one runs, and must contain all named mice; name the cohort that holds them.
- In a partial run that selects no drifting problem the extra set's analyses are skipped like
  the plain pool's, but its sessions-per-problem figure and table are redrawn, as the plain
  ones are. The extra figure and table folders are emptied before its analyses are rewritten.
- A wrong mouse name is a one-line error in both scripts (no traceback).
- An empty `--exclude-mice` list is an error rather than "no exclusion".
- `excluded_mice.txt` rather than a README.

## Results change
- New optional folders only: `drifting/all-drifting-problems-without-…/` (figures),
  `tables/open-field/<cohort>/all-drifting-problems-without-…/`, and with the all-cohorts script
  `all-cohorts-without-…/`. Nothing a run without the options writes changes.
