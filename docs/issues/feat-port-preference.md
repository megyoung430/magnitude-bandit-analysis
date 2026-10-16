---
type: feature
complexity: complex
status: in review
---

# Port preference: chosen versus best, X versus plus (drifting task)

## Description

PI request: across mice, how much ports on an X from the initiation port (A1, C1, A3, C3) are
chosen versus ports on a plus (B1, B3, A2, C2), normalised by how often they were best; and within
a mouse, the same per port, across problems and for each problem.

Measure: `P(chosen) / P(best)` over the trials where the port (or any port of the category) was a
choice tower. 1 = chosen exactly as often as best.

## Acceptance criteria

- Counting (choice tower, chosen, best per port and category) is tested on synthetic data with ties,
  a port never best and a different initiation tower.
- Ties for best split evenly; ratios are NaN where never best, and those cases are counted and returned.
- Mice are the replicates (mean ± `nan_standard_error`); counts, not ratios, are pooled over problems.
- Figures (PNG + PDF): X vs plus bars with per-mouse lines and a line at 1; per-mouse 3 x 3 maze
  heatmap (log2, diverging, centred on 1, grey for undefined, ratio and n in each cell).
- `run_port_preference_analysis` is on in `DRIFTING_SELECTION` and in the drifting-only guard.

## Decisions

- Geometry is read from each problem's `initiation_tower` (`B2` if not logged, as in the poke
  analysis); sessions of one problem disagreeing raises.
- Categories are relative to the problem's initiation port: plus = orthogonal neighbours, X =
  diagonal neighbours; other ports are "other" and not plotted in the bars.
- A category's "best" is the sum of the tie shares of its ports, so a tie across categories
  gives each half.
- Trials without a choice or magnitudes are skipped (counted per mouse); a choice that was not offered raises.
- The entry point writes both figures into whichever folder it is given, so a problem folder
  also gets its own X vs plus bars (a small extra beyond the request).
- Nothing is written as CSV; DataFrames are returned.

## Results change

New outputs only: `drifting/problem-KK/port-preference/` and
`drifting/all-drifting-problems/port-preference/`, each with `Port Preference X vs Plus` and
`Port Preference Per Mouse` (PNG and PDF). Existing figures are unchanged. A blocked problem with
`run_port_preference_analysis` selected now raises.
