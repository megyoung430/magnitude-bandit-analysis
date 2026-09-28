# Cohort 03 session-boundary reversal audit

Audited 253 runs with performance-dependent reversal counters (245 consecutive run boundaries, eight mice) from `/Volumes/behrens/meg/3x3_field_magnitude_bandit/rawdata/cohort-03`. Compared the last logged trial with `run_end`, and each `run_end` with the following `run_start`. Both long and abbreviated counter names are supported. Three runs have no end record; missing values are not treated as unchanged.

| Mouse | Boundary | Evidence | Interpretation |
|---|---|---|---|
| MY_66_LL | Session 22, 2026-09-09, last trial → end; carried into session 23 | Local bad counter 0 → 1; total bad 3 at start → 4 at end; A1/C3 magnitudes 1/0 → 0/1 | Confirmed terminal bad reversal missed by trial-only counting. Align post-reversal behavior to first trial of session 23. |
| MY_66_LR | Session 32 → 33, 2026-09-16 | Total good 8 → 9; A3/C3 magnitudes 4/0 → 0/4 | Additional good reversal supported by boundary records; exact trigger time is not recorded in these snapshots. |
| MY_66_R | Session 32 → 33, 2026-09-16 | Total good 9 → 8; A3/C3 magnitudes 0/4 → 4/0 | Counter rollback/state discontinuity; do not classify as an additional good reversal. |
| MY_66_R | Session 31 → 32, 2026-09-15 → 16 | A1/C3 magnitudes 1/0 → 0/1; cumulative totals unchanged | Unclassified magnitude change. |
| MY_66_LR | Session 31 → 32, 2026-09-15 → 16 | A1/C3 magnitudes 0/1 → 1/0; cumulative totals unchanged | Unclassified magnitude change. |
| MY_66_LL | Session 17, 2026-09-02, last trial → end; carried into session 18 | C3 magnitude 1 → 0; local good/bad counters unchanged | Unclassified terminal magnitude change. |

No observed end-to-next-start changes in the total bad counter. The confirmed terminal bad reversal is already present in the end record and persists into the next start, so checking only end-to-start differences would also miss it.

The existing merger uses trial-level session counters and therefore misses the confirmed terminal bad reversal and the boundary-supported good reversal. Magnitude-only changes should remain flagged for review, and negative counter changes must not be counted as reversals. This audit does not modify analysis counts or regenerate figures.

Full snapshots, paths, and comparisons: `session_reversal_audit_cohort03.json`. Reproduce using `python3 scripts/audit_session_reversals.py RAW_ROOT OUTPUT_JSON`.
