---
type: feature
complexity: moderate
status: in review
---

# Photometry: outcome responses split by number of reward pulses

Section 7e of `scripts/photometry/cohort_photometry_preprocessing.ipynb` splits the outcome-aligned responses of
`feat-photometry-notebook` by `num_rew_pulses` (0 = unrewarded, 1 to 4), for both pipelines.

## What it adds

- `src/photometry_analysis/pulse_groups.py` (tests in `tests/test_pulse_groups.py`): `pulse_categories`,
  `split_by_pulses`, `categories_with_min_trials`, `pulse_response_correlation` (Spearman over rewarded trials; no p-value).
  The photometry preprocessing stays in the notebook; the split is generic and lives in `src/`.
- Configuration: `PULSE_PHASES = ("task",)` (port-finding 2-pulse trials come from a phase with all arms equal and are
  left out; add `"port_finding"` to include them) and `MIN_TRIALS_PER_PULSE = 5` (a mouse's category is drawn, enters group
  means and the correlation only with that many usable trials; dropped ones are printed).
- Windows are normalised per recording exactly as in section 7 and split afterwards, so the split cannot change a trace.
  `real_condition_windows` now also returns the pulse count of each used trial; `build_real` takes a `conditions` argument.
- Figures (`results/figures/photometry/cohort-04/`): `pulses_per_mouse_pipeline_{A,B}`, the 405-nm control versions
  `..._control_own_scale`, `pulses_group_comparison`, `pulses_single_mouse_comparison`. Table: `pulse_responses.csv`
  (mean z response 0-2 s minus -2-0 s per category with n, and rho) and `pulse_trial_counts.csv`.

## Caveats

0 pulses is aligned to the no-reward cue and 1-4 pulses to reward onset (different events). Reward size is confounded with
arm, value and time at the port (4 pulses take about 0.9 s longer). The 2- and 3-pulse lines come from `RT_177_N` and
`RT_177_LR` only, so group-figure lines average different mice (dashed, mice named). Trials of a session are not independent.

## Results change

None for existing analyses; new outputs only (git-ignored).
