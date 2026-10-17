---
type: feature
complexity: complex
status: in review
---

# Shared sync, import and outcome-event code for photometry

Cohort-04 mice have fiber photometry (pyPhotometry `.ppd`, `2EX_1EM_pulsed`, 130 Hz, analog 1 =
nLightG2 signal, analog 2 = isosbestic control, rsync pulses on digital 1), pyControl behaviour
(`.tsv`) and video (`*_GPIO_data.csv`, rsync on GPIO1). The lab convention is to put everything on
**pyControl time, in seconds**. This item adds the code that does that and that finds the trial
outcome times; the photometry preprocessing notebook imports these modules.

## Modules

- `src/sync/rsync.py`: `RsyncAligner` (alias `Rsync_aligner`), `RsyncError`, `RsyncWarning`,
  `min_pulses_required`. Port of Thomas Akam's `Rsync_aligner` (GPL-3, header kept): same chunk
  matching and numerics. Plotting is gone from `__init__`; the log-MSE values and match labels are
  stored as `log_mse` and `valid_matches` for `plot_rsync_alignment`. Inputs are validated
  (1-D, finite, strictly increasing, at least `2 * chunk_size + 1` pulses). The mixture model is
  seeded (`random_state=0`) so an alignment is reproducible. The upstream order check tested
  `cor_times_A` twice; it now tests A and B. `interval_discrepancy_ms()` gives the interval
  disagreement over matched neighbouring pulses.
- `src/behavior_visualization/plot_rsync_alignment.py`: `plot_rsync_alignment(aligner, axes=None)`,
  three panels, draws into given axes or makes a figure.
- `src/sync/pycontrol_time.py`: pure functions that put other clocks on pyControl seconds:
  `pycontrol_sync_times`, `align_to_pycontrol`, `to_pycontrol_time`, `photometry_sample_times`,
  `video_frame_times`, `sync_quality`. Times outside the matched pulses are NaN (no extrapolation
  unless asked).
- `src/photometry_import/` (I/O only, no preprocessing): `pyphotometry.py` (official `import_ppd`
  unchanged, `read_ppd_header`), `sessions.py` (`find_photometry_sessions`, `PhotometrySession`),
  `recordings.py` (`match_ppd_to_tsv`, `load_photometry_recordings`, `PhotometryRecording`,
  `SkippedRecording`).
- `src/behavior_analysis/outcome_events.py`: `outcome_events`, `outcome_settings`,
  `session_intervals` (pure), `load_outcome_events` (reads a `.tsv`).

## Outcome-time definitions and why

The log order around a trial print (verified on RT_176_L ses-46 and on all 17 staged cohort-04
recordings) fixes what each time means:

- **Rewarded trial** (`num_rew_pulses > 0`): print, then `state choice` a few ms later. The first
  reward pulse starts on entry to `choice`, so `outcome_time` is that entry
  (`outcome_kind = "reward_onset"`). `check_time` is the first `pulse_off_timer` after it, which
  ends pulse 1; `check_time - outcome_time` should be about `pulse_isi`.
- **Unrewarded trial**: print, then `state ITI` with no `choice`. The no-reward cue starts on `ITI`
  entry, so `outcome_time` is that entry (`"no_reward_cue_onset"`). `no_rew_cue_timer` fires
  `no_rew_cue_dur` later, so it marks cue **offset**, not onset; it is `check_time`, to verify the
  onset against the logged duration.
- Only rows between a trial's print and the next print are used. A missing or duplicated expected
  state, a state more than 100 ms after the print, a rewarded trial without `choice`, an unrewarded
  trial with one, or a missing check timer raises `OutcomeParseError` naming trial and file. The
  100 ms limit is generous: observed print-to-state delays are 5-14 ms.
- Traceback rows are dropped by the repo's `drop_non_numeric_times` (through `read_tsv_file`).

## Pairing a `.ppd` with a `.tsv`

A session folder can hold several `.ppd` and `.tsv` files. `match_ppd_to_tsv` aligns the `.ppd`
rsync train with each candidate; a candidate counts only if the aligner accepts it, at least half of
the shorter train is paired, and the median interval error is under 50 ms. The aligner alone is not
enough: it accepted a 21-pulse restart `.ppd` against an unrelated 232-pulse `.tsv` (5 of 21 pulses
paired, errors over 600 ms). A `.ppd` with fewer than `min_pulses=20` pulses is skipped, reported
as a `SkippedRecording` with its reason. A `.ppd` header whose subject differs from the folder's
raises.

## Results change

None for existing analyses. New modules, one new dependency pair (`scikit-learn`, and `packaging`
which `import_ppd` imports and which was only installed indirectly), no existing function changed.
