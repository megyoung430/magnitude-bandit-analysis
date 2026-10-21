---
type: feature
complexity: moderate
status: in review
---

# Imaging: frames on pyControl time, reward responses by pulse count, most responsive cells

`scripts/funcimg/reward_responses.ipynb` puts mini2p imaging frames on pyControl time with the same rsync aligner as the
photometry, then plots outcome-aligned responses split by `num_rew_pulses` and each run's most responsive cells, for both
destriping variants of the mini2p preprocessing.

## What it adds

- **Sync** (`src/funcimg_import/daq_sync.py`): the acquisition's `file_*.h5` holds two 5 kHz lines, `FrameClock` (one
  rising edge per imaging timepoint, both channels) and `PyControlPulse` (rsync). Edges are midpoint crossings.
  `src/sync/pycontrol_time.imaging_frame_times` aligns them like `photometry_sample_times`.
- **Which files form a run** (`src/funcimg_import/runs.py`): TIFFs (by `source_substring`) and DAQ files are paired in
  acquisition-counter order, as the batch numbers `run-XX`. The behaviour `.tsv` is the one whose rsync pulses align with the
  DAQ's: at least 90 % of pulses matched and interval discrepancies ≤ 10 ms, because the aligner alone accepted an
  unrelated train of 30 pulses in testing.
- **Run identity**: the run numbering is worked out from the raw folder, then checked against the batch's
  `run-XX/<variant>/source_tiff.json` before frame times are written and again when traces are loaded; a mismatch (a
  stray `._` file, a renamed TIFF) is refused, because runs of equal length would otherwise pair one acquisition's traces
  with another's frame clock and behaviour.
- **Frame times** (`src/funcimg_analysis/frame_times.py`): `derivatives/.../funcimg/run-XX/frame_times.npz` with
  `time_s` (pyControl s per frame, NaN outside the sync pulses), `frame_index`, `daq_frame_s`, the three source files and
  the sync quality. Shared by both variants. Frames between the last *matched* rsync pulse and the DAQ's last pulse take
  the extrapolated clock (the trailing pulses rarely fill a 5-pulse chunk, so the last ~30 s of every run would otherwise be
  lost; drift is ~0.1 ms/s); frames outside the DAQ's pulses stay NaN.
- **Responses** (`src/funcimg_analysis/event_frames.py`, `run_responses.py`, `collect.py`): each outcome goes to its
  nearest frame (dropped unless within half a frame of a frame with a known time); windows are whole frames (no
  interpolation), so the x axis reads in frames or seconds. Views: raw F and ΔF/F0 of green and red. ΔF/F0 comes from
  `mini2p_preprocessing.fluorescence.compute_dff` with the fluorescence settings recorded at extraction, so it matches the
  QC audit. Each cell is averaged over its trials per pulse count; the figure shows mean ± SEM across all cells of all
  runs (each cell counts once). Runs without frame times or an extraction, or whose trace length differs from the frame
  clock, are listed with the reason.
- **Top cells**: per run, the `N_TOP` cells with the largest green ΔF/F0 anywhere in the trace, stacked traces with the
  outcomes marked by pulse count, and their outlines on the mean image. One PDF page per run per variant.
- **Optional dependency** `funcimg` (`h5py`, `mini2p-preprocessing` from GitHub); nothing else in the repo needs it.
- **rsync fix**: `_align_chunks` clamps each chunk's mean squared error at 0. The expansion used to compute it cancels to a
  tiny negative number for an exact match of intervals of seconds in ms, which raised in `_classify_chunks` on the first
  real imaging session. Only inputs that raised before are affected.

## Results change

None for existing analyses. Photometry and video alignments never had a negative error (it raised), so they are unchanged.

## Caveats

"Cells" are all Cellpose ROIs (`iscell` is not used). Frame times are frame *starts*; a frame takes ~128 of its 133 ms
to acquire, so an event's frame 0 starts within half a frame of it. `frame_times.npz` is not invalidated automatically:
set `OVERWRITE_FRAME_TIMES` after a change to the sync code. With every chunk an exact match, `RsyncAligner` can still
raise (zero errors are replaced by the smallest non-zero one); it never returns a wrong alignment, and the real imaging
session aligns. The `funcimg` extra brings in suite2p and torch, and CI installs all extras.

0 pulses is aligned to the no-reward cue, 1-4 pulses to reward onset. Raw F is not baseline-subtracted. Ranking by max
ΔF/F0 favours cells with one bright frame; check the traces. Cells of different runs of the same field of view may be the
same neurons, and trials of a session are not independent. Outputs go to `results/figures/funcimg/` (git-ignored).

## Choosing sessions and runs

`SESSION_FIRST, SESSION_LAST` pick the session range (the same number twice for one session); folders without a
session number (`ses-baseplating`, `ses-test`) are skipped. `RUNS = None` uses every run; `RUNS = ("run-02",)` uses
only those run folders of each session, in all three steps (frame times, pooled responses, top cells). Runs not named
are neither pooled nor listed as left out; naming a run that no session in the range has raises an error listing the
runs that exist. Figure titles and file names carry the sessions and runs (`pulse_responses_ses-220-220_run-02_…`), so
a single-run figure does not overwrite the all-runs one.
