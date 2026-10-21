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
  with another's frame clock and behaviour. A variant folder without a `source_tiff.json` (e.g. `with_destriping` when
  the batch wrote no record) is accepted if it has processed `.tif`/`.tiff` files and every one is named
  `<source stem>_dynamic_bidiphase[_destriped][_undistorted]` (the batch's output names; `._` AppleDouble files ignored,
  `_retry_...` or a further acquisition number refused; stems and the processed suffix are compared exactly, extensions ignore case, the same rule as for the batch record's TIFF name); the record is read from whichever variant folder is checked; a `UserWarning` names the run, variant and TIFFs used. No TIFF,
  another stem, or a record naming another TIFF is still refused. Variant folders that do not exist are not checked, but
  at least one listed variant must exist. The warning is raised once per run and step (frame times, pooled responses, top
  cells), not per frame.
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
- **Z-scored population responses** (`event_frames.robust_z`, `baseline_subtract`): cells with the same trend but
  different raw F or ΔF/F0 scale average out, so the default views are robust z-scores: per cell and run,
  `(x - median) / (1.4826 * MAD)` over the whole trace (NaN-safe), then each trial's mean over the pre-event part of the
  window (offsets < 0) is subtracted before averaging over trials; mean ± SEM across cells as for the raw views. A cell
  with MAD 0 (or no finite value) has no scale: its z is NaN, it adds nothing to the pool, and it is listed per run
  (`flat_cells`, printed by the notebook). Only the requested `measures` are computed (`F`, `dF/F0`, `F z`, `dF/F0 z`; the notebook's `MEASURES`, default z;
  titles and y labels say "robust z, baseline-subtracted" and that F is not neuropil-corrected; file names carry the
  measure set). `measures` are validated against `MEASURES` (`F`, `dF/F0`, `F z`, `dF/F0 z`; the background accepts `F`, `F z`) by
  `check_measures`, which raises a `ValueError` naming the unknown names and listing the valid ones, in `pulse_responses`,
  `pool_runs` and both `collect_*` functions (before any run is read). A z measure needs at least 2 frames before the outcome
  (a window starting at 0 s, or with one pre-event frame, raises unless raw measures are asked for). For an event that falls
  before its nearest frame's start, the frame at offset -1 was being acquired when it occurred and is left out of that
  trial's baseline; a trial left with no baseline frame at all is excluded from `n_trials` and counted in the run's
  `n_trials_no_baseline` (printed by the notebook; with the 2-frame minimum it is 0 unless the guard is relaxed).
- **Top cells**: per run, the `N_TOP` cells with the largest green ΔF/F0 anywhere in the trace, stacked traces with the
  outcomes marked by pulse count, and their outlines on the mean image. One PDF page per run per variant. Each ROI has one
  colour (matplotlib `tab10`, cycling in rank order, best ROI on top) shared by its trace and its outline and number in the
  field of view; traces are 0.8 pt, with `ROI <n> (p=<p(cell)>, max <max>)` as y tick labels at each baseline, drawn clipped to each ROI's
  own 1st-99th percentile (`clip_percentiles`, `CLIP_PERCENTILES` in the notebook; None = unclipped; display only, the
  ranking uses the unclipped values; validated: two numbers, `0 <= low < high <= 100`). Clipping flattens every peak above
  the 99th percentile, so the label carries the ROI's true full-run maximum ΔF/F0 and a small tick on the x axis, in the ROI's
  colour, marks the frame of that maximum when it is among the frames shown. Rows are spaced by one constant distance: the largest, over adjacent rows, of the lower
  trace's high percentile minus the upper trace's low percentile, plus 10 % (with clipping, nothing overlaps; unclipped
  uses the 0.5th and 99.5th percentiles, so only a brief transient can reach the next row). A floor keeps flat traces
  apart. A ROI with no finite value in the frames shown (possible with `middle_frames_shown`) is an empty row labelled
  "(no data in shown frames)" that takes no part in the spacing or y limits (without this the y limits were NaN and
  `set_ylim` raised). With no top cells the panel says so; a top ROI hidden by overlaps in the label image is listed in the
  FOV title.
- **Top-cell rankings**: `top_cells(..., rank_by="max_dff" | "p_cell")`. `p_cell` is Suite2p's classifier
  probability (`iscell.npy` column 1); mini2p's `classify_rois` runs Suite2p's built-in classifier on the Cellpose ROIs,
  so it is advisory. Ties go to the lower ROI index; ROIs without a probability or without a defined ΔF/F0 are left
  out. The notebook (`N_TOP = 30`, `RANKINGS`) writes one PDF per ranking, `top30_cells_by_p_cell_<tag>_<variant>_<units>.pdf`
  and `top30_cells_by_max_dff_….pdf`, followed by the display settings when set (`_frames_mid8000_clip1-99`; nothing for
  None), so figures made with other settings are not overwritten; the traces' title names the ranking and each label reads
  `ROI 12 (p=0.93, max 2.4)`. The notebook closes the PDF in a `finally`, so an error never leaves a truncated file.
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

0 pulses is aligned to the no-reward cue, 1-4 pulses to reward onset. Raw F is not baseline-subtracted. A long window
(e.g. (−10, 30) s) can include the previous or next trial's outcome in the baseline or the response when trials are closer
together than the window; `n_events_dropped` shows outcomes lost at run ends or sync gaps. Ranking by max
ΔF/F0 favours cells with one bright frame; check the traces. Cells of different runs of the same field of view may be the
same neurons, and trials of a session are not independent. Outputs go to `results/figures/funcimg/` (git-ignored).

## Choosing sessions and runs

`SESSION_FIRST, SESSION_LAST` pick the session range (the same number twice for one session); folders without a
session number (`ses-baseplating`, `ses-test`) are skipped. `RUNS = None` uses every run; `RUNS = ("run-02",)` uses
only those run folders of each session, in all three steps (frame times, pooled responses, top cells). Runs not named
are neither pooled nor listed as left out; naming a run that no session in the range has raises an error listing the
runs that exist. Figure titles and file names carry the sessions and runs (`pulse_responses_ses-220-220_run-02_…`), so
a single-run figure does not overwrite the all-runs one. The pooled and background PNGs also carry the window
(`pulse_responses_<tag>_<variant>_<measures>_<units>_win-10to30s.png`).

## Middle frames of the top-cell traces

`TRACES_X_UNITS` (default `"frames"`) and `TRACES_MIDDLE_FRAMES` (default 8000) set the top-cell trace panels apart
from the pooled figures (`X_UNITS`): each run's traces show only its middle 8000 frames (`middle_frames`, centred;
all frames when the run is shorter or the setting is None). The per-ROI clipping percentiles and the row spacing are
taken over the frames shown, outcomes outside them are not marked, and the title says "middle 8000 of N frames". The
ranking still uses the whole run. `middle_frames_shown` must be None or an integer (a bool or float raises `TypeError`) and
positive (checked before the "shorter than the run" shortcut). With `x_units="s"`, if none of the frames shown has a
pyControl time the panel is empty, a `UserWarning` is raised and the title says so.

## Background (non-ROI) fluorescence

A check for a field-wide effect on the reward responses (blood flow, neuropil, motion, a diffuse neuromodulator).
`background_mask` (`src/funcimg_analysis/background.py`) is True for pixels in no ROI of `stat.npy` (cells and non-cells,
the ROIs' `ypix/xpix` in full, so overlapping pixels count as ROI) and inside the registration crop (`ops["yrange"]`,
`ops["xrange"]`); `roi_margin_px` (default 0) also drops a square ring around every ROI. All those pixels are one large
ROI: its F is the per-frame mean of its pixels (Suite2p's ROI mean with uniform weights, no neuropil subtraction) in the
green (`data.bin`) and red (`data_chan2.bin`) registered binaries. `load_background` (`background_io.py`) reads them
as a memmap in chunks of 200 frames (never a whole ~3.4 GB binary) and caches the two traces in
`<cache_dir>/funcimg_background/` (the repo's git-ignored `.cache/`), keyed on the plane's path, the sizes and mtimes of
the two binaries, `stat.npy` and `ops.npy`, the margin and a hash of the code; nothing is written beside the data.
`collect_background_responses` passes the traces through `pulse_responses` like a cell's F, so frame times, window,
phases, 0-pulse no-reward cue, pulse categories and robust z with baseline subtraction are identical (a test builds a
cell with the background's trace and checks every view agrees). Measures are raw `F` (main view) and `F z`; there is no
ΔF/F0 (no neuropil or F0 model for the background). Each run is one trace, so the pooled figure is mean ± SEM across
runs (`n_cells` in the pooled dict is the number of runs). `plot_background_responses` draws the background rows with
`plot_pulse_responses` (one line per pulse count, thin lines per run) and the cells' `F z` beneath on shared y limits;
`plot_background_mask` shows each run's mean images with the included pixels shaded and the ROIs outlined (title:
number of pixels and fraction of the FOV). The notebook's section 2b (`BACKGROUND = True`, `ROI_MARGIN_PX = 0`) writes
`background_responses_<tag>_<variant>_<units>.png` and `background_pixels_<tag>_<variant>.pdf` (one page per run).
Caveats: out-of-focus neuropil is in the background; the registration edges are excluded; a run whose binaries are
missing or whose size does not match `ops` is left out with the reason.

Details of the background analysis: with a single contributing run the SEM across runs is undefined, so
`collect_background_responses` shows the SEM across that run's trials of the pulse count (`sem_kind = "trials"`; the
legend says "SEM across trials"); runs whose background z is NaN (constant background, MAD 0) are listed by the notebook
like `flat_cells` and are not counted in the legend's n. The cells' row in the notebook uses only the runs that also have
a background (differences are printed). A run whose `ops.npy` lacks `yrange`, `xrange` or a mean image (or whose mean
image is not the image's shape) is left out with that reason (`BackgroundUnavailableError`); both binaries' sizes are
checked against `ops` before either is read. Caveat on the crop: Suite2p takes `yrange`/`xrange` from the non-bad
frames only, so on frames with large motion some edge pixels inside the crop can be invalid; this affects the cells
too. `plot_pulse_responses` now makes one row per measure (it raised for one or three measures without `axes`).
