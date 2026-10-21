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

## Shared vs cell-specific signal

Asks whether cells carry signal beyond the field-wide background, and whether the reward response survives removing it
(notebook section 2c, `SHARED_SIGNAL = True`; settings `DETREND_WINDOW_S` = 60, `N_SHIFTS` = 100, `MIN_SHIFT_S` = 30,
`CELLS_ONLY` = False, `FIT_EXCLUDE_OUTCOMES` = True, `FIT_EXCLUDE_WINDOW_S` = (-1, 5), `NULL_KINDS`). Code:
`shared_signal.py` (detrending, correlations, partial r with several controls, reliabilities, regression),
`shared_signal_null.py` (circular-shift chance levels), `run_statistics.py` (all statistics of one run and the table row),
`shared_signal_run.py` (one run: statistics plus original and residual responses), `collect_shared.py` (from disk, cached
by `shared_cache.py`), `plot_funcimg_shared.py` (figures).

- **Inputs**: every ROI's raw F and neuropil-corrected ΔF/F0 (green and red) from `plane_views`, each ROI's own neuropil
  (`Fneu`, `Fneu_chan2`), and the cached background F of the same channel with its two halves (`load_background`: a
  checkerboard split of the background pixels, read in the same pass over the binaries). `CELLS_ONLY` keeps ROIs with
  `iscell[:, 0] > 0`; the default uses all ROIs. A run with fewer than two ROIs is listed and left out
  (`TooFewCellsError`).
- **Preparation**: each trace minus a running 10th percentile over `DETREND_WINDOW_S` (a centred window, evaluated on a grid
  of window/10 frames and interpolated; NaN stays local; within half a window of either end the window is truncated to one
  side, so the baseline there is less reliable), then robust z (median / MAD). It is a new implementation, not mini2p's
  ΔF/F0 baseline, because that one runs through Suite2p/torch on a whole plane. Correlations use every frame finite in all
  the traces involved, including frames without a pyControl time.
- **Statistics per run, channel and measure**: r(background, mean of the cells); per cell r(cell, leave-one-out average of
  the other cells) and r(cell, background); partial r(cell, LOO | background); partial r given the background **and** the
  cell's own and the other cells' mean neuropil (both sides regressed on all controls with an intercept; computed from the
  correlation matrix, equal to correlating the residuals, tested against `lstsq`); R² and slope of `cell ~ a + b *
  background`; the background's split-half correlation and Spearman-Brown reliability; the reliability of each LOO average
  (random splits of the other cells into halves differing by at most one cell, averaged over 10 splits, Spearman-Brown, at
  most 1) and the **attenuation-corrected** r(cell, LOO) = r / sqrt(reliability) (at most 1). Without the correction the
  comparison with r(cell, background) is biased against the cells when few are averaged (3 cells: about -0.2); with it,
  field-only cells lie on the identity line for 3 to 50 cells (tested). The background's own noise lowers r(cell,
  background); that is why its reliability is reported (it is not used to correct the x axis).
- **How to read partial r**: the circular-shift null only tests partial r = 0. A noisy or spatially different background
  leaves some of the field signal in the cells (a test builds field-only cells with a background that is the field plus
  noise: partial r is clearly positive), so partial r > 0 is **not by itself evidence of shared neural signal**. Read it
  with the neuropil-controlled value (it removes most of that residual when the neuropil carries the field, tested; it does
  not remove a signal the neuropil does not carry), the background's reliability and the residual reward test.
- **Chance levels** (`shared_signal_null.py`): the statistics recomputed with one reference trace circularly shifted by a
  random offset of at least `MIN_SHIFT_S` in both directions, `N_SHIFTS`, seeded from the seed and the run's name. Only the
  `NULL_KINDS` are computed: by default `r_bg` (background shifted) and `r_loo` (the other cells' average shifted), which the
  scatter draws as ticks on its axes (the y one divided by sqrt(reliability) like the markers), and `partial_average` and
  `partial_neuropil_average` (the average shifted, controls real), which the histograms draw; opt-in `partial_background`
  and `r_bg_avg`. Rows without missing frames get all shifts at once from FFT cross-correlations; rows with a missing frame
  (whose frames in common change with every shift) go through the plain loop, with identical numbers (tested). The
  histograms mark the 95th percentile pooled over cells and shifts; the table's `null 95th` columns are the 95th percentile
  over shifts of the run's median. The paired Wilcoxon p across cells is descriptive (cells within a run are not
  independent; zero differences count half, so identical inputs give p = 1).
- **Residual response**: `cell - b * background` in detrended-z units (the cell keeps its offset), aligned with the same
  `pulse_responses` path (window, phases, pulse categories, robust z and pre-outcome baseline subtraction) as the detrended
  cells. `b` is fitted only on frames outside a `FIT_EXCLUDE_WINDOW_S` window around every outcome of the analysed phases
  (all pulse counts) and applied to all frames: fitting on all frames lets a reward response that leaks into the background
  raise the slope and remove the cells' own response (tested: the masked residual keeps about 80 % where the unmasked one
  keeps about 55 %). The window defaults to `WINDOW_S` in the function, but the notebook uses (-1, 5) s because a (-10, 30) s
  window around outcomes about 10 s apart covers every frame (`NoFitFramesError`: the run is left out with the fix when
  under 10 % of the frames remain; the fraction used is in the table). Z-scoring the residual by its own, smaller, scale
  would inflate a surviving response, so the residual response is multiplied back by the residual's robust scale: both
  responses are in units of the detrended cell's z. The "original" is the detrended cell, so the only difference is the
  background removal; section 2's responses differ from it through detrending alone.
- **Scatter**: filled markers are the corrected r(cell, LOO), hollow markers the raw one, both against r(cell, background),
  with the identity line, the null ticks, and per-run text (`session run`: median difference corrected and raw, Wilcoxon p,
  n, cells outside the axes). The histograms count cells outside their bins in the panel text and show the background's
  split-half r and reliability.
- **Cache**: each run's result is pickled in `<cache_dir>/funcimg_shared/` (the repo's git-ignored `.cache/`), keyed on
  hashes of the content of the raw traces (cells, neuropil, background and halves), the extraction record, the ROIs, frame
  times, outcomes, every setting and seed, and the source of the analysis code; a truncated file is recomputed. A 200 ROI x
  27000 frame run (4 views, 100 shifts) took about 700 s (two measurements, 697 s and 700 s) before and about 40 s (the neuropil-controlled partial r and the reliabilities included) now; a cache hit takes the time of loading
  the traces.
- **Pooling** refuses runs whose window offsets differ or whose frame period differs by more than 1 % (a pooled figure's time
  axis comes from the first run); the check is shared with the other pooled figures.
- **Figures** (`shared_{scatter,partial_r,r2}_<tag>_<variant>_<settings>.png`,
  `shared_residual_responses_<tag>_<variant>_<units>_win…_<settings>.png`; the settings tag is e.g.
  `detrend60s_shifts100_min30s_allrois_fitoutside-1to5s`).
- **Caveats**: detrending matters (the window is in the file names); removing a real diffuse signal removes signal that
  belongs to the cells; raw F is not neuropil-corrected whereas ΔF/F0 is; cells within a run are not independent.
- **Tests**: `tests/test_funcimg_shared_signal.py` (detrending, correlations, partial r, regression),
  `test_funcimg_shared_controls.py` (several controls, noisy background and neuropil, reliabilities, corrected r),
  `test_funcimg_shared_null.py` (batched against plain loops, which trace each branch shifts), `test_funcimg_shared_run.py`
  (reward only in the background vanishes and is not overcorrected below 0, only in the cells survives, leaks, red analysed
  against the red background, refusals), `test_funcimg_shared_collect.py` (from disk, `cells_only`, left-out runs, cache),
  `test_funcimg_shared_plot.py`.

## Background-corrected ΔF/F0 and the transient check

Notebook section 2d (`BG_CORRECTED_DFF = True`; settings `RESIDUAL_VARIANTS` = `("without_destriping",)`,
`MIN_F0_FRACTION` = 0.1, `TRANSIENT_CHECK`, `TRANSIENT_Z` = 3, `MIN_EVENT_GAP_S` = 1, `EXAMPLE_TRANSIENT_CELLS` = 6; the fit
window and the detrend window are 2c's `FIT_EXCLUDE_WINDOW_S` and `DETREND_WINDOW_S`). Code: `residual_dff.py` (pure),
`collect_residual.py` (from disk, cached in `.cache/funcimg_shared/`), `transients.py` (pure), `plot_funcimg_residual.py`,
`plot_funcimg_transients.py`.

- **Correction**: per cell and channel `F_corr = F - b * (background - mean(background))`, `b` the OLS slope of the raw F on
  the same channel's background (`fit_background`), fitted on the frames outside the outcome windows (`fit_frames`, the helper
  2c uses, so the same `NoFitFramesError` applies) and applied to every frame. The cell keeps its own offset.
- **ΔF/F0** of `F_corr` with mini2p's baseline and the `FluorescenceSettings` recorded in `cellpose_extraction.json`
  (`load_fluorescence_settings`, shared with `plane_views`). The correction **replaces** the neuropil subtraction: the
  baseline step of `compute_dff` is used without subtracting `Fneu` (a test checks equality with `compute_dff(F_corr, 0)`).
- **Exclusions**: a cell is excluded, with the reason, if its F0 is ≤ 0 or below `MIN_F0_FRACTION` x its median F in any
  frame, or if its F has no variance (no slope). Both measures are compared on the same remaining cells of a channel; a run
  with no usable cell in a channel is left out with the reasons.
- **Reported**: per run and channel the number of cells and exclusions, the median `b` and the median r(F_corr, background)
  over all frames (not exactly 0, the slope being fitted without the outcome windows; exactly 0 when
  `FIT_EXCLUDE_OUTCOMES = False`, tested), plus a list of the excluded cells.
- **Figures**: pooled pulse-count responses (robust z, baseline-subtracted, same window and phases as section 2) of Suite2p's
  and the corrected ΔF/F0 on shared y limits, and example cells over the middle frames. Each is z-scored by its own scale,
  so a response that survives a correction that removes shared variance looks larger (the 2c residual figure rescales
  instead); the markdown says so.
- **Example cells** are spread over the ranking of `b` (largest, ..., smallest), not the cells that change most.
- **Transient check** on the corrected ΔF/F0 of green and red (red has no indicator, so it is the noise reference): each trace
  detrended with a running **median** and robust-z-scored (the 10th percentile used for the correlations follows the noise's
  lower tail and makes positive excursions about 25 % more frequent than negative ones even for symmetric noise, tested);
  skewness, the robust skewness `((p99 - median) - (median - p1)) / MAD`, and the rates per minute of positive (> +z) and
  negative (< -z) events, an event being a threshold crossing at least `MIN_EVENT_GAP_S` after the previous one. The table
  gives median skewness (green, red), the fraction of green cells above the red 95th percentile of skewness, and the median
  event rates; figures are overlaid histograms (skewness, positive minus negative rate) and example traces of the most
  positively skewed green cells with their red channel. Caveats are in the notebook: red is dimmer, and motion artefacts also
  skew traces, so the example traces decide.
- **Tests**: `test_funcimg_residual_dff.py` (fluctuation removed, offset kept, fit mask honoured, equality with
  `compute_dff`, flags), `test_funcimg_residual_collect.py`, `test_funcimg_residual_plot.py`, `test_funcimg_transients.py`
  (planted exponential transients give skew and excess positive events in green, ~0 in noise red, symmetric noise ~0, event
  gap honoured, drift), `test_funcimg_transients_plot.py`.

## Most reward-responsive cells, cross-validated

Notebook section 3b (`REWARD_CELLS = True`; `RESPONSIVE_CONTRAST` = `"rewarded"` or `"4_vs_0"`, `RESPONSIVE_MEASURES` =
Suite2p's and the background-corrected ΔF/F0 of the green channel, `RESPONSE_S` = (0, 2), `BASELINE_S` = (-2, 0),
`PLOT_WINDOW_S` = (-2, 5), `N_REWARD_CELLS` = 20, `RANK_ON` = `"split_half"` or `"all"`, `SPLIT_SEED`; `N_SHIFTS` and
`MIN_SHIFT_S` are 2c's). Code: `reward_cells.py` (pure), `collect_reward_cells.py`, `plot_funcimg_reward_cells.py`.

- **Trial response**: the mean over the frames starting in `[0, 2)` s of the outcome's frame minus the mean over `[-2, 0)` s,
  on the robust-z ΔF/F0 (the index is scale-free); the frame the outcome fell in is left out of the baseline where the outcome
  came before its start, as in section 2.
- **Index**: `"rewarded"`: mean / SD (ddof 1) over the rewarded trials of the chosen half; `"4_vs_0"`: Welch's t of the 4-pulse
  against the 0-pulse trials (tested against `scipy.stats.ttest_ind`). NaN, not inf, where fewer than two trials have a value
  or there is no spread. The first does not grow with the number of trials, the second does.
- **Split**: trials alternate A, B, A, ... in time within each pulse count; the half each pulse count starts with is drawn from
  the seeded generator, so the split is deterministic. A half with fewer than two trials of a needed group raises
  `TooFewTrialsError`, and the run is left out with that reason.
- **Null**: the outcomes (all trials together) are shifted circularly by `shift_offsets` (at least `MIN_SHIFT_S`, `N_SHIFTS`
  shifts, seeded from the seed and the run's name, as in 2c); pulse counts and halves stay attached to their trials, and
  windows wrap around the end of the trace. A cell exceeds its null if its index is above the 95th percentile of its own
  shifted indices. Reported: cells above the null on half A (expected 5 % of the cells with an index), those also above their
  half-B null (expected 5 % of the cells chosen on half A, which is the right chance level for a second test of a selected set),
  and the Spearman rank correlation of the half-A and half-B indices.
- **Figure**: per measure and run, the top cells' panels (mean ± SEM over half-B trials per pulse count, title ROI, index A,
  index B, p(cell), title colour = the cell's outline colour in the field of view), a heatmap of all ranked cells, the mean
  image with the top cells outlined, and the null counts under the panels. `RANK_ON = "all"` ranks and plots all trials and
  prints "SELECTION BIAS ... not cross-validated" in red in the title.
- **Selection-bias test**: over 8 synthetic datasets of pure noise (200 cells), the top 20 cells ranked on half A average
  about 0 (|mean| < 0.04) on half B, against about 0.17 when ranked and scored on the same trials.
- **Tests**: `test_funcimg_reward_cells.py` (index, Welch t, masks, NaN handling, frame windows and wrapping, halves, null about
  5 % on noise and well above with planted cells, planted cells recovered, selection bias), `test_funcimg_reward_cells_collect.py`,
  `test_funcimg_reward_cells_plot.py`.
- **Refactors**: `outcome_fit_mask` and `fit_frames` (shared by 2c and 2d) are public in `shared_signal_run.py`;
  `run_seed` in `collect_shared.py` and `draw_fov` in `plot_funcimg_responses.py` are public; `shared_cache` hashes
  `residual_dff.py` into every cached result.
