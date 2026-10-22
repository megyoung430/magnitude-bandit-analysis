---
type: feature
complexity: complex
status: in review
---

# Imaging: the Zong et al. 2022 analysis pipeline as its own notebook

`scripts/funcimg/zong2022_pipeline.ipynb` runs the neuronal-activity pipeline of Zong et al. 2022 (Cell 185:1240, STAR
Methods) on our runs, starting from the registered planes and the Cellpose masks (registration and cell detection done,
extraction not). It extracts the traces itself and never reads or writes the plane's own `F`, `Fneu`, `stat` files.

## What it adds

- **`zong2022_signal.py`** (pure): `Fcorr = Fcell - 0.7 Fnp`; running baseline `F0 = Fs + m` (8th percentile in a ±15 s
  centred window, baseline points at the lowest 10 % of the local-SD range, `m` centring `Fcorr - F0` on the baseline
  points); `ΔF/F`; significant transients (above 2 local baseline SDs for more than 0.75 s); OASIS deconvolution (Suite2p's);
  `ΔF/F_clear`, `E_clear` (scaled to the maximum of `ΔF/F_clear`), calcium events; SNR and inclusion (SNR > 3, optional
  `MIN_EVENTS`); event counts for several τ.
- **`zong2022_responses.py`** (pure): whole-frame windows around outcomes by pulse count; the escape-active score of the paper
  against 1000 circular shifts (at least 30 s from either end), all shifts at once by FFT cross-correlation (checked against
  `np.roll` and rescoring); counts against the 5 % expected.
- **`zong2022_motion.py`** (pure): Suite2p's registration metrics (`pclowhigh`, `pc_register`) with 500 frames per extreme
  image and up to `MAX_PC_SOURCE_FRAMES` sampled frames (default 2000, the paper used 5000); the paper's non-rigid motion per frame from `xoff1`, `yoff1`.
- **`zong2022_io.py`, `zong2022_run.py`**: Cellpose seg/image located as the mini2p batch does
  (`BatchSettings` defaults), masks aligned onto each variant (`align_to_plane`), extraction with Suite2p's `create_masks`
  defaults and `extraction_wrapper`; the slow stages (extraction, signal, registration metrics) cached in `.cache/zong2022/`
  keyed on file size and mtime, content, settings and code; `analyse_runs` lists runs left out with the reason.
- **`plot_funcimg_zong2022.py`**: baseline and transient examples, SNR, τ sensitivity, ΔF/F_clear and E_clear responses by
  pulse count (SNR-passing and all cells), score against null, top responsive cells (SNR and score in titles, FOV outlines),
  motion summary, z-drift review (one panel per PC; PDF of all PCs).
- Figures and per-run CSV tables go to `results/figures/funcimg/<cohort>/<subject>/zong2022/` (git-ignored). Nothing is
  written to the share unless `WRITE_DERIVATIVES = True`, and then only to `<run>/<variant>/zong2022/`.

## Interpretations of underspecified steps (also in the notebook's first cell)

- jGCaMP8s: `TAU_S = 0.7` s is a placeholder (culture half-decay about 0.31 s, τ about 0.44 s; in vivo slower); the user
  sets it. Event counts for 0.45, 0.7 and 1.0 s are compared in the notebook.
- OASIS is given `Fcorr - F0` (the paper says `Fcorr`; OASIS has no baseline term).
- `m = mean(Fcorr - Fs)` over baseline points; windows are cut, not padded, at the recording's ends.
- "Local SD of the baseline": robust SD (1.4826 × MAD) of `ΔF/F` over the non-transient frames of the ±15 s window, two
  passes; fewer than `MIN_BASELINE_S` (4 s) of such frames falls back to the cell's overall value.
- Cells with fewer than `MIN_BASELINE_S` of baseline points are excluded with a reason, never an error.
- Edge windows (cut, not padded) make frames near the recording's ends somewhat over-represented among baseline points.
- "More than 0.75 s": `k × period > 0.75`.
- SNR noise: mean absolute consecutive-frame difference over pairs both outside transients; no transient or zero noise gives
  no SNR (not passing).
- Cells with `F0 <= 0` anywhere: excluded and listed with the reason (the paper is silent). A run with no usable cell is left
  out. Aligned labels are renumbered 1..n by row; the table, figures and `traces.npz` carry each row's Cellpose label.
- Registration metrics: Itop registered to Ibottom as in Suite2p, not both to the stack mean; fewer extreme frames when the
  movie is short.
- Responsive cells: score on `ΔF/F_clear` (setting `SCORE_TRACE`), post-outcome window from the outcome's frame
  (15 frames for 2 s at 7.5 Hz), separately for rewarded and no-reward outcomes, on the outcomes with a complete −2 to 5 s
  window; the scored trace (`ΔF/F_clear`) is shifted against the behaviour, whereas the paper shifts the deconvolved
  activity, so the null keeps the structure of what is compared. Chance counts use only cells with a finite score.
- Cells are all Cellpose ROIs (no `iscell`); Cellpose labels cannot overlap, so no pixels are removed as overlapping.

## Results change

New notebook and modules only; no existing analysis or figure changes.

## Caveats

The paper's 30 s baseline, transient filter and deconvolution remove slow field-wide signals by design, unlike
`reward_responses.ipynb`. Cells of different runs of one field may be the same neurons; cells within a run are not
independent, so the binomial p against 5 % is a guide. Tests use synthetic fixtures only. The registration QC samples
`MAX_PC_SOURCE_FRAMES` frames (default 2000); its PCA peaks at about 4 GB for 5000 frames of 256 × 256 and 16 GB at
512 × 512, in proportion to the frame count.

Correlated noise is a limitation: the 2 SD / 0.75 s rule assumes roughly white noise. Cells with no transient pass the
SNR > 3 inclusion in 0 % of synthetic cases with white noise, 4 % with AR(1) noise of τ = 0.2 s and 85 % with τ = 0.5 s
(`noise_pass_rates` in `tests/test_zong2022_signal.py`). The paper's rule is kept; check example traces.
