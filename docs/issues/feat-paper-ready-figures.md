---
type: feat
complexity: moderate
status: in review
---

# Paper-style versions of the pooled drifting figures

Requested: "fix the plotting style of my figures so that they look more like what you would
expect from a scientific paper", in a separate folder `open-field/paper-ready/`; the pooled
drifting set only, Nature/Cell style, PDF + SVG + PNG.

## Acceptance criteria
- `paper_style()` sets the rc parameters (Arial with Linux fallbacks, 7 pt, thin lines, outward
  ticks, editable text) and restores them afterwards.
- `polish(fig)` removes top/right spines, trims spines to the outermost ticks, unboxes legends,
  caps markers, recolours to a colourblind-safe set, picks a column width and moves the suptitle,
  figure notes and long titles to a caption string.
- `run_cohort(paper_ready=True)` / `run_cohorts` and `run_all_cohorts_drifting(paper_ready=True)`
  write pdf, svg, png (600 dpi) and `.caption.txt` for every pooled figure, including the set
  without excluded mice, under `figures/open-field/paper-ready/`, in the same subfolders.
  Normal outputs (tables and PNGs) are identical with and without the option.
- `--paper-ready` on `run_cohorts.py` (with `--pooled-only`, `--exclude-mice`) and
  `run_all_drifting_problems.py`.
- The paper-ready "Rank Proportions (...) by Average Across Mice" figures mark best vs second,
  best vs third and second vs third with significance brackets and stars instead of p-values:
  two-sided Wilcoxon signed-rank tests across mice, Holm-corrected for the comparisons of the
  figure, stars from the corrected p (`***` < 0.001, `**` < 0.01, `*` < 0.05, `n.s.`). The
  caption gives n, W, raw and corrected p and marks, says why a comparison was not tested
  (fewer than 2 mice) and, when too few mice make significance impossible, the smallest
  attainable exact p. A comparison whose differences are all zero counts as p = 1 (n.s.) and
  stays in the Holm family. The normal figures and tables keep the paired
  t-tests.

## Decisions (PI to check)
- One switch, no per-plot changes: `paper_figures()` is a context the save step in `plot_helpers`
  reads; the pooled analyses are drawn a second time inside it. The cost is that the analyses
  run twice with `--paper-ready`.
- `polish` restyles the finished figure (text sizes, line widths, colours) because the plot
  functions hard-code them; colours are mapped from the old hexes (best/second/third, mouse
  palette) to Okabe-Ito, so best is blue, second green and third grey, as before.
- Figures with a legend below the axes get a figure-level legend (two columns) so the layout
  reserves room for it; figures with a colorbar made under another layout engine fall back to
  tight layout.
- Title rule: a single-line title of up to 36 characters stays (axes title, 8 pt); anything
  longer, multi-line, or the suptitle goes to the caption. The scope marker ("without ...")
  is in the caption only.
- Narrow figures are at most 0.8 times their width tall; grids of per-mouse panels may run down
  the page (max 230 mm).
- Output folder is `figures/open-field/paper-ready/<cohort>/drifting/...` (cohort level kept so
  cohorts do not overwrite each other) and `paper-ready/all-cohorts[-without-...]/`.
- sciplotlib is not a dependency; the spine trimming is a small helper that credits it.
- Significance stars (requested: "do the stars for the comparisons ... a Wilcoxon signed rank
  test and correct for multiple comparisons"): the unit is the mouse, with the same per-mouse
  proportions the bars and lines show (both pooling modes). scipy's default
  `zero_method="wilcox"`, `method="auto"`: exact for up to 50 mice without ties or zero
  differences, an exact permutation test for ties or zeros up to 13 mice, the normal
  approximation otherwise; the method used is named in the caption. All-zero differences stay
  in the Holm family as p = 1, so the correction is always for every tested comparison (the
  more conservative choice). A printed p gets extra decimals rather than round across a star
  threshold (0.0496, not 0.050). Brackets are drawn only inside `paper_figures()`: the means
  and statistics reach the reader only through the paper caption, so elsewhere it raises. Holm is a small function
  of our own (statsmodels is only a notebooks extra). `n.s.` is drawn too (`show_ns=True`).
- In the paper rank figure the bar means move to the caption (they were moved there by the
  polish step anyway), so the brackets sit just above the data; the y ticks stay at 0-1 and the
  mouse legend goes beside the axes (so the figure is 120 mm wide when it names the mice).

## Results change
- New folder only: `figures/open-field/paper-ready/`. Nothing a run without `--paper-ready`
  writes changes.
