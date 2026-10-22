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

## Results change
- New folder only: `figures/open-field/paper-ready/`. Nothing a run without `--paper-ready`
  writes changes.
