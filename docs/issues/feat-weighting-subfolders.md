---
type: feature
complexity: medium
status: in review
---

# Mouse-weighted and event-weighted versions of the value-change and arm figures

## Goal

Every averaged figure of two analyses comes in two versions, each in its own subfolder:
`mouse-weighted/` (each mouse's events pooled into one curve, then mean ± SE across mice) and
`event-weighted/` (every event counts once, mean ± SE across events).

## Change

- Value change (`run_value_change_analysis`): the mouse-weighted figures move from
  `<value-change folder>/` to `<value-change folder>/mouse-weighted/`; the event-weighted ones stay
  in `event-weighted/`. Nothing is written at the top level. The folder names are constants in
  `src/pipelines/weighting_folders.py`.
- Reversal-aligned arms: `compute_reversal_aligned_arms(..., weighting="subject" | "event")`
  (default `"subject"`; anything else raises) and `ReversalAlignedArms.weighting`.
  - `"subject"`: block changes averaged into a subject curve (choice probabilities smoothed by the
    moving average), then mean and SE across the subject curves.
  - `"event"`: the mean is the moving average of the mean of all block changes' one-hot rows; the SE
    is across the block changes' rows after the same moving average. Magnitudes are not smoothed;
    their mean and SE are across block changes. The probabilities-sum-to-1 check applies to the
    mean before and after smoothing.
  - `run_reversal_aligned_arm_analysis` computes both. Figures go to `<folder>/mouse-weighted/` and
    `<folder>/event-weighted/` under the same file name; `results` stays the mouse-weighted
    structure and `event_results` holds the event-weighted one; `counts` has a `weighting` column
    (`subject` or `event`) and one row per scope, scheme, type and weighting; `figure_paths` lists
    both.
  - Titles: "mean ± se across mice" or "mean ± se across block changes, each weighted equally"
    followed, for event weighting, by "SE treats block changes as independent, so it is narrower
    than the true uncertainty". The count in the title says `n=K mice` (was `subjects`).

## Results change

- `value-change/`: the `Choice Probabilities Around ...` figures that sat at the top of each
  value-change folder are now in `value-change/mouse-weighted/`, unchanged. `event-weighted/` is
  unchanged. A rerun does not delete the old top-level files (Meg's existing results were already
  moved by hand).
- `reversal-aligned-arms/`: the figures move from the top of the folder to
  `reversal-aligned-arms/mouse-weighted/` (curves unchanged; the title now says "across mice" and
  "n=K mice" instead of "across subjects" and "n=K subjects"), and new event-weighted copies appear
  in `reversal-aligned-arms/event-weighted/`. Old top-level figures from earlier runs stay until
  deleted (the all-cohorts runner clears its folders itself).
- `tables/.../reversal_aligned_arms_counts.csv`: a new `weighting` column and twice as many rows;
  filter on `weighting == "subject"` for the old table. The same holds for the stacked tables of
  the cohort runner and the all-cohorts runner.
- Value-change tables are unchanged.
