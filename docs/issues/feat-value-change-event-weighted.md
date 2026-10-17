# Event-weighted value-change figures

## Goal

The value-change choice-probability curves are a mean across mice: each mouse's events are
pooled into one probability per trial offset, and every mouse then counts once. A mouse with
few events weighs as much as one with many. Meg asked for the event-weighted version, where
every event counts once, to see whether a few low-count mice drive the pre-window pattern.

## Change

- `aggregate_windows(..., weighting="subject" | "event")` and
  `summarize_windows(windows, weighting=...)`. With `"event"`, the mean at t is the share over
  all events with a trial at t, and the SE is across those events. The per-mouse curves and
  the per-bin counts are the same for both. `ValueChangeCurves` has a `weighting` field.
- The figure title says "mean ± se across events, each event weighted equally" for the
  event-weighted figure.
- `run_value_change_analysis` writes each figure twice: the mouse-weighted one where it was,
  and the event-weighted one with the same name in an `event-weighted/` subfolder. Each
  summary has an `event_weighted` entry with that `mean` and `se`.

## Results change

The existing mouse-weighted figures and tables are unchanged. New figures appear in
`<value-change folder>/event-weighted/` for every problem, the pooled drifting problems and
the all-cohorts runner. For a single mouse the two versions are identical.

The event SE treats events as independent, but events from one mouse are not. So the
event-weighted band is narrower than the real uncertainty: use it to compare the shape of the
curves, not to judge significance.
