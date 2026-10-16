---
type: feature
complexity: complex
status: in review
---

# Long pokes per problem and poke heatmaps

Ports the `fixing_old_analyses_problem1/2` poke analyses into the package: non-choice long pokes per
relative session, and maze heatmaps of (a) pokes after the choice and (b) long pokes after
initiation, per mouse and problem, plus the two session-level maps in the single-session figure.

## Acceptance criteria
- Long pokes per problem: mean `num_long_pokes` per trial per session, sessions re-keyed
  `rel_ses-01, 02, ...` per mouse, bars = mean +/- SE across mice (NaN SE omitted), own folder.
- Pokes after choice: every `*_in` from the trial's `ITI` state to the next initiation, the
  initiating `B2_in` excluded, repeats kept, leading pokes at the trial's chosen tower dropped until
  another port was poked.
- Pokes after initiation: long pokes (`hold_poke` -> `poked_in`) from initiation to the choice,
  choice poke included; leading initiation-tower long pokes dropped until a long poke elsewhere.
- Green outlines = current problem's choice ports, red = the mouse's previous problem's (overall
  problem order, any task type), equally visible when both; blue = initiation port (after-initiation
  maps). No red for a mouse without a previous problem.
- Problem-level figures (one panel per mouse, 0-to-max scale, honest colorbars) in the same folder
  as the long-poke plot; single-session figure gets the two session maps; `plot_single_session` takes
  `previous_choice_towers`.

## Entry points
- `src.pipelines.poke_analysis.run_poke_analysis(subjects_trials_by_problem: dict, out_dir: Path, *,
  previous_choice_towers: dict | None = None) -> dict`. `subjects_trials_by_problem` is the sessions
  of ONE problem, `{mouse: {session_key: session_dict}}` (what `run_problem` receives); a dict keyed
  by problem raises. `previous_choice_towers` is `{mouse: frozenset | None}` and must cover every
  mouse when given. Returns `long_pokes`, `counts`, `per_session`, `cross_check`, `files` (only the
  files this call wrote). Writes into `out_dir` (the runner should pass the problem's `long-pokes`
  folder). `Poke Trial Summary.csv` columns: `mouse, session, n_trials, n_trials_counted,
  n_trials_dropped_no_print, n_trials_dropped_inconsistent, n_open_after_choice_windows,
  n_long_poke_checked, n_long_poke_agree, parse_error`.
- `src.behavior_analysis.poke_counts.previous_problem_choice_towers(all_problems) ->
  {mouse: {overall_problem: frozenset | None}}` and
  `previous_choice_towers_for_problem(subjects_sessions, previous) -> {mouse: frozenset | None}`
  (reads each session's `"problem"`, so it works on task-type-regrouped problems; a mouse or problem
  missing from `previous` raises a `ValueError` naming it).
- `plot_single_session(..., previous_choice_towers=None)`.

## What the runner (a later integration item) must call
No runner is wired here (`cohort_analysis.py` and `open_field.py` are untouched), so a cohort run does
not yet produce the `long-pokes` folder or red outlines. The wiring is:
1. Once per cohort, from the full `{overall problem: {mouse: sessions}}` dict (before regrouping by
   task type): `previous = previous_problem_choice_towers(trials_by_problem)`.
2. For every (task type, k) problem `subs = {mouse: sessions}`:
   `prev_ports = previous_choice_towers_for_problem(subs, previous)`, then
   `run_poke_analysis(subs, <problem folder>/"long-pokes", previous_choice_towers=prev_ports)`.
3. In the single-session loop, for each mouse and session:
   `plot_single_session(session, ..., previous_choice_towers=prev_ports[mouse])`.
`tests/test_poke_analysis_pipeline.py::test_red_outlines_of_the_previous_problem_appear_in_problem_two_only`
and `..._single_session_of_problem_two_draws_red_outlines_...` run exactly this sequence on two
problems and assert red outline patches and pixels in problem 2 only.

## Design decisions (PI to check)
- **Layout** (`MAZE_POSITIONS`, confirmed by the PI): letters left to right, number 1 at the bottom
  (A1 bottom-left, C3 top-right).
- **Equal visibility**: concentric octagonal rings of identical radial width (data units), green
  innermost, then red, then blue. A split ring would hide half of each colour on a small port; the
  only asymmetry is that the outer ring is longer. Green/red is as requested but is hard for
  red-green colour-blind readers.
- **Open windows**: only a recording's last printed trial may have no following initiation; its
  after-choice window is excluded (`n_open_after_choice_windows`), not run to the end of the log.
- **Dropped trials** (PI decision): a trial's pokes are dropped, with one warning per session giving
  the count and reasons, when it ended without a summary print (timeout/abort: ITI with no print, or a
  plain-text print), when the held choice port differs from the printed `chosen_tow`, or when it went
  back to `wait_init` with no ITI. Aligned printed trials keep their place so the merged-choice check
  still holds. A print with no initiation since the previous trial drops both that trial and the
  previous one (its window never closed). Impossible logs raise `PokeParseError` (a `ValueError`).
- **Misalignment stops the run** (PI decision): parsed trials that differ from the imported ones (count
  or chosen tower) raise `PokeAlignmentError` (a `ValueError`, deliberately NOT a `PokeParseError`),
  as a likely code bug. `plot_single_session` and `run_poke_analysis` do not catch it.
- **`run_poke_analysis` skips unreadable sessions** (PI decision, as the single-session figure does):
  a session raising `PokeParseError` warns, gets zero counts and a `parse_error` column entry in
  `Poke Trial Summary.csv` (empty for readable sessions; last column).
- **Parse once**: both entry points take optional already-parsed counts: `run_poke_analysis(...,
  poke_counts={mouse: {session: SessionPokeCounts}})` (e.g. `per_session` from an earlier call or
  `mouse_poke_counts(mouse, sessions)`; must cover every session) and `plot_single_session(...,
  poke_counts=SessionPokeCounts)` (e.g. `session_poke_counts(session)`). A result with `parse_error`
  draws no maps and does not warn again.
- **Single-session figure** catches only `PokeParseError`: it warns with the title and draws the
  four standard panels without the maze row.
- **After-choice window** starts at the `ITI` state; pokes during the rewarded `choice` state are
  excluded (documented in `poke_events`). A non-initiating B2 poke in the ITI counts as "elsewhere"
  (PI decision; pinned by a test).
- **Initiation outline** is per mouse (the mouse's sessions must agree).
- Trials are the recording's `print` rows (same rule as the importer, non-bandit recordings skipped);
  the k-th trial is checked against the session's merged `choice` list and a mismatch raises.
- Initiation tower: session `initiation_tower`, protocol default `B2` when unlogged. The initiating
  poke is the `_in` that triggers `wait_init -> wait_choice`; if it is at another port the parser raises.
- Single-session figure: maps appear whenever the session has `"df"`, in a row added below; standard
  panels keep size and distance from the top (tested).
- Mean long pokes per problem is trial-pooled per mouse (`problem_mean_by_mouse`); the figure is per
  relative session as in the notebook.

## Bugs found in the notebook
1. Window end included the initiating `B2_in` (window cut at the `wait_choice` state row, after the
   poke) -> B2 inflated. Fixed: the initiating poke is popped; test
   `test_initiating_poke_of_the_next_trial_is_not_counted`. Confirmed by running the old function.
2. Consecutive pokes collapsed (`collapse_consecutive=True` default). Fixed, no collapse; test
   `test_pokes_are_not_collapsed_when_repeated`.
3. After-initiation function added +1 to the choice on top of the printed list and fell back to the
   last cleaned poke when the choice was missing; it also collapsed repeats. Fixed: long pokes come
   from raw events with the choice poke counted once; tests
   `test_long_pokes_include_the_choice_poke_exactly_once`, `..._count_only_after_a_long_poke_elsewhere`.
4. `mpl.cm.get_cmap` is gone (matplotlib 3.11); `matplotlib.colormaps[name]` used; all drawing tests.
5. Broad `except Exception`/silent returns (`_coerce_trial_pokes_list`, missing-column returns) and
   `get_num_long_pokes`'s `except (KeyError, TypeError)` print-and-skip: new code raises
   (`test_session_missing_num_long_pokes_raises_instead_of_being_skipped`, parser error tests).
6. Per-mouse colorbar used `vmin = min(counts)` but labelled it "0": scale is now fixed 0..max and ticks
   show the real numbers (`test_scale_starts_at_zero_even_when_every_port_was_poked`,
   `test_colorbar_ticks_are_zero_and_the_actual_maximum_with_the_given_label`).
7. k-th ITI assumed to belong to trial k (ITIs counted, trials skipped when a slice had no ITI;
   multi-recording frames were concatenated). Fixed by attaching each window to the print row before it
   and checking against the merged choice list
   (`test_windows_belong_to_the_trial_whose_print_precedes_them_across_recordings`,
   `test_alignment_with_the_sessions_merged_choice_list_is_checked`).
8. Dict-order sessions: `to_relative_sessions` and everything new use `ordered_session_keys`
   (`test_sessions_are_in_date_order_not_dict_order`, `test_mouse_counts_sum_over_sessions_in_session_order_not_dict_order`).
9. Hard-coded maze layout and y-limit (`y_text = 3` clipped bars above 3): layout is a documented
   constant tested against the notebook mapping; the y-limit is computed from the data.
10. SE was `0.0` for one mouse; now NaN via `mean_se_n` and omitted from the plot.
11. The drop-leading-choice-poke rule in the notebook ignored initiation-tower long pokes and only
    worked when a choice list was passed; both rules are now explicit functions.

## Results change
- New outputs (in `out_dir`): long-poke relative-session figure, two problem-level maze figures, four CSVs.
- Existing single-session figures for sessions with raw `"df"` (all imported sessions) get an extra
  row of two maze maps and are taller; their standard panels are unchanged. Sessions without `"df"`
  produce the identical figure. No other output changes.
- `plot_num_pokes_panel` / `plot_num_pokes_across_mice` gained optional `xlabel`, `title`,
  `tick_label` (and `rng` on the wrapper); defaults reproduce the old figure.
