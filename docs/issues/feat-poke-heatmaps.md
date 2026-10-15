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
  mouse when given. Returns `long_pokes`, `counts`, `per_session`, `cross_check`, `files`. Writes
  into `out_dir` (the runner should pass the problem's `long-pokes` folder).
- `src.behavior_analysis.poke_counts.previous_problem_choice_towers(all_problems) ->
  {mouse: {overall_problem: frozenset | None}}` and
  `previous_choice_towers_for_problem(subjects_sessions, previous) -> {mouse: frozenset | None}`
  (reads each session's `"problem"`, so it works on task-type-regrouped problems). Runner use:
  `prev = previous_problem_choice_towers(trials_by_problem)`, then per grouped problem
  `run_poke_analysis(subs, out, previous_choice_towers=previous_choice_towers_for_problem(subs, prev))`.
- `plot_single_session(..., previous_choice_towers=None)`.

## Design decisions (PI to check)
- **Layout** (`MAZE_POSITIONS`): letters left to right, number 1 at the BOTTOM (A1 bottom-left, C3
  top-right). This is what the notebook drew (`WELL_MAP` + `(x, y) = (col, 2 - row)`); its comment
  says "row 1 is top", which contradicts its own coordinates. Not checked against the physical maze.
- **Equal visibility**: concentric octagonal rings of identical radial width (data units), green
  innermost, then red, then blue. A split ring would hide half of each colour on a small port; the
  only asymmetry is that the outer ring is longer. Green/red is as requested but is hard for
  red-green colour-blind readers.
- **Open windows**: the last trial of a recording has no following initiation, so its after-choice
  window is excluded (counted in `n_open_windows` and the cross-check table), not run to the end of
  the log.
- Trials are the recording's `print` rows (same rule as the importer, non-bandit recordings skipped);
  the k-th trial is checked against the session's merged `choice` list and a mismatch raises.
- Initiation tower: session `initiation_tower`, protocol default `B2` when unlogged. The initiating
  poke is the `_in` that triggers `wait_init -> wait_choice`; if it is at another port the parser raises.
- Pokes during the rewarded `choice` state are outside the window (window starts at `ITI`).
- Single-session figure: maps appear whenever the session has `"df"`, in a row added below; standard
  panels keep size and distance from the top (tested). Parse errors there now raise.
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
