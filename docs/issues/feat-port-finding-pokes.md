---
type: feature
complexity: simple
status: in review
---

# Poke heatmaps for the port-finding section of each problem

PI request: separate poke heatmaps for the port-finding (`num_t_found`) trials of each problem for
each mouse; one map per problem, not per session.

## Acceptance criteria

- The parser reads the pokes of port-finding trials with the two task rules (long pokes after
  initiation; visits after the choice from the ITI to the next initiation, initiating B2 included,
  leading visits to the chosen tower dropped), using the print's `chosen_tow`.
- Trials stay in file order within a recording.
- Every task-trial count and figure is unchanged.
- Two problem-level figures in `long-pokes/`, "Maze Heatmaps - Pokes After Choice (Port Finding)"
  and "... After Initiation (Port Finding)", per mouse, same outlines, port-finding sessions of the
  problem summed. Panel titles give n port-finding trials; a mouse with none is left out; no figure
  if no mouse has any. Not in the single-session figures.
- The port-finding counts are in the returned summary; no CSVs.

## Findings on the raw files

- Port-finding prints carry `chosen_tow` (and `long_pokes`, `num_long_pokes`). The state machine is
  the task's: `wait_init` -> initiating `B2_in` -> `wait_choice` -> long pokes (`hold_poke` ->
  `poked_in`) -> the choice long poke, the print row, `choice` (only if rewarded) -> `ITI` ->
  `wait_init`. So both rules apply unchanged. MY_175_L ses-33 and MY_71_R ses-229 have port finding
  first in the file, then task trials, in the same recording.
- Staged data: no port-finding print without `chosen_tow`, no timed-out port-finding trial, no
  port-finding trial whose held port differs from the print, no port-finding-only recording.
- The long pokes of all 52 + 9 staged port-finding trials agree with the printed `long_pokes` lists
  (checked while hand-checking, see below).

## Ordering bug

Confirmed. `parse_recording` returned the task trials in file order and then appended one
placeholder per port-finding trial (`trials + [placeholder] * n_port_finding`), so a recording that
begins with port finding listed it last (MY_175_L ses-33: port finding at indices 4-55 of 56;
MY_71_R ses-229: indices 25-33 for the 9 port-finding trials, which are the first 9 in the file).
It did not change any task result, because the placeholders carried nothing and the state machine
itself reads the file in order; but a trial's after-choice window closes at the *next* initiation,
which for the last port-finding trial is the first task trial's, so once port-finding trials carry
pokes the list order matters. Port-finding trials are now built by the same state machine in the
order they are printed; indices are 0-51 (ses-33) and 0-8 (ses-229). Regression tests:
`test_port_finding_trials_keep_their_pokes_and_stay_in_file_order`,
`test_port_finding_trials_stay_in_file_order_across_recordings`.

## Decisions

- Port-finding trials are flagged `TrialPokes.port_finding` instead of `drop_reason ==
  PORT_FINDING` (constant removed). With the drop reason they would also have had to be dropped
  when untrustworthy, losing the flag, and every `drop_reason is None` filter would pick them up
  once they carry pokes. Task counts are computed from `not port_finding` trials only.
- An untrustworthy port-finding trial (held port differs from the print, no initiation, a print
  with no `chosen_tow`) is dropped and warned about like a task trial ("1 x port finding: ...").
  A print with no `chosen_tow` is not guessed at.
- A timed-out port-finding trial has no print, hence no chosen tower: it counts in
  `n_port_finding_trials` and `n_port_finding_dropped`, carries no pokes, and (being expected) is
  not warned about.
- A recording with no task trials (port finding or pretraining only) is now also read by the state
  machine. If it cannot be read, its port-finding trials are dropped with the reason and the
  session is not skipped; otherwise the session's task counts would be lost to an unreadable
  pretraining file. A recording that holds task trials raises as before.
- Panel n is the number of port-finding trials in that map: for the after-choice map those with a
  closed window (a recording's last trial has none), for the after-initiation map all counted ones.
- New summary columns (after `n_port_finding_trials`): `n_port_finding_counted`,
  `n_port_finding_dropped`, `n_port_finding_open_after_choice_windows`. The result also has
  `port_finding_counts` (`{kind: {mouse: {port: count}}}`, only mice with port finding). The
  per-port counts are on the figures, not in any table.
- `SessionPokeCounts` has a new required field `port_finding` (a `PortFindingPokeCounts`).

## Results change

- Two new figures per problem in `long-pokes/` when some mouse has port finding in it.
- `poke_trial_summary.csv`: three new columns.
- Task-trial counts, maps and the single-session figures: unchanged (compared on all staged
  sessions, before and after: after-initiation and after-choice counts, trial counts, long-poke
  check and every parsed task trial are identical).
