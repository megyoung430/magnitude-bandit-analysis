---
type: bug
complexity: complex
status: in review
---

# GLM design matrices: session boundaries, NaN rows, labels, merged sessions, errors

`src/behavior_analysis/glm_utils.py` builds the multinomial-GLM design matrices
(`format_glm_input`, `format_glm_input_by_rank`). Sessions were concatenated
before the history lags were formed, NaN outcomes could stay in the fit, the
column labels could disagree with the matrix, a short list in one session
shifted every later session, and malformed subjects failed with unrelated
errors.

## Acceptance criteria

1. Both public functions take a keyword-only `reset_history_per_session: bool = True`.
   - `True`: lags never cross a session boundary; the first `n_lags` trials of
     each session are dropped.
   - `False`: sessions are concatenated as before; output identical to main
     apart from criteria 2-5.
   - The value is recorded in the returned `GlmDesign` (new
     `format_glm_input_with_info` / `format_glm_input_by_rank_with_info`; the
     existing functions keep returning their 3-tuple). Both settings are tested,
     including the boundary rows.
   - The rank design also takes a keyword-only `reset_beliefs_per_session: bool = True`,
     independent of the history option (relevant with `use_believed=True`).
     `True`: believed ranks (last-seen magnitudes) start empty in every session.
     `False`: ses-2's first trials use ses-1's last-seen magnitudes. All four
     combinations are tested, boundary rows included. With both options `False`
     the result equals one concatenated session (and main). Recorded in
     `GlmDesign.reset_beliefs_per_session` (`None` for the arm design, which has
     no beliefs and so no such option).
2. Rows with a NaN response or a NaN in any regressor are dropped in both
   settings, and counted per subject (`RowCounts`).
3. The returned labels always equal the returned columns (asserted for both
   functions, with and without `drop_redundant_cols`).
4. A choice or reward list that does not have one entry per trial in any session
   but the last raises a `ValueError` naming subject and session (otherwise
   concatenation shifts the later sessions).
5. Clear `ValueError`s (with the subject) for: a tower count other than three
   (both designs); a choice series shorter than `trial`; a last-session reward
   list more than one entry short when `n_lags >= 1` (names the session).

## Results change

Sample data: `data/3x3_field_blocked_reward_bandit/cohort-01/rawdata` (one
subject, one session, 93 trials, no NaN, `statsmodels` `MNLogit`, first class as
reference). The sample has a single session, so `reset=True` only drops the
first `n_lags` trials instead of the first one; `reset=False` equals main. The
sample does not exercise the NaN, merged-session, label (with `drop_redundant_cols=True`)
or error fixes.

Rows used (main / branch reset=False / branch reset=True):

| design | n_lags=1 | n_lags=2 | n_lags=4 |
|---|---|---|---|
| arm | 92 / 92 / 92 | 92 / 92 / 91 | 92 / 92 / 89 |
| rank, true | 92 / 92 / 92 | 91 / 91 / 91 | 89 / 89 / 89 |
| rank, believed | 89 / 89 / 89 | 88 / 88 / 88 | 86 / 86 / 86 |

(The rank designs already dropped `n_lags` leading rows on main, so reset=True
changes nothing there on one session.)

Coefficients: `reset=False` gives coefficients identical to main (compared as
printed to 4 decimals) for all three designs at n_lags = 1, 2, 4. At n_lags=1
`reset=True` is identical too. For the arm design at n_lags=2, dropping one more
row changes the fit, but the unpenalised fit is near-separable on 91 rows, so
individual coefficients are not meaningful (for example `Reward t-1`, class 1 vs
0: 0.488 on main and reset=False, 16.32 with reset=True). Column labels for
`drop_redundant_cols=True` are unchanged:
`Bias, Choice Arm1/Arm2 t-k, Reward t-k, Chx Rew Arm1/Arm2 t-k` (arm) and
`Bias, Chose Best/Second t-k, Reward t-k, Best*Reward/Second*Reward t-k` (rank).
With `drop_redundant_cols=False` the arm labels now include the `Arm3` columns.

Synthetic two sessions (ses-1 A,B,C,A,B; ses-2 C,A,B,C,A; constant magnitudes
1, 2, 3; `n_lags=1`): reset=False keeps 9 of 10 trials (1 dropped), reset=True
keeps 8 (the first trial of each session dropped). The extra reset=False row is
ses-2's first trial, which sees ses-1's last trial (chose B, reward 2) as history.

## Believed ranks: all four combinations on a synthetic example

`use_believed=True`, `n_lags=1`. ses-1 chooses A,B,C,A,B with magnitudes
A=3, B=2, C=1 (beliefs exist from its fourth trial); ses-2 chooses C,A,B with
A=1, B=2, C=3. Rows are `x` (Bias, Chose Best, Chose Second, Reward, Best*Reward,
Second*Reward at t-1) and `y` (best, second, third).

| reset_history | reset_beliefs | rows | which |
|---|---|---|---|
| False | False | 4 | ses-1 trial 5; ses-2 trials 1-3. Trial 1 (C) is believed third from ses-1's last-seen magnitudes (y = 0,0,1) although it pays best in ses-2. Equals main. |
| True | False | 3 | as above without ses-2's first trial (no in-session history); beliefs still carry, so ses-2 trials 2-3 are ranked |
| False | True | 1 | ses-1 trial 5 only: ses-2 never visits all arms within its 3 trials, so it has no belief |
| True | True | 1 | the same single row |

The sample data has one session, so `reset_beliefs_per_session` changes nothing
there: rows and coefficients are as in the table above for every setting.

## Review follow-ups

- Sessions are validated up front for both designs: every session must have the
  tower set of the first session (`session 2 of 2 has towers {'A', 'B'}, expected
  {'A', 'B', 'C'}`), and each choice and reward list must have one entry per trial
  of its session. The last session's reward list may lack its final entry in the
  arm design only; the rank design needs every tower's magnitude on every trial.
  Previously the rank design used the shortest list as its trial count and
  dropped trials silently. `RowCounts.n_trials` is now always `len(trial)`.
  This is stricter than main also for `n_lags=0`.
- `reset_beliefs_per_session=False` carries last-seen magnitudes from one session
  into the next. It is meant for the sessions of one problem. The function cannot
  tell problems apart, so the caller's grouping (`load_trials_by_problem` yields
  `{problem: {subject: sessions}}`) must pass one problem at a time.
- A NaN magnitude makes the ranking of that trial unknown; the trial gets no rank
  and its row (and rows using it as history) are dropped.
- The arm design's "expected exactly one chosen tower" error names subject,
  session and the session's own (1-based) trial.
- NaN-row drops are logged at WARNING.
- `scripts/to_integrate/testing_code.ipynb` passes `reset_history_per_session=True`
  explicitly (cell sources only); `False` reproduces the old concatenated behaviour.
