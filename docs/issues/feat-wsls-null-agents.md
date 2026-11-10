# feat-wsls-null-agents

## Request
Negative-control agents with no value-guided switching, fitted to each mouse, simulated on the
mouse's own task and run through the choice-logit pipeline, to interpret the logit's reward
kernels, value psychometrics and stay/switch curves. One agent lets *staying* depend on the
magnitude just received.

## Change
- `src/agents/null_agents.py`: three soft win-stay/lose-shift agents (frozen dataclasses
  `WslsParams`, `WslsRecencyParams`, `WslsMagnitudeParams`).
  - `wsls`: P(stay) = sigmoid(`a_win`) after any reward, sigmoid(`a_lose`) after none; switches
    uniform over the two other arms.
  - `wsls_recency`: same stay rule; a switch picks the first other arm (sorted port order) with
    sigmoid(`w` * (h_first - h_second)).
  - `wsls_magnitude`: P(stay) = sigmoid(`a_win` + `b` * (m - 1)) after m >= 1 pulses; recency
    switches (`w` = 0 gives uniform ones, `b` = 0 gives plain WSLS).
  - History trace `h`: exponentially decaying count of past choices, decay 0.9 (the logit's reward
    tail default; half-life about 7 trials against 30-35 trial blocks), fixed, not fitted. A count
    says how much an arm was visited; "trials since last chosen" would not.
  - Session starts and the first trial after a dropped trial are uniform and clear the trace, like
    the logit's choice history. No tower bias, on first choices or on switches: it would add up to
    eight weights identified from the same switches that identify `w`.
  - Fits per mouse: `a_win`, `a_lose` are logits of the stay fractions (clipped to
    [0.005, 0.995]); `a_win`, `b` by a two-parameter logistic fit (analytic gradient); `w` by a
    one-dimensional maximum likelihood over switches with a non-zero trace difference. Fits raise
    on an empty table, fewer than 20 rewarded or unrewarded pairs, fewer than 30 informative
    switches, fewer than two distinct rewarded magnitudes, NaN trace differences,
    non-convergence, or a parameter at the +-20 bound (perfect separation). Parameters accept
    numpy scalars and reject None and non-finite values. Likelihood is scored in-sample (2-4
    parameters, thousands of choices), the logit's held out.
- `src/agents/drifting_simulation.py`: `play_problem` runs any agent with `start_session`,
  `probabilities` and `update` through the task; `simulate_problem` is `play_problem` with a
  `ValuePolicy`. The value learner's simulations are pinned by golden digests of
  `simulate_problem` and `simulate_mouse` (both learner kinds, two seeds).
- `src/pipelines/null_agents.py`: `simulate_null_cohort` plays one fitted agent per mouse on that
  mouse's towers, initiation port, starting magnitudes and session lengths and returns a
  `SyntheticCohort` tagged with the agent kind; `null_controls` fits, simulates and analyzes all
  three; `compare_with_controls` returns every table of the comparison with a reference analysis
  (parameters, in-sample scores next to the logit's held-out ones, observed and implied stay
  rule, psychometrics with baselines, stay/switch and switch-target curves, switch-target
  excess over the baseline); `plot_comparison` in `plot_null_agents.py` draws them.
  - Magnitudes follow the task rules, not the mouse's recorded schedule: a block change zeroes the
    arm the agent chose most, so the recorded schedule only suits the mouse's own choices.
  - Drift rate and block lengths are measured within each cohort (problems keyed `(cohort, k)`),
    never pooled across cohorts; a cohort whose rules cannot be estimated (no complete block, or
    the block-change rule is not followed) raises, naming the cohort, and so does a mouse with
    problems in two cohorts. Explicit rules may be one set or one per cohort.
  - `generating_values`, `reference_weights`, `value_agreement` and `kernel_comparison` raise on
    a cohort whose `kind` is not `"value"`.
- `src/behavior_visualization/plot_null_agents.py`: fit and implied stay rule figure, kernels by
  source overlay (each source's n in the legend when the sources' n differ).
  `plot_choice_logit.kernel_matrix` and `mice_stats` are public so the new figures reuse them.
- `scripts/choice_logit_values.ipynb`: sections 2.7 (Q-learner reference) and 3.3 (mice), figures
  in `negative-controls/` subfolders of the synthetic and real output folders.

## What the switch-target comparison can and cannot say
The logit's choice-history baseline has five choice lags and no choice tail. A recency trace with
decay 0.9 mostly lies beyond them, so recency alone gives above-baseline switch targets with no
value guidance: with `w` = 1.5 the rewarded-row excess (observed minus baseline, averaged over
bins) is about 0.30 at decay 0.9 and about 0 at decay 0.5, where the trace fits in five lags
(tested for `wsls_recency` and `wsls_magnitude`; `w` = 0 also gives about 0). The agents fitted to
the synthetic Q-learners have `w` of about 0.2 and excesses between -0.004 and 0.041. The mice's
perseveration is long, so their own excess may include the same effect. The logit and the baseline
are unchanged here; the switch regression of the next analysis controls for it with a long
choice trace and trials-since-chosen.

## Results change
None: new agents and notebook section. Existing figures, tables and file names are unchanged.
