# feat-wsls-null-agents

## Request
Negative-control agents with no magnitude-guided switching, fitted to each mouse, simulated on
the mouse's own task and run through the choice-logit pipeline, to interpret the logit's reward
kernels, value psychometrics and stay/switch curves. A fourth request added an agent whose
*staying* depends on the magnitude just received.

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
    on fewer than 20 rewarded or unrewarded pairs, fewer than 30 informative switches, fewer than
    two distinct rewarded magnitudes, non-convergence, or a parameter at the +-20 bound (perfect
    separation). Likelihood is scored in-sample (2-4 parameters, thousands of choices).
- `src/agents/drifting_simulation.py`: `play_problem` runs any agent with `start_session`,
  `probabilities` and `update` through the task; `simulate_problem` is now `play_problem` with a
  `ValuePolicy`. Value-learner simulations are unchanged (identical output for the same seed).
- `src/pipelines/null_agents.py`: `simulate_null_cohort` plays one fitted agent per mouse on that
  mouse's towers, initiation port, starting magnitudes and session lengths and returns a
  `SyntheticCohort`; `null_controls` fits, simulates and analyses all three; parameter, score,
  observed-stay and implied-stay tables. Magnitudes follow the task rules measured from the data
  (drift, block lengths), not the mouse's recorded schedule: a block change zeroes the arm the
  agent chose most, so the recorded schedule only suits the mouse's own choices.
- `src/behavior_visualization/plot_null_agents.py`: fit and implied stay rule figure, kernels by
  source overlay. `plot_choice_logit.kernel_matrix` is now public (was `_kernel_matrix`).
- `scripts/choice_logit_values.ipynb`: sections 2.7 (Q-learner reference) and 3.3 (mice), figures
  in `negative-controls/` subfolders of the synthetic and real output folders.

## Results change
None: new agents and notebook section. Existing figures, tables and file names are unchanged.
