# feat-glm-hmm

**Results change: none — new analysis.** Nothing that existed produces different numbers.

## Request
Do the mice switch between strategy states? Fit a GLM-HMM (a mixture-of-agents HMM) to the
choices, select the number of states by held-out likelihood, and calibrate the gain over one state
against replays of a fitted static agent. Notebook: `scripts/glm_hmm.ipynb`; figures in
`results/figures/glm-hmm/`.

## Model
- **Own implementation** (`src/behavior_analysis/glm_hmm/glm_hmm.py`); no new dependency.
- **Emissions**: a three-alternative conditional logit. In state `z`, `P(arm j) = softmax_j(w_z . x_j)`
  with per-arm features `x_j`. Weights are state specific.
- **Features** (`src/agents/glm_hmm_agent.py`, `FEATURES`): `q` (kernel-Q value of the arm), `habit`
  (the choice kernel), `stay_win` and `stay_magnitude` and `stay_zero` (stay after a reward, per
  extra pulse, after a zero), `switch_trace` (a decaying trace, decay 0.9, of the arms switched to
  recently, centered over the two non-previous arms) and `plus_bias`.
- **Signals come from the mouse's fitted kernel-Q** (full-data fit). This leaks a little
  information about the held-out sessions into the features but is identical across K, so it
  cancels in the gain over K = 1 (it is also the same fit the null agent uses).
- **WSLS (Magnitude)** maps exactly onto the weights only when the magnitude weight is 0 (stay =
  `a + ln 2`); otherwise approximately, because the three-way softmax scales the switch mass by a
  cosh factor. The exact case and the switch-target rule are tested.
- **Transitions**: sticky Dirichlet prior (MAP): `alpha = 2` off-diagonal, extra `kappa = 20` on
  the diagonal, initial-state `alpha = 2`; ridge `l2 = 1` on the weights. Each session is an
  independent sequence and the state is reset at the session start (initial distribution learned);
  a segment also restarts after a gap. The alternative (carry the state across sessions) is not
  fitted.
- **Fitting**: EM (generalized: L-BFGS M-step, monotonicity checked each iteration) with 10
  restarts (a split of the K - 1 fit, k-means of per-session weights, random perturbations), a
  12-iteration screening stage whose best 2 continue; states sorted by the weight on `q`. Fails
  loudly if no restart converges.

## Model selection
- K = 1 to 4. **5 folds of sessions**; the sessions of each problem are dealt round-robin with a
  running counter across problems, so every fold holds every problem.
- Score: forward-filter predictive log probability of the held-out sessions (scored trials only; a
  segment's first trial is not scored). **Gain** = nats(K = 1) - nats(K), per trial.
- Selected K: the smallest K within one standard error of the best (paired per-fold differences).
- Static models from #119 (kernel-Q, WSLS (Habit)) are scored on the same folds.

## Simulations
Three-state switching agent (value, WSLS (Magnitude), habit/biased), dwell 300 trials (slow) or 50
(fast), plus no-switching controls: the fitted kernel-Q, the fitted WSLS (Habit), and a stationary
blend of the three weight vectors. Run on three mice's schedules.

## Null
Replays of the fitted kernel-Q agent on the mouse's problems, same CV protocol (K up to 3), then
`rank_calibration` (z, empirical p, Wilcoxon across mice). **12 replays per mouse, not the 20
asked for** (runtime: about 80 minutes on 2 cores).

## Caveats
- The static WSLS (Habit) control produces a small positive gain (selected K = 2 or 3 in the
  three mice tested), so a gain of that size is not by itself evidence of switching. The mice's
  gains (0.004 to 0.02 nats per trial) are in that range and about ten times below the genuine
  switching simulations (0.045 to 0.084).
- Kernel by state: the choice logit (5 lags) refit on trials with posterior above 0.8 of a state,
  needing at least 200 trials; flatness = mean(lag 3-5) / lag 1.
- Data: cohort 1 problems up to 7, cohort 3 all (as `feat-rank-signatures`).

## References
- Ashwood, Roy, Stone et al. (2022), Mice alternate between discrete strategies during
  perceptual decision-making, Nature Neuroscience.
- Venditto, Miller, Brody, Daw (2024), Dynamic reinforcement learning reveals time-dependent
  shifts in strategy during reward learning, eLife.
