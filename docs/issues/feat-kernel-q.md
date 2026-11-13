# feat-kernel-q

**Results change: none. New model.** Nothing that existed produces different numbers, with one
exception stated under "Fitted-Q random streams": the fitted Q-learner's simulated mice use a
different random stream than before.

## Request
Add the Q-learner of Aloor, Sit, Geuter, Tutt & Duan (bioRxiv 2026.10.31.763226, Methods "Q-learning
model", eqs 7-14), with asymmetric learning rates, forgetting of unchosen values and a multi-trial
choice kernel, to the strategy comparison, with four nested ablations, model and parameter
recovery, a fitted positive control and a held-out profile of the WSLS trace decay. Addendum: mice's
choice-logit weights over the last five choices are flat (about 0.2 to 0.3) while the agents' are
sharp (0.75 to 1.1 at lag 1, then about 0). The hypothesis is a slowly built choice kernel (habit)
that affects staying. Add a long-lag choice kernel with a choice tail and a known-kernel recovery,
P(stay) by run length, the fitted kernel rate per mouse and a WSLS (Habit) agent.

## Change
- `src/agents/kernel_q_agent.py`: the model. Exact likelihood vectorized over a mouse's problems
  with the forward-mode gradient (derivative of the values with respect to alpha_pos, alpha_neg
  and zeta and of the kernel with respect to alpha_k carried along), multi-start L-BFGS-B, `Variant`
  (free, fixed and tied parameters) and the five variants in `VARIANTS`, `parameters_at_bounds`,
  `value_agent_parameters`. Non-convergence of every start raises.
- `src/agents/drifting_simulation.py`: `KernelQPolicy`, so the model is simulated with the
  existing machinery.
- `src/behavior_analysis/strategy_models/strategy_models.py`: `KERNEL_Q`, `WSLS_HABIT` and the four ablations in the
  registry (`MODELS` are the candidates and recovery generators, `COMPARED` also holds the
  ablations, which are scored on the same folds but never win and are not generators). New paired
  comparisons (kernel Q against WSLS (Magnitude) and the Q-learner, WSLS (Habit) against WSLS
  (Magnitude) and the kernel Q-learner), `ABLATION_COMPARISONS`, and `decay_profile`: held-out nats
  of WSLS (Magnitude) at trace decays 0 (no trace), 0.3, 0.5, 0.7, 0.8, 0.9 and 0.95.
- `src/agents/null_agents.py`: WSLS (Habit) (`wsls_habit`), a null agent, a comparison candidate
  and a recovery generator.
- `src/behavior_analysis/choice_logit/choice_logit.py`: the `CHOICE_TAIL` extra and `ALL_EXTRAS`
  (`EXTRAS` is unchanged), `choice_logit_extensions.TAIL_VARIANTS`.
- `src/behavior_analysis/choice_logit/choice_kernel_timescale.py`, `stay_by_run_length.py`;
  `src/pipelines/choice_logit/choice_kernel_long.py` (long-lag fits cached per mouse, `recover_choice_kernel`),
  `strategy_comparison.py` (ablation, parameter and timescale tables, decay profile), `fitted_q.py`
  (the control generalized over `CONTROLS`).
- Figures (each saved as pdf, png, svg and a `.caption.txt`): `Kernel Q Ablations`, `Kernel Q
  Parameters`, `Choice Kernel Decay` (`plot_kernel_q.py`), `Choice Kernel Long`, `Stay by Run
  Length` (`plot_choice_kernel.py`). The comparison and recovery figures now show the new models
  (`plot_strategy_models.py`: 4-column grid, short panel-title name "Kernel Q-Learner").
- `scripts/choice_logit_values.ipynb`: sections 2.11 and 3.7, the kernel + forgetting fitted
  control next to the first one in 3.3 to 3.6, new parameters (`KERNEL_LONG_LAGS`,
  `KERNEL_RECOVERY_RATES`, `RECOVERY_REAL_MICE`).

## Equation mapping (paper to code)
With three arms `j`, graded reward `r` (pulses), values `Q`, kernel `K`:

| Paper | Here |
|---|---|
| eq 7: softmax of `beta Q + beta_k K + b` | `choice_probabilities`: softmax of `beta Q + beta_k K + plus_bias c` |
| eqs 8-9: `delta = r - Q_chosen` | same, `r` the pulses paid (not 0/1) |
| eqs 10-11: `alpha+` if `delta > 0`, else `alpha-` | `alpha_pos`, `alpha_neg`; `delta = 0` uses `alpha_neg` |
| eq 12: unchosen `Q <- zeta Q` | `zeta`, 1 is no forgetting |
| eqs 13-14: `K += alpha_k (onehot - K)` | `alpha_k`; every arm's kernel moves, the chosen toward 1 |

## Differences from the paper
1. **Reset.** The paper resets Q and K at every session start. Here Q starts at 0 for a problem
   and carries over its sessions (as the value agent's does, and as a mouse does: the reward
   probabilities drift within a problem across sessions), and K is cleared at each session start
   and after a dropped trial, as the value agent's previous choice is.
2. **Bias.** The paper has two arms and one left-right bias `b`. Here the spatial term is the plus-port
   bias of the value agent (`plus_bias`, 1 for a port in the row or column of the initiation port), so that
   the two Q-learners differ in the learning rule and the kernel only.
3. **Bounds.** Rates have a lower bound of 1e-3 (0 gives a value or kernel that never moves);
   `zeta` may reach 0 and 1. A fitted parameter within 1e-6 of a bound is reported (`at_bounds`).
4. **Graded reward.** `delta > 0` means better than expected, not rewarded.

Special case: `alpha_pos = alpha_neg`, `zeta = 1`, `alpha_k = 1` is the value agent with `beta_k` as
its stickiness; the likelihoods match (tested). The WSLS choice trace `h <- 0.9 h + chosen` is the
kernel with `alpha_k = 0.1`, scaled by 10 (tested).

## Ablations
`no_asymmetry` (alpha_neg tied to alpha_pos), `no_forgetting` (zeta = 1), `no_kernel` (beta_k = 0,
alpha_k held at 0.5, unidentified then) and `kernel_only`. The last is the value agent with its
one-trial stickiness replaced by a free multi-trial kernel (zeta = 1 and one learning rate): this is
the interpretation of "kernel only", and the one that separates the kernel from the other pieces.

## WSLS (Habit) and the choice tail
- WSLS (Habit): the stay logit of WSLS (Magnitude) plus `w_h` times the kernel of the arm just
  chosen (after its update; cleared at a session start and after a gap). It reduces to WSLS
  (Magnitude) at `w_h = 0` (tested). The kernel rate is chosen from `HABIT_RATES` (0.02 to 0.5, eight
  values) by the stay log likelihood on the training problems, because the likelihood is
  logistic regression at a fixed rate; the switch rule is unchanged. The grid costs up to about 20
  nats of log likelihood against the true rate in a test.
- Choice tail: an exponentially weighted indicator of the choice at each arm older than the last K
  choices, decay 0.9 (the reward tail's default, so both reach as far). Run with 15 lags and
  without or with the tail; the held-out nats are in `LongKernel.comparison`.
- Run length is the consecutive trials at the previous port within a session segment. P(stay) is
  reported after a zero payout and after a non-zero one (pooled over the non-zero magnitudes).
  Pooling is a choice: a port is kept longer when it pays well, so the magnitude mix of long runs
  differs from that of short runs. The agents are replayed on the same task so they carry the same
  mix; read the slope of an agent that ignores magnitude as what the mix alone produces. A bin
  needs 20 decisions, else it is left out with a warning. Slopes are per bin step.

## Fitted-Q random streams
`null_agents.KINDS` gained `wsls_habit`, and the fitted controls' stream indices follow
`len(na.KINDS)`. The first control (Q-Learner (Fitted)) is simulated from another stream than
before, so its simulated numbers differ in the last digit (second best after non-zero
0.537 ± 0.012 against 0.540 ± 0.017). Its fitted parameters are unchanged. The second control
"Q-Learner (Kernel + Forgetting, Fitted)" takes the next stream, so no two agents share numbers.

## Results (cohorts 1 and 3, 14 mice; notebook 3.6 and 3.7)
Model recovery (6 models, one replicate per mouse): WSLS (Magnitude) 64%, Heuristic+ 57%, Q-learner
71% (29% won by the kernel Q-learner, which contains it), WSLS (Graded) 93%, kernel Q 93%, WSLS (Habit)
100%. Recovered parameters (correlation): alpha_pos 0.93, zeta 0.96, alpha_k 0.96, beta 0.92, beta_k
0.97, w_h 0.98; **alpha_neg poorly (0.61)**, as 9 mice sit at its bound of 1.

Held-out comparison (nats per choice, first minus second, negative: first better): kernel Q minus
WSLS (Magnitude) -0.086 ± 0.012 (14 of 14 mice, p 0.0001), kernel Q minus Q-learner -0.094 ± 0.011
(14 of 14), WSLS (Habit) minus WSLS (Magnitude) -0.081 ± 0.008 (14 of 14), WSLS (Habit) minus kernel Q
+0.006 ± 0.006 (not distinguishable). Wins: kernel Q 9, WSLS (Habit) 5. Ablations (ablation minus
full): no kernel +0.030, kernel only +0.015, no forgetting +0.014, no asymmetry +0.004; all help
(Wilcoxon p 0.0001 to 0.004).

Fitted kernel Q-learner (mean ± se): alpha_pos 0.31 ± 0.04, alpha_neg 0.94 ± 0.03, zeta 0.41 ± 0.09,
alpha_k 0.26 ± 0.04, beta 0.71 ± 0.05, beta_k 1.59 ± 0.06. Per mouse (alpha_pos, alpha_neg, zeta,
alpha_k, beta, beta_k), alpha_neg at its bound of 1 where marked `*`, zeta at 0 where marked `^`:

| mouse | alpha_pos | alpha_neg | zeta | alpha_k | beta | beta_k |
|---|---|---|---|---|---|---|
| MY_174_LR | 0.284 | 1.000* | 0.219 | 0.136 | 0.789 | 1.801 |
| MY_174_R | 0.278 | 1.000* | 0.022 | 0.159 | 1.021 | 1.449 |
| MY_175_L | 0.173 | 1.000* | 0.000^ | 0.186 | 1.150 | 1.434 |
| MY_175_N | 0.631 | 0.742 | 0.110 | 0.122 | 0.500 | 1.791 |
| MY_66_LL | 0.148 | 1.000* | 0.889 | 0.179 | 0.684 | 1.797 |
| MY_66_LR | 0.083 | 0.754 | 0.910 | 0.540 | 0.612 | 1.845 |
| MY_66_N | 0.232 | 0.763 | 0.802 | 0.295 | 0.584 | 1.784 |
| MY_66_R | 0.317 | 1.000* | 0.741 | 0.634 | 0.719 | 1.730 |
| MY_71_L | 0.589 | 1.000* | 0.268 | 0.272 | 0.614 | 1.750 |
| MY_71_N | 0.271 | 1.000* | 0.034 | 0.162 | 0.878 | 1.263 |
| MY_71_R | 0.272 | 0.974 | 0.842 | 0.248 | 0.388 | 1.739 |
| MY_72_L | 0.394 | 1.000* | 0.231 | 0.255 | 0.675 | 1.447 |
| MY_72_N | 0.432 | 0.925 | 0.432 | 0.218 | 0.559 | 1.272 |
| MY_72_R | 0.262 | 1.000* | 0.268 | 0.232 | 0.777 | 1.220 |

Timescale of the kernel: fitted 1 - alpha_k of the kernel Q-learner 0.74 ± 0.04 (alpha_k 0.12 to
0.63), of WSLS (Habit) 0.66 ± 0.03 (rate 0.2 to 0.5 on the grid), geometric decay of the long-lag
choice weights 0.76 ± 0.05 (timescale 0.5 to 15 trials, median 4.7). WSLS (Magnitude)'s held-out
nats are lowest at trace decay 0.9 for 6 mice, 0.8 for 4, 0.7 for 3, 0.95 for 1; no trace is worse
by 0.005 to 0.031 nats for every mouse.

Long-lag kernel (15 lags): the mice's choice weights are flat, about 0.2 to 0.25 at lags 1 to 5,
near 0 by lags 8 to 10. The kernel + forgetting and habit agents are flat likewise and the sharp-kernel
agents (WSLS, WSLS (Recency), the first fitted Q-learner) are not. The choice tail does not help held out
(+0.0001 to +0.0086 nats, worse or equal in 13 of 14 mice). Known-kernel recovery (kernel-only agents,
alpha_k 0.05, 0.1, 0.3 on the mice's schedules) is recovered within the se bands at every lag.

P(stay) by run length: the slope per bin step is 0.083 (non-zero) and 0.055 (zero) for the mice;
0.001 for WSLS and WSLS (Recency), 0.006 WSLS (Magnitude), 0.011 WSLS (Graded), 0.023 the first
fitted Q-learner, and 0.078 and 0.064 for the kernel Q-learner control, 0.068 and 0.095 for WSLS (Habit).
The rise of P(stay) with run length needs a slowly built kernel.

## Habit as a null (switch regression and second best)
`switch_regression.NULL_AGENTS` gained `wsls_habit` as its fourth agent (the drawn band stays WSLS
(Magnitude), the first), so the switch-regression and second-best calibrations also run each mouse's
fitted WSLS (Habit) on the mouse's own task (100 replicates). A habit agent revisits its usual arms,
which may also be the arms that paid more, so habit alone could produce a value weight. Result
(cohorts 1 and 3, 14 mice): it does not. The habit source's raw weights are about 0 (believed 0.021 ±
0.013 after zero, 0.003 ± 0.010 after non-zero; true 0.008 ± 0.014, -0.005 ± 0.010; mice 0.088, 0.087,
0.016, 0.048). Calibrated z of the mice against the habit null, mean ± se (Wilcoxon p): believed after
non-zero 2.30 ± 0.52 (0.0006; against WSLS (Magnitude) 2.20 ± 0.50), believed after zero 1.21 ± 0.36
(0.013), true after non-zero 1.37 ± 0.46 (0.020), true after zero 0.24 ± 0.37 (0.58). Second best
against the habit null: second best -0.005 ± 0.454, believed-better 1.12 ± 0.54 (after zero) and 0.84 ±
0.80 (after non-zero), truly better 0.45 ± 0.42 and 0.11 ± 0.58, none different from 0. Mice that beat
the habit null on believed value after non-zero (empirical p < 0.05): MY_66_LL, MY_71_N, MY_71_R,
MY_72_L, MY_72_N, MY_72_R; on true value after non-zero: MY_66_LL, MY_71_R, MY_72_N, MY_72_R. The kernel
+ forgetting Q-learner source (a positive comparison, not a null) has raw weights between the habit and
the first fitted Q-learner's: believed 0.068 ± 0.020 / 0.056 ± 0.017, true 0.038 ± 0.023 / 0.038 ±
0.012 (after zero / after non-zero). The per-mouse z against that source is not computed; the power
check of 3.6 gives each mouse's rank among its replicates.

## Limits
- alpha_neg is at its bound for 9 of 14 mice; the asymmetry is poorly identified in this regime.
- The kernel Q-learner and WSLS (Habit) cannot be told apart by held-out fit.
- The model recovery of WSLS (Magnitude), Heuristic+ and the Q-learner is poor (nested models).
- The real recovery refits six models on every simulated mouse; with 2 workers it took 2 h 10 min
  for 14 mice (`RECOVERY_REAL_MICE` limits it).
- Run times on 2 cores, cold caches, 14 mice: comparison with the fitted controls 50 min, recovery
  2 h 10 min, regression extensions 23 min, switch regression 22 min, the whole notebook about 4.6 h.
