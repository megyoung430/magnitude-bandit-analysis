# Published paper and released-code alignment

The notebooks now use the published article, **Hybrid neural–cognitive models reveal how memory shapes human reward learning**, Nature Human Behaviour 10, 972–987 (2026), [DOI 10.1038/s41562-025-02324-0](https://doi.org/10.1038/s41562-025-02324-0), and the authors' [released repository](https://github.com/google-deepmind/hybrid_rnns_reward_learning).

The implementation is pinned to commit **baf38da49fd1c1537affbc3745e961bbf0ae547d**. Unmodified Apache-2.0 reference files and their licence are in `tests/reference/hybrid_rnns_reward_learning`. Numerical tests compare outputs, recurrent states, likelihood gradients and AdamW updates with these files at matched parameter values, for both three and four arms. Our functional JAX implementation uses float64 and a different PRNG key allocation; matching initialization distributions does not mean identical random initial weights for the same integer seed.

## Model definitions

### Three distinct RL labels

- `sticky_rl`: the original notebook's three-parameter Q-learning plus nonnegative stickiness. The original parameter bounds and per-mouse L-BFGS-B analysis remain available. This is a local comparison model, not an upstream named preset.
- `best_rl`: the **six-parameter written-paper equations**: alpha, beta, update bias b, transient perseverance kappa, forgetting f, initial Q. Chosen delta-rule update includes b; all values then decay toward initial Q. The paper retains this specification in Eq. 2. Bias and initial Q are confounded through `b - alpha*Q_init` in choice likelihood under these equations; this warning still applies to this six-parameter model.
- `released_rl`: the **five-parameter released CogMod example**: alpha, beta, transient perseverance, f and initial Q. The example sets `fit_bias=True`, but [CogMod](https://github.com/google-deepmind/hybrid_rnns_reward_learning/blob/baf38da49fd1c1537affbc3745e961bbf0ae547d/hybrid_rnns_reward_learning/cogmod.py) never reads it or creates a bias parameter. Its persistent perseverance is disabled in the example. We reproduce this actual five-parameter computation rather than describing it as the paper's six-parameter model. It is equivalent to `best_rl` with b fixed to zero. ReLU beta can be exactly zero in code even though the paper states beta>0.

The published article also discusses variable-learning-rate Pearce–Hall variants (Eqs. 6–8). They are not the main Best RL model and are not implemented by the released CogMod class; these notebooks do not claim to reproduce that additional ablation sweep.

### Neural models

Use one-hot actions for all models. Hybrids have independent reward/action tanh networks with linear readouts, additive logits Q+c, beta=1 and channel weights both fixed to 1. Fit **three** scalar parameters in each hybrid: Q_init, c_init and a sigmoid-transformed forgetting rate. Hidden states start at zero. Initial Q and c are common scalars broadcast across arms.

Let D(Q)=(1-f)Q+f Q_init, and U be the reward network's scalar output. Inputs refer to state before the observation's update.

| Model | Reward input | Chosen Q after observation | Unchosen Q | Action-network input |
|---|---|---|---|---|
| RL-ANN | [Q(chosen), reward] | D(Q)(chosen)+U | D(Q) | one-hot action |
| Context-ANN | [Q(chosen), reward, Q vector] | D(Q)(chosen)+U | D(Q) | [one-hot action, c vector] |
| Memory-ANN | [0, reward, previous reward hidden state] | U | D(Q) | [one-hot action, previous action hidden state] |

The zero Memory-ANN input is a real input slot with H weights. Those weights have zero data gradient, but are included in parameter counts and AdamW decay. The action network replaces the entire c vector. No additional action-kernel decay is applied. The reward and action hidden states are global per session, not per arm. Reward memory does not directly observe actions; the external chosen-arm write routes its output to Q. For RL-ANN and Memory-ANN the scalar initial c does not affect later action-network outputs; it remains part of the released parameterization.

Vanilla RNN uses `[one-hot action, reward, previous hidden state]`, tanh hidden activation and a linear K-logit readout. No separate Q/c, forgetting, or beta parameter is fitted. Its output bias can encode first-choice arm preferences, but first choices are excluded from the released training objective.

These definitions follow [BiRNN](https://github.com/google-deepmind/hybrid_rnns_reward_learning/blob/baf38da49fd1c1537affbc3745e961bbf0ae547d/hybrid_rnns_reward_learning/bi_rnn.py) and [RNN](https://github.com/google-deepmind/hybrid_rnns_reward_learning/blob/baf38da49fd1c1537affbc3745e961bbf0ae547d/hybrid_rnns_reward_learning/rnn.py). The winning Memory preset is given in `train_models.ipynb`; RL-ANN/Context-ANN are configured via `s=False` with `o=False/True`, `zero_values=False`, fitted forgetting/Q_init/c_init and unit channel weights. Bare `rnn_config.py` defaults are an example configuration, not the final paper presets.

For K arms and H hidden units, counts are RL-ANN `H(2K+5)+K+4`; Context-ANN `H(4K+5)+K+4`; Memory-ANN `2H²+(2K+5)H+K+4`; Vanilla `H²+(2K+2)H+K`. At the paper's K=4 and H=16/64/32/64 these are **216/1352/2472/4740**, now matching the reported counts exactly.

### Initialization

Haiku Linear initializes weights from a normal truncated at ±2 standard deviations, with scale `1/sqrt(fan_in)`; biases are zero. Hybrid Q_init and c_init use N(1,1); raw forgetting uses N(0,1). Initial hidden states are zero. Released CogMod initializes raw alpha=1, raw beta=1, raw forgetting=0, raw transient perseverance=0 and Q_init~N(1,1). Its transforms are sigmoid, ReLU, sigmoid, tanh and identity. Our six-parameter paper model adds raw bias=0 with tanh; the original sticky model retains its established bounded parameterization.

## Training: paper, repository and mouse adaptation

| Detail | Released code | Notebook choice |
|---|---|---|
| Optimizer | `optax.adamw`, decoupled weight decay | Same; no squared-parameter penalty is added to likelihood |
| Sampling | Without replacement within each minibatch | Same; cap requested batch size at available training sessions and record the effective size |
| Inputs/targets | Feed observation t, predict choice t+1 | Same timing; score choices 2…T; first observation still updates state |
| Batch reduction | Sum over trials; divide by number of sampled sessions | Same; variable session lengths use padding masked out of loss and state updates |
| Probability adjustment | `p_adjusted=(1-1e-5)*p+5e-4` | Same training/selection score; also report proper unadjusted likelihood metrics |
| Data split | Utility slices first 10% blocks into train, next 10% validation, remaining 80% test; no subject split | Retain the **paper's** subject-disjoint 80/10/10 design, with mice as subjects and audited integer rounding |
| Monitoring | Every 500 loop iterations, sampled validation and test batches | Full validation at configured checkpoints (full profile: 500 updates); test withheld until selection completes |
| Sweep/selection | Public demo trains one configuration and returns last weights | Notebook implements the paper's grid, five seeds and validation-based checkpoint/configuration selection |

The paper describes Adam, summed NLL and sampling with replacement; the release concretely uses AdamW, session-mean loss and sampling without replacement. These disagreements are not silently treated as equivalent. The code's probability adjustment is **not a normalized categorical distribution**: its sum is `1-1e-5+K*5e-4`. It is retained as a released loss proxy. `cross_entropy`, NLL, argmax accuracy and geometric mean choice probability use the proper softmax; `release_nll`, `release_score_per_trial`, and selected-fit `*_release_score` name the adjusted proxy. Validation selection minimizes the proxy per scored trial; on a fixed validation dataset this ranks candidates identically to mean session summed proxy. `n_trials` in population reports is T−1; `n_observed_trials` retains T. Existing L-BFGS-B reports still include all T choices and use proper NLL; their scores must not be directly mixed with these population scores.

The exact released split/monitoring utility is retained for audit/reference tests but is not used for mouse data. We do not copy its hard-coded four-arm target slice: the models and targets use the task's actual arm count. Rewards remain delivered pulses; the released formatter does not rescale rewards. Whole extracted sessions serve as blocks, with no arbitrary 150-trial chopping. The notebooks continue to retain/exclude sessions according to the existing data audit.

## Profiles, outputs and migration

Restart the notebook kernel after updating so imported modules cannot retain the previous definitions. `pilot` checks execution; `full` uses the published grid. RL population fitting now offers original sticky RL, written-paper Best RL and released CogMod together; the neural section fits the four released-aligned neural architectures. The manuscript's hidden-width sweep applies only to neural networks. Full training remains explicit opt-in.

Old notebook/source versions were archived under `results/model_implementation_archive/pre_release_alignment_*`. Existing fits and results were not overwritten. A changed `ALGORITHM_VERSION`, configuration, protocol or upstream SHA produces a new manifest-specific run directory (source text no longer does; it is kept in `manifest.json` as provenance); old checkpoints are not resumed as though they used the new models, so bump `ALGORITHM_VERSION` (or the model module's `MODEL_VERSION`) when the model code changes results. Neural weights from before this update lack c_init and Memory's extra input row and must be refitted. Notebook outputs derived from superseded model code were cleared to avoid presenting them as current results.

Install `requirements-reference-tests.txt` only to run upstream parity tests; normal fitting still uses `requirements-paper-rl.txt`. Source code is Apache-2.0; the reference directory includes its license and provenance.


## Local pooled problem cross-validation

The RNN notebook's `local` profile now uses pooled leave-one-problem-out fitting,
via `split_by_problem` and `run_problem_sweep`. The same problem index is held out
across mice. The next available index cyclically is validation; all other indices
are training. This deterministic rule is independent of outcomes. Indices are
mouse-relative problem order, not a claim of identical task configurations.
Every validation/test mouse must have training data; otherwise fitting fails with
an explicit error. Sessions remain intact and recurrent state resets per session.

This asks about new problems in familiar mice; `pilot` and `full` retain the
paper-inspired mouse-disjoint split. Local uses four architectures, width 16,
learning rates 1e-3/1e-4, weight decay 1e-4, batch size 16, two seeds and 20,000
updates, with validation every 500 updates. That is 16 fits per fold, or 112 fits
and 2,240,000 updates for seven folds. `rnn_test_problems=None` requests every
available problem; a list such as `[5]` requests a preliminary subset. The
`max_new_runs` budget applies across folds. Each fold selects its own checkpoint,
configuration and seed using only validation; test predictions are aggregated
once per session after all requested folds complete. Fold-specific checkpoints,
splits and results are retained. Training-domain probes and held-out inspection
plots use the selected fold's partitions. Test results must not guide subsequent
hyperparameter selection. Per-mouse RL comparisons still require matched sessions
and scoring conventions; this change does not alter the RL notebook.
