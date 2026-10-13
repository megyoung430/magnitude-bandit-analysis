"""Subject-disjoint fitting with the released AdamW and shifted-loss convention.

Paper-style procedure, with explicit mouse/session adaptations. Optional JAX and
Optax imports are delayed so the existing SciPy workflows need no new packages.
"""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

SOURCE_COMMIT = 'baf38da49fd1c1537affbc3745e961bbf0ae547d'
SOURCE_URL = 'https://github.com/google-deepmind/hybrid_rnns_reward_learning'
MODEL_NAMES = ('sticky_rl', 'best_rl', 'released_rl')
PARAMETER_NAMES = {
    'sticky_rl': ['learning_rate', 'beta', 'alpha'],
    'best_rl': ['learning_rate', 'beta', 'kappa', 'bias', 'forgetting', 'q_init'],
    'released_rl': ['learning_rate', 'beta', 'kappa', 'forgetting', 'q_init'],
}
FULL_GRID = [dict(optimizer_learning_rate=lr, weight_decay=l2, batch_size=bs)
             for lr, l2, bs in itertools.product(
                 [1e-3, 1e-4, 1e-5], [1e-3, 1e-4, 1e-5], [32, 64, 128])]


def dependencies():
    try:
        import jax
        import jax.numpy as jnp
        import optax
    except ImportError as exc:
        raise ImportError('Paper-style fitting needs jax and optax in the notebook kernel. '
                          'Install requirements-paper-rl.txt in that environment.') from exc
    jax.config.update('jax_enable_x64', True)
    return jax, jnp, optax


def split_by_mouse(sequences, seed=42, fractions=(.8, .1, .1)):
    """Seeded mouse split: no session of a mouse can cross partitions.

    Floor training and validation counts, with at least one mouse per partition.
    Small cohorts cannot exactly match 80/10/10; returned identities are auditable.
    """
    if len(fractions) != 3 or min(fractions) <= 0 or not np.isclose(sum(fractions), 1):
        raise ValueError('Need three positive fractions summing to one.')
    mice = sorted({s['mouse'] for s in sequences})
    if len(mice) < 3:
        raise ValueError('Participant-disjoint training/validation/test needs at least three mice.')
    shuffled = np.random.default_rng(seed).permutation(mice).tolist()
    n_train = min(max(1, int(len(mice)*fractions[0])), len(mice)-2)
    n_val = min(max(1, int(len(mice)*fractions[1])), len(mice)-n_train-1)
    ids = dict(train=shuffled[:n_train], validation=shuffled[n_train:n_train+n_val],
               test=shuffled[n_train+n_val:])
    return {part: [s for s in sequences if s['mouse'] in group] for part, group in ids.items()}, ids


def split_by_problem(sequences, test_problem, validation_problem=None):
    """Pooled, whole-problem split; numbers are mouse-relative problem indices.

    Hold the same index out across mice. By default validation is the next
    available index cyclically, chosen without inspecting behavioral outcomes.
    Every validation/test mouse must be represented in training.
    """
    problems = sorted({int(s['problem']) for s in sequences})
    if len(problems) < 3 or test_problem not in problems:
        raise ValueError('Problem folds need at least three problems and a valid test problem.')
    if validation_problem is None:
        validation_problem = problems[(problems.index(test_problem)+1) % len(problems)]
    if validation_problem not in problems or validation_problem == test_problem:
        raise ValueError('Validation must be an available problem distinct from test.')
    ids = dict(train=[p for p in problems if p not in (test_problem, validation_problem)],
               validation=[int(validation_problem)], test=[int(test_problem)])
    parts = {part: [s for s in sequences if int(s['problem']) in group]
             for part, group in ids.items()}
    train_mice = {s['mouse'] for s in parts['train']}
    unseen = {s['mouse'] for part in ('validation', 'test') for s in parts[part]} - train_mice
    if unseen:
        raise ValueError(f'Problem folds require training data for held-out mice: {sorted(unseen)}')
    return parts, ids


def pack_sequences(sequences, max_trials=None, num_arms=3):
    """Pad complete sessions; masked trials contribute no loss and no state update."""
    if not sequences:
        raise ValueError('Empty data partition.')
    length = max(len(s['actions']) for s in sequences)
    if max_trials is not None and length > max_trials:
        raise ValueError('max_trials would truncate a session.')
    length = max_trials or length
    a = np.zeros((len(sequences), length), dtype=np.int32)
    r = np.zeros_like(a, dtype=float)
    m = np.zeros_like(a, dtype=bool)
    for i, s in enumerate(sequences):
        actions, rewards = np.asarray(s['actions']), np.asarray(s['rewards'], dtype=float)
        if (len(s['towers']) != num_arms or actions.ndim != 1 or not len(actions)
                or rewards.shape != actions.shape
                or not np.issubdtype(actions.dtype, np.integer)
                or np.any(actions < 0) or np.any(actions >= num_arms)
                or not np.all(np.isfinite(rewards))):
            raise ValueError('Invalid sequence, reward history or arm count.')
        n = len(actions)
        a[i, :n], r[i, :n], m[i, :n] = actions, rewards, True
    return a, r, m


def transform_parameters(raw, model):
    """Transform trainable coordinates; regularization is on these raw coordinates.

    Best/published RL: sigmoid rates, ReLU beta, tanh perseverance, free q_init.
    Paper-equation Best RL additionally has tanh bias, absent in released CogMod.
    Sticky RL: sigmoid maps to existing fitting bounds for rate/beta/stickiness.
    """
    jax, jnp, _ = dependencies()
    if model == 'sticky_rl':
        return jnp.stack([.001+.999*jax.nn.sigmoid(raw[0]), .01+9.99*jax.nn.sigmoid(raw[1]),
                          3*jax.nn.sigmoid(raw[2])])
    if model == 'best_rl':
        return jnp.stack([jax.nn.sigmoid(raw[0]), jax.nn.relu(raw[1]),
                          jnp.tanh(raw[2]), jnp.tanh(raw[3]),
                          jax.nn.sigmoid(raw[4]), raw[5]])
    if model == 'released_rl':
        return jnp.stack([jax.nn.sigmoid(raw[0]), jax.nn.relu(raw[1]),
                          jnp.tanh(raw[2]), jax.nn.sigmoid(raw[3]), raw[4]])
    raise ValueError('Unknown model.')


def make_backend(model, num_arms=3, hidden_units=16):
    if model not in MODEL_NAMES:
        from src.agents.neural_bandit import make_backend as neural_backend
        return neural_backend(model, hidden_units=hidden_units, num_arms=num_arms)
    jax, jnp, optax = dependencies()
    if model not in MODEL_NAMES:
        raise ValueError('Unknown model.')

    @jax.jit
    def statistics(raw, actions, rewards, mask, released_loss=False, skip_first=False):
        params = transform_parameters(raw, model)
        if model == 'sticky_rl':
            lr, beta, stickiness = params
            bias, forgetting, q_init = 0., 0., 0.
        elif model == 'released_rl':
            lr, beta, stickiness, forgetting, q_init = params
            bias = 0.
        else:
            lr, beta, stickiness, bias, forgetting, q_init = params
        batch = actions.shape[0]
        initial = (jnp.full((batch, num_arms), q_init), jnp.full(batch, -1, dtype=actions.dtype),
                   jnp.zeros(batch), jnp.zeros(batch))
        index = jnp.arange(batch)

        def trial(carry, inputs):
            q, previous, nll, correct = carry
            action, reward, valid, t = inputs
            choice_bonus = jax.nn.one_hot(previous, num_arms)*stickiness
            logits = beta*(q+choice_bonus)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            chosen = log_probs[index, action]
            adjusted = jnp.logaddexp(jnp.log1p(-1e-5)+chosen, jnp.log(5e-4))
            scored = valid & ((t > 0) | ~jnp.asarray(skip_first))
            nll += jnp.where(scored, -jnp.where(released_loss, adjusted, chosen), 0.)
            correct += jnp.where(scored, jnp.argmax(logits, axis=-1) == action, 0.)
            updated = q.at[index, action].add(lr*(reward-q[index, action])+bias)
            updated = (1-forgetting)*updated+forgetting*q_init
            q = jnp.where(valid[:, None], updated, q)
            previous = jnp.where(valid, action, previous)
            return (q, previous, nll, correct), None

        final, _ = jax.lax.scan(trial, initial, (actions.T, rewards.T, mask.T, jnp.arange(actions.shape[1])))
        return final[2], final[3]

    def objective(raw, actions, rewards, mask, l2_weight):
        # Released fit_hyb_rnn.loss_fn: score t>=1, affine probability safeguard,
        # sum across trials and average across sampled sessions. Decay is AdamW.
        nll, _ = statistics(raw, actions, rewards, mask, True, True)
        return nll.sum()/actions.shape[0]

    value_grad = jax.value_and_grad(objective)

    def trainer(setting):
        optimizer = optax.adamw(setting['optimizer_learning_rate'],
                                weight_decay=setting.get('weight_decay', setting.get('l2_weight', 0.)))
        batch_size = setting['batch_size']
        l2 = 0.  # Decoupled AdamW decay, not a term in the likelihood.

        def step(carry, unused):
            raw, state, key, actions, rewards, mask = carry
            key, sampling_key = jax.random.split(key)
            ids = jax.random.choice(sampling_key, actions.shape[0], (batch_size,), replace=False)
            loss, grad = value_grad(raw, actions[ids], rewards[ids], mask[ids], l2)
            updates, state = optimizer.update(grad, state, raw)
            raw = optax.apply_updates(raw, updates)
            return (raw, state, key, actions, rewards, mask), loss

        def chunk(raw, state, key, actions, rewards, mask, steps):
            carry, losses = jax.lax.scan(step, (raw, state, key, actions, rewards, mask),
                                         xs=None, length=steps)
            return carry[:3], losses[-1]

        return optimizer, jax.jit(chunk, static_argnums=(6,))

    return statistics, objective, trainer


def score_sequences(raw, sequences, model, statistics=None, evaluation_batch_size=32):
    if evaluation_batch_size < 1:
        raise ValueError('evaluation_batch_size must be positive.')
    statistics = statistics or make_backend(model)[0]
    packed = pack_sequences(sequences)
    rows = []
    for offset in range(0, len(sequences), evaluation_batch_size):
        batch = tuple(x[offset:offset+evaluation_batch_size] for x in packed)
        nll, correct = statistics(raw, *batch, False, True)
        release_nll, _ = statistics(raw, *batch, True, True)
        for s, loss, count, release_loss in zip(sequences[offset:offset+evaluation_batch_size], np.asarray(nll), np.asarray(correct), np.asarray(release_nll)):
            n = len(s['actions'])-1
            if n < 1:
                raise ValueError('Released scoring needs at least two trials per session.')
            rows.append(dict(mouse=s['mouse'], problem=int(s['problem']), session=s['session'],
                             n_trials=n, n_observed_trials=n+1, nll=float(loss), cross_entropy=float(loss/n),
                             release_nll=float(release_loss), release_score_per_trial=float(release_loss/n),
                             geometric_mean_choice_probability=float(np.exp(-loss/n)),
                             argmax_accuracy=float(count/n)))
    total_trials = sum(r['n_trials'] for r in rows)
    return sum(r['nll'] for r in rows)/total_trials, rows


def _save_checkpoint(path, raw, state, key, best_raw, metadata):
    jax, _, _ = dependencies()
    leaves, _ = jax.tree_util.tree_flatten(state)
    raw_leaves, _ = jax.tree_util.tree_flatten(raw)
    best_leaves, _ = jax.tree_util.tree_flatten(best_raw)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as file:
        np.savez(file, key=np.asarray(key),
                 metadata=np.array(json.dumps(metadata, allow_nan=False)),
                 **{f'raw_{i}': np.asarray(v) for i, v in enumerate(raw_leaves)},
                 **{f'best_{i}': np.asarray(v) for i, v in enumerate(best_leaves)},
                 **{f'opt_{i}': np.asarray(v) for i, v in enumerate(leaves)})
    tmp.replace(path)


def run_paper_sweep(sequences, output_root, grid=None, n_seeds=5, max_steps=1_000_000,
                    validation_every=500, split_seed=42, fit_seed=123,
                    max_new_runs=None, profile='full', models=MODEL_NAMES,
                    test_problem=None, validation_problem=None):
    """Run/resume AdamW sweep; select only on validation, score test after completion.

    Checkpoints contain parameters, AdamW moments, sampling RNG and best validation
    iterate, so restarting an interrupted run resumes exactly at its saved step.
    Runs use matched initialization seeds/minibatch streams across candidate models.
    Test scores are never computed during optimization or hyperparameter selection.
    """
    jax, jnp, optax = dependencies()
    from src.agents import neural_bandit
    if not models or len(set(models)) != len(models) or any(
        model not in (*MODEL_NAMES, *neural_bandit.NEURAL_MODELS) for model in models
    ):
        raise ValueError('Need unique supported model names.')
    if grid is None:
        grid = FULL_GRID
    if (not grid or n_seeds < 1 or max_steps < 1 or validation_every < 1
            or (max_new_runs is not None and max_new_runs < 0)):
        raise ValueError('Invalid sweep sizes.')
    for setting in grid:
        if (setting['optimizer_learning_rate'] <= 0 or setting.get('weight_decay', setting.get('l2_weight', 0.)) < 0
                or setting['batch_size'] < 1 or setting.get('hidden_units', 16) < 1):
            raise ValueError('Invalid optimizer setting.')
    if any(len(s['actions']) < 2 for s in sequences):
        raise ValueError('Released scoring requires at least two observed trials per session.')
    if test_problem is None:
        if validation_problem is not None:
            raise ValueError('validation_problem requires test_problem.')
        parts, identities = split_by_mouse(sequences, split_seed)
        split_details = dict(split_strategy='mouse', split_problems=None)
    else:
        parts, problem_ids = split_by_problem(sequences, test_problem, validation_problem)
        identities = {part: sorted({s['mouse'] for s in data}) for part, data in parts.items()}
        split_details = dict(split_strategy='pooled_problem', split_problems=problem_ids)
    # Only training and validation histories enter training/selection functions.
    train_arrays = tuple(jnp.asarray(x) for x in pack_sequences(parts['train']))
    fingerprint = hashlib.sha256()
    for s in sequences:
        fingerprint.update(json.dumps([s['mouse'], int(s['problem']), s['session'], s['towers']]).encode())
        fingerprint.update(np.asarray(s['actions'], dtype=np.int64).tobytes())
        fingerprint.update(np.asarray(s['rewards'], dtype=np.float64).tobytes())
    manifest = dict(profile=profile, models=list(models), grid=grid, n_seeds=n_seeds, max_steps=max_steps,
                    validation_every=validation_every, split_seed=split_seed, fit_seed=fit_seed,
                    **split_details, split_mice=identities, data_sha256=fingerprint.hexdigest(),
                    upstream_repository=SOURCE_URL, upstream_commit=SOURCE_COMMIT,
                    objective='mean_session_sum_t>=1(-log((1-1e-5)*p+5e-4))',
                    optimizer='optax.adamw; decoupled weight decay',
                    sampling='without replacement per minibatch; cap batch size at number of training sessions',
                    selection='lowest full-validation affine-adjusted score per trial, including step zero',
                    initialization='CogMod alpha_raw=1, beta_raw=1, perseverance_raw=0, forgetting_raw=0, q_init~N(1,1); paper bias_raw=0; sticky model retains N(0,.5)', precision='float64',
                    reset='each full extracted session', reward='unaltered delivered pulses',
                    versions=dict(jax=jax.__version__, optax=optax.__version__, numpy=np.__version__),
                    implementation=Path(__file__).read_text())
    if any(model in neural_bandit.NEURAL_MODELS for model in models):
        manifest['neural_implementation'] = Path(neural_bandit.__file__).read_text()
        manifest['neural_initialization'] = 'Haiku Linear truncated normal / sqrt(fan_in); zero biases/hidden; q_init,c_init~N(1,1), forgetting_raw~N(0,1)'
    from src.agents import best_rl, released_rl
    manifest['rl_implementations'] = {m: Path(module.__file__).read_text() for m, module in [('best_rl', best_rl), ('released_rl', released_rl)]}
    encoded = json.dumps(manifest, sort_keys=True, indent=2)
    output = Path(output_root)/hashlib.sha256(encoded.encode()).hexdigest()[:12]
    output.mkdir(parents=True, exist_ok=True)
    (output/'manifest.json').write_text(encoded)
    records, completed_now = [], 0
    for model in models:
        for setting_index, setting in enumerate(grid):
            stats, _, make_trainer = make_backend(model, hidden_units=setting.get('hidden_units', 16))
            training_setting = dict(setting, batch_size=min(setting['batch_size'], len(parts['train'])))
            optimizer, chunk = make_trainer(training_setting)
            for repeat in range(n_seeds):
                run_id = f'{model}_setting-{setting_index:02d}_seed-{repeat:02d}'
                done_file, checkpoint = output/f'{run_id}.json', output/f'{run_id}.npz'
                if done_file.exists():
                    records.append(json.loads(done_file.read_text()))
                    continue
                if max_new_runs is not None and completed_now >= max_new_runs:
                    continue
                init_key, sample_key = jax.random.split(jax.random.PRNGKey(fit_seed+repeat))
                if model == 'sticky_rl':
                    raw = .5*jax.random.normal(init_key, (3,), dtype=jnp.float64)
                elif model in ('best_rl', 'released_rl'):
                    q_init = 1+jax.random.normal(init_key, (), dtype=jnp.float64)
                    raw = jnp.array([1., 1., 0., 0., 0., q_init] if model == 'best_rl' else [1., 1., 0., 0., q_init])
                else:
                    raw = neural_bandit.initialize_parameters(init_key, model, setting.get('hidden_units', 16))
                state = optimizer.init(raw)
                if checkpoint.exists():
                    with np.load(checkpoint, allow_pickle=False) as saved:
                        metadata = json.loads(str(saved['metadata']))
                        raw_leaves, raw_tree = jax.tree_util.tree_flatten(raw)
                        raw = jax.tree_util.tree_unflatten(raw_tree, [jnp.asarray(saved[f'raw_{i}']) for i in range(len(raw_leaves))])
                        best_raw = jax.tree_util.tree_unflatten(raw_tree, [jnp.asarray(saved[f'best_{i}']) for i in range(len(raw_leaves))])
                        sample_key = jnp.asarray(saved['key'])
                        leaves, tree = jax.tree_util.tree_flatten(state)
                        state = jax.tree_util.tree_unflatten(tree, [jnp.asarray(saved[f'opt_{i}']) for i in range(len(leaves))])
                else:
                    validation_ce, validation_rows = score_sequences(raw, parts['validation'], model, stats)
                    validation = sum(r['release_nll'] for r in validation_rows)/sum(r['n_trials'] for r in validation_rows)
                    best_raw = raw
                    metadata = dict(step=0, best_step=0, best_validation=validation,
                                    history=[dict(step=0, validation_cross_entropy=validation_ce, validation_release_score=validation)])
                while metadata['step'] < max_steps:
                    steps = min(validation_every, max_steps-metadata['step'])
                    (raw, state, sample_key), train_loss = chunk(raw, state, sample_key, *train_arrays, steps)
                    validation_ce, validation_rows = score_sequences(raw, parts['validation'], model, stats)
                    validation = sum(r['release_nll'] for r in validation_rows)/sum(r['n_trials'] for r in validation_rows)
                    if not np.isfinite(validation) or not np.isfinite(float(train_loss)):
                        raise FloatingPointError(f'{run_id}: nonfinite objective; last finite checkpoint retained.')
                    metadata['step'] += steps
                    metadata['history'].append(dict(step=metadata['step'], validation_cross_entropy=validation_ce,
                                                    validation_release_score=validation, last_batch_objective=float(train_loss)))
                    if validation < metadata['best_validation']:
                        metadata.update(best_step=metadata['step'], best_validation=validation)
                        best_raw = raw
                    _save_checkpoint(checkpoint, raw, state, sample_key, best_raw, metadata)
                    print(f'{run_id}: {metadata["step"]:,}/{max_steps:,} steps; '
                          f'validation release score {validation:.5f}', flush=True)
                serialized = ({k: np.asarray(v).tolist() for k, v in best_raw.items()}
                              if isinstance(best_raw, dict) else np.asarray(best_raw).tolist())
                summary = (dict(zip(PARAMETER_NAMES[model], np.asarray(transform_parameters(best_raw, model)).tolist()))
                           if model in MODEL_NAMES else neural_bandit.parameter_summary(best_raw, model))
                record = dict(model=model, setting_index=setting_index, repeat=repeat, setting=setting,
                              effective_batch_size=training_setting['batch_size'],
                              best_raw=serialized, parameters=summary,
                              **metadata)
                done_file.write_text(json.dumps(record, indent=2, allow_nan=False))
                records.append(record)
                completed_now += 1
    expected = len(models)*len(grid)*n_seeds
    if len(records) != expected:
        return dict(complete=False, completed_runs=len(records), expected_runs=expected,
                    output_dir=str(output), split_mice=identities, **split_details, selected=[], test_sessions=[])
    selected, test_sessions = [], []
    for model in models:
        winner = min((r for r in records if r['model'] == model), key=lambda r: r['best_validation'])
        # The winner is fixed before the test data reach the scoring function.
        best_raw = ({k: jnp.asarray(v) for k, v in winner['best_raw'].items()}
                    if isinstance(winner['best_raw'], dict) else jnp.asarray(winner['best_raw']))
        validation_ce, _ = score_sequences(best_raw, parts['validation'], model)
        test_ce, rows = score_sequences(best_raw, parts['test'], model)
        winner_file = output/f'{model}_selected_parameters.npz'
        with winner_file.open('wb') as file:
            np.savez(file, **best_raw) if isinstance(best_raw, dict) else np.savez(file, parameters=np.asarray(best_raw))
        selected.append(dict(model=model, setting=winner['setting'], repeat=winner['repeat'],
                             selected_step=winner['best_step'], parameters=winner['parameters'],
                             parameters_file=winner_file.name,
                             validation_cross_entropy=validation_ce, validation_release_score=winner['best_validation'],
                             test_cross_entropy=test_ce,
                             test_release_score=sum(r['release_nll'] for r in rows)/sum(r['n_trials'] for r in rows),
                             effective_batch_size=winner['effective_batch_size'],
                             geometric_mean_choice_probability=float(np.exp(-test_ce))))
        test_sessions.extend(dict(model=model, **r) for r in rows)
    result = dict(complete=True, completed_runs=len(records), expected_runs=expected,
                  output_dir=str(output), split_mice=identities, **split_details, selected=selected, test_sessions=test_sessions)
    (output/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def run_problem_sweep(sequences, output_root, test_problems=None, max_new_runs=None, **kwargs):
    """Run pooled leave-one-problem-out fits, with independent selection per fold.

    max_new_runs is a global budget across folds, including resumable partial runs.
    Aggregate test reports are withheld until all requested folds complete.
    """
    problems = sorted({int(s['problem']) for s in sequences})
    tests = problems if test_problems is None else list(test_problems)
    if not tests or len(set(tests)) != len(tests):
        raise ValueError('Need unique nonempty test problems.')
    if max_new_runs is not None and max_new_runs < 0:
        raise ValueError('max_new_runs must be nonnegative.')
    # Validate every fold before starting any training.
    for problem in tests:
        split_by_problem(sequences, problem)
    folds, remaining = [], max_new_runs
    for problem in tests:
        print(f'Pooled problem fold: test problem {problem}', flush=True)
        before = run_paper_sweep(sequences, output_root, test_problem=problem,
                                 max_new_runs=0, **kwargs)
        if not before['complete'] and (remaining is None or remaining > 0):
            result = run_paper_sweep(sequences, output_root, test_problem=problem,
                                     max_new_runs=remaining, **kwargs)
            if remaining is not None:
                remaining -= result['completed_runs'] - before['completed_runs']
        else:
            result = before
        folds.append(dict(test_problem=int(problem), **result))
    identity = json.dumps([f['output_dir'] for f in folds])
    output = Path(output_root)/('problem-cv-'+hashlib.sha256(identity.encode()).hexdigest()[:12])
    output.mkdir(parents=True, exist_ok=True)
    complete = all(f['complete'] for f in folds)
    result = dict(complete=complete, split_strategy='pooled_problem',
                  completed_runs=sum(f['completed_runs'] for f in folds),
                  expected_runs=sum(f['expected_runs'] for f in folds),
                  output_dir=str(output), folds=folds, selected=[], test_sessions=[])
    if complete:
        result['selected'] = [dict(row, test_problem=f['test_problem'],
                                  parameters_file=str((Path(f['output_dir'])/row['parameters_file']).resolve()))
                              for f in folds for row in f['selected']]
        result['test_sessions'] = [dict(row, test_problem=f['test_problem'])
                                   for f in folds for row in f['test_sessions']]
    (output/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
