"""Participant-disjoint JAX/Optax Adam fitting for the two notebook RL models.

Paper-style procedure, with explicit mouse/session adaptations. Optional JAX and
Optax imports are delayed so the existing SciPy workflows need no new packages.
"""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

MODEL_NAMES = ('sticky_rl', 'best_rl')
PARAMETER_NAMES = {
    'sticky_rl': ['learning_rate', 'beta', 'alpha'],
    'best_rl': ['learning_rate', 'beta', 'kappa', 'bias', 'forgetting', 'q_init'],
}
FULL_GRID = [dict(optimizer_learning_rate=lr, l2_weight=l2, batch_size=bs)
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

    Best RL: sigmoid rates, positive softplus beta, tanh bias/kappa, free q_init.
    Sticky RL: sigmoid maps to existing fitting bounds for rate/beta/stickiness.
    """
    jax, jnp, _ = dependencies()
    if model == 'sticky_rl':
        return jnp.stack([.001+.999*jax.nn.sigmoid(raw[0]), .01+9.99*jax.nn.sigmoid(raw[1]),
                          3*jax.nn.sigmoid(raw[2])])
    if model == 'best_rl':
        return jnp.stack([jax.nn.sigmoid(raw[0]), jax.nn.softplus(raw[1])+1e-8,
                          jnp.tanh(raw[2]), jnp.tanh(raw[3]),
                          jax.nn.sigmoid(raw[4]), raw[5]])
    raise ValueError('Unknown model.')


def make_backend(model, num_arms=3, hidden_units=16):
    if model not in MODEL_NAMES:
        from src.agents.neural_bandit import make_backend as neural_backend
        return neural_backend(model, hidden_units=hidden_units, num_arms=num_arms)
    jax, jnp, optax = dependencies()
    if model not in MODEL_NAMES:
        raise ValueError('Unknown model.')

    @jax.jit
    def statistics(raw, actions, rewards, mask):
        params = transform_parameters(raw, model)
        if model == 'sticky_rl':
            lr, beta, stickiness = params
            bias, forgetting, q_init = 0., 0., 0.
        else:
            lr, beta, stickiness, bias, forgetting, q_init = params
        batch = actions.shape[0]
        initial = (jnp.full((batch, num_arms), q_init), jnp.full(batch, -1, dtype=jnp.int32),
                   jnp.zeros(batch), jnp.zeros(batch))
        index = jnp.arange(batch)

        def trial(carry, inputs):
            q, previous, nll, correct = carry
            action, reward, valid = inputs
            choice_bonus = jax.nn.one_hot(previous, num_arms)*stickiness
            logits = beta*(q+choice_bonus)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            nll += jnp.where(valid, -log_probs[index, action], 0.)
            correct += jnp.where(valid, jnp.argmax(logits, axis=-1) == action, 0.)
            updated = q.at[index, action].add(lr*(reward-q[index, action])+bias)
            updated = (1-forgetting)*updated+forgetting*q_init
            q = jnp.where(valid[:, None], updated, q)
            previous = jnp.where(valid, action, previous)
            return (q, previous, nll, correct), None

        final, _ = jax.lax.scan(trial, initial, (actions.T, rewards.T, mask.T))
        return final[2], final[3]

    def objective(raw, actions, rewards, mask, l2_weight):
        # Paper Eq. L: SUM over sessions and observed trials, not mean CE.
        nll, _ = statistics(raw, actions, rewards, mask)
        return nll.sum()+l2_weight*jnp.square(raw).sum()

    value_grad = jax.value_and_grad(objective)

    def trainer(setting):
        optimizer = optax.adam(setting['optimizer_learning_rate'])
        batch_size = setting['batch_size']
        l2 = setting['l2_weight']

        def step(carry, unused):
            raw, state, key, actions, rewards, mask = carry
            key, sampling_key = jax.random.split(key)
            ids = jax.random.randint(sampling_key, (batch_size,), 0, actions.shape[0])
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
        nll, correct = statistics(raw, *(x[offset:offset+evaluation_batch_size] for x in packed))
        for s, loss, count in zip(sequences[offset:offset+evaluation_batch_size], np.asarray(nll), np.asarray(correct)):
            n = len(s['actions'])
            rows.append(dict(mouse=s['mouse'], problem=int(s['problem']), session=s['session'],
                             n_trials=n, nll=float(loss), cross_entropy=float(loss/n),
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
                    validation_every=10_000, split_seed=42, fit_seed=123,
                    max_new_runs=None, profile='full', models=MODEL_NAMES):
    """Run/resume Adam sweep; select only on validation, score test after completion.

    Checkpoints contain parameters, Adam moments, sampling RNG and best validation
    iterate, so restarting an interrupted run resumes exactly at its saved step.
    Runs use matched initialization seeds/minibatch streams across the two models.
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
        if (setting['optimizer_learning_rate'] <= 0 or setting['l2_weight'] < 0
                or setting['batch_size'] < 1 or setting.get('hidden_units', 16) < 1):
            raise ValueError('Invalid optimizer setting.')
    parts, identities = split_by_mouse(sequences, split_seed)
    # Only training and validation histories enter training/selection functions.
    train_arrays = tuple(jnp.asarray(x) for x in pack_sequences(parts['train']))
    fingerprint = hashlib.sha256()
    for s in sequences:
        fingerprint.update(json.dumps([s['mouse'], int(s['problem']), s['session'], s['towers']]).encode())
        fingerprint.update(np.asarray(s['actions'], dtype=np.int64).tobytes())
        fingerprint.update(np.asarray(s['rewards'], dtype=np.float64).tobytes())
    manifest = dict(profile=profile, models=list(models), grid=grid, n_seeds=n_seeds, max_steps=max_steps,
                    validation_every=validation_every, split_seed=split_seed, fit_seed=fit_seed,
                    split_mice=identities, data_sha256=fingerprint.hexdigest(),
                    objective='sum_batch_nll + l2_weight * sum(raw_parameters ** 2)',
                    selection='lowest unregularized validation mean NLL, including step zero',
                    initialization='normal(0, .5) raw coordinates', precision='float64',
                    reset='each full extracted session', reward='unaltered delivered pulses',
                    versions=dict(jax=jax.__version__, optax=optax.__version__, numpy=np.__version__),
                    implementation=Path(__file__).read_text())
    if any(model in neural_bandit.NEURAL_MODELS for model in models):
        manifest['neural_implementation'] = Path(neural_bandit.__file__).read_text()
        manifest['neural_initialization'] = 'Glorot uniform weights; zero biases/states; forgetting .05'
    encoded = json.dumps(manifest, sort_keys=True, indent=2)
    output = Path(output_root)/hashlib.sha256(encoded.encode()).hexdigest()[:12]
    output.mkdir(parents=True, exist_ok=True)
    (output/'manifest.json').write_text(encoded)
    records, completed_now = [], 0
    for model in models:
        for setting_index, setting in enumerate(grid):
            stats, _, make_trainer = make_backend(model, hidden_units=setting.get('hidden_units', 16))
            optimizer, chunk = make_trainer(setting)
            for repeat in range(n_seeds):
                run_id = f'{model}_setting-{setting_index:02d}_seed-{repeat:02d}'
                done_file, checkpoint = output/f'{run_id}.json', output/f'{run_id}.npz'
                if done_file.exists():
                    records.append(json.loads(done_file.read_text()))
                    continue
                if max_new_runs is not None and completed_now >= max_new_runs:
                    continue
                init_key, sample_key = jax.random.split(jax.random.PRNGKey(fit_seed+repeat))
                if model in MODEL_NAMES:
                    raw = .5*jax.random.normal(init_key, (len(PARAMETER_NAMES[model]),), dtype=jnp.float64)
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
                    validation, _ = score_sequences(raw, parts['validation'], model, stats)
                    best_raw = raw
                    metadata = dict(step=0, best_step=0, best_validation=validation,
                                    history=[dict(step=0, validation_cross_entropy=validation)])
                while metadata['step'] < max_steps:
                    steps = min(validation_every, max_steps-metadata['step'])
                    (raw, state, sample_key), train_loss = chunk(raw, state, sample_key, *train_arrays, steps)
                    validation, _ = score_sequences(raw, parts['validation'], model, stats)
                    if not np.isfinite(validation) or not np.isfinite(float(train_loss)):
                        raise FloatingPointError(f'{run_id}: nonfinite objective; last finite checkpoint retained.')
                    metadata['step'] += steps
                    metadata['history'].append(dict(step=metadata['step'], validation_cross_entropy=validation,
                                                    last_regularized_batch_loss=float(train_loss)))
                    if validation < metadata['best_validation']:
                        metadata.update(best_step=metadata['step'], best_validation=validation)
                        best_raw = raw
                    _save_checkpoint(checkpoint, raw, state, sample_key, best_raw, metadata)
                    print(f'{run_id}: {metadata["step"]:,}/{max_steps:,} steps; '
                          f'validation CE {validation:.5f}', flush=True)
                serialized = ({k: np.asarray(v).tolist() for k, v in best_raw.items()}
                              if isinstance(best_raw, dict) else np.asarray(best_raw).tolist())
                summary = (dict(zip(PARAMETER_NAMES[model], np.asarray(transform_parameters(best_raw, model)).tolist()))
                           if model in MODEL_NAMES else neural_bandit.parameter_summary(best_raw, model))
                record = dict(model=model, setting_index=setting_index, repeat=repeat, setting=setting,
                              best_raw=serialized, parameters=summary,
                              **metadata)
                done_file.write_text(json.dumps(record, indent=2, allow_nan=False))
                records.append(record)
                completed_now += 1
    expected = len(models)*len(grid)*n_seeds
    if len(records) != expected:
        return dict(complete=False, completed_runs=len(records), expected_runs=expected,
                    output_dir=str(output), split_mice=identities, selected=[], test_sessions=[])
    selected, test_sessions = [], []
    for model in models:
        winner = min((r for r in records if r['model'] == model), key=lambda r: r['best_validation'])
        # The winner is fixed before the test data reach the scoring function.
        best_raw = ({k: jnp.asarray(v) for k, v in winner['best_raw'].items()}
                    if isinstance(winner['best_raw'], dict) else jnp.asarray(winner['best_raw']))
        test_ce, rows = score_sequences(best_raw, parts['test'], model)
        winner_file = output/f'{model}_selected_parameters.npz'
        with winner_file.open('wb') as file:
            np.savez(file, **best_raw) if isinstance(best_raw, dict) else np.savez(file, parameters=np.asarray(best_raw))
        selected.append(dict(model=model, setting=winner['setting'], repeat=winner['repeat'],
                             selected_step=winner['best_step'], parameters=winner['parameters'],
                             parameters_file=winner_file.name,
                             validation_cross_entropy=winner['best_validation'], test_cross_entropy=test_ce,
                             geometric_mean_choice_probability=float(np.exp(-test_ce))))
        test_sessions.extend(dict(model=model, **r) for r in rows)
    result = dict(complete=True, completed_runs=len(records), expected_runs=expected,
                  output_dir=str(output), split_mice=identities, selected=selected, test_sessions=test_sessions)
    (output/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
