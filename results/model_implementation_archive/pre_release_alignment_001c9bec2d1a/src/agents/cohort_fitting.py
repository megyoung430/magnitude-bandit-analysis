"""Fit one Q-learning model per mouse with whole-problem cross-validation.

Each extracted session starts at Q=0 and no previous choice. Reward is the
recorded number of delivered reward pulses, not a post-trial arm magnitude.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from src.behavior_analysis.problem_utils import session_sort_key

PARAMETER_NAMES = ['learning_rate', 'beta', 'alpha']
DEFAULT_BOUNDS = [(0.001, 1.0), (0.01, 10.0), (0.0, 3.0)]


def prepare_cohort_sequences(problems, problem_numbers=range(5, 12)):
    """Convert grouped extracted sessions to arrays; return a full session audit.

    Reject a session with any invalid/missing choice or delivered reward rather
    than silently joining trials across missing learning history. All choices,
    including zero-reward choices, are retained in otherwise valid sessions.
    """
    sequences, audit = [], []
    for problem in problem_numbers:
        for mouse, sessions in sorted(problems.get(problem, {}).items()):
            for session in sorted(sessions, key=session_sort_key):
                data = sessions[session]
                choices = data.get('choice', [])
                row = dict(mouse=mouse, problem=int(problem), session=session,
                           n_trials=len(choices), included=False, reason='')
                try:
                    towers = sorted(data.get('choices_by_tower', {}))
                    if len(towers) != 3:
                        raise ValueError('Expected exactly three choice towers.')
                    if not choices:
                        raise ValueError('No completed choices.')
                    if any(choice not in towers for choice in choices):
                        raise ValueError('Missing or unrecognised choice.')
                    rewards = np.asarray(data.get('reward_pulses', []), dtype=float)
                    if rewards.shape != (len(choices),):
                        raise ValueError('Missing or misaligned delivered reward pulses.')
                    if not np.all(np.isfinite(rewards)) or np.any(rewards < 0):
                        raise ValueError('Missing, nonfinite, or negative delivered reward.')
                    available = data.get('choice_towers', [])
                    if len(available) != len(choices) or any(
                        not isinstance(value, (list, tuple, set, dict))
                        or set(value) != set(towers) for value in available
                    ):
                        raise ValueError('Missing/changing available towers within session.')
                    actions = np.array([towers.index(choice) for choice in choices], dtype=int)
                    sequences.append(dict(mouse=mouse, problem=int(problem), session=session,
                                          towers=towers, actions=actions, rewards=rewards))
                    row.update(included=True, reward_min=float(rewards.min()),
                               reward_max=float(rewards.max()), towers=','.join(towers))
                except (TypeError, ValueError) as error:
                    row['reason'] = str(error)
                audit.append(row)
    columns = ['mouse', 'problem', 'session', 'n_trials', 'included', 'reason',
               'reward_min', 'reward_max', 'towers']
    return sequences, pd.DataFrame(audit, columns=columns)


def sequence_log_probabilities(params, sequence, return_gradient=False):
    """Pre-choice log probabilities, with Q updated only after observing reward."""
    learning_rate, beta, alpha = params
    actions, rewards = sequence['actions'], sequence['rewards']
    n, k = len(actions), len(sequence['towers'])
    q, dq = np.zeros(k), np.zeros(k)
    preferences, derivatives, previous_choices = (np.zeros((n, k)) for _ in range(3))
    previous = None
    for t, (action, reward) in enumerate(zip(actions, rewards)):
        preferences[t] = q
        derivatives[t] = dq
        if previous is not None:
            preferences[t, previous] += alpha
            previous_choices[t, previous] = 1
        # d Q_new / d learning_rate = (1-lr)*d Q_old + reward-Q_old
        dq[action] = (1 - learning_rate) * dq[action] + reward - q[action]
        q[action] += learning_rate * (reward - q[action])
        previous = action
    logits = beta * preferences
    log_probs = logits - logsumexp(logits, axis=1, keepdims=True)
    if not return_gradient:
        return log_probs
    residual = np.exp(log_probs)
    residual[np.arange(n), actions] -= 1
    gradient = np.array([
        np.sum(residual * beta * derivatives),
        np.sum(residual * preferences),
        np.sum(residual * beta * previous_choices),
    ])
    return log_probs, gradient


def multi_session_loss(params, sequences):
    """Mean trial-weighted cross entropy and analytic gradient; reset per session."""
    total_loss, total_trials = 0.0, 0
    gradient = np.zeros(3)
    for sequence in sequences:
        log_probs, derivative = sequence_log_probabilities(params, sequence, True)
        actions = sequence['actions']
        total_loss -= log_probs[np.arange(len(actions)), actions].sum()
        total_trials += len(actions)
        gradient += derivative
    if total_trials == 0:
        raise ValueError('No trials available for fitting.')
    return total_loss / total_trials, gradient / total_trials


def fit_mouse_sessions(sequences, bounds=DEFAULT_BOUNDS, n_starts=10, seed=0):
    if n_starts < 1:
        raise ValueError('n_starts must be positive.')
    rng = np.random.default_rng(seed)
    starts = []
    for start in range(n_starts):
        initial = [rng.uniform(low, high) for low, high in bounds]
        result = minimize(multi_session_loss, initial, args=(sequences,), jac=True,
                          method='L-BFGS-B', bounds=bounds)
        starts.append(dict(start=start, success=bool(result.success),
                           loss=float(result.fun) if np.isfinite(result.fun) else None,
                           parameters=result.x.tolist(), message=str(result.message)))
    candidates = [fit for fit in starts if fit['success'] and fit['loss'] is not None]
    if not candidates:
        return None, starts
    return min(candidates, key=lambda fit: fit['loss']), starts


def run_problem_cross_validation(sequences, output_root, problem_numbers=range(5, 12),
                                 bounds=DEFAULT_BOUNDS, n_starts=10, seed=42,
                                 source_metadata=None, model="sticky_rl"):
    """Seven whole-problem folds per eligible mouse, with resumable saved fits.

    Only mice with usable trials in every requested problem are eligible.
    Training is trial-weighted within each mouse. Validation updates Q from
    past held-out observations, but never changes the fitted parameters.
    """
    if model == "sticky_rl":
        parameter_names = PARAMETER_NAMES
        fit_function = fit_mouse_sessions
        prediction_function = sequence_log_probabilities
    elif model == "best_rl":
        from src.agents import best_rl
        parameter_names = best_rl.PARAMETER_NAMES
        fit_function = best_rl.fit_mouse_sessions
        prediction_function = best_rl.sequence_log_probabilities
        if len(bounds) != 6:
            raise ValueError("Best RL requires six bounds; pass best_rl.DEFAULT_BOUNDS.")
    else:
        raise ValueError("Unknown model.")
    problem_numbers = sorted(set(problem_numbers))
    if len(problem_numbers) < 2:
        raise ValueError('Cross-validation requires at least two problems.')
    sequences = [s for s in sequences if s['problem'] in problem_numbers]
    fingerprint = hashlib.sha256()
    for s in sequences:
        fingerprint.update(json.dumps([s['mouse'], s['problem'], s['session'], s['towers']]).encode())
        fingerprint.update(np.asarray(s['actions'], dtype=np.int64).tobytes())
        fingerprint.update(np.asarray(s['rewards'], dtype=np.float64).tobytes())
    manifest = dict(problems=problem_numbers, bounds=bounds, n_starts=n_starts, seed=seed,
                    model=model, state_reset='each_extracted_session', reward='delivered_reward_pulses',
                    data_sha256=fingerprint.hexdigest(), source=source_metadata,
                    implementation=Path(__file__).read_text())
    if model == "best_rl":
        manifest['model_implementation'] = Path(best_rl.__file__).read_text()
        manifest['update_order'] = 'choice_then_chosen_update_then_all_action_forgetting'
    encoded = json.dumps(manifest, sort_keys=True, indent=2)
    output = Path(output_root) / hashlib.sha256(encoded.encode()).hexdigest()[:12]
    output.mkdir(parents=True, exist_ok=True)
    (output / 'manifest.json').write_text(encoded)
    eligibility = []
    for mouse in sorted({s['mouse'] for s in sequences}):
        found = {s['problem'] for s in sequences if s['mouse'] == mouse}
        missing = sorted(set(problem_numbers) - found)
        eligibility.append(dict(mouse=mouse, eligible=not missing,
                                missing_problems=','.join(map(str, missing))))
    eligibility = pd.DataFrame(eligibility, columns=['mouse', 'eligible', 'missing_problems'])
    eligibility.to_csv(output / 'mouse_eligibility.csv', index=False)
    rows, session_rows = [], []
    for mouse in eligibility.loc[eligibility.eligible, 'mouse']:
        mouse_sequences = [s for s in sequences if s['mouse'] == mouse]
        for held_out in problem_numbers:
            training = [s for s in mouse_sequences if s['problem'] != held_out]
            validation = [s for s in mouse_sequences if s['problem'] == held_out]
            fold_key = hashlib.sha256(f'{mouse}:{held_out}'.encode()).hexdigest()[:12]
            fold_file = output / f'fold_{fold_key}.json'
            if fold_file.exists():
                saved = json.loads(fold_file.read_text())
                rows.append(saved['fold'])
                session_rows.extend(saved['sessions'])
                continue
            fold_seed = int(np.random.SeedSequence([seed, int(fold_key[:8], 16)]).generate_state(1)[0])
            fit, starts = fit_function(training, bounds, n_starts, fold_seed)
            row = dict(mouse=mouse, held_out_problem=held_out,
                       training_problems=','.join(str(p) for p in problem_numbers if p != held_out),
                       n_train_trials=sum(len(s['actions']) for s in training),
                       n_test_trials=sum(len(s['actions']) for s in validation),
                       n_train_sessions=len(training), n_test_sessions=len(validation),
                       success=fit is not None, converged_starts=sum(s['success'] for s in starts))
            row.update(train_cross_entropy=None, test_cross_entropy=None,
                       uniform_cross_entropy=float(np.log(3)), improvement_over_uniform=None)
            for name in parameter_names:
                row[name] = None
                row[f'at_bound_{name}'] = None
            fold_sessions = []
            if fit is not None:
                params = fit['parameters']
                row['train_cross_entropy'] = fit['loss']
                nll = 0.0
                predictions = []
                for s in validation:
                    log_probs = prediction_function(params, s)
                    probabilities = np.exp(log_probs)
                    actions = s['actions']
                    losses = -log_probs[np.arange(len(actions)), actions]
                    nll += losses.sum()
                    fold_sessions.append(dict(mouse=mouse, held_out_problem=held_out,
                        session=s['session'], n_trials=len(actions), cross_entropy=float(losses.mean())))
                    for t, (action, reward, loss) in enumerate(zip(actions, s['rewards'], losses)):
                        predictions.append(dict(mouse=mouse, held_out_problem=held_out,
                            session=s['session'], trial=t + 1, chosen_tower=s['towers'][action],
                            reward=reward, negative_log_probability=loss,
                            **{f'p_{tower}': probabilities[t, a] for a, tower in enumerate(s['towers'])}))
                pd.DataFrame(predictions).to_csv(output / f'predictions_{fold_key}.csv', index=False)
                row.update(test_cross_entropy=float(nll / row['n_test_trials']),
                           uniform_cross_entropy=float(np.log(3)))
                row['improvement_over_uniform'] = row['uniform_cross_entropy'] - row['test_cross_entropy']
                for name, value, limits in zip(parameter_names, params, bounds):
                    row[name] = float(value)
                    row[f'at_bound_{name}'] = bool(any(np.isclose(value, b, atol=1e-4, rtol=0) for b in limits if b is not None))
            saved = dict(fold=row, sessions=fold_sessions, starts=starts)
            tmp = fold_file.with_suffix('.tmp')
            tmp.write_text(json.dumps(saved, indent=2, allow_nan=False))
            tmp.replace(fold_file)
            rows.append(row)
            session_rows.extend(fold_sessions)
            pd.DataFrame(rows).to_csv(output / 'fold_results.csv', index=False)
            print(f'{mouse}: held out problem {held_out}; '
                  f"{'fit saved' if fit else 'all starts failed'}", flush=True)
    results = pd.DataFrame(rows)
    sessions = pd.DataFrame(session_rows)
    results.to_csv(output / 'fold_results.csv', index=False)
    sessions.to_csv(output / 'held_out_sessions.csv', index=False)
    return results, sessions, eligibility, output
