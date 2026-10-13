"""Reproducible, resumable synthetic recovery sweeps for the drifting task."""
import hashlib
import inspect
import itertools
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.mouse_agents import QLearningMouse
from src.tasks.drifting_bandit import DriftingBanditTask

NAMES = ['learning_rate', 'beta', 'alpha']


def recovery_design(config):
    """Full factorial design, with independent reproducible seeds per session."""
    combinations = itertools.product(
        config['swap_modes'], config['trial_lengths'],
        *[config['parameter_grid'][name] for name in NAMES],
        range(config['n_repeats']),
    )
    rows = []
    for mode, length, lr, beta, alpha, repeat in combinations:
        identity = [mode, length, lr, beta, alpha, repeat]
        digest = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        seed = int(np.random.SeedSequence(
            [config['seed'], int(digest[:8], 16)]
        ).generate_state(1)[0])
        rows.append(dict(run_id=digest[:16], swap_mode=mode, num_trials=length,
                         repeat=repeat, seed=seed, true_learning_rate=lr,
                         true_beta=beta, true_alpha=alpha))
    return pd.DataFrame(rows)


def run_recovery(config, fit_function, loss_function, bounds, output_root,
                 max_new_runs=None):
    """Save each fit and its synthetic data atomically; resume identical runs.

    max_new_runs limits new work for a pilot. Failed fits remain visible in the
    checkpoint; they are not silently discarded or retried on resume.
    """
    design = recovery_design(config)
    if design.empty or design.run_id.duplicated().any():
        raise ValueError('Design must contain unique, nonempty conditions.')
    manifest = dict(config=config, bounds=bounds, code={
        'task': inspect.getsource(DriftingBanditTask),
        'mouse': inspect.getsource(QLearningMouse),
        'runner': inspect.getsource(run_recovery),
        'design': inspect.getsource(recovery_design),
        'fit': inspect.getsource(fit_function),
        'loss': inspect.getsource(loss_function),
    })
    serialized = json.dumps(manifest, sort_keys=True, indent=2)
    experiment_id = hashlib.sha256(serialized.encode()).hexdigest()[:12]
    output = Path(output_root) / experiment_id
    output.mkdir(parents=True, exist_ok=True)
    (output / 'manifest.json').write_text(serialized)
    design.to_csv(output / 'design.csv', index=False)
    checkpoint = output / 'results.json'
    rows = json.loads(checkpoint.read_text()) if checkpoint.exists() else []
    complete = {row['run_id'] for row in rows}
    pending = design.loc[~design.run_id.isin(complete)]
    if max_new_runs is not None:
        pending = pending.head(max_new_runs)
    print(f'{len(rows)}/{len(design)} sessions saved; running {len(pending)}.\n{output}', flush=True)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    started = time.perf_counter()
    try:
        for count, row in enumerate(pending.to_dict('records'), 1):
            random.seed(row['seed'])
            np.random.seed(row['seed'])
            params = {name: row[f'true_{name}'] for name in NAMES}
            task = DriftingBanditTask(
                list(config['initial_arms']), swap_mode=row['swap_mode'],
                **config['task_parameters'],
            )
            mouse = QLearningMouse(len(config['initial_arms']), **params)
            actions, rewards, _ = task.simulate(mouse, row['num_trials'])
            actions, rewards = np.asarray(actions), np.asarray(rewards)
            num_arms = len(config['initial_arms'])
            data_file = output / f"session_{row['run_id']}.npz"
            with data_file.with_suffix('.tmp').open('wb') as handle:
                np.savez_compressed(handle, actions=actions, rewards=rewards,
                                    arm_history=np.asarray(task.arm_history))
            data_file.with_suffix('.tmp').replace(data_file)
            row['true_loss'] = float(loss_function(
                [params[name] for name in NAMES], actions, rewards, num_arms))
            row['uniform_loss'] = float(np.log(num_arms))
            fit_started = time.perf_counter()
            try:
                estimates, result = fit_function(
                    actions, rewards, num_arms=num_arms,
                    n_starts=config['n_starts'], seed=row['seed'],
                )
                row.update(success=True, loss=float(result.fun), message=str(result.message))
                for name, limits in zip(NAMES, bounds):
                    row[f'recovered_{name}'] = float(estimates[name])
                    row[f'at_bound_{name}'] = bool(any(
                        np.isclose(estimates[name], b, atol=1e-4, rtol=0) for b in limits))
            except RuntimeError as error:
                row.update(success=False, loss=None, message=str(error))
                for name in NAMES:
                    row[f'recovered_{name}'] = None
                    row[f'at_bound_{name}'] = None
            row['fit_seconds'] = time.perf_counter() - fit_started
            rows.append(row)
            temporary = checkpoint.with_suffix('.tmp')
            temporary.write_text(json.dumps(rows, indent=2, allow_nan=False))
            temporary.replace(checkpoint)
            elapsed = time.perf_counter() - started
            remaining = (len(pending) - count) * elapsed / count
            if count == 1 or count % 10 == 0 or count == len(pending):
                print(f'{len(rows)}/{len(design)} saved; remaining estimate '
                      f'{remaining / 60:.1f} min', flush=True)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    results = pd.DataFrame(rows)
    results.to_csv(output / 'results.csv', index=False)
    return results, output


def summarize_recovery(results, group_columns):
    """Summarise estimation error and failures without pooling task conditions."""
    summaries = []
    for keys, group in results.groupby(group_columns):
        if not isinstance(keys, tuple):
            keys = (keys,)
        successful = group.loc[group.success]
        for name in NAMES:
            errors = successful[f'recovered_{name}'] - successful[f'true_{name}']
            actual = successful[f'true_{name}']
            recovered = successful[f'recovered_{name}']
            correlation = (actual.corr(recovered) if len(actual) > 1
                           and actual.nunique() > 1 and recovered.nunique() > 1 else np.nan)
            summaries.append(dict(
                zip(group_columns, keys), parameter=name,
                n_sessions=len(group), n_success=len(successful),
                failure_rate=1 - len(successful) / len(group),
                bias=errors.mean(), mae=errors.abs().mean(),
                rmse=np.sqrt((errors ** 2).mean()), correlation=correlation,
                boundary_rate=successful[f'at_bound_{name}'].mean(),
            ))
    return pd.DataFrame(summaries)
