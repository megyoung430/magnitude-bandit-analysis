"""Reproducible, resumable synthetic recovery sweeps for the drifting task."""

import contextlib
import hashlib
import inspect
import itertools
import json
import random
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.mouse_agents import QLearningMouse
from src.tasks.drifting_bandit import DriftingBanditTask

NAMES = ["learning_rate", "beta", "alpha"]
# A bound is "hit" when the recovered value lies within this distance of it.
AT_BOUND_ATOL = 1e-4


def recovery_design(config):
    """Full factorial design, with independent reproducible seeds per session."""
    combinations = itertools.product(
        config["swap_modes"],
        config["trial_lengths"],
        *[config["parameter_grid"][name] for name in NAMES],
        range(config["n_repeats"]),
    )
    rows = []
    for mode, length, lr, beta, alpha, repeat in combinations:
        identity = [mode, length, lr, beta, alpha, repeat]
        digest = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        seed = int(
            np.random.SeedSequence([config["seed"], int(digest[:8], 16)]).generate_state(1)[0]
        )
        rows.append(
            dict(
                run_id=digest[:16],
                swap_mode=mode,
                num_trials=length,
                repeat=repeat,
                seed=seed,
                true_learning_rate=lr,
                true_beta=beta,
                true_alpha=alpha,
            )
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# run_recovery building blocks
# ---------------------------------------------------------------------------


def _write_json_atomically(path: Path, payload, *, indent: int | None = None) -> None:
    """Write via a temporary file so an interrupted run never leaves a partial file."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=indent, allow_nan=False))
    temporary.replace(path)


def _build_manifest(config: dict, bounds: list, fit_function: Callable, loss_function: Callable):
    """Config, bounds and the source of everything that determines the results.

    The whole module text is hashed, so editing a helper or a module constant (such as the
    at-bound tolerance) starts a new experiment instead of resuming stale checkpoints.

    Known issue: inspect.getsource needs plain functions, so a callable object passed as
    fit_function or loss_function raises TypeError.
    """
    return dict(
        config=config,
        bounds=bounds,
        code={
            "task": inspect.getsource(DriftingBanditTask),
            "mouse": inspect.getsource(QLearningMouse),
            "runner": inspect.getsource(run_recovery),
            "module": Path(__file__).read_text(),
            "design": inspect.getsource(recovery_design),
            "fit": inspect.getsource(fit_function),
            "loss": inspect.getsource(loss_function),
        },
    )


def _create_experiment_dir(output_root, manifest: dict, design: pd.DataFrame) -> Path:
    """Directory named by the manifest hash; holds the manifest and the design table."""
    serialized = json.dumps(manifest, sort_keys=True, indent=2)
    experiment_id = hashlib.sha256(serialized.encode()).hexdigest()[:12]
    output = Path(output_root) / experiment_id
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(serialized)
    design.to_csv(output / "design.csv", index=False)
    return output


@contextlib.contextmanager
def _preserved_global_rng() -> Iterator[None]:
    """Restore the global Python and NumPy RNG states on exit.

    The task and the mouse draw from the global generators, which each run seeds.
    """
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _simulate_session(row: dict, config: dict) -> tuple[dict, np.ndarray, np.ndarray, list]:
    """Simulate one synthetic session from the row's true parameters and seed.

    Seeds the global generators with the row's seed: the task and mouse draw from them.
    """
    random.seed(row["seed"])
    np.random.seed(row["seed"])
    params = {name: row[f"true_{name}"] for name in NAMES}
    task = DriftingBanditTask(
        list(config["initial_arms"]), swap_mode=row["swap_mode"], **config["task_parameters"]
    )
    mouse = QLearningMouse(len(config["initial_arms"]), **params)
    actions, rewards, _ = task.simulate(mouse, row["num_trials"])
    return params, np.asarray(actions), np.asarray(rewards), task.arm_history


def _save_session_data(path: Path, actions, rewards, arm_history) -> None:
    """Save the synthetic session atomically."""
    with path.with_suffix(".tmp").open("wb") as handle:
        np.savez_compressed(
            handle, actions=actions, rewards=rewards, arm_history=np.asarray(arm_history)
        )
    path.with_suffix(".tmp").replace(path)


def _at_bound(value: float, limits) -> bool:
    # Known issue: a None limit (an unbounded side) raises TypeError here, unlike the
    # cohort fits, which skip None limits.
    return bool(any(np.isclose(value, b, atol=AT_BOUND_ATOL, rtol=0) for b in limits))


def _fit_session(
    row: dict,
    fit_function: Callable,
    bounds: list,
    *,
    actions,
    rewards,
    num_arms: int,
    config: dict,
) -> None:
    """Fit one session and record the outcome in *row*; a failed fit stays visible.

    Only RuntimeError marks a failed fit; any other exception propagates.
    """
    try:
        estimates, result = fit_function(
            actions,
            rewards,
            num_arms=num_arms,
            n_starts=config["n_starts"],
            seed=row["seed"],
        )
        row.update(success=True, loss=float(result.fun), message=str(result.message))
        # Known issue: zip truncates when bounds has fewer entries than NAMES, silently
        # dropping the later parameters' recovered values from the row.
        for name, limits in zip(NAMES, bounds, strict=False):
            row[f"recovered_{name}"] = float(estimates[name])
            row[f"at_bound_{name}"] = _at_bound(estimates[name], limits)
    except RuntimeError as error:
        row.update(success=False, loss=None, message=str(error))
        for name in NAMES:
            row[f"recovered_{name}"] = None
            row[f"at_bound_{name}"] = None


def _recover_one_run(
    row: dict, config: dict, fit_function: Callable, loss_function: Callable, bounds: list, output
) -> None:
    """Simulate, save and fit one design row, adding the results to *row* in place."""
    params, actions, rewards, arm_history = _simulate_session(row, config)
    num_arms = len(config["initial_arms"])
    _save_session_data(output / f"session_{row['run_id']}.npz", actions, rewards, arm_history)
    row["true_loss"] = float(
        loss_function([params[name] for name in NAMES], actions, rewards, num_arms)
    )
    row["uniform_loss"] = float(np.log(num_arms))
    fit_started = time.perf_counter()
    _fit_session(
        row,
        fit_function,
        bounds,
        actions=actions,
        rewards=rewards,
        num_arms=num_arms,
        config=config,
    )
    row["fit_seconds"] = time.perf_counter() - fit_started


def _progress_line(n_saved: int, n_total: int, remaining_seconds: float) -> str:
    return f"{n_saved}/{n_total} saved; remaining estimate {remaining_seconds / 60:.1f} min"


def run_recovery(config, fit_function, loss_function, bounds, output_root, max_new_runs=None):
    """Save each fit and its synthetic data atomically; resume identical runs.

    max_new_runs limits new work for a pilot. Failed fits remain visible in the
    checkpoint; they are not silently discarded or retried on resume.
    """
    design = recovery_design(config)
    if design.empty or design.run_id.duplicated().any():
        raise ValueError("Design must contain unique, nonempty conditions.")
    manifest = _build_manifest(config, bounds, fit_function, loss_function)
    output = _create_experiment_dir(output_root, manifest, design)
    checkpoint = output / "results.json"
    rows = json.loads(checkpoint.read_text()) if checkpoint.exists() else []
    complete = {row["run_id"] for row in rows}
    pending = design.loc[~design.run_id.isin(complete)]
    if max_new_runs is not None:
        pending = pending.head(max_new_runs)
    print(
        f"{len(rows)}/{len(design)} sessions saved; running {len(pending)}.\n{output}", flush=True
    )
    started = time.perf_counter()
    with _preserved_global_rng():
        for count, row in enumerate(pending.to_dict("records"), 1):
            _recover_one_run(row, config, fit_function, loss_function, bounds, output)
            rows.append(row)
            _write_json_atomically(checkpoint, rows, indent=2)
            elapsed = time.perf_counter() - started
            remaining = (len(pending) - count) * elapsed / count
            if count == 1 or count % 10 == 0 or count == len(pending):
                print(_progress_line(len(rows), len(design), remaining), flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(output / "results.csv", index=False)
    return results, output


def summarize_recovery(results, group_columns):
    """Summarise estimation error and failures without pooling task conditions."""
    summaries = []
    for keys, group in results.groupby(group_columns):
        if not isinstance(keys, tuple):
            keys = (keys,)
        successful = group.loc[group.success]
        for name in NAMES:
            errors = successful[f"recovered_{name}"] - successful[f"true_{name}"]
            actual = successful[f"true_{name}"]
            recovered = successful[f"recovered_{name}"]
            correlation = (
                actual.corr(recovered)
                if len(actual) > 1 and actual.nunique() > 1 and recovered.nunique() > 1
                else np.nan
            )
            summaries.append(
                dict(
                    # Known issue: a bare string group_columns is zipped character by
                    # character, so only a list of column names labels the rows correctly.
                    zip(group_columns, keys, strict=False),
                    parameter=name,
                    n_sessions=len(group),
                    n_success=len(successful),
                    failure_rate=1 - len(successful) / len(group),
                    bias=errors.mean(),
                    mae=errors.abs().mean(),
                    rmse=np.sqrt((errors**2).mean()),
                    correlation=correlation,
                    boundary_rate=successful[f"at_bound_{name}"].mean(),
                )
            )
    return pd.DataFrame(summaries)
