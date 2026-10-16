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

from src.agents import mouse_agents
from src.agents.mouse_agents import QLearningMouse
from src.agents.run_store import (
    at_bound,
    library_versions,
    manifest_dir,
    write_atomically,
    write_json_atomically,
)
from src.tasks import drifting_bandit
from src.tasks.drifting_bandit import DriftingBanditTask

NAMES = ["learning_rate", "beta", "alpha"]
# A bound is "hit" when the recovered value lies within this distance of it.
_AT_BOUND_ATOL = 1e-4
# Names the experiment directory together with the configuration. It stands for this module's
# runner and design; the task and mouse have their own MODEL_VERSION, and the fit and loss
# functions are hashed by source.
# Bump when a change alters results; comments and refactors don't need a bump.
ALGORITHM_VERSION = 1


def recovery_design(config):
    """Full factorial design, with independent reproducible seeds per session.

    Args:
        config (dict): Needs ``swap_modes`` (list of task swap modes), ``trial_lengths`` (list of
            ints), ``parameter_grid`` (``{'learning_rate': [...], 'beta': [...], 'alpha': [...]}``),
            ``n_repeats`` (int) and ``seed`` (int).

    Returns:
        pd.DataFrame: One row per combination of swap mode, trial length, true parameter values and
            repeat, with columns ``run_id`` (16 hex characters naming the condition), ``swap_mode``,
            ``num_trials``, ``repeat``, ``seed`` (derived from ``config['seed']`` and the condition,
            independent of row order), ``true_learning_rate``, ``true_beta`` and ``true_alpha``.
    """
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


def _function_source(function: Callable) -> str:
    """Source text of a caller-supplied function; raise if it has none.

    The fit and loss functions are inputs to the experiment, so redefining one (even in a
    notebook's ``__main__``) must start a new experiment rather than reuse old results.
    """
    try:
        return inspect.getsource(function)
    except (OSError, TypeError) as error:
        raise ValueError(
            f"Cannot hash {function!r}: it has no retrievable source. Pass a function defined "
            "in a file or notebook cell, not a callable object or a function built with exec."
        ) from error


def _is_simple(value) -> bool:
    """True for immutable JSON-able values: numbers, strings, booleans, None, tuples of those.

    Lists and dicts are excluded: a captured list is often mutable state (a call log), and
    hashing it would move the directory between a run and its resume.
    """
    if isinstance(value, (bool, int, float, str, type(None), np.generic)):
        return True
    return isinstance(value, tuple) and all(_is_simple(item) for item in value)


def _closure_values(function: Callable) -> dict:
    """Closure variables of *function* that hold simple values, by name.

    A factory such as ``make(0.2)`` returns functions with identical source, so the captured
    values must enter the identity. Captured objects that are not simple values (arrays,
    functions, instances) are not hashed.
    """
    cells = getattr(function, "__closure__", None) or ()
    names = function.__code__.co_freevars
    return {
        name: cell.cell_contents
        for name, cell in zip(names, cells, strict=True)
        if _is_simple(cell.cell_contents)
    }


def _build_manifest(
    config: dict, *, bounds: list, fit_function: Callable, loss_function: Callable
) -> tuple[dict, dict]:
    """The identity of an experiment, and the source of this module's code.

    The identity (algorithm version, model versions, library versions, config, bounds,
    at-bound tolerance, and the source and simple closure values of the fit and loss
    functions) names the experiment
    directory. This module's own source is provenance only, so editing a helper does not
    start a new experiment; a change to the runner or design that alters results needs an
    ALGORITHM_VERSION bump.
    """
    code = {
        "task": inspect.getsource(DriftingBanditTask),
        "mouse": inspect.getsource(QLearningMouse),
        "runner": inspect.getsource(run_recovery),
        "module": Path(__file__).read_text(),
        "design": inspect.getsource(recovery_design),
    }
    identity = dict(
        algorithm_version=ALGORITHM_VERSION,
        model_versions=dict(
            drifting_bandit=drifting_bandit.MODEL_VERSION, mouse_agents=mouse_agents.MODEL_VERSION
        ),
        versions=library_versions("numpy", "scipy"),
        config=config,
        bounds=bounds,
        at_bound_atol=_AT_BOUND_ATOL,
        fit_source=_function_source(fit_function),
        fit_closure=_closure_values(fit_function),
        loss_source=_function_source(loss_function),
        loss_closure=_closure_values(loss_function),
    )
    return identity, dict(code=code)


def _create_experiment_dir(
    output_root, identity: dict, provenance: dict, design: pd.DataFrame
) -> Path:
    """Directory named by the identity hash; holds the manifest and the design table."""
    output = manifest_dir(output_root, identity, provenance)
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


def _save_session_data(path: Path, *, actions, rewards, arm_history) -> None:
    """Save the synthetic session atomically."""
    write_atomically(
        path,
        lambda handle: np.savez_compressed(
            handle, actions=actions, rewards=rewards, arm_history=np.asarray(arm_history)
        ),
    )


def _fit_session(
    row: dict,
    *,
    fit_function: Callable,
    bounds: list,
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
            row[f"at_bound_{name}"] = at_bound(estimates[name], limits, atol=_AT_BOUND_ATOL)
    except RuntimeError as error:
        row.update(success=False, loss=None, message=str(error))
        for name in NAMES:
            row[f"recovered_{name}"] = None
            row[f"at_bound_{name}"] = None


def _recover_one_run(
    row: dict,
    *,
    config: dict,
    fit_function: Callable,
    loss_function: Callable,
    bounds: list,
    output: Path,
) -> None:
    """Recover one design row's parameters, adding the results to *row* in place.

    Simulates the session, saves it, scores the true parameters and fits.
    """
    params, actions, rewards, arm_history = _simulate_session(row, config)
    num_arms = len(config["initial_arms"])
    _save_session_data(
        output / f"session_{row['run_id']}.npz",
        actions=actions,
        rewards=rewards,
        arm_history=arm_history,
    )
    row["true_loss"] = float(
        loss_function([params[name] for name in NAMES], actions, rewards, num_arms)
    )
    row["uniform_loss"] = float(np.log(num_arms))
    fit_started = time.perf_counter()
    _fit_session(
        row,
        fit_function=fit_function,
        bounds=bounds,
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

    max_new_runs limits new work for a pilot. Failed fits remain visible in the checkpoint; they are
    not silently discarded or retried on resume. The experiment directory is named by a hash of the
    configuration, bounds and the source of both functions, so changing any of them starts a new
    experiment. The global Python and NumPy random generators are restored afterwards.

    Args:
        config (dict): Everything that defines the experiment: the keys of ``recovery_design`` plus
            ``initial_arms`` (list of arm values, one of them 0), ``task_parameters`` (dict of extra
            ``DriftingBanditTask`` keyword arguments) and ``n_starts`` (int, fit restarts per
            session).
        fit_function (Callable): ``fit_function(actions, rewards, num_arms=, n_starts=, seed=)``
            returning ``(estimates, result)``: a dict mapping ``'learning_rate'``, ``'beta'`` and
            ``'alpha'`` to floats, and an optimizer result with ``fun`` and ``message``. It signals
            a failed fit by raising ``RuntimeError``. It must have retrievable source.
        loss_function (Callable): ``loss_function([learning_rate, beta, alpha], actions, rewards,
            num_arms)`` returning the mean negative log likelihood per trial. It must have
            retrievable source.
        bounds (list[tuple]): Three ``(low, high)`` pairs in the order learning_rate, beta, alpha,
            used to flag fits at a bound.
        output_root (str | Path): Directory that holds one subdirectory per experiment (created if
            missing).
        max_new_runs (int | None): Maximum number of design rows to run in this call; None runs all
            pending rows. Default None.

    Returns:
        tuple[pd.DataFrame, Path]: ``(results, output)``. ``results`` has one row per finished run:
            the ``recovery_design`` columns plus ``true_loss``, ``uniform_loss``, ``success``
            (bool), ``loss``, ``message``, ``fit_seconds``, and ``recovered_<name>`` and
            ``at_bound_<name>`` for each of learning_rate, beta and alpha (None for a failed fit).
            ``output`` is the experiment directory, which also holds the manifest, design,
            per-session ``.npz`` data, ``results.json`` and ``results.csv``.

    Raises:
        ValueError: If the design is empty or has duplicate conditions, or a function has no
            retrievable source.
    """
    design = recovery_design(config)
    if design.empty or design.run_id.duplicated().any():
        raise ValueError("Design must contain unique, nonempty conditions.")
    identity, provenance = _build_manifest(
        config, bounds=bounds, fit_function=fit_function, loss_function=loss_function
    )
    output = _create_experiment_dir(output_root, identity, provenance, design)
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
            _recover_one_run(
                row,
                config=config,
                fit_function=fit_function,
                loss_function=loss_function,
                bounds=bounds,
                output=output,
            )
            rows.append(row)
            write_json_atomically(checkpoint, rows, indent=2)
            elapsed = time.perf_counter() - started
            remaining = (len(pending) - count) * elapsed / count
            if count == 1 or count % 10 == 0 or count == len(pending):
                print(_progress_line(len(rows), len(design), remaining), flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(output / "results.csv", index=False)
    return results, output


def summarize_recovery(results, group_columns):
    """Summarise estimation error and failures without pooling task conditions.

    Args:
        results (pd.DataFrame): Output of ``run_recovery``.
        group_columns (list[str]): Columns whose combinations define the groups, e.g.
            ``['swap_mode', 'num_trials']``. Pass a list, not a bare string.

    Returns:
        pd.DataFrame: One row per group and parameter, with the ``group_columns``, ``parameter``
            (``'learning_rate'``, ``'beta'`` or ``'alpha'``), ``n_sessions``, ``n_success``,
            ``failure_rate``, and over successful fits only: ``bias`` (mean of recovered minus
            true), ``mae``, ``rmse``, ``correlation`` (NaN when fewer than two sessions or either
            value is constant) and ``boundary_rate`` (fraction of fits at a bound).
    """
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
