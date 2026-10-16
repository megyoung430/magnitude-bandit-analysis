"""Fit one Q-learning model per mouse with whole-problem cross-validation.

Each extracted session starts at Q=0 and no previous choice. Reward is the
recorded number of delivered reward pulses, not a post-trial arm magnitude.
"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from src.agents.run_store import (
    at_bound,
    fingerprint_sequences,
    library_versions,
    manifest_dir,
    write_json_atomically,
)
from src.behavior_import.session_order import ordered_session_keys

PARAMETER_NAMES = ["learning_rate", "beta", "alpha"]
DEFAULT_BOUNDS = [(0.001, 1.0), (0.01, 10.0), (0.0, 3.0)]
_AUDIT_COLUMNS = [
    "mouse",
    "problem",
    "session",
    "n_trials",
    "included",
    "reason",
    "reward_min",
    "reward_max",
    "towers",
]
# A bound is "hit" when the fitted value lies within this distance of it.
_AT_BOUND_ATOL = 1e-4
# Names the saved-run directory together with the configuration. It stands for the code in
# this module (the sticky RL model and the cross-validation); best_rl and released_rl have
# their own MODEL_VERSION.
# Bump when a change alters results; comments and refactors don't need a bump.
ALGORITHM_VERSION = 1


def _validated_session_arrays(data: dict, choices: list) -> tuple[list, np.ndarray, np.ndarray]:
    """Towers, action indices and delivered rewards of one session; raise if unusable."""
    towers = sorted(data.get("choices_by_tower", {}))
    if len(towers) != 3:
        raise ValueError("Expected exactly three choice towers.")
    if not choices:
        raise ValueError("No completed choices.")
    if any(choice not in towers for choice in choices):
        raise ValueError("Missing or unrecognised choice.")
    rewards = np.asarray(data.get("reward_pulses", []), dtype=float)
    if rewards.shape != (len(choices),):
        raise ValueError("Missing or misaligned delivered reward pulses.")
    if not np.all(np.isfinite(rewards)) or np.any(rewards < 0):
        raise ValueError("Missing, nonfinite, or negative delivered reward.")
    available = data.get("choice_towers", [])
    if len(available) != len(choices) or any(
        not isinstance(value, (list, tuple, set, dict)) or set(value) != set(towers)
        for value in available
    ):
        raise ValueError("Missing/changing available towers within session.")
    actions = np.array([towers.index(choice) for choice in choices], dtype=int)
    return towers, actions, rewards


def prepare_cohort_sequences(problems, problem_numbers=range(5, 12)):
    """Convert grouped extracted sessions to arrays; return a full session audit.

    Reject a session with any invalid/missing choice or delivered reward rather than silently
    joining trials across missing learning history. All choices, including zero-reward choices, are
    retained in otherwise valid sessions.

    Args:
        problems (dict): ``{problem: {mouse: {session: session_data}}}``. Each ``session_data`` dict
            needs ``choice`` (list of chosen tower names, one per completed trial),
            ``choices_by_tower`` (dict with exactly three tower names as keys), ``reward_pulses``
            (one delivered-pulse count per choice) and ``choice_towers`` (per trial, the collection
            of available towers, which must equal the three towers). A problem absent from the dict
            is skipped.
        problem_numbers (Iterable[int]): Problems to extract, in this order. Default ``range(5,
            12)``.

    Returns:
        tuple[list[dict], pd.DataFrame]: ``(sequences, audit)``. ``sequences`` has one dict per
            valid session, ordered by *problem_numbers*, then mouse name, then session order, with
            keys ``mouse`` (str), ``problem`` (int), ``session``, ``towers`` (sorted list of three
            tower names), ``actions`` (``(n_trials,)`` int array of indices into ``towers``) and
            ``rewards`` (``(n_trials,)`` float array). ``audit`` has one row per session seen,
            including rejected ones, with columns ``mouse``, ``problem``, ``session``, ``n_trials``,
            ``included`` (bool), ``reason`` (rejection message, empty if included), ``reward_min``,
            ``reward_max`` and ``towers`` (comma-joined; the last three are NaN for rejected
            sessions).
    """
    sequences, audit = [], []
    for problem in problem_numbers:
        for mouse, sessions in sorted(problems.get(problem, {}).items()):
            for session in ordered_session_keys(mouse, sessions):
                data = sessions[session]
                choices = data.get("choice", [])
                row = dict(
                    mouse=mouse,
                    problem=int(problem),
                    session=session,
                    n_trials=len(choices),
                    included=False,
                    reason="",
                )
                try:
                    towers, actions, rewards = _validated_session_arrays(data, choices)
                    sequences.append(
                        dict(
                            mouse=mouse,
                            problem=int(problem),
                            session=session,
                            towers=towers,
                            actions=actions,
                            rewards=rewards,
                        )
                    )
                    row.update(
                        included=True,
                        reward_min=float(rewards.min()),
                        reward_max=float(rewards.max()),
                        towers=",".join(towers),
                    )
                except (TypeError, ValueError) as error:
                    row["reason"] = str(error)
                audit.append(row)
    return sequences, pd.DataFrame(audit, columns=_AUDIT_COLUMNS)


def sequence_log_probabilities(params, sequence, return_gradient=False):
    """Pre-choice log probabilities, with Q updated only after observing reward.

    The model is Q-learning with softmax choice and a stickiness bonus on the previous arm. Q starts
    at 0 and there is no previous choice at the session start.

    Args:
        params (array-like): Three floats ``(learning_rate, beta, alpha)`` in the order of
            ``PARAMETER_NAMES``; ``alpha`` is the stickiness bonus.
        sequence (dict): One session as built by ``prepare_cohort_sequences``. Uses ``actions``
            (``(n_trials,)`` int array of chosen-tower indices into ``towers``), ``rewards``
            (``(n_trials,)`` float array of delivered reward pulses) and ``towers`` (list of tower
            names; its length is the number of arms).
        return_gradient (bool): If True also return the gradient of the summed chosen-arm log
            probability. Default False.

    Returns:
        np.ndarray | tuple[np.ndarray, np.ndarray]: ``log_probs``, a ``(n_trials, n_arms)`` array of
            log probabilities over arms. With ``return_gradient`` a tuple ``(log_probs, gradient)``;
            ``gradient`` is a ``(3,)`` array in ``PARAMETER_NAMES`` order.
    """
    learning_rate, beta, alpha = params
    actions, rewards = sequence["actions"], sequence["rewards"]
    n, k = len(actions), len(sequence["towers"])
    q, dq = np.zeros(k), np.zeros(k)
    preferences, derivatives, previous_choices = (np.zeros((n, k)) for _ in range(3))
    previous = None
    # Known issue: zip truncates silently when actions and rewards differ in length;
    # prepare_cohort_sequences guarantees equal lengths, hand-built sequences are unchecked.
    for t, (action, reward) in enumerate(zip(actions, rewards, strict=False)):
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
    gradient = np.array(
        [
            np.sum(residual * beta * derivatives),
            np.sum(residual * preferences),
            np.sum(residual * beta * previous_choices),
        ]
    )
    return log_probs, gradient


def multi_session_loss(params, sequences):
    """Mean trial-weighted cross entropy and analytic gradient; reset per session.

    Args:
        params (array-like): Three floats ``(learning_rate, beta, alpha)``.
        sequences (list[dict]): Sessions as built by ``prepare_cohort_sequences``; each dict needs
            ``actions``, ``rewards`` and ``towers``.

    Returns:
        tuple[float, np.ndarray]: ``(loss, gradient)``: the mean negative log likelihood per trial
            over all sessions, and its ``(3,)`` gradient with respect to ``params``.

    Raises:
        ValueError: If the sessions contain no trials.
    """
    total_loss, total_trials = 0.0, 0
    gradient = np.zeros(3)
    for sequence in sequences:
        log_probs, derivative = sequence_log_probabilities(params, sequence, True)
        actions = sequence["actions"]
        total_loss -= log_probs[np.arange(len(actions)), actions].sum()
        total_trials += len(actions)
        gradient += derivative
    if total_trials == 0:
        raise ValueError("No trials available for fitting.")
    return total_loss / total_trials, gradient / total_trials


def fit_mouse_sessions(sequences, bounds=DEFAULT_BOUNDS, n_starts=10, seed=0):
    """Fit the sticky Q-learning model to a mouse's sessions by multi-start L-BFGS-B.

    Starting points are drawn uniformly inside ``bounds`` (seeded).

    Args:
        sequences (list[dict]): Sessions as built by ``prepare_cohort_sequences``; each dict needs
            ``actions``, ``rewards`` and ``towers``.
        bounds (list[tuple[float, float]]): Three ``(low, high)`` pairs in ``PARAMETER_NAMES``
            order. Default ``DEFAULT_BOUNDS``.
        n_starts (int): Number of random starts. Default 10.
        seed (int): Seed of the random generator that draws the starts. Default 0.

    Returns:
        tuple[dict | None, list[dict]]: ``(best, starts)``. ``starts`` has one dict per start with
            keys ``start`` (int), ``success`` (bool), ``loss`` (float, or None if not finite),
            ``parameters`` (list of three floats) and ``message`` (str). ``best`` is the successful
            start with the lowest loss, or None if no start succeeded.

    Raises:
        ValueError: If ``n_starts`` < 1.
    """
    if n_starts < 1:
        raise ValueError("n_starts must be positive.")
    rng = np.random.default_rng(seed)
    starts = []
    for start in range(n_starts):
        initial = [rng.uniform(low, high) for low, high in bounds]
        result = minimize(
            multi_session_loss,
            initial,
            args=(sequences,),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
        )
        starts.append(
            dict(
                start=start,
                success=bool(result.success),
                loss=float(result.fun) if np.isfinite(result.fun) else None,
                parameters=result.x.tolist(),
                message=str(result.message),
            )
        )
    candidates = [fit for fit in starts if fit["success"] and fit["loss"] is not None]
    if not candidates:
        return None, starts
    return min(candidates, key=lambda fit: fit["loss"]), starts


# ---------------------------------------------------------------------------
# Cross-validation: model selection and run manifest
# ---------------------------------------------------------------------------


class _ModelSpec(NamedTuple):
    """What the cross-validation needs from one model family."""

    parameter_names: list[str]
    fit_function: Callable
    prediction_function: Callable
    # Source files of the model code, recorded as provenance (not hashed), and extra
    # identity entries such as the model's MODEL_VERSION.
    manifest_sources: dict[str, Path]
    manifest_fields: dict


def _model_spec(model: str, bounds: list) -> _ModelSpec:
    """Resolve *model* to its functions; the other families are imported only when used.

    Looks up ``fit_mouse_sessions`` at call time so it stays replaceable in tests.
    """
    if model == "sticky_rl":
        return _ModelSpec(PARAMETER_NAMES, fit_mouse_sessions, sequence_log_probabilities, {}, {})
    if model == "best_rl":
        from src.agents import best_rl

        if len(bounds) != 6:
            raise ValueError("Best RL requires six bounds; pass best_rl.DEFAULT_BOUNDS.")
        return _ModelSpec(
            best_rl.PARAMETER_NAMES,
            best_rl.fit_mouse_sessions,
            best_rl.sequence_log_probabilities,
            {"model_implementation": Path(best_rl.__file__)},
            {
                "update_order": "choice_then_chosen_update_then_all_action_forgetting",
                "model_versions": {"best_rl": best_rl.MODEL_VERSION},
            },
        )
    if model == "released_rl":
        from src.agents import best_rl, released_rl

        if len(bounds) != 5:
            raise ValueError("Released CogMod requires five bounds.")
        return _ModelSpec(
            released_rl.PARAMETER_NAMES,
            released_rl.fit_mouse_sessions,
            released_rl.sequence_log_probabilities,
            {
                "model_implementation": Path(released_rl.__file__),
                "value_update_implementation": Path(__file__).with_name("best_rl.py"),
            },
            {
                "model_versions": {
                    "released_rl": released_rl.MODEL_VERSION,
                    "best_rl": best_rl.MODEL_VERSION,
                }
            },
        )
    raise ValueError("Unknown model.")


def _build_manifest(
    spec: _ModelSpec,
    sequences: list[dict],
    *,
    problem_numbers: list[int],
    bounds: list,
    n_starts: int,
    seed: int,
    model: str,
    source_metadata,
) -> tuple[dict, dict]:
    """The identity a saved fold depends on, and the source text it was produced by.

    Only the identity (algorithm version and configuration) names the output directory;
    the source text is provenance.
    """
    identity = dict(
        algorithm_version=ALGORITHM_VERSION,
        problems=problem_numbers,
        bounds=bounds,
        n_starts=n_starts,
        seed=seed,
        model=model,
        state_reset="each_extracted_session",
        reward="delivered_reward_pulses",
        data_sha256=fingerprint_sequences(sequences),
        source=source_metadata,
        at_bound_atol=_AT_BOUND_ATOL,
        versions=library_versions("numpy", "scipy"),
        **spec.manifest_fields,
    )
    provenance = dict(
        implementation=Path(__file__).read_text(),
        **{key: path.read_text() for key, path in spec.manifest_sources.items()},
    )
    return identity, provenance


def _mouse_eligibility(sequences: list[dict], problem_numbers: list[int]) -> pd.DataFrame:
    """One row per mouse: eligible only if it has data in every requested problem."""
    rows = []
    for mouse in sorted({s["mouse"] for s in sequences}):
        found = {s["problem"] for s in sequences if s["mouse"] == mouse}
        missing = sorted(set(problem_numbers) - found)
        rows.append(
            dict(mouse=mouse, eligible=not missing, missing_problems=",".join(map(str, missing)))
        )
    return pd.DataFrame(rows, columns=["mouse", "eligible", "missing_problems"])


# ---------------------------------------------------------------------------
# Cross-validation: one fold
# ---------------------------------------------------------------------------


def _fold_key(mouse: str, held_out: int) -> str:
    return hashlib.sha256(f"{mouse}:{held_out}".encode()).hexdigest()[:12]


def _fold_seed(seed: int, fold_key: str) -> int:
    """Independent optimiser seed per fold, so folds are reproducible in any order."""
    return int(np.random.SeedSequence([seed, int(fold_key[:8], 16)]).generate_state(1)[0])


def _fold_row_skeleton(
    mouse: str,
    *,
    held_out: int,
    problem_numbers: list[int],
    training: list[dict],
    validation: list[dict],
    fit,
    starts: list[dict],
    parameter_names: list[str],
) -> dict:
    """Fold summary before scoring: trial counts, convergence, and empty metric fields."""
    row = dict(
        mouse=mouse,
        held_out_problem=held_out,
        training_problems=",".join(str(p) for p in problem_numbers if p != held_out),
        n_train_trials=sum(len(s["actions"]) for s in training),
        n_test_trials=sum(len(s["actions"]) for s in validation),
        n_train_sessions=len(training),
        n_test_sessions=len(validation),
        success=fit is not None,
        converged_starts=sum(s["success"] for s in starts),
    )
    row.update(
        train_cross_entropy=None,
        test_cross_entropy=None,
        uniform_cross_entropy=float(np.log(3)),
        improvement_over_uniform=None,
    )
    for name in parameter_names:
        row[name] = None
        row[f"at_bound_{name}"] = None
    return row


def _score_held_out(
    prediction_function: Callable, params, validation: list[dict], *, mouse: str, held_out: int
) -> tuple[float, list[dict], pd.DataFrame]:
    """Summed held-out negative log likelihood, per-session summaries, per-trial predictions."""
    nll = 0.0
    fold_sessions, predictions = [], []
    for s in validation:
        log_probs = prediction_function(params, s)
        probabilities = np.exp(log_probs)
        actions = s["actions"]
        losses = -log_probs[np.arange(len(actions)), actions]
        nll += losses.sum()
        fold_sessions.append(
            dict(
                mouse=mouse,
                held_out_problem=held_out,
                session=s["session"],
                n_trials=len(actions),
                cross_entropy=float(losses.mean()),
            )
        )
        # Known issue: zip truncates to the shorter of actions and rewards, so a hand-built
        # session with fewer rewards than actions loses its trailing trials from the
        # predictions (prepare_cohort_sequences guarantees equal lengths).
        for t, (action, reward, loss) in enumerate(
            zip(actions, s["rewards"], losses, strict=False)
        ):
            predictions.append(
                dict(
                    mouse=mouse,
                    held_out_problem=held_out,
                    session=s["session"],
                    trial=t + 1,
                    chosen_tower=s["towers"][action],
                    reward=reward,
                    negative_log_probability=loss,
                    **{f"p_{tower}": probabilities[t, a] for a, tower in enumerate(s["towers"])},
                )
            )
    return nll, fold_sessions, pd.DataFrame(predictions)


def _record_fit_metrics(
    row: dict, fit: dict, nll: float, *, parameter_names: list[str], bounds: list
) -> None:
    """Fill the fold row's training/test loss, parameters and at-bound flags in place."""
    params = fit["parameters"]
    row["train_cross_entropy"] = fit["loss"]
    row["test_cross_entropy"] = float(nll / row["n_test_trials"])
    row["improvement_over_uniform"] = row["uniform_cross_entropy"] - row["test_cross_entropy"]
    # Known issue: zip truncates to the shortest of names, parameters and bounds, so a
    # bounds list shorter than the parameters drops them silently instead of raising. The
    # best_rl and released_rl models check their bounds length up front; sticky_rl never
    # validates it.
    for name, value, limits in zip(parameter_names, params, bounds, strict=False):
        row[name] = float(value)
        row[f"at_bound_{name}"] = at_bound(value, limits, atol=_AT_BOUND_ATOL)


def _save_fold(fold_file: Path, saved: dict) -> None:
    """Write via a temporary file so an interrupted run never leaves a partial fold."""
    write_json_atomically(fold_file, saved, indent=2)


class _FoldSettings(NamedTuple):
    """Settings shared by every fold of one cross-validation run."""

    spec: _ModelSpec
    problem_numbers: list[int]
    bounds: list
    n_starts: int
    seed: int


class _FoldResult(NamedTuple):
    row: dict
    sessions: list[dict]
    loaded: bool  # True when read back from a saved fold instead of fitted
    fit_succeeded: bool


def _fit_and_score_fold(
    mouse: str,
    mouse_sequences: list[dict],
    held_out: int,
    *,
    settings: _FoldSettings,
    output: Path,
    fold_key: str,
) -> tuple[dict, list[dict], dict, bool]:
    """Fit on the other problems and score the held-out one.

    Returns the fold row, per-session rows, the dict to save, and whether a fit was found.
    """
    spec = settings.spec
    training = [s for s in mouse_sequences if s["problem"] != held_out]
    validation = [s for s in mouse_sequences if s["problem"] == held_out]
    fold_seed = _fold_seed(settings.seed, fold_key)
    fit, starts = spec.fit_function(training, settings.bounds, settings.n_starts, fold_seed)
    row = _fold_row_skeleton(
        mouse,
        held_out=held_out,
        problem_numbers=settings.problem_numbers,
        training=training,
        validation=validation,
        fit=fit,
        starts=starts,
        parameter_names=spec.parameter_names,
    )
    fold_sessions = []
    if fit is not None:
        nll, fold_sessions, predictions = _score_held_out(
            spec.prediction_function, fit["parameters"], validation, mouse=mouse, held_out=held_out
        )
        predictions.to_csv(output / f"predictions_{fold_key}.csv", index=False)
        _record_fit_metrics(
            row, fit, nll, parameter_names=spec.parameter_names, bounds=settings.bounds
        )
    saved = dict(fold=row, sessions=fold_sessions, starts=starts)
    return row, fold_sessions, saved, bool(fit)


def _run_fold(
    mouse: str,
    mouse_sequences: list[dict],
    held_out: int,
    *,
    settings: _FoldSettings,
    output: Path,
) -> _FoldResult:
    """Load the saved fold if present, otherwise fit, score and save it."""
    fold_key = _fold_key(mouse, held_out)
    fold_file = output / f"fold_{fold_key}.json"
    if fold_file.exists():
        saved = json.loads(fold_file.read_text())
        # Known issue: fit_succeeded is True here even when the saved fold's fit failed
        # (saved["fold"]["success"] is False); callers do not read it for loaded folds.
        return _FoldResult(saved["fold"], saved["sessions"], loaded=True, fit_succeeded=True)
    row, fold_sessions, saved, fit_succeeded = _fit_and_score_fold(
        mouse, mouse_sequences, held_out, settings=settings, output=output, fold_key=fold_key
    )
    _save_fold(fold_file, saved)
    return _FoldResult(row, fold_sessions, loaded=False, fit_succeeded=fit_succeeded)


def run_problem_cross_validation(
    sequences,
    output_root,
    problem_numbers=range(5, 12),
    bounds=DEFAULT_BOUNDS,
    n_starts=10,
    seed=42,
    source_metadata=None,
    model="sticky_rl",
):
    """One whole-problem fold per requested problem (seven by default) for each eligible mouse, with
    resumable saved fits.

    Only mice with usable trials in every requested problem are eligible. Training is trial-weighted
    within each mouse. Validation updates Q from past held-out observations, but never changes the
    fitted parameters. Each fold is saved under ``<output_root>/<identity hash>/``; rerunning with
    the same inputs reloads finished folds instead of refitting them. Progress is printed for each
    newly fitted fold.

    Args:
        sequences (list[dict]): Sessions as built by ``prepare_cohort_sequences``. Sessions from
            problems outside *problem_numbers* are ignored.
        output_root (str | Path): Directory that holds one subdirectory per run identity (created if
            missing).
        problem_numbers (Iterable[int]): Problems to cross-validate over; each is held out once.
            Duplicates are dropped and the list is sorted. Default ``range(5, 12)``.
        bounds (list[tuple]): ``(low, high)`` pairs in the model's parameter order: three for
            ``"sticky_rl"``, six for ``"best_rl"``, five for ``"released_rl"``. Default
            ``DEFAULT_BOUNDS`` (the sticky model's).
        n_starts (int): Random optimiser starts per fold. Default 10.
        seed (int): Base seed; each fold derives its own from it and the fold name. Default 42.
        source_metadata (object | None): JSON-compatible description of where the data came from,
            recorded in the run identity. Default None.
        model (str): ``"sticky_rl"``, ``"best_rl"`` or ``"released_rl"``. Default ``"sticky_rl"``.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]: ``(results, sessions, eligibility,
            output)``. ``results`` has one row per mouse and held-out problem with columns
            ``mouse``, ``held_out_problem``, ``training_problems``, ``n_train_trials``,
            ``n_test_trials``, ``n_train_sessions``, ``n_test_sessions``, ``success``,
            ``converged_starts``, ``train_cross_entropy``, ``test_cross_entropy``,
            ``uniform_cross_entropy``, ``improvement_over_uniform``, and per parameter its fitted
            value and an ``at_bound_<name>`` flag (parameter columns are None when no start
            converged). ``sessions`` has one row per held-out session with columns ``mouse``,
            ``held_out_problem``, ``session``, ``n_trials`` and ``cross_entropy``. ``eligibility``
            has one row per mouse with columns ``mouse``, ``eligible`` (bool) and
            ``missing_problems`` (comma-joined). ``output`` is the run directory, which also holds
            per-fold JSON, per-trial prediction CSVs and the three tables as CSV.

    Raises:
        ValueError: If fewer than two distinct problems are requested, ``model`` is unknown, or the
            number of ``bounds`` does not match the model.
    """
    spec = _model_spec(model, bounds)
    problem_numbers = sorted(set(problem_numbers))
    if len(problem_numbers) < 2:
        raise ValueError("Cross-validation requires at least two problems.")
    sequences = [s for s in sequences if s["problem"] in problem_numbers]
    identity, provenance = _build_manifest(
        spec,
        sequences,
        problem_numbers=problem_numbers,
        bounds=bounds,
        n_starts=n_starts,
        seed=seed,
        model=model,
        source_metadata=source_metadata,
    )
    output = manifest_dir(output_root, identity, provenance)
    eligibility = _mouse_eligibility(sequences, problem_numbers)
    eligibility.to_csv(output / "mouse_eligibility.csv", index=False)
    settings = _FoldSettings(spec, problem_numbers, bounds, n_starts, seed)
    rows, session_rows = [], []
    for mouse in eligibility.loc[eligibility.eligible, "mouse"]:
        mouse_sequences = [s for s in sequences if s["mouse"] == mouse]
        for held_out in problem_numbers:
            fold = _run_fold(mouse, mouse_sequences, held_out, settings=settings, output=output)
            rows.append(fold.row)
            session_rows.extend(fold.sessions)
            if fold.loaded:
                continue
            pd.DataFrame(rows).to_csv(output / "fold_results.csv", index=False)
            outcome = "fit saved" if fold.fit_succeeded else "all starts failed"
            print(f"{mouse}: held out problem {held_out}; {outcome}", flush=True)
    results = pd.DataFrame(rows)
    sessions = pd.DataFrame(session_rows)
    results.to_csv(output / "fold_results.csv", index=False)
    sessions.to_csv(output / "held_out_sessions.csv", index=False)
    return results, sessions, eligibility, output
