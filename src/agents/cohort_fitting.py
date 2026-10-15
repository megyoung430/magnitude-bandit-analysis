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

from src.behavior_analysis.problem_utils import session_sort_key

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

    Reject a session with any invalid/missing choice or delivered reward rather
    than silently joining trials across missing learning history. All choices,
    including zero-reward choices, are retained in otherwise valid sessions.
    """
    sequences, audit = [], []
    for problem in problem_numbers:
        for mouse, sessions in sorted(problems.get(problem, {}).items()):
            for session in sorted(sessions, key=session_sort_key):
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
    """Pre-choice log probabilities, with Q updated only after observing reward."""
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
    """Mean trial-weighted cross entropy and analytic gradient; reset per session."""
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
    # Manifest entries that make the saved folds depend on the model code.
    manifest_sources: dict[str, Path]
    manifest_fields: dict[str, str]


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
            {"update_order": "choice_then_chosen_update_then_all_action_forgetting"},
        )
    if model == "released_rl":
        from src.agents import released_rl

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
            {},
        )
    raise ValueError("Unknown model.")


def _data_fingerprint(sequences: list[dict]) -> str:
    """SHA-256 over every sequence's identity, towers, actions and rewards."""
    fingerprint = hashlib.sha256()
    for s in sequences:
        identity = [s["mouse"], s["problem"], s["session"], s["towers"]]
        fingerprint.update(json.dumps(identity).encode())
        fingerprint.update(np.asarray(s["actions"], dtype=np.int64).tobytes())
        fingerprint.update(np.asarray(s["rewards"], dtype=np.float64).tobytes())
    return fingerprint.hexdigest()


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
) -> dict:
    """Everything a saved fold depends on, including the source text of the model code."""
    manifest = dict(
        problems=problem_numbers,
        bounds=bounds,
        n_starts=n_starts,
        seed=seed,
        model=model,
        state_reset="each_extracted_session",
        reward="delivered_reward_pulses",
        data_sha256=_data_fingerprint(sequences),
        source=source_metadata,
        implementation=Path(__file__).read_text(),
    )
    manifest.update({key: path.read_text() for key, path in spec.manifest_sources.items()})
    manifest.update(spec.manifest_fields)
    return manifest


def _create_output_dir(output_root, manifest: dict) -> Path:
    """Create the directory named by the manifest hash and write the manifest into it."""
    encoded = json.dumps(manifest, sort_keys=True, indent=2)
    output = Path(output_root) / hashlib.sha256(encoded.encode()).hexdigest()[:12]
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(encoded)
    return output


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


def _at_bound(value: float, limits) -> bool:
    return bool(
        any(np.isclose(value, b, atol=_AT_BOUND_ATOL, rtol=0) for b in limits if b is not None)
    )


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
        row[f"at_bound_{name}"] = _at_bound(value, limits)


def _save_fold(fold_file: Path, saved: dict) -> None:
    """Write via a temporary file so an interrupted run never leaves a partial fold."""
    tmp = fold_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(saved, indent=2, allow_nan=False))
    tmp.replace(fold_file)


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
    """Seven whole-problem folds per eligible mouse, with resumable saved fits.

    Only mice with usable trials in every requested problem are eligible.
    Training is trial-weighted within each mouse. Validation updates Q from
    past held-out observations, but never changes the fitted parameters.
    """
    spec = _model_spec(model, bounds)
    problem_numbers = sorted(set(problem_numbers))
    if len(problem_numbers) < 2:
        raise ValueError("Cross-validation requires at least two problems.")
    sequences = [s for s in sequences if s["problem"] in problem_numbers]
    manifest = _build_manifest(
        spec,
        sequences,
        problem_numbers=problem_numbers,
        bounds=bounds,
        n_starts=n_starts,
        seed=seed,
        model=model,
        source_metadata=source_metadata,
    )
    output = _create_output_dir(output_root, manifest)
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
