"""Subject-disjoint fitting with the released AdamW and shifted-loss convention.

Paper-style procedure, with explicit mouse/session adaptations. Optional JAX and
Optax imports are delayed so the existing SciPy workflows need no new packages.
"""

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from src.agents.run_store import fingerprint_sequences, manifest_dir, write_atomically

# Names the sweep directory together with the settings, data and library versions. It stands
# for the fitting code here and in the neural_bandit, best_rl and released_rl modules.
# Bump when a change alters results; comments and refactors don't need a bump.
ALGORITHM_VERSION = 1
SOURCE_COMMIT = "baf38da49fd1c1537affbc3745e961bbf0ae547d"
SOURCE_URL = "https://github.com/google-deepmind/hybrid_rnns_reward_learning"
MODEL_NAMES = ("sticky_rl", "best_rl", "released_rl")
PARAMETER_NAMES = {
    "sticky_rl": ["learning_rate", "beta", "alpha"],
    "best_rl": ["learning_rate", "beta", "kappa", "bias", "forgetting", "q_init"],
    "released_rl": ["learning_rate", "beta", "kappa", "forgetting", "q_init"],
}
FULL_GRID = [
    dict(optimizer_learning_rate=lr, weight_decay=l2, batch_size=bs)
    for lr, l2, bs in itertools.product([1e-3, 1e-4, 1e-5], [1e-3, 1e-4, 1e-5], [32, 64, 128])
]


def dependencies() -> tuple[Any, Any, Any]:
    try:
        import jax
        import jax.numpy as jnp
        import optax
    except ImportError as exc:
        raise ImportError(
            "Paper-style fitting needs jax and optax in the notebook kernel. "
            "Install requirements-paper-rl.txt in that environment."
        ) from exc
    jax.config.update("jax_enable_x64", True)
    return jax, jnp, optax


def split_by_mouse(
    sequences: list[dict],
    seed: int = 42,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> tuple[dict[str, list[dict]], dict[str, list[str]]]:
    """Seeded mouse split: no session of a mouse can cross partitions.

    Floor training and validation counts, with at least one mouse per partition.
    Small cohorts cannot exactly match 80/10/10; returned identities are auditable.
    """
    if len(fractions) != 3 or min(fractions) <= 0 or not np.isclose(sum(fractions), 1):
        raise ValueError("Need three positive fractions summing to one.")
    mice = sorted({s["mouse"] for s in sequences})
    if len(mice) < 3:
        raise ValueError("Participant-disjoint training/validation/test needs at least three mice.")
    shuffled = np.random.default_rng(seed).permutation(mice).tolist()
    n_train = min(max(1, int(len(mice) * fractions[0])), len(mice) - 2)
    n_val = min(max(1, int(len(mice) * fractions[1])), len(mice) - n_train - 1)
    ids = dict(
        train=shuffled[:n_train],
        validation=shuffled[n_train : n_train + n_val],
        test=shuffled[n_train + n_val :],
    )
    return {part: [s for s in sequences if s["mouse"] in group] for part, group in ids.items()}, ids


def split_by_problem(
    sequences: list[dict], test_problem: int, validation_problem: int | None = None
) -> tuple[dict[str, list[dict]], dict[str, list[int]]]:
    """Pooled, whole-problem split; numbers are mouse-relative problem indices.

    Hold the same index out across mice. By default validation is the next
    available index cyclically, chosen without inspecting behavioral outcomes.
    Every validation/test mouse must be represented in training.
    """
    problems = sorted({int(s["problem"]) for s in sequences})
    if len(problems) < 3 or test_problem not in problems:
        raise ValueError("Problem folds need at least three problems and a valid test problem.")
    if validation_problem is None:
        validation_problem = problems[(problems.index(test_problem) + 1) % len(problems)]
    if validation_problem not in problems or validation_problem == test_problem:
        raise ValueError("Validation must be an available problem distinct from test.")
    ids = dict(
        train=[p for p in problems if p not in (test_problem, validation_problem)],
        validation=[int(validation_problem)],
        test=[int(test_problem)],
    )
    parts = {
        part: [s for s in sequences if int(s["problem"]) in group] for part, group in ids.items()
    }
    train_mice = {s["mouse"] for s in parts["train"]}
    unseen = {s["mouse"] for part in ("validation", "test") for s in parts[part]} - train_mice
    if unseen:
        raise ValueError(f"Problem folds require training data for held-out mice: {sorted(unseen)}")
    return parts, ids


def pack_sequences(
    sequences: list[dict], max_trials: int | None = None, num_arms: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad complete sessions; masked trials contribute no loss and no state update."""
    if not sequences:
        raise ValueError("Empty data partition.")
    length = max(len(s["actions"]) for s in sequences)
    if max_trials is not None and length > max_trials:
        raise ValueError("max_trials would truncate a session.")
    length = max_trials or length
    a = np.zeros((len(sequences), length), dtype=np.int32)
    r = np.zeros_like(a, dtype=float)
    m = np.zeros_like(a, dtype=bool)
    for i, s in enumerate(sequences):
        actions, rewards = np.asarray(s["actions"]), np.asarray(s["rewards"], dtype=float)
        if (
            len(s["towers"]) != num_arms
            or actions.ndim != 1
            or not len(actions)
            or rewards.shape != actions.shape
            or not np.issubdtype(actions.dtype, np.integer)
            or np.any(actions < 0)
            or np.any(actions >= num_arms)
            or not np.all(np.isfinite(rewards))
        ):
            raise ValueError("Invalid sequence, reward history or arm count.")
        n = len(actions)
        a[i, :n], r[i, :n], m[i, :n] = actions, rewards, True
    return a, r, m


def transform_parameters(raw: Any, model: str) -> Any:
    """Transform trainable coordinates; regularization is on these raw coordinates.

    Best/published RL: sigmoid rates, ReLU beta, tanh perseverance, free q_init.
    Paper-equation Best RL additionally has tanh bias, absent in released CogMod.
    Sticky RL: sigmoid maps to existing fitting bounds for rate/beta/stickiness.
    """
    jax, jnp, _ = dependencies()
    if model == "sticky_rl":
        return jnp.stack(
            [
                0.001 + 0.999 * jax.nn.sigmoid(raw[0]),
                0.01 + 9.99 * jax.nn.sigmoid(raw[1]),
                3 * jax.nn.sigmoid(raw[2]),
            ]
        )
    if model == "best_rl":
        return jnp.stack(
            [
                jax.nn.sigmoid(raw[0]),
                jax.nn.relu(raw[1]),
                jnp.tanh(raw[2]),
                jnp.tanh(raw[3]),
                jax.nn.sigmoid(raw[4]),
                raw[5],
            ]
        )
    if model == "released_rl":
        return jnp.stack(
            [
                jax.nn.sigmoid(raw[0]),
                jax.nn.relu(raw[1]),
                jnp.tanh(raw[2]),
                jax.nn.sigmoid(raw[3]),
                raw[4],
            ]
        )
    raise ValueError("Unknown model.")


class _RLParameters(NamedTuple):
    """Transformed RL parameters; models without a term fix it at a constant."""

    learning_rate: Any
    beta: Any
    stickiness: Any
    bias: Any
    forgetting: Any
    q_init: Any


def _split_rl_parameters(params, model):
    """Name the transformed parameter vector; absent terms are fixed at zero."""
    if model == "sticky_rl":
        lr, beta, stickiness = params
        return _RLParameters(lr, beta, stickiness, 0.0, 0.0, 0.0)
    if model == "released_rl":
        lr, beta, stickiness, forgetting, q_init = params
        return _RLParameters(lr, beta, stickiness, 0.0, forgetting, q_init)
    if model == "best_rl":
        lr, beta, stickiness, bias, forgetting, q_init = params
        return _RLParameters(lr, beta, stickiness, bias, forgetting, q_init)
    raise ValueError("Unknown model.")


def _make_trial_step(jax, jnp, params, batch, num_arms, released_loss, skip_first):
    """Scan body: score one trial for every session in the batch, then update values."""
    index = jnp.arange(batch)

    def trial(carry, inputs):
        q, previous, nll, correct = carry
        action, reward, valid, t = inputs
        choice_bonus = jax.nn.one_hot(previous, num_arms) * params.stickiness
        logits = params.beta * (q + choice_bonus)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        chosen = log_probs[index, action]
        adjusted = jnp.logaddexp(jnp.log1p(-1e-5) + chosen, jnp.log(5e-4))
        scored = valid & ((t > 0) | ~jnp.asarray(skip_first))
        nll += jnp.where(scored, -jnp.where(released_loss, adjusted, chosen), 0.0)
        correct += jnp.where(scored, jnp.argmax(logits, axis=-1) == action, 0.0)
        updated = q.at[index, action].add(
            params.learning_rate * (reward - q[index, action]) + params.bias
        )
        updated = (1 - params.forgetting) * updated + params.forgetting * params.q_init
        q = jnp.where(valid[:, None], updated, q)
        previous = jnp.where(valid, action, previous)
        return (q, previous, nll, correct), None

    return trial


def _make_statistics(jax, jnp, model, num_arms):
    """Jitted per-session (negative log-likelihood, correct count) for an RL model."""

    @jax.jit
    def statistics(raw, actions, rewards, mask, released_loss=False, skip_first=False):
        params = _split_rl_parameters(transform_parameters(raw, model), model)
        batch = actions.shape[0]
        initial = (
            jnp.full((batch, num_arms), params.q_init),
            jnp.full(batch, -1, dtype=actions.dtype),
            jnp.zeros(batch),
            jnp.zeros(batch),
        )
        trial = _make_trial_step(jax, jnp, params, batch, num_arms, released_loss, skip_first)
        inputs = (actions.T, rewards.T, mask.T, jnp.arange(actions.shape[1]))
        final, _ = jax.lax.scan(trial, initial, inputs)
        return final[2], final[3]

    return statistics


def _make_objective(statistics):
    def objective(raw, actions, rewards, mask, l2_weight):
        # Released fit_hyb_rnn.loss_fn: score t>=1, affine probability safeguard,
        # sum across trials and average across sampled sessions. Decay is AdamW.
        # Known issue: ``l2_weight`` is unused here; the trainer always passes 0.0
        # and the setting's decay is applied by the optimizer instead.
        nll, _ = statistics(raw, actions, rewards, mask, True, True)
        return nll.sum() / actions.shape[0]

    return objective


def _weight_decay(setting):
    """AdamW weight decay of an optimizer setting; ``l2_weight`` is the legacy key.

    Known issues, left as they are because this module preserves fitting behaviour
    exactly (changing them is a separate piece of work): ``weight_decay`` silently
    wins when both keys are given, and a setting with neither silently trains
    without decay instead of failing.
    """
    return setting.get("weight_decay", setting.get("l2_weight", 0.0))


def _make_trainer(jax, optax, value_grad):
    def trainer(setting):
        optimizer = optax.adamw(
            setting["optimizer_learning_rate"], weight_decay=_weight_decay(setting)
        )
        batch_size = setting["batch_size"]
        l2 = 0.0  # Decoupled AdamW decay, not a term in the likelihood (see ``objective``).

        def step(carry, unused):
            raw, state, key, actions, rewards, mask = carry
            key, sampling_key = jax.random.split(key)
            ids = jax.random.choice(sampling_key, actions.shape[0], (batch_size,), replace=False)
            loss, grad = value_grad(raw, actions[ids], rewards[ids], mask[ids], l2)
            updates, state = optimizer.update(grad, state, raw)
            raw = optax.apply_updates(raw, updates)
            return (raw, state, key, actions, rewards, mask), loss

        def chunk(raw, state, key, actions, rewards, mask, steps):
            carry, losses = jax.lax.scan(
                step, (raw, state, key, actions, rewards, mask), xs=None, length=steps
            )
            return carry[:3], losses[-1]

        return optimizer, jax.jit(chunk, static_argnums=(6,))

    return trainer


def make_backend(model: str, num_arms: int = 3, hidden_units: int = 16) -> tuple[Any, Any, Any]:
    if model not in MODEL_NAMES:
        from src.agents.neural_bandit import make_backend as neural_backend

        return neural_backend(model, hidden_units=hidden_units, num_arms=num_arms)
    jax, jnp, optax = dependencies()
    statistics = _make_statistics(jax, jnp, model, num_arms)
    objective = _make_objective(statistics)
    trainer = _make_trainer(jax, optax, jax.value_and_grad(objective))
    return statistics, objective, trainer


def score_sequences(
    raw: Any,
    sequences: list[dict],
    model: str,
    statistics: Any = None,
    evaluation_batch_size: int = 32,
) -> tuple[float, list[dict]]:
    if evaluation_batch_size < 1:
        raise ValueError("evaluation_batch_size must be positive.")
    statistics = statistics or make_backend(model)[0]
    packed = pack_sequences(sequences)
    rows = []
    for offset in range(0, len(sequences), evaluation_batch_size):
        batch = tuple(x[offset : offset + evaluation_batch_size] for x in packed)
        nll, correct = statistics(raw, *batch, False, True)
        release_nll, _ = statistics(raw, *batch, True, True)
        for s, loss, count, release_loss in zip(
            sequences[offset : offset + evaluation_batch_size],
            np.asarray(nll),
            np.asarray(correct),
            np.asarray(release_nll),
            strict=True,
        ):
            n = len(s["actions"]) - 1
            if n < 1:
                raise ValueError("Released scoring needs at least two trials per session.")
            rows.append(
                dict(
                    mouse=s["mouse"],
                    problem=int(s["problem"]),
                    session=s["session"],
                    n_trials=n,
                    n_observed_trials=n + 1,
                    nll=float(loss),
                    cross_entropy=float(loss / n),
                    release_nll=float(release_loss),
                    release_score_per_trial=float(release_loss / n),
                    geometric_mean_choice_probability=float(np.exp(-loss / n)),
                    argmax_accuracy=float(count / n),
                )
            )
    total_trials = sum(r["n_trials"] for r in rows)
    return sum(r["nll"] for r in rows) / total_trials, rows


def _save_checkpoint(path, raw, state, key, best_raw, metadata):
    jax, _, _ = dependencies()
    leaves, _ = jax.tree_util.tree_flatten(state)
    raw_leaves, _ = jax.tree_util.tree_flatten(raw)
    best_leaves, _ = jax.tree_util.tree_flatten(best_raw)
    write_atomically(
        path,
        lambda file: np.savez(
            file,
            key=np.asarray(key),
            metadata=np.array(json.dumps(metadata, allow_nan=False)),
            **{f"raw_{i}": np.asarray(v) for i, v in enumerate(raw_leaves)},
            **{f"best_{i}": np.asarray(v) for i, v in enumerate(best_leaves)},
            **{f"opt_{i}": np.asarray(v) for i, v in enumerate(leaves)},
        ),
    )


_MANIFEST_DESCRIPTIONS = dict(
    upstream_repository=SOURCE_URL,
    upstream_commit=SOURCE_COMMIT,
    objective="mean_session_sum_t>=1(-log((1-1e-5)*p+5e-4))",
    optimizer="optax.adamw; decoupled weight decay",
    sampling=("without replacement per minibatch; cap batch size at number of training sessions"),
    selection="lowest full-validation affine-adjusted score per trial, including step zero",
    initialization=(
        "CogMod alpha_raw=1, beta_raw=1, perseverance_raw=0, forgetting_raw=0, q_init~N(1,1); "
        "paper bias_raw=0; sticky model retains N(0,.5)"
    ),
    precision="float64",
    reset="each full extracted session",
    reward="unaltered delivered pulses",
)
_NEURAL_INITIALIZATION = (
    "Haiku Linear truncated normal / sqrt(fan_in); zero biases/hidden; "
    "q_init,c_init~N(1,1), forgetting_raw~N(0,1)"
)


@dataclass(frozen=True, kw_only=True)
class _FitConfig:
    """What every run in one sweep shares: training and validation data and the schedule.

    The test partition is deliberately absent, so nothing that trains or selects can
    reach it.
    """

    validation_sequences: list
    train_arrays: tuple
    fit_seed: int
    max_steps: int
    validation_every: int

    @property
    def n_train_sessions(self):
        return self.train_arrays[0].shape[0]


class _Trainer(NamedTuple):
    """The jitted statistics, the optimizer and the jitted training chunk of one setting."""

    stats: Any
    optimizer: Any
    chunk: Any


def _validate_models(models, supported):
    """Require a nonempty list of distinct, supported model names."""
    if not models or len(set(models)) != len(models) or any(m not in supported for m in models):
        raise ValueError("Need unique supported model names.")


def _validate_sweep_sizes(grid, *, n_seeds, max_steps, validation_every, max_new_runs):
    """Require a nonempty grid and positive counts; ``max_new_runs=None`` means no budget."""
    if (
        not grid
        or n_seeds < 1
        or max_steps < 1
        or validation_every < 1
        or (max_new_runs is not None and max_new_runs < 0)
    ):
        raise ValueError("Invalid sweep sizes.")


def _hidden_units(setting):
    return setting.get("hidden_units", 16)


def _validate_settings(grid):
    """Require positive learning rate, batch size and width, and nonnegative decay."""
    for setting in grid:
        if (
            setting["optimizer_learning_rate"] <= 0
            or _weight_decay(setting) < 0
            or setting["batch_size"] < 1
            or _hidden_units(setting) < 1
        ):
            raise ValueError("Invalid optimizer setting.")


def _validate_sequence_lengths(sequences):
    """Released scoring drops the first trial, so every session needs at least two."""
    if any(len(s["actions"]) < 2 for s in sequences):
        raise ValueError("Released scoring requires at least two observed trials per session.")


def _split_sequences(sequences, split_seed, test_problem=None, validation_problem=None):
    """Partition sessions by mouse, or by whole problem when ``test_problem`` is given.

    Returns ``(parts, identities, split_details)``; identities list the mice per partition
    either way, and ``split_details`` is what the manifest and results record.
    """
    if test_problem is None:
        if validation_problem is not None:
            raise ValueError("validation_problem requires test_problem.")
        parts, identities = split_by_mouse(sequences, split_seed)
        return parts, identities, dict(split_strategy="mouse", split_problems=None)
    parts, problem_ids = split_by_problem(sequences, test_problem, validation_problem)
    identities = {part: sorted({s["mouse"] for s in data}) for part, data in parts.items()}
    return (
        parts,
        identities,
        dict(split_strategy="pooled_problem", split_problems=problem_ids),
    )


def _read_source_snapshots(models):
    """Source text of the code that produced a sweep, recorded as provenance (not hashed)."""
    from src.agents import best_rl, neural_bandit, released_rl

    snapshots = dict(implementation=Path(__file__).read_text())
    if any(model in neural_bandit.NEURAL_MODELS for model in models):
        snapshots["neural_implementation"] = Path(neural_bandit.__file__).read_text()
    modules = [("best_rl", best_rl), ("released_rl", released_rl)]
    snapshots["rl_implementations"] = {m: Path(mod.__file__).read_text() for m, mod in modules}
    return snapshots


def _neural_identity(models):
    """Manifest fields that only matter when a neural model is among ``models``."""
    from src.agents import neural_bandit

    if any(model in neural_bandit.NEURAL_MODELS for model in models):
        return dict(neural_initialization=_NEURAL_INITIALIZATION)
    return {}


def _build_manifest(settings, split_details, identities, data_sha256, versions, model_fields):
    """Everything that determines a sweep's results; its hash names the output directory.

    Holds the algorithm version but no source text (see ``_read_source_snapshots``).
    """
    return dict(
        **settings,
        **split_details,
        split_mice=identities,
        data_sha256=data_sha256,
        **_MANIFEST_DESCRIPTIONS,
        algorithm_version=ALGORITHM_VERSION,
        versions=versions,
        **model_fields,
    )


def _initial_raw_parameters(model, init_key, hidden_units=16):
    """Draw the starting raw parameters of one run from its initialization key."""
    jax, jnp, _ = dependencies()
    if model == "sticky_rl":
        return 0.5 * jax.random.normal(init_key, (3,), dtype=jnp.float64)
    if model in ("best_rl", "released_rl"):
        q_init = 1 + jax.random.normal(init_key, (), dtype=jnp.float64)
        return jnp.array(
            [1.0, 1.0, 0.0, 0.0, 0.0, q_init]
            if model == "best_rl"
            else [1.0, 1.0, 0.0, 0.0, q_init]
        )
    from src.agents import neural_bandit

    return neural_bandit.initialize_parameters(init_key, model, hidden_units)


def _release_score_per_trial(rows):
    """Released-loss score per scored trial, pooled over the sessions in ``rows``."""
    return sum(r["release_nll"] for r in rows) / sum(r["n_trials"] for r in rows)


def _initial_metadata(validation_ce, *, validation_score):
    """Run history at step zero, before any optimization."""
    return dict(
        step=0,
        best_step=0,
        best_validation=validation_score,
        history=[
            dict(
                step=0,
                validation_cross_entropy=validation_ce,
                validation_release_score=validation_score,
            )
        ],
    )


def _advance_metadata(metadata, *, steps, validation_ce, validation_score, train_loss):
    """Metadata after ``steps`` more optimizer steps, and whether validation improved.

    Only a strictly lower validation score replaces the best iterate, so ties keep
    the earlier step.
    """
    step = metadata["step"] + steps
    entry = dict(
        step=step,
        validation_cross_entropy=validation_ce,
        validation_release_score=validation_score,
        last_batch_objective=train_loss,
    )
    updated = dict(metadata, step=step, history=[*metadata["history"], entry])
    improved = validation_score < metadata["best_validation"]
    if improved:
        updated.update(best_step=step, best_validation=validation_score)
    return updated, improved


def _load_checkpoint(checkpoint, raw, state):
    """Restore (raw, best_raw, sampling key, optimizer state, metadata) from a checkpoint.

    ``raw`` and ``state`` supply the pytree structures the saved leaves are poured into.
    """
    jax, jnp, _ = dependencies()

    def restore(prefix, template):
        leaves, tree = jax.tree_util.tree_flatten(template)
        return jax.tree_util.tree_unflatten(
            tree, [jnp.asarray(saved[f"{prefix}_{i}"]) for i in range(len(leaves))]
        )

    with np.load(checkpoint, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        return (
            restore("raw", raw),
            restore("best", raw),
            jnp.asarray(saved["key"]),
            restore("opt", state),
            metadata,
        )


def _score_validation(raw, model, stats, config):
    """Validation cross-entropy per trial and released-loss score per trial."""
    validation_ce, rows = score_sequences(raw, config.validation_sequences, model, stats)
    return validation_ce, _release_score_per_trial(rows)


def _fit_run(run_id, model, *, setting, repeat, trainer, config, checkpoint):
    """Train one (model, setting, seed) run, resuming from ``checkpoint`` when present.

    Returns the best-validation raw parameters and the run metadata.
    """
    jax, _, _ = dependencies()
    stats, optimizer, chunk = trainer
    # Matched initialization seeds across models: the stream depends only on ``repeat``.
    init_key, sample_key = jax.random.split(jax.random.PRNGKey(config.fit_seed + repeat))
    raw = _initial_raw_parameters(model, init_key, _hidden_units(setting))
    state = optimizer.init(raw)
    if checkpoint.exists():
        raw, best_raw, sample_key, state, metadata = _load_checkpoint(checkpoint, raw, state)
    else:
        validation_ce, validation_score = _score_validation(raw, model, stats, config)
        best_raw = raw
        metadata = _initial_metadata(validation_ce, validation_score=validation_score)
    while metadata["step"] < config.max_steps:
        steps = min(config.validation_every, config.max_steps - metadata["step"])
        (raw, state, sample_key), train_loss = chunk(
            raw, state, sample_key, *config.train_arrays, steps
        )
        validation_ce, validation_score = _score_validation(raw, model, stats, config)
        if not np.isfinite(validation_score) or not np.isfinite(float(train_loss)):
            raise FloatingPointError(
                f"{run_id}: nonfinite objective; last finite checkpoint retained."
            )
        metadata, improved = _advance_metadata(
            metadata,
            steps=steps,
            validation_ce=validation_ce,
            validation_score=validation_score,
            train_loss=float(train_loss),
        )
        if improved:
            best_raw = raw
        _save_checkpoint(checkpoint, raw, state, sample_key, best_raw, metadata)
        print(
            f"{run_id}: {metadata['step']:,}/{config.max_steps:,} steps; "
            f"validation release score {validation_score:.5f}",
            flush=True,
        )
    return best_raw, metadata


def _serialize_raw(raw):
    """JSON-ready (nested lists) form of raw parameters, which may be a dict of arrays."""
    if isinstance(raw, dict):
        return {k: np.asarray(v).tolist() for k, v in raw.items()}
    return np.asarray(raw).tolist()


def restore_raw(serialized: list | dict) -> Any:
    """Inverse of :func:`_serialize_raw`, as JAX arrays."""
    _, jnp, _ = dependencies()
    if isinstance(serialized, dict):
        return {k: jnp.asarray(v) for k, v in serialized.items()}
    return jnp.asarray(serialized)


def _summarize_parameters(best_raw, model):
    """Named, human-scale parameters of a fitted model."""
    if model in MODEL_NAMES:
        values = np.asarray(transform_parameters(best_raw, model)).tolist()
        return dict(zip(PARAMETER_NAMES[model], values, strict=True))
    from src.agents import neural_bandit

    return neural_bandit.parameter_summary(best_raw, model)


def _run_record(model, setting_index, *, setting, repeat, effective_batch_size, best_raw, metadata):
    return dict(
        model=model,
        setting_index=setting_index,
        repeat=repeat,
        setting=setting,
        effective_batch_size=effective_batch_size,
        best_raw=_serialize_raw(best_raw),
        parameters=_summarize_parameters(best_raw, model),
        **metadata,
    )


def _sweep_fits(models, grid, *, n_seeds, config, output, max_new_runs):
    """Load finished runs and fit missing ones, at most ``max_new_runs`` new fits."""
    records, completed_now = [], 0
    for model in models:
        for setting_index, setting in enumerate(grid):
            stats, _, make_trainer = make_backend(model, hidden_units=_hidden_units(setting))
            # The batch cannot exceed the number of training sessions.
            batch_size = min(setting["batch_size"], config.n_train_sessions)
            optimizer, chunk = make_trainer(dict(setting, batch_size=batch_size))
            trainer = _Trainer(stats, optimizer, chunk)
            for repeat in range(n_seeds):
                run_id = f"{model}_setting-{setting_index:02d}_seed-{repeat:02d}"
                done_file, checkpoint = output / f"{run_id}.json", output / f"{run_id}.npz"
                if done_file.exists():
                    records.append(json.loads(done_file.read_text()))
                    continue
                if max_new_runs is not None and completed_now >= max_new_runs:
                    continue
                best_raw, metadata = _fit_run(
                    run_id,
                    model,
                    setting=setting,
                    repeat=repeat,
                    trainer=trainer,
                    config=config,
                    checkpoint=checkpoint,
                )
                record = _run_record(
                    model,
                    setting_index,
                    setting=setting,
                    repeat=repeat,
                    effective_batch_size=batch_size,
                    best_raw=best_raw,
                    metadata=metadata,
                )
                done_file.write_text(json.dumps(record, indent=2, allow_nan=False))
                records.append(record)
                completed_now += 1
    return records


def _select_winner(records, model):
    """Record of ``model`` with the lowest validation score; ties go to the first."""
    return min((r for r in records if r["model"] == model), key=lambda r: r["best_validation"])


def _save_selected_parameters(path, best_raw):
    """Save a winner's raw parameters; neural models are saved one array per key."""
    with path.open("wb") as file:
        if isinstance(best_raw, dict):
            np.savez(file, **best_raw)
        else:
            np.savez(file, parameters=np.asarray(best_raw))


def _selected_parameters_name(model):
    """File name under which a model's winning raw parameters are saved."""
    return f"{model}_selected_parameters.npz"


def _score_winner(model, winner, parts):
    """Score a validation-selected winner on held-out data; returns its summary and test rows."""
    best_raw = restore_raw(winner["best_raw"])
    validation_ce, _ = score_sequences(best_raw, parts["validation"], model)
    test_ce, rows = score_sequences(best_raw, parts["test"], model)
    selected = dict(
        model=model,
        setting=winner["setting"],
        repeat=winner["repeat"],
        selected_step=winner["best_step"],
        parameters=winner["parameters"],
        parameters_file=_selected_parameters_name(model),
        validation_cross_entropy=validation_ce,
        validation_release_score=winner["best_validation"],
        test_cross_entropy=test_ce,
        test_release_score=_release_score_per_trial(rows),
        effective_batch_size=winner["effective_batch_size"],
        geometric_mean_choice_probability=float(np.exp(-test_ce)),
    )
    return selected, [dict(model=model, **r) for r in rows]


def _sweep_result(records, expected, *, output, identities, split_details, selected, test_sessions):
    return dict(
        complete=len(records) == expected,
        completed_runs=len(records),
        expected_runs=expected,
        output_dir=str(output),
        split_mice=identities,
        **split_details,
        selected=selected,
        test_sessions=test_sessions,
    )


def _library_versions():
    """Versions of the numerical libraries a run depends on, recorded in the manifest."""
    jax, _, optax = dependencies()
    return dict(jax=jax.__version__, optax=optax.__version__, numpy=np.__version__)


def _finish_sweep(models, records, *, expected, parts, output, identities, split_details):
    """Pick each winner on validation alone, score it on test, then write the outputs."""
    scored = []
    # Each model's winner is picked, scored and saved before the next model is looked
    # at, so a selection or scoring failure part-way through leaves the earlier
    # winners' files on disk and writes no results.json.
    for model in models:
        winner = _select_winner(records, model)
        scored.append(_score_winner(model, winner, parts))
        _save_selected_parameters(
            output / _selected_parameters_name(model), restore_raw(winner["best_raw"])
        )
    result = _sweep_result(
        records,
        expected,
        output=output,
        identities=identities,
        split_details=split_details,
        selected=[summary for summary, _ in scored],
        test_sessions=[row for _, rows in scored for row in rows],
    )
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def run_paper_sweep(
    sequences: list[dict],
    output_root: str | Path,
    grid: list[dict] | None = None,
    n_seeds: int = 5,
    max_steps: int = 1_000_000,
    validation_every: int = 500,
    split_seed: int = 42,
    fit_seed: int = 123,
    max_new_runs: int | None = None,
    profile: str = "full",
    models: tuple[str, ...] = MODEL_NAMES,
    test_problem: int | None = None,
    validation_problem: int | None = None,
) -> dict:
    """Run/resume AdamW sweep; select only on validation, score test after completion.

    Checkpoints contain parameters, AdamW moments, sampling RNG and best validation
    iterate, so restarting an interrupted run resumes exactly at its saved step.
    Runs use matched initialization seeds/minibatch streams across candidate models.
    Test scores are never computed during optimization or hyperparameter selection.
    """
    _, jnp, _ = dependencies()
    from src.agents import neural_bandit

    _validate_models(models, (*MODEL_NAMES, *neural_bandit.NEURAL_MODELS))
    if grid is None:
        grid = FULL_GRID
    _validate_sweep_sizes(
        grid,
        n_seeds=n_seeds,
        max_steps=max_steps,
        validation_every=validation_every,
        max_new_runs=max_new_runs,
    )
    _validate_settings(grid)
    _validate_sequence_lengths(sequences)
    parts, identities, split_details = _split_sequences(
        sequences, split_seed, test_problem, validation_problem
    )
    # Only the training arrays and validation sequences enter ``config``; the test
    # partition is first used by ``_finish_sweep``, after every run has finished.
    train_arrays = tuple(jnp.asarray(x) for x in pack_sequences(parts["train"]))
    settings = dict(
        profile=profile,
        models=list(models),
        grid=grid,
        n_seeds=n_seeds,
        max_steps=max_steps,
        validation_every=validation_every,
        split_seed=split_seed,
        fit_seed=fit_seed,
    )
    identity = _build_manifest(
        settings,
        split_details,
        identities,
        fingerprint_sequences(sequences),
        _library_versions(),
        _neural_identity(models),
    )
    output = manifest_dir(output_root, identity, _read_source_snapshots(models))
    config = _FitConfig(
        validation_sequences=parts["validation"],
        train_arrays=train_arrays,
        fit_seed=fit_seed,
        max_steps=max_steps,
        validation_every=validation_every,
    )
    records = _sweep_fits(
        models, grid, n_seeds=n_seeds, config=config, output=output, max_new_runs=max_new_runs
    )
    expected = len(models) * len(grid) * n_seeds
    if len(records) != expected:
        return _sweep_result(
            records,
            expected,
            output=output,
            identities=identities,
            split_details=split_details,
            selected=[],
            test_sessions=[],
        )
    return _finish_sweep(
        models,
        records,
        expected=expected,
        parts=parts,
        output=output,
        identities=identities,
        split_details=split_details,
    )


def run_problem_sweep(
    sequences: list[dict],
    output_root: str | Path,
    test_problems: list[int] | None = None,
    max_new_runs: int | None = None,
    **kwargs: Any,
) -> dict:
    """Run pooled leave-one-problem-out fits, with independent selection per fold.

    max_new_runs is a global budget across folds, including resumable partial runs.
    Aggregate test reports are withheld until all requested folds complete.
    """
    problems = sorted({int(s["problem"]) for s in sequences})
    tests = problems if test_problems is None else list(test_problems)
    if not tests or len(set(tests)) != len(tests):
        raise ValueError("Need unique nonempty test problems.")
    if max_new_runs is not None and max_new_runs < 0:
        raise ValueError("max_new_runs must be nonnegative.")
    # Validate every fold before starting any training.
    for problem in tests:
        split_by_problem(sequences, problem)
    folds, remaining = [], max_new_runs
    for problem in tests:
        print(f"Pooled problem fold: test problem {problem}", flush=True)
        before = run_paper_sweep(
            sequences, output_root, test_problem=problem, max_new_runs=0, **kwargs
        )
        if not before["complete"] and (remaining is None or remaining > 0):
            result = run_paper_sweep(
                sequences, output_root, test_problem=problem, max_new_runs=remaining, **kwargs
            )
            if remaining is not None:
                remaining -= result["completed_runs"] - before["completed_runs"]
        else:
            result = before
        folds.append(dict(test_problem=int(problem), **result))
    identity = json.dumps([f["output_dir"] for f in folds])
    output = Path(output_root) / (
        "problem-cv-" + hashlib.sha256(identity.encode()).hexdigest()[:12]
    )
    output.mkdir(parents=True, exist_ok=True)
    complete = all(f["complete"] for f in folds)
    result = dict(
        complete=complete,
        split_strategy="pooled_problem",
        completed_runs=sum(f["completed_runs"] for f in folds),
        expected_runs=sum(f["expected_runs"] for f in folds),
        output_dir=str(output),
        folds=folds,
        selected=[],
        test_sessions=[],
    )
    if complete:
        result["selected"] = [
            dict(
                row,
                test_problem=f["test_problem"],
                parameters_file=str((Path(f["output_dir"]) / row["parameters_file"]).resolve()),
            )
            for f in folds
            for row in f["selected"]
        ]
        result["test_sessions"] = [
            dict(row, test_problem=f["test_problem"]) for f in folds for row in f["test_sessions"]
        ]
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
