"""Do the mice switch between strategy states? The GLM-HMM on mice, simulations and replays.

Per mouse (:func:`analyze_mouse`): the signals of
:mod:`src.agents.glm_hmm_agent` are built from the trial table with the mouse's fitted kernel
Q rates (:func:`fit_agents`: one set for the mouse, so every number of states sees the same
signals); GLM-HMMs with one to four states are scored on held-out sessions (5 folds, the
sessions of each problem dealt across them); the number of states is chosen by the one-standard-
error rule (:mod:`src.behavior_analysis.glm_hmm.glm_hmm_selection`); and the fit at the chosen count
(at
least two, for the descriptions) gives the state weights, occupancy, dwell times, state
probability by trial in session, by problem and around block changes, and the choice kernel
within each state. The static kernel Q-learner and WSLS (Habit) are refitted on the same folds
as reference levels. The kernel Q rates are those of the fit to all of the mouse's trials, as in
the simulations and the replays below, so the mouse and its null see the same signals.

Simulations (:func:`simulate_scenario`, :func:`recover_scenario`): agents with known switching
(three states: a value learner, WSLS (Magnitude) and a habit/biased strategy, with the mouse's
fitted parameters, sticky transitions with a slow or a fast dwell time) and agents without
(the fitted static agents and a stationary blend), played on the mouse's own problems.

Replays (:func:`replay_source`) are the null of the real gain: the fitted kernel Q agent plays
the mouse's problems, and the same cross-validation finds the gain of more states that the method
produces from a static agent. The mouse's gain is calibrated against it
(:func:`calibrate_gains`: z and empirical p per mouse, Wilcoxon across mice).
"""

import hashlib
import inspect
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed

from src.agents import drifting_simulation as sim
from src.agents import glm_hmm_agent as gha
from src.agents import kernel_q_agent as kq
from src.agents import null_agents as na
from src.behavior_analysis.data.drifting_trial_table import drifting_trial_table
from src.behavior_analysis.glm_hmm import glm_hmm as hmm
from src.behavior_analysis.glm_hmm import glm_hmm_features as feats
from src.behavior_analysis.glm_hmm import glm_hmm_kernel as ker
from src.behavior_analysis.glm_hmm import glm_hmm_recovery as rec
from src.behavior_analysis.glm_hmm import glm_hmm_selection as sel
from src.behavior_analysis.rank_tests import rank_calibration as rcal
from src.behavior_analysis.strategy_models import strategy_models as sm
from src.pipelines.switching import switch_regression as switch_pipe

CACHE_VERSION = 1
SLOW_DWELL, FAST_DWELL = 300, 50
SCENARIOS = ("switch_slow", "switch_fast", "static_kernel_q", "static_wsls_habit", "static_blend")
SWITCHING = ("switch_slow", "switch_fast")
STATE_NAMES = ("Value", "WSLS (Magnitude)", "Habit")
# Each readout of the calibration, in the direction a real switching process would push it.
PREDICTED_GAIN = +1
BLOCK_WINDOW = (-10, 20)
TRIAL_BIN = 10
MODEL_LABELS = {sm.KERNEL_Q: "Q-Learner (Kernel + Forgetting)", sm.WSLS_HABIT: "WSLS (Habit)"}


def code_version() -> str:
    """A cache version that changes with ``CACHE_VERSION`` and the source of the analysis."""
    modules = (hmm, feats, ker, rec, sel, gha, sm, sim)
    digest = hashlib.sha256(
        "\n".join([str(CACHE_VERSION), *(inspect.getsource(m) for m in modules)]).encode()
    )
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class Config:
    """The protocol of the analysis; the same for the mice, the simulations and the replays.

    Attributes:
        n_states (tuple[int, ...]): State counts of the model selection; includes 1.
        replay_states (tuple[int, ...]): The counts the replays and simulations' gains cover;
            a subset of *n_states* (the null needs the same fits as the mouse's number).
        n_restarts (int): EM starts per fit.
        screen_iterations (int | None): EM iterations of the screening stage of the starts.
        n_keep (int): Starts that continue after screening.
        max_iterations (int): EM iterations allowed.
        n_folds (int): Cross-validation folds (of sessions).
        prior (hmm.Prior): Penalties of the log posterior.
        kernel_threshold (float): Posterior above which a trial belongs to a state.
        agent_starts (int): Random starts of the static agents' fits.
    """

    n_states: tuple = (1, 2, 3, 4)
    replay_states: tuple = (1, 2, 3)
    n_restarts: int = 10
    screen_iterations: int | None = 12
    n_keep: int = 2
    max_iterations: int = 150
    n_folds: int = 5
    prior: hmm.Prior = field(default_factory=hmm.Prior)
    kernel_threshold: float = ker.DEFAULT_THRESHOLD
    agent_starts: int = sm.Q_STARTS

    def __post_init__(self):
        if 1 not in self.n_states or 1 not in self.replay_states:
            raise ValueError("n_states and replay_states must include one state")
        if not set(self.replay_states) <= set(self.n_states):
            raise ValueError("replay_states must be among n_states")
        if self.n_restarts < 1:
            raise ValueError(f"n_restarts must be at least 1, got {self.n_restarts}")

    def fit_kwargs(self) -> dict:
        return dict(
            max_iterations=self.max_iterations,
            screen_iterations=self.screen_iterations,
            n_keep=self.n_keep,
        )


class Agents(NamedTuple):
    """The static agents fitted to one mouse's trials.

    Attributes:
        kq (tuple[float, ...]): Kernel Q parameters (``kernel_q_agent.PARAMETER_NAMES`` order).
        habit (na.WslsHabitParams): WSLS (Habit) parameters.
    """

    kq: tuple
    habit: na.WslsHabitParams


# ---------------------------------------------------------------------------
# Fitting the static agents
# ---------------------------------------------------------------------------


def fit_agents(trials: pd.DataFrame, n_starts: int, seed: int) -> Agents:
    """The kernel Q-learner and WSLS (Habit) fitted to all of one mouse's trials.

    Raises:
        RuntimeError: If a fit does not converge.
    """
    return Agents(
        kq=sm.SPECS[sm.KERNEL_Q].fit(trials, n_starts, seed),
        habit=sm.SPECS[sm.WSLS_HABIT].fit(trials, n_starts, seed),
    )


def _memoized(function, cache_dir: Path | None):
    return function if cache_dir is None else Memory(cache_dir, verbose=0).cache(function)


def _fit_agents_cached(trials, n_starts, seed, version):
    return fit_agents(trials, n_starts, seed)


def fit_cohort_agents(
    trial_table: pd.DataFrame,
    n_starts: int = sm.Q_STARTS,
    seed: int = 0,
    n_jobs: int = 1,
    cache_dir: Path | None = None,
) -> dict[str, Agents]:
    """``{mouse: Agents}`` for every mouse of *trial_table*, in parallel over mice."""
    by_mouse = dict(sorted(trial_table.groupby("mouse"), key=lambda item: item[0]))
    fit = _memoized(_fit_agents_cached, cache_dir)
    version = code_version()
    results = Parallel(n_jobs=n_jobs)(
        delayed(fit)(rows.reset_index(drop=True), n_starts, seed, version)
        for rows in by_mouse.values()
    )
    return dict(zip(by_mouse, results, strict=True))


# ---------------------------------------------------------------------------
# Cross-validation readouts
# ---------------------------------------------------------------------------


def cv_scores(data: feats.SessionData, folds: np.ndarray, config: Config, seed: int):
    """Held-out scores of every state count of *config* on the sessions of *data*."""
    return sel.cross_validate(
        data.sequences, folds, config.n_states, config.prior, config.n_restarts, seed,
        **config.fit_kwargs(),
    )  # fmt: skip


def gain_readouts(
    scores: pd.DataFrame, states: Sequence[int], best_over: Sequence[int] | None = None
) -> dict:
    """The gain over one state of each count above one, and the best of them, in nats per trial.

    Args:
        scores (pd.DataFrame):
            :func:`src.behavior_analysis.glm_hmm.glm_hmm_selection.cross_validate`.
        states (Sequence[int]): The counts to report a gain for.
        best_over (Sequence[int] | None): The counts ``best_gain`` ranges over. Default
            ``None``: *states*.

    Returns:
        dict: ``gain_k<K>`` for each count above one in *states* and ``best_gain``.
    """
    table = sel.selection_table(scores).set_index("n_states")
    readouts = {f"gain_k{k}": float(table.loc[k, "gain"]) for k in states if k > 1}
    best_states = states if best_over is None else best_over
    readouts["best_gain"] = sel.best_gain(scores[scores["n_states"].isin(best_states)])
    return readouts


# ---------------------------------------------------------------------------
# Static reference levels on the same folds
# ---------------------------------------------------------------------------


def _session_keys(frame: pd.DataFrame) -> pd.Series:
    return frame["problem"].map(str) + "|" + frame["session_index"].astype(str)


def static_levels(
    trials: pd.DataFrame, data: feats.SessionData, folds: np.ndarray, n_starts: int, seed: int
) -> pd.DataFrame:
    """Held-out log likelihood of the kernel Q-learner and WSLS (Habit) on the HMM's folds.

    Each static model is fitted on the training sessions and scores the held-out sessions' scored
    trials. The kernel Q-learner predicts from the values of the whole problem history (values
    carry across sessions), as the HMM's signals do.

    Args:
        trials (pd.DataFrame): The mouse's trial table (the one *data* was made from).
        data (feats.SessionData): The mouse's sessions.
        folds (np.ndarray): Fold of every session.
        n_starts (int): Random starts of the kernel Q fit.
        seed (int): Base seed.

    Returns:
        pd.DataFrame: ``fold, model, log_likelihood, n_scored``; summarized by
        :func:`static_nats`.

    Raises:
        RuntimeError: If a fit does not converge, or the trial order disagrees with *data*.
    """
    full = sim.problem_sequences(trials, "magnitude")
    seq = data.sequences
    if not np.array_equal(full.actions[full.mask], seq.chosen[seq.mask]):
        raise RuntimeError("the trial table's order disagrees with the session data's")
    scored = seq.scored[seq.mask]
    trial_fold = folds[data.trials["session"].to_numpy()]
    session_fold = dict(zip(_session_keys(data.sessions), folds, strict=True))
    row_fold = _session_keys(trials).map(session_fold).to_numpy()
    rows = []
    for fold in np.unique(folds):
        train, test = trials[row_fold != fold], trials[row_fold == fold]
        n_scored = int((scored & (trial_fold == fold)).sum())
        params = sm.SPECS[sm.KERNEL_Q].fit(train, n_starts, seed + int(fold))
        per_trial = kq.trial_log_probabilities(params, full)[full.mask]
        rows.append(
            dict(fold=int(fold), model=sm.KERNEL_Q, n_scored=n_scored,
                 log_likelihood=float(per_trial[scored & (trial_fold == fold)].sum()))
        )  # fmt: skip
        habit = sm.SPECS[sm.WSLS_HABIT].fit(train, n_starts, seed + int(fold))
        rows.append(
            dict(fold=int(fold), model=sm.WSLS_HABIT, n_scored=n_scored,
                 log_likelihood=float(sm.SPECS[sm.WSLS_HABIT].score(test, habit)))
        )  # fmt: skip
    return pd.DataFrame(rows)


def static_nats(levels: pd.DataFrame) -> pd.DataFrame:
    """``model, nats, n_scored``: pooled held-out nats per scored trial of each static model."""
    pooled = levels.groupby("model", sort=False).agg(
        log_likelihood=("log_likelihood", "sum"), n_scored=("n_scored", "sum")
    )
    pooled["nats"] = -pooled["log_likelihood"] / pooled["n_scored"]
    return pooled.reset_index()[["model", "nats", "n_scored"]]


# ---------------------------------------------------------------------------
# Describing a fitted model
# ---------------------------------------------------------------------------


def weights_table(params: hmm.Params) -> pd.DataFrame:
    """``n_states, state, feature, weight`` of a fit (the states in their sorted order)."""
    k = params.n_states
    return pd.DataFrame(
        [
            dict(n_states=k, state=z, feature=name, weight=float(params.weights[z, i]))
            for z in range(k)
            for i, name in enumerate(gha.FEATURES)
        ]
    )


def occupancy_table(params: hmm.Params, gamma: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    """``state, occupancy, dwell`` (trials) and ``initial`` probability of each state."""
    return pd.DataFrame(
        dict(
            state=np.arange(params.n_states),
            occupancy=hmm.occupancy(gamma, mask),
            dwell=hmm.dwell_times(params.transition),
            initial=params.initial,
        )
    )


def _mean_posterior(trials: pd.DataFrame, gamma: np.ndarray, by: pd.Series, name: str):
    """Mean posterior of each state over the trials grouped by *by*."""
    probs = gamma[trials["session"].to_numpy(), trials["step"].to_numpy()]
    frame = pd.DataFrame(probs).assign(**{name: by.to_numpy()})
    long = frame.melt(id_vars=name, var_name="state", value_name="probability")
    out = long.groupby([name, "state"], as_index=False).agg(
        probability=("probability", "mean"), n=("probability", "size")
    )
    return out[["state", name, "probability", "n"]]


def by_trial_in_session(data: feats.SessionData, gamma: np.ndarray) -> pd.DataFrame:
    """``state, trial_in_session, probability, n``: posterior by trial in the session (bins)."""
    binned = (data.trials["trial_in_session"] // TRIAL_BIN) * TRIAL_BIN
    return _mean_posterior(data.trials, gamma, binned, "trial_in_session")


def by_problem(data: feats.SessionData, gamma: np.ndarray) -> pd.DataFrame:
    """``state, problem_number, probability, n``: posterior by the order of the mouse's problems."""
    order = {p: i + 1 for i, p in enumerate(dict.fromkeys(data.trials["problem"].map(str)))}
    numbers = data.trials["problem"].map(str).map(order)
    return _mean_posterior(data.trials, gamma, numbers, "problem_number")


def block_aligned(data: feats.SessionData, gamma: np.ndarray, window=BLOCK_WINDOW) -> pd.DataFrame:
    """``state, offset, probability, n``: posterior around block changes within a session.

    An event is a trial whose block differs from the session's previous trial's (time 0 is the
    first trial of the new block); an offset counts the trials of the same session that exist.
    Empty if the table has no ``block`` column.
    """
    trials = data.trials
    if "block" not in trials.columns:
        return pd.DataFrame(columns=["state", "offset", "probability", "n"])
    block = trials["block"].to_numpy()
    same_session = trials["session"].to_numpy()[1:] == trials["session"].to_numpy()[:-1]
    changed = np.flatnonzero(same_session & (block[1:] != block[:-1])) + 1
    rows = []
    for event in changed:
        session = trials["session"].iloc[event]
        for offset in range(window[0], window[1] + 1):
            i = event + offset
            if 0 <= i < len(trials) and trials["session"].iloc[i] == session:
                p = gamma[session, trials["step"].iloc[i]]
                rows += [dict(state=z, offset=offset, probability=p[z]) for z in range(len(p))]
    if not rows:
        return pd.DataFrame(columns=["state", "offset", "probability", "n"])
    out = (
        pd.DataFrame(rows)
        .groupby(["state", "offset"], as_index=False)
        .agg(probability=("probability", "mean"), n=("probability", "size"))
    )
    return out


class MouseResult(NamedTuple):
    """One mouse's GLM-HMM analysis.

    Attributes:
        mouse (str): Mouse id.
        n_trials (int): Trials analyzed.
        n_sessions (int): Sessions analyzed.
        scores (pd.DataFrame): Held-out score per fold and state count.
        table (pd.DataFrame): ``selection_table``: nats, gain over one state, gain se.
        selected_k (int): States chosen by the one-standard-error rule.
        summary_k (int): States of the descriptions below: the chosen count, at least 2.
        readouts (dict): ``gain_readouts`` over the replay's state counts.
        static_levels (pd.DataFrame): ``model, nats, n_scored`` of the static models.
        weights (pd.DataFrame): ``weights_table`` of the one-state fit and of ``summary_k``.
        occupancy (pd.DataFrame): ``occupancy_table`` of the ``summary_k`` fit.
        transition (np.ndarray): Transition matrix of the ``summary_k`` fit.
        by_trial_in_session (pd.DataFrame): State probability by trial in the session.
        by_problem (pd.DataFrame): State probability by problem.
        block_aligned (pd.DataFrame): State probability around block changes.
        kernel (pd.DataFrame): ``glm_hmm_kernel.kernel_by_state`` of the ``summary_k`` fit.
        n_converged (tuple): ``(starts converged, starts)`` of the ``summary_k`` fit.
    """

    mouse: str
    n_trials: int
    n_sessions: int
    scores: pd.DataFrame
    table: pd.DataFrame
    selected_k: int
    summary_k: int
    readouts: dict
    static_levels: pd.DataFrame
    weights: pd.DataFrame
    occupancy: pd.DataFrame
    transition: np.ndarray
    by_trial_in_session: pd.DataFrame
    by_problem: pd.DataFrame
    block_aligned: pd.DataFrame
    kernel: pd.DataFrame
    n_converged: tuple


def _fit(data: feats.SessionData, k: int, config: Config, seed: int) -> hmm.HmmFit:
    return hmm.fit_restarts(
        data.sequences, k, config.prior, config.n_restarts, seed, **config.fit_kwargs()
    )


def analyze_mouse(
    trials: pd.DataFrame, agents: Agents, config: Config, seed: int, version: str = ""
) -> MouseResult:
    """The whole GLM-HMM analysis of one mouse (see the module text).

    Args:
        trials (pd.DataFrame): One mouse's rows of the drifting trial table.
        agents (Agents): The static agents fitted to the mouse; the kernel Q rates make the
            signals.
        config (Config): The protocol.
        seed (int): Base seed of the fits.
        version (str): Only part of the cache key. Default empty.

    Returns:
        MouseResult: Selection, descriptions and kernel.

    Raises:
        ValueError: If the mouse has fewer sessions than folds.
        RuntimeError: If a fit does not converge.
    """
    data = feats.session_arrays(trials, agents.kq)
    folds = feats.assign_folds(data, config.n_folds)
    scores = cv_scores(data, folds, config, seed)
    selected = sel.select_n_states(scores)
    summary_k = max(selected, 2) if max(config.n_states) >= 2 else 1
    one, fit = _fit(data, 1, config, seed), _fit(data, summary_k, config, seed + 1)
    gamma = hmm.posteriors(fit.params, data.sequences)
    return MouseResult(
        mouse=str(trials["mouse"].iloc[0]),
        n_trials=int(data.sequences.mask.sum()),
        n_sessions=len(data.sessions),
        scores=scores,
        table=sel.selection_table(scores),
        selected_k=selected,
        summary_k=summary_k,
        readouts=gain_readouts(scores, config.n_states, best_over=config.replay_states),
        static_levels=static_nats(static_levels(trials, data, folds, config.agent_starts, seed)),
        weights=pd.concat(
            [weights_table(one.params), weights_table(fit.params)], ignore_index=True
        ).drop_duplicates(["n_states", "state", "feature"]),
        occupancy=occupancy_table(fit.params, gamma, data.sequences.mask),
        transition=fit.params.transition,
        by_trial_in_session=by_trial_in_session(data, gamma),
        by_problem=by_problem(data, gamma),
        block_aligned=block_aligned(data, gamma),
        kernel=ker.kernel_by_state(trials, data, gamma, config.kernel_threshold),
        n_converged=(fit.n_converged, fit.n_restarts),
    )


def analyze_mice(
    trial_table: pd.DataFrame,
    agents: Mapping[str, Agents],
    config: Config,
    seed: int,
    *,
    n_jobs: int = 1,
    cache_dir: Path | None = None,
) -> dict[str, MouseResult]:
    """:func:`analyze_mouse` for every mouse of *trial_table*, in parallel over mice."""
    by_mouse = dict(sorted(trial_table.groupby("mouse"), key=lambda item: item[0]))
    analyze = _memoized(analyze_mouse, cache_dir)
    version = code_version()
    results = Parallel(n_jobs=n_jobs)(
        delayed(analyze)(rows.reset_index(drop=True), agents[mouse], config, seed, version)
        for mouse, rows in by_mouse.items()
    )
    return dict(zip(by_mouse, results, strict=True))


# ---------------------------------------------------------------------------
# Simulated agents
# ---------------------------------------------------------------------------


def sticky_transition(n_states: int, dwell: float) -> np.ndarray:
    """Transition matrix of states that last *dwell* trials on average, leaving evenly."""
    if dwell <= 1:
        raise ValueError(f"dwell must exceed 1 trial, got {dwell}")
    stay = 1.0 - 1.0 / dwell
    transition = np.full((n_states, n_states), (1 - stay) / (n_states - 1))
    np.fill_diagonal(transition, stay)
    return transition


def strategy_weights(agents: Agents) -> np.ndarray:
    """``(3, features)`` weights of the three strategies: value, WSLS (Magnitude), habit."""
    return np.stack(
        [gha.value_weights(agents.kq), gha.wsls_weights(agents.habit), gha.habit_weights()]
    )


class Simulated(NamedTuple):
    """An agent's play of a mouse's problems.

    Attributes:
        table (pd.DataFrame): The simulated trial table, under the mouse's id.
        true_states (np.ndarray | None): The generating state of every trial in the order of the
            session data's trials; ``None`` for a static agent.
        weights (np.ndarray | None): ``(states, features)`` generating weights.
        transition (np.ndarray | None): Generating transition matrix.
    """

    table: pd.DataFrame
    true_states: np.ndarray | None
    weights: np.ndarray | None
    transition: np.ndarray | None


def _static_policy(scenario: str, agents: Agents, rng: np.random.Generator):
    if scenario == "static_kernel_q":
        return lambda layout: sim.KernelQPolicy(agents.kq, "magnitude", layout)
    if scenario == "static_wsls_habit":
        return lambda layout: na.NullPolicy(agents.habit)
    blend = gha.blend_weights(strategy_weights(agents))[None, :]
    return lambda layout: gha.FeaturePolicy(
        blend, np.ones((1, 1)), np.ones(1), agents.kq, layout,
        np.random.default_rng(rng.integers(2**32)),
    )  # fmt: skip


def simulate_scenario(
    trials: pd.DataFrame, agents: Agents, scenario: str, rules: sim.TaskRules, seed: int
) -> Simulated:
    """One agent of *scenario* playing the mouse's problems once.

    Args:
        trials (pd.DataFrame): The mouse's trial table (the layouts to play).
        agents (Agents): The mouse's fitted agents (they set the strategies' parameters).
        scenario (str): One of ``SCENARIOS``.
        rules (sim.TaskRules): Drift and block lengths of the simulated task.
        seed (int): Seed.

    Returns:
        Simulated: The trials and, for a switching scenario, the true states and parameters.

    Raises:
        ValueError: For an unknown scenario.
        RuntimeError: If a switching agent's logged signals disagree with the signals rebuilt from
            its trial table.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")
    mouse = str(trials["mouse"].iloc[0])
    rng = np.random.default_rng([seed, zlib.crc32(mouse.encode()), SCENARIOS.index(scenario)])
    layouts = sim.mouse_layouts(trials)
    if scenario in SWITCHING:
        weights = strategy_weights(agents)
        transition = sticky_transition(3, SLOW_DWELL if scenario == "switch_slow" else FAST_DWELL)
        policies = []

        def make(layout):
            policies.append(
                gha.FeaturePolicy(
                    weights,
                    transition,
                    np.full(3, 1 / 3),
                    agents.kq,
                    layout,
                    np.random.default_rng(rng.integers(2**32)),
                )  # fmt: skip
            )
            return policies[-1]

        table = drifting_trial_table(sim.simulate_policy_mouse(make, layouts, rules, rng, mouse))
        states = np.concatenate([p.true_states() for p in policies])
        data = feats.session_arrays(table, agents.kq)
        logged = np.concatenate([p.logged_features() for p in policies])
        if not np.allclose(logged, data.sequences.X[data.sequences.mask]):
            raise RuntimeError(
                "the simulated agent's signals differ from those rebuilt for the fit"
            )
        return Simulated(table, states, weights, transition)
    played = sim.simulate_policy_mouse(
        _static_policy(scenario, agents, rng), layouts, rules, rng, mouse
    )
    return Simulated(drifting_trial_table(played), None, None, None)


class Recovery(NamedTuple):
    """The analysis of one simulated agent.

    Attributes:
        scenario (str): One of ``SCENARIOS``.
        mouse (str): The mouse whose problems and parameters the agent used.
        readouts (dict): ``selected_k`` and ``gain_readouts``.
        scores (pd.DataFrame): Held-out scores per fold and state count.
        table (pd.DataFrame): ``selection_table``.
        accuracy (float): Posterior state accuracy of the three-state fit (NaN for a static
            agent).
        weights (pd.DataFrame): ``state, fitted_state, feature, true, recovered`` of the
            three-state fit (empty for a static agent).
        dwell (pd.DataFrame): ``state, true, recovered`` mean dwell times (empty likewise).
    """

    scenario: str
    mouse: str
    readouts: dict
    scores: pd.DataFrame
    table: pd.DataFrame
    accuracy: float
    weights: pd.DataFrame
    dwell: pd.DataFrame


def recover_scenario(
    trials: pd.DataFrame,
    agents: Agents,
    scenario: str,
    rules: sim.TaskRules,
    config: Config,
    seed: int,
    version: str = "",
) -> Recovery:
    """Simulate *scenario* on the mouse's problems and analyze it like a mouse.

    For a switching scenario the three-state fit (the truth) is also compared with the generating
    weights, states and dwell times after matching its states to the true ones.

    Args:
        trials (pd.DataFrame): The mouse's trial table.
        agents (Agents): The mouse's fitted agents.
        scenario (str): One of ``SCENARIOS``.
        rules (sim.TaskRules): The simulated task.
        config (Config): The protocol; includes three states for a switching scenario.
        seed (int): Seed of the simulation and the fits.
        version (str): Only part of the cache key. Default empty.

    Returns:
        Recovery: Selection, gains and, for a switching scenario, recovery of the truth.

    Raises:
        ValueError: For an unknown scenario, or a switching scenario without three states in
            *config*.
    """
    simulated = simulate_scenario(trials, agents, scenario, rules, seed)
    data = feats.session_arrays(simulated.table, agents.kq)
    folds = feats.assign_folds(data, config.n_folds)
    scores = cv_scores(data, folds, config, seed)
    readouts = {
        "selected_k": sel.select_n_states(scores),
        **gain_readouts(scores, config.replay_states),
    }
    mouse = str(trials["mouse"].iloc[0])
    table = sel.selection_table(scores)
    if simulated.true_states is None:
        return Recovery(
            scenario, mouse, readouts, scores, table, np.nan, pd.DataFrame(), pd.DataFrame()
        )
    if 3 not in config.n_states:
        raise ValueError("a switching scenario needs three states in the config")
    fit = _fit(data, 3, config, seed + 1)
    gamma = hmm.posteriors(fit.params, data.sequences)
    matching = rec.match_states(simulated.weights, fit.params.weights)
    posterior = gamma[data.sequences.mask]
    return Recovery(
        scenario, mouse, readouts, scores, table,
        rec.state_accuracy(posterior, simulated.true_states, matching),
        rec.weight_recovery(simulated.weights, fit.params.weights, matching, gha.FEATURES),
        rec.dwell_recovery(simulated.transition, fit.params.transition, matching),
    )  # fmt: skip


# ---------------------------------------------------------------------------
# Replays: the null of the real gain
# ---------------------------------------------------------------------------


def replay_one(
    trials: pd.DataFrame,
    agents: Agents,
    rules: sim.TaskRules,
    config: Config,
    replicate: int,
    seed: int,
    version: str,
    kernel_states: int | None = None,
) -> dict:
    """One replay of the mouse by its fitted kernel Q agent, analyzed like the mouse.

    Args:
        trials (pd.DataFrame): The mouse's trial table (the problems to play).
        agents (Agents): The mouse's fitted agents; the kernel Q agent plays.
        rules (sim.TaskRules): The simulated task.
        config (Config): The protocol; the gains cover ``config.replay_states``.
        replicate (int): Replay number.
        seed (int): Base seed.
        version (str): Only part of the cache key.
        kernel_states (int | None): If given, also the choice kernel within each state of a fit
            with this many states (``kernel`` entry). Default ``None``.

    Returns:
        dict: ``replicate``, ``selected_k``, the gain readouts and, if asked, ``kernel``.
    """
    mouse = str(trials["mouse"].iloc[0])
    rng = np.random.default_rng([seed, zlib.crc32(mouse.encode()), replicate])
    played = sim.simulate_policy_mouse(
        lambda layout: sim.KernelQPolicy(agents.kq, "magnitude", layout),
        sim.mouse_layouts(trials), rules, rng, mouse,
    )  # fmt: skip
    table = drifting_trial_table(played)
    data = feats.session_arrays(table, agents.kq)
    folds = feats.assign_folds(data, config.n_folds)
    replay_config = replace(config, n_states=config.replay_states)
    scores = cv_scores(data, folds, replay_config, seed + replicate)
    out = {
        "replicate": replicate,
        "selected_k": sel.select_n_states(scores),
        **gain_readouts(scores, config.replay_states),
    }
    if kernel_states is not None:
        fit = _fit(data, kernel_states, config, seed + replicate + 1)
        gamma = hmm.posteriors(fit.params, data.sequences)
        out["kernel"] = ker.kernel_by_state(table, data, gamma, config.kernel_threshold)
    return out


def replay_source(
    trial_table: pd.DataFrame,
    agents: Mapping[str, Agents],
    config: Config,
    replicates: int,
    *,
    rules=None,
    seed: int = 0,
    kernel_states: Mapping[str, int] | None = None,
    kernel_replicates: int = 0,
    n_jobs: int = 1,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Replays of every mouse of *agents* by its fitted kernel Q agent.

    Parallel over (mouse, replicate); each replay is cached on its own, so more replicates can be
    added later at the cost of the new ones only.

    Args:
        trial_table (pd.DataFrame): The mice's drifting trial table.
        agents (Mapping[str, Agents]): ``{mouse: fitted agents}``.
        config (Config): The protocol.
        replicates (int): Replays per mouse, at least 2.
        rules (sim.TaskRules | Mapping | None): The simulated task: one rule set, ``{cohort:
            rules}`` or ``None`` (measured within each cohort of *trial_table*).
        seed (int): Base seed.
        kernel_states (Mapping[str, int] | None): ``{mouse: states}`` for the replays that also
            get a choice kernel by state. Default none.
        kernel_replicates (int): The first this many replays of each mouse get that kernel.
        n_jobs (int): Parallel workers.
        cache_dir (Path | None): Cache each replay here. Default ``None``.

    Returns:
        pd.DataFrame: ``mouse, replicate, selected_k`` and the gain readouts, one row per replay;
        the kernels are in the ``kernel`` column (a table or NaN).

    Raises:
        ValueError: For fewer than 2 replicates or a mouse without fitted agents.
    """
    if replicates < 2:
        raise ValueError(f"replicates must be at least 2, got {replicates}")
    mice = sorted(set(trial_table["mouse"]))
    missing = [m for m in mice if m not in agents]
    if missing:
        raise ValueError(f"no fitted agents for mice {missing}")
    rules_of = switch_pipe.rules_by_mouse(trial_table, mice, rules)
    by_mouse = {m: rows.reset_index(drop=True) for m, rows in trial_table.groupby("mouse")}
    replay = _memoized(replay_one, cache_dir)
    version = code_version()
    jobs = [(m, r) for m in mice for r in range(replicates)]
    results = Parallel(n_jobs=n_jobs)(
        delayed(replay)(
            by_mouse[m], agents[m], rules_of[m], config, r, seed, version,
            (kernel_states or {}).get(m) if r < kernel_replicates else None,
        )
        for m, r in jobs
    )  # fmt: skip
    rows = [{"mouse": m, **res} for (m, _), res in zip(jobs, results, strict=True)]
    return pd.DataFrame(rows)


class Calibration(NamedTuple):
    """The mice's gains against the replays' null.

    Attributes:
        table (pd.DataFrame): ``rank_calibration.calibrate`` with ``predicted`` and ``passes``.
        summary (pd.DataFrame): ``rank_calibration.summarize``: mean ± se of z and the Wilcoxon p
            across mice.
    """

    table: pd.DataFrame
    summary: pd.DataFrame


def calibrate_gains(observed: pd.DataFrame, null: pd.DataFrame) -> Calibration:
    """Each mouse's gains against its replays (z, empirical p; Wilcoxon across mice).

    Args:
        observed (pd.DataFrame): ``mouse`` and one column per gain readout, one row per mouse.
        null (pd.DataFrame): :func:`replay_source` output (extra columns are ignored).

    Returns:
        Calibration: Per-mouse z and p, and the summary across mice.

    Raises:
        ValueError: If a mouse of *observed* has no replay.
    """
    readouts = [c for c in observed.columns if c != "mouse"]
    table = rcal.calibrate(observed, null[["mouse", "replicate", *readouts]])
    predicted = dict.fromkeys(readouts, PREDICTED_GAIN)
    return Calibration(rcal.passes(table, predicted), rcal.summarize(table, predicted))
