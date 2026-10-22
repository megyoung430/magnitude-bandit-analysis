"""Baseline agents for the drifting task: fitted to each mouse, then run through the same task.

Two agents share one model, the four-parameter value learner of :mod:`src.agents.value_agent`
(softmax over learned values, a bonus for repeating the previous choice and a bias toward plus
ports), and differ only in what they learn from:

- ``"magnitude"``: the reward magnitude received (0-4);
- ``"binary"``: whether any reward was received (0 or 1), so magnitude is invisible to it.

Both are fitted to each mouse's own choices (:func:`fit_agent`), which gives a likelihood
comparison with equal numbers of parameters. Each fitted agent then plays the drifting task
(:func:`simulate_mouse`) with that mouse's ports, initiation port, starting magnitudes and
trials per session, in a task whose drift rate and block lengths are measured from the data and
whose block-change rule is the one the data follow
(:mod:`src.behavior_analysis.drifting_task_rules`; the pipeline checks the data against it):

- each non-zero arm drifts by one step to the nearest free value in 1-4 with probability
  ``drift_rate`` per trial (as in :class:`src.tasks.drifting_bandit.DriftingBanditTask`);
- at the end of a block, whose length is drawn from the observed complete block lengths, the
  non-zero arm chosen most in that block drops to 0 and the arm that was at 0 takes exactly its
  value (no drift on that trial);
- blocks run on across sessions; the agent's values carry over between the sessions of a
  problem and its previous choice resets at each session start, as in the likelihood.

The simulated sessions come out as importer session dicts (:func:`simulated_sessions`): each
recording logs ``run_start`` and ``run_end`` snapshots with the block totals as the rig does,
and the sessions go through the importer's ``extract_session_trials`` and
``reconcile_reversals``, so a block change after a session's last trial is recovered as it is
for real data and every existing drifting analysis runs on them unchanged.

Randomness comes only from the ``numpy.random.Generator`` passed in; nothing touches the global
random state.
"""

import json
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.agents import value_agent
from src.behavior_analysis.port_preference import port_category
from src.behavior_import.extract_trials import extract_session_trials
from src.behavior_import.reconcile_reversals import reconcile_reversals

AGENT_KINDS = ("magnitude", "binary")
REWARD_VALUES = (1, 2, 3, 4)
DRIFTING_TASK_NAME = "simulated\\3x3_field_drifting_value_bandit"
RANKS = ("best", "second", "third")


# ---------------------------------------------------------------------------
# Rewards and fitting
# ---------------------------------------------------------------------------


def agent_rewards(rewards: np.ndarray, kind: str) -> np.ndarray:
    """What an agent of *kind* learns from on each trial.

    Args:
        rewards (np.ndarray): Reward magnitudes received.
        kind (str): One of ``AGENT_KINDS``.

    Returns:
        np.ndarray: The magnitudes for ``"magnitude"``; 1.0 where a reward was received and
        0.0 elsewhere for ``"binary"``.

    Raises:
        ValueError: For an unknown *kind*.
    """
    rewards = np.asarray(rewards, dtype=float)
    if kind == "magnitude":
        return rewards
    if kind == "binary":
        return (rewards > 0).astype(float)
    raise ValueError(f"unknown agent kind {kind!r}; expected one of {AGENT_KINDS}")


def plus_ports(ports: list, initiation: str) -> np.ndarray:
    """1.0 for each of *ports* in the same row or column as *initiation*, else 0.0."""
    return np.array([float(port_category(p, initiation) == "plus") for p in ports])


def problem_sequences(mouse_trials: pd.DataFrame, kind: str) -> value_agent.Sequences:
    """One mouse's problems as padded arrays for :mod:`src.agents.value_agent`.

    A session start, and the first trial after a dropped trial, reset the previous choice.

    Args:
        mouse_trials (pd.DataFrame): One mouse's rows of the drifting trial table.
        kind (str): One of ``AGENT_KINDS``; sets the rewards.

    Returns:
        value_agent.Sequences: Ports are indexed in sorted order within each problem.

    Raises:
        ValueError: If the offered ports change within a problem.
    """
    problems = []
    for _, rows in mouse_trials.groupby(mouse_trials["problem"].map(str), sort=False):
        rows = rows.sort_values("trial")
        ports = sorted(rows["magnitudes"].iloc[0])
        if any(sorted(m) != ports for m in rows["magnitudes"]):
            raise ValueError(f"problem {rows['problem'].iloc[0]!r}: offered ports change")
        problems.append(
            dict(
                actions=np.array([ports.index(c) for c in rows["choice"]], dtype=int),
                rewards=agent_rewards(rows["reward"].to_numpy(), kind),
                session_start=(
                    (rows["trial_in_session"].to_numpy() == 0) | rows["after_gap"].to_numpy()
                ),
                plus=plus_ports(ports, rows["initiation"].iloc[0]),
            )
        )
    return value_agent.pad_sequences(problems)


class AgentFit(NamedTuple):
    """The value agent fitted to one mouse.

    Attributes:
        mouse (str): Mouse id.
        kind (str): One of ``AGENT_KINDS``.
        parameters (tuple[float, ...]): In ``value_agent.PARAMETER_NAMES`` order.
        log_likelihood (float): Summed log likelihood at the fit.
        n_trials (int): Trials fitted.
    """

    mouse: str
    kind: str
    parameters: tuple
    log_likelihood: float
    n_trials: int


def fit_agent(
    mouse: str,
    mouse_trials: pd.DataFrame,
    kind: str,
    n_starts: int,
    seed: int,
    model_version: int = value_agent.MODEL_VERSION,
):
    """Fit the value agent with *kind* rewards to one mouse's choices.

    Args:
        mouse (str): Mouse id, kept in the result.
        mouse_trials (pd.DataFrame): The mouse's rows of the drifting trial table.
        kind (str): One of ``AGENT_KINDS``.
        n_starts (int): Random starts of the optimiser.
        seed (int): Seed of the starting points.
        model_version (int): ``value_agent.MODEL_VERSION``. Not used in the fit; it is an
            argument so that a disk cache of this function is keyed on the model version, and
            cached fits of an older model are not reused.

    Returns:
        AgentFit: The best start.
    """
    result = value_agent.fit(problem_sequences(mouse_trials, kind), n_starts, seed)
    return AgentFit(
        mouse, kind, tuple(map(float, result.parameters)), result.log_likelihood, result.n_trials
    )


def fits_table(fits: list[AgentFit]) -> pd.DataFrame:
    """The fits as a table, with the per-mouse likelihood comparison of the two agents.

    Args:
        fits (list[AgentFit]): Fits of both kinds for each mouse.

    Returns:
        pd.DataFrame: One row per mouse and kind with ``mouse, kind, n_trials,
        log_likelihood, nll_per_trial``, one column per parameter, and
        ``delta_nll_vs_magnitude``: the agent's negative log likelihood minus that of the same
        mouse's magnitude agent (positive: it fits worse than the magnitude agent). Both agents
        have four parameters, so this is also half the difference in AIC or BIC.
        ``at_bounds`` names the parameters fitted at a bound (comma-separated; empty if none).

    Raises:
        ValueError: If a mouse lacks the magnitude fit, which every comparison is made against.
    """
    without = sorted({f.mouse for f in fits} - {f.mouse for f in fits if f.kind == "magnitude"})
    if without:
        raise ValueError(f"no magnitude-agent fit for mice {without}")
    table = pd.DataFrame(
        [
            dict(
                mouse=f.mouse,
                kind=f.kind,
                n_trials=f.n_trials,
                log_likelihood=f.log_likelihood,
                nll_per_trial=-f.log_likelihood / f.n_trials,
                **dict(zip(value_agent.PARAMETER_NAMES, f.parameters, strict=True)),
                at_bounds=",".join(value_agent.parameters_at_bounds(f.parameters)),
            )
            for f in fits
        ]
    )
    magnitude = table.loc[table["kind"] == "magnitude"].set_index("mouse")["log_likelihood"]
    table["delta_nll_vs_magnitude"] = table["mouse"].map(magnitude) - table["log_likelihood"]
    return table


# ---------------------------------------------------------------------------
# The task
# ---------------------------------------------------------------------------


class TaskRules(NamedTuple):
    """The drifting task as simulated.

    Attributes:
        drift_rate (float): Per-trial probability that a non-zero arm attempts a drift step.
        block_lengths (np.ndarray): Observed complete block lengths, sampled with replacement.
    """

    drift_rate: float
    block_lengths: np.ndarray


def drift_step(magnitudes: dict, drift_rate: float, rng: np.random.Generator) -> dict:
    """One trial of drift, as in ``DriftingBanditTask.drift_arms``.

    Each non-zero arm, with probability *drift_rate*, moves up or down (equally likely) to the
    nearest value in ``REWARD_VALUES`` that no other arm holds; it stays put if there is none.
    Arms are visited in their dict order and each sees the others' updated values.

    Args:
        magnitudes (dict): ``{port: magnitude}`` with one port at 0.
        drift_rate (float): Per-arm attempt probability.
        rng (np.random.Generator): Source of randomness.

    Returns:
        dict: The new magnitudes (*magnitudes* is not changed).
    """
    new = dict(magnitudes)
    low, high = REWARD_VALUES[0], REWARD_VALUES[-1]
    for arm, value in magnitudes.items():
        if value == 0 or rng.random() >= drift_rate:
            continue
        direction = 1 if rng.random() < 0.5 else -1
        occupied = {v for a, v in new.items() if a != arm}
        candidate = new[arm] + direction
        while low <= candidate <= high and candidate in occupied:
            candidate += direction
        if low <= candidate <= high:
            new[arm] = candidate
    return new


def observed_drift_rate(drift_rate: float, rng: np.random.Generator, n_trials: int = 20000):
    """Fraction of trials on which a non-zero arm actually changes, for a given attempt rate.

    Attempts that hit the edge of the value range or find no free value do nothing, so the
    observed rate is below the attempt rate.

    Args:
        drift_rate (float): Attempt probability.
        rng (np.random.Generator): Source of randomness.
        n_trials (int): Trials simulated. Default 20000.

    Returns:
        float: Changes per non-zero arm per trial.
    """
    magnitudes = {"a": 4, "b": 2, "c": 0}
    changes = 0
    for _ in range(n_trials):
        new = drift_step(magnitudes, drift_rate, rng)
        changes += sum(new[a] != magnitudes[a] for a in ("a", "b"))
        magnitudes = new
    return changes / (2 * n_trials)


def calibrate_drift_rate(target: float, seed: int = 0, tolerance: float = 0.002) -> float:
    """The attempt rate whose observed change rate matches *target*, by bisection.

    Args:
        target (float): Observed per-arm change rate in the data, in (0, 1).
        seed (int): Seed of the simulations. Default 0.
        tolerance (float): Stop when the observed rate is this close to *target*.

    Returns:
        float: The attempt rate.

    Raises:
        ValueError: If *target* is not in (0, 1).
    """
    if not 0 < target < 1:
        raise ValueError(f"target drift rate must be in (0, 1), got {target}")
    low, high = target, 1.0
    for _ in range(30):
        middle = (low + high) / 2
        observed = observed_drift_rate(middle, np.random.default_rng(seed))
        if abs(observed - target) < tolerance:
            return middle
        low, high = (middle, high) if observed < target else (low, middle)
    return (low + high) / 2


def block_change(magnitudes: dict, counts: dict, rng: np.random.Generator) -> dict:
    """Zero the non-zero arm chosen most in the block; the old zero arm takes its value.

    Args:
        magnitudes (dict): ``{port: magnitude}`` with one port at 0.
        counts (dict): ``{port: choices in the block that ends}``.
        rng (np.random.Generator): Breaks ties between equally chosen arms.

    Returns:
        dict: The magnitudes of the new block.
    """
    old_zero = next(a for a, v in magnitudes.items() if v == 0)
    candidates = [a for a in magnitudes if a != old_zero]
    most = max(counts.get(a, 0) for a in candidates)
    tied = [a for a in candidates if counts.get(a, 0) == most]
    zeroed = tied[rng.integers(len(tied))]
    return {**magnitudes, zeroed: 0, old_zero: magnitudes[zeroed]}


# ---------------------------------------------------------------------------
# Agent play
# ---------------------------------------------------------------------------


class MouseLayout(NamedTuple):
    """What a simulated problem copies from a mouse's real problem.

    Attributes:
        ports (list[str]): The three offered ports, sorted.
        initiation (str): The initiation port.
        start_magnitudes (dict): ``{port: magnitude}`` on the problem's first trial.
        session_lengths (list[int]): Trials in each session, in order.
    """

    ports: list
    initiation: str
    start_magnitudes: dict
    session_lengths: list


def mouse_layouts(mouse_trials: pd.DataFrame) -> dict:
    """``{problem: MouseLayout}`` of one mouse's real problems.

    Raises:
        ValueError: If the offered ports change within a problem.
    """
    layouts = {}
    for problem, rows in mouse_trials.groupby(mouse_trials["problem"].map(str), sort=False):
        rows = rows.sort_values("trial")
        port_sets = {tuple(sorted(m)) for m in rows["magnitudes"]}
        if len(port_sets) != 1:
            raise ValueError(f"problem {problem}: offered ports change within the problem")
        layouts[rows["problem"].iloc[0]] = MouseLayout(
            ports=list(port_sets.pop()),
            initiation=rows["initiation"].iloc[0],
            start_magnitudes=dict(rows["magnitudes"].iloc[0]),
            session_lengths=rows.groupby("session_index", sort=True).size().tolist(),
        )
    return layouts


def _rank(magnitudes: dict, port: str) -> str:
    order = sorted(magnitudes, key=lambda a: -magnitudes[a])
    return RANKS[order.index(port)]


class SimulatedSession(NamedTuple):
    """One simulated recording.

    Attributes:
        trials (list[dict]): One print payload per trial (see :func:`simulate_problem`).
        start (dict): ``run_start`` snapshot: ``num_blocks`` (1), ``tot_num_blocks`` (blocks
            begun in the problem so far) and ``curr_rew_mag`` at the session start.
        end (dict): ``run_end`` snapshot: the same fields after the last trial, so a block
            change after the last trial shows in the counters and the magnitudes.
    """

    trials: list
    start: dict
    end: dict


def simulate_problem(
    parameters, kind: str, layout: MouseLayout, rules: TaskRules, rng: np.random.Generator
) -> list[SimulatedSession]:
    """One agent's play of one problem, one :class:`SimulatedSession` per session.

    Each trial's payload is in the pyControl print format: ``num_t``, ``num_blocks`` (block
    counter within the session, from 1), ``curr_rew_mag`` (magnitudes offered before the
    choice), ``chosen_tow``, ``c_tows``, ``chosen_rank`` (true-value rank of the chosen port)
    and ``rank_counts`` (choices of each rank so far in the session, this one included).

    Args:
        parameters (Sequence[float]): In ``value_agent.PARAMETER_NAMES`` order.
        kind (str): One of ``AGENT_KINDS``; what the agent learns from.
        layout (MouseLayout): Ports, initiation port, starting values and session lengths.
        rules (TaskRules): Drift attempt rate and block lengths.
        rng (np.random.Generator): Source of all randomness.

    Returns:
        list[SimulatedSession]: The sessions, in order (an empty session has no trials).
    """
    learning_rate = parameters[0]
    ports = list(layout.ports)
    plus = plus_ports(ports, layout.initiation)
    magnitudes = dict(layout.start_magnitudes)
    block_left = int(rng.choice(rules.block_lengths))
    counts: dict = {}
    total_blocks = 1
    q = np.zeros(len(ports))
    sessions = []
    for n_trials in layout.session_lengths:
        start = {"num_blocks": 1, "tot_num_blocks": total_blocks, "curr_rew_mag": dict(magnitudes)}
        previous, block_in_session, trials = None, 1, []
        rank_counts = dict.fromkeys(RANKS, 0)
        for t in range(n_trials):
            p = value_agent.choice_probabilities(q, previous, plus, parameters)
            action = int(rng.choice(len(ports), p=p))
            port = ports[action]
            rank = _rank(magnitudes, port)
            rank_counts[rank] += 1
            trials.append(
                {
                    "num_t": t + 1,
                    "num_blocks": block_in_session,
                    "curr_rew_mag": dict(magnitudes),
                    "chosen_tow": port,
                    "c_tows": list(ports),
                    "chosen_rank": rank,
                    "rank_counts": dict(rank_counts),
                }
            )
            reward = agent_rewards([magnitudes[port]], kind)[0]
            q[action] += learning_rate * (reward - q[action])
            counts[port] = counts.get(port, 0) + 1
            previous = action
            block_left -= 1
            if block_left <= 0:
                # the swap replaces drift on this trial, so the old zero arm takes exactly the
                # zeroed arm's value, as in the real task
                magnitudes = block_change(magnitudes, counts, rng)
                counts, block_left = {}, int(rng.choice(rules.block_lengths))
                block_in_session += 1
                total_blocks += 1
            else:
                magnitudes = drift_step(magnitudes, rules.drift_rate, rng)
        end = {
            "num_blocks": block_in_session,
            "tot_num_blocks": total_blocks,
            "curr_rew_mag": dict(magnitudes),
        }
        sessions.append(SimulatedSession(trials, start, end))
    return sessions


# ---------------------------------------------------------------------------
# Session dicts
# ---------------------------------------------------------------------------


def _frame(session: SimulatedSession, problem_number: int) -> pd.DataFrame:
    """A pyControl-like event log: run_start, one print per trial, run_end."""
    start = {"num_problem": problem_number, **session.start}
    end = {"num_problem": problem_number, **session.end}
    rows = [(0.0, "variable", "run_start", json.dumps(start))]
    rows += [(1.0 + i, "variable", "print", json.dumps(t)) for i, t in enumerate(session.trials)]
    rows.append((2.0 + len(session.trials), "variable", "run_end", json.dumps(end)))
    return pd.DataFrame(rows, columns=["time", "type", "subtype", "content"])


def simulated_sessions(
    mouse: str, problem_number: int, layout: MouseLayout, sessions: list[SimulatedSession]
) -> dict:
    """``{session_key: session dict}`` of a simulated problem, imported like real sessions.

    Each session is extracted with ``extract_session_trials`` and the problem's sessions are
    then reconciled together with ``reconcile_reversals``, as the importer does for real data.
    Session keys are ``ses-<n>_date-2000MMDD`` so that they sort in play order.

    Args:
        mouse (str): Id given to the simulated animal.
        problem_number (int): Problem number written into the sessions.
        layout (MouseLayout): The layout the sessions were simulated with.
        sessions (list[SimulatedSession]): Output of :func:`simulate_problem`; empty sessions
            are skipped.

    Returns:
        dict: Session dicts with the fields the importer adds.
    """
    out = {}
    for n, session in enumerate(sessions, start=1):
        if not session.trials:
            continue
        key = f"ses-{n}_date-2000{(n - 1) // 28 + 1:02d}{(n - 1) % 28 + 1:02d}"
        frame = _frame(session, problem_number)
        record = {
            "df": frame,
            "data": frame,
            "choice_towers": set(layout.ports),
            "initiation_tower": layout.initiation,
            "problem": problem_number,
            "task_name": DRIFTING_TASK_NAME,
        }
        out[key] = extract_session_trials(record, mouse, key)
    return reconcile_reversals({mouse: out})[mouse] if out else out


def simulate_mouse(
    fit: AgentFit, layouts: dict, rules: TaskRules, rng: np.random.Generator, label: str
) -> dict:
    """``{problem: {simulated mouse: sessions}}`` of one fitted agent playing a mouse's problems.

    Args:
        fit (AgentFit): The agent fitted to the mouse.
        layouts (dict): ``{problem: MouseLayout}`` of the mouse (:func:`mouse_layouts`).
        rules (TaskRules): The task rules.
        rng (np.random.Generator): Source of randomness.
        label (str): Id of the simulated animal, e.g. ``"MY_71_L-magnitude-1"``.

    Returns:
        dict: The simulated problems, keyed like the real ones.
    """
    problems = {}
    for k, (problem, layout) in enumerate(layouts.items(), start=1):
        sessions = simulate_problem(fit.parameters, fit.kind, layout, rules, rng)
        problems[problem] = {label: simulated_sessions(label, k, layout, sessions)}
    return problems
