"""One mouse's trials as GLM-HMM sessions: signals, trial index and cross-validation folds.

The signals are those of :mod:`src.agents.glm_hmm_agent` (``FEATURES``), computed from the
mouse's history with a kernel Q parameter set, so they are the same for every number of states.
A *session* is one (problem, session) pair; the state process restarts at its first trial.

**Scored trials** are the choices with a previous choice in their segment: a session's first trial
and the first trial after a dropped trial are left out, as in the strategy comparison
(:mod:`src.behavior_analysis.strategy_models`), so held-out numbers are comparable with its.

**Folds** hold out whole sessions (the sessions of a problem are not independent trials of the
mouse's strategy, but a held-out session never trains the model that scores it). Within each
problem the sessions are dealt to the folds in turn, continuing the count across problems, so
every problem is spread over as many folds as it has sessions (up to the number of folds) and the
folds hold about the same number of sessions.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd

from src.agents import drifting_simulation as sim
from src.agents import glm_hmm_agent as agent
from src.behavior_analysis import glm_hmm as hmm

FEATURES = agent.FEATURES
DEFAULT_FOLDS = 5


class SessionData(NamedTuple):
    """One mouse's sessions.

    Attributes:
        sequences (hmm.Sequences): The padded sessions.
        sessions (pd.DataFrame): One row per session (the row number is the session's index in
            *sequences*): ``mouse, problem, session_index, n_trials``.
        trials (pd.DataFrame): One row per trial, sessions in order: ``session, step`` (position in
            *sequences*) and the trial table's ``mouse, problem, session_index, trial,
            trial_in_session``.
    """

    sequences: hmm.Sequences
    sessions: pd.DataFrame
    trials: pd.DataFrame


def _problem_ports(rows: pd.DataFrame) -> list[str]:
    ports = sorted(rows["magnitudes"].iloc[0])
    if any(sorted(m) != ports for m in rows["magnitudes"]):
        raise ValueError(f"problem {rows['problem'].iloc[0]!r}: offered ports change")
    return ports


def _problem_signals(rows: pd.DataFrame, kq_params) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(signals, chosen, segment start)`` of one problem's rows, in trial order."""
    ports = _problem_ports(rows)
    unknown = sorted(set(rows["choice"]) - set(ports))
    if unknown:
        raise ValueError(f"choices {unknown} are not among the offered ports {ports}")
    tracker = agent.FeatureTracker(kq_params, sim.plus_ports(ports, rows["initiation"].iloc[0]))
    chosen = np.array([ports.index(c) for c in rows["choice"]], dtype=int)
    start = (rows["trial_in_session"].to_numpy() == 0) | rows["after_gap"].to_numpy(bool)
    start[0] = True
    signals = np.empty((len(rows), len(ports), len(FEATURES)))
    for i, (arm, reward) in enumerate(zip(chosen, rows["reward"].to_numpy(float), strict=True)):
        if start[i]:
            tracker.start_segment()
        signals[i] = tracker.features()
        tracker.update(int(arm), reward)
    return signals, chosen, start


def session_arrays(trials: pd.DataFrame, kq_params) -> SessionData:
    """The mouse's sessions with their signals.

    Args:
        trials (pd.DataFrame): One mouse's rows of the drifting trial table.
        kq_params (Sequence[float]): Kernel Q parameters (``kernel_q_agent.PARAMETER_NAMES``
            order) whose rates make the value and habit signals.

    Returns:
        SessionData: Sessions ordered by problem (first appearance) and session number.

    Raises:
        ValueError: If *trials* hold other than one mouse, a problem's offered ports change, a
            choice is not an offered port, or *kq_params* are not seven finite values.
    """
    if trials["mouse"].nunique() != 1:
        raise ValueError(f"session_arrays takes one mouse, got {trials['mouse'].nunique()}")
    blocks = []
    for _, rows in trials.groupby(trials["problem"].map(str), sort=False):
        rows = rows.sort_values("trial").reset_index(drop=True)
        signals, chosen, start = _problem_signals(rows, kq_params)
        for _, index in rows.groupby("session_index", sort=True).indices.items():
            blocks.append((rows.iloc[index], signals[index], chosen[index], start[index]))
    n_sessions, n_steps = len(blocks), max(len(b[0]) for b in blocks)
    X = np.zeros((n_sessions, n_steps, hmm.N_ARMS, len(FEATURES)))
    chosen = np.zeros((n_sessions, n_steps), dtype=int)
    mask = np.zeros((n_sessions, n_steps), dtype=bool)
    scored = np.zeros_like(mask)
    for s, (rows, signals, picked, start) in enumerate(blocks):
        n = len(rows)
        X[s, :n], chosen[s, :n], mask[s, :n], scored[s, :n] = signals, picked, True, ~start
    meta_columns = ["mouse", "problem", "session_index", "trial", "trial_in_session"]
    trial_rows = pd.concat([b[0][meta_columns] for b in blocks], ignore_index=True)
    trial_rows.insert(0, "session", np.repeat(np.arange(n_sessions), mask.sum(axis=1)))
    trial_rows.insert(1, "step", np.concatenate([np.arange(len(b[0])) for b in blocks]))
    sessions = pd.DataFrame(
        [
            dict(mouse=b[0]["mouse"].iloc[0], problem=b[0]["problem"].iloc[0],
                 session_index=b[0]["session_index"].iloc[0], n_trials=len(b[0]))
            for b in blocks
        ]
    )  # fmt: skip
    return SessionData(hmm.Sequences(X, chosen, mask, scored), sessions, trial_rows)


def assign_folds(data: SessionData, n_folds: int = DEFAULT_FOLDS) -> np.ndarray:
    """The fold of every session: within each problem the sessions are dealt out in turn.

    Args:
        data (SessionData): The mouse's sessions.
        n_folds (int): Number of folds. Default ``DEFAULT_FOLDS``.

    Returns:
        np.ndarray: ``(sessions,)`` fold numbers ``0 .. n_folds - 1``; deterministic.

    Raises:
        ValueError: With fewer sessions than folds or fewer than two folds.
    """
    n_sessions = len(data.sessions)
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")
    if n_sessions < n_folds:
        raise ValueError(f"{n_sessions} sessions cannot fill {n_folds} folds")
    folds = np.empty(n_sessions, dtype=int)
    counter = 0
    for _, group in data.sessions.groupby(data.sessions["problem"].map(str), sort=False):
        folds[group.index] = (counter + np.arange(len(group))) % n_folds
        counter += len(group)
    return folds


def subset(seqs: hmm.Sequences, sessions: np.ndarray) -> hmm.Sequences:
    """The sessions at the given indices, with the padding trimmed to the longest of them."""
    picked = hmm.Sequences(*(a[sessions] for a in seqs))
    n_steps = int(picked.mask.sum(axis=1).max())
    return hmm.Sequences(*(a[:, :n_steps] for a in picked))
