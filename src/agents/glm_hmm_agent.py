"""The agent signals of the GLM-HMM and the switching agent that plays them.

Every trial gives each arm a vector of *signals*, computed from the history alone (they do not
depend on a hidden state), so a strategy is a weight vector on them:

====================  ======================================================================
``q``                 the arm's value in the kernel Q-learner of
                      :mod:`src.agents.kernel_q_agent` at the mouse's fitted rates (asymmetric
                      learning, forgetting); starts at 0 in a problem and carries over its sessions
``habit``             the arm's choice kernel, a trace of recent choices at the fitted rate;
                      cleared at a segment start
``stay_win``          1 on the previous arm if it paid, else 0
``stay_magnitude``    (pulses - 1) on the previous arm if it paid, else 0
``stay_zero``         1 on the previous arm if it paid 0, else 0
``switch_trace``      the WSLS choice trace ``h <- 0.9 h + chosen`` on the two arms other than the
                      previous one, centered on their mean; 0 on the previous arm
``plus_bias``         1 for a port in the same row or column as the initiation port
====================  ======================================================================

A *segment* starts at a session's first trial and after a dropped trial; nothing is carried over
a segment start except the values. The first trial of a segment has no previous arm, so its stay
and trace signals are 0.

**Strategies as weights.** A Q-learner is a weight on ``q`` (and the port bias). Win-stay /
lose-shift (Magnitude) is a weight vector too: its stay logit ``a`` against two arms at logit 0
is ``a + log 2`` in a three-way softmax, and its switch rule ``expit(w (h_first - h_second))`` is
the softmax over the two other arms with weight ``w`` on their centered trace. The centering keeps
the arms' common trace level out of the stay-or-switch split; a three-way softmax still scales the
switch mass by ``cosh(w (h_first - h_second) / 2)``, so ``wsls_weights`` reproduces WSLS exactly
when ``w = 0`` and as a close relative otherwise (the simulated "WSLS state" is that relative).
A habit/biased strategy is a strong weight on ``habit`` and on the port bias.

The signals come from one :class:`FeatureTracker`, used both to build features from a trial table
and by :class:`FeaturePolicy`, so a simulated agent sees what the fit sees.
"""

import numpy as np

from src.agents import drifting_simulation as sim
from src.agents import kernel_q_agent as kq
from src.agents import null_agents as na

FEATURES = (
    "q",
    "habit",
    "stay_win",
    "stay_magnitude",
    "stay_zero",
    "switch_trace",
    "plus_bias",
)
INDEX = {name: i for i, name in enumerate(FEATURES)}
# The habit/biased strategy of the simulations: a strong choice kernel and a port bias.
HABIT_WEIGHT = 3.0
HABIT_PLUS_BIAS = 1.0


def check_kernel_q_parameters(params) -> np.ndarray:
    """The seven kernel Q parameters as an array.

    Raises:
        ValueError: For other than seven finite values.
    """
    values = np.asarray(params, dtype=float)
    if values.shape != (len(kq.PARAMETER_NAMES),) or not np.all(np.isfinite(values)):
        raise ValueError(
            f"expected seven finite kernel Q parameters {kq.PARAMETER_NAMES}, got {params!r}"
        )
    return values


class FeatureTracker:
    """The signals of one problem, trial by trial.

    Args:
        kq_params (Sequence[float]): The kernel Q parameters (``kernel_q_agent.PARAMETER_NAMES``
            order) whose rates make ``q`` and ``habit``.
        plus (np.ndarray): ``(3,)`` 1.0 for the plus ports of the problem.
        decay (float): Decay of the WSLS choice trace. Default ``null_agents.DEFAULT_DECAY``.
    """

    def __init__(self, kq_params, plus: np.ndarray, decay: float = na.DEFAULT_DECAY):
        self.params = check_kernel_q_parameters(kq_params)
        self.plus, self.decay = np.asarray(plus, dtype=float), decay
        self.q = np.zeros(len(plus))
        self.start_segment()

    def start_segment(self) -> None:
        """Clear what a segment start clears: the kernel, the trace and the previous choice."""
        self.k = np.zeros_like(self.q)
        self.trace = np.zeros_like(self.q)
        self.previous: int | None = None
        self.last_reward = 0.0

    def features(self) -> np.ndarray:
        """``(3, len(FEATURES))`` signals before the next choice."""
        out = np.zeros((len(self.q), len(FEATURES)))
        out[:, INDEX["q"]] = self.q
        out[:, INDEX["habit"]] = self.k
        out[:, INDEX["plus_bias"]] = self.plus
        if self.previous is not None:
            paid = self.last_reward > 0
            out[self.previous, INDEX["stay_win"]] = float(paid)
            out[self.previous, INDEX["stay_magnitude"]] = (self.last_reward - 1.0) * paid
            out[self.previous, INDEX["stay_zero"]] = float(not paid)
            others = np.arange(len(self.q)) != self.previous
            out[others, INDEX["switch_trace"]] = self.trace[others] - self.trace[others].mean()
        return out

    def update(self, action: int, reward: float) -> None:
        """Advance after choosing *action* and receiving *reward* pulses."""
        self.q, self.k = kq.update_state(self.q, self.k, action, reward, self.params)
        self.trace = self.decay * self.trace
        self.trace[action] += 1.0
        self.previous, self.last_reward = action, float(reward)


# ---------------------------------------------------------------------------
# Strategies as weight vectors
# ---------------------------------------------------------------------------


def value_weights(kq_params, habit: float = 0.0) -> np.ndarray:
    """The kernel Q-learner's choice rule: ``beta`` on ``q``, *habit* on ``habit``, its port bias.

    Args:
        kq_params (Sequence[float]): Kernel Q parameters.
        habit (float): Weight on the choice kernel (the Q-learner's ``beta_k``; 0 is a pure value
            learner). Default 0.
    """
    params = check_kernel_q_parameters(kq_params)
    weights = np.zeros(len(FEATURES))
    weights[INDEX["q"]] = params[kq.INDEX["beta"]]
    weights[INDEX["habit"]] = habit
    weights[INDEX["plus_bias"]] = params[kq.INDEX["plus_bias"]]
    return weights


def wsls_weights(params) -> np.ndarray:
    """WSLS (Magnitude) as weights (the habit of a WSLS (Habit) agent is dropped).

    Args:
        params (na.WslsMagnitudeParams | na.WslsHabitParams): Stay logits ``a_win``, ``a_lose``,
            the per-pulse slope ``b`` and the switch weight ``w``.
    """
    weights = np.zeros(len(FEATURES))
    weights[INDEX["stay_win"]] = params.a_win + np.log(2.0)
    weights[INDEX["stay_magnitude"]] = params.b
    weights[INDEX["stay_zero"]] = params.a_lose + np.log(2.0)
    weights[INDEX["switch_trace"]] = params.w
    return weights


def habit_weights(habit: float = HABIT_WEIGHT, plus_bias: float = HABIT_PLUS_BIAS) -> np.ndarray:
    """A habit/biased strategy: repeat recent choices and prefer the plus ports."""
    weights = np.zeros(len(FEATURES))
    weights[INDEX["habit"]] = habit
    weights[INDEX["plus_bias"]] = plus_bias
    return weights


def blend_weights(state_weights: np.ndarray) -> np.ndarray:
    """A stationary blend: the mean of the strategies' weights, applied on every trial."""
    return np.asarray(state_weights, dtype=float).mean(axis=0)


# ---------------------------------------------------------------------------
# The switching agent
# ---------------------------------------------------------------------------


class FeaturePolicy:
    """A hidden-state agent as a :class:`src.agents.drifting_simulation.Policy`.

    The state is drawn from *initial* at each session start (it never carries over a session)
    and follows *transition* after every trial; the choice is a softmax of the state's weights on
    the signals. With one state it is a stationary agent.

    Args:
        weights (np.ndarray): ``(states, len(FEATURES))``.
        transition (np.ndarray): ``(states, states)`` row-stochastic.
        initial (np.ndarray): ``(states,)`` distribution of a session's first state.
        kq_params (Sequence[float]): Kernel Q parameters behind the signals.
        layout (sim.MouseLayout): Gives the ports and the initiation port.
        rng (np.random.Generator): Draws the states (the task draws the choices itself).

    Raises:
        ValueError: For inconsistent shapes or rows that are not distributions.
    """

    def __init__(self, weights, transition, initial, kq_params, layout, rng):
        self.weights = np.asarray(weights, dtype=float)
        self.transition = np.asarray(transition, dtype=float)
        self.initial = np.asarray(initial, dtype=float)
        k = len(self.initial)
        if self.weights.shape != (k, len(FEATURES)) or self.transition.shape != (k, k):
            raise ValueError("weights, transition and initial disagree on the number of states")
        for probs in (self.initial, self.transition):
            if not np.allclose(probs.sum(axis=-1), 1.0) or np.any(probs < 0):
                raise ValueError("initial and transition rows must be distributions")
        plus = sim.plus_ports(list(layout.ports), layout.initiation)
        self.tracker = FeatureTracker(kq_params, plus)
        self.rng = rng
        self.state = 0
        self._features: list = []
        self._sessions: list = []

    def start_session(self) -> None:
        self.tracker.start_segment()
        self.state = int(self.rng.choice(len(self.initial), p=self.initial))
        self._sessions.append([])

    def probabilities(self) -> np.ndarray:
        logits = self.tracker.features() @ self.weights[self.state]
        p = np.exp(logits - logits.max())
        return p / p.sum()

    def update(self, action: int, magnitude: float) -> None:
        self._features.append(self.tracker.features())
        self._sessions[-1].append(self.state)
        self.tracker.update(action, magnitude)
        self.state = int(self.rng.choice(len(self.initial), p=self.transition[self.state]))

    def logged_features(self) -> np.ndarray:
        """``(trials played, 3, len(FEATURES))`` signals the agent saw."""
        return np.array(self._features)

    def sessions_true_states(self) -> list[np.ndarray]:
        """The state of every trial, one array per session played."""
        return [np.array(s, dtype=int) for s in self._sessions]

    def true_states(self) -> np.ndarray:
        """The state of every trial played, sessions in order."""
        return np.concatenate(self.sessions_true_states())
