"""Heuristic+: a win-stay/lose-shift agent that also looks at what the alternatives last paid.

It is the magnitude-graded win-stay/lose-shift agent of :mod:`src.agents.null_agents`
(``"wsls_magnitude"``) with a richer rule for where a switch goes. Each trial after a segment's
first has two steps:

1. **Stay or switch**, as WSLS (Magnitude): after a reward of ``m`` pulses P(stay) =
   sigmoid(``a_win`` + ``b`` (m - 1)); after none sigmoid(``a_lose``).
2. **Where to**, on a switch: the first of the two other arms (sorted port order) is chosen with
   probability sigmoid(``w`` dh + ``v`` db + ``u`` dn), the first arm's value minus the second's of

   - ``h``: the decaying count of past choices of the arm within the segment (the null agents'
     trace), weight ``w``;
   - ``b``: the believed value, the magnitude the arm paid when last chosen in the problem, 0
     before its first visit (the choice logit's ``believed_value``), weight ``v``;
   - ``n``: 1 while the arm has not been chosen in the problem (the logit's
     ``believed_unvisited``), weight ``u``.

With ``v = u = 0`` it is WSLS (Magnitude). The believed value is the only place the alternatives'
magnitudes enter, and only as the last payout at that arm; a value learner, by contrast, blends
all its past payouts at an arm with a learning rate. The history trace's decay is fixed at
``DEFAULT_DECAY`` (0.9, the null agents' decay): the model then nests WSLS (Magnitude), so a
difference between them is attributable to ``v`` and ``u``, and a decay fitted next to ``w`` is
weakly identified. The decay is a field of the parameters so another value can be tried.

Session starts, and the first trial after a dropped trial, have no previous choice: the choice is
uniform over the three arms and the trace is cleared; believed values carry over a problem's
sessions. Everything here is pure; fits take the drifting trial table
(:func:`src.behavior_analysis.drifting_trial_table.drifting_trial_table`) of one mouse.
"""

import warnings
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.special import expit, log_expit

from src.agents import null_agents as na
from src.behavior_analysis import choice_logit as cl
from src.behavior_analysis import switch_regression as sr

N_ARMS = na.N_ARMS
DEFAULT_DECAY = na.DEFAULT_DECAY
SWITCH_FEATURES = ("trace", "believed", "never_visited")


@dataclass(frozen=True)
class HeuristicPlusParams:
    """The parameters of Heuristic+.

    Attributes:
        a_win (float): Stay logit after a reward of 1 pulse.
        a_lose (float): Stay logit after no reward.
        b (float): Change of the stay logit per further pulse.
        w (float): Weight of the trace difference on a switch.
        v (float): Weight of the believed-value difference (logit per pulse).
        u (float): Weight of the never-visited difference (logit).
        decay (float): Decay of the history trace, in (0, 1). Default ``DEFAULT_DECAY``.
    """

    a_win: float
    a_lose: float
    b: float
    w: float
    v: float
    u: float
    decay: float = DEFAULT_DECAY

    def __post_init__(self):
        for name, value in self.__dict__.items():
            if not np.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number, got {value!r}")
        if not 0 < self.decay < 1:
            raise ValueError(f"decay must be in (0, 1), got {self.decay}")

    @property
    def switch_weights(self) -> np.ndarray:
        """``(w, v, u)``, the weights of ``SWITCH_FEATURES``."""
        return np.array([self.w, self.v, self.u])

    def stay_logit(self, magnitude) -> np.ndarray:
        """Logit of staying after a choice that paid *magnitude* pulses (scalar or array)."""
        stay_rule = na.WslsMagnitudeParams(self.a_win, self.a_lose, self.b, self.w, self.decay)
        return stay_rule.stay_logit(magnitude)


# ---------------------------------------------------------------------------
# Choice rule and policy
# ---------------------------------------------------------------------------


def choice_probabilities(
    params: HeuristicPlusParams,
    previous: int | None,
    magnitude: float,
    h: np.ndarray,
    believed: np.ndarray,
    visited: np.ndarray,
) -> np.ndarray:
    """The agent's probability of each arm on one trial.

    Args:
        params (HeuristicPlusParams): The agent.
        previous (int | None): Arm chosen on the previous trial of the segment; ``None`` at a
            segment's first trial.
        magnitude (float): Pulses the previous choice paid.
        h (np.ndarray): ``(3,)`` history trace before this trial.
        believed (np.ndarray): ``(3,)`` magnitude each arm last paid in the problem (0 if never).
        visited (np.ndarray): ``(3,)`` bool, the arm has been chosen in the problem.

    Returns:
        np.ndarray: ``(3,)`` probabilities.
    """
    if previous is None:
        return np.full(N_ARMS, 1 / N_ARMS)
    first, second = na.OTHERS[previous]
    features = np.array(
        [h[first] - h[second], believed[first] - believed[second],
         float(visited[second]) - float(visited[first])]
    )  # fmt: skip
    stay = float(expit(params.stay_logit(magnitude)))
    to_first = float(expit(params.switch_weights @ features))
    p = np.zeros(N_ARMS)
    p[previous] = stay
    p[first] = (1 - stay) * to_first
    p[second] = (1 - stay) * (1 - to_first)
    return p


class HeuristicPolicy:
    """Heuristic+ as a :class:`src.agents.drifting_simulation.Policy`.

    One policy plays one problem: the believed values and the visited arms carry over its
    sessions; the previous choice and the trace are cleared at each session start.

    Args:
        params (HeuristicPlusParams): The agent.
    """

    def __init__(self, params: HeuristicPlusParams):
        self.params = params
        self.believed = np.zeros(N_ARMS)
        self.visited = np.zeros(N_ARMS, dtype=bool)
        self.start_session()

    def start_session(self) -> None:
        self.previous: int | None = None
        self.magnitude = 0.0
        self.trace = np.zeros(N_ARMS)

    def probabilities(self) -> np.ndarray:
        return choice_probabilities(
            self.params, self.previous, self.magnitude, self.trace, self.believed, self.visited
        )

    def update(self, action: int, magnitude: float) -> None:
        self.trace = self.params.decay * self.trace
        self.trace[action] += 1.0
        self.believed[action] = magnitude
        self.visited[action] = True
        self.previous, self.magnitude = action, magnitude


# ---------------------------------------------------------------------------
# Features and likelihood
# ---------------------------------------------------------------------------


class Features(NamedTuple):
    """One mouse's trials as stay/switch decisions with the switch features.

    Attributes:
        start (np.ndarray): Bool; the trial has no previous choice.
        stayed (np.ndarray): Bool; the trial repeats the previous choice.
        switched (np.ndarray): Bool; the trial leaves the previous choice.
        prev_magnitude (np.ndarray): Pulses the previous choice paid (NaN at starts).
        chose_first (np.ndarray): Bool; on a switch, the first other arm was chosen.
        delta (np.ndarray): ``(n_trials, 3)`` first minus second other arm of the trace, the
            believed value and the never-visited indicator (``SWITCH_FEATURES``); 0 off switches.
    """

    start: np.ndarray
    stayed: np.ndarray
    switched: np.ndarray
    prev_magnitude: np.ndarray
    chose_first: np.ndarray
    delta: np.ndarray


def features(trials: pd.DataFrame, decay: float = DEFAULT_DECAY) -> Features:
    """The stay/switch decisions and switch features of one mouse's trials.

    Args:
        trials (pd.DataFrame): One mouse's rows of the drifting trial table.
        decay (float): Decay of the history trace. Default ``DEFAULT_DECAY``.

    Returns:
        Features: One row per trial, problems in order of first appearance.

    Raises:
        ValueError: As :func:`null_agents.decisions` and ``choice_logit.design_matrix`` (one
            mouse, three offered ports per problem).
    """
    d = na.decisions(trials, decay)
    with warnings.catch_warnings():
        # tower-group warnings concern tower biases, which this agent does not have
        warnings.simplefilter("ignore")
        design = cl.design_matrix(trials, 1, None, extras=(cl.BELIEVED,))
    previous = np.roll(design.chosen, 1)
    if len(d.start) != len(design.chosen) or not np.array_equal(
        d.stayed, ~d.start & (design.chosen == previous)
    ):
        raise RuntimeError("the stay/switch decisions and the design have different trials")
    believed = design.X[:, :, design.feature_names.index("believed_value")]
    unvisited = design.X[:, :, design.feature_names.index("believed_unvisited")]
    first, second = na.OTHERS[previous].T
    rows = np.arange(len(previous))
    delta = np.stack(
        [
            np.nan_to_num(d.delta_h),
            believed[rows, first] - believed[rows, second],
            unvisited[rows, first] - unvisited[rows, second],
        ],
        axis=1,
    )
    delta[~d.switched] = 0.0
    return Features(d.start, d.stayed, d.switched, d.prev_magnitude, d.chose_first, delta)


def log_likelihood(
    trials: pd.DataFrame, params: HeuristicPlusParams, include_starts: bool = True
) -> float:
    """Summed log likelihood of one mouse's choices under Heuristic+.

    Args:
        trials (pd.DataFrame): One mouse's rows of the drifting trial table.
        params (HeuristicPlusParams): The agent.
        include_starts (bool): Count the segment starts, each ``log(1/3)``, as
            :func:`null_agents.log_likelihood` does. ``False`` scores the stay/switch decisions
            only. Default True.

    Returns:
        float: Natural-log likelihood (nats).
    """
    f = features(trials, params.decay)
    stay_logit = params.stay_logit(np.nan_to_num(f.prev_magnitude))
    sign = np.where(f.chose_first, 1.0, -1.0)
    switch_term = log_expit(sign * (f.delta @ params.switch_weights))
    per_pair = np.where(f.stayed, log_expit(stay_logit), log_expit(-stay_logit) + switch_term)
    total = float(per_pair[~f.start].sum())
    return total + float(f.start.sum() * np.log(1 / N_ARMS)) if include_starts else total


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def fit(
    trials: pd.DataFrame, decay: float = DEFAULT_DECAY, l2: float = cl.DEFAULT_L2
) -> HeuristicPlusParams:
    """Heuristic+ fitted to one mouse's trials by maximum likelihood.

    The stay rule and the switch rule are separate likelihood terms (a switch is conditioned on),
    so the stay rule is the WSLS (Magnitude) fit and the three switch weights are a
    two-alternative logit on the switches with no intercept
    (:func:`switch_regression.fit_switch_logit`). The switch weights carry the same L2 penalty
    as the switch regression (a Gaussian prior with SD ``1/sqrt(l2)``): the never-visited
    indicator differs between the alternatives on a handful of switches per problem and the
    mouse nearly always takes the unvisited arm there, which an unpenalized fit would send to
    infinity.

    Args:
        trials (pd.DataFrame): One mouse's rows of the drifting trial table.
        decay (float): Decay of the history trace (fixed, not fitted). Default
            ``DEFAULT_DECAY``.
        l2 (float): Penalty of the switch weights. Default ``choice_logit.DEFAULT_L2``.

    Returns:
        HeuristicPlusParams: The fitted agent.

    Raises:
        ValueError: If the mouse has too few rewarded or unrewarded pairs, or too few switches
            (the message names the mouse; the limits are those of the null agents' fits).
        RuntimeError: If a fit does not converge or a parameter runs to a bound.
    """
    mouse = trials["mouse"].iloc[0]
    stay_rule = na.fit_null_agents(trials, "wsls_magnitude", decay)[mouse]
    f = features(trials, decay)
    # only the difference of the two alternatives' features enters, so the second is zero
    delta = f.delta[f.switched]
    X = np.stack([delta, np.zeros_like(delta)], axis=1)
    weights = sr.fit_switch_logit(
        X, np.where(f.chose_first[f.switched], 0, 1), SWITCH_FEATURES, l2
    ).beta
    return HeuristicPlusParams(stay_rule.a_win, stay_rule.a_lose, stay_rule.b, *weights, decay)
