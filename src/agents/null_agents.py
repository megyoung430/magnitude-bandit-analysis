"""Soft win-stay/lose-shift agents: negative controls for value-guided choice.

Each agent decides in two steps on every trial of a session after the first. **Stay or switch**:
the probability of repeating the previous choice depends only on what that choice paid.
**Where to**: after a switch, a choice between the two other arms. None of them looks at the
magnitudes the other arms offer, so none can show value-guided switching; what they show in the
analyses of the choice logit is what outcome-dependent staying alone produces.

- ``"wsls"`` (:class:`WslsParams`): P(stay) = sigmoid(``a_win``) after any reward (magnitude is
  ignored) and sigmoid(``a_lose``) after none; a switch goes to either other arm with probability
  1/2.
- ``"wsls_recency"`` (:class:`WslsRecencyParams`): the same stay rule; on a switch the first other
  arm (in sorted port order) is chosen with probability sigmoid(``w`` * (h_first - h_second)),
  where ``h`` is the arm's choice-history trace.
- ``"wsls_magnitude"`` (:class:`WslsMagnitudeParams`): after a reward of ``m`` pulses P(stay) =
  sigmoid(``a_win`` + ``b`` * (m - 1)), so magnitude grades staying (``b`` = 0 is plain WSLS);
  after none sigmoid(``a_lose``). Switches follow the recency rule. Magnitude acts through the
  agent's own last outcome only, never through the other arms' values.
- ``"wsls_graded"`` (:class:`WslsGradedParams`): P(stay) = sigmoid(``w0`` + ``w_m`` * m), one linear
  term in the pulses ``m`` the previous choice paid (0 when unrewarded, so ``w0`` is the stay
  logit after none; ``w0`` = 0 fixes P(stay) after none to 1/2). A switch goes with probability
  ``p_recent`` to whichever of the two other arms was chosen most recently in the *problem* (the
  memory runs on across sessions and resets at a problem start, as the switch regression's
  trials-since-chosen does) and otherwise to the other one, the third arm; if neither was chosen
  yet in the problem it is 50/50. It keeps no decaying trace.
- ``"wsls_habit"`` (:class:`WslsHabitParams`): WSLS (Magnitude) whose stay logit also grows with
  the *choice kernel* of the arm just chosen, ``w_h * K``, so staying builds up over a run of
  consecutive choices (a habit) instead of depending on the last outcome alone. The kernel is
  that of :mod:`src.agents.kernel_q_agent` (``K <- K + alpha_k (1 - K)`` for the chosen arm,
  ``K <- (1 - alpha_k) K`` for the others, cleared at a segment start), evaluated after the
  previous trial's update. Switches follow the recency rule, unchanged. ``w_h = 0`` is WSLS
  (Magnitude).

**History trace.** ``h_a(t)`` is the exponentially decaying count of past choices of arm ``a``,
``h_a(t) = sum_k decay**(k-1) * [arm chosen on trial t-k = a]``. A count (rather than "trials
since last chosen") says how much an arm was visited, which is what "ports that were good were
chosen a lot" leaves behind. The default decay 0.9 (half-life about 7 trials) is the default of
the choice logit's reward tail; blocks last 30-35 trials, so it keeps the recent block's ports
ahead of the previous block's. The decay is fixed, not fitted: ``w`` alone sets how strongly the
trace guides a switch. The choice logit's choice-history baseline has five choice lags and no
tail, so a trace with decay 0.9 (most of it beyond five lags) gives above-baseline switch targets
in the logit's analyses with no value guidance; a fast trace (decay 0.5) does not.

**Session starts.** The first trial of a session, and the first trial after a dropped trial,
has no previous choice: it is uniform over the three arms, and the trace is cleared, as the
choice history of :mod:`src.behavior_analysis.choice_logit` is. A pair of consecutive trials
within such a segment is one stay/switch decision. There is no tower bias: a first choice and a
switch target are uniform (or trace-guided), so the agents have no preference between ports.

Everything here is pure; fits take the drifting trial table
(:func:`src.behavior_analysis.drifting_trial_table.drifting_trial_table`) and are per mouse.
"""

from dataclasses import dataclass, fields
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import lfilter
from scipy.special import expit, log_expit

N_ARMS = 3
KINDS = ("wsls", "wsls_recency", "wsls_magnitude", "wsls_graded", "wsls_habit")
DEFAULT_DECAY = 0.9
# Candidate rates of the habit kernel (time constants 1 / alpha_k of 2 to 50 trials). The rate
# enters the likelihood non-linearly and the stay rule is a logistic regression at each fixed
# rate, so the fit takes the best rate of this grid on the training trials.
HABIT_RATES = (0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.5)
# Stay fractions are clipped to [CLIP, 1 - CLIP] before the logit, so a mouse that never (or
# always) stays gets a finite, large logit (about 5.3) instead of an infinite one.
CLIP = 0.005
MIN_PAIRS = 20  # rewarded and unrewarded stay/switch pairs each, to fit the stay rule
MIN_SWITCHES = 30  # switches with a trace difference, to fit the switch weight
LOGIT_BOUND = 20.0  # a fitted logit at this bound means the data separate perfectly

# The two arms other than arm p, in sorted order: OTHERS[p]
OTHERS = np.array([[1, 2], [0, 2], [0, 1]])


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


def _check_parameters(params, decay: float) -> None:
    for field in fields(params):
        value = getattr(params, field.name)
        try:
            finite = bool(np.isfinite(float(value)))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            raise ValueError(f"{field.name} must be a finite number, got {value!r}")
    if not 0 < decay < 1:
        raise ValueError(f"decay must be in (0, 1), got {decay}")


@dataclass(frozen=True)
class WslsParams:
    """Plain soft WSLS: magnitude ignored, uniform switches.

    Attributes:
        a_win (float): Stay logit after a reward of any size.
        a_lose (float): Stay logit after no reward.
    """

    a_win: float
    a_lose: float

    def __post_init__(self):
        _check_parameters(self, self.decay)

    @property
    def switch_weight(self) -> float:
        """Weight of the trace difference on a switch; 0 means a uniform switch."""
        return 0.0

    @property
    def decay(self) -> float:
        """Decay of the history trace (unused here: the weight is 0)."""
        return DEFAULT_DECAY

    def stay_logit(self, magnitude) -> np.ndarray:
        """Logit of staying after a choice that paid *magnitude* pulses (scalar or array)."""
        return np.where(np.asarray(magnitude) > 0, self.a_win, self.a_lose).astype(float)


@dataclass(frozen=True)
class WslsRecencyParams:
    """Soft WSLS whose switches favor the recently chosen arm.

    Attributes:
        a_win (float): Stay logit after a reward of any size.
        a_lose (float): Stay logit after no reward.
        w (float): Weight of the trace difference on a switch.
        decay (float): Decay of the history trace, in (0, 1). Default ``DEFAULT_DECAY``.
    """

    a_win: float
    a_lose: float
    w: float
    decay: float = DEFAULT_DECAY

    def __post_init__(self):
        _check_parameters(self, self.decay)

    @property
    def switch_weight(self) -> float:
        return self.w

    def stay_logit(self, magnitude) -> np.ndarray:
        return WslsParams(self.a_win, self.a_lose).stay_logit(magnitude)


@dataclass(frozen=True)
class WslsMagnitudeParams:
    """Soft WSLS whose staying grows with the magnitude just received; recency switches.

    Attributes:
        a_win (float): Stay logit after a reward of 1 pulse.
        a_lose (float): Stay logit after no reward.
        b (float): Change of the stay logit per further pulse.
        w (float): Weight of the trace difference on a switch (0: uniform switches).
        decay (float): Decay of the history trace, in (0, 1). Default ``DEFAULT_DECAY``.
    """

    a_win: float
    a_lose: float
    b: float
    w: float
    decay: float = DEFAULT_DECAY

    def __post_init__(self):
        _check_parameters(self, self.decay)

    @property
    def switch_weight(self) -> float:
        return self.w

    def stay_logit(self, magnitude) -> np.ndarray:
        magnitude = np.asarray(magnitude, dtype=float)
        return np.where(magnitude > 0, self.a_win + self.b * (magnitude - 1), self.a_lose)


@dataclass(frozen=True)
class WslsGradedParams:
    """Soft WSLS with one stay logit linear in the magnitude; switches to the recent arm.

    Attributes:
        w0 (float): Stay logit after no reward (the intercept).
        w_m (float): Change of the stay logit per pulse just received.
        p_recent (float): Probability that a switch goes to the more recently chosen of the two
            other arms (in [0, 1]); 1 - p_recent to the other. 50/50 if neither was chosen yet.
    """

    w0: float
    w_m: float
    p_recent: float

    def __post_init__(self):
        _check_parameters(self, DEFAULT_DECAY)
        if not 0 <= self.p_recent <= 1:
            raise ValueError(f"p_recent must be in [0, 1], got {self.p_recent}")

    @property
    def decay(self) -> float:
        """Decay of the history trace (unused here: the agent keeps no trace)."""
        return DEFAULT_DECAY

    def stay_logit(self, magnitude) -> np.ndarray:
        return self.w0 + self.w_m * np.asarray(magnitude, dtype=float)

    def to_first(self, recent) -> np.ndarray:
        """Probability that a switch goes to the first other arm, given who is more recent.

        Args:
            recent: ``+1`` (the first other arm was chosen more recently), ``-1`` (the second)
                or ``0`` (neither was chosen yet in the problem); scalar or array.
        """
        recent = np.asarray(recent)
        return np.where(recent == 0, 0.5, np.where(recent > 0, self.p_recent, 1 - self.p_recent))


@dataclass(frozen=True)
class WslsHabitParams:
    """WSLS (Magnitude) with a habit: staying also grows with the current arm's choice kernel.

    Attributes:
        a_win (float): Stay logit after a reward of 1 pulse, with no habit.
        a_lose (float): Stay logit after no reward, with no habit.
        b (float): Change of the stay logit per further pulse.
        w (float): Weight of the trace difference on a switch (0: uniform switches).
        w_h (float): Weight of the chosen arm's choice kernel on the stay logit (0: none).
        alpha_k (float): Rate of the choice kernel, in (0, 1].
        decay (float): Decay of the switch trace, in (0, 1). Default ``DEFAULT_DECAY``.
    """

    a_win: float
    a_lose: float
    b: float
    w: float
    w_h: float
    alpha_k: float
    decay: float = DEFAULT_DECAY

    def __post_init__(self):
        _check_parameters(self, self.decay)
        if not 0 < self.alpha_k <= 1:
            raise ValueError(f"alpha_k must be in (0, 1], got {self.alpha_k}")

    @property
    def switch_weight(self) -> float:
        return self.w

    def stay_logit(self, magnitude, kernel=0.0) -> np.ndarray:
        """Logit of staying after *magnitude* pulses, with the chosen arm's *kernel* (scalar or
        array)."""
        magnitude = np.asarray(magnitude, dtype=float)
        base = np.where(magnitude > 0, self.a_win + self.b * (magnitude - 1), self.a_lose)
        return base + self.w_h * np.asarray(kernel, dtype=float)


AGENT_PARAMS = {
    "wsls": WslsParams,
    "wsls_recency": WslsRecencyParams,
    "wsls_magnitude": WslsMagnitudeParams,
    "wsls_graded": WslsGradedParams,
    "wsls_habit": WslsHabitParams,
}


# ---------------------------------------------------------------------------
# Choice rule and policy
# ---------------------------------------------------------------------------


def choice_probabilities(
    params,
    previous: int | None,
    magnitude: float,
    h: np.ndarray,
    recent: int = 0,
    kernel: np.ndarray | None = None,
):
    """The agent's probability of each arm on one trial.

    Args:
        params (WslsParams | WslsRecencyParams | WslsMagnitudeParams | WslsGradedParams): The
            agent.
        previous (int | None): Arm chosen on the previous trial of the segment; ``None`` at a
            segment's first trial.
        magnitude (float): Pulses the previous choice paid.
        h (np.ndarray): ``(3,)`` history trace before this trial.
        recent (int): For the graded agent: ``+1`` if the first other arm was chosen more
            recently in the problem than the second, ``-1`` if the second, ``0`` if neither was
            chosen yet. Ignored by the other agents. Default 0.
        kernel (np.ndarray | None): For the habit agent: ``(3,)`` choice kernel after the previous
            trial. Ignored by the other agents. Default ``None`` (zeros).

    Returns:
        np.ndarray: ``(3,)`` probabilities.
    """
    if previous is None:
        return np.full(N_ARMS, 1 / N_ARMS)
    first, second = OTHERS[previous]
    if isinstance(params, WslsHabitParams):
        habit = 0.0 if kernel is None else kernel[previous]
        stay = float(expit(params.stay_logit(magnitude, habit)))
    else:
        stay = float(expit(params.stay_logit(magnitude)))
    if isinstance(params, WslsGradedParams):
        to_first = float(params.to_first(recent))
    else:
        to_first = float(expit(params.switch_weight * (h[first] - h[second])))
    p = np.zeros(N_ARMS)
    p[previous] = stay
    p[first] = (1 - stay) * to_first
    p[second] = (1 - stay) * (1 - to_first)
    return p


class NullPolicy:
    """A null agent as a :class:`src.agents.drifting_simulation.Policy`.

    Nothing carries over between sessions (the agent has no values); the previous choice and
    the trace are cleared at each session start. The one exception is the graded agent's record
    of when each arm was last chosen: it runs on across sessions and lasts as long as the policy,
    which plays one problem.

    Args:
        params (WslsParams | WslsRecencyParams | WslsMagnitudeParams | WslsGradedParams): The
            agent.
    """

    def __init__(self, params):
        self.params = params
        self.last_chosen = np.full(N_ARMS, -1)  # trial index in the problem, -1: never
        self.n_trials = 0
        self.start_session()

    def start_session(self) -> None:
        self.previous: int | None = None
        self.magnitude = 0.0
        self.trace = np.zeros(N_ARMS)
        self.kernel = np.zeros(N_ARMS)

    def _recent(self) -> int:
        if self.previous is None:
            return 0
        first, second = OTHERS[self.previous]
        return int(np.sign(self.last_chosen[first] - self.last_chosen[second]))

    def probabilities(self) -> np.ndarray:
        return choice_probabilities(
            self.params, self.previous, self.magnitude, self.trace, self._recent(), self.kernel
        )

    def update(self, action: int, magnitude: float) -> None:
        self.trace = self.params.decay * self.trace
        self.trace[action] += 1.0
        if isinstance(self.params, WslsHabitParams):
            chosen = np.arange(N_ARMS) == action
            self.kernel = self.kernel + self.params.alpha_k * (chosen - self.kernel)
        self.previous, self.magnitude = action, magnitude
        self.last_chosen[action] = self.n_trials
        self.n_trials += 1


# ---------------------------------------------------------------------------
# Decisions of one mouse
# ---------------------------------------------------------------------------


class Decisions(NamedTuple):
    """One mouse's trials as stay/switch decisions, rows in trial order within each problem.

    Attributes:
        start (np.ndarray): Bool; the trial has no previous choice (session start or after a
            dropped trial).
        stayed (np.ndarray): Bool; the trial repeats the previous choice (``False`` at starts).
        switched (np.ndarray): Bool; the trial leaves the previous choice.
        prev_magnitude (np.ndarray): Pulses the previous choice paid (NaN at starts).
        delta_h (np.ndarray): On a switch, the trace of the first other arm minus the second's
            (NaN elsewhere).
        chose_first (np.ndarray): Bool; on a switch, the first other arm was chosen.
        recent (np.ndarray): Int; at every pair, ``+1`` if the first other arm was chosen more
            recently in the problem than the second, ``-1`` if the second, ``0`` if neither has
            been chosen yet (0 at starts). The record runs across sessions and gaps and resets at
            a problem start, as ``switch_regression.trials_since_chosen`` does.
    """

    start: np.ndarray
    stayed: np.ndarray
    switched: np.ndarray
    prev_magnitude: np.ndarray
    delta_h: np.ndarray
    chose_first: np.ndarray
    recent: np.ndarray


def _problem_ports(rows: pd.DataFrame) -> list[str]:
    port_sets = {tuple(sorted(m)) for m in rows["magnitudes"]}
    problem = rows["problem"].iloc[0]
    if len(port_sets) != 1 or len(next(iter(port_sets))) != N_ARMS:
        raise ValueError(
            f"problem {problem!r}: expected the same three offered ports on every trial"
        )
    return list(port_sets.pop())


def _traces(chosen: np.ndarray, start: np.ndarray, decay: float) -> np.ndarray:
    """``(n, 3)`` trace before each trial, restarted at every segment start."""
    picked = np.zeros((len(chosen), N_ARMS))
    picked[np.arange(len(chosen)), chosen] = 1.0
    bounds = [*np.flatnonzero(start), len(chosen)]
    h = np.zeros_like(picked)
    for first, last in zip(bounds[:-1], bounds[1:], strict=True):
        # y[t] = decay * y[t-1] + picked[t-1]
        h[first:last] = lfilter([0.0, 1.0], [1.0, -decay], picked[first:last], axis=0)
    return h


def _recent_alternative(chosen: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """``+1``/``-1``/``0``: which of two arms per row was chosen more recently before it.

    Rows are the trials of one problem in order. The last trial an arm was chosen strictly
    before each row (-1: never) is compared between the *first* and *second* arm of the row.
    """
    position = np.arange(len(chosen))
    picked = np.zeros((len(chosen), N_ARMS), dtype=bool)
    picked[position, chosen] = True
    last = np.maximum.accumulate(np.where(picked, position[:, None], -1), axis=0)
    before = np.vstack([np.full((1, N_ARMS), -1), last[:-1]])
    return np.sign(before[position, first] - before[position, second])


def _problem_decisions(rows: pd.DataFrame, decay: float) -> Decisions:
    ports = _problem_ports(rows)
    chosen = np.array([ports.index(c) for c in rows["choice"]], dtype=int)
    reward = rows["reward"].to_numpy(dtype=float)
    start = (rows["trial_in_session"].to_numpy() == 0) | rows["after_gap"].to_numpy(dtype=bool)
    start[0] = True
    previous = np.roll(chosen, 1)
    h = _traces(chosen, start, decay)
    first, second = OTHERS[previous].T
    rows_index = np.arange(len(chosen))
    stayed = ~start & (chosen == previous)
    switched = ~start & ~stayed
    return Decisions(
        start=start,
        stayed=stayed,
        switched=switched,
        prev_magnitude=np.where(start, np.nan, np.roll(reward, 1)),
        delta_h=np.where(switched, h[rows_index, first] - h[rows_index, second], np.nan),
        chose_first=switched & (chosen == first),
        recent=np.where(start, 0, _recent_alternative(chosen, first, second)),
    )


def decisions(trials: pd.DataFrame, decay: float) -> Decisions:
    """The stay/switch decisions of one mouse's trials.

    Args:
        trials (pd.DataFrame): One mouse's rows of the drifting trial table; rows are put in
            trial order within each problem.
        decay (float): Decay of the history trace.

    Returns:
        Decisions: One row per trial, problems in order of first appearance.

    Raises:
        ValueError: If *trials* hold other than one mouse, or a problem does not offer the same
            three ports on every trial.
    """
    if trials.empty:
        raise ValueError("the trial table is empty")
    if trials["mouse"].nunique() != 1:
        raise ValueError(
            f"decisions takes the trials of one mouse, got {trials['mouse'].nunique()} mice"
        )
    parts = [
        _problem_decisions(rows.sort_values("trial"), decay)
        for _, rows in trials.groupby("problem", sort=False)
    ]
    return Decisions(*(np.concatenate(field) for field in zip(*parts, strict=True)))


def _problem_kernel(rows: pd.DataFrame, alpha_k: float) -> np.ndarray:
    ports = _problem_ports(rows)
    chosen = np.array([ports.index(c) for c in rows["choice"]], dtype=int)
    start = (rows["trial_in_session"].to_numpy() == 0) | rows["after_gap"].to_numpy(dtype=bool)
    start[0] = True
    picked = np.zeros((len(chosen), N_ARMS))
    picked[np.arange(len(chosen)), chosen] = 1.0
    bounds = [*np.flatnonzero(start), len(chosen)]
    after = np.zeros_like(picked)
    for first, last in zip(bounds[:-1], bounds[1:], strict=True):
        # K[t] = (1 - alpha_k) K[t-1] + alpha_k picked[t], from 0 at the segment start
        after[first:last] = lfilter([alpha_k], [1.0, -(1.0 - alpha_k)], picked[first:last], axis=0)
    current = np.full(len(chosen), np.nan)
    current[1:] = after[np.arange(len(chosen) - 1), chosen[:-1]]
    return np.where(start, np.nan, current)


def current_kernel(trials: pd.DataFrame, alpha_k: float) -> np.ndarray:
    """The choice kernel of the arm chosen on the previous trial, at every decision.

    The kernel follows ``K <- K + alpha_k (1 - K)`` for the arm chosen and ``K <- (1 - alpha_k) K``
    for the others, from 0 at a segment start, and the value returned for trial ``t`` includes
    the choice on trial ``t - 1``. Rows are aligned with :func:`decisions`.

    Args:
        trials (pd.DataFrame): One mouse's rows of the drifting trial table.
        alpha_k (float): Kernel rate, in (0, 1].

    Returns:
        np.ndarray: One value per trial, NaN at segment starts.

    Raises:
        ValueError: For an *alpha_k* outside (0, 1], an empty table, or a problem that does not
            offer the same three ports on every trial.
    """
    if not 0 < alpha_k <= 1:
        raise ValueError(f"alpha_k must be in (0, 1], got {alpha_k}")
    if trials.empty:
        raise ValueError("the trial table is empty")
    return np.concatenate(
        [
            _problem_kernel(rows.sort_values("trial"), alpha_k)
            for _, rows in trials.groupby("problem", sort=False)
        ]
    )


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------


def log_likelihood(trials: pd.DataFrame, params) -> float:
    """Summed log likelihood of one mouse's choices under a null agent.

    Args:
        trials (pd.DataFrame): One mouse's rows of the drifting trial table.
        params (WslsParams | WslsRecencyParams | WslsMagnitudeParams | WslsGradedParams): The
            agent.

    Returns:
        float: Natural-log likelihood (nats) of every choice.
    """
    d = decisions(trials, params.decay)
    pairs = ~d.start
    if isinstance(params, WslsHabitParams):
        kernel = np.nan_to_num(current_kernel(trials, params.alpha_k))
        stay_logit = params.stay_logit(np.nan_to_num(d.prev_magnitude), kernel)
    else:
        stay_logit = params.stay_logit(np.nan_to_num(d.prev_magnitude))
    if isinstance(params, WslsGradedParams):
        to_first = params.to_first(d.recent)
        with np.errstate(divide="ignore"):  # p_recent of 0 or 1 gives -inf, as it should
            switch_term = np.log(np.where(d.chose_first, to_first, 1 - to_first))
    else:
        sign = np.where(d.chose_first, 1.0, -1.0)
        switch_term = log_expit(sign * params.switch_weight * np.nan_to_num(d.delta_h))
    per_pair = np.where(d.stayed, log_expit(stay_logit), log_expit(-stay_logit) + switch_term)
    return float(d.start.sum() * np.log(1 / N_ARMS) + per_pair[pairs].sum())


def nats_per_choice(trials: pd.DataFrame, params) -> float:
    """Negative log likelihood per choice (nats), see :func:`log_likelihood`."""
    return -log_likelihood(trials, params) / len(trials)


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------


def _clipped_logit(n_stay: int, n_pairs: int) -> float:
    fraction = np.clip(n_stay / n_pairs, CLIP, 1 - CLIP)
    return float(np.log(fraction / (1 - fraction)))


def _require_pairs(d: Decisions, min_pairs: int) -> tuple[np.ndarray, np.ndarray]:
    """Masks of the rewarded and unrewarded pairs; raises if either has fewer than *min_pairs*."""
    pairs = ~d.start
    rewarded = pairs & (np.nan_to_num(d.prev_magnitude) > 0)
    unrewarded = pairs & ~rewarded
    for name, mask in (("rewarded", rewarded), ("unrewarded", unrewarded)):
        if mask.sum() < min_pairs:
            raise ValueError(f"{mask.sum()} {name} pairs; at least {min_pairs} are needed")
    return rewarded, unrewarded


def win_lose_logits(d: Decisions, min_pairs: int = MIN_PAIRS) -> tuple[float, float]:
    """``(a_win, a_lose)``: logits of the stay fractions after rewarded and unrewarded choices.

    This is the maximum-likelihood fit of the plain stay rule; fractions are clipped to
    ``[CLIP, 1 - CLIP]``.

    Raises:
        ValueError: If there are fewer than *min_pairs* rewarded or unrewarded pairs.
    """
    rewarded, unrewarded = _require_pairs(d, min_pairs)
    return (
        _clipped_logit(d.stayed[rewarded].sum(), rewarded.sum()),
        _clipped_logit(d.stayed[unrewarded].sum(), unrewarded.sum()),
    )


def fit_win_curve(magnitudes: np.ndarray, stayed: np.ndarray) -> tuple[float, float]:
    """``(a_win, b)`` of P(stay) = sigmoid(a_win + b * (m - 1)) by maximum likelihood.

    Args:
        magnitudes (np.ndarray): Pulses paid by the previous choice, at least 1 each.
        stayed (np.ndarray): Whether the next trial repeated the choice.

    Returns:
        tuple[float, float]: Intercept at 1 pulse and slope per pulse.

    Raises:
        ValueError: If fewer than two distinct magnitudes occur.
        RuntimeError: If the optimizer does not converge, or a parameter ends at
            ``LOGIT_BOUND`` (the stays are perfectly predicted by magnitude).
    """
    magnitudes = np.asarray(magnitudes, dtype=float)
    stayed = np.asarray(stayed, dtype=float)
    if len(np.unique(magnitudes)) < 2:
        raise ValueError(
            f"{len(np.unique(magnitudes))} distinct rewarded magnitudes; at least 2 are needed "
            "to fit a slope"
        )
    return _fit_stay_logistic(magnitudes - 1.0, stayed, True, "win-curve")


def _fit_stay_logistic(
    x: np.ndarray, stayed: np.ndarray, free_intercept: bool, label: str
) -> tuple[float, float]:
    """``(intercept, slope)`` of P(stay) = sigmoid(intercept + slope * x) by maximum likelihood.

    With ``free_intercept=False`` the intercept is held at 0. Raises ``RuntimeError`` if the
    optimizer does not converge or a free parameter ends at ``LOGIT_BOUND``.
    """

    def nll(theta):
        z = theta[0] + theta[1] * x
        gradient = expit(z) - stayed
        # sum of softplus(z) - stayed * z
        return (np.logaddexp(0, z) - stayed * z).sum(), np.array(
            [gradient.sum(), (gradient * x).sum()]
        )

    bounds = [(-LOGIT_BOUND, LOGIT_BOUND) if free_intercept else (0.0, 0.0)]
    bounds.append((-LOGIT_BOUND, LOGIT_BOUND))
    result = minimize(nll, np.zeros(2), jac=True, method="L-BFGS-B", bounds=bounds)
    if not result.success:
        raise RuntimeError(f"the {label} fit did not converge: {result.message}")
    if np.any(np.abs(result.x) > LOGIT_BOUND - 1e-2):
        raise RuntimeError(f"the {label} fit ended at the bound {LOGIT_BOUND}: {result.x}")
    return float(result.x[0]), float(result.x[1])


def _fit_logistic_design(
    design: np.ndarray, stayed: np.ndarray, label: str
) -> tuple[np.ndarray, float]:
    """Weights and negative log likelihood of P(stay) = sigmoid(design @ weights)."""

    def nll(theta):
        z = design @ theta
        return (np.logaddexp(0, z) - stayed * z).sum(), design.T @ (expit(z) - stayed)

    bounds = [(-LOGIT_BOUND, LOGIT_BOUND)] * design.shape[1]
    result = minimize(nll, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B", bounds=bounds)
    if not result.success:
        raise RuntimeError(f"the {label} fit did not converge: {result.message}")
    if np.any(np.abs(result.x) > LOGIT_BOUND - 1e-2):
        raise RuntimeError(f"the {label} fit ended at the bound {LOGIT_BOUND}: {result.x}")
    return result.x, float(result.fun)


def fit_habit_stay(
    magnitudes: np.ndarray, stayed: np.ndarray, kernel: np.ndarray
) -> tuple[tuple[float, float, float, float], float]:
    """``(a_win, a_lose, b, w_h)`` of the habit agent's stay rule at a fixed kernel, by ML.

    P(stay) = sigmoid(a_win + b * (m - 1) + w_h * K) after a reward of ``m`` pulses and
    sigmoid(a_lose + w_h * K) after none, with ``K`` the chosen arm's kernel.

    Args:
        magnitudes (np.ndarray): Pulses paid by the previous choice (0 when unrewarded).
        stayed (np.ndarray): Whether the next trial repeated the choice.
        kernel (np.ndarray): The chosen arm's kernel at each pair (:func:`current_kernel`).

    Returns:
        tuple[tuple[float, float, float, float], float]: The weights and the negative log
        likelihood of the stays.

    Raises:
        ValueError: If fewer than two distinct rewarded magnitudes occur.
        RuntimeError: If the optimizer does not converge or a weight ends at ``LOGIT_BOUND``.
    """
    magnitudes = np.asarray(magnitudes, dtype=float)
    win = magnitudes > 0
    if len(np.unique(magnitudes[win])) < 2:
        raise ValueError("fewer than two distinct rewarded magnitudes; a magnitude slope needs two")
    design = np.column_stack([win, ~win, win * (magnitudes - 1.0), kernel]).astype(float)
    weights, nll = _fit_logistic_design(design, np.asarray(stayed, dtype=float), "habit stay")
    return tuple(float(w) for w in weights), nll


def fit_stay_line(
    magnitudes: np.ndarray, stayed: np.ndarray, fix_intercept: bool = False
) -> tuple[float, float]:
    """``(w0, w_m)`` of P(stay) = sigmoid(w0 + w_m * m) by maximum likelihood.

    Args:
        magnitudes (np.ndarray): Pulses paid by the previous choice (0 when unrewarded).
        stayed (np.ndarray): Whether the next trial repeated the choice.
        fix_intercept (bool): Hold ``w0`` at 0 (P(stay) = 1/2 after none). Default False.

    Raises:
        ValueError: If fewer than two distinct magnitudes occur.
        RuntimeError: If the optimizer does not converge or a free parameter ends at
            ``LOGIT_BOUND``.
    """
    magnitudes = np.asarray(magnitudes, dtype=float)
    if len(np.unique(magnitudes)) < 2:
        raise ValueError("fewer than two distinct magnitudes; a slope needs two")
    return _fit_stay_logistic(
        magnitudes, np.asarray(stayed, dtype=float), not fix_intercept, "stay-line"
    )


def recent_fraction(
    recent: np.ndarray, chose_first: np.ndarray, min_switches: int = MIN_SWITCHES
) -> float:
    """``p_recent``: the fraction of informative switches that went to the more recent arm.

    A switch is informative if one of the two other arms was chosen more recently in the problem
    (``recent`` is not 0). This is the maximum-likelihood fit; it is clipped to
    ``[CLIP, 1 - CLIP]``.

    Args:
        recent (np.ndarray): ``Decisions.recent`` on each switch.
        chose_first (np.ndarray): Whether the first other arm was chosen on each switch.
        min_switches (int): Informative switches needed. Default ``MIN_SWITCHES``.

    Raises:
        ValueError: If there are fewer than *min_switches* informative switches.
    """
    recent, chose_first = np.asarray(recent), np.asarray(chose_first, dtype=bool)
    informative = recent != 0
    if informative.sum() < min_switches:
        raise ValueError(
            f"{informative.sum()} informative switches (one alternative chosen more recently); "
            f"at least {min_switches} are needed"
        )
    to_recent = chose_first[informative] == (recent[informative] > 0)
    return float(np.clip(to_recent.mean(), CLIP, 1 - CLIP))


def fit_switch_weight(
    delta_h: np.ndarray, chose_first: np.ndarray, min_switches: int = MIN_SWITCHES
) -> float:
    """``w`` of P(first other arm) = sigmoid(w * delta_h) by maximum likelihood.

    Switches with ``delta_h = 0`` carry no information about ``w`` and are not counted.

    Args:
        delta_h (np.ndarray): Trace difference (first minus second other arm) on each switch.
        chose_first (np.ndarray): Whether the first other arm was chosen.
        min_switches (int): Informative switches needed. Default ``MIN_SWITCHES``.

    Returns:
        float: The weight.

    Raises:
        ValueError: If *delta_h* has NaN, or there are fewer than *min_switches* informative
            switches.
        RuntimeError: If the optimizer does not converge, or ``w`` ends at ``LOGIT_BOUND``
            (the switch targets are perfectly predicted by the trace).
    """
    delta_h = np.asarray(delta_h, dtype=float)
    if np.isnan(delta_h).any():
        raise ValueError("delta_h contains NaN; pass only the switches' trace differences")
    informative = delta_h != 0
    x = delta_h[informative]
    sign = np.where(np.asarray(chose_first)[informative], 1.0, -1.0)
    if len(x) < min_switches:
        raise ValueError(
            f"{len(x)} switches with a history difference; at least {min_switches} are needed"
        )
    result = minimize_scalar(
        lambda w: -log_expit(sign * w * x).sum(),
        bounds=(-LOGIT_BOUND, LOGIT_BOUND),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"the switch-weight fit did not converge: {result.message}")
    if abs(result.x) > LOGIT_BOUND - 1e-2:
        raise RuntimeError(f"the switch-weight fit ended at the bound {LOGIT_BOUND}: {result.x}")
    return float(result.x)


def _fit_wsls(d: Decisions, decay: float, min_pairs: int, min_switches: int) -> WslsParams:
    return WslsParams(*win_lose_logits(d, min_pairs))


def _fit_recency(d: Decisions, decay: float, min_pairs: int, min_switches: int):
    a_win, a_lose = win_lose_logits(d, min_pairs)
    w = fit_switch_weight(d.delta_h[d.switched], d.chose_first[d.switched], min_switches)
    return WslsRecencyParams(a_win, a_lose, w, decay)


def _fit_magnitude(d: Decisions, decay: float, min_pairs: int, min_switches: int):
    _, a_lose = win_lose_logits(d, min_pairs)
    rewarded = ~d.start & (np.nan_to_num(d.prev_magnitude) > 0)
    a_win, b = fit_win_curve(d.prev_magnitude[rewarded], d.stayed[rewarded])
    w = fit_switch_weight(d.delta_h[d.switched], d.chose_first[d.switched], min_switches)
    return WslsMagnitudeParams(a_win, a_lose, b, w, decay)


def _fit_graded(
    d: Decisions, decay: float, min_pairs: int, min_switches: int, fix_intercept: bool = False
) -> WslsGradedParams:
    _require_pairs(d, min_pairs)  # the same minimums as the other agents
    pairs = ~d.start
    w0, w_m = fit_stay_line(np.nan_to_num(d.prev_magnitude)[pairs], d.stayed[pairs], fix_intercept)
    p_recent = recent_fraction(d.recent[d.switched], d.chose_first[d.switched], min_switches)
    return WslsGradedParams(w0, w_m, p_recent)


def _fit_habit(rows: pd.DataFrame, d: Decisions, decay: float, min_pairs: int, min_switches: int):
    """The habit agent: the best of ``HABIT_RATES`` by the training likelihood of the stays."""
    _require_pairs(d, min_pairs)
    pairs = ~d.start
    magnitudes, stayed = np.nan_to_num(d.prev_magnitude)[pairs], d.stayed[pairs]
    fits = {
        rate: fit_habit_stay(magnitudes, stayed, current_kernel(rows, rate)[pairs])
        for rate in HABIT_RATES
    }
    rate = min(fits, key=lambda r: fits[r][1])
    a_win, a_lose, b, w_h = fits[rate][0]
    w = fit_switch_weight(d.delta_h[d.switched], d.chose_first[d.switched], min_switches)
    return WslsHabitParams(a_win, a_lose, b, w, w_h, rate, decay)


_FITS = {
    "wsls": _fit_wsls,
    "wsls_recency": _fit_recency,
    "wsls_magnitude": _fit_magnitude,
    "wsls_graded": _fit_graded,
}


def fit_null_agents(
    trial_table: pd.DataFrame,
    kind: str,
    decay: float = DEFAULT_DECAY,
    min_pairs: int = MIN_PAIRS,
    min_switches: int = MIN_SWITCHES,
    fix_intercept: bool = False,
) -> dict:
    """The null agent *kind* fitted to every mouse of *trial_table* by maximum likelihood.

    Two to four parameters per mouse against thousands of choices, so the likelihood is not
    cross-validated.

    Args:
        trial_table (pd.DataFrame): A drifting trial table.
        kind (str): One of ``KINDS``.
        decay (float): Decay of the history trace (fixed, not fitted). Default
            ``DEFAULT_DECAY``.
        min_pairs (int): Rewarded and unrewarded pairs each mouse needs. Default ``MIN_PAIRS``.
        min_switches (int): Switches with a trace difference each mouse needs (recency and
            magnitude agents), or informative switches (graded agent). Default ``MIN_SWITCHES``.
        fix_intercept (bool): For ``"wsls_graded"``: hold the stay intercept ``w0`` at 0, so
            P(stay) after no reward is 1/2. Default False (free). Ignored by the other agents.

    Returns:
        dict: ``{mouse: params}``, mice in sorted order.

    Raises:
        ValueError: For an unknown *kind*, an empty table, a mouse with too few pairs or switches
            (the message names the mouse), or one rewarded magnitude for the magnitude agent.
        RuntimeError: If a fit does not converge or runs to a bound (named by mouse).
    """
    if kind not in KINDS:
        raise ValueError(f"unknown agent {kind!r}; expected one of {KINDS}")
    if trial_table.empty:
        raise ValueError("the trial table is empty")
    fits = {}
    for mouse, rows in sorted(trial_table.groupby("mouse"), key=lambda item: item[0]):
        try:
            d = decisions(rows, decay)
            if kind == "wsls_graded":
                fits[mouse] = _fit_graded(d, decay, min_pairs, min_switches, fix_intercept)
            elif kind == "wsls_habit":
                fits[mouse] = _fit_habit(rows, d, decay, min_pairs, min_switches)
            else:
                fits[mouse] = _FITS[kind](d, decay, min_pairs, min_switches)
        except (ValueError, RuntimeError) as error:
            raise type(error)(f"mouse {mouse}: {error}") from error
    return fits


def fit_wsls(trial_table: pd.DataFrame, min_pairs: int = MIN_PAIRS) -> dict:
    """``{mouse: WslsParams}``, see :func:`fit_null_agents`."""
    return fit_null_agents(trial_table, "wsls", min_pairs=min_pairs)


def fit_wsls_recency(
    trial_table: pd.DataFrame,
    decay: float = DEFAULT_DECAY,
    min_pairs: int = MIN_PAIRS,
    min_switches: int = MIN_SWITCHES,
) -> dict:
    """``{mouse: WslsRecencyParams}``, see :func:`fit_null_agents`."""
    return fit_null_agents(trial_table, "wsls_recency", decay, min_pairs, min_switches)


def fit_wsls_magnitude(
    trial_table: pd.DataFrame,
    decay: float = DEFAULT_DECAY,
    min_pairs: int = MIN_PAIRS,
    min_switches: int = MIN_SWITCHES,
) -> dict:
    """``{mouse: WslsMagnitudeParams}``, see :func:`fit_null_agents`."""
    return fit_null_agents(trial_table, "wsls_magnitude", decay, min_pairs, min_switches)
