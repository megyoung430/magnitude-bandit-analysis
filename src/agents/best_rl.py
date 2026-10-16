"""Six-parameter Best RL, using full-data likelihood fitting rather than paper Adam.

Predict first, update chosen value with bias, then forget ALL values toward q_init.
The paper gives these operations but does not explicitly settle their ordering.
Rewards retain their supplied units; parameters therefore depend on reward scale.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

# Bump when a change alters what this model computes; comments and refactors don't need a bump.
# Runs that use this model record the version in their saved-run identity.
MODEL_VERSION = 1
PARAMETER_NAMES = ['learning_rate', 'beta', 'kappa', 'bias', 'forgetting', 'q_init']
EPS = 1e-6
DEFAULT_BOUNDS = [(EPS, 1-EPS), (EPS, None), (-1+EPS, 1-EPS),
                  (-1+EPS, 1-EPS), (EPS, 1-EPS), (None, None)]


class BestRLMouse:
    """Simulated agent for the six-parameter Best RL model.

    Each trial the agent picks an arm by softmax over its values plus a bonus on the previous arm;
    then the chosen value is updated and all values decay toward ``q_init``.

    Args:
        num_arms (int): Number of arms.
        learning_rate (float): Value learning rate, in (0, 1).
        beta (float): Softmax inverse temperature, > 0.
        kappa (float): Bonus added to the previous arm's preference, in (-1, 1).
        bias (float): Constant added to the chosen arm's value at each update, in (-1, 1).
        forgetting (float): Per-trial fraction by which every value decays toward ``q_init``, in (0,
            1).
        q_init (float): Initial value of every arm, and the level values decay toward.

    Attributes:
        learning_rate (float): As in Args.
        beta (float): As in Args.
        kappa (float): As in Args.
        bias (float): As in Args.
        forgetting (float): As in Args.
        q_init (float): As in Args.
        q_values (np.ndarray): ``(num_arms,)`` float array of current arm values.

    Raises:
        ValueError: If any parameter is non-finite or outside its range.
    """
    def __init__(self, num_arms, learning_rate, beta, kappa, bias, forgetting, q_init):
        params = [learning_rate, beta, kappa, bias, forgetting, q_init]
        if not np.all(np.isfinite(params)) or not (
            0 < learning_rate < 1 and beta > 0 and -1 < kappa < 1
            and -1 < bias < 1 and 0 < forgetting < 1
        ):
            raise ValueError('Parameters must satisfy the Best RL ranges.')
        for name, value in zip(PARAMETER_NAMES, params):
            setattr(self, name, value)
        self.q_values = np.full(num_arms, q_init, dtype=float)

    def choose_action(self, previous_action):
        preferences = self.q_values.copy()
        if previous_action is not None:
            preferences[previous_action] += self.kappa
        logits = self.beta * preferences
        probabilities = np.exp(logits - logsumexp(logits))
        return np.random.choice(len(logits), p=probabilities)

    def update_q_values(self, action, reward):
        self.q_values[action] += self.learning_rate * (reward-self.q_values[action]) + self.bias
        self.q_values = (1-self.forgetting)*self.q_values + self.forgetting*self.q_init


def sequence_log_probabilities(params, sequence, return_gradient=False):
    """Per-trial log choice probabilities of one session under Best RL.

    Each row is the prediction made before that trial's choice is seen; values are updated
    afterwards. State starts fresh (``q_init``, no previous action) at the session start.

    Args:
        params (array-like): Six floats in the order of ``PARAMETER_NAMES``: ``learning_rate``,
            ``beta``, ``kappa``, ``bias``, ``forgetting``, ``q_init``.
        sequence (dict): One session as built by ``prepare_cohort_sequences``. Uses ``actions``
            (``(n_trials,)`` int array of chosen-tower indices into ``towers``), ``rewards``
            (``(n_trials,)`` float array of delivered reward pulses) and ``towers`` (list of tower
            names; its length is the number of arms).
        return_gradient (bool): If True also return the gradient of the summed chosen-arm log
            probability. Default False.

    Returns:
        np.ndarray | tuple[np.ndarray, np.ndarray]: ``log_probs``, a ``(n_trials, n_arms)`` array of
            log probabilities over arms. With ``return_gradient`` a tuple ``(log_probs, gradient)``;
            ``gradient`` is a ``(6,)`` array in ``PARAMETER_NAMES`` order.
    """
    learning_rate, beta, kappa, bias, forgetting, q_init = np.asarray(params, dtype=float)
    actions, rewards = sequence['actions'], sequence['rewards']
    n, k = len(actions), len(sequence['towers'])
    q = np.full(k, q_init, dtype=float)
    dq = np.zeros((k, 6))
    dq[:, 5] = 1
    log_probs = np.empty((n, k))
    gradient = np.zeros(6)
    previous = None
    for t, (action, reward) in enumerate(zip(actions, rewards)):
        preferences = q.copy()
        dp = dq.copy()
        if previous is not None:
            preferences[previous] += kappa
            dp[previous, 2] += 1
        logits = beta * preferences
        log_probs[t] = logits - logsumexp(logits)
        if return_gradient:
            dl = beta * dp
            dl[:, 1] += preferences
            residual = np.exp(log_probs[t])
            residual[action] -= 1
            gradient += residual @ dl
        error = reward - q[action]
        dq[action] *= 1-learning_rate
        dq[action, 0] += error
        dq[action, 3] += 1
        q[action] += learning_rate*error + bias
        dq *= 1-forgetting
        dq[:, 4] += q_init-q
        dq[:, 5] += forgetting
        q = (1-forgetting)*q + forgetting*q_init
        previous = action
    return (log_probs, gradient) if return_gradient else log_probs


def multi_session_loss(params, sequences):
    """Trial-weighted mean negative log likelihood over sessions, with its gradient.

    Args:
        params (array-like): Six floats in the order of ``PARAMETER_NAMES``.
        sequences (list[dict]): Sessions as built by ``prepare_cohort_sequences``; each dict needs
            ``actions`` (``(n_trials,)`` int array of indices into ``towers``), ``rewards``
            (``(n_trials,)`` float array) and ``towers``.

    Returns:
        tuple[float, np.ndarray]: ``(loss, gradient)``: the mean cross entropy per trial over all
            sessions, and its gradient with respect to ``params`` (shape ``(n_params,)``).

    Raises:
        ValueError: If the sessions contain no trials.
    """
    total, count, gradient = 0., 0, np.zeros(6)
    for sequence in sequences:
        log_probs, derivative = sequence_log_probabilities(params, sequence, True)
        actions = sequence['actions']
        total -= log_probs[np.arange(len(actions)), actions].sum()
        count += len(actions)
        gradient += derivative
    if not count:
        raise ValueError('No trials available for fitting.')
    return total/count, gradient/count


def fit_mouse_sessions(sequences, bounds=DEFAULT_BOUNDS, n_starts=10, seed=0):
    """Fit Best RL to a mouse's sessions by multi-start L-BFGS-B.

    Starting points are drawn at random (seeded) and clipped into ``bounds``.

    Args:
        sequences (list[dict]): Sessions as built by ``prepare_cohort_sequences``; each dict needs
            ``actions`` (``(n_trials,)`` int array of indices into ``towers``), ``rewards``
            (``(n_trials,)`` float array) and ``towers``.
        bounds (list[tuple]): Six ``(low, high)`` pairs in ``PARAMETER_NAMES`` order; None means
            unbounded. Default ``DEFAULT_BOUNDS``.
        n_starts (int): Number of random starts. Default 10.
        seed (int): Seed of the random generator that draws the starts. Default 0.

    Returns:
        tuple[dict | None, list[dict]]: ``(best, starts)``. ``starts`` has one dict per start with
            keys ``start`` (int), ``success`` (bool), ``loss`` (float, or None if not finite),
            ``parameters`` (list of floats) and ``message`` (str). ``best`` is the successful start
            with the lowest loss, or None if no start succeeded.

    Raises:
        ValueError: If ``n_starts`` < 1 or ``bounds`` does not have six entries.
    """
    if n_starts < 1 or len(bounds) != 6:
        raise ValueError('Need positive n_starts and six parameter bounds.')
    rng = np.random.default_rng(seed)
    starts = []
    for start in range(n_starts):
        initial = [rng.uniform(.05, .95), rng.lognormal(0, .7), rng.uniform(-.8, .8),
                   rng.uniform(-.8, .8), rng.uniform(.01, .5), rng.normal(0, 1)]
        initial = [np.clip(v, -np.inf if lo is None else lo,
                           np.inf if hi is None else hi)
                   for v, (lo, hi) in zip(initial, bounds)]
        result = minimize(multi_session_loss, initial, args=(sequences,), jac=True,
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 2000})
        starts.append(dict(start=start, success=bool(result.success),
                           loss=float(result.fun) if np.isfinite(result.fun) else None,
                           parameters=result.x.tolist(), message=str(result.message)))
    candidates = [s for s in starts if s['success'] and s['loss'] is not None]
    return (min(candidates, key=lambda s: s['loss']) if candidates else None), starts
