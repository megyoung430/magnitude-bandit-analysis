"""Five-parameter CogMod configured by the upstream 'Best RL' example.

Pinned source: google-deepmind/hybrid_rnns_reward_learning, commit
baf38da49fd1c1537affbc3745e961bbf0ae547d, cogmod.py/train_models.ipynb.
fit_bias is requested by the example but UNUSED by CogMod. fit_persev_p=False;
we therefore implement only alpha, beta, transient perseverance, f and Q_init.
The six-parameter written-paper model remains separately available in best_rl.
"""
import numpy as np
from scipy.optimize import minimize
from src.agents import best_rl

# Bump when a change alters what this model computes; comments and refactors don't need a bump.
# Runs that use this model record the version in their saved-run identity.
MODEL_VERSION = 1
PARAMETER_NAMES = ['learning_rate', 'beta', 'kappa', 'forgetting', 'q_init']
DEFAULT_BOUNDS = [(1e-6, 1-1e-6), (0., None), (-1+1e-6, 1-1e-6),
                  (1e-6, 1-1e-6), (None, None)]


def sequence_log_probabilities(params, sequence, return_gradient=False):
    """Per-trial log choice probabilities of one session under the released CogMod model.

    The model is Best RL with the bias fixed to zero; the gradient drops the bias entry.

    Args:
        params (array-like): Five floats in the order of ``PARAMETER_NAMES``: ``learning_rate``,
            ``beta``, ``kappa``, ``forgetting``, ``q_init``.
        sequence (dict): One session as built by ``prepare_cohort_sequences``. Uses ``actions``
            (``(n_trials,)`` int array of chosen-tower indices into ``towers``), ``rewards``
            (``(n_trials,)`` float array of delivered reward pulses) and ``towers`` (list of tower
            names; its length is the number of arms).
        return_gradient (bool): If True also return the gradient of the summed chosen-arm log
            probability. Default False.

    Returns:
        np.ndarray | tuple[np.ndarray, np.ndarray]: ``log_probs``, a ``(n_trials, n_arms)`` array of
            log probabilities over arms. With ``return_gradient`` a tuple ``(log_probs, gradient)``;
            ``gradient`` is a ``(5,)`` array in ``PARAMETER_NAMES`` order.
    """
    lr, beta, kappa, forgetting, q_init = params
    result = best_rl.sequence_log_probabilities(
        [lr, beta, kappa, 0., forgetting, q_init], sequence, return_gradient)
    if return_gradient:
        lp, gradient = result
        return lp, gradient[[0, 1, 2, 4, 5]]
    return result


def multi_session_loss(params, sequences):
    """Trial-weighted mean negative log likelihood over sessions, with its gradient.

    Args:
        params (array-like): Five floats in the order of ``PARAMETER_NAMES``.
        sequences (list[dict]): Sessions as built by ``prepare_cohort_sequences``; each dict needs
            ``actions`` (``(n_trials,)`` int array of indices into ``towers``), ``rewards``
            (``(n_trials,)`` float array) and ``towers``.

    Returns:
        tuple[float, np.ndarray]: ``(loss, gradient)``: the mean cross entropy per trial over all
            sessions, and its gradient with respect to ``params`` (shape ``(n_params,)``).

    Raises:
        ValueError: If the sessions contain no trials.
    """
    total, count, gradient = 0., 0, np.zeros(5)
    for s in sequences:
        lp, derivative = sequence_log_probabilities(params, s, True)
        actions = s['actions']
        total -= lp[np.arange(len(actions)), actions].sum()
        count += len(actions)
        gradient += derivative
    if not count:
        raise ValueError('No trials available for fitting.')
    return total/count, gradient/count


def fit_mouse_sessions(sequences, bounds=DEFAULT_BOUNDS, n_starts=10, seed=0):
    """Fit the released CogMod model to a mouse's sessions by multi-start L-BFGS-B.

    Starting points are drawn at random (seeded) and clipped into ``bounds``.

    Args:
        sequences (list[dict]): Sessions as built by ``prepare_cohort_sequences``; each dict needs
            ``actions`` (``(n_trials,)`` int array of indices into ``towers``), ``rewards``
            (``(n_trials,)`` float array) and ``towers``.
        bounds (list[tuple]): Five ``(low, high)`` pairs in ``PARAMETER_NAMES`` order; None means
            unbounded. Default ``DEFAULT_BOUNDS``.
        n_starts (int): Number of random starts. Default 10.
        seed (int): Seed of the random generator that draws the starts. Default 0.

    Returns:
        tuple[dict | None, list[dict]]: ``(best, starts)``. ``starts`` has one dict per start with
            keys ``start`` (int), ``success`` (bool), ``loss`` (float, or None if not finite),
            ``parameters`` (list of floats) and ``message`` (str). ``best`` is the successful start
            with the lowest loss, or None if no start succeeded.

    Raises:
        ValueError: If ``n_starts`` < 1 or ``bounds`` does not have five entries.
    """
    if n_starts < 1 or len(bounds) != 5:
        raise ValueError('Need positive n_starts and five bounds.')
    rng = np.random.default_rng(seed)
    starts = []
    for start in range(n_starts):
        initial = [rng.uniform(.05, .95), rng.lognormal(), rng.uniform(-.8, .8),
                   rng.uniform(.01, .5), rng.normal(1, 1)]
        initial = [np.clip(v, -np.inf if lo is None else lo, np.inf if hi is None else hi)
                   for v, (lo, hi) in zip(initial, bounds)]
        result = minimize(multi_session_loss, initial, args=(sequences,), jac=True,
                          method='L-BFGS-B', bounds=bounds, options={'maxiter': 2000})
        starts.append(dict(start=start, success=bool(result.success),
                           loss=float(result.fun) if np.isfinite(result.fun) else None,
                           parameters=result.x.tolist(), message=str(result.message)))
    candidates = [s for s in starts if s['success'] and s['loss'] is not None]
    return (min(candidates, key=lambda s: s['loss']) if candidates else None), starts
