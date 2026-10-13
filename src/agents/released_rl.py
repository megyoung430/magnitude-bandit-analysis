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

PARAMETER_NAMES = ['learning_rate', 'beta', 'kappa', 'forgetting', 'q_init']
DEFAULT_BOUNDS = [(1e-6, 1-1e-6), (0., None), (-1+1e-6, 1-1e-6),
                  (1e-6, 1-1e-6), (None, None)]


def sequence_log_probabilities(params, sequence, return_gradient=False):
    lr, beta, kappa, forgetting, q_init = params
    result = best_rl.sequence_log_probabilities(
        [lr, beta, kappa, 0., forgetting, q_init], sequence, return_gradient)
    if return_gradient:
        lp, gradient = result
        return lp, gradient[[0, 1, 2, 4, 5]]
    return result


def multi_session_loss(params, sequences):
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
