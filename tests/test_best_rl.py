import tempfile
import unittest
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from src.agents import best_rl
from src.agents.cohort_fitting import (
    sequence_log_probabilities as sticky_predictions, run_problem_cross_validation,
)


def sequence(problem=5):
    return dict(mouse='m', problem=problem, session=f'ses-{problem}',
                towers=['A', 'B', 'C'], actions=np.array([0, 0, 1, 2, 1, 0]),
                rewards=np.array([1., 2., 4., 0., 1., 3.]))


class BestRLTests(unittest.TestCase):
    def test_gradient_and_reset(self):
        p = np.array([.2, 1.7, -.3, .12, .08, .7])
        s = sequence()
        loss, grad = best_rl.multi_session_loss(p, [s, s])
        numeric = approx_derivative(lambda x: best_rl.multi_session_loss(x, [s, s])[0], p)
        np.testing.assert_allclose(grad, numeric.ravel(), atol=1e-7)
        self.assertAlmostEqual(loss, best_rl.multi_session_loss(p, [s])[0])
        np.testing.assert_allclose(np.exp(best_rl.sequence_log_probabilities(p, s)[0]), 1/3)

    def test_reduces_to_existing_model(self):
        # Zero forgetting is used only to check the mathematical limiting case.
        np.testing.assert_allclose(best_rl.sequence_log_probabilities([.1, 2, .25, 0, 0, 0], sequence()),
                                   sticky_predictions([.1, 2, .25], sequence()))

    def test_manual_update_and_simulation_scoring_agree(self):
        p = [.2, 2., -.3, .1, .25, .5]
        mouse = best_rl.BestRLMouse(3, *p)
        mouse.update_q_values(0, 4)
        np.testing.assert_allclose(mouse.q_values, [.75*(.5+.2*3.5+.1)+.25*.5, .5, .5])
        s = sequence()
        mouse = best_rl.BestRLMouse(3, *p)
        log_probs = best_rl.sequence_log_probabilities(p, s)
        previous = None
        for t, (a, r) in enumerate(zip(s['actions'], s['rewards'])):
            logits = mouse.beta*mouse.q_values.copy()
            if previous is not None:
                logits[previous] += mouse.beta*mouse.kappa
            logits -= np.max(logits)
            probs = np.exp(logits)/np.exp(logits).sum()
            np.testing.assert_allclose(np.exp(log_probs[t]), probs)
            mouse.update_q_values(a, r)
            previous = a

    def test_causality_and_bias_initial_value_confound(self):
        p = np.array([.2, 2., -.3, .1, .25, .5])
        s = sequence()
        changed = dict(s, rewards=s['rewards'].copy())
        changed['rewards'][-2:] = 100
        np.testing.assert_allclose(best_rl.sequence_log_probabilities(p, s)[:5],
                                   best_rl.sequence_log_probabilities(p, changed)[:5])
        alternative = p.copy()
        alternative[5] += 1
        alternative[3] += p[0]
        np.testing.assert_allclose(best_rl.sequence_log_probabilities(p, s),
                                   best_rl.sequence_log_probabilities(alternative, s))

    def test_cv_fit_and_resume(self):
        with tempfile.TemporaryDirectory() as root:
            args = dict(problem_numbers=[5, 6], bounds=best_rl.DEFAULT_BOUNDS,
                        n_starts=1, seed=1, model='best_rl')
            rows, sessions, _, output = run_problem_cross_validation([sequence(5), sequence(6)], root, **args)
            self.assertTrue(rows.success.all())
            self.assertEqual(len(rows), 2)
            self.assertTrue(set(best_rl.PARAMETER_NAMES).issubset(rows.columns))
            self.assertEqual(len(sessions), 2)
            resumed, _, _, same_output = run_problem_cross_validation([sequence(5), sequence(6)], root, **args)
            self.assertEqual(output, same_output)
            np.testing.assert_allclose(rows.test_cross_entropy, resumed.test_cross_entropy)


if __name__ == '__main__':
    unittest.main()
