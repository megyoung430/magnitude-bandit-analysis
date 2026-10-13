import importlib.util
import tempfile
import unittest
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from src.agents import released_rl as released
from src.agents import best_rl
from src.agents.cohort_fitting import run_problem_cross_validation
from tests.test_cohort_fitting import sequence


class ReleasedRLTests(unittest.TestCase):
    def test_five_parameter_gradient_and_bias_zero_equivalence(self):
        p=np.array([.2, 1.7, -.3, .1, .6])
        seq=sequence()
        loss, gradient=released.multi_session_loss(p,[seq])
        numeric=approx_derivative(lambda x: released.multi_session_loss(x,[seq])[0],p)
        np.testing.assert_allclose(gradient,numeric.ravel(),atol=1e-7)
        np.testing.assert_allclose(released.sequence_log_probabilities(p,seq),
                                  best_rl.sequence_log_probabilities([.2,1.7,-.3,0.,.1,.6],seq))

    def test_per_mouse_five_parameter_fit(self):
        with tempfile.TemporaryDirectory() as root:
            rows,_,_,_=run_problem_cross_validation([sequence(5),sequence(6)],root,
                model='released_rl',bounds=released.DEFAULT_BOUNDS,problem_numbers=[5,6],n_starts=1)
            self.assertTrue(rows.success.all())
            self.assertTrue(set(released.PARAMETER_NAMES).issubset(rows.columns))
            self.assertNotIn('bias',rows.columns)

    @unittest.skipUnless(importlib.util.find_spec('jax') and importlib.util.find_spec('optax'), 'Optional JAX/Optax missing')
    def test_population_metric_alignment_and_small_batch_cap(self):
        from src.agents import paper_rl_fitting as paper
        from tests.test_paper_rl_fitting import sequences
        data=sequences()
        with tempfile.TemporaryDirectory() as root:
            result=paper.run_paper_sweep(data,root,models=('released_rl',),
                grid=[dict(optimizer_learning_rate=.001,weight_decay=1e-5,batch_size=99)],
                n_seeds=1,max_steps=2,validation_every=2,profile='test')
            self.assertEqual(result['selected'][0]['effective_batch_size'],16)
            for row in result['test_sessions']:
                self.assertEqual(row['n_trials'],row['n_observed_trials']-1)
                self.assertAlmostEqual(row['cross_entropy'],row['nll']/row['n_trials'])
                self.assertLess(row['release_nll'],row['nll'])


if __name__ == '__main__':
    unittest.main()
