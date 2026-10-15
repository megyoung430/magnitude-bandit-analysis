import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.optimize._numdiff import approx_derivative
from src.agents import paper_rl_fitting as paper
from src.agents.best_rl import sequence_log_probabilities as best_predictions
from src.agents.cohort_fitting import sequence_log_probabilities as sticky_predictions


def sequences():
    return [dict(mouse=f'm{i}', problem=5+j, session=f'ses-{j}', towers=['A', 'B', 'C'],
                 actions=np.array([0, 0, 1, 2, 1, 0][:6-j], dtype=int),
                 rewards=np.array([1., 2., 4., 0., 1., 3.][:6-j]))
            for i in range(10) for j in range(2)]


class PaperSplitTests(unittest.TestCase):
    def test_split_is_disjoint_reproducible_and_auditable(self):
        parts, ids = paper.split_by_mouse(sequences())
        self.assertEqual([len(ids[p]) for p in ('train', 'validation', 'test')], [8, 1, 1])
        self.assertEqual(ids, paper.split_by_mouse(sequences())[1])
        self.assertFalse(set(ids['train']) & set(ids['test']))
        self.assertFalse(set(ids['train']) & set(ids['validation']))
        self.assertFalse(set(ids['validation']) & set(ids['test']))
        self.assertEqual(sum(map(len, parts.values())), 20)
        with self.assertRaises(ValueError):
            paper.split_by_mouse(sequences()[:2])


@unittest.skipUnless(importlib.util.find_spec('jax') and importlib.util.find_spec('optax'),
                     'Optional JAX/Optax dependencies not installed')
class PaperJaxTests(unittest.TestCase):
    def test_likelihood_parity_padding_and_autodiff(self):
        import jax
        import jax.numpy as jnp
        for model, reference in [('sticky_rl', sticky_predictions), ('best_rl', best_predictions)]:
            raw = np.linspace(-.6, .4, len(paper.PARAMETER_NAMES[model]))
            params = np.asarray(paper.transform_parameters(raw, model))
            stats, objective, _ = paper.make_backend(model)
            subset = sequences()[:2]
            packed = paper.pack_sequences(subset)
            nll, _ = stats(raw, *packed)
            expected = [-reference(params, s)[np.arange(len(s['actions'])), s['actions']].sum()
                        for s in subset]
            np.testing.assert_allclose(nll, expected, atol=1e-10)
            padded = list(paper.pack_sequences(subset, max_trials=12))
            # Padded inputs must not enter either the likelihood or state updates.
            padded[1][~padded[2]] = 999
            np.testing.assert_allclose(stats(raw, *padded)[0], expected, atol=1e-10)
            l2 = .03
            value = objective(jnp.asarray(raw), *packed, l2)
            expected_proxy = sum(-np.log((1-1e-5)*np.exp(reference(params, seq)[np.arange(1, len(seq['actions'])), seq['actions'][1:]])+5e-4).sum() for seq in subset)/len(subset)
            self.assertAlmostEqual(float(value), expected_proxy)
            grad = jax.grad(objective)(jnp.asarray(raw), *packed, l2)
            numeric = approx_derivative(lambda x: float(objective(x, *packed, l2)), raw)
            np.testing.assert_allclose(grad, numeric.ravel(), atol=1e-7)

    def test_partial_withholds_test_then_complete_and_test_does_not_select(self):
        data = sequences()
        args = dict(models=('sticky_rl', 'best_rl'), grid=[dict(optimizer_learning_rate=.001, l2_weight=.00001, batch_size=2)],
                    n_seeds=1, max_steps=4, validation_every=2, profile='test')
        with tempfile.TemporaryDirectory() as root:
            partial = paper.run_paper_sweep(data, root, max_new_runs=1, **args)
            self.assertFalse(partial['complete'])
            self.assertEqual(partial['test_sessions'], [])
            self.assertEqual(partial['selected'], [])
            complete = paper.run_paper_sweep(data, root, **args)
            self.assertTrue(complete['complete'])
            self.assertEqual(len(complete['selected']), 2)
            self.assertEqual(len(complete['test_sessions']), 4)
            resumed = paper.run_paper_sweep(data, root, **args)
            self.assertEqual(complete, resumed)
            test_mice = set(complete['split_mice']['test'])
            changed = [dict(s, rewards=np.full_like(s['rewards'], 100)) if s['mouse'] in test_mice else s for s in data]
            different_test = paper.run_paper_sweep(changed, root, **args)
            for before, after in zip(complete['selected'], different_test['selected']):
                self.assertEqual(before['parameters'], after['parameters'])
                self.assertEqual(before['selected_step'], after['selected_step'])
                self.assertEqual(before['validation_cross_entropy'], after['validation_cross_entropy'])

    def test_interrupted_checkpoint_resume_matches_uninterrupted(self):
        args = dict(models=('sticky_rl', 'best_rl'), grid=[dict(optimizer_learning_rate=.001, l2_weight=.00001, batch_size=2)],
                    n_seeds=1, max_steps=4, validation_every=2, profile='test-resume')
        original = paper._save_checkpoint
        def stop_after_save(*params):
            original(*params)
            raise KeyboardInterrupt('Simulate notebook interruption after a checkpoint')
        with tempfile.TemporaryDirectory() as root:
            with patch.object(paper, '_save_checkpoint', side_effect=stop_after_save):
                with self.assertRaises(KeyboardInterrupt):
                    paper.run_paper_sweep(sequences(), Path(root)/'resumed', **args)
            resumed = paper.run_paper_sweep(sequences(), Path(root)/'resumed', **args)
            fresh = paper.run_paper_sweep(sequences(), Path(root)/'fresh', **args)
            self.assertEqual(resumed['selected'], fresh['selected'])
            self.assertEqual(resumed['test_sessions'], fresh['test_sessions'])


if __name__ == '__main__':
    unittest.main()


class PaperFinishSweepOrderTests(unittest.TestCase):
    def test_earlier_winner_is_saved_before_a_later_selection_fails(self):
        # A later model whose record has no validation score must not stop the earlier
        # model's winner from being picked, scored and saved first.
        records = [dict(model='sticky_rl', best_validation=0.5, best_raw=[0.1, 0.2]),
                   dict(model='best_rl', best_validation=0.4, best_raw=[0.3]),
                   dict(model='best_rl', best_validation=None, best_raw=[0.5])]
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(paper, 'restore_raw', side_effect=np.asarray), \
                patch.object(paper, '_score_winner', return_value=({}, [])):
            output = Path(tmp)
            with self.assertRaises(TypeError):
                paper._finish_sweep(['sticky_rl', 'best_rl'], records, expected=None,
                                    parts=None, output=output, identities=None,
                                    split_details=None)
            self.assertEqual(sorted(p.name for p in output.iterdir()),
                             [paper._selected_parameters_name('sticky_rl')])
