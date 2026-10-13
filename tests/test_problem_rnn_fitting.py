"""Leakage, fold coverage and resumability checks for pooled problem fitting."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from src.agents import paper_rl_fitting as paper


def data():
    return [dict(mouse=f'm{i}', problem=p, session=f's{p}', towers=['A', 'B', 'C'],
                 actions=np.array([0, 1, 2, 1]), rewards=np.array([1., 0., 4., 1.]))
            for i in range(2) for p in (5, 6, 7)]


class ProblemSplitTests(unittest.TestCase):
    def test_whole_problem_disjointness_and_coverage(self):
        rows = data()
        tested = []
        for problem in (5, 6, 7):
            parts, ids = paper.split_by_problem(rows, problem)
            self.assertEqual(ids['test'], [problem])
            self.assertEqual(ids['validation'], [6 if problem == 5 else 7 if problem == 6 else 5])
            self.assertEqual(sum(map(len, parts.values())), len(rows))
            for a, b in [('train', 'test'), ('train', 'validation'), ('test', 'validation')]:
                self.assertFalse(set(ids[a]) & set(ids[b]))
            self.assertEqual({s['mouse'] for s in parts['train']}, {'m0', 'm1'})
            tested.extend((s['mouse'], s['session']) for s in parts['test'])
        self.assertEqual(len(tested), len(set(tested)))
        self.assertEqual(len(tested), len(rows))
        with self.assertRaises(ValueError):
            paper.split_by_problem(rows, 5, 5)
        with self.assertRaises(ValueError):
            paper.split_by_problem([s for s in rows if s['problem'] != 7], 5)
        with self.assertRaises(ValueError):
            paper.split_by_problem(rows+[dict(rows[0], mouse='unseen')], 5)


@unittest.skipUnless(importlib.util.find_spec('jax') and importlib.util.find_spec('optax'), 'Requires JAX')
class ProblemFitTests(unittest.TestCase):
    def test_fold_resume_aggregation_and_test_isolation(self):
        args = dict(models=('memory_ann',),
                    grid=[dict(hidden_units=2, optimizer_learning_rate=.001, weight_decay=.0001, batch_size=2)],
                    n_seeds=1, max_steps=2, validation_every=1, profile='problem-test')
        with tempfile.TemporaryDirectory() as root:
            partial = paper.run_problem_sweep(data(), root, max_new_runs=1, **args)
            self.assertFalse(partial['complete'])
            self.assertEqual(partial['completed_runs'], 1)
            self.assertEqual(partial['test_sessions'], [])
            result = paper.run_problem_sweep(data(), root, **args)
            self.assertTrue(result['complete'])
            self.assertEqual(result['expected_runs'], 3)
            self.assertEqual(len(result['test_sessions']), 6)
            self.assertEqual(result, paper.run_problem_sweep(data(), root, **args))
            for row in result['selected']:
                self.assertTrue(Path(row['parameters_file']).exists())
            altered = [dict(s, rewards=np.full(4, 50.)) if s['problem'] == 5 else s for s in data()]
            changed = paper.run_problem_sweep(altered, root, test_problems=[5], **args)
            before, after = result['folds'][0]['selected'][0], changed['folds'][0]['selected'][0]
            for key in ('setting', 'repeat', 'selected_step', 'validation_cross_entropy'):
                self.assertEqual(before[key], after[key])
            with np.load(Path(result['folds'][0]['output_dir'])/before['parameters_file']) as a, np.load(Path(changed['folds'][0]['output_dir'])/after['parameters_file']) as b:
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])


if __name__ == '__main__':
    unittest.main()
