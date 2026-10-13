import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize._numdiff import approx_derivative

from src.agents.cohort_fitting import (
    prepare_cohort_sequences, sequence_log_probabilities,
    multi_session_loss, run_problem_cross_validation,
)


def sequence(problem=5, rewards=None):
    return dict(mouse='mouse', problem=problem, session=f'ses-{problem}',
                towers=['A1', 'B1', 'C1'], actions=np.array([0, 0, 1, 2, 1, 0]),
                rewards=np.array([1., 2., 4., 0., 1., 3.] if rewards is None else rewards))


class CohortFittingTests(unittest.TestCase):
    def test_gradient_and_session_resets(self):
        params = np.array([.12, 1.7, .3])
        seq = sequence()
        loss, gradient = multi_session_loss(params, [seq, seq])
        np.testing.assert_allclose(loss, multi_session_loss(params, [seq])[0])
        numeric = approx_derivative(lambda p: multi_session_loss(p, [seq, seq])[0], params)
        np.testing.assert_allclose(gradient, numeric.ravel(), atol=1e-7)
        np.testing.assert_allclose(np.exp(sequence_log_probabilities(params, seq)[0]), 1 / 3)

    def test_future_rewards_do_not_change_earlier_predictions(self):
        params = [.1, 2., .25]
        original = sequence()
        changed = sequence(rewards=[1., 2., 4., 0., 100., 100.])
        np.testing.assert_allclose(sequence_log_probabilities(params, original)[:5],
                                   sequence_log_probabilities(params, changed)[:5])

    def test_zero_rewards_and_missing_data_audit(self):
        session = dict(choice=['A1', 'B1'], reward_pulses=[0, 4],
                       choice_towers=[['A1', 'B1', 'C1']] * 2,
                       choices_by_tower={k: [] for k in ['A1', 'B1', 'C1']})
        bad = dict(session, reward_pulses=[None, 4])
        sequences, audit = prepare_cohort_sequences(
            {5: {'mouse': {'ses-1': session, 'ses-2': bad}}})
        self.assertEqual(len(sequences), 1)
        np.testing.assert_array_equal(sequences[0]['rewards'], [0, 4])
        self.assertEqual(audit.included.tolist(), [True, False])
        self.assertTrue(audit.iloc[1].reason)

    def test_seven_folds_training_excludes_held_out_and_resume(self):
        seen = []
        def fit(training, bounds, n_starts, seed):
            seen.append({s['problem'] for s in training})
            return dict(parameters=[.1, 2., .25], loss=.8), [dict(success=True)]
        sequences = [sequence(problem) for problem in range(5, 12)]
        with tempfile.TemporaryDirectory() as root, patch(
            'src.agents.cohort_fitting.fit_mouse_sessions', side_effect=fit
        ):
            results, sessions, eligibility, _ = run_problem_cross_validation(sequences, root)
            self.assertEqual(len(results), 7)
            for row, used in zip(results.to_dict('records'), seen):
                self.assertNotIn(row['held_out_problem'], used)
                self.assertEqual(len(used), 6)
            self.assertEqual(results.n_test_trials.sum(), 42)
            run_problem_cross_validation(sequences, root)
            self.assertEqual(len(seen), 7)

    def test_failed_folds_and_missing_mouse_remain_visible(self):
        sequences = [sequence(problem) for problem in range(5, 12)]
        sequences.append(dict(sequence(), mouse='incomplete'))
        with tempfile.TemporaryDirectory() as root, patch(
            'src.agents.cohort_fitting.fit_mouse_sessions', return_value=(None, [dict(success=False)])
        ):
            results, _, eligibility, _ = run_problem_cross_validation(sequences, root)
            self.assertEqual(len(results), 7)
            self.assertFalse(results.success.any())
            self.assertIn('test_cross_entropy', results)
            self.assertFalse(eligibility.set_index('mouse').loc['incomplete', 'eligible'])


if __name__ == '__main__':
    unittest.main()
