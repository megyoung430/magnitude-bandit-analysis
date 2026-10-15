import json
import unittest

import pandas as pd

from src.behavior_import.extract_trials import (
    extract_trials_grouped_by_problem,
    unpack_reward_magnitudes,
)


def recording(trials):
    return pd.DataFrame([
        dict(time=i, type='variable', subtype='print', content=json.dumps(trial))
        for i, trial in enumerate(trials)
    ])


class MissingMagnitudeTests(unittest.TestCase):
    def test_missing_values_preserve_trial_alignment_and_later_towers(self):
        session = {'reward_magnitudes': [None, {'A1': 4}, {}, {'A1': 2, 'B1': 1}]}
        unpack_reward_magnitudes(session)
        self.assertEqual(session['reward_magnitudes_by_tower'], {
            'A1': [None, 4, None, 2], 'B1': [None, None, None, 1],
        })

    def test_empty_and_all_missing_magnitudes(self):
        for values in ([], [None, None]):
            self.assertEqual(unpack_reward_magnitudes({'reward_magnitudes': values})
                             ['reward_magnitudes_by_tower'], {})

    def test_pretraining_does_not_become_a_problem(self):
        training = recording([{'num_trials': 1, 'chosen_tower': 'B1',
                               'num_reward_pulses': 4}])
        bandit = recording([{
            'num_trials': 1, 'num_blocks': 1, 'chosen_tower': 'A1',
            'current_reward_magnitudes': {'A1': 4, 'B1': 1},
            'current_choice_towers': ['A1', 'B1'],
            'chosen_rank': 'best', 'rank_counts': {'best': 1, 'second': 0},
        }])
        evidence = {
            'starts_found_ports_false': True,
            'has_port_finding_trials': False, 'task_name': None,
        }
        data = {'mouse': {
            'ses-1': {'data': training, 'settings_evidence': evidence},
            'ses-2': {'data': bandit, 'settings_evidence': evidence},
        }}
        problems = extract_trials_grouped_by_problem(data)
        self.assertEqual(list(problems), [1])
        self.assertEqual(list(problems[1]['mouse']), ['ses-2'])
        self.assertEqual(problems[1]['mouse']['ses-2']['chosen_magnitude'], [4])


if __name__ == '__main__':
    unittest.main()
