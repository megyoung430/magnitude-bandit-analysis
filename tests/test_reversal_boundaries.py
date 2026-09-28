import unittest
import pandas as pd

from src.behavior_import.reconcile_reversals import reconcile_reversals
from src.behavior_analysis.get_variables_across_sessions import get_vars_across_all_sessions
from src.behavior_analysis.get_total_reversals import get_total_reversals, get_all_reversal_indices
from src.behavior_analysis.get_good_reversal_info import get_good_reversal_info
from src.behavior_analysis.get_bad_reversal_info import get_bad_reversal_info


def snapshot(good=0, bad=0, tg=0, tb=0, mags=None, problem=1):
    return dict(num_good_rev=good, num_bad_rev=bad, tot_num_good_rev=tg,
                tot_num_bad_rev=tb, curr_rew_mag=mags or {'A': 4, 'B': 1, 'C': 0}, num_problem=problem)


def session(start, end, good=(0, 0), bad=(0, 0)):
    n = len(good)
    mags = start['curr_rew_mag']
    tv = dict(trial=list(range(1, n + 1)), good_reversals=list(good), bad_reversals=list(bad),
              reward_magnitudes=[mags] * n)
    return dict(data=pd.DataFrame([dict(type='variable', subtype='run_start', content=start),
                                  dict(type='variable', subtype='run_end', content=end)]),
                trial_variables=[tv], trial_info=[tv], trial=tv['trial'],
                good_reversals=list(good), bad_reversals=list(bad), has_good=True, has_bad=True,
                blocks=[1] * n, trials_in_block=list(range(1, n + 1)),
                reward_magnitudes_by_tower={k: [v] * n for k, v in mags.items()},
                choices_by_tower={k: [int(k == 'A')] * n for k in mags},
                choices_by_rank={'best': [1] * n})


class BoundaryTests(unittest.TestCase):
    def test_terminal_bad_and_between_session_good_reach_all_consumers(self):
        a = snapshot()
        b = snapshot(bad=1, tb=1, mags={'A': 4, 'B': 0, 'C': 1})
        c = snapshot(tg=1, tb=1, mags={'A': 0, 'B': 4, 'C': 1})
        data = {'mouse': {'ses-1': session(a, b), 'ses-2': session(c, c)}}
        reconcile_reversals(data)
        merged, _ = get_vars_across_all_sessions(data)
        self.assertEqual(merged['mouse']['bad_reversals'], [0, 0, 1, 1])
        self.assertEqual(merged['mouse']['good_reversals'], [0, 0, 1, 1])
        self.assertEqual(get_total_reversals(data['mouse']), dict(total_reversals=2, good_reversals=1, bad_reversals=1))
        self.assertEqual(get_all_reversal_indices(data)[:2], ({'mouse': [2]}, {'mouse': [2]}))
        self.assertEqual(get_good_reversal_info(data)['mouse'][0]['reversal_idx'], 2)
        self.assertEqual(get_bad_reversal_info(data)['mouse'][0]['reversal_idx'], 2)
        reconcile_reversals(data)
        self.assertEqual(get_vars_across_all_sessions(data)[0], merged)

    def test_terminal_event_counted_without_next_trial_and_not_double_counted(self):
        a, b = snapshot(), snapshot(bad=1, tb=1)
        data = {'mouse': {'ses-1': session(a, b)}}
        reconcile_reversals(data)
        self.assertEqual(get_total_reversals(data['mouse'])['bad_reversals'], 1)
        self.assertEqual(get_all_reversal_indices(data)[1]['mouse'], [])
        data['mouse']['ses-2'] = session(snapshot(tb=1), snapshot(tb=1))
        reconcile_reversals(data)
        self.assertEqual(get_total_reversals(data['mouse'])['bad_reversals'], 1)
        self.assertEqual(get_all_reversal_indices(data)[1]['mouse'], [2])

    def test_rollback_magnitudes_and_problem_changes_do_not_create_events(self):
        a = snapshot(tg=9)
        b = snapshot(tg=8, mags={'A': 0, 'B': 1, 'C': 4})
        c = snapshot(tg=20, problem=2)
        data = {'mouse': {'ses-1': session(a, a), 'ses-2': session(b, b), 'ses-3': session(c, c)}}
        reconcile_reversals(data)
        self.assertEqual(get_total_reversals(data['mouse'])['total_reversals'], 0)
        self.assertTrue(data['mouse']['ses-2']['reversal_diagnostics'])

    def test_recording_restart_preserves_terminal_event_and_subtracts_baseline(self):
        a, b = snapshot(), snapshot(bad=1, tb=1)
        first, second = session(a, b), session(b, snapshot(bad=2, tb=2), bad=(1, 2))
        first['data'] = [first['data'], second['data']]
        first['trial_variables'] += second['trial_variables']
        first['trial'] = [1, 2, 3, 4]
        first['blocks'] = [1, 1, 1, 2]
        data = {'mouse': {'ses-1': first}}
        reconcile_reversals(data)
        self.assertEqual(first['bad_reversals'], [0, 0, 1, 2])
        self.assertEqual(first['reversal_totals']['bad'], 2)

    def test_missing_end_uses_start_total_plus_observed_trials(self):
        a, b = snapshot(), snapshot(good=1, tg=1)
        first = session(a, b, good=(0, 1))
        first['data'] = first['data'].iloc[:1]
        data = {'mouse': {'ses-1': first, 'ses-2': session(snapshot(tg=1), snapshot(tg=1))}}
        reconcile_reversals(data)
        self.assertEqual(get_total_reversals(data['mouse'])['good_reversals'], 1)

    def test_boundary_correction_survives_session_subset(self):
        a, b = snapshot(), snapshot(tg=1)
        data = {'mouse': {'ses-1': session(a, a), 'ses-2': session(b, b)}}
        reconcile_reversals(data)
        subset = {'ses-2': data['mouse']['ses-2']}
        self.assertEqual(get_total_reversals(subset)['good_reversals'], 1)
        self.assertEqual(get_vars_across_all_sessions({'mouse': subset})[0]['mouse']['good_reversals'], [1, 1])

    def test_missing_counter_samples_keep_trial_alignment(self):
        a = snapshot()
        data = {'mouse': {'ses-1': session(a, snapshot(good=1, tg=1), good=(0, None, 1), bad=(0, None, 0))}}
        reconcile_reversals(data)
        merged, _ = get_vars_across_all_sessions(data)
        self.assertEqual(merged['mouse']['good_reversals'], [0, 0, 1])
        self.assertEqual(get_all_reversal_indices(data)[0]['mouse'], [2])

    def test_empty_run_terminal_event_carries_forward(self):
        a, b = snapshot(), snapshot(bad=1, tb=1)
        data = {'mouse': {'ses-1': session(a, b, good=(), bad=()),
                          'ses-2': session(snapshot(tb=1), snapshot(tb=1))}}
        reconcile_reversals(data)
        merged, _ = get_vars_across_all_sessions(data)
        self.assertEqual(merged['mouse']['bad_reversals'], [1, 1])
        self.assertEqual(get_total_reversals(data['mouse'])['bad_reversals'], 1)


if __name__ == '__main__':
    unittest.main()
