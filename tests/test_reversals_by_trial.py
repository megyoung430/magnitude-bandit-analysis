import unittest
import numpy as np
from test_reversal_boundaries import session, snapshot
from src.behavior_import.reconcile_reversals import reconcile_reversals
from src.behavior_analysis.reversals_by_trial import reversal_trial_series, trailing_reversal_rate


class TrialReversalTests(unittest.TestCase):
    def test_terminal_event_at_exact_completed_trial_count(self):
        a,b=snapshot(),snapshot(bad=1,tb=1)
        data={'mouse':{'ses-1':session(a,b),'ses-2':session(snapshot(tb=1),snapshot(tb=1))}}
        reconcile_reversals(data)
        values=reversal_trial_series(data)['mouse']['bad']
        np.testing.assert_array_equal(values,[0,0,1,1,1])
        np.testing.assert_allclose(trailing_reversal_rate(values,2),[0,50,50,0])
        one={'mouse':{'ses-1':data['mouse']['ses-1']}}
        np.testing.assert_array_equal(reversal_trial_series(one)['mouse']['bad'],[0,0,1])

    def test_window_uses_actual_exposure_and_does_not_look_ahead(self):
        np.testing.assert_allclose(trailing_reversal_rate([0,0,1,1],100),[0,50,100/3])
        with self.assertRaises(ValueError):trailing_reversal_rate([0,1],0)


if __name__ == '__main__':unittest.main()
