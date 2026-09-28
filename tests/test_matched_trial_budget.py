import unittest
from test_reversal_boundaries import session, snapshot
from src.behavior_analysis.matched_trial_budget import match_trial_budget
from src.behavior_analysis.get_total_reversals import get_total_reversals


class BudgetTests(unittest.TestCase):
    def test_cutoff_counts_boundary_but_not_later_reversals(self):
        subjects={f'm{i}':{'ses-1':session(snapshot(),snapshot(),good=(0,0),bad=(0,0))} for i in range(5,9)}
        subjects['m1']={'ses-1':session(snapshot(),snapshot(),good=(0,0,1,2),bad=(0,0,0,0))}
        subjects['m2']={'ses-1':session(snapshot(),snapshot(),good=(0,),bad=(0,))}
        matched,report=match_trial_budget(subjects,{m:int(m[1:]) for m in subjects})
        self.assertEqual(report['trial_budget'],2)
        self.assertEqual(len(matched['m1']['ses-1']['trial']),2)
        self.assertEqual(get_total_reversals(matched['m1'])['good_reversals'],1)
        self.assertEqual(len(subjects['m1']['ses-1']['trial']),4)
        self.assertTrue(next(r for r in report['mice'] if r['mouse']=='m2')['below_budget'])

    def test_whole_session_preserves_terminal_total(self):
        subjects={f'm{i}':{'ses-1':session(snapshot(),snapshot())} for i in range(5,9)}
        subjects['m5']['ses-1']['reversal_totals']={'good':0,'bad':1}
        result,report=match_trial_budget(subjects,{m:int(m[1:]) for m in subjects})
        self.assertEqual(get_total_reversals(result['m5'])['bad_reversals'],1)
        with self.assertRaises(ValueError):match_trial_budget({'m5':subjects['m5']},{'m5':5})

    def test_fixed_budget_needs_no_reference_group(self):
        subjects={'m1':{'ses-1':session(snapshot(),snapshot())}}
        result,report=match_trial_budget(subjects,{'m1':1},trial_budget=510)
        self.assertEqual(report['trial_budget'],510)
        self.assertIsNone(report['reference_mean'])
        self.assertEqual(report['reference_trial_counts'],{})
        self.assertEqual(len(result['m1']['ses-1']['trial']),2)


if __name__ == '__main__':unittest.main()
