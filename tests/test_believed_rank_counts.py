import unittest
from src.behavior_analysis.get_rank_counts_by_good_reversal import get_rank_counts_by_value


def session(choices, rewards):
    return dict(trial=list(range(1, len(choices) + 1)), blocks=[1]*len(choices),
                trials_in_block=list(range(1, len(choices) + 1)),
                choices_by_tower={a: [int(c == a) for c in choices] for a in rewards[0]},
                reward_magnitudes_by_tower={a: [r[a] for r in rewards] for a in rewards[0]},
                choices_by_rank={})


class BelievedRankTests(unittest.TestCase):
    def test_prechoice_beliefs_cross_sessions_and_split_ties(self):
        old = {'A': 4, 'B': 1, 'C': 0}
        changed = {'A': 0, 'B': 4, 'C': 1}
        data = {'mouse': {'ses-1': session(['A','B','C'], [old]*3),
                          'ses-2': session(['A','A','C'], [changed]*3)}}
        windows = {'mouse': [{'trial_window_idx': {'post': [0,1,2]}},
                             {'trial_window_idx': {'post': [3,4,5]}}]}
        result = get_rank_counts_by_value(data, windows, 'believed')['mouse']
        self.assertEqual(result[0]['total'], 0)  # Each arm initially unseen.
        # First A is still believed best; the next A and C tie for second/third.
        self.assertEqual([result[1][r] for r in ('best','second','third')], [1,1,1])
        self.assertEqual(result[1]['total'], 3)

    def test_true_rank_uses_existing_counts(self):
        window = {'choices_by_rank': {r: {'post': v} for r,v in
                  [('best',[1,0]),('second',[0,1]),('third',[0,0])]}}
        result = get_rank_counts_by_value({}, {'mouse':[window]}, 'true')['mouse'][0]
        self.assertEqual(result['total'], 2)
        self.assertEqual(result['best_prop'], .5)


if __name__ == '__main__':
    unittest.main()
