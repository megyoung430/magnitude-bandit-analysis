import unittest
from src.behavior_analysis.trial_block_summary import build_trial_block_tables, subject_numbers_from_paths


def session(blocks):
    return {'trial': list(range(1,len(blocks)+1)), 'blocks': blocks,
            'trials_in_block': [], 'choices_by_tower': {}, 'choices_by_rank': {},
            'reward_magnitudes_by_tower': {}}


class SummaryTests(unittest.TestCase):
    def test_blocks_continue_across_sessions_and_reset_across_problems(self):
        data = {1: {'mouse': {'ses-1_date-20260901': session([1,1,2]),
                              'ses-2_date-20260902': session([1,1,2,2])}},
                2: {'mouse': {'ses-3_date-20260903': session([1,1])}}}
        sessions,blocks = build_trial_block_tables(data, {'mouse':5})
        self.assertEqual(sessions.trials.tolist(), [3,4,2])
        self.assertEqual(blocks.trials.tolist(), [2,3,2,2])
        self.assertEqual(blocks.edge_block.tolist(), [True,False,True,True])
        self.assertEqual(set(sessions.group), {'sub-05–08'})
        self.assertEqual(sessions.trials.sum(), blocks.trials.sum())

    def test_subject_group_uses_directory_number(self):
        self.assertEqual(subject_numbers_from_paths(['/raw/sub-03_id-MY_X/ses-1/file.tsv']), {'MY_X':3})


if __name__ == '__main__':unittest.main()
