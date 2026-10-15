import unittest
import pandas as pd
from src.behavior_import.split_recording_configurations import split_recording_configurations


def frame(arms):
    return pd.DataFrame([{'type':'variable','subtype':'run_start','content':{'curr_rew_mag':dict.fromkeys(arms,1)}}])


class ConfigurationTests(unittest.TestCase):
    def test_changed_towers_split_and_keep_raw_frames(self):
        a,b=frame(['C1','A1','B3']),frame(['C2','A1','B3'])
        data={'mouse':{'ses-213_date-20260725':{'data':[a,b],'df':[a.copy(),b.copy()]}}}
        split_recording_configurations(data)
        keys=list(data['mouse'])
        self.assertEqual(keys,['ses-213_date-20260725_part-01','ses-213_date-20260725_part-02'])
        self.assertIs(data['mouse'][keys[0]]['data'],a)
        self.assertIs(data['mouse'][keys[1]]['data'],b)
        self.assertEqual(data['mouse'][keys[1]]['choice_towers'],{'C2','A1','B3'})
        split_recording_configurations(data)
        self.assertEqual(list(data['mouse']),keys)

    def test_segments_do_not_inherit_the_parents_problem(self):
        a,b=frame(['C1','A1','B3']),frame(['C2','A1','B3'])
        session={'data':[a,b],'df':[a.copy(),b.copy()],'problem':2,'problem_flag':'review_settings','problem_provisional':True}
        data={'mouse':{'ses-1_date-20260101':session}}
        split_recording_configurations(data)
        for segment in data['mouse'].values():
            self.assertNotIn('problem',segment)
            self.assertNotIn('problem_provisional',segment)

    def test_only_the_first_segment_keeps_the_parents_flag(self):
        # import_data numbered the session by its first recording, so only the segment that
        # starts with it continues what it reported.
        a,b=frame(['C1','A1','B3']),frame(['C2','A1','B3'])
        session={'data':[a,b],'df':[a.copy(),b.copy()],'problem':2,'problem_flag':'review_settings'}
        data={'mouse':{'ses-1_date-20260101':session}}
        split_recording_configurations(data)
        first,second=data['mouse'].values()
        self.assertEqual(first['problem_flag'],'review_settings')
        self.assertNotIn('problem_flag',second)

    def test_same_towers_stay_one_session(self):
        a,b=frame(['C1','A1','B3']),frame(['B3','C1','A1'])
        data={'mouse':{'ses-1':{'data':[a,b]}}}
        split_recording_configurations(data)
        self.assertEqual(list(data['mouse']),['ses-1'])

    def test_corrupt_run_start_raises_naming_mouse_session_and_file(self):
        a = frame(['C1','A1','B3'])
        b = pd.DataFrame([{'type':'variable','subtype':'run_start','content':'{not json'}])
        b.attrs['source_path'] = '/raw/sub-01_id-mouse/ses-1/b.tsv'
        data={'mouse':{'ses-1':{'data':[a,b]}}}
        with self.assertRaisesRegex(ValueError, r'mouse ses-1 \(/raw/sub-01_id-mouse/ses-1/b.tsv\).*not valid JSON'):
            split_recording_configurations(data)


if __name__ == '__main__':unittest.main()
