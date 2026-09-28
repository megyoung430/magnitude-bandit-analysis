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

    def test_same_towers_stay_one_session(self):
        a,b=frame(['C1','A1','B3']),frame(['B3','C1','A1'])
        data={'mouse':{'ses-1':{'data':[a,b]}}}
        split_recording_configurations(data)
        self.assertEqual(list(data['mouse']),['ses-1'])


if __name__ == '__main__':unittest.main()
