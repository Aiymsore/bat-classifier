"""离线判族的行为测试：连续CF短片段、真CF-FM和混群保护。"""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from bat_classifier.sequence_analysis import SequenceConfig, analyze_sequences


def calls(durations, tails=None, file='one.wav', label='CF3_66K', start=0):
    return pd.DataFrame([dict(segment_file=file, start_s=start+i*.1,
        end_s=start+i*.1+d/1000, gate_status='bat',
        predicted_label=label if d >=17 else 'CFFM1_67K',
        cf_plateau_khz=66., cf_plateau_ms=float(d),
        sequence_reference_eligible=d>=17,
        tail_low_energy_ratio=(tails[i] if tails is not None else .05),
        template_cf_label=label, template_cffm_label='CFFM1_67K', template_cffm_score=.8)
        for i,d in enumerate(durations)])


class SequenceTests(unittest.TestCase):
    def test_long_cf_sequence_corrects_one_short_call_without_mutating_input(self):
        d=calls([30,8,28,31]); old=d.copy(deep=True)
        r,s=analyze_sequences(d)
        self.assertEqual(r.sequence_predicted_label.tolist(), ['CF3_66K']*4)
        self.assertEqual(int(r.sequence_changed.sum()),1)
        pd.testing.assert_frame_equal(d,old)
        pd.testing.assert_frame_equal(r[d.columns],d)

    def test_short_tailed_sequence_stays_cffm(self):
        r,_=analyze_sequences(calls([8,9,10,7],[.4,.5,.3,.2]))
        self.assertTrue(r.sequence_predicted_label.eq('CFFM1_67K').all())

    def test_short_tailless_sequence_uses_duration_only(self):
        r,_=analyze_sequences(calls([8,9,10]))
        self.assertTrue(r.sequence_predicted_label.eq('CFFM1_67K').all())

    def test_tail_values_and_legacy_tail_switch_have_no_effect(self):
        d=calls([8,9,10])
        before,_=analyze_sequences(d,SequenceConfig(short_requires_tail=True))
        d['tail_low_energy_ratio']=[0.,np.nan,1.]
        after,_=analyze_sequences(d,SequenceConfig(short_requires_tail=False))
        self.assertEqual(before.sequence_predicted_label.tolist(),after.sequence_predicted_label.tolist())
        d=d.drop(columns='tail_low_energy_ratio')
        after,_=analyze_sequences(d)
        self.assertEqual(before.sequence_predicted_label.tolist(),after.sequence_predicted_label.tolist())

    def test_duration_only_variant_preserves_short_cffm_without_tail(self):
        r,_=analyze_sequences(calls([8,9,10]), SequenceConfig(short_requires_tail=False))
        self.assertTrue(r.sequence_predicted_label.eq('CFFM1_67K').all())

    def test_two_calls_fall_back_to_single(self):
        r,_=analyze_sequences(calls([8,30]))
        self.assertTrue(r.family_source.eq('single').all())
        self.assertFalse(r.sequence_changed.any())

    def test_obvious_mixed_durations_require_review(self):
        d=calls([8,30,9,31,7,29]);d['sequence_reference_eligible']=True
        r,_=analyze_sequences(d)
        self.assertTrue(r.family_source.eq('sequence_review').all())
        self.assertFalse(r.sequence_changed.any())
        self.assertTrue(r.sequence_review_reason.str.contains('mixed_duration').all())

    def test_two_long_duration_clusters_are_both_cf_not_mixed_families(self):
        r,_=analyze_sequences(calls([20,41,21,40,19,42]))
        self.assertTrue(r.sequence_family.eq('CF').all())
        self.assertFalse(r.sequence_review_reason.str.contains('mixed_duration').any())

    def test_recordings_and_frequency_steps_do_not_link(self):
        a=calls([8,9]); b=calls([8],file='two.wav',start=.2)
        d=pd.concat([a,b],ignore_index=True)
        r,s=analyze_sequences(d)
        self.assertEqual(len(s),2)
        d=calls([8,9,8]); d.loc[2,'cf_plateau_khz']=68.
        _,s=analyze_sequences(d)
        self.assertEqual(len(s),2)

    def test_unknown_noise_and_single_family_are_unchanged(self):
        d=calls([8,9,8,9,8,9])
        d.loc[0,'gate_status']='noise'
        d.loc[1,'predicted_label']='unknown'; d.loc[1,'template_cf_label']=''
        d.loc[2,'template_cffm_score']=.1
        r,_=analyze_sequences(d)
        self.assertEqual(r.loc[0,'sequence_predicted_label'],d.loc[0,'predicted_label'])
        self.assertEqual(r.loc[1,'sequence_predicted_label'],'unknown')
        self.assertEqual(r.loc[2,'sequence_predicted_label'],d.loc[2,'predicted_label'])

    def test_same_frequency_overlap_is_reviewed(self):
        d=calls([30,8,30,30]); d.loc[1,['start_s','end_s']]=[.01,.018]
        r,_=analyze_sequences(d)
        self.assertFalse(r.sequence_changed.any())
        self.assertTrue(r.sequence_review_reason.str.contains('overlap').any())

    def test_missing_values_do_not_change_label(self):
        d=calls([8,9,10]); d.loc[0,'cf_plateau_ms']=np.nan
        r,_=analyze_sequences(d)
        self.assertFalse(r.sequence_changed.any())


if __name__=='__main__':
    unittest.main()
