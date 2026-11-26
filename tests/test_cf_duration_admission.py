"""Final CF duration guard: no single-family bypass, no automatic CFFM fallback."""
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'src'),str(Path(__file__).resolve().parent)]
from test_sequence_analysis import calls
from bat_classifier.sequence_analysis import classify_sequence_rows,SequenceConfig


def cf_only(durations):
    d=calls(durations)
    d['predicted_label']='CF3_66K'
    d['confidence']=1.
    d['is_uncertain']=False
    d['template_cffm_score']=.2
    return d


class CfAdmissionTests(unittest.TestCase):
    def test_two_short_cf_only_calls_become_review_not_cffm(self):
        d=cf_only([2.048,3.2]);before=d.copy(deep=True)
        r,_=classify_sequence_rows(d)
        self.assertEqual(r.predicted_label.tolist(),['uncertain']*2)
        self.assertEqual(r.gate_status.tolist(),['uncertain']*2)
        self.assertTrue(r.cf_duration_review.all())
        self.assertEqual(r.single_gate_status.tolist(),['bat']*2)
        self.assertEqual(r.single_predicted_label.tolist(),['CF3_66K']*2)
        pd.testing.assert_frame_equal(d,before)

    def test_exact_threshold_is_accepted_below_is_rejected(self):
        r,_=classify_sequence_rows(cf_only([16.999,17.]))
        self.assertEqual(r.predicted_label.tolist(),['uncertain','CF3_66K'])
        self.assertEqual(r.cf_duration_source.tolist(),['review','single'])

    def test_cf_only_clean_long_sequence_supports_short_member(self):
        r,_=classify_sequence_rows(cf_only([30.,8.,28.]))
        self.assertEqual(r.predicted_label.tolist(),['CF3_66K']*3)
        self.assertEqual(r.cf_duration_source.tolist(),['single','sequence','single'])
        self.assertTrue(r.cf_duration_accepted.all())

    def test_short_sequence_does_not_create_cffm_without_template_support(self):
        r,_=classify_sequence_rows(cf_only([8.,9.,10.]))
        self.assertEqual(r.predicted_label.tolist(),['uncertain']*3)

    def test_mixed_sequence_cannot_support_short_members(self):
        d=cf_only([8,30,9,31,7,29]);d['sequence_reference_eligible']=True
        r,_=classify_sequence_rows(d)
        self.assertEqual(int(r.cf_duration_review.sum()),3)
        self.assertEqual(int(r.cf_duration_accepted.sum()),3)
        self.assertTrue(r.sequence_review_reason.str.contains('mixed_duration').all())

    def test_overlap_cannot_support_short_member(self):
        d=cf_only([30,8,30,30]);d.loc[1,['start_s','end_s']]=[.01,.018]
        r,_=classify_sequence_rows(d)
        self.assertEqual(r.loc[1,'predicted_label'],'uncertain')

    def test_existing_uncertain_noise_unknown_are_not_promoted(self):
        d=cf_only([30.,30.,30.]);d['gate_status']=['uncertain','noise','bat']
        d['predicted_label']=['uncertain','noise','unknown']
        r,_=classify_sequence_rows(d)
        self.assertEqual(r.predicted_label.tolist(),d.predicted_label.tolist())
        self.assertFalse(r.cf_duration_accepted.any())

    def test_finalization_is_idempotent_including_gate_reason(self):
        r,_=classify_sequence_rows(cf_only([2.048,3.2]))
        again,_=classify_sequence_rows(r)
        pd.testing.assert_frame_equal(r,again)

    def test_missing_platform_is_not_accepted(self):
        d=cf_only([8.]);d.loc[0,'cf_plateau_ms']=np.nan
        r,_=classify_sequence_rows(d)
        self.assertTrue(r.cf_duration_review.all())

    def test_configured_threshold_used_by_guard(self):
        r,_=classify_sequence_rows(cf_only([15.]),SequenceConfig(plateau_threshold_ms=14.))
        self.assertEqual(r.predicted_label.tolist(),['CF3_66K'])


if __name__=='__main__':unittest.main()
