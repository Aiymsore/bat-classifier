"""Raw bridge versus natural silence, and reliable sequence references."""
import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from bat_classifier.boundary_reliability import assess_boundaries,gap_decision,ridge_evidence
from bat_classifier.detection import merge_adjacent_fragments
from bat_classifier.sequence_analysis import classify_sequence_rows
from test_sequence_analysis import calls


class ReliabilityTests(unittest.TestCase):
    def test_stationary_noise_not_reported_as_persistent_ridge(self):
        fractions=[]
        for seed in range(10):
            y=np.random.default_rng(seed).normal(0,.02,25000)
            _,support,_=ridge_evidence(y,250000,0,66000)
            fractions.append(support.mean())
        # Stay below the 20% support allowed for a quiet exterior interval.
        self.assertLess(max(fractions),.2)

    def audio(self,intervals):
        sr=250000;t=np.arange(sr//2)/sr
        y=np.random.default_rng(31).normal(0,.0002,len(t))
        for a,b,amplitude in intervals:
            active=(t>=a)&(t<b)
            y[active]+=amplitude*np.sin(2*np.pi*66000*t[active])
        return y,sr

    def test_actual_bridge_required_for_fragment_merge(self):
        pulses=[dict(start_s=.2,end_s=.22,peak_freq_khz=66.),
                dict(start_s=.23,end_s=.25,peak_freq_khz=66.)]
        for p in pulses:
            p.update(snr_db=30.,candidate_start_s=p['start_s'],candidate_end_s=p['end_s'])
        for continuous in (True,False):
            intervals=[(.2,.25,.1)] if continuous else [(.2,.22,.1),(.23,.25,.1)]
            y,sr=self.audio(intervals)
            self.assertEqual(gap_decision(y,sr,0,.22,.23,66000), 'continuous' if continuous else 'separate')
            result=merge_adjacent_fragments(pulses,y=y,sr=sr,max_duration_ms=108.)
            self.assertEqual(len(result),1 if continuous else 2)

    def test_boundary_completeness_does_not_depend_on_17ms(self):
        y,sr=self.audio([(.2,.25,.1),(.3,.308,.1)])
        rows=pd.DataFrame([dict(start_s=a,end_s=b,cf_plateau_khz=66.,gate_status='bat',
            gate_quality=.9,gate_coverage=1.) for a,b in [(.2,.25),(.22,.23),(.3,.308)]])
        folder=ROOT/'.tmp';folder.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as tmp:
            wav=Path(tmp)/'test.wav';sf.write(wav,y,sr,subtype='FLOAT')
            r=assess_boundaries(rows,wav)
        self.assertEqual(r.loc[2,'boundary_status'],'complete')
        self.assertTrue(r.loc[2,'sequence_reference_eligible'])
        self.assertFalse(r.loc[1,'sequence_reference_eligible'])
        pd.testing.assert_series_equal(r.start_s,rows.start_s)

    def test_weak_majority_cannot_drag_reliable_reference_duration(self):
        d=calls([30,8,8,8,8,28]);d['sequence_reference_eligible']=[True,False,False,False,False,True]
        r,_=classify_sequence_rows(d)
        self.assertTrue(r.predicted_label.eq('CF3_66K').all())
        self.assertTrue(r.sequence_reference_count.eq(2).all())
        self.assertTrue(r.sequence_plateau_median_ms.eq(29).all())
        self.assertTrue(r.sequence_observed_median_ms.eq(8).all())
        pd.testing.assert_series_equal(r.cf_plateau_ms,d.cf_plateau_ms)

    def test_one_or_conflicting_references_cannot_correct_short_member(self):
        for eligible in ([True,False,False],[True,True,True]):
            d=calls([30,8,9]);d['sequence_reference_eligible']=eligible
            r,_=classify_sequence_rows(d)
            self.assertEqual(r.loc[1,'predicted_label'],'CFFM1_67K')
            self.assertTrue(r.sequence_review_reason.ne('').all())

    def test_unresolved_fragments_not_counted_as_confirmed_calls(self):
        d=calls([30,8,28]);d['segmentation_review']=[False,True,False]
        r,_=classify_sequence_rows(d)
        self.assertEqual(r.gate_status.tolist(),['bat','uncertain','bat'])
        self.assertEqual(len(r),3)
        self.assertEqual(r.loc[1,'gate_reason_code'],'segmentation_unresolved')
        again,_=classify_sequence_rows(r)
        pd.testing.assert_frame_equal(r,again)
