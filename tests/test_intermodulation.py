"""Conservative offline intermodulation evidence and counting safeguards."""
import sys
import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from bat_classifier.intermodulation import evaluate_profiles, resolve_relations, audit_intermodulation, measure_relation
import soundfile as sf


class IntermodulationTests(unittest.TestCase):
    def profiles(self):
        t=np.linspace(-.012,.042,200)
        a=np.exp(-((t-.015)/.010)**8)
        b=np.exp(-((t-.015)/.008)**8)
        powers=np.array([.1*a*a*b+1e-9,2*a+1e-9,b+1e-9])
        frequencies=np.array([np.full(len(t),47000.),np.full(len(t),59000.),np.full(len(t),71000.)])
        prominence=np.full((3,len(t)),25.)
        return t,frequencies,powers,prominence,[(0.,.030)]*3,122.

    def test_true_cubic_product(self):
        r=evaluate_profiles(*self.profiles())
        self.assertTrue(r['passed'])

    def test_coincident_frequency_independent_envelope(self):
        args=list(self.profiles()); args[2][0]=.1*np.exp(-((args[0]-.023)/.003)**2)+1e-9
        self.assertFalse(evaluate_profiles(*args)['passed'])

    def test_independent_extension_is_protected(self):
        args=list(self.profiles()); args[2][0]=.1
        r=evaluate_profiles(*args)
        self.assertTrue(r['independent_activity']); self.assertFalse(r['passed'])

    def test_missing_source_time_is_protected(self):
        args=list(self.profiles()); args[4]=[(0,.03),(0,.03),(.012,.03)]
        self.assertFalse(evaluate_profiles(*args)['passed'])

    def test_frequency_relation_must_hold_framewise(self):
        args=list(self.profiles()); args[1][0]+=900*np.sin(np.arange(200))
        self.assertFalse(evaluate_profiles(*args)['passed'])

    def test_shorter_than_one_spectral_window_is_not_auto_excluded(self):
        args=list(self.profiles()); args[4]=[(.013,.018)]*3
        r=evaluate_profiles(*args)
        self.assertFalse(r['passed']); self.assertIn('duration_support',r['failed_checks'])

    def test_weaker_parent_cannot_explain_stronger_product(self):
        args=list(self.profiles()); args[2][1]*=.001
        self.assertFalse(evaluate_profiles(*args)['passed'])

    def test_ambiguous_explanations_and_parent_conflict(self):
        p=[dict(product_id='c',parent_a_id='a',parent_b_id='b',passed=True),
           dict(product_id='c',parent_a_id='d',parent_b_id='e',passed=True)]
        self.assertTrue(all(x['decision']=='review_ambiguous' for x in resolve_relations(p)))
        p=[dict(product_id='c',parent_a_id='a',parent_b_id='b',passed=True),
           dict(product_id='a',parent_a_id='c',parent_b_id='d',passed=True)]
        self.assertTrue(all(x['decision']=='review_parent_conflict' for x in resolve_relations(p)))

    def test_single_real_47_call_preserved_and_input_unchanged(self):
        d=pd.DataFrame([dict(segment_file='a.wav',pulse_id='p1',start_s=0.,end_s=.03,
                            peak_freq_khz=47.,gate_status='bat',predicted_label='CF1_47K')])
        original=d.copy(deep=True)
        r,e=audit_intermodulation(d,{'a.wav':'unused'})
        pd.testing.assert_frame_equal(d,original)
        pd.testing.assert_frame_equal(r[d.columns],d)
        self.assertTrue(r.corrected_counted.all()); self.assertTrue(e.empty)

    def test_raw_wav_cubic_product(self):
        sr=250000; t=np.arange(int(.1*sr))/sr
        a=np.exp(-((t-.05)/.014)**8)
        b=np.exp(-((t-.05)/.011)**8)
        y=.35*a*np.sin(2*np.pi*59000*t)+.25*b*np.sin(2*np.pi*71000*t)
        y+=.08*a*a*b*np.sin(2*np.pi*47000*t)
        y+=np.random.default_rng(7).normal(0,1e-5,len(t))
        rows=pd.DataFrame([dict(start_s=.03,end_s=.07,peak_freq_khz=f) for f in [47,59,71]])
        rows.loc[0,['start_s','end_s']]=[.039,.061]
        workspace_tmp=Path(__file__).resolve().parents[1]/'.tmp'
        workspace_tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workspace_tmp) as tmp:
            assert Path(tmp).resolve().is_relative_to(workspace_tmp.resolve())
            path=Path(tmp)/'synthetic.wav'; sf.write(path,y,sr,subtype='FLOAT')
            r=measure_relation(path,*list(rows.itertuples()))
        self.assertTrue(r['passed'],r)

    def test_uncertain_parent_not_promoted_and_no_double_count(self):
        d=pd.DataFrame([dict(segment_file='a.wav',pulse_id=f'p{i}',start_s=0.,end_s=.03,
            peak_freq_khz=f,gate_status=g,predicted_label=l) for i,(f,g,l) in enumerate([
                (47,'bat','CF1'),(59,'bat','CF2'),(71,'uncertain','uncertain')])])
        old=d.copy(deep=True)
        with patch('bat_classifier.intermodulation.measure_relation',
                   side_effect=lambda wav,p,a,b,c: {'passed':p.peak_freq_khz==47}):
            r,e=audit_intermodulation(d,{'a.wav':'unused'})
        pd.testing.assert_frame_equal(d,old)
        self.assertEqual(r.corrected_counted.tolist(),[False,True,False])
        self.assertEqual(r.gate_status.tolist(),['bat','bat','uncertain'])


if __name__=='__main__':
    unittest.main()
