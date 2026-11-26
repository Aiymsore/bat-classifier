"""Weak boundary recovery with separation/feature and adjacent-call safeguards."""
import os,sys
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('BAT_WORKSPACE',str(ROOT))
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'.cache/numba'))
os.environ.setdefault('NUMBA_CACHE_LOCATOR_CLASSES','bat_classifier.runtime_cache.WorkspaceCacheLocator')
import numpy as np
from bat_classifier.detection import extend_pulse_boundary
from bat_classifier.separation import separate_simultaneous_pulse_components
from bat_classifier.features import compute_pulse_spectrum,with_observed_support,extract_plateau_features
from bat_classifier.morphology_gate import score_morphology

SR=250000

def recording(second=None):
    rng=np.random.default_rng(21)
    t=np.arange(int(.5*SR))/SR;y=rng.normal(0,.0002,len(t))
    env=np.zeros(len(t));env[(t>=.20)&(t<.26)]=.04;env[(t>=.225)&(t<.230)]=.8
    y+=env*np.sin(2*np.pi*58000*t)
    if second is not None:
        start,freq=second
        on=(t>=start)&(t<start+.04)
        y[on]+=.6*np.sin(2*np.pi*freq*t[on])
    return y


class BoundaryTrackingTests(unittest.TestCase):
    def test_gate_uses_observed_weak_ends_instead_of_short_core(self):
        t=np.arange(int(.08*SR))/SR
        env=np.where((t>=.005)&(t<.075),.10,0.)
        env[(t>=.030)&(t<.054)]=.8
        y=env*np.sin(2*np.pi*83500*t)
        y+=np.random.default_rng(8).normal(0,.0001,len(t))
        spectrum=compute_pulse_spectrum(y,SR)
        def track(a,b):
            times=np.arange(a,b,32/SR)
            return dict(times_s=times,frequencies_hz=np.full(len(times),83500.),
                        coverage=1.,original_concentration=.95,original_concentration_local=.98,original_entropy=.1)
        core=track(.030,.054)
        args=dict(context_audio=np.pad(y,(2500,2500)),start_sample=2500,end_sample=2500+len(y))
        old=score_morphology(spectrum,component={'_gate_track':core},**args)
        restored=score_morphology(spectrum,component={'_gate_track':core,'_boundary_track':track(.005,.075)},**args)
        self.assertEqual(old['gate_status'],'uncertain')
        self.assertLess(old['gate_coverage'],.55)
        self.assertEqual(restored['gate_status'],'bat')
        self.assertGreater(restored['gate_track_duration_ms'],65.)
        self.assertGreater(restored['gate_coverage'],.9)
        self.assertEqual(restored['gate_track_source'],'observed_boundary')

    def test_extended_mask_cannot_override_poor_original_mixture_support(self):
        t=np.arange(int(.07*SR))/SR
        y=np.sin(2*np.pi*83500*t)
        track=dict(times_s=t[::32],frequencies_hz=np.full(len(t[::32]),83500.),coverage=1.,
                   original_concentration=.03,original_concentration_local=.25,original_entropy=.95)
        result=score_morphology(compute_pulse_spectrum(y,SR),component={'_boundary_track':track},
            context_audio=np.pad(y,(2500,2500)),start_sample=2500,end_sample=2500+len(y))
        self.assertNotEqual(result['gate_status'],'bat')
        self.assertLess(result['gate_concentration'],.25)
        self.assertLess(result['gate_concentration_local'],.90)

    def extend(self,y):
        a,b=extend_pulse_boundary(y,SR,int(.225*SR),int(.230*SR),58000.,max_duration_ms=108.)
        return a/SR,b/SR

    def test_weak_both_ends_survive_short_strong_core(self):
        a,b=self.extend(recording())
        self.assertLess(a,.202);self.assertGreater(b,.258)
        self.assertGreater(a,.197);self.assertLess(b,.263)

    def test_same_frequency_next_call_is_not_absorbed(self):
        a,b=self.extend(recording((.266,58000.)))
        self.assertLess(b,.264);self.assertGreater(b,.258)

    def test_other_frequency_next_call_is_not_followed(self):
        a,b=self.extend(recording((.262,52000.)))
        self.assertLess(b,.265)

    def test_noise_only_does_not_expand_to_duration_limit(self):
        y=np.random.default_rng(99).normal(0,.02,int(.5*SR))
        a,b=self.extend(y)
        self.assertLess(b-a,.015)

    def test_total_duration_budget_is_respected(self):
        y=recording()
        a,b=extend_pulse_boundary(y,SR,int(.225*SR),int(.230*SR),58000.,max_duration_ms=20.)
        self.assertLessEqual((b-a)/SR,.0202)

    def test_separation_and_plateau_keep_weak_observed_ends(self):
        t=np.arange(int(.08*SR))/SR
        env=np.zeros(len(t));env[(t>=.015)&(t<.065)]=.003
        env[(t>=.038)&(t<.043)]=.7
        y=env*np.sin(2*np.pi*58000*t)+.8*np.sin(2*np.pi*85000*t)
        y+=np.random.default_rng(4).normal(0,1e-5,len(t))
        components=separate_simultaneous_pulse_components(y.astype(np.float32),SR,harmonic_grouping_mode='enhanced')
        observed=[c for c in components if np.isfinite(c['track_median_freq_khz'])]
        self.assertTrue(observed)
        low=min(observed,key=lambda c:abs(c['track_median_freq_khz']-58.))
        self.assertLess(abs(low['track_median_freq_khz']-58.),2.)
        self.assertLess(low['start_sample']/SR,.018)
        self.assertGreater(low['end_sample']/SR,.062)
        spectrum=with_observed_support(compute_pulse_spectrum(low['audio'],SR),low.get('_boundary_track'))
        self.assertIsNotNone(spectrum.supported_frames)
        self.assertFalse(spectrum.supported_frames.flags.writeable)
        self.assertGreater(extract_plateau_features(spectrum)['cf_plateau_ms'],40.)


if __name__=='__main__':unittest.main()
