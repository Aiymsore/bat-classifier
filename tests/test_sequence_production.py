"""Production sequence finalization: disk ledger, cross-chunk groups and images."""
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
os.environ.setdefault('BAT_WORKSPACE',str(ROOT))
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'.cache/numba'))
os.environ.setdefault('NUMBA_CACHE_LOCATOR_CLASSES','bat_classifier.runtime_cache.WorkspaceCacheLocator')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import numpy as np
import pandas as pd
import soundfile as sf
from test_sequence_analysis import calls
from bat_classifier.segment_analysis import (SegmentAnalysisConfig,_SegmentAccumulator,
    _finalize_segment_sequences,_export_batch_pulse_spectrograms,_refresh_sequence_images,analyze_segment_folder)


class ProductionSequenceTests(unittest.TestCase):
    def test_duration_review_reconciles_gate_counts_and_replay(self):
        auditor=patch('bat_classifier.segment_analysis.assess_boundaries',side_effect=lambda rows,path:rows)
        auditor.start();self.addCleanup(auditor.stop)
        d=calls([2.048,3.2]);d['predicted_label']='CF3_66K';d['template_cffm_score']=.2
        d['confidence']=1.;d['is_uncertain']=False
        cfg=SegmentAnalysisConfig(save_figures=False,save_pulse_spectrograms=False)
        acc=_SegmentAccumulator(['CF3_66K'],'unknown',cfg)
        rows=d.to_dict('records')
        acc.add_batch(raw_candidate_count=2,pulse_rows=rows,result_rows=rows,
            noise_count=0,next_pulse_id=3,skipped=False)
        for _ in range(2):
            _finalize_segment_sequences(acc,None,Path('one.wav'),cfg)
            self.assertEqual(acc.detected_call_count,0)
            self.assertEqual(acc.valid_call_count,0)
            self.assertEqual(acc.gate_uncertain_count,2)
            self.assertEqual(acc.label_counts['CF3_66K'],0)
            self.assertEqual(acc.candidate_call_count,2)

    def test_review_images_follow_uncertain_switch(self):
        tmp_root=ROOT/'.tmp';tmp_root.mkdir(exist_ok=True)
        for show_review in (False,True):
            with self.subTest(show_review=show_review),tempfile.TemporaryDirectory(dir=tmp_root) as folder:
                assert Path(folder).resolve().is_relative_to(tmp_root.resolve())
                folder=Path(folder);wav=folder/'one.wav';root=folder/'images'
                sr=250000;t=np.arange(int(.5*sr))/sr;y=.3*np.sin(2*np.pi*66000*t)
                sf.write(wav,y,sr)
                cfg=SegmentAnalysisConfig(save_figures=False,save_uncertain_spectrograms=show_review)
                row=dict(pulse_id='p001',candidate_id=1,component_id=1,start_s=.2,end_s=.208,
                    duration_ms=8.,peak_freq_khz=66.,predicted_label='CF3_66K',gate_status='bat',
                    separation_applied=False,source_path=str(wav))
                _export_batch_pulse_spectrograms([row],segment_id='one',spectrogram_root=root,
                    read_y=y,read_start_s=0.)
                old=folder/row['pulse_spectrogram_file']
                self.assertTrue(old.exists())
                row.update(gate_status='uncertain',predicted_label='uncertain',sequence_changed=True)
                _refresh_sequence_images([row],wav,cfg,root)
                self.assertFalse(old.exists())
                if show_review:
                    new=folder/row['pulse_spectrogram_file']
                    self.assertTrue(new.exists());self.assertEqual(new.parent.name,'uncertain')
                else:
                    self.assertEqual(row['pulse_spectrogram_file'],'')

    def test_incomplete_preview_cannot_silently_replace_full_recording(self):
        cfg=SegmentAnalysisConfig()
        acc=_SegmentAccumulator(['CF3_66K'],'unknown',cfg)
        acc.rows_total=3
        with self.assertRaisesRegex(ValueError,'完整声波记录'):
            _finalize_segment_sequences(acc,None,Path('one.wav'),cfg)

    def test_real_pipeline_complete_short_call_conflicts_across_ten_second_boundary(self):
        from test_tail_features import family_model_and_rows, cf_call
        model,_=family_model_and_rows([cf_call(66500.,30.)])
        tmp_root=ROOT/'.tmp';tmp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as folder:
            assert Path(folder).resolve().is_relative_to(tmp_root.resolve())
            folder=Path(folder);wav=folder/'cross_boundary.wav'
            sr=250000;y=np.zeros(int(10.3*sr))
            for i,duration in enumerate([30.,8.,28.,31.]):
                audio=cf_call(66500.,duration)
                start=int((9.8+i*.1)*sr)
                y[start:start+len(audio)]=audio
            sf.write(wav,y,sr)
            result=analyze_segment_folder(wav,SegmentAnalysisConfig(min_confidence=0.,
                save_pulse_audio=False,save_figures=False,save_pulse_spectrograms=False),
                output_root=folder/'out',template_payload=model,resume=False)
            p=result['pulse_predictions']
            bat=p[p.gate_status.eq('bat')]
            self.assertEqual(len(bat),4)
            self.assertEqual(int(bat.sequence_changed.sum()),0)
            self.assertEqual(bat.predicted_label.tolist(),['CF3_66K','CFFM1_67K','CF3_66K','CF3_66K'])
            self.assertTrue(bat.sequence_review_reason.str.contains('reference_family_conflict').all())
            summary=result['segment_summary'].iloc[0]
            self.assertEqual(summary['CF3_66K'],3)
            self.assertEqual(summary['CFFM1_67K'],1)
            disk=pd.read_csv(folder/'out/pulse_predictions.csv')
            self.assertEqual(disk.predicted_label.tolist(),p.predicted_label.tolist())
            sequence=pd.read_csv(folder/'out/segments/cross_boundary/sequence_summary.csv')
            self.assertEqual(sequence.sequence_size.tolist(),[4])

    def test_cross_chunk_disk_ledger_survives_row_cap_and_finalization_replay(self):
        auditor=patch('bat_classifier.segment_analysis.assess_boundaries',side_effect=lambda rows,path:rows)
        auditor.start();self.addCleanup(auditor.stop)
        d=calls([30,8,28,31],start=9.8)
        d['confidence']=.8
        d['is_uncertain']=False
        cfg=SegmentAnalysisConfig(save_figures=False,save_pulse_spectrograms=False)
        acc=_SegmentAccumulator(['CF3_66K','CFFM1_67K'],'unknown',cfg)
        tmp_root=ROOT/'.tmp';tmp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as folder:
            assert Path(folder).resolve().is_relative_to(tmp_root.resolve())
            ledger=Path(folder)/'segment/.checkpoint/pulse_results.partial.csv'
            ledger.parent.mkdir(parents=True)
            with patch('bat_classifier.segment_analysis.SEGMENT_MAX_INMEMORY_ROWS',2):
                for chunk in (d.iloc[:2],d.iloc[2:]):
                    rows=chunk.to_dict('records')
                    acc.add_batch(raw_candidate_count=2,pulse_rows=rows,result_rows=rows,
                        noise_count=0,next_pulse_id=5,skipped=False)
                self.assertEqual(acc.label_counts['CFFM1_67K'],1)
                d.to_csv(ledger,index=False,encoding='utf-8-sig')
                cp=SimpleNamespace(partial_csv_path=ledger)
                _finalize_segment_sequences(acc,cp,Path('one.wav'),cfg)
                self.assertEqual(acc.label_counts['CF3_66K'],4)
                self.assertEqual(acc.label_counts['CFFM1_67K'],0)
                self.assertEqual(acc.detected_call_count,4)
                self.assertTrue(acc.rows_truncated)
                self.assertEqual(len(acc.rows),2)
                output=pd.read_csv(ledger)
                self.assertEqual(output.sequence_size.tolist(),[4.]*4)
                self.assertEqual(output.single_predicted_label.tolist(),d.predicted_label.tolist())
                _finalize_segment_sequences(acc,cp,Path('one.wav'),cfg)
                replay=pd.read_csv(ledger)
                pd.testing.assert_frame_equal(output,replay)
                self.assertEqual(acc.label_counts['CF3_66K'],4)

    def test_changed_event_image_is_regenerated_in_final_class_directory(self):
        cfg=SegmentAnalysisConfig(save_figures=False,save_pulse_spectrograms=True)
        tmp_root=ROOT/'.tmp';tmp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as folder:
            assert Path(folder).resolve().is_relative_to(tmp_root.resolve())
            folder=Path(folder);wav=folder/'one.wav';root=folder/'images'
            sr=250000;t=np.arange(int(.5*sr))/sr;y=.3*np.sin(2*np.pi*66000*t)
            sf.write(wav,y,sr)
            row=dict(pulse_id='p001',candidate_id=1,component_id=1,start_s=.2,end_s=.208,
                duration_ms=8.,peak_freq_khz=66.,predicted_label='CFFM1_67K',gate_status='bat',
                separation_applied=False,source_path=str(wav))
            self.assertEqual(_export_batch_pulse_spectrograms([row],segment_id='one',
                spectrogram_root=root,read_y=y,read_start_s=0.),1)
            old=Path(row['pulse_spectrogram_file'])
            if not old.is_absolute(): old=folder/old
            self.assertTrue(old.exists())
            row.update(predicted_label='CF3_66K',sequence_changed=True)
            _refresh_sequence_images([row],wav,cfg,root)
            new=Path(row['pulse_spectrogram_file'])
            if not new.is_absolute(): new=folder/new
            self.assertTrue(new.exists());self.assertEqual(new.parent.name,'CF3_66K')
            self.assertFalse(old.exists());self.assertFalse(old.with_suffix('.json').exists())


if __name__=='__main__':unittest.main()
