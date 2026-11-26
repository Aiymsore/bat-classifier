"""检测诊断 CSV 的行为验证：只读、判据可复现、聚合指标正确。"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("BAT_WORKSPACE", str(ROOT))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache" / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import numpy as np  # noqa: E402

from bat_classifier.config import (  # noqa: E402
    DETECTION_ADAPTIVE_MARGIN_DB, TARGET_SAMPLE_RATE,
)
from bat_classifier.detection import detect_pulses, diagnose_band_detection  # noqa: E402
from bat_classifier.segment_analysis import (  # noqa: E402
    SegmentAnalysisConfig, _max_subband_contrast, _subband_only_candidates,
)

SR = TARGET_SAMPLE_RATE


def flat_noise(seconds: float = 0.2, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.05, int(SR * seconds)).astype(np.float32)


def chirp_in_silence(seconds: float = 0.2, frequency: float = 47000.0) -> np.ndarray:
    """静音里放一段 10 ms 的 47 kHz 纯音，全带对比度应远高于门限。"""
    y = np.zeros(int(SR * seconds), dtype=np.float32)
    count = int(SR * 0.010)
    start = int(SR * 0.05)
    phase = 2 * np.pi * frequency * np.arange(count) / SR
    y[start:start + count] = (0.8 * np.sin(phase)
                              * np.sin(np.linspace(0, np.pi, count)) ** 0.6)
    return y


class DetectionDiagnosticsTests(unittest.TestCase):
    def test_rows_cover_full_band_and_subbands(self):
        rows = diagnose_band_detection(chirp_in_silence(), SR)
        bands = [row["band_khz"] for row in rows]
        self.assertEqual(bands[0], "all")
        self.assertIn("40-55", bands)
        self.assertGreaterEqual(len(bands), 4)
        for row in rows:
            self.assertIn("contrast_db", row)
            self.assertIn("threshold_db", row)
            self.assertIn("raw_candidates", row)
            # 三个数值都做过 3 位小数舍入，比较时留 0.002 的容差。
            self.assertAlmostEqual(
                row["contrast_db"],
                row["peak_db"] - row["background_median_db"],
                delta=0.002,
            )
            # 检测器判据：阈值 = max(-28, 背景 + 余量)；活动帧等价于对比度 ≥ 余量
            self.assertAlmostEqual(
                row["threshold_db"],
                max(-28.0, row["background_median_db"] + DETECTION_ADAPTIVE_MARGIN_DB),
                delta=0.002,
            )

    def test_full_band_row_matches_detector(self):
        for audio in (flat_noise(), chirp_in_silence()):
            with self.subTest(signal=float(np.max(np.abs(audio)))):
                rows = diagnose_band_detection(audio, SR, include_refined=False)
                full = next(row for row in rows if row["band_khz"] == "all")
                pulses, *_ = detect_pulses(audio, SR)
                self.assertEqual(full["raw_candidates"], len(pulses))

    def test_flat_noise_cannot_trigger_any_band(self):
        rows = diagnose_band_detection(flat_noise(), SR)
        for row in rows:
            self.assertLess(row["contrast_db"], DETECTION_ADAPTIVE_MARGIN_DB)
            self.assertEqual(row["raw_candidates"], 0)
            self.assertEqual(row["active_frames"], 0)

    def test_visible_chirp_triggers_full_band(self):
        rows = diagnose_band_detection(chirp_in_silence(), SR)
        full = next(row for row in rows if row["band_khz"] == "all")
        self.assertGreaterEqual(full["contrast_db"], DETECTION_ADAPTIVE_MARGIN_DB)
        self.assertGreaterEqual(full["raw_candidates"], 1)
        narrow = next(row for row in rows if row["band_khz"] == "40-55")
        self.assertGreaterEqual(narrow["raw_candidates"], 1)

    def test_diagnosis_does_not_modify_audio(self):
        audio = chirp_in_silence()
        before = audio.copy()
        diagnose_band_detection(audio, SR, include_refined=False)
        np.testing.assert_array_equal(audio, before)

    def test_aggregation_marks_files_only_visible_in_subbands(self):
        # 全带 0 候选、子带 12 条：应记 12 条"只有子带能看见"的候选。
        rows = [
            {"chunk_index": 0, "band_khz": "all", "contrast_db": 6.0,
             "raw_candidates": 0, "skipped": False},
            {"chunk_index": 0, "band_khz": "55-70", "contrast_db": 20.0,
             "raw_candidates": 12, "skipped": False},
            {"chunk_index": 0, "band_khz": "70-85", "contrast_db": 16.0,
             "raw_candidates": 7, "skipped": False},
            # 第二个块全带已有候选：不计入 subband_only
            {"chunk_index": 1, "band_khz": "all", "contrast_db": 12.0,
             "raw_candidates": 3, "skipped": False},
            {"chunk_index": 1, "band_khz": "70-85", "contrast_db": 18.0,
             "raw_candidates": 40, "skipped": False},
        ]
        self.assertEqual(_subband_only_candidates(rows), 12)
        self.assertEqual(_max_subband_contrast(rows), 20.0)

    def test_diagnostics_are_opt_in(self):
        self.assertFalse(SegmentAnalysisConfig().save_detection_diagnostics)
        self.assertTrue(
            SegmentAnalysisConfig(save_detection_diagnostics=True).validate()
            .save_detection_diagnostics
        )


if __name__ == "__main__":
    unittest.main()
