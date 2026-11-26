"""模板子带检测：弱窄带叫声在全带被底噪稀释时，子带检测应能找回。"""
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

from bat_classifier.config import TARGET_SAMPLE_RATE  # noqa: E402
from bat_classifier.detection import (  # noqa: E402
    detect_pulses,
    normalize_detection_bands,
    refine_all_pulses,
)
from bat_classifier.segment_analysis import SegmentAnalysisConfig  # noqa: E402
from bat_classifier.segment_stream import dedupe_cross_candidate_rows  # noqa: E402
from bat_classifier.template_matching import template_detection_bands  # noqa: E402

SR = TARGET_SAMPLE_RATE
BAND_83K = (81.0, 86.0)


def weak_tone_in_white_noise(
    seconds: float = 0.6,
    frequency: float = 83500.0,
    amplitude: float = 0.1,
    starts_s: tuple[float, ...] = (0.15, 0.40),
    seed: int = 11,
) -> np.ndarray:
    """白噪声里放两段 5 ms 弱纯音：全带对比度不到 10 dB，5 kHz 子带内超过 15 dB。"""
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 0.05, int(SR * seconds))
    count = int(SR * 0.005)
    envelope = np.sin(np.linspace(0, np.pi, count)) ** 0.3
    phase = 2 * np.pi * frequency * np.arange(count) / SR
    for start_s in starts_s:
        start = int(SR * start_s)
        y[start:start + count] += amplitude * envelope * np.sin(phase)
    return y


def tone_burst(frequency: float, seconds: float = 0.3, start_s: float = 0.1) -> np.ndarray:
    y = np.zeros(int(SR * seconds))
    count = int(SR * 0.008)
    start = int(SR * start_s)
    phase = 2 * np.pi * frequency * np.arange(count) / SR
    y[start:start + count] = 0.8 * np.sin(phase) * np.sin(np.linspace(0, np.pi, count)) ** 0.5
    return y


class TemplateBandDetectionTests(unittest.TestCase):
    def test_fullband_misses_weak_tone_but_subband_finds_it(self):
        y = weak_tone_in_white_noise()
        fullband, *_ = detect_pulses(y, SR)
        self.assertEqual(fullband, [])
        combined, *_ = detect_pulses(y, SR, bands_khz=[BAND_83K])
        band_hits = [p for p in combined if p.get("detect_band_khz")]
        self.assertEqual(len(band_hits), 2)
        refined = refine_all_pulses(y, SR, combined)
        self.assertEqual(len(refined), 2)
        for pulse, expected_s in zip(refined, (0.15, 0.40)):
            self.assertEqual(pulse["detect_source"], "band:81-86")
            self.assertLess(abs(pulse["start_s"] - expected_s), 0.003)

    def test_without_bands_behaviour_is_unchanged(self):
        y = tone_burst(47000.0)
        plain, *_rest_plain = detect_pulses(y, SR)
        explicit_none, *_rest_none = detect_pulses(y, SR, bands_khz=None)
        self.assertEqual(plain, explicit_none)
        self.assertTrue(plain)
        self.assertTrue(all("detect_band_khz" not in p for p in plain))
        refined = refine_all_pulses(y, SR, plain)
        self.assertEqual([p["detect_source"] for p in refined], ["fullband"])

    def test_overlapping_band_and_fullband_candidates_merge_into_one(self):
        y = tone_burst(47000.0)
        pulses, *_ = detect_pulses(y, SR, bands_khz=[(45.0, 49.0)])
        self.assertGreaterEqual(len(pulses), 2)
        refined = refine_all_pulses(y, SR, pulses)
        self.assertEqual(len(refined), 1)
        self.assertEqual(refined[0]["detect_source"], "fullband|band:45-49")

    def test_normalize_detection_bands_clips_and_drops_invalid(self):
        bands = normalize_detection_bands([(10.0, 20.0), (50.0, 50.5), (110.0, 140.0), (float("nan"), 3)], SR)
        self.assertEqual(bands, [(15.0, 20.0), (110.0, 120.0)])

    def test_template_bands_from_tracks_merge_and_skip_noise(self):
        def template(track):
            return {"features": {"template_track_khz": list(track)}}
        payload = {
            "kind": "bat_template_model",
            "groups": {
                "CF1_47K": {"templates": [template([46.0] * 32), template([47.0] * 32)]},
                "CF5_83K": {"templates": [template(np.linspace(85, 80, 32))]},
                "CFFM2_86K": {"templates": [template(np.r_[[86.0] * 28, np.linspace(85, 79, 4)])]},
                "noise pulse": {"templates": [template([30.0] * 32)]},
            },
        }
        from bat_classifier import template_matching

        original = template_matching.is_template_model
        template_matching.is_template_model = lambda _payload: True
        try:
            bands = template_detection_bands(payload)
        finally:
            template_matching.is_template_model = original
        self.assertEqual(len(bands), 2)
        self.assertEqual(bands[0]["labels"], ["CF1_47K"])
        self.assertAlmostEqual(bands[0]["low_khz"], 44.5)
        self.assertAlmostEqual(bands[0]["high_khz"], 48.5)
        self.assertEqual(sorted(bands[1]["labels"]), ["CF5_83K", "CFFM2_86K"])
        self.assertLess(bands[1]["low_khz"], 80.0)
        self.assertAlmostEqual(bands[1]["high_khz"], 87.5)

    def test_cross_candidate_duplicates_removed_but_same_candidate_kept(self):
        rows = [
            {"candidate_id": 1, "component_id": 1, "start_s": 0.100, "end_s": 0.108, "track_median_freq_khz": 47.0, "snr_db": 20.0},
            {"candidate_id": 1, "component_id": 2, "start_s": 0.100, "end_s": 0.108, "track_median_freq_khz": 83.0, "snr_db": 8.0},
            {"candidate_id": 2, "component_id": 1, "start_s": 0.101, "end_s": 0.108, "track_median_freq_khz": 83.5, "snr_db": 12.0},
            {"candidate_id": 2, "component_id": 2, "start_s": 0.101, "end_s": 0.108, "track_median_freq_khz": 47.2, "snr_db": 6.0},
        ]
        kept = dedupe_cross_candidate_rows(rows)
        kept_keys = sorted((r["candidate_id"], r["component_id"]) for r in kept)
        self.assertEqual(kept_keys, [(1, 1), (2, 1)])

    def test_config_default_enables_template_bands(self):
        config = SegmentAnalysisConfig().validate()
        self.assertTrue(config.template_band_detection)


if __name__ == "__main__":
    unittest.main()
