"""直接查看的自动频率范围：覆盖所有有叫声的频段，不被连续低频底噪带偏；可选完整频段。"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("BAT_WORKSPACE", str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
(ROOT / ".tmp").mkdir(exist_ok=True)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from bat_classifier.visualization import _auto_frequency_limits, save_quick_overview  # noqa: E402

SR = 250000


def synthetic_matrix() -> tuple[np.ndarray, np.ndarray]:
    """频率 × 时间的 dB 矩阵：20–35 kHz 连续强底噪，66 kHz 强叫声、85 kHz 弱 15 dB 的叫声间歇出现。"""
    rng = np.random.default_rng(1)
    freqs = np.arange(15000.0, 120001.0, 500.0)
    frames = 400
    db = rng.normal(-60.0, 2.0, (len(freqs), frames))
    db[(freqs >= 20000) & (freqs <= 35000)] += 35.0
    on = np.zeros(frames, dtype=bool)
    for start in range(10, frames, 40):
        on[start:start + 8] = True
    rows = np.flatnonzero(np.abs(freqs - 66000) <= 500)
    db[np.ix_(rows, np.flatnonzero(on))] = -5.0
    rows = np.flatnonzero(np.abs(freqs - 85000) <= 500)
    db[np.ix_(rows, np.flatnonzero(np.roll(on, 20)))] = -20.0
    return db, freqs


class AutoFocusTests(unittest.TestCase):
    def test_range_covers_every_call_band_not_the_noise(self):
        db, freqs = synthetic_matrix()
        low, high, mode = _auto_frequency_limits(db, freqs, SR)
        self.assertEqual(mode, "auto-all-bands")
        self.assertLess(low, 66000)
        self.assertGreater(high, 85000)
        self.assertGreater(low, 35000)

    def test_short_clip_keeps_temporally_supported_call(self):
        db, freqs = synthetic_matrix()
        low, high, _mode = _auto_frequency_limits(db[:, :20], freqs, SR)
        self.assertLess(low, 66000)
        self.assertGreater(high, 66000)

    def test_rare_weak_high_band_is_included(self):
        db, freqs = synthetic_matrix()
        db[np.abs(freqs - 85000) <= 500] = -60.0
        # 仅 2/400 帧，且比 66 kHz 强叫声弱 40 dB；不能靠全局峰值或占时比例筛掉。
        rows = np.flatnonzero(np.abs(freqs - 86000) <= 500)
        db[np.ix_(rows, [201, 202])] = -45.0
        low, high, mode = _auto_frequency_limits(db, freqs, SR)
        self.assertEqual(mode, "auto-all-bands")
        self.assertGreater(low, 35000)
        self.assertGreaterEqual(high, 94500)

    def test_isolated_pixels_and_broadband_click_do_not_expand_range(self):
        db, freqs = synthetic_matrix()
        expected = _auto_frequency_limits(db, freqs, SR)
        db[np.argmin(abs(freqs - 115000)), 50:55] = 0.0
        db[np.abs(freqs - 110000) <= 500, 80] = 0.0
        db[:, 100:103] = 0.0
        self.assertEqual(_auto_frequency_limits(db, freqs, SR), expected)

    def test_sustained_short_clip_fallback_covers_both_bands(self):
        freqs = np.arange(15000.0, 120001.0, 500.0)
        db = np.full((len(freqs), 20), -70.0)
        db[np.abs(freqs - 66000) <= 500] = -5.0
        db[np.abs(freqs - 86000) <= 500] = -20.0
        low, high, mode = _auto_frequency_limits(db, freqs, SR)
        self.assertEqual(mode, "auto")
        self.assertLess(low, 66000)
        self.assertGreater(high, 86000)

    def test_empty_time_axis_returns_full_band(self):
        freqs = np.arange(15000.0, 120001.0, 500.0)
        self.assertEqual(_auto_frequency_limits(np.empty((len(freqs), 0)), freqs, SR),
                         (15000.0, 120000.0, "full-fallback"))

    def test_full_band_option(self):
        rng = np.random.default_rng(2)
        t = np.arange(int(SR * 0.4)) / SR
        y = 0.02 * rng.normal(size=t.size)
        for start in np.arange(0.02, 0.38, 0.05):
            k = (t >= start) & (t < start + 0.02)
            y[k] += 0.5 * np.sin(2 * np.pi * 66000 * t[k])
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as folder:
            wav = Path(folder) / "clip.wav"
            sf.write(wav, y, SR)
            for full_band, expected in ((False, None), (True, [15000.0, 120000.0])):
                out = Path(folder) / f"view_{full_band}.png"
                self.assertTrue(save_quick_overview(wav, out, full_band=full_band))
                limits = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))["freq_ylim"]
                if expected:
                    self.assertEqual([round(v) for v in limits], expected)
                else:
                    self.assertLess(limits[1] - limits[0], 105000)

    def test_direct_view_checkbox_defaults_off(self):
        from PySide6.QtWidgets import QApplication

        from bat_classifier.app.segment_analysis_tab import SegmentAnalysisTab

        _app = QApplication.instance() or QApplication([])
        tab = SegmentAnalysisTab(lambda _: None, lambda _: None)
        self.assertFalse(tab._direct_full_band_check.isChecked())


if __name__ == "__main__":
    unittest.main()
