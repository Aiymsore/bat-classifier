"""Run with: .venv/Scripts/python.exe -m unittest discover -s tests -v."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("BAT_WORKSPACE", str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache" / "numba"))

import numpy as np
import soundfile as sf
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QFont, QFontDatabase, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from bat_classifier.audio_io import load_audio_interval
from bat_classifier.app.linear_spectrogram_view import (
    LinearSpectrogramDialog, load_spectrogram_meta, resolve_source_audio,
)
from bat_classifier.segment_analysis import _export_batch_pulse_spectrograms
from bat_classifier.visualization import save_direct_overview, save_wav_interval_overview
from bat_classifier.local_spectrogram import LocalSpectrogramCache


class LocalSpectrogramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        # The Windows offscreen plugin starts without a system font database.
        for font in ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/msyh.ttc"):
            if Path(font).exists():
                QFontDatabase.addApplicationFont(font)
        cls.app.setFont(QFont("Microsoft YaHei", 9))
        (ROOT / ".tmp").mkdir(exist_ok=True)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".tmp", prefix="test_detail_")
        self.folder = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def wait_for_worker(self, dialog):
        deadline = time.monotonic() + 30
        while (dialog._worker is not None or dialog._zoom_timer.isActive()) and time.monotonic() < deadline:
            # qWait can retain the GIL on Windows, starving Python QThread.run.
            time.sleep(.01)
            self.app.processEvents()
        self.assertIsNone(dialog._worker, "Local render failed to finish")
        self.assertFalse(dialog._zoom_timer.isActive())

    def test_interval_uses_seek_and_bounded_read(self):
        sr = 250000
        t = np.arange(sr * 8) / sr
        audio = np.sin(2 * np.pi * 47000 * t)
        wav = self.folder / "long.wav"
        sf.write(wav, audio, sr, subtype="FLOAT")
        original = sf.SoundFile
        reads, seeks = [], []

        class RecordingFile:
            def __init__(self, *args, **kwargs):
                self.handle = original(*args, **kwargs)
                self.samplerate = self.handle.samplerate
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.handle.close()
            def __len__(self):
                return len(self.handle)
            def seek(self, frame):
                seeks.append(frame)
                return self.handle.seek(frame)
            def read(self, frames, **kwargs):
                reads.append(frames)
                return self.handle.read(frames, **kwargs)

        with patch("soundfile.SoundFile", RecordingFile):
            selected, actual_sr, start = load_audio_interval(wav, 3., 3.02, preprocessed=True)
        self.assertEqual(actual_sr, sr)
        self.assertAlmostEqual(start, 3.)
        self.assertEqual(len(selected), 5000)
        np.testing.assert_allclose(selected, audio[sr * 3:sr * 3 + 5000], atol=4e-8)
        self.assertEqual(len(reads), 1)
        self.assertLessEqual(reads[0], int(sr * .121))
        self.assertGreater(seeks[0], 0)
        # Mono conversion and resampling remain supported.
        other_sr = 192000
        tone = np.sin(2 * np.pi * 40000 * np.arange(other_sr) / other_sr)
        stereo = self.folder / "stereo.wav"
        sf.write(stereo, np.column_stack((tone, tone * .5)), other_sr, subtype="FLOAT")
        y, target_sr, start = load_audio_interval(stereo, .2, .21)
        self.assertEqual(target_sr, sr)
        self.assertEqual(len(y), 2500)
        self.assertEqual(y.ndim, 1)
        self.assertTrue(np.isfinite(y).all())

    def test_recomputed_range_preserves_time_and_increases_frame_density(self):
        wav = ROOT / "data/sample/demo_xianren_0_4s.wav"
        overview = self.folder / "overview.png"
        detail = self.folder / "detail.png"
        self.assertTrue(save_direct_overview(wav, overview))
        before = overview.read_bytes()
        meta = load_spectrogram_meta(overview)
        self.assertTrue(save_wav_interval_overview(meta["source_audio"], 1.71, 1.73, detail,
                                                  frequency_limits_hz=tuple(meta["freq_ylim"])))
        fine = load_spectrogram_meta(detail)
        np.testing.assert_allclose(fine["time_xlim"], [1.71, 1.73], atol=1 / 250000)
        self.assertLess(fine["time_step_s"], meta["time_step_s"])
        self.assertEqual(fine["stft_n_fft"], meta["stft_n_fft"])
        self.assertEqual(fine["freq_ylim"], meta["freq_ylim"])
        self.assertEqual(overview.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "2 秒"):
            save_wav_interval_overview(meta["source_audio"], 0, 3, detail)

    def test_ui_selection_worker_wheel_restore_and_legacy_fallback(self):
        wav = ROOT / "data/sample/demo_xianren_0_4s.wav"
        png = self.folder / "overview.png"
        self.assertTrue(save_direct_overview(wav, png))
        meta = load_spectrogram_meta(png)
        dialog = LinearSpectrogramDialog(png, meta)
        dialog.show()
        QTest.qWait(30)
        view = dialog._view
        rect = view._axes_rect(view._scaled_pixmap())
        p1 = QPoint(round(rect.left() + rect.width() * .42), round(rect.center().y()))
        p2 = QPoint(p1.x() + 30, p1.y())
        QTest.mousePress(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, p1)
        QTest.mouseMove(view, p2)
        QTest.mouseRelease(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, p2)
        self.assertIsNotNone(dialog._selected_range)
        self.assertTrue(dialog._detail_button.isEnabled())
        selected = dialog._selected_range
        with patch("matplotlib.figure.Figure.savefig", side_effect=AssertionError("Unexpected PNG render")):
            QTest.mouseClick(dialog._detail_button, Qt.MouseButton.LeftButton)
            self.assertIsNotNone(dialog._worker)
            dialog.accept()  # Closing while rendering must not destroy a running QThread.
            self.assertTrue(dialog.isVisible())
            self.wait_for_worker(dialog)
        np.testing.assert_allclose(view._meta["time_xlim"], selected, atol=1 / 250000)
        self.assertIsNotNone(view._detail_data)
        self.assertTrue(view.isEnabled())
        old_span = selected[1] - selected[0]
        rect = view._axes_rect(view._scaled_pixmap())
        pos = rect.center()
        event = QWheelEvent(pos, QPointF(view.mapToGlobal(pos.toPoint())), QPoint(), QPoint(0, 120),
                            Qt.MouseButton.NoButton, Qt.KeyboardModifier.ControlModifier,
                            Qt.ScrollPhase.NoScrollPhase, False)
        with patch.object(dialog._cache, "compute", wraps=dialog._cache.compute) as compute:
            # Three notches accumulate around the pending view; only the final one runs.
            for _ in range(3):
                self.app.sendEvent(view, event)
            self.assertIsNone(dialog._worker)
            self.wait_for_worker(dialog)
            self.assertEqual(compute.call_count, 1)
        low, high = view._meta["time_xlim"]
        self.assertAlmostEqual(high - low, old_span / 8, delta=2 / 250000)
        preview = ROOT / ".tmp/local_detail_preview"
        preview.mkdir(exist_ok=True)
        view.grab().save(str(preview / "detail.png"))
        dialog.grab().save(str(preview / "viewer.png"))
        view.resize(800, 450)
        rect = view._axes_rect(view._scaled_pixmap())
        self.assertAlmostEqual(view._hz_at(rect.top(), rect), view._meta["freq_ylim"][1])
        self.assertAlmostEqual(view._hz_at(rect.bottom(), rect), view._meta["freq_ylim"][0])
        self.assertAlmostEqual(view._time_at(rect.left(), rect), low)
        self.assertAlmostEqual(view._time_at(rect.right(), rect), high)
        self.assertGreater(dialog._cache.results.nbytes, 0)
        QTest.mouseClick(dialog._overview_button, Qt.MouseButton.LeftButton)
        self.assertEqual(view._meta["time_xlim"], meta["time_xlim"])
        dialog.accept()
        self.assertEqual(dialog._cache.results.nbytes, 0)
        self.assertEqual(dialog._cache.audio.nbytes, 0)
        legacy = dict(meta)
        legacy.pop("source_audio")
        self.assertIsNone(resolve_source_audio(png, legacy))
        self.assertIsNotNone(resolve_source_audio(png, legacy, meta["source_audio"]))

    def test_drag_pans_time_waveform_and_spectrum_and_clamps_edges(self):
        wav = ROOT / "data/sample/demo_xianren_0_4s.wav"
        png = self.folder / "overview.png"
        save_direct_overview(wav, png)
        meta = load_spectrogram_meta(png)
        dialog = LinearSpectrogramDialog(png, meta)
        dialog.show()
        dialog.request_detail(1.71, 1.73)
        self.wait_for_worker(dialog)
        view = dialog._view
        self.assertEqual(view.cursor().shape(), Qt.CursorShape.OpenHandCursor)

        def drag(dx, waveform=False):
            scaled = view._scaled_pixmap()
            rect = view._axes_rect(scaled)
            y = (view._image_origin(scaled).y() + scaled.height() * .3
                 if waveform else rect.center().y())
            p1 = QPoint(round(rect.center().x()), round(y))
            p2 = QPoint(p1.x() + dx, p1.y())
            QTest.mousePress(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p1)
            self.assertTrue(view._time_dragging)
            QTest.mouseMove(view, p2)
            pending = view._pan_preview_range
            self.assertIsNotNone(pending)
            # Preview moves immediately; it does not change the underlying source data.
            self.assertIsNotNone(view._detail_data)
            view.grab()
            QTest.mouseRelease(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p2)
            self.wait_for_worker(dialog)
            self.assertFalse(view._time_dragging)
            np.testing.assert_allclose(view._meta["time_xlim"], pending, atol=1 / 250000)
            self.assertAlmostEqual(view._meta["time_xlim"][1] - view._meta["time_xlim"][0], .02,
                                   delta=2 / 250000)
            return pending

        width = view._axes_rect(view._scaled_pixmap()).width()
        after = drag(-round(width / 4))
        self.assertGreater(after[0], 1.71)
        back = drag(round(width / 4), waveform=True)
        np.testing.assert_allclose(back, [1.71, 1.73], atol=2 / 250000)
        # Shift still selects a range in the local plot rather than panning it.
        rect = view._axes_rect(view._scaled_pixmap())
        p1 = rect.center().toPoint()
        p2 = p1 + QPoint(50, 0)
        QTest.mousePress(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, p1)
        self.assertFalse(view._time_dragging)
        QTest.mouseRelease(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, p2)
        self.assertIsNotNone(dialog._selected_range)
        for low, high, delta in ((0., .02, 150), (meta["time_xlim"][1] - .02, meta["time_xlim"][1], -150)):
            dialog.request_detail(low, high)
            self.wait_for_worker(dialog)
            bounded = drag(delta)
            self.assertGreaterEqual(bounded[0], meta["time_xlim"][0] - 1e-9)
            self.assertLessEqual(bounded[1], meta["time_xlim"][1] + 1e-9)
        preview = ROOT / ".tmp/local_detail_preview"
        dialog.request_detail(*after)
        self.wait_for_worker(dialog)
        dialog.grab().save(str(preview / "dragged.png"))
        dialog.accept()

    def test_wheel_zoom_out_and_pointer_anchor_and_two_second_limit(self):
        wav = ROOT / "data/sample/demo_xianren_0_4s.wav"
        png = self.folder / "overview.png"
        save_direct_overview(wav, png)
        meta = load_spectrogram_meta(png)
        # Give the viewer a longer overview range to reproduce the old 2s cap bug.
        # All tested local intervals still lie within the real 4s source.
        meta["time_xlim"] = [0., 10.]
        dialog = LinearSpectrogramDialog(png, meta)
        dialog.show()
        view = dialog._view

        def wheel(delta, ratio=.25):
            rect = view._axes_rect(view._scaled_pixmap())
            pos = QPointF(rect.left() + ratio * rect.width(), rect.center().y())
            event = QWheelEvent(pos, QPointF(view.mapToGlobal(pos.toPoint())), QPoint(), QPoint(0, delta),
                                Qt.MouseButton.NoButton, Qt.KeyboardModifier.ControlModifier,
                                Qt.ScrollPhase.NoScrollPhase, False)
            self.app.sendEvent(view, event)

        dialog.request_detail(1., 1.2)
        self.wait_for_worker(dialog)
        anchor = 1.05
        wheel(120)
        self.wait_for_worker(dialog)
        low, high = view._meta["time_xlim"]
        self.assertAlmostEqual(high - low, .1, delta=2 / 250000)
        self.assertAlmostEqual(low + .25 * (high - low), anchor, delta=2 / 250000)
        wheel(-120)
        self.wait_for_worker(dialog)
        np.testing.assert_allclose(view._meta["time_xlim"], [1., 1.2], atol=2 / 250000)
        # Forward/backward gestures before the debounce fires must also be reversible.
        wheel(120)
        wheel(-120)
        self.wait_for_worker(dialog)
        np.testing.assert_allclose(view._meta["time_xlim"], [1., 1.2], atol=2 / 250000)
        dialog.request_detail(1., 3.)
        self.wait_for_worker(dialog)
        wheel(-120, .8)
        self.assertIsNone(dialog._worker)
        self.assertFalse(dialog._zoom_timer.isActive())
        self.assertEqual(view._meta["time_xlim"], [0., 10.])
        self.assertIsNone(view._detail_data)
        # A long overview zooms into 2s while retaining the pointed-to time.
        wheel(120, .2)
        self.wait_for_worker(dialog)
        low, high = view._meta["time_xlim"]
        self.assertAlmostEqual(high - low, 2., delta=2 / 250000)
        self.assertAlmostEqual(low + .2 * (high - low), 2., delta=2 / 250000)
        dialog.accept()

    def test_data_matches_export_cache_reuse_invalidation_and_bounds(self):
        from bat_classifier.audio_io import load_audio_interval as read
        from bat_classifier.visualization import compute_overview_spectrum as stft
        wav = self.folder / "tone.wav"
        sr = 250000
        sf.write(wav, np.sin(2 * np.pi * 47000 * np.arange(sr) / sr), sr, subtype="FLOAT")
        source = {"path": str(wav), "offset_s": .1, "preprocessed": False}
        limits = (35000., 60000.)
        cache = LocalSpectrogramCache()
        with patch("bat_classifier.local_spectrogram.load_audio_interval", wraps=read) as reads, \
             patch("bat_classifier.local_spectrogram.compute_overview_spectrum", wraps=stft) as transforms:
            first = cache.compute(source, .2, .22, limits)
            self.assertIs(first, cache.compute(source, .2, .22, limits))
            self.assertEqual(reads.call_count, 1)
            self.assertEqual(transforms.call_count, 1)
            cache.compute(source, .2, .22, (40000., 55000.))
            self.assertEqual(reads.call_count, 1, "Frequency-only change should reuse audio")
            y, actual_sr, start = read(wav, .3, .32)
            expected, times, freqs, *_ = stft(y, actual_sr, 8000)
            np.testing.assert_array_equal(first.power_db, expected)
            np.testing.assert_array_equal(first.frequencies, freqs)
            np.testing.assert_allclose(first.times, times + start - .1)
            np.testing.assert_allclose(first.meta["time_xlim"], [.2, .22], atol=1 / sr)
            sf.write(wav, np.zeros(sr), sr, subtype="FLOAT")
            silent = cache.compute(source, .2, .22, limits)
            self.assertIsNot(first, silent)
            self.assertEqual(reads.call_count, 2)
            np.testing.assert_array_equal(silent.rgb, np.broadcast_to(silent.rgb[0, 0], silent.rgb.shape))
            self.assertLess(int(silent.rgb.max()), 30)
        small = LocalSpectrogramCache(audio_limit=50000, result_limit=400000)
        for low in (.1, .2, .3, .4, .5):
            small.compute(source, low, low + .02, limits)
            self.assertLessEqual(small.audio.nbytes, small.audio.limit)
            self.assertLessEqual(small.results.nbytes, small.results.limit)
        self.assertLess(len(small.audio.items), 5)
        self.assertLess(len(small.results.items), 5)
        wav.unlink()
        with self.assertRaises(FileNotFoundError):
            cache.compute(source, .2, .22, limits)

    def test_latest_request_wins_and_restore_cancels_pending_zoom(self):
        wav = ROOT / "data/sample/demo_xianren_0_4s.wav"
        png = self.folder / "overview.png"
        save_direct_overview(wav, png)
        meta = load_spectrogram_meta(png)
        dialog = LinearSpectrogramDialog(png, meta)
        dialog.show()
        original = dialog._cache.compute
        entered, release = threading.Event(), threading.Event()
        published = []
        set_detail = dialog._view.set_detail

        def slow(source, low, high, limits):
            if low == .3:
                entered.set()
                release.wait(5)
            return original(source, low, high, limits)

        def record(data):
            published.append(data.meta["time_xlim"])
            set_detail(data)

        with patch.object(dialog._cache, "compute", side_effect=slow) as compute, \
             patch.object(dialog._view, "set_detail", side_effect=record):
            dialog.request_detail(.3, .32)
            self.assertTrue(entered.wait(5))
            dialog.request_detail(.4, .42)
            dialog.request_detail(.5, .52)
            release.set()
            self.wait_for_worker(dialog)
            self.assertEqual(compute.call_count, 2)
            self.assertEqual(len(published), 1)
            np.testing.assert_allclose(published[0], [.5, .52], atol=1 / 250000)
        # A fresh wheel gesture must supersede an earlier return-to-overview request,
        # even when the old worker finishes before the debounce timer expires.
        entered.clear()
        release.clear()
        published.clear()
        with patch.object(dialog._cache, "compute", side_effect=slow), \
             patch.object(dialog._view, "set_detail", side_effect=record):
            dialog.request_detail(.3, .32)
            self.assertTrue(entered.wait(5))
            dialog._restore_overview()
            self.assertTrue(dialog._restore_requested)
            dialog._zoom_time(.8, .02)
            self.assertFalse(dialog._restore_requested)
            release.set()
            self.wait_for_worker(dialog)
            self.assertEqual(len(published), 1)
            np.testing.assert_allclose(published[0], [.79, .81], atol=1 / 250000)
        dialog._zoom_time(.51, .01)
        self.assertTrue(dialog._zoom_timer.isActive())
        dialog._restore_overview()
        self.assertFalse(dialog._zoom_timer.isActive())
        self.assertIsNone(dialog._view._detail_data)
        self.assertEqual(dialog._view._meta["time_xlim"], meta["time_xlim"])
        dialog._zoom_time(.5, .01)
        dialog.accept()
        self.assertFalse(dialog._zoom_timer.isActive())

    def test_export_provenance_keeps_separated_audio(self):
        sr = 250000
        t = np.arange(5000) / sr
        component = np.sin(2 * np.pi * 47000 * t).astype(np.float32)
        isolated = self.folder / "isolated.wav"
        mixed = self.folder / "recording.wav"
        sf.write(isolated, component, sr, subtype="FLOAT")
        sf.write(mixed, np.concatenate((np.zeros(sr * 2), component)), sr, subtype="FLOAT")
        base = {"pulse_id": "p00001", "event_id": "e1", "predicted_label": "unknown",
                "start_s": 2., "end_s": 2.02, "source_path": str(mixed),
                "candidate_id": 1, "component_id": 1}
        row = dict(base, separation_applied=True, pulse_file=str(isolated))
        root = self.folder / "pulse_spectrograms"
        count = _export_batch_pulse_spectrograms([row], segment_id="recording", spectrogram_root=root,
            read_y=component, read_start_s=2., mode="separate", component_audio={(1, 1): component})
        self.assertEqual(count, 1)
        meta = load_spectrogram_meta(next(root.rglob("*.png")))
        self.assertEqual(Path(meta["source_audio"]["path"]), isolated)
        self.assertEqual(meta["source_audio"]["offset_s"], 0.)
        self.assertTrue(meta["source_audio"]["preprocessed"])
        row = dict(base)
        root = self.folder / "merged"
        self.assertEqual(_export_batch_pulse_spectrograms([row], segment_id="recording", spectrogram_root=root,
            read_y=component, read_start_s=2., mode="merged"), 1)
        meta = load_spectrogram_meta(next(root.rglob("*.png")))
        self.assertEqual(Path(meta["source_audio"]["path"]), mixed)
        self.assertEqual(meta["source_audio"]["offset_s"], 2.)


if __name__ == "__main__":
    unittest.main()
