"""形态门控的行为验证：形态、噪声掩膜、统计和独立调试图片开关。"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("BAT_WORKSPACE", str(ROOT))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache" / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
(ROOT / ".tmp").mkdir(exist_ok=True)

import numpy as np

from bat_classifier.features import compute_pulse_spectrum
from bat_classifier.morphology_gate import score_morphology, unavailable_gate
from bat_classifier.pulse_table import create_pulse_rows
from bat_classifier.segment_analysis import (
    SegmentAnalysisConfig, _SegmentAccumulator, _export_batch_pulse_spectrograms,
    _finalize_chunk_rows, _image_outputs, _validate_checkpoint_image_outputs,
)
from bat_classifier.template_matching import build_template_model, classify_template_rows

SR = 250000


def signal(shape: str, duration_ms: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    count = int(SR * duration_ms / 1000)
    frequency = np.full(count, 47000.0)
    if shape == "FM":
        frequency = np.linspace(90000, 35000, count)
    elif shape in {"CF-FM", "FM-CF"}:
        tail = int(count * 0.30)
        if shape == "CF-FM":
            frequency[-tail:] = np.linspace(47000, 33000, tail)
        else:
            frequency[:tail] = np.linspace(65000, 47000, tail)
    elif shape == "QCF":
        frequency = np.linspace(47000, 45000, count)
    phase = 2 * np.pi * np.cumsum(frequency) / SR
    y = np.sin(phase)
    if shape == "harmonic":
        y += 0.8 * np.sin(2 * phase)
    y *= np.sin(np.linspace(0, np.pi, count)) ** 0.6
    return y.astype(np.float32), frequency


def pulse_rows(audio: np.ndarray) -> list[dict]:
    context = np.pad(audio, (2500, 2500))
    return create_pulse_rows(
        Path("synthetic.wav"), context, SR,
        [dict(start_s=.01, end_s=.01 + len(audio) / SR,
              candidate_start_s=.01, candidate_end_s=.01 + len(audio) / SR,
              snr_db=20., core_threshold_db=-24.)],
        save_separated_audio=False, include_template_track=True,
        harmonic_grouping_mode="enhanced",
    )


def fake_template_payload(features: dict, label: str = "CF4_72K", offset: float = 0.0) -> dict:
    """无文件的模板快照：行特征与模板一致时相似度为 1.0（offset 可造出低分）。"""
    from bat_classifier.config import TEMPLATE_FEATURE_COLUMNS
    values = {column: float(features[column]) + offset for column in TEMPLATE_FEATURE_COLUMNS}
    return {
        "kind": "bat_template_matching",
        "version": 1,
        "labels": [label],
        "unknown_label": "unknown",
        "min_similarity": 0.6,
        "feature_columns": list(TEMPLATE_FEATURE_COLUMNS),
        "groups": {
            label: {
                "templates": [{"file": "fake.wav", "name": "fake.wav", "features": values}],
                "scales": {column: 1.0 for column in TEMPLATE_FEATURE_COLUMNS},
            }
        },
        "template_count": 1,
        "max_pulse_ms": 40.0,
    }


class MorphologyGateTests(unittest.TestCase):
    def test_three_shapes_and_qcf_with_no_required_harmonics(self):
        for shape in ("CF", "FM", "CF-FM", "FM-CF", "QCF"):
            with self.subTest(shape=shape):
                rows = pulse_rows(signal(shape)[0])
                self.assertTrue(rows)
                strongest = max(rows, key=lambda r: r["gate_score"])
                self.assertEqual(strongest["gate_status"], "bat")
                expected = "CF-FM" if shape == "FM-CF" else "CF" if shape == "QCF" else shape
                self.assertEqual(strongest["gate_shape"], expected)

    def test_harmonic_family_remains_one_call(self):
        rows = pulse_rows(signal("harmonic")[0])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["gate_status"], "bat")
        self.assertGreater(rows[0]["gate_harmonic_support"], 0)

    def test_short_fast_sweeps_are_retained_for_review(self):
        for duration in (1., 2., 3., 5.):
            with self.subTest(duration=duration):
                rows = pulse_rows(signal("FM", duration)[0])
                self.assertTrue(rows)
                self.assertNotIn("noise", [row["gate_status"] for row in rows])

    def test_broadband_masks_do_not_manufacture_bat_calls(self):
        for seed in range(4):
            noise = np.random.default_rng(seed).normal(size=2500) * np.sin(np.linspace(0, np.pi, 2500))
            rows = pulse_rows(noise.astype(np.float32))
            self.assertTrue(rows)
            self.assertNotIn("bat", [row["gate_status"] for row in rows])

    def test_persistent_tone_uses_both_background_sides(self):
        y = np.sin(2 * np.pi * 47000 * np.arange(7500) / SR)
        spectrum = compute_pulse_spectrum(y[2500:5000], SR)
        result = score_morphology(spectrum, component={}, context_audio=y,
                                  start_sample=2500, end_sample=5000)
        self.assertEqual(result["gate_status"], "noise")
        self.assertEqual(result["gate_reason_code"], "persistent_tone")
        edge = score_morphology(spectrum, component={}, context_audio=y[:2500],
                                start_sample=0, end_sample=2500)
        self.assertFalse(edge["gate_background_available"])
        self.assertNotEqual(edge["gate_status"], "noise")

    def test_shared_spectrum_is_not_recomputed_or_mutated(self):
        import librosa
        audio, _ = signal("CF-FM")
        with patch.object(librosa, "stft", wraps=librosa.stft) as stft:
            rows = pulse_rows(audio)
        # 一次分离频谱 + 一次共用的建表/模板/门控频谱。
        self.assertEqual(stft.call_count, 2)
        self.assertTrue(rows[0]["template_track_khz"])
        spectrum = compute_pulse_spectrum(audio, SR)
        saved = spectrum.power.copy()
        score_morphology(spectrum, component={}, context_audio=np.pad(audio, (2500, 2500)),
                         start_sample=2500, end_sample=5000)
        np.testing.assert_array_equal(saved, spectrum.power)
        self.assertFalse(spectrum.power.flags.writeable)

    def test_short_evidence_stays_uncertain(self):
        audio, _ = signal("CF", .128)
        result = score_morphology(compute_pulse_spectrum(audio, SR), component={},
                                  context_audio=audio, start_sample=0, end_sample=len(audio))
        self.assertEqual(result["gate_status"], "uncertain")


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import soundfile as sf
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as tmp:
            sf.write(Path(tmp) / "CF1_47K1.wav", signal("CF")[0], SR)
            cls.payload = build_template_model(tmp)

    def test_gate_status_survives_matching_and_counts_are_disjoint(self):
        source = pulse_rows(signal("CF")[0])[0]
        rows = [{**source, "gate_status": state} for state in ("bat", "uncertain", "noise")]
        expected = classify_template_rows(__import__("pandas").DataFrame([source]), self.payload, min_similarity=0).iloc[0]
        config = SegmentAnalysisConfig(min_confidence=0)
        result, noises, next_id = _finalize_chunk_rows(rows, config, 1, self.payload)
        self.assertEqual([r["predicted_label"] for r in result], [expected["predicted_label"], "uncertain", "noise"])
        self.assertEqual(result[1]["template_predicted_label"], expected["predicted_label"])
        self.assertIsNone(result[2]["template_match_score"])
        acc = _SegmentAccumulator(self.payload["labels"], "unknown", config)
        acc.add_batch(raw_candidate_count=3, pulse_rows=rows, result_rows=result,
                      noise_count=noises, next_pulse_id=next_id, skipped=False)
        self.assertEqual((acc.bat_call_count, acc.gate_uncertain_count, acc.noise_count), (1, 1, 1))
        self.assertEqual(sum(acc.label_counts.values()), 1)
        restored = _SegmentAccumulator(acc.labels, "unknown", config)
        restored.from_state(acc.to_state())
        self.assertEqual(restored.to_state(), acc.to_state())

    def test_uncertain_switch_is_independent_and_does_not_change_counts(self):
        audio, _ = signal("CF")
        rows = [{**pulse_rows(audio)[0], **unavailable_gate(), "predicted_label": "uncertain",
                 "pulse_id": "p001", "event_id": "e1", "template_predicted_label": "CF1_47K"}]
        context = np.pad(audio, (2500, 2500))
        for mode in ("merged", "separate"):
            for regular, debug in ((False, False), (True, False), (False, True), (True, True)):
                with self.subTest(mode=mode, regular=regular, debug=debug), tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as tmp:
                    current = [dict(r) for r in rows]
                    with patch("bat_classifier.visualization.save_pulse_spectrogram", return_value=True) as save:
                        count = _export_batch_pulse_spectrograms(
                            current, segment_id="sample", spectrogram_root=Path(tmp) / "pulse_spectrograms",
                            read_y=context, read_start_s=0, mode=mode, save_regular=regular, save_uncertain=debug)
                    self.assertEqual(count, int(debug))
                    self.assertEqual(save.call_count, int(debug))
                    self.assertEqual(current[0]["gate_status"], "uncertain")
                    if debug:
                        self.assertIn("/uncertain/", current[0]["pulse_spectrogram_file"])
                    else:
                        self.assertNotIn("pulse_spectrogram_file", current[0])

    def test_checkpoint_rejects_debug_option_change(self):
        config = SegmentAnalysisConfig()
        state = {"image_outputs": _image_outputs(config)}
        _validate_checkpoint_image_outputs(state, config)
        config.save_uncertain_spectrograms = True
        with self.assertRaises(ValueError):
            _validate_checkpoint_image_outputs(state, config)

    def test_only_noise_skips_templates_and_unknown_is_still_bat(self):
        source = pulse_rows(signal("CF")[0])[0]
        with patch("bat_classifier.segment_analysis.classify_template_rows", wraps=classify_template_rows) as match:
            result, count, _ = _finalize_chunk_rows(
                [{**source, "gate_status": "noise"}], SegmentAnalysisConfig(), 1, self.payload)
            self.assertTrue(match.call_args.args[0].empty)
        self.assertEqual(count, 1)
        self.assertEqual(result[0]["predicted_label"], "noise")
        distant = {**source, "peak_freq_khz": 120.0, "cf_plateau_khz": 120.0}
        result, _, _ = _finalize_chunk_rows([distant], SegmentAnalysisConfig(min_confidence=1), 1, self.payload)
        self.assertEqual(result[0]["gate_status"], "bat")
        self.assertEqual(result[0]["predicted_label"], "unknown")
        self.assertEqual(result[0]["template_predicted_label"], "unknown")


class LocalConcentrationTests(unittest.TestCase):
    """局部带集中度：只作旁路使用（救回 / 宽带瞬态），不替代全带阈值。"""

    @staticmethod
    def _spectrum(power, frequencies, centers, *, sample_count=8000, hop=16):
        from bat_classifier.features import PulseSpectrum
        return PulseSpectrum(
            sample_count=sample_count, sample_rate=SR, hop_length=hop,
            power=power, frequencies=frequencies,
            frame_peak_power=power.max(axis=0), dominant_freqs_hz=centers,
        )

    @classmethod
    def _grid(cls, high_hz=120000.0):
        frequencies = np.fft.rfftfreq(512, 1.0 / SR)
        keep = (frequencies >= 15000) & (frequencies <= high_hz)
        return frequencies[keep]

    def test_mask_concentrations_flat_noise_baseline_matches_bandwidth_ratio(self):
        from bat_classifier.features import mask_concentrations
        frequencies = self._grid()
        power = np.ones((len(frequencies), 20))
        centers = np.full(20, 47000.0)
        near = np.abs(frequencies[:, None] - centers[None, :]) <= 1500.0
        local = np.abs(frequencies[:, None] - centers[None, :]) <= 6000.0
        full, local_value = mask_concentrations(power, near, local)
        # 平谱：全带值 ≈ 3kHz/105kHz，局部带值 ≈ 3kHz/12kHz。
        self.assertAlmostEqual(full, 3.0 / 105.0, delta=0.02)
        self.assertAlmostEqual(local_value, 0.25, delta=0.04)

    def test_rescue_path_keeps_a_clean_tone_in_a_noisy_band(self):
        """47 kHz 窄带叫声 + 100 kHz 强宽带驼峰：全带集中度很低但局部带干净。"""
        from bat_classifier.morphology_gate import score_morphology
        frequencies = self._grid()
        tone = np.exp(-0.5 * ((frequencies - 47000.0) / 400.0) ** 2)
        hump = 8.0 * np.exp(-0.5 * ((frequencies - 100000.0) / 9000.0) ** 2)
        power = np.tile(tone + hump, (120, 1)).T * 1e-6
        spectrum = self._spectrum(power, frequencies, np.full(120, 47000.0))
        result = score_morphology(spectrum, component={}, context_audio=np.zeros(8000, np.float32),
                                  start_sample=0, end_sample=8000)
        self.assertEqual(result["gate_status"], "bat")
        self.assertEqual(result["gate_pass_path"], "rescue")
        self.assertLess(result["gate_concentration"], 0.25)
        self.assertGreaterEqual(result["gate_concentration_local"], 0.90)
        self.assertFalse(any(result[f"gate_fail_{key}"] for key in
                             ("score", "quality", "concentration", "coverage",
                              "continuity", "duration", "concentration_local")))

    def test_broadband_impulse_is_noise_not_uncertain(self):
        """中心乱跳的宽带驼峰：全带和局部带都没有窄带峰 → 直接判噪声。"""
        from bat_classifier.morphology_gate import score_morphology
        frequencies = self._grid()
        rng = np.random.default_rng(3)
        centers = 47000.0 + np.cumsum(rng.normal(0, 750.0, 120))
        power = np.zeros((len(frequencies), 120))
        for index, center in enumerate(centers):
            power[:, index] = np.exp(-0.5 * ((frequencies - center) / (20000.0 / 2.355)) ** 2)
        spectrum = self._spectrum(power * 1e-6, frequencies, centers)
        result = score_morphology(spectrum, component={}, context_audio=np.zeros(8000, np.float32),
                                  start_sample=0, end_sample=8000)
        self.assertEqual(result["gate_status"], "noise")
        self.assertEqual(result["gate_reason_code"], "broadband_impulse")
        self.assertTrue(result["gate_fail_concentration"])
        self.assertEqual(result["gate_pass_path"], "")

    def test_fail_columns_only_describe_non_bat_rows(self):
        audio, _ = signal("CF")
        rows = pulse_rows(audio)
        self.assertTrue(rows)
        row = rows[0]
        self.assertEqual(row["gate_status"], "bat")
        self.assertFalse(any(row[f"gate_fail_{key}"] for key in
                             ("score", "quality", "concentration", "coverage",
                              "continuity", "duration", "concentration_local")))
        self.assertIn(row["gate_pass_path"], {"band", "rescue"})

    def test_spectrogram_title_shows_template_score_and_local_concentration(self):
        audio, _ = signal("CF")
        row = {**pulse_rows(audio)[0], **unavailable_gate()}
        row.update({
            "predicted_label": "CF1_47K", "template_predicted_label": "CF1_47K",
            "template_match_score": 0.73, "pulse_id": "p001", "event_id": "e1",
            "gate_status": "bat", "gate_pass_path": "band",
        })
        root = ROOT / ".tmp" / "title_spectrogram"
        context = np.pad(audio, (2500, 2500))
        with patch("bat_classifier.visualization.save_pulse_spectrogram", return_value=True) as save:
            _export_batch_pulse_spectrograms(
                [row], segment_id="sample", spectrogram_root=root / "pulse_spectrograms",
                read_y=context, read_start_s=0, mode="merged",
                save_regular=True, save_uncertain=False,
            )
        title = "\n".join(save.call_args.args[3])
        self.assertIn("template=CF1_47K (0.73)", title)
        self.assertIn("C_loc=", title)


    def test_unknown_images_use_their_own_switch(self):
        audio, _ = signal("CF")
        context = np.pad(audio, (2500, 2500))
        base = {**pulse_rows(audio)[0], **unavailable_gate(),
                "gate_status": "bat", "gate_pass_path": "band",
                "event_id": "e1", "pulse_spectrogram_mode": "merged"}
        unknown_row = {**base, "pulse_id": "p001", "predicted_label": "unknown",
                       "template_predicted_label": "unknown", "template_match_score": 0.2}
        root = ROOT / ".tmp" / "unknown_switch"
        # 常规图片开关不再包含 unknown。
        with patch("bat_classifier.visualization.save_pulse_spectrogram", return_value=True) as save:
            count = _export_batch_pulse_spectrograms(
                [dict(unknown_row)], segment_id="sample", spectrogram_root=root / "off",
                read_y=context, read_start_s=0, mode="merged",
                save_regular=True, save_uncertain=False, save_unknown=False)
        self.assertEqual(count, 0)
        self.assertEqual(save.call_count, 0)
        # 单独开启 unknown 开关才输出，并落在 unknown/ 目录。
        row = dict(unknown_row)
        with patch("bat_classifier.visualization.save_pulse_spectrogram", return_value=True):
            count = _export_batch_pulse_spectrograms(
                [row], segment_id="sample", spectrogram_root=root / "on",
                read_y=context, read_start_s=0, mode="merged",
                save_regular=False, save_uncertain=False, save_unknown=True)
        self.assertEqual(count, 1)
        self.assertIn("/unknown/", row["pulse_spectrogram_file"])


class TemplatePromotionTests(unittest.TestCase):
    """V1.5 模板放行：只在缺能量证据、且模板与形态互相印证时生效。"""

    @staticmethod
    def _row() -> dict:
        audio, _ = signal("CF")
        source = pulse_rows(audio)[0]
        return {
            **source, **unavailable_gate(),
            "gate_status": "uncertain", "gate_shape": "CF",
            "gate_fail_score": True, "gate_fail_concentration": True,
            "gate_fail_concentration_local": True,
        }

    def test_promotion_is_off_by_default(self):
        row = self._row()
        result, _, _ = _finalize_chunk_rows(
            [row], SegmentAnalysisConfig(min_confidence=0.6), 1,
            fake_template_payload(row),
        )
        self.assertEqual(result[0]["gate_status"], "uncertain")
        self.assertFalse(result[0]["gate_promoted"])
        self.assertEqual(result[0]["predicted_label"], "uncertain")

    def test_energy_only_failure_is_promoted_and_marked(self):
        row = self._row()
        config = SegmentAnalysisConfig(min_confidence=0.6, allow_template_promotion=True)
        result, _, _ = _finalize_chunk_rows([row], config, 1, fake_template_payload(row))
        promoted = result[0]
        self.assertEqual(promoted["gate_status"], "bat")
        self.assertEqual(promoted["gate_pass_path"], "template")
        self.assertTrue(promoted["gate_promoted"])
        self.assertEqual(promoted["gate_reason_code"], "template_promoted")
        self.assertEqual(promoted["predicted_label"], "CF4_72K")
        self.assertEqual(promoted["template_predicted_label"], "CF4_72K")
        self.assertFalse(promoted["is_uncertain"])
        # 原始未达标项保留，便于事后剔除重算。
        self.assertTrue(promoted["gate_fail_concentration"])

    def test_integrity_failure_blocks_promotion(self):
        for key in ("coverage", "continuity", "duration", "quality"):
            with self.subTest(blocker=key):
                row = self._row()
                row[f"gate_fail_{key}"] = True
                config = SegmentAnalysisConfig(min_confidence=0.6, allow_template_promotion=True)
                result, _, _ = _finalize_chunk_rows([row], config, 1, fake_template_payload(row))
                self.assertEqual(result[0]["gate_status"], "uncertain")
                self.assertFalse(result[0]["gate_promoted"])

    def test_shape_mismatch_blocks_promotion(self):
        row = self._row()
        row["gate_shape"] = "FM"  # 模板是 CF 类
        config = SegmentAnalysisConfig(min_confidence=0.6, allow_template_promotion=True)
        result, _, _ = _finalize_chunk_rows([row], config, 1, fake_template_payload(row))
        self.assertEqual(result[0]["gate_status"], "uncertain")

    def test_low_template_similarity_blocks_promotion(self):
        row = self._row()
        config = SegmentAnalysisConfig(min_confidence=0.9, allow_template_promotion=True)
        payload = fake_template_payload(row, offset=40.0)  # 相似度远低于阈值
        result, _, _ = _finalize_chunk_rows([row], config, 1, payload)
        self.assertEqual(result[0]["gate_status"], "uncertain")
        self.assertFalse(result[0]["gate_promoted"])

    def test_noise_rows_are_never_promoted(self):
        row = {**self._row(), "gate_status": "noise"}
        config = SegmentAnalysisConfig(min_confidence=0.6, allow_template_promotion=True)
        result, noise_count, _ = _finalize_chunk_rows([row], config, 1, fake_template_payload(row))
        self.assertEqual(noise_count, 1)
        self.assertEqual(result[0]["gate_status"], "noise")
        self.assertEqual(result[0]["predicted_label"], "noise")

    def test_promoted_rows_are_counted_as_bat_and_tallied(self):
        row = self._row()
        config = SegmentAnalysisConfig(min_confidence=0.6, allow_template_promotion=True)
        result, noise_count, next_id = _finalize_chunk_rows([row], config, 1, fake_template_payload(row))
        accumulator = _SegmentAccumulator(["CF4_72K"], "unknown", config)
        accumulator.add_batch(raw_candidate_count=1, pulse_rows=[row], result_rows=result,
                             noise_count=noise_count, next_pulse_id=next_id, skipped=False)
        self.assertEqual(accumulator.bat_call_count, 1)
        self.assertEqual(accumulator.gate_uncertain_count, 0)
        self.assertEqual(accumulator.template_promoted_count, 1)
        restored = _SegmentAccumulator(["CF4_72K"], "unknown", config)
        restored.from_state(accumulator.to_state())
        self.assertEqual(restored.template_promoted_count, 1)

    def test_checkpoint_rejects_promotion_change(self):
        from bat_classifier.segment_analysis import _validate_checkpoint_promotion
        config = SegmentAnalysisConfig()
        state = {"template_promotion": False, "cf_min_plateau_ms": config.cf_min_plateau_ms}
        _validate_checkpoint_promotion(state, config)
        config.allow_template_promotion = True
        with self.assertRaises(ValueError):
            _validate_checkpoint_promotion(state, config)

    def test_checkpoint_rejects_cf_plateau_change(self):
        from bat_classifier.segment_analysis import _validate_checkpoint_promotion
        config = SegmentAnalysisConfig()
        state = {"template_promotion": False, "cf_min_plateau_ms": config.cf_min_plateau_ms}
        config.cf_min_plateau_ms += 2.0
        with self.assertRaises(ValueError):
            _validate_checkpoint_promotion(state, config)
        with self.assertRaises(ValueError):
            _validate_checkpoint_promotion({"template_promotion": False}, SegmentAnalysisConfig())


class GuiTests(unittest.TestCase):
    def test_debug_checkbox_controls_config_busy_state_and_summary(self):
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QFont, QFontDatabase
        from bat_classifier.app.segment_analysis_tab import SegmentAnalysisTab
        from bat_classifier.segment_analysis import summarize_segment_results
        app = QApplication.instance() or QApplication([])
        for font in ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/msyh.ttc"):
            if Path(font).exists():
                QFontDatabase.addApplicationFont(font)
        app.setFont(QFont("Microsoft YaHei", 9))
        tab = SegmentAnalysisTab(lambda _: None, lambda _: None)
        self.assertFalse(tab._save_uncertain_check.isChecked())
        self.assertFalse(tab._save_unknown_check.isChecked())
        self.assertFalse(tab._template_promote_check.isChecked())
        tab._save_images_check.setChecked(False)
        self.assertFalse(tab._pulse_spectrogram_mode_combo.isEnabled())
        tab._save_uncertain_check.setChecked(True)
        self.assertTrue(tab._pulse_spectrogram_mode_combo.isEnabled())
        tab._save_uncertain_check.setChecked(False)
        self.assertFalse(tab._pulse_spectrogram_mode_combo.isEnabled())
        tab._save_unknown_check.setChecked(True)
        self.assertTrue(tab._pulse_spectrogram_mode_combo.isEnabled())
        config = SegmentAnalysisConfig(**tab._collect_config()).validate()
        self.assertTrue(config.save_unknown_spectrograms)
        self.assertFalse(config.save_uncertain_spectrograms)
        self.assertFalse(config.allow_template_promotion)
        self.assertFalse(config.save_pulse_spectrograms)
        tab._set_busy(True)
        self.assertFalse(tab._save_uncertain_check.isEnabled())
        self.assertFalse(tab._save_unknown_check.isEnabled())
        self.assertFalse(tab._template_promote_check.isEnabled())
        self.assertFalse(tab._pulse_spectrogram_mode_combo.isEnabled())
        tab._set_busy(False)
        self.assertTrue(tab._save_uncertain_check.isEnabled())
        summary, _ = summarize_segment_results(
            [{"label_counts": {"CF": 1, "unknown": 1}, "detected_call_count": 2,
              "gate_uncertain_count": 3, "template_promoted_count": 1,
              "noise_count": 4}], ["CF"], "unknown")
        tab._labels = ["CF"]
        tab._populate_table(summary)
        self.assertEqual(tab._result_model.columnCount(), 8)
        self.assertEqual(tab._result_model.item(0, 5).text(), "3")
        self.assertEqual(tab._result_model.item(0, 6).text(), "1")
        self.assertEqual(tab._result_model.item(0, 7).text(), "4")
        tab.close()


if __name__ == "__main__":
    unittest.main()
