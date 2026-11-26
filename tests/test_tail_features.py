"""CF-FM 尾段特征：CF 主线持续时，尾段 CF 下方的较弱能量也要被量化。"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("BAT_WORKSPACE", str(ROOT))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache" / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from bat_classifier.config import TARGET_SAMPLE_RATE, TEMPLATE_FEATURE_COLUMNS  # noqa: E402
from bat_classifier.features import (  # noqa: E402
    compute_pulse_spectrum,
    extract_pulse_features,
    extract_tail_features,
)
from bat_classifier.template_matching import build_template_model, classify_template_rows  # noqa: E402

SR = TARGET_SAMPLE_RATE


def cf_call(cf_hz: float = 67000.0, ms: float = 10.0, tail_ms: float = 0.0,
            tail_drop_hz: float = 4000.0, tail_gain: float = 0.6, noise: float = 0.0,
            seed: int = 3) -> np.ndarray:
    """CF 主线贯穿全程；可在末尾叠加一段与之并存、向下扫的较弱 FM 尾巴。"""
    count = int(SR * ms / 1000)
    t = np.arange(count) / SR
    envelope = np.sin(np.linspace(0, np.pi, count)) ** 0.2
    y = np.sin(2 * np.pi * cf_hz * t) * envelope
    if tail_ms > 0:
        tail_count = int(SR * tail_ms / 1000)
        tt = np.arange(tail_count) / SR
        sweep = cf_hz - 1500.0 - tail_drop_hz * tt / tt[-1]
        phase = 2 * np.pi * np.cumsum(sweep) / SR
        tail = tail_gain * np.sin(phase) * np.hanning(tail_count) ** 0.5
        y[-tail_count:] += tail
    if noise:
        y = y + np.random.default_rng(seed).normal(0, noise, count)
    return 0.8 * y / np.max(np.abs(y))


def tail_of(y: np.ndarray) -> dict:
    return extract_tail_features(compute_pulse_spectrum(y, SR))


class TailFeatureTests(unittest.TestCase):
    def test_pure_cf_has_no_tail_energy(self):
        for cf_hz, ms in ((47000.0, 30.0), (67000.0, 10.0), (105000.0, 20.0)):
            tail = tail_of(cf_call(cf_hz, ms))
            self.assertLess(tail["tail_low_energy_ratio"], 0.08, (cf_hz, ms, tail))
            self.assertLess(tail["tail_drop_khz"], 1.0, (cf_hz, ms, tail))

    def test_cffm_tail_beside_continuing_cf_is_detected(self):
        tail = tail_of(cf_call(67000.0, 10.0, tail_ms=2.5))
        self.assertGreater(tail["tail_low_energy_ratio"], 0.15, tail)
        self.assertGreater(tail["tail_drop_khz"], 1.5, tail)

    def test_noisy_cf_does_not_fake_a_tail(self):
        tail = tail_of(cf_call(72000.0, 20.0, noise=0.05))
        self.assertLess(tail["tail_low_energy_ratio"], 0.10, tail)

    def test_short_pulse_returns_zero_instead_of_nan(self):
        tail = tail_of(cf_call(67000.0, 0.4))
        self.assertEqual(set(tail), {"tail_low_energy_ratio", "tail_drop_khz"})
        self.assertTrue(all(np.isfinite(value) for value in tail.values()))

    def test_pulse_features_include_tail_columns(self):
        features = extract_pulse_features(cf_call(67000.0, 10.0, tail_ms=2.5), SR)
        for column in ("tail_low_energy_ratio", "tail_drop_khz"):
            self.assertNotIn(column, TEMPLATE_FEATURE_COLUMNS)
            self.assertTrue(np.isfinite(features[column]))

    def test_template_model_carries_tail_columns_end_to_end(self):
        # 合成声波较干净，旧特征也能分开；这里只验证新列贯通建模与匹配。
        # 真实模板上的区分效果见变更说明中的模板评估。
        model, rows = family_model_and_rows([cf_call(66500.0, 30.0), cf_call(66500.0, 10.2, tail_ms=2.5)])
        self.assertNotIn("tail_low_energy_ratio", model["feature_columns"])
        result = classify_template_rows(rows, model, min_similarity=0.0)
        self.assertEqual(list(result["raw_label"]), ["CF3_66K", "CFFM1_67K"])


def fm_cf_fm_call(cf_hz: float, ms: float) -> np.ndarray:
    """实测 CF 叫声：开头上扫 + 横频平台 + 末尾下扫。"""
    head = int(SR * 0.0015)
    tail = int(SR * 0.002)
    body = int(SR * ms / 1000) - head - tail
    freq = np.concatenate([
        np.linspace(cf_hz - 5000.0, cf_hz, head),
        np.full(body, cf_hz),
        np.linspace(cf_hz, cf_hz - 9000.0, tail),
    ])
    y = np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.sin(np.linspace(0, np.pi, len(freq))) ** 0.2
    return 0.8 * y


def family_model_and_rows(calls: list[np.ndarray]):
    """CF 模板只保留横频段（20–40 ms）；CF-FM 模板约 10 ms 且带尾巴。"""
    import pandas as pd

    with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as folder:
        for index, ms in enumerate((20.0, 30.0, 40.0), start=1):
            sf.write(Path(folder) / f"CF3_66K{index}.wav", cf_call(66500.0, ms), SR)
            sf.write(Path(folder) / f"CF1_47K{index}.wav", cf_call(46900.0, ms), SR)
        for index, ms in enumerate((9.5, 10.0, 10.5), start=1):
            sf.write(Path(folder) / f"CFFM1_67K{index}.wav", cf_call(67000.0, ms, tail_ms=2.5), SR)
        model = build_template_model(folder)
    rows = [extract_pulse_features(y, SR) | {"duration_ms": len(y) / SR * 1000} for y in calls]
    return model, pd.DataFrame(rows)


class TemplateFamilyTests(unittest.TestCase):
    def test_cf_family_ignores_fm_ends_of_field_calls(self):
        model, rows = family_model_and_rows([fm_cf_fm_call(46900.0, 25.0)])
        result = classify_template_rows(rows, model, min_similarity=0.6).iloc[0]
        self.assertEqual(result["predicted_label"], "CF1_47K")
        self.assertEqual(result["template_family"], "CF")
        self.assertGreater(rows.iloc[0]["tail_low_energy_ratio"], 0.15)

    def test_plateau_duration_decides_between_cf_and_cffm(self):
        model, rows = family_model_and_rows([
            cf_call(66800.0, 14.0, tail_ms=2.5),
            cf_call(66800.0, 9.0, tail_ms=2.5),
        ])
        result = classify_template_rows(rows, model, min_similarity=0.3, cf_min_plateau_ms=11.0)
        self.assertEqual(list(result["template_family_rule"]), ["plateau_long", "plateau_short"])
        self.assertEqual(list(result["predicted_label"]), ["CF3_66K", "CFFM1_67K"])
        self.assertTrue((result["template_cf_label"] == "CF3_66K").all())
        self.assertTrue((result["template_cffm_label"] == "CFFM1_67K").all())

    def test_plateau_threshold_can_be_overridden(self):
        model, rows = family_model_and_rows([cf_call(66800.0, 14.0, tail_ms=2.5)])
        result = classify_template_rows(rows, model, min_similarity=0.3, cf_min_plateau_ms=20.0)
        self.assertEqual(result.iloc[0]["predicted_label"], "CFFM1_67K")

    def test_old_snapshot_without_family_columns_keeps_single_best_match(self):
        model, rows = family_model_and_rows([cf_call(66800.0, 14.0, tail_ms=2.5)])
        for key in ("family_columns", "default_family_columns"):
            model.pop(key)
        result = classify_template_rows(rows, model, min_similarity=0.3)
        self.assertNotIn(result.iloc[0]["template_family_rule"], {"plateau_long", "plateau_short"})


class CfIntervalTests(unittest.TestCase):
    """CF 族按模板平台主频 [最低-2, 最高+2] kHz 区间识别。"""

    @classmethod
    def setUpClass(cls):
        model, _ = family_model_and_rows([cf_call(46900.0, 20.0)])
        cls.model = model
        freqs = [t["features"]["cf_plateau_khz"] for t in model["groups"]["CF1_47K"]["templates"]]
        cls.low, cls.high = min(freqs), max(freqs)

    def classify(self, row_overrides: dict):
        import pandas as pd

        row = extract_pulse_features(cf_call(46900.0, 20.0), SR) | {"duration_ms": 20.0} | row_overrides
        return classify_template_rows(pd.DataFrame([row]), self.model, min_similarity=0.6).iloc[0]

    def test_inside_margin_matches_and_reports_offset(self):
        result = self.classify({"cf_plateau_khz": self.high + 1.8})
        self.assertEqual(result["predicted_label"], "CF1_47K")
        self.assertEqual(result["template_family_rule"], "single")
        self.assertAlmostEqual(result["template_cf_offset_khz"], 1.8, places=6)
        self.assertEqual(result["template_cf_interval_khz"], f"{self.low - 2:.1f}-{self.high + 2:.1f}")

    def test_outside_margin_is_unknown(self):
        result = self.classify({"cf_plateau_khz": self.low - 2.2})
        self.assertEqual(result["predicted_label"], "unknown")
        self.assertEqual(result["template_cf_label"], "")

    def test_pulse_without_plateau_is_not_cf(self):
        result = self.classify({"cf_plateau_ms": 1.0})
        self.assertEqual(result["predicted_label"], "unknown")

    def test_fm_sweep_has_no_usable_plateau(self):
        sweep = np.linspace(60000.0, 35000.0, int(SR * 0.004))
        y = 0.8 * np.sin(2 * np.pi * np.cumsum(sweep) / SR) * np.hanning(len(sweep))
        self.assertLess(extract_pulse_features(y, SR)["cf_plateau_ms"], 2.0)



class PartialCsvTests(unittest.TestCase):
    def test_batches_with_different_columns_stay_aligned(self):
        import csv

        from bat_classifier.segment_stream import ChunkCheckpoint

        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as folder:
            checkpoint = ChunkCheckpoint(Path(folder), Path(folder) / "seg.wav")
            checkpoint.append_rows([{"a": 1, "b": 2}], columns=["a", "b"])
            checkpoint.append_rows([{"b": 20, "a": 10}], columns=["b", "a"])
            checkpoint.append_rows([{"a": 100, "c": "x", "b": 200}], columns=["a", "c", "b"])
            with open(checkpoint.partial_csv_path, newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual([(r["a"], r["b"], r["c"]) for r in rows],
                         [("1", "2", ""), ("10", "20", ""), ("100", "200", "x")])



class GateTailPathTests(unittest.TestCase):
    """门控 CF-FM 通道：尾段路径 + 与判族一致的短平台条件。"""

    @staticmethod
    def gate(y: np.ndarray) -> dict:
        from bat_classifier.pulse_table import create_pulse_rows

        context = np.pad(np.asarray(y, dtype=np.float32), (2500, 2500))
        rows = create_pulse_rows(
            Path("synthetic.wav"), context, SR,
            [dict(start_s=.01, end_s=.01 + len(y) / SR, candidate_start_s=.01,
                  candidate_end_s=.01 + len(y) / SR, snr_db=20., core_threshold_db=-24.)],
            save_separated_audio=False, include_template_track=True,
            harmonic_grouping_mode="enhanced",
        )
        return max(rows, key=lambda row: row["gate_score"])

    def test_short_call_with_fm_end_is_cffm(self):
        row = self.gate(fm_cf_fm_call(46900.0, 9.0))
        self.assertEqual(row["gate_shape"], "CF-FM")
        self.assertEqual(row["gate_tail_score"], 0.0)
        self.assertEqual(row["gate_cffm_source"], "track")
        self.assertEqual(row["gate_status"], "bat")

    def test_long_cf_with_fm_ends_stays_cf(self):
        row = self.gate(fm_cf_fm_call(46900.0, 25.0))
        self.assertEqual(row["gate_shape"], "CF")
        self.assertEqual(row["gate_cffm_score"], 0.0)
        self.assertEqual(row["gate_status"], "bat")

    def test_pure_cf_has_no_tail_support(self):
        row = self.gate(cf_call(67000.0, 10.0))
        self.assertEqual(row["gate_shape"], "CF")
        self.assertEqual(row["gate_tail_score"], 0.0)
        self.assertEqual(row["gate_cffm_source"], "")



class CffmTailRequirementTests(unittest.TestCase):
    def test_tail_not_required_by_default(self):
        model, rows = family_model_and_rows([cf_call(66800.0, 9.0)])
        self.assertIsNone(model["cffm_min_tail_ratio"])
        result = classify_template_rows(rows, model, min_similarity=0.3)
        self.assertEqual(result.iloc[0]["predicted_label"], "CFFM1_67K")

    def test_legacy_tail_requirement_is_ignored(self):
        model, rows = family_model_and_rows([cf_call(66800.0, 9.0), cf_call(66800.0, 9.0, tail_ms=2.5)])
        model["cffm_min_tail_ratio"] = 0.10
        result = classify_template_rows(rows, model, min_similarity=0.3)
        self.assertEqual(result.iloc[0]["predicted_label"], "CFFM1_67K")
        self.assertEqual(result.iloc[1]["predicted_label"], "CFFM1_67K")

    def test_legacy_snapshot_tail_features_cannot_affect_scores_or_validity(self):
        model, rows = family_model_and_rows([cf_call(66800.0, 9.0)])
        expected=classify_template_rows(rows,model,min_similarity=.3)
        for column in ('tail_low_energy_ratio','tail_drop_khz'):
            model['feature_columns'].append(column)
            model['default_family_columns'].append(column)
            rows[column]=float('nan')
            for group in model['groups'].values():
                group['scales'][column]=1e-20
                for template in group['templates']:
                    template['features'][column]=float('inf')
        model['cffm_min_tail_ratio']=.99
        actual=classify_template_rows(rows,model,min_similarity=.3)
        self.assertEqual(actual.predicted_label.tolist(),expected.predicted_label.tolist())
        np.testing.assert_allclose(actual.confidence,expected.confidence)


class BoundaryExtensionTests(unittest.TestCase):
    @staticmethod
    def call_with_faded_end(total_ms: float = 30.0, loud_ms: float = 12.0, fade_db: float = 22.0) -> np.ndarray:
        """前 loud_ms 为强 CF，其后同频 CF 低 fade_db（仍远高于底噪）。"""
        rng = np.random.default_rng(4)
        pad = int(SR * 0.2)  # 检测的背景取中位数，片段须以背景为主
        count = int(SR * total_ms / 1000)
        t = np.arange(count) / SR
        envelope = np.full(count, 10 ** (-fade_db / 20))
        envelope[:int(SR * loud_ms / 1000)] = 1.0
        call = 0.8 * envelope * np.sin(2 * np.pi * 46900.0 * t)
        y = np.concatenate([np.zeros(pad), call, np.zeros(pad)])
        return y + rng.normal(0, 0.0005, len(y)), pad / SR, (pad + count) / SR

    def test_faded_cf_end_is_kept(self):
        from bat_classifier.detection import refine_pulse_boundary

        y, start, end = self.call_with_faded_end()
        pulse = {"start_s": start, "end_s": start + 0.013}
        refined = refine_pulse_boundary(y, SR, pulse, max_duration_ms=100.0)
        self.assertIsNotNone(refined)
        self.assertGreater(refined["end_s"], end - 0.001)
        self.assertLess(refined["start_s"], start + 0.001)

    def test_long_cf_survives_with_template_max_duration(self):
        from bat_classifier.detection import detect_pulses, refine_all_pulses

        y, start, end = self.call_with_faded_end(total_ms=60.0, loud_ms=60.0)
        default = refine_all_pulses(y, SR, detect_pulses(y, SR)[0])
        self.assertFalse(any(p["end_s"] - p["start_s"] > 0.05 for p in default))
        raw = detect_pulses(y, SR, max_duration_ms=108.0)[0]
        longer = refine_all_pulses(y, SR, raw, max_duration_ms=108.0)
        self.assertTrue(any(p["end_s"] - p["start_s"] > 0.055 for p in longer))

    def _count_extension_passes(self, reaches_edge: bool) -> int:
        """假延伸：每次都能再多抓几帧；按 reaches_edge 报告是否碰到窗口边缘。"""
        from unittest.mock import patch

        import bat_classifier.detection as detection

        calls = []

        def fake(y, sr, start, end, freq, *, max_duration_ms, report_edges=False):
            calls.append((start, end))
            grown = (start - 64, end + 64)
            return (*grown, reaches_edge, reaches_edge) if report_edges else grown

        y, start, end = self.call_with_faded_end(total_ms=20.0, loud_ms=20.0)
        with patch.object(detection, "extend_pulse_boundary", side_effect=fake):
            detection.refine_pulse_boundary(y, SR, {"start_s": start, "end_s": end}, max_duration_ms=100.0)
        return len(calls)

    def test_extension_stops_unless_window_edge_reached(self):
        # 旧逻辑只要还能多抓几帧就继续，每条叫声都跑满上限、一点点蹭进噪声。
        self.assertEqual(self._count_extension_passes(reaches_edge=False), 1)

    def test_extension_continues_while_window_edge_reached(self):
        self.assertGreater(self._count_extension_passes(reaches_edge=True), 1)

    def test_overlap_requirement_follows_max_duration(self):
        from bat_classifier.segment_stream import minimum_overlap_seconds

        # 108ms calls also need 8ms more background context on each side.
        self.assertAlmostEqual(minimum_overlap_seconds(108.0) - minimum_overlap_seconds(40.0), 0.084, places=6)



class FragmentMergeTests(unittest.TestCase):
    @staticmethod
    def frag(start, end, freq, snr=10.0):
        return {"start_s": start, "end_s": end, "candidate_start_s": start, "candidate_end_s": end,
                "duration_ms": (end - start) * 1000, "snr_db": snr, "peak_freq_khz": freq,
                "detect_source": "fullband", "fragment_count": 1}

    def test_touching_same_frequency_fragments_merge(self):
        from bat_classifier.detection import merge_adjacent_fragments

        out = merge_adjacent_fragments([self.frag(0.100, 0.112, 59.1), self.frag(0.1126, 0.125, 59.0, snr=12.0)],
                                       max_duration_ms=108.0)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0]["start_s"], 0.100)
        self.assertAlmostEqual(out[0]["end_s"], 0.125)
        self.assertEqual(out[0]["fragment_count"], 2)

    def test_merge_skips_over_other_frequency(self):
        from bat_classifier.detection import merge_adjacent_fragments

        out = merge_adjacent_fragments([
            self.frag(0.100, 0.112, 59.1), self.frag(0.105, 0.110, 85.0), self.frag(0.114, 0.125, 59.0),
        ], max_duration_ms=108.0)
        self.assertEqual(sorted(round(p["peak_freq_khz"]) for p in out), [59, 85])

    def test_separate_calls_and_other_frequencies_stay_apart(self):
        from bat_classifier.detection import merge_adjacent_fragments

        out = merge_adjacent_fragments([
            self.frag(0.100, 0.140, 59.0), self.frag(0.170, 0.210, 59.0),   # 30 ms 间隔：两条叫声
            self.frag(0.105, 0.120, 85.0),                                   # 同时、不同频：另一只
        ], max_duration_ms=108.0)
        self.assertEqual(len(out), 3)

    def test_merge_respects_max_duration(self):
        from bat_classifier.detection import merge_adjacent_fragments

        out = merge_adjacent_fragments([self.frag(0.100, 0.160, 59.0), self.frag(0.162, 0.220, 59.0)],
                                       max_duration_ms=108.0)
        self.assertEqual(len(out), 2)

    def test_two_calls_with_fm_ends_and_short_gap_stay_apart(self):
        from bat_classifier.detection import detect_pulses, refine_all_pulses

        rng = np.random.default_rng(5)
        pad, gap = int(SR * 0.2), int(SR * 0.004)
        call = fm_cf_fm_call(84700.0, 30.0)
        y = np.concatenate([np.zeros(pad), call, np.zeros(gap), call, np.zeros(pad)])
        y = y + rng.normal(0, 0.003, len(y))
        out = refine_all_pulses(y, SR, detect_pulses(y, SR, max_duration_ms=108.0)[0], max_duration_ms=108.0)
        end = (2 * len(call) + gap) / SR + 0.2
        out = [p for p in out if p["end_s"] > 0.2 and p["start_s"] < end]
        self.assertEqual(len(out), 2, [(round(p["start_s"], 4), round(p["end_s"], 4)) for p in out])

    def test_dipped_long_cf_is_one_candidate(self):
        from bat_classifier.detection import detect_pulses, refine_all_pulses

        rng = np.random.default_rng(2)
        pad, count = int(SR * 0.2), int(SR * 0.045)
        t = np.arange(count) / SR
        env = np.ones(count); dip = slice(int(SR * 0.020), int(SR * 0.0225)); env[dip] = 0.02  # 2.5 ms 凹陷
        call = 0.6 * env * np.sin(2 * np.pi * 58600.0 * t)
        y = np.concatenate([np.zeros(pad), call, np.zeros(pad)]) + rng.normal(0, 0.003, 2 * pad + count)
        out = refine_all_pulses(y, SR, detect_pulses(y, SR, max_duration_ms=108.0)[0], max_duration_ms=108.0)
        # 只看与叫声重叠的候选（合成信号首尾的边缘瞬态不在考查范围）。
        out = [p for p in out if p["end_s"] > 0.2 and p["start_s"] < 0.245]
        self.assertEqual(len(out), 1)
        self.assertGreater(out[0]["end_s"] - out[0]["start_s"], 0.040)


if __name__ == "__main__":
    unittest.main()
