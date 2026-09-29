"""⑦ 超长 WAV 流式分析（本轮长录音优化）的回归护栏测试。

覆盖本轮要求：
1. 长 WAV 流式读取（chunk+overlap 网格、覆盖铺满、最小重叠校验）；
2. overlap 区间与边界 pulse 不丢失、不重复（跨块归属 + 去重安全网）；
3. 断点续跑（checkpoint/取消/文件变更防错误续跑/续跑后不重复计数）；
4. 静音快筛保守（弱脉冲不丢、快筛开/关计数一致）；
5. 并行/单线程计数与结果完全一致；
6. CSV/PNG/pulse WAV 导出上限绝不改变计数；
7. 短 WAV 与优化前结果完全一致（golden 回归）；
8. 全时段统计（FULL RECORDING）+ 速率分母 = 有效监测总时长。
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tests.test_segment_analysis import (  # noqa: E402
    SR,
    _make_case_dir,
    _noise_only,
    _remove_case_dir,
    _segment_with_calls,
    _sweep,
    _write,
)

from bat_classifier import segment_stream  # noqa: E402
from bat_classifier.config import (  # noqa: E402
    CLUSTER_FEATURE_COLUMNS,
    NOISE_GATE_FEATURE_COLUMNS,
)
from bat_classifier.segment_analysis import (  # noqa: E402
    SegmentAnalysisConfig,
    analyze_segment,
)
from bat_classifier.segment_stream import (  # noqa: E402
    ChunkCheckpoint,
    build_chunk_specs,
    chunk_may_contain_ultrasound,
    dedupe_cross_chunk_rows,
    minimum_overlap_seconds,
    probe_audio,
    StreamStats,
)


# ---------------------------------------------------------------------------
# 测试工具
# ---------------------------------------------------------------------------
class DeterministicClassifier:
    """行内容决定预测（与 batch 切分无关），置信度固定 0.7。"""

    classes_ = np.array(["A", "B", "C", "D"])

    def _labels(self, X):
        dur = np.asarray(X["duration_ms"], dtype=float)
        idx = np.round(dur).astype(int) % 4
        return np.array(["A", "B", "C", "D"])[idx]

    def predict(self, X):
        return self._labels(X)

    def predict_proba(self, X):
        labels = self._labels(X)
        out = np.full((len(X), 4), 0.1)
        for row, label in enumerate(labels):
            out[row, list(self.classes_).index(label)] = 0.7
        return out


class AllBatNoiseModel:
    classes_ = np.array(["bat", "noise"])

    def predict_proba(self, X):
        return np.tile([0.99, 0.01], (len(X), 1))


def _deterministic_payload() -> dict:
    return {
        "noise_gate": {
            "model": AllBatNoiseModel(),
            "feature_columns": list(NOISE_GATE_FEATURE_COLUMNS),
            "classes": ["bat", "noise"],
            "bat_probability_threshold": 0.5,
        },
        "acoustic_types": {
            "scaler": None,
            "kmeans": None,
            "temporary_classifier": DeterministicClassifier(),
            "raw_cluster_to_type": {},
            "feature_columns": list(CLUSTER_FEATURE_COLUMNS),
            "labels": ["A", "B", "C", "D"],
        },
    }


def _row_keys(result: dict) -> list[tuple[float, float]]:
    return [
        (round(float(row["start_s"]), 3), round(float(row["end_s"]), 3))
        for row in result.get("pulse_rows", [])
        if str(row.get("predicted_label", "")) != "noise"
    ]


def _long_file_with_calls(
    path: Path,
    length_s: float,
    calls: list[tuple[float, float, float]],
    quiet_floor: float = 0.0008,
) -> Path:
    """构造长 WAV：安静底噪 + 指定 (start_s, end_khz, amplitude) 的扫频 call。"""
    total = np.zeros(int(SR * length_s), dtype=np.float32)
    rng = np.random.default_rng(7)
    total += rng.uniform(
        -quiet_floor, quiet_floor, size=len(total)
    ).astype(np.float32)
    for start_s, end_khz, amplitude in calls:
        sweep = _sweep(95.0, end_khz, amplitude=amplitude)
        start = int(start_s * SR)
        end = min(len(total), start + len(sweep))
        total[start:end] += sweep[: end - start]
    return _write(path, np.clip(total, -1.0, 1.0))


class _StreamTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = _make_case_dir()
        self.cfg = SegmentAnalysisConfig(
            min_confidence=0.5,
            low_confidence_action="uncertain",
            save_pulse_audio=False,
            save_figures=False,
        ).validate()

    def tearDown(self) -> None:
        _remove_case_dir(self.root)


# ---------------------------------------------------------------------------
# 1：chunk 网格（流式读取的骨架）
# ---------------------------------------------------------------------------
class ChunkGridTest(_StreamTestCase):
    def test_coverage_covers_full_recording_without_gaps(self) -> None:
        total_samples = int(SR * 61.3)
        specs = build_chunk_specs(total_samples, SR)
        self.assertGreater(len(specs), 1)
        # coverage 连续铺满 0 → 总时长
        self.assertAlmostEqual(specs[0].coverage_start_s, 0.0)
        for prev, nxt in zip(specs, specs[1:]):
            self.assertAlmostEqual(prev.coverage_end_s, nxt.coverage_start_s)
        self.assertAlmostEqual(
            specs[-1].coverage_end_s, total_samples / SR, places=6
        )
        # read 区间向两侧扩 overlap，首尾贴文件边界
        self.assertAlmostEqual(specs[0].read_start_s, 0.0)
        self.assertAlmostEqual(specs[-1].read_end_s, total_samples / SR, places=6)
        for spec in specs:
            self.assertGreaterEqual(spec.read_start_s, 0.0)
            self.assertGreaterEqual(spec.coverage_start_s, spec.read_start_s)
            self.assertLessEqual(spec.read_end_s, total_samples / SR + 1e-9)

    def test_overlap_below_minimum_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_chunk_specs(SR * 10, SR, overlap_seconds=0.05)
        # 边界值：恰好等于最小重叠必须放行
        specs = build_chunk_specs(
            SR * 10, SR, overlap_seconds=minimum_overlap_seconds()
        )
        self.assertTrue(specs)

    def test_minimum_overlap_covers_pulse_and_context(self) -> None:
        from bat_classifier.config import (
            CORE_CONTEXT_MS,
            CORE_MAX_GAP_MS,
            CORE_PADDING_MS,
            MAX_PULSE_MS,
            MERGE_GAP_MS,
            PULSE_PADDING_MS,
        )

        lower_bound_ms = (
            MAX_PULSE_MS
            + 2 * PULSE_PADDING_MS
            + 2 * CORE_CONTEXT_MS
            + 2 * CORE_PADDING_MS
            + MERGE_GAP_MS
        )
        self.assertGreaterEqual(
            minimum_overlap_seconds(), lower_bound_ms / 1000.0
        )

    def test_probe_audio_reports_duration(self) -> None:
        wav = _write(self.root / "probe.wav", _segment_with_calls(1, 2.0))
        frames, sr, duration = probe_audio(wav)
        self.assertEqual(sr, SR)
        self.assertAlmostEqual(duration, frames / SR, places=6)
        self.assertAlmostEqual(duration, 2.0, places=2)


# ---------------------------------------------------------------------------
# 2：边界 pulse 不丢不重 + global 时间正确
# ---------------------------------------------------------------------------
class BoundaryPulseTest(_StreamTestCase):
    def _run_with_small_chunks(self, wav: Path, workers: int = 1) -> dict:
        with mock.patch.object(
            segment_stream, "SEGMENT_STREAM_CHUNK_SECONDS", 2.0
        ):
            return analyze_segment(
                wav,
                None,
                self.cfg,
                max_workers=workers,
            )

    def test_boundary_pulse_counted_exactly_once(self) -> None:
        # pulse 恰好横跨 chunk 边界 2.0s（起于 1.995，宽 8ms，中心 1.999s）
        total = np.zeros(int(SR * 6.0), dtype=np.float32)
        sweep = _sweep(90, 55)
        start = int(1.995 * SR)
        total[start : start + len(sweep)] = sweep
        wav = _write(self.root / "boundary.wav", total)

        result = self._run_with_small_chunks(wav)
        keys = _row_keys(result)
        self.assertEqual(
            len(keys), 1, f"边界 pulse 必须恰好计数一次，实际 {keys}"
        )
        (start_s, end_s), = keys
        self.assertAlmostEqual(start_s, 1.995, delta=0.01)

    def test_pulse_inside_overlap_not_doubled(self) -> None:
        # pulse 完整落在两个 chunk 的 read 重叠区内（1.7s < 2.0s 边界）
        total = np.zeros(int(SR * 6.0), dtype=np.float32)
        sweep = _sweep(85, 50)
        start = int(1.70 * SR)
        total[start : start + len(sweep)] = sweep
        wav = _write(self.root / "overlap.wav", total)

        result = self._run_with_small_chunks(wav)
        keys = _row_keys(result)
        self.assertEqual(len(keys), 1, f"重叠区 pulse 不得重复计数：{keys}")

    def test_times_are_global_absolute(self) -> None:
        expected_starts = [0.35, 3.10, 5.25, 8.40]
        total = np.zeros(int(SR * 10.0), dtype=np.float32)
        for offset in expected_starts:
            sweep = _sweep(90 - 6 * expected_starts.index(offset), 55)
            start = int(offset * SR)
            total[start : start + len(sweep)] = sweep
        wav = _write(self.root / "global.wav", total)

        result = self._run_with_small_chunks(wav)
        keys = _row_keys(result)
        self.assertEqual(len(keys), 4)
        for (start_s, _), expected in zip(sorted(keys), expected_starts):
            self.assertAlmostEqual(
                start_s, expected, delta=0.02,
                msg="pulse 时间必须是 global 绝对时间",
            )

    def test_cross_chunk_dedupe_helper_keeps_higher_snr(self) -> None:
        old = [{"start_s": 1.0, "end_s": 1.01, "snr_db": 12.0, "_chunk_index": 0}]
        new = [
            {"start_s": 1.001, "end_s": 1.011, "snr_db": 18.0, "_chunk_index": 1},
            {"start_s": 1.2, "end_s": 1.21, "snr_db": 9.0, "_chunk_index": 1},
        ]
        kept = dedupe_cross_chunk_rows(old, new)
        self.assertEqual(len(kept), 1)
        self.assertAlmostEqual(kept[0]["start_s"], 1.2)
        # 同 chunk 内的行互不去重（可能是合法的多轨迹行）
        same_chunk = [
            {"start_s": 1.0, "end_s": 1.01, "snr_db": 5.0, "_chunk_index": 3},
            {"start_s": 1.0, "end_s": 1.01, "snr_db": 7.0, "_chunk_index": 3},
        ]
        self.assertEqual(
            len(dedupe_cross_chunk_rows([same_chunk[0]], [same_chunk[1]])), 1
        )
        self.assertEqual(
            len(dedupe_cross_chunk_rows([], same_chunk)), 2
        )


# ---------------------------------------------------------------------------
# 8：FULL RECORDING 统计 + 速率分母（本轮核心修复）
# ---------------------------------------------------------------------------
class FullRecordingStatisticsTest(_StreamTestCase):
    def test_30s_twelve_calls_all_counted(self) -> None:
        wav = _write(self.root / "seg_30s.wav", _segment_with_calls(12, 30.0))
        result = analyze_segment(wav, _deterministic_payload(), self.cfg)
        self.assertEqual(result["detected_call_count"], 12)
        self.assertEqual(result["raw_candidate_count"], 12)

    def test_rate_denominator_is_valid_monitoring_duration(self) -> None:
        wav = _write(self.root / "seg_30s.wav", _segment_with_calls(12, 30.0))
        result = analyze_segment(wav, _deterministic_payload(), self.cfg)
        self.assertAlmostEqual(result["recording_duration_s"], 30.0, places=2)
        self.assertAlmostEqual(
            result["valid_monitoring_duration_s"], 30.0, places=2
        )
        self.assertAlmostEqual(
            result["calls_per_second"],
            result["detected_call_count"] / result["valid_monitoring_duration_s"],
            places=6,
        )
        self.assertAlmostEqual(result["calls_per_second"], 0.4, places=6)

    def test_chunk_accounting_fields_present(self) -> None:
        wav = _write(self.root / "seg_30s.wav", _segment_with_calls(4, 10.0))
        result = analyze_segment(wav, None, self.cfg)
        for key in (
            "recording_duration_s",
            "valid_monitoring_duration_s",
            "failed_duration_s",
            "processed_chunk_count",
            "fast_skipped_chunk_count",
            "chunk_seconds",
            "overlap_seconds",
        ):
            self.assertIn(key, result)
        self.assertGreater(result["processed_chunk_count"], 0)


# ---------------------------------------------------------------------------
# 4：静音快筛（保守，recall 优先）
# ---------------------------------------------------------------------------
class FastSkipTest(_StreamTestCase):
    def test_flat_quiet_chunk_is_skipped(self) -> None:
        flat = np.full(int(SR * 2.0), 0.001, dtype=np.float32)
        self.assertFalse(chunk_may_contain_ultrasound(flat, SR))

    def test_weak_pulse_in_quiet_chunk_never_skipped(self) -> None:
        rng = np.random.default_rng(3)
        noise = rng.uniform(-0.001, 0.001, size=int(SR * 2.0)).astype(np.float32)
        weak = _sweep(80, 55, amplitude=0.006)  # 弱但远高于底噪
        start = int(1.0 * SR)
        noise[start : start + len(weak)] += weak
        self.assertTrue(
            chunk_may_contain_ultrasound(noise, SR),
            "弱真实脉冲的 chunk 绝不允许快筛跳过",
        )

    def test_fastskip_on_off_identical_counts(self) -> None:
        calls = [
            (1.2, 60.0, 0.8),
            (6.5, 55.0, 0.008),   # 弱 call，放在安静段
            (12.3, 50.0, 0.8),
            (18.0, 52.0, 0.012),  # 另一个弱 call
        ]
        wav = _long_file_with_calls(self.root / "quiet_calls.wav", 24.0, calls)

        with mock.patch.object(
            segment_stream, "SEGMENT_STREAM_CHUNK_SECONDS", 2.0
        ):
            with mock.patch.object(
                segment_stream, "SEGMENT_FASTSKIP_ENABLED", True
            ):
                result_fast = analyze_segment(wav, None, self.cfg, max_workers=1)
            with mock.patch.object(
                segment_stream, "SEGMENT_FASTSKIP_ENABLED", False
            ):
                result_full = analyze_segment(wav, None, self.cfg, max_workers=1)

        self.assertEqual(
            result_fast["detected_call_count"],
            result_full["detected_call_count"],
            "快筛开/关必须得到相同 call 数（不丢 call）",
        )
        self.assertEqual(_row_keys(result_fast), _row_keys(result_full))

    def test_skipped_chunks_still_count_in_denominator(self) -> None:
        # 纯静音长文件：快筛可跳过块，但速率分母仍是整条录音时长
        wav = _write(self.root / "silence.wav", _noise_only(12.0))
        with mock.patch.object(
            segment_stream, "SEGMENT_STREAM_CHUNK_SECONDS", 2.0
        ):
            result = analyze_segment(wav, None, self.cfg)
        self.assertEqual(result["detected_call_count"], 0)
        self.assertAlmostEqual(
            result["valid_monitoring_duration_s"], 12.0, places=2
        )
        self.assertAlmostEqual(result["calls_per_second"], 0.0, places=6)


# ---------------------------------------------------------------------------
# 5：并行/单线程一致
# ---------------------------------------------------------------------------
class ParallelConsistencyTest(_StreamTestCase):
    def test_workers_do_not_change_results(self) -> None:
        wav = _write(
            self.root / "par.wav", _segment_with_calls(8, 12.0, seed=4)
        )
        with mock.patch.object(
            segment_stream, "SEGMENT_STREAM_CHUNK_SECONDS", 2.0
        ):
            single = analyze_segment(wav, _deterministic_payload(), self.cfg,
                                     max_workers=1)
            parallel = analyze_segment(wav, _deterministic_payload(), self.cfg,
                                       max_workers=3)
        self.assertEqual(_row_keys(single), _row_keys(parallel))
        self.assertEqual(
            single["detected_call_count"], parallel["detected_call_count"]
        )
        self.assertEqual(single["label_counts"], parallel["label_counts"])
        self.assertEqual(single["raw_candidate_count"],
                         parallel["raw_candidate_count"])


# ---------------------------------------------------------------------------
# 3：断点续跑 / 取消 / 增量落盘
# ---------------------------------------------------------------------------
class ResumeTest(_StreamTestCase):
    def test_graceful_stop_checkpoint_then_resume_completes(self) -> None:
        wav = _write(self.root / "resume.wav", _segment_with_calls(8, 12.0, seed=5))
        out1 = self.root / "out_stop"

        # baseline：不中断跑完
        baseline = analyze_segment(wav, _deterministic_payload(), self.cfg)
        self.assertEqual(baseline["detected_call_count"], 8)

        # 第一次：2 个 chunk 后优雅停止
        state = {"chunks": 0, "stop": False}

        def chunk_progress(payload):
            state["chunks"] += 1
            if state["chunks"] >= 2:
                state["stop"] = True

        with mock.patch.object(
            segment_stream, "SEGMENT_STREAM_CHUNK_SECONDS", 2.0
        ):
            partial = analyze_segment(
                wav,
                _deterministic_payload(),
                self.cfg,
                segment_output_dir=out1,
                chunk_progress=chunk_progress,
                should_stop=lambda: state["stop"],
                max_workers=1,
            )
        self.assertEqual(partial["status"], "cancelled")
        self.assertLess(partial["detected_call_count"], 8)

        # 停止后已有部分结果落盘（增量输出，不等全跑完）；
        # 取消时 checkpoint 保留（partial 可见 + 可续跑）
        partial_csv = out1 / ".checkpoint" / "pulse_results.partial.csv"
        self.assertTrue(partial_csv.exists(), "取消时部分结果必须已落盘可见")
        self.assertTrue((out1 / ".checkpoint" / "run_state.json").exists())

        # 第二次：断点续跑
        with mock.patch.object(
            segment_stream, "SEGMENT_STREAM_CHUNK_SECONDS", 2.0
        ):
            resumed = analyze_segment(
                wav,
                _deterministic_payload(),
                self.cfg,
                segment_output_dir=out1,
                resume=True,
                max_workers=1,
            )
        self.assertTrue(resumed["resumed"])
        self.assertEqual(resumed["status"], "success")
        self.assertEqual(
            resumed["detected_call_count"],
            baseline["detected_call_count"],
            "续跑后计数必须与不中断一致",
        )
        keys = _row_keys(resumed)
        self.assertEqual(len(keys), len(set(keys)), "续跑不得产生重复 pulse")
        self.assertEqual(sorted(keys), sorted(_row_keys(baseline)))

    def test_resume_restarts_when_file_changed(self) -> None:
        wav = _write(self.root / "changed.wav", _segment_with_calls(6, 10.0, seed=6))
        out1 = self.root / "out_changed"
        state = {"chunks": 0, "stop": False}

        def chunk_progress(payload):
            state["chunks"] += 1
            state["stop"] = state["chunks"] >= 2

        with mock.patch.object(
            segment_stream, "SEGMENT_STREAM_CHUNK_SECONDS", 2.0
        ):
            analyze_segment(
                wav,
                _deterministic_payload(),
                self.cfg,
                segment_output_dir=out1,
                chunk_progress=chunk_progress,
                should_stop=lambda: state["stop"],
                max_workers=1,
            )
        # 换文件内容：必须拒绝错误续跑并重新开始（不重复计数）
        _write(wav, _segment_with_calls(6, 10.0, seed=99))
        with mock.patch.object(
            segment_stream, "SEGMENT_STREAM_CHUNK_SECONDS", 2.0
        ):
            result = analyze_segment(
                wav,
                _deterministic_payload(),
                self.cfg,
                segment_output_dir=out1,
                resume=True,
                max_workers=1,
            )
        self.assertFalse(result["resumed"])
        self.assertEqual(result["detected_call_count"], 6)

    def test_checkpoint_state_survives_and_is_cleared_on_success(self) -> None:
        wav = _write(self.root / "ckpt.wav", _segment_with_calls(3, 4.0))
        out1 = self.root / "out_ckpt"
        with mock.patch.object(
            segment_stream, "SEGMENT_STREAM_CHUNK_SECONDS", 2.0
        ):
            result = analyze_segment(
                wav, None, self.cfg, segment_output_dir=out1, max_workers=1
            )
        self.assertEqual(result["status"], "success")
        checkpoint_dir = out1 / ".checkpoint"
        self.assertFalse(
            (checkpoint_dir / "run_state.json").exists(),
            "成功结束后 checkpoint 状态必须清理",
        )
        self.assertTrue((out1 / "pulse_results.csv").exists())


# ---------------------------------------------------------------------------
# 6：导出上限绝不影响计数
# ---------------------------------------------------------------------------
class ExportLimitTest(_StreamTestCase):
    def test_export_cap_never_changes_counts(self) -> None:
        wav = _write(self.root / "exp.wav", _segment_with_calls(6, 8.0, seed=8))
        cfg = SegmentAnalysisConfig(
            min_confidence=0.5,
            low_confidence_action="uncertain",
            save_pulse_audio=True,
            save_figures=False,
        ).validate()
        out1 = self.root / "out_export"

        with mock.patch.object(
            segment_stream, "SEGMENT_STREAM_CHUNK_SECONDS", 2.0
        ), mock.patch(
            "bat_classifier.segment_analysis.SEGMENT_MAX_PULSE_WAV_EXPORTS", 1
        ):
            limited = analyze_segment(
                wav, None, cfg, segment_output_dir=out1, max_workers=1
            )

        self.assertEqual(limited["detected_call_count"], 6)
        self.assertEqual(limited["exported_pulse_count"], 1)
        self.assertTrue(limited["export_truncated"])
        wav_files = sorted(out1.glob("*.wav"))
        self.assertEqual(len(wav_files), 1, "导出上限必须真正限制 WAV 文件数")

        # 不限时：计数相同（上限只影响磁盘）
        out2 = self.root / "out_export_full"
        with mock.patch.object(
            segment_stream, "SEGMENT_STREAM_CHUNK_SECONDS", 2.0
        ), mock.patch(
            "bat_classifier.segment_analysis.SEGMENT_MAX_PULSE_WAV_EXPORTS", 0
        ):
            unlimited = analyze_segment(
                wav, None, cfg, segment_output_dir=out2, max_workers=1
            )
        self.assertEqual(
            unlimited["detected_call_count"], limited["detected_call_count"]
        )
        self.assertFalse(unlimited["export_truncated"])
        self.assertGreaterEqual(len(list(out2.glob("*.wav"))), 6)


# ---------------------------------------------------------------------------
# 7：短 WAV golden 回归（优化前后完全一致）
# ---------------------------------------------------------------------------
class GoldenRegressionTest(_StreamTestCase):
    EXPECTED = {
        "seg_1s_2calls": {
            "calls": 2,
            "rate": 2.0,
            "rows": [
                (0.059, 0.067, "A", 0.7),
                (0.899, 0.907, "A", 0.7),
            ],
        },
        "seg_2s_3calls": {
            "calls": 3,
            "rate": 1.5,
            "rows": [
                (0.059, 0.067, "A", 0.7),
                (0.979, 0.987, "A", 0.7),
                (1.899, 1.907, "A", 0.7),
            ],
        },
        "seg_3s_5calls": {
            "calls": 5,
            "rate": 1.666667,
            "rows": [
                (0.059, 0.067, "A", 0.7),
                (0.769, 0.777, "A", 0.7),
                (1.479, 1.487, "A", 0.7),
                (2.189, 2.197, "A", 0.7),
                (2.899, 2.907, "A", 0.7),
            ],
        },
    }

    def test_short_wav_results_identical_to_pre_optimization(self) -> None:
        for name, expected in self.EXPECTED.items():
            length = float(name.split("_")[1].rstrip("s"))
            calls = int(name.split("_")[2].replace("calls", ""))
            wav = _write(
                self.root / f"{name}.wav",
                _segment_with_calls(calls, length),
            )
            result = analyze_segment(wav, _deterministic_payload(), self.cfg)
            self.assertEqual(result["detected_call_count"], expected["calls"], name)
            self.assertAlmostEqual(
                result["calls_per_second"], expected["rate"], places=6, msg=name
            )
            observed = [
                (
                    round(float(row["start_s"]), 3),
                    round(float(row["end_s"]), 3),
                    str(row.get("predicted_label")),
                    round(float(row.get("confidence")), 4),
                )
                for row in result["pulse_rows"]
                if str(row.get("predicted_label", "")) != "noise"
            ]
            self.assertEqual(observed, expected["rows"], name)


if __name__ == "__main__":
    unittest.main()
