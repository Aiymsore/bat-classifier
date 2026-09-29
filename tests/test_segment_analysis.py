"""⑦ 片段识别与叫声统计（segment_analysis）单元/集成测试。

覆盖 README 第三十二节要求：
1. Call counting——构造含 N 个明显 pulse 的 WAV，验证检测数量稳定；
2. 一个 segment 多 pulse——产生多个 pulse row，summary total_calls == 明细数；
3. 多类别汇总——A,A,A,B,B,uncertain → A=3,B=2,uncertain=1（绝不 major A=6）；
4. 文件夹分组——每段独立输出目录，pulse 不串目录；
5. 损坏 WAV——一条失败不终止整个 batch；
6. No-call WAV——call_count=0 且 status=success；
7. 模型不存在——明确提示，且可仅计数。
另测：逐 pulse 分类置信度/不确定标记逻辑、真实模型推理链可跑通、
     统计守恒（total == 各类之和）。不测 Qt 像素布局。
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bat_classifier.config import (  # noqa: E402
    CLUSTER_FEATURE_COLUMNS,
    NOISE_GATE_FEATURE_COLUMNS,
)
from bat_classifier.segment_analysis import (  # noqa: E402
    SegmentAnalysisConfig,
    _aggregate_label_counts,
    analyze_segment,
    analyze_segment_folder,
    classify_pulse_rows,
    describe_model,
    load_classification_model,
    summarize_segment_results,
)

SR = 250_000


# ---------------------------------------------------------------------------
# 测试工具
# ---------------------------------------------------------------------------
def _make_case_dir() -> Path:
    """建一个独立的临时工作目录（Path.mkdir，便于清理，兼容受限环境）。"""
    base = Path(tempfile.gettempdir()) / "_bat_segment_analysis_tests"
    base.mkdir(parents=True, exist_ok=True)
    case = base / f"case_{uuid.uuid4().hex}"
    case.mkdir(parents=True, exist_ok=True)
    return case


def _remove_case_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _sweep(start_khz: float, end_khz: float, duration_s: float = 0.006,
           amplitude: float = 0.8) -> np.ndarray:
    samples = int(SR * duration_s)
    frequencies = np.linspace(start_khz * 1000, end_khz * 1000, samples)
    phase = 2 * np.pi * np.cumsum(frequencies) / SR
    return (amplitude * np.sin(phase)).astype(np.float32)


def _segment_with_calls(n_calls: int, length_s: float = 1.0,
                        seed: int = 0) -> np.ndarray:
    """在静音里等距摆放 n_calls 个清晰 FM 扫频，代表 n_calls 次独立叫声。"""
    total = np.zeros(int(SR * length_s), dtype=np.float32)
    offsets = np.linspace(0.06, length_s - 0.10, max(n_calls, 1))
    for index, offset in enumerate(offsets):
        sweep = _sweep(90 - 6 * index, 50 - 4 * index)
        start = int(offset * SR)
        end = min(len(total), start + len(sweep))
        total[start:end] = sweep[: end - start]
    rng = np.random.default_rng(seed)
    total += rng.uniform(-0.001, 0.001, size=len(total)).astype(np.float32)
    return total


def _noise_only(length_s: float = 0.5, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.0008, 0.0008, size=int(SR * length_s)).astype(np.float32)


def _write(path: Path, audio: np.ndarray, sr: int = SR) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr, subtype="PCM_16")
    return path


class _StubAcousticClassifier:
    """可控 predict/predict_proba 的分类器桩，用于确定性验证置信度/不确定逻辑。

    classes：完整类别表（对应 predict_proba 列）；
    predictions：逐行预测标签；
    proba：形状 (n_rows, n_classes) 的概率。
    """

    def __init__(self, classes, predictions, proba):
        self.classes_ = np.array(classes)
        self._predictions = np.array(predictions)
        self._proba = np.asarray(proba, dtype=float)

    def predict(self, X):
        return self._predictions[: len(X)]

    def predict_proba(self, X):
        return self._proba[: len(X)]


class _StubNoiseModel:
    classes_ = np.array(["bat", "noise"])

    def predict_proba(self, X):
        n = len(X)
        return np.tile([0.99, 0.01], (n, 1))


class _StubNoiseModelAllNoise:
    classes_ = np.array(["bat", "noise"])

    def predict_proba(self, X):
        n = len(X)
        return np.tile([0.01, 0.99], (n, 1))


def _acoustic_stub_payload(classifier, labels):
    return {
        "scaler": None,
        "kmeans": None,
        "temporary_classifier": classifier,
        "raw_cluster_to_type": {},
        "feature_columns": list(CLUSTER_FEATURE_COLUMNS),
        "labels": list(labels),
    }


def _noise_stub_payload(model, threshold: float = 0.5):
    return {
        "model": model,
        "feature_columns": list(NOISE_GATE_FEATURE_COLUMNS),
        "classes": ["bat", "noise"],
        "bat_probability_threshold": threshold,
    }


def _feature_frame(n_rows: int, value: float = 1.0) -> pd.DataFrame:
    data = {
        column: np.full(n_rows, value, dtype=float)
        for column in CLUSTER_FEATURE_COLUMNS
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# 1 & 2 & 6：Call counting / 多 pulse / No-call
# ---------------------------------------------------------------------------
class CallCountingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = _make_case_dir()
        self.cfg = SegmentAnalysisConfig(
            save_pulse_audio=False,
            save_figures=False,
            use_quality_filter=True,
        ).validate()

    def tearDown(self) -> None:
        _remove_case_dir(self.root)

    def test_n_clear_pulses_count_stable(self) -> None:
        for n in (1, 3, 5, 7):
            wav = _write(self.root / f"seg_{n}.wav", _segment_with_calls(n))
            result = analyze_segment(wav, None, self.cfg)
            self.assertEqual(
                result["detected_call_count"],
                n,
                f"{n} 个 pulse 应检测到 {n} 次叫声",
            )

    def test_multi_pulse_produces_rows_and_summary_matches(self) -> None:
        wav = _write(self.root / "seg_multi.wav", _segment_with_calls(4))
        result = analyze_segment(wav, None, self.cfg)
        # 多个 pulse row
        bat_rows = [
            row for row in result["pulse_rows"]
            if row.get("predicted_label") != "noise"
        ]
        self.assertEqual(len(bat_rows), 4)
        self.assertGreaterEqual(len(result["pulse_rows"]), 4)
        # summary total_calls == 逐 pulse 明细数量
        summary_df, _ = summarize_segment_results(
            [result], ["unavailable"], "uncertain"
        )
        total_calls = int(summary_df.iloc[0]["total_calls"])
        self.assertEqual(total_calls, len(bat_rows))
        self.assertEqual(total_calls, result["detected_call_count"])

    def test_no_call_wav_is_success_with_zero(self) -> None:
        wav = _write(self.root / "noise.wav", _noise_only())
        result = analyze_segment(wav, None, self.cfg)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["detected_call_count"], 0)
        self.assertEqual(result["bat_call_count"], 0)


# ---------------------------------------------------------------------------
# 3：多类别汇总（统计正确性）
# ---------------------------------------------------------------------------
class MultiClassAggregationTest(unittest.TestCase):
    def test_aggregate_counts_not_major_attributed(self) -> None:
        pulse_labels = ["A", "A", "A", "B", "B", "uncertain"]
        counts = _aggregate_label_counts(
            pulse_labels, ["A", "B", "C"], "uncertain"
        )
        self.assertEqual(counts["A"], 3)
        self.assertEqual(counts["B"], 2)
        self.assertEqual(counts["uncertain"], 1)
        self.assertEqual(counts["C"], 0)

    def test_summary_total_equals_sum_of_columns(self) -> None:
        result = {
            "segment_id": "s",
            "segment_file": "s.wav",
            "segment_path": "s.wav",
            "duration_s": 1.0,
            "major_label": "A",
            "detected_call_count": 6,
            "classified_call_count": 5,
            "uncertain_count": 1,
            "purity": 0.6,
            "consistency_note": "",
            "prediction_confidence": None,
            "raw_candidate_count": 9,
            "candidate_call_count": 8,
            "valid_call_count": 7,
            "noise_count": 1,
            "calls_per_second": 6.0,
            "status": "success",
            "error_message": "",
            "label_counts": {"A": 3, "B": 2, "C": 0, "uncertain": 1},
        }
        summary_df, overall_df = summarize_segment_results(
            [result], ["A", "B", "C"], "uncertain"
        )
        row = summary_df.iloc[0]
        self.assertEqual(int(row["total_calls"]), 6)
        self.assertEqual(int(row["A"]), 3)
        self.assertEqual(int(row["B"]), 2)
        self.assertEqual(int(row["uncertain"]), 1)
        # total == 各类之和（守恒）
        column_sum = int(row["A"] + row["B"] + row["C"] + row["uncertain"])
        self.assertEqual(int(row["total_calls"]), column_sum)
        # major 不吞并所有 call
        self.assertNotEqual(int(row["A"]), int(row["total_calls"]))

    def test_overall_accumulates_per_label_not_major(self) -> None:
        # 两段：段1 A=3,B=2,unc=1；段2 A=0,B=5,unc=0
        results = [
            {
                "label_counts": {"A": 3, "B": 2, "uncertain": 1},
                "detected_call_count": 6,
            },
            {
                "label_counts": {"A": 0, "B": 5, "uncertain": 0},
                "detected_call_count": 5,
            },
        ]
        _, overall_df = summarize_segment_results(
            results, ["A", "B"], "uncertain"
        )
        by_species = {
            row["species"]: row for _, row in overall_df.iterrows()
        }
        self.assertEqual(int(by_species["A"]["call_count"]), 3)   # 不是 6
        self.assertEqual(int(by_species["A"]["segment_count"]), 1)
        self.assertEqual(int(by_species["B"]["call_count"]), 7)   # 2+5
        self.assertEqual(int(by_species["B"]["segment_count"]), 2)
        self.assertEqual(int(by_species["uncertain"]["call_count"]), 1)

    def test_mixed_low_consistency_note(self) -> None:
        # A=4,B=3,C=3 → purity 0.4 < 0.5 → 混合/低一致性（仅描述）
        case_dir = _make_case_dir()
        try:
            wav = _write(case_dir / "m.wav", _segment_with_calls(3))
            cfg = SegmentAnalysisConfig(
                save_pulse_audio=False, save_figures=False, min_confidence=0.0
            ).validate()
            # 用交替标签的桩强制混杂
            classifier = _StubAcousticClassifier(
                ["A", "B", "C"],
                ["A", "B", "C"],
                np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]),
            )
            payload = {
                "noise_gate": _noise_stub_payload(_StubNoiseModel()),
                "acoustic_types": _acoustic_stub_payload(
                    classifier, ["A", "B", "C"]
                ),
            }
            result = analyze_segment(wav, payload, cfg)
            self.assertEqual(result["detected_call_count"], 3)
            self.assertEqual(result["consistency_note"], "混合 / 低一致性")
            self.assertEqual(
                result["detected_call_count"],
                sum(result["label_counts"].values()),
            )
        finally:
            _remove_case_dir(case_dir)


# ---------------------------------------------------------------------------
# 4：文件夹分组（pulse 不串目录）
# ---------------------------------------------------------------------------
class FolderGroupingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = _make_case_dir()

    def tearDown(self) -> None:
        _remove_case_dir(self.root)

    def test_each_segment_gets_own_output_dir(self) -> None:
        input_dir = self.root / "input"
        _write(input_dir / "segment_001.wav", _segment_with_calls(2))
        _write(input_dir / "segment_002.wav", _segment_with_calls(3))
        output_root = self.root / "run"

        cfg = SegmentAnalysisConfig(
            save_pulse_audio=True, save_figures=False
        ).validate()
        result = analyze_segment_folder(
            input_dir, None, cfg, output_root=output_root
        )

        segments_dir = output_root / "segments"
        dir1 = segments_dir / "segment_001"
        dir2 = segments_dir / "segment_002"
        self.assertTrue(dir1.is_dir())
        self.assertTrue(dir2.is_dir())

        wavs1 = [p.name for p in dir1.glob("*.wav")]
        wavs2 = [p.name for p in dir2.glob("*.wav")]
        # 每段独立目录里有 pulse WAV
        self.assertGreaterEqual(len(wavs1), 2)
        self.assertGreaterEqual(len(wavs2), 3)
        # pulse 不串目录：文件名前缀必须属于本段
        for name in wavs1:
            self.assertTrue(name.startswith("segment_001__"), name)
        for name in wavs2:
            self.assertTrue(name.startswith("segment_002__"), name)
        # pulse_results.csv 每段独立
        self.assertTrue((dir1 / "pulse_results.csv").exists())
        self.assertTrue((dir2 / "pulse_results.csv").exists())

        # 汇总文件写出
        self.assertTrue((output_root / "segment_summary.csv").exists())
        self.assertTrue((output_root / "pulse_predictions.csv").exists())
        self.assertTrue((output_root / "overall_species_summary.csv").exists())
        self.assertTrue((output_root / "run_metadata.json").exists())
        self.assertEqual(result["segment_count"], 2)


# ---------------------------------------------------------------------------
# 5：失败隔离
# ---------------------------------------------------------------------------
class FailureIsolationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = _make_case_dir()

    def tearDown(self) -> None:
        _remove_case_dir(self.root)

    def test_corrupt_wav_does_not_abort_batch(self) -> None:
        input_dir = self.root / "input"
        _write(input_dir / "good.wav", _segment_with_calls(2))
        bad = input_dir / "bad.wav"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_bytes(b"this is definitely not a wav file")
        _write(input_dir / "good2.wav", _segment_with_calls(1))

        cfg = SegmentAnalysisConfig(
            save_pulse_audio=False, save_figures=False
        ).validate()
        result = analyze_segment_folder(
            input_dir, None, cfg, output_root=self.root / "run"
        )

        self.assertEqual(result["segment_count"], 3)
        self.assertEqual(result["success_count"], 2)
        self.assertEqual(result["failed_count"], 1)
        # 失败的那条被标记 failed 并带错误信息
        failed = [
            r for r in result["segment_results"] if r["status"] == "failed"
        ]
        self.assertEqual(len(failed), 1)
        self.assertTrue(failed[0]["error_message"])
        # 成功的仍被分析
        success = [
            r for r in result["segment_results"] if r["status"] == "success"
        ]
        self.assertEqual(len(success), 2)


# ---------------------------------------------------------------------------
# 7：模型不存在 + 仅计数
# ---------------------------------------------------------------------------
class ModelAbsentTest(unittest.TestCase):
    def test_describe_model_none(self) -> None:
        info = describe_model(None)
        self.assertFalse(info["loaded"])
        self.assertIn("未检测到", info["warning"])

    def test_load_missing_model_returns_none(self) -> None:
        payload = load_classification_model(
            Path("does_not_exist_model.joblib")
        )
        self.assertIsNone(payload)

    def test_count_only_uses_unavailable_label(self) -> None:
        case_dir = _make_case_dir()
        try:
            wav = _write(case_dir / "seg.wav", _segment_with_calls(3))
            cfg = SegmentAnalysisConfig(
                save_pulse_audio=False, save_figures=False
            ).validate()
            result = analyze_segment(wav, None, cfg)
            self.assertEqual(result["detected_call_count"], 3)
            # 无模型 → 标签 = unavailable，不是某个物种
            self.assertEqual(result["label_counts"].get("unavailable"), 3)
            self.assertEqual(result["major_label"], "unavailable")
        finally:
            _remove_case_dir(case_dir)


# ---------------------------------------------------------------------------
# 逐 pulse 分类置信度 / 不确定标记
# ---------------------------------------------------------------------------
class ClassifyUncertaintyTest(unittest.TestCase):
    def _classify(self, min_confidence: float, action: str):
        classifier = _StubAcousticClassifier(
            ["A", "B"],
            ["A", "B", "A"],
            np.array([
                [0.90, 0.10],   # → A, conf 0.90
                [0.20, 0.80],   # → B, conf 0.80
                [0.55, 0.45],   # → A, conf 0.55
            ]),
        )
        payload = _acoustic_stub_payload(classifier, ["A", "B"])
        cfg = SegmentAnalysisConfig(
            min_confidence=min_confidence,
            low_confidence_action=action,
        ).validate()
        df = _feature_frame(3)
        return classify_pulse_rows(df, payload, cfg)

    def test_all_classified_below_threshold_relaxed(self) -> None:
        df = self._classify(0.5, "uncertain")
        self.assertEqual(
            list(df["predicted_label"]), ["A", "B", "A"]
        )
        self.assertFalse(any(df["is_uncertain"]))

    def test_low_confidence_marked_uncertain(self) -> None:
        df = self._classify(0.7, "uncertain")
        # 第三行 conf 0.55 < 0.7 → uncertain，其余保留
        self.assertEqual(df.iloc[0]["predicted_label"], "A")
        self.assertEqual(df.iloc[1]["predicted_label"], "B")
        self.assertEqual(df.iloc[2]["predicted_label"], "uncertain")
        self.assertTrue(bool(df.iloc[2]["is_uncertain"]))

    def test_keep_action_retains_label(self) -> None:
        df = self._classify(0.7, "keep")
        # keep：不改标签，不标 uncertain
        self.assertEqual(
            list(df["predicted_label"]), ["A", "B", "A"]
        )
        self.assertFalse(any(df["is_uncertain"]))

    def test_feature_order_uses_metadata_columns(self) -> None:
        # 打乱 df 列顺序也应按 feature_columns 正确取特征（不报错、标签稳定）
        classifier = _StubAcousticClassifier(
            ["A"], ["A"], np.array([[1.0]])
        )
        payload = _acoustic_stub_payload(classifier, ["A"])
        cfg = SegmentAnalysisConfig(min_confidence=0.0).validate()
        df = _feature_frame(1)
        df = df[list(reversed(list(df.columns)))]  # 打乱列序
        out = classify_pulse_rows(df, payload, cfg)
        self.assertEqual(out.iloc[0]["predicted_label"], "A")


# ---------------------------------------------------------------------------
# 真实 sklearn 推理链可跑通（特征顺序 / RF / predict_proba）
# ---------------------------------------------------------------------------
class RealModelChainTest(unittest.TestCase):
    def _real_payload(self) -> dict:
        from sklearn.cluster import KMeans
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(0)
        # 用带列名的 DataFrame 训练，与 prediction.py 的真实推理链一致
        feature_cols = list(CLUSTER_FEATURE_COLUMNS)
        X = pd.DataFrame(
            rng.normal(size=(40, len(feature_cols))), columns=feature_cols
        )
        y = np.array(["A", "B"] * 20)
        scaler = StandardScaler().fit(X)
        scaled = scaler.transform(X)
        kmeans = KMeans(n_clusters=2, n_init=5, random_state=0).fit(scaled)
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)

        noise_cols = list(NOISE_GATE_FEATURE_COLUMNS)
        Xn = pd.DataFrame(
            rng.normal(size=(30, len(noise_cols))), columns=noise_cols
        )
        yn = np.array(["bat", "noise"] * 15)
        noise_rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(Xn, yn)

        return {
            "noise_gate": {
                "model": noise_rf,
                "feature_columns": noise_cols,
                "classes": ["bat", "noise"],
                "bat_probability_threshold": 0.0,  # 全部进入分类，便于确定计数
            },
            "acoustic_types": {
                "scaler": scaler,
                "kmeans": kmeans,
                "temporary_classifier": rf,
                "raw_cluster_to_type": {0: "A", 1: "B"},
                "feature_columns": feature_cols,
                "labels": ["A", "B"],
            },
        }

    def test_real_chain_runs_and_conserves_counts(self) -> None:
        case_dir = _make_case_dir()
        try:
            wav = _write(case_dir / "seg.wav", _segment_with_calls(3))
            cfg = SegmentAnalysisConfig(
                save_pulse_audio=False,
                save_figures=False,
                min_confidence=0.0,
            ).validate()
            result = analyze_segment(wav, self._real_payload(), cfg)
            # 阈值 0.0 的 noise gate → 3 次全部计入
            self.assertEqual(result["detected_call_count"], 3)
            # 守恒：total == 各类之和
            self.assertEqual(
                result["detected_call_count"],
                sum(result["label_counts"].values()),
            )
            # 标签只在 {A,B,uncertain} 内，置信度在 [0,1]
            for row in result["pulse_rows"]:
                if row.get("predicted_label") == "noise":
                    continue
                self.assertIn(
                    row["predicted_label"], ("A", "B", "uncertain")
                )
                if row.get("confidence") is not None:
                    self.assertGreaterEqual(row["confidence"], 0.0)
                    self.assertLessEqual(row["confidence"], 1.0)
        finally:
            _remove_case_dir(case_dir)


if __name__ == "__main__":
    unittest.main()
