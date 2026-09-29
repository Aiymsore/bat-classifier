"""⑦ 概率校准（CalibratedClassifierCV）测试。

覆盖：
- 校准方法选择（isotonic 小样本自动降级 sigmoid）；
- 校准切分按 source_recording 分组隔离（train/calib 录音绝不重合）；
- 数据不足 → 不伪造：classifier=None，只给未校准置信度；
- attach_calibration 补校准：不重训、不覆盖原模型，特征顺序自动对齐；
- ⑦ 推理链：置信度性质 calibrated_probability / model_support 明确区分；
- detect_label_column / detect_source_column。

只测算法与结构，不测 Qt 控件。
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bat_classifier.calibration import (  # noqa: E402
    CALIBRATED_PROBABILITY,
    MODEL_SUPPORT,
    attach_calibration,
    build_calibration_folds,
    calibrate_classifier,
    describe_calibration,
    detect_label_column,
    detect_source_column,
    resolve_calibration_method,
    select_classifier,
)
from bat_classifier.config import (  # noqa: E402
    SEGMENT_ISOTONIC_MIN_SAMPLES,
    SEGMENT_MIN_RECORDINGS_FOR_CALIBRATION,
)
from bat_classifier.segment_analysis import (  # noqa: E402
    SegmentAnalysisConfig,
    classify_pulse_rows,
    describe_model,
)


def _make_case_dir() -> Path:
    base = Path(tempfile.gettempdir()) / "_bat_calibration_tests"
    base.mkdir(parents=True, exist_ok=True)
    case = base / f"case_{uuid.uuid4().hex}"
    case.mkdir(parents=True, exist_ok=True)
    return case


def _synthetic(
    n_recordings: int = 8,
    per_recording: int = 8,
    seed: int = 0,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """构造“特征与标签相关”的合成脉冲表（每条录音都含 A/B 两类）。"""
    rng = np.random.default_rng(seed)
    rows: list[list[float]] = []
    labels: list[str] = []
    groups: list[str] = []
    for rec in range(n_recordings):
        for position in range(per_recording):
            label = "A" if (rec + position) % 2 == 0 else "B"
            mean = 1.0 if label == "A" else -1.0
            rows.append(
                [
                    float(rng.normal(loc=mean, scale=0.5)),
                    float(rng.normal(size=1)[0]),
                ]
            )
            labels.append(label)
            groups.append(f"rec_{rec:03d}")
    frame = pd.DataFrame(rows, columns=["f1", "f2"])
    return frame, labels, groups


def _acoustic_payload(
    features: pd.DataFrame,
    labels: list[str],
    groups: list[str] | None,
    calibrated: bool = True,
) -> dict:
    from sklearn.ensemble import RandomForestClassifier

    base = RandomForestClassifier(n_estimators=20, random_state=0).fit(
        features, labels
    )
    acoustic: dict = {
        "temporary_classifier": base,
        "feature_columns": ["f1", "f2"],
        "labels": ["A", "B"],
    }
    if calibrated:
        result = calibrate_classifier(
            features, labels, groups=groups, base=base
        )
        if result["classifier"] is not None:
            acoustic["calibrated_classifier"] = result["classifier"]
            acoustic["calibration"] = result["info"]
    return {"noise_gate": {}, "acoustic_types": acoustic}


# ---------------------------------------------------------------------------
# 校准方法
# ---------------------------------------------------------------------------
class ResolveMethodTest(unittest.TestCase):
    def test_isotonic_downgraded_for_small_sample(self) -> None:
        method, note = resolve_calibration_method("isotonic", 10)
        self.assertEqual(method, "sigmoid")
        self.assertIn("isotonic 不可靠", note)

    def test_isotonic_kept_for_large_sample(self) -> None:
        method, note = resolve_calibration_method(
            "isotonic", SEGMENT_ISOTONIC_MIN_SAMPLES
        )
        self.assertEqual(method, "isotonic")
        self.assertEqual(note, "")

    def test_auto_picks_by_sample_count(self) -> None:
        self.assertEqual(
            resolve_calibration_method("auto", 10)[0], "sigmoid"
        )
        self.assertEqual(
            resolve_calibration_method(
                "auto", SEGMENT_ISOTONIC_MIN_SAMPLES
            )[0],
            "isotonic",
        )

    def test_invalid_falls_back_to_default(self) -> None:
        self.assertEqual(resolve_calibration_method("nonsense", 999)[0], "sigmoid")


# ---------------------------------------------------------------------------
# 分组隔离的校准切分
# ---------------------------------------------------------------------------
class CalibrationFoldsTest(unittest.TestCase):
    def test_group_isolation_no_recording_overlap(self) -> None:
        _, labels, groups = _synthetic(n_recordings=9)
        result = build_calibration_folds(labels, groups=groups)
        self.assertTrue(result["ok"])
        self.assertTrue(result["group_isolated"])
        self.assertGreater(len(result["folds"]), 0)
        for record in result["split"]["folds"]:
            train = set(record["train_recordings"])
            calib = set(record["calibration_recordings"])
            self.assertTrue(train and calib)
            self.assertEqual(train & calib, set())  # 绝不跨两侧

    def test_too_few_recordings_never_fake_calibration(self) -> None:
        _, labels, groups = _synthetic(
            n_recordings=SEGMENT_MIN_RECORDINGS_FOR_CALIBRATION - 1
        )
        result = build_calibration_folds(labels, groups=groups)
        self.assertFalse(result["ok"])
        self.assertEqual(result["folds"], [])
        self.assertTrue(
            any("不足以" in note for note in result["notes"])
        )

    def test_no_groups_flagged_as_not_isolated(self) -> None:
        _, labels, _ = _synthetic()
        result = build_calibration_folds(labels, groups=None)
        self.assertFalse(result["group_isolated"])
        self.assertTrue(any("未按录音隔离" in note for note in result["notes"]))

    def test_single_class_rejected(self) -> None:
        result = build_calibration_folds(["A"] * 30, groups=None)
        self.assertFalse(result["ok"])

    def test_too_few_samples_per_class_rejected(self) -> None:
        result = build_calibration_folds(
            ["A", "B", "A", "B"], groups=None
        )
        self.assertFalse(result["ok"])


# ---------------------------------------------------------------------------
# calibrate_classifier
# ---------------------------------------------------------------------------
class CalibrateClassifierTest(unittest.TestCase):
    def test_calibrated_when_data_sufficient(self) -> None:
        features, labels, groups = _synthetic()
        result = calibrate_classifier(features, labels, groups=groups)
        self.assertIsNotNone(result["classifier"])
        info = result["info"]
        self.assertTrue(info["is_calibrated"])
        self.assertIn(info["calibration_method"], ("sigmoid", "isotonic"))
        self.assertGreater(info["fold_count"], 0)
        self.assertEqual(info["sample_count"], len(labels))
        # 输出是合法概率
        probabilities = result["classifier"].predict_proba(features)
        self.assertEqual(probabilities.shape[0], len(labels))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6)

    def test_insufficient_data_returns_none_never_fakes(self) -> None:
        features, labels, groups = _synthetic(n_recordings=2)
        result = calibrate_classifier(features, labels, groups=groups)
        self.assertIsNone(result["classifier"])
        self.assertFalse(result["info"]["is_calibrated"])
        self.assertTrue(result["info"]["warnings"])

    def test_labels_used_verbatim(self) -> None:
        features, labels, groups = _synthetic()
        result = calibrate_classifier(features, labels, groups=groups)
        self.assertEqual(
            result["info"]["class_counts"],
            {"A": labels.count("A"), "B": labels.count("B")},
        )


# ---------------------------------------------------------------------------
# 补校准（attach_calibration）
# ---------------------------------------------------------------------------
class AttachCalibrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.case = _make_case_dir()

    def tearDown(self) -> None:
        shutil.rmtree(self.case, ignore_errors=True)

    def test_attaches_without_retraining_base(self) -> None:
        features, labels, groups = _synthetic()
        payload = _acoustic_payload(features, labels, groups, calibrated=False)
        acoustic = payload["acoustic_types"]
        base_before = acoustic["temporary_classifier"]

        attach_calibration(payload, features, labels, groups=groups)

        self.assertIs(acoustic["temporary_classifier"], base_before)
        self.assertIn("calibrated_classifier", acoustic)
        self.assertTrue(acoustic["calibration"]["is_calibrated"])

    def test_feature_columns_aligned_despite_shuffled_order(self) -> None:
        features, labels, groups = _synthetic()
        payload = _acoustic_payload(features, labels, groups, calibrated=False)
        shuffled = features[list(reversed(list(features.columns)))]
        attach_calibration(payload, shuffled, labels, groups=groups)
        self.assertTrue(payload["acoustic_types"]["calibration"]["is_calibrated"])

    def test_missing_feature_column_raises(self) -> None:
        features, labels, groups = _synthetic()
        payload = _acoustic_payload(features, labels, groups, calibrated=False)
        with self.assertRaises(ValueError):
            attach_calibration(
                payload, features.drop(columns=["f2"]), labels, groups=groups
            )

    def test_non_two_stage_payload_raises(self) -> None:
        features, labels, groups = _synthetic()
        with self.assertRaises(ValueError):
            attach_calibration({"model": "x"}, features, labels, groups=groups)

    def test_insufficient_data_keeps_model_uncalibrated(self) -> None:
        features, labels, groups = _synthetic(n_recordings=2)
        payload = _acoustic_payload(features, labels, groups, calibrated=False)
        attach_calibration(payload, features, labels, groups=groups)
        self.assertNotIn("calibrated_classifier", payload["acoustic_types"])
        self.assertFalse(payload["acoustic_types"]["calibration"]["is_calibrated"])


# ---------------------------------------------------------------------------
# 读侧：select_classifier / describe_calibration / ⑦ 推理链
# ---------------------------------------------------------------------------
class ReadSideTest(unittest.TestCase):
    def test_select_prefers_calibrated(self) -> None:
        features, labels, groups = _synthetic()
        payload = _acoustic_payload(features, labels, groups, calibrated=True)
        acoustic = payload["acoustic_types"]
        if "calibrated_classifier" not in acoustic:
            self.skipTest("本次合成数据未触发校准")
        classifier, is_calibrated = select_classifier(acoustic)
        self.assertTrue(is_calibrated)
        self.assertIs(classifier, acoustic["calibrated_classifier"])

    def test_select_falls_back_to_temporary(self) -> None:
        features, labels, groups = _synthetic()
        acoustic = _acoustic_payload(
            features, labels, groups, calibrated=False
        )["acoustic_types"]
        classifier, is_calibrated = select_classifier(acoustic)
        self.assertFalse(is_calibrated)
        self.assertIs(classifier, acoustic["temporary_classifier"])

    def test_describe_marks_uncalibrated(self) -> None:
        features, labels, groups = _synthetic()
        acoustic = _acoustic_payload(
            features, labels, groups, calibrated=False
        )["acoustic_types"]
        info = describe_calibration(acoustic)
        self.assertFalse(info["is_calibrated"])
        self.assertEqual(info["probability_kind"], MODEL_SUPPORT)
        self.assertIn("未校准", info["calibration_note"])

    def test_seven_chain_reports_calibrated_probability(self) -> None:
        features, labels, groups = _synthetic()
        payload = _acoustic_payload(features, labels, groups, calibrated=True)
        acoustic = payload["acoustic_types"]
        if "calibrated_classifier" not in acoustic:
            self.skipTest("本次合成数据未触发校准")

        cfg = SegmentAnalysisConfig(
            min_confidence=0.0, low_confidence_action="uncertain"
        ).validate()
        out = classify_pulse_rows(features.copy(), acoustic, cfg)

        self.assertEqual(set(out["confidence_kind"]), {CALIBRATED_PROBABILITY})
        for value in out["confidence"]:
            self.assertGreaterEqual(float(value), 0.0)
            self.assertLessEqual(float(value), 1.0)

        info = describe_model(payload)
        self.assertTrue(info["is_calibrated"])
        self.assertEqual(info["probability_kind"], CALIBRATED_PROBABILITY)
        self.assertEqual(
            info["calibration_method"],
            acoustic["calibration"]["calibration_method"],
        )
        self.assertIn("校准概率", info["calibration_note"])

    def test_seven_chain_reports_model_support_without_calibration(self) -> None:
        features, labels, groups = _synthetic()
        payload = _acoustic_payload(
            features, labels, groups, calibrated=False
        )
        cfg = SegmentAnalysisConfig(
            min_confidence=0.0, low_confidence_action="uncertain"
        ).validate()
        out = classify_pulse_rows(features.copy(), payload["acoustic_types"], cfg)
        self.assertEqual(set(out["confidence_kind"]), {MODEL_SUPPORT})

        info = describe_model(payload)
        self.assertFalse(info["is_calibrated"])
        self.assertEqual(info["probability_kind"], MODEL_SUPPORT)
        self.assertIn("未校准", info["calibration_note"])


# ---------------------------------------------------------------------------
# 标注 CSV 列探测
# ---------------------------------------------------------------------------
class LabelColumnTest(unittest.TestCase):
    def test_detect_label_column_priority(self) -> None:
        frame = pd.DataFrame({"predicted_label": ["A"], "expert_label": ["A"]})
        self.assertEqual(detect_label_column(frame), "expert_label")

    def test_no_label_column(self) -> None:
        frame = pd.DataFrame({"predicted_label": ["A"], "x": [1]})
        self.assertEqual(detect_label_column(frame), "")

    def test_detect_source_column(self) -> None:
        frame = pd.DataFrame({"source_recording": ["r1"], "y": [1]})
        self.assertEqual(detect_source_column(frame), "source_recording")
        self.assertEqual(detect_source_column(pd.DataFrame({"y": [1]})), "")


if __name__ == "__main__":
    unittest.main()
