"""已确认物种模型（⑥ 模型对照）训练 / 校准 / 导入 / 推理 测试。

覆盖：概率校准（is_calibrated、合法概率）、录音级 group split 无重合、
单物种拒绝、数据不足时不伪造校准概率、导入模型特征一致性校验、
uncertain 阈值、未校准模型不得输出“校准概率”。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bat_classifier.config import (  # noqa: E402
    SPECIES_FEATURE_COLUMNS,
    SPECIES_UNCERTAIN_LABEL,
)
from bat_classifier.species_model import (  # noqa: E402
    INSUFFICIENT_CALIBRATION_WARNING,
    SINGLE_CLASS_WARNING,
    build_group_splits,
    calibrate_species_model_with_labels,
    describe_species_model,
    import_species_model,
    model_status_text,
    normalize_training_frame,
    predict_pulse_frame,
    source_recording_series,
    train_species_model,
    validate_imported_payload,
)

SPECIES_PEAKS = {
    "Pipistrellus abramus": 25.0,
    "Myotis sp.": 45.0,
    "Rhinolophus sp.": 65.0,
}


def _make_dataset(
    recording_counts: dict[str, int] | None = None,
    pulses_per_recording: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    """合成专家标注数据：每个物种若干条“录音”，每条录音多个 pulse。"""
    recording_counts = recording_counts or {
        "Pipistrellus abramus": 6,
        "Myotis sp.": 6,
        "Rhinolophus sp.": 6,
    }
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for species, rec_count in recording_counts.items():
        peak = SPECIES_PEAKS.get(species, 35.0)
        for rec_index in range(rec_count):
            source = f"{species.replace(' ', '_')}_rec{rec_index}.wav"
            for pulse_index in range(pulses_per_recording):
                row = {
                    "pulse_id": f"{source}_p{pulse_index:03d}",
                    "expert_label": species,
                    "source_file": source,
                    "source_path": f"/data/{source}",
                    "audio_hash": f"hash_{source}",
                    "review_status": "reviewed",
                }
                for column in SPECIES_FEATURE_COLUMNS:
                    base = peak if column == "peak_freq_khz" else 30.0
                    row[column] = float(rng.normal(base, 1.5))
                rows.append(row)
    return pd.DataFrame(rows)


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


class _ScriptedModel:
    """按 duration_ms 编码返回固定概率的极简分类器（不依赖 sklearn 训练）。"""

    classes_ = np.array(["Species A", "Species B", "Species C"])

    def __init__(
        self,
        proba_map: dict[int, list[float]],
        default: list[float] | None = None,
    ) -> None:
        self.proba_map = proba_map
        self.default = default or [1.0 / len(self.classes_)] * len(self.classes_)

    def predict_proba(self, x_values):  # noqa: ANN001
        rows = []
        for _, row in x_values.iterrows():
            key = int(round(float(row["duration_ms"])))
            rows.append(self.proba_map.get(key, self.default))
        return np.asarray(rows, dtype=float)

    def predict(self, x_values):  # noqa: ANN001
        probabilities = self.predict_proba(x_values)
        return np.asarray(
            [self.classes_[int(index)] for index in probabilities.argmax(axis=1)]
        )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
class TestCalibration(unittest.TestCase):
    def test_trained_model_is_calibrated_with_valid_probabilities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = _write_csv(_make_dataset(), root / "train.csv")
            result = train_species_model(
                [csv_path],
                "species_test",
                calibration_method="auto",
                output_root=root / "models",
            )
            payload = result["payload"]

            self.assertTrue(result["is_calibrated"])
            self.assertEqual(payload["calibration_method"], "sigmoid")
            self.assertTrue(payload["is_calibrated"])

            probabilities = np.asarray(
                payload["model"].predict_proba(
                    normalize_training_frame(_make_dataset())[SPECIES_FEATURE_COLUMNS]
                ),
                dtype=float,
            )
            self.assertTrue((probabilities >= 0.0).all())
            self.assertTrue((probabilities <= 1.0).all())
            self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6))

            model_dir = Path(result["model_dir"])
            self.assertTrue((model_dir / "model.joblib").exists())
            self.assertTrue((model_dir / "metadata.json").exists())
            metadata = json.loads(
                (model_dir / "metadata.json").read_text(encoding="utf-8")
            )
            for key in (
                "model_type",
                "calibration_method",
                "is_calibrated",
                "feature_columns",
                "class_labels",
                "training_sample_count",
                "recording_count",
                "evaluation",
            ):
                self.assertIn(key, metadata)
            self.assertTrue(metadata["is_calibrated"])
            self.assertEqual(metadata["calibration_method"], "sigmoid")

            evaluation = result["evaluation"]
            self.assertIsNotNone(evaluation)
            self.assertIn("accuracy", evaluation)
            self.assertIn("macro_precision", evaluation)
            self.assertIn("macro_recall", evaluation)
            self.assertIn("macro_f1", evaluation)
            self.assertIn("confusion_matrix", evaluation)
            self.assertIn("log_loss", evaluation)
            self.assertIn("brier_score", evaluation)
            self.assertIn("calibration", evaluation)

    def test_predict_reports_calibrated_probability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = _write_csv(_make_dataset(), root / "train.csv")
            result = train_species_model(
                [csv_path], "species_test", output_root=root / "models"
            )
            frame = _make_dataset().head(20)
            predictions = predict_pulse_frame(result["payload"], frame)
            self.assertTrue(
                (predictions["probability_kind"] == "calibrated_probability").all()
            )
            self.assertTrue(predictions["calibrated_probability"].between(0, 1).all())
            for text in predictions["all_class_probabilities"]:
                probabilities = json.loads(text)
                self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=5)

    def test_insufficient_data_does_not_fake_calibration(self) -> None:
        # 5 条录音 < SPECIES_MIN_RECORDINGS_FOR_CALIBRATION（6）→ 不允许校准
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = _make_dataset(
                {"Pipistrellus abramus": 3, "Myotis sp.": 2},
            )
            csv_path = _write_csv(data, root / "train.csv")
            result = train_species_model(
                [csv_path], "species_small", output_root=root / "models"
            )
            self.assertFalse(result["is_calibrated"])
            self.assertTrue(
                any(INSUFFICIENT_CALIBRATION_WARNING in note for note in result["notes"])
            )
            description = describe_species_model(result["payload"])
            self.assertFalse(description["is_calibrated"])
            self.assertIn("未校准模型", model_status_text(description))
            predictions = predict_pulse_frame(result["payload"], data.head(5))
            self.assertTrue(
                (predictions["probability_kind"] == "model_support").all()
            )
            self.assertTrue(predictions["calibrated_probability"].isna().all())

    def test_single_species_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = _make_dataset({"Pipistrellus abramus": 3})
            csv_path = _write_csv(data, root / "train.csv")
            with self.assertRaises(ValueError) as context:
                train_species_model(
                    [csv_path], "species_one", output_root=root / "models"
                )
            self.assertIn(SINGLE_CLASS_WARNING, str(context.exception))


# ---------------------------------------------------------------------------
# Group split
# ---------------------------------------------------------------------------
class TestGroupSplit(unittest.TestCase):
    def test_same_recording_never_crosses_sets(self) -> None:
        data = normalize_training_frame(_make_dataset())
        groups = source_recording_series(data)
        splits = build_group_splits(data, random_state=42)

        fit_groups = set(groups.iloc[splits["fit_idx"]])
        test_groups = set(groups.iloc[splits["test_idx"]])
        self.assertFalse(fit_groups & test_groups)
        self.assertTrue(splits["calibration_ok"])
        self.assertTrue(splits["calibration_folds"])

        for fold in splits["recording_split"]["folds"]:
            train_recordings = set(fold["train_recordings"])
            calibration_recordings = set(fold["calibration_recordings"])
            # 同一录音绝不同时进入训练与校准两侧
            self.assertFalse(train_recordings & calibration_recordings)
            self.assertTrue(calibration_recordings <= fit_groups)
            self.assertTrue(train_recordings <= fit_groups)

        for train_idx, calib_idx in splits["calibration_folds"]:
            train_groups = set(groups.iloc[train_idx])
            calib_groups = set(groups.iloc[calib_idx])
            self.assertFalse(train_groups & calib_groups)
            self.assertFalse(set(groups.iloc[splits["test_idx"]]) & train_groups)
            self.assertFalse(set(groups.iloc[splits["test_idx"]]) & calib_groups)

    def test_pulse_level_rows_of_one_recording_stay_together(self) -> None:
        data = normalize_training_frame(_make_dataset())
        groups = source_recording_series(data)
        splits = build_group_splits(data, random_state=7)
        assignments: dict[str, set[str]] = {}
        for name, indices in (
            ("fit", splits["fit_idx"]),
            ("test", splits["test_idx"]),
        ):
            for group in set(groups.iloc[indices]):
                assignments.setdefault(group, set()).add(name)
        for sides in assignments.values():
            self.assertEqual(len(sides), 1)


# ---------------------------------------------------------------------------
# Imported model
# ---------------------------------------------------------------------------
class TestImportedModel(unittest.TestCase):
    def _trained_payload(self, root: Path) -> dict:
        csv_path = _write_csv(_make_dataset(), root / "train.csv")
        result = train_species_model(
            [csv_path], "species_import", output_root=root / "models"
        )
        return result["payload"]

    def test_feature_consistent_model_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._trained_payload(root)
            imported = import_species_model(
                root / "models" / "species_import",
                "species_imported",
                output_root=root / "models",
            )
            self.assertTrue(imported["payload"]["is_calibrated"])
            description = describe_species_model(imported["payload"])
            self.assertTrue(description["loaded"])
            self.assertEqual(description["class_count"], 3)

    def test_feature_mismatch_rejected_with_missing_columns(self) -> None:
        payload = {
            "model": _ScriptedModel({}),
            "feature_columns": ["mfcc_1", "peak_freq_khz"],
            "class_labels": ["Species A"],
            "is_calibrated": False,
        }
        with self.assertRaises(ValueError) as context:
            validate_imported_payload(payload)
        message = str(context.exception)
        self.assertIn("mfcc_1", message)
        self.assertIn("缺失列", message)

    def test_predict_rejects_missing_feature_columns(self) -> None:
        payload = {
            "model": _ScriptedModel({1: [1.0, 0.0, 0.0]}),
            "feature_columns": list(SPECIES_FEATURE_COLUMNS),
            "class_labels": ["Species A", "Species B", "Species C"],
            "is_calibrated": False,
        }
        frame = _make_dataset().head(3).drop(columns=[SPECIES_FEATURE_COLUMNS[0]])
        with self.assertRaises(ValueError) as context:
            predict_pulse_frame(payload, frame)
        self.assertIn(SPECIES_FEATURE_COLUMNS[0], str(context.exception))

    def test_temporary_acoustic_model_rejected(self) -> None:
        payload = {
            "temporary_classifier": object(),
            "raw_cluster_to_type": {},
            "feature_columns": list(SPECIES_FEATURE_COLUMNS),
            "class_labels": ["A", "B", "C", "D"],
        }
        with self.assertRaises(ValueError) as context:
            validate_imported_payload(payload)
        self.assertIn("临时声型", str(context.exception))


# ---------------------------------------------------------------------------
# Uncalibrated model / uncertain
# ---------------------------------------------------------------------------
class TestUncalibratedAndUncertain(unittest.TestCase):
    def _payload(self, proba_map: dict[int, list[float]], calibrated: bool = False) -> dict:
        return {
            "model": _ScriptedModel(proba_map),
            "feature_columns": list(SPECIES_FEATURE_COLUMNS),
            "class_labels": ["Species A", "Species B", "Species C"],
            "is_calibrated": calibrated,
            "calibration_method": "sigmoid" if calibrated else "",
            "uncertainty_threshold": 0.60,
        }

    def _frame(self) -> pd.DataFrame:
        frame = _make_dataset(recording_counts={"Myotis sp.": 1}).head(3).copy()
        frame["duration_ms"] = [1.0, 2.0, 3.0]
        return frame

    def test_below_threshold_becomes_uncertain(self) -> None:
        payload = self._payload(
            {
                1: [0.48, 0.43, 0.09],
                2: [0.90, 0.05, 0.05],
                3: [0.55, 0.35, 0.10],
            }
        )
        predictions = predict_pulse_frame(payload, self._frame(), uncertain_threshold=0.60)
        labels = list(predictions["model_inferred_label"])
        self.assertEqual(labels[0], SPECIES_UNCERTAIN_LABEL)
        self.assertEqual(labels[1], "Species A")
        self.assertEqual(labels[2], SPECIES_UNCERTAIN_LABEL)
        self.assertTrue(bool(predictions["is_uncertain"].iloc[0]))
        self.assertFalse(bool(predictions["is_uncertain"].iloc[1]))

    def test_uncalibrated_model_never_reports_calibrated_probability(self) -> None:
        payload = self._payload({1: [0.48, 0.43, 0.09]})
        predictions = predict_pulse_frame(payload, self._frame())
        self.assertTrue(predictions["calibrated_probability"].isna().all())
        self.assertTrue((predictions["probability_kind"] == "model_support").all())
        description = describe_species_model(payload)
        self.assertFalse(description["is_calibrated"])
        self.assertIn("未校准模型", model_status_text(description))
        self.assertNotIn("已启用（校准概率）", model_status_text(description))

    def test_calibrated_model_reports_calibrated_probability(self) -> None:
        payload = self._payload({1: [0.48, 0.43, 0.09]}, calibrated=True)
        predictions = predict_pulse_frame(payload, self._frame())
        self.assertTrue(
            (predictions["probability_kind"] == "calibrated_probability").all()
        )
        self.assertTrue(predictions["calibrated_probability"].between(0, 1).all())

    def test_calibrate_imported_raw_model_with_labels(self) -> None:
        from sklearn.ensemble import RandomForestClassifier

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training = normalize_training_frame(_make_dataset())
            raw = RandomForestClassifier(n_estimators=30, random_state=0)
            raw.fit(training[SPECIES_FEATURE_COLUMNS], training["expert_label"])
            payload = {
                "model": raw,
                "feature_columns": list(SPECIES_FEATURE_COLUMNS),
                "class_labels": sorted(set(training["expert_label"])),
                "is_calibrated": False,
            }
            calibration_csv = _write_csv(
                _make_dataset(seed=3), root / "calibration.csv"
            )
            result = calibrate_species_model_with_labels(
                payload, [calibration_csv], output_root=root / "models"
            )
            self.assertTrue(result["payload"]["is_calibrated"])
            self.assertEqual(result["payload"]["calibration_method"], "sigmoid")

    def test_calibrate_rejects_unknown_species(self) -> None:
        from sklearn.ensemble import RandomForestClassifier

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training = normalize_training_frame(_make_dataset())
            raw = RandomForestClassifier(n_estimators=30, random_state=0)
            raw.fit(training[SPECIES_FEATURE_COLUMNS], training["expert_label"])
            payload = {
                "model": raw,
                "feature_columns": list(SPECIES_FEATURE_COLUMNS),
                "class_labels": sorted(set(training["expert_label"])),
                "is_calibrated": False,
            }
            alien = _make_dataset({"Unknown sp.": 2, "Pipistrellus abramus": 2})
            csv_path = _write_csv(alien, root / "alien.csv")
            with self.assertRaises(ValueError) as context:
                calibrate_species_model_with_labels(
                    payload, [csv_path], output_root=root / "models"
                )
            self.assertIn("未知的物种", str(context.exception))


# ---------------------------------------------------------------------------
# 专家标签保护（模型输出绝不写回 expert_label）
# ---------------------------------------------------------------------------
class TestExpertLabelSafety(unittest.TestCase):
    def test_prediction_columns_never_contain_expert_label(self) -> None:
        payload = {
            "model": _ScriptedModel({1: [0.9, 0.05, 0.05]}),
            "feature_columns": list(SPECIES_FEATURE_COLUMNS),
            "class_labels": ["Species A", "Species B", "Species C"],
            "is_calibrated": False,
        }
        frame = _make_dataset(recording_counts={"Myotis sp.": 1}).head(2).copy()
        frame["duration_ms"] = [1.0, 1.0]
        frame["expert_label"] = ""
        predictions = predict_pulse_frame(payload, frame)
        self.assertNotIn("expert_label", [
            column for column in predictions.columns if column.startswith("model")
        ])
        self.assertTrue((predictions["model_inferred_label"] != "").all())
        self.assertTrue((frame["expert_label"] == "").all())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
