"""Cluster 四类 representative × 物种模型对照统计 测试。

覆盖：agreement_rate 定义、average_calibrated_probability（含预测错样本）、
Center/Boundary/Random/Outlier 四组独立统计、综合统计按 pulse 去重、
多 Cluster 可输出同一物种、状态判定、TRAINING_DATA_OVERLAP、
未校准模型不得输出“校准概率”、模型结果绝不覆盖 expert_label。
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
    MODEL_STATUS_HIGH,
    MODEL_STATUS_MIXED,
    MODEL_OVERLAP_FLAG,
    MODEL_STATUS_MODERATE,
    SPECIES_FEATURE_COLUMNS,
    SPECIES_UNCERTAIN_LABEL,
)
from bat_classifier.curation.model_comparison import (  # noqa: E402
    compute_group_stats,
    compute_overall_stats,
    determine_cluster_status,
    run_model_comparison,
    split_reasons,
)
from bat_classifier.curation.pipeline import CurationRun  # noqa: E402


# ---------------------------------------------------------------------------
# 测试工具
# ---------------------------------------------------------------------------
def _pred_row(
    pulse_id: str,
    label: str,
    probabilities: dict[str, float],
    uncertain: bool = False,
    selection_reason: str = "center",
    cluster_id: str = "001",
) -> dict:
    return {
        "pulse_id": pulse_id,
        "cluster_id": cluster_id,
        "selection_reason": selection_reason,
        "model_inferred_label": label,
        "predicted_label": label,
        "calibrated_probability": max(probabilities.values()),
        "all_class_probabilities": json.dumps(probabilities),
        "is_uncertain": uncertain,
    }


class _ScriptedModel:
    """按 duration_ms 编码返回固定概率的极简分类器。"""

    classes_ = np.array(["Species A", "Species B"])

    def __init__(
        self,
        proba_map: dict[int, list[float]],
        default: list[float] | None = None,
    ) -> None:
        self.proba_map = proba_map
        self.default = default or [0.5, 0.5]

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


def _write_representatives(run: CurationRun, rows: pd.DataFrame) -> None:
    run.tables_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(
        run.tables_dir / "representatives.csv", index=False, encoding="utf-8-sig"
    )


def _rep_rows(cluster_id: str, spec: list[tuple[str, str]]) -> pd.DataFrame:
    """spec：[(pulse_id, selection_reason), ...]"""
    rng = np.random.default_rng(abs(hash(cluster_id)) % (2**31))
    rows = []
    for pulse_id, reason in spec:
        row = {
            "pulse_id": pulse_id,
            "cluster_id": cluster_id,
            "cluster_label": f"Cluster {cluster_id}",
            "selection_reason": reason,
            "source_file": f"rec_{pulse_id}.wav",
            "source_path": f"/data/rec_{pulse_id}.wav",
            "audio_hash": f"hash_{pulse_id}",
            "pulse_file": f"clusters/cluster_{cluster_id}/audio/{pulse_id}.wav",
            "spectrogram_file": "",
            "duration_ms": float(abs(hash(pulse_id)) % 900 + 1),
        }
        for column in SPECIES_FEATURE_COLUMNS:
            row[column] = float(rng.normal(30.0, 1.0))
        row["duration_ms"] = float(abs(hash(pulse_id)) % 900 + 1)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------
class TestAgreement(unittest.TestCase):
    def test_aaaab_dominant_a_agreement_08(self) -> None:
        rows = [
            _pred_row(f"p{index}", "A", {"A": 0.9, "B": 0.1}) for index in range(4)
        ]
        rows.append(_pred_row("p5", "B", {"A": 0.4, "B": 0.6}))
        stats = compute_group_stats(pd.DataFrame(rows), "center")
        self.assertEqual(stats["dominant_label"], "A")
        self.assertEqual(stats["dominant_count"], 4)
        self.assertAlmostEqual(stats["agreement_rate"], 0.8)

    def test_uncertain_counts_in_denominator(self) -> None:
        rows = [
            _pred_row("p1", "A", {"A": 0.9, "B": 0.1}),
            _pred_row("p2", "A", {"A": 0.9, "B": 0.1}),
            _pred_row("p3", "A", {"A": 0.9, "B": 0.1}),
            _pred_row("p4", "B", {"A": 0.4, "B": 0.6}),
            _pred_row("p5", SPECIES_UNCERTAIN_LABEL, {"A": 0.4, "B": 0.4}, uncertain=True),
        ]
        stats = compute_group_stats(pd.DataFrame(rows), "outlier")
        self.assertEqual(stats["dominant_label"], "A")
        self.assertEqual(stats["sample_count"], 5)
        self.assertAlmostEqual(stats["agreement_rate"], 0.6)
        self.assertEqual(stats["uncertain_count"], 1)
        self.assertEqual(stats["species_distribution"].get(SPECIES_UNCERTAIN_LABEL), 1)


# ---------------------------------------------------------------------------
# Average calibrated probability
# ---------------------------------------------------------------------------
class TestAverageCalibratedProbability(unittest.TestCase):
    def test_average_uses_every_sample_including_wrong_predictions(self) -> None:
        probabilities = [0.96, 0.94, 0.91, 0.88]
        rows = [
            _pred_row(f"p{index}", "Species A", {"Species A": value, "Species B": 1 - value})
            for index, value in enumerate(probabilities)
        ]
        # 第 5 个样本实际预测成 B，但 P(A)=0.43 仍必须计入平均
        rows.append(
            _pred_row("p5", "Species B", {"Species A": 0.43, "Species B": 0.57})
        )
        stats = compute_group_stats(pd.DataFrame(rows), "boundary")
        self.assertEqual(stats["dominant_label"], "Species A")
        self.assertAlmostEqual(stats["average_calibrated_probability"], 0.824, places=6)
        self.assertAlmostEqual(stats["median_calibrated_probability"], 0.91, places=6)
        self.assertAlmostEqual(stats["min_calibrated_probability"], 0.43, places=6)
        self.assertAlmostEqual(stats["max_calibrated_probability"], 0.96, places=6)


# ---------------------------------------------------------------------------
# Four representative groups
# ---------------------------------------------------------------------------
class TestFourGroups(unittest.TestCase):
    def test_each_strategy_gets_independent_stats(self) -> None:
        rows = []
        rows += [
            _pred_row(f"c{index}", "Species A", {"Species A": 0.95, "Species B": 0.05}, selection_reason="center")
            for index in range(5)
        ]
        rows += [
            _pred_row(f"b{index}", "Species A" if index < 4 else "Species B", {"Species A": 0.7, "Species B": 0.3}, selection_reason="boundary")
            for index in range(5)
        ]
        rows += [
            _pred_row(f"r{index}", "Species A", {"Species A": 0.9, "Species B": 0.1}, selection_reason="random")
            for index in range(5)
        ]
        rows += [
            _pred_row(f"o{index}", "Species A" if index < 3 else SPECIES_UNCERTAIN_LABEL, {"Species A": 0.5, "Species B": 0.3}, uncertain=index >= 3, selection_reason="outlier")
            for index in range(5)
        ]
        frame = pd.DataFrame(rows)
        for reason, expected_agreement in (
            ("center", 1.0),
            ("boundary", 0.8),
            ("random", 1.0),
            ("outlier", 0.6),
        ):
            members = frame[
                frame["selection_reason"].map(lambda value, r=reason: r in split_reasons(value))
            ]
            stats = compute_group_stats(members, reason)
            self.assertEqual(stats["sample_count"], 5, reason)
            self.assertAlmostEqual(stats["agreement_rate"], expected_agreement, reason)
            self.assertEqual(stats["representative_type"], reason)

    def test_split_reasons_handles_joined_values(self) -> None:
        self.assertEqual(split_reasons("center|boundary"), ["center", "boundary"])
        self.assertEqual(split_reasons("outlier"), ["outlier"])
        self.assertEqual(split_reasons(""), [])


# ---------------------------------------------------------------------------
# Overall dedupe
# ---------------------------------------------------------------------------
class TestOverallDedupe(unittest.TestCase):
    def test_pulse_in_multiple_strategies_counted_once(self) -> None:
        rows = [
            _pred_row("shared", "Species A", {"Species A": 0.9, "Species B": 0.1}, selection_reason="center|boundary|outlier"),
            _pred_row("c2", "Species A", {"Species A": 0.9, "Species B": 0.1}, selection_reason="center"),
            _pred_row("b2", "Species B", {"Species A": 0.3, "Species B": 0.7}, selection_reason="boundary"),
        ]
        frame = pd.DataFrame(rows)
        center = compute_group_stats(
            frame[frame["selection_reason"].map(lambda v: "center" in split_reasons(v))], "center"
        )
        boundary = compute_group_stats(
            frame[frame["selection_reason"].map(lambda v: "boundary" in split_reasons(v))], "boundary"
        )
        self.assertEqual(center["sample_count"], 2)
        self.assertEqual(boundary["sample_count"], 2)

        overall = compute_overall_stats(frame)
        # 三个 unique pulse，shared 只能算一次
        self.assertEqual(overall["sample_count"], 3)
        self.assertEqual(overall["dominant_count"], 2)
        self.assertAlmostEqual(overall["agreement_rate"], 2 / 3)


# ---------------------------------------------------------------------------
# 状态判定
# ---------------------------------------------------------------------------
class TestClusterStatus(unittest.TestCase):
    def test_high_consistency_thresholds(self) -> None:
        center = {"sample_count": 5, "dominant_label": "A", "agreement_rate": 1.0}
        boundary = {"sample_count": 5, "dominant_label": "A", "agreement_rate": 0.8}
        overall = {
            "sample_count": 8,
            "dominant_label": "A",
            "agreement_rate": 0.9,
            "average_calibrated_probability": 0.9,
        }
        self.assertEqual(
            determine_cluster_status(center, boundary, overall), MODEL_STATUS_HIGH
        )

    def test_center_boundary_disagree_is_mixed(self) -> None:
        center = {"sample_count": 5, "dominant_label": "A", "agreement_rate": 1.0}
        boundary = {"sample_count": 5, "dominant_label": "B", "agreement_rate": 1.0}
        overall = {
            "sample_count": 8,
            "dominant_label": "A",
            "agreement_rate": 0.9,
            "average_calibrated_probability": 0.9,
        }
        self.assertEqual(
            determine_cluster_status(center, boundary, overall), MODEL_STATUS_MIXED
        )

    def test_low_overall_agreement_needs_review(self) -> None:
        center = {"sample_count": 5, "dominant_label": "A", "agreement_rate": 1.0}
        boundary = {"sample_count": 5, "dominant_label": "A", "agreement_rate": 1.0}
        overall = {
            "sample_count": 10,
            "dominant_label": "A",
            "agreement_rate": 0.65,
            "average_calibrated_probability": 0.7,
        }
        # Overall agreement < 0.70 → LOW_CONSISTENCY（与 MIXED 一样必须人工审核）
        from bat_classifier.config import MODEL_STATUS_LOW

        self.assertEqual(
            determine_cluster_status(center, boundary, overall),
            MODEL_STATUS_LOW,
        )


# ---------------------------------------------------------------------------
# 端到端：run_model_comparison
# ---------------------------------------------------------------------------
class TestRunModelComparison(unittest.TestCase):
    def _make_run(self, root: Path) -> CurationRun:
        input_dir = root / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        return CurationRun.create(input_dir, base_dir=root / "runs")

    def _payload(self, proba_map: dict[int, list[float]], calibrated: bool = True) -> dict:
        return {
            "model": _ScriptedModel(proba_map),
            "feature_columns": list(SPECIES_FEATURE_COLUMNS),
            "class_labels": ["Species A", "Species B"],
            "is_calibrated": calibrated,
            "calibration_method": "sigmoid" if calibrated else "",
            "uncertainty_threshold": 0.60,
            "training_pulse_ids": [],
            "training_audio_hashes": [],
        }

    def test_end_to_end_writes_tables_and_allows_shared_species(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self._make_run(root)

            rows = pd.concat(
                [
                    _rep_rows(
                        "001",
                        [
                            ("c1", "center"),
                            ("c2", "center"),
                            ("c3", "center"),
                            ("c4", "center"),
                            ("c5", "center"),
                            ("b1", "boundary"),
                            ("b2", "boundary"),
                            ("b3", "boundary"),
                            ("b4", "boundary"),
                            ("b5", "boundary"),
                            ("cb", "center|boundary"),
                            ("r1", "random"),
                            ("o1", "outlier"),
                        ],
                    ),
                    _rep_rows(
                        "002",
                        [
                            ("d1", "center"),
                            ("d2", "center"),
                            ("d3", "center"),
                            ("e1", "boundary"),
                            ("e2", "boundary"),
                            ("e3", "boundary"),
                        ],
                    ),
                ],
                ignore_index=True,
            )
            _write_representatives(run, rows)

            # 全部 pulse 都预测成 Species A：两个 Cluster 允许同一物种
            proba_map = {
                int(round(float(row["duration_ms"]))): [0.95, 0.05]
                for _, row in rows.iterrows()
            }
            payload = self._payload(proba_map)
            result = run_model_comparison(run, payload, uncertain_threshold=0.60)

            self.assertTrue(result["ok"])
            self.assertEqual(len(result["clusters"]), 2)
            self.assertTrue(
                (run.tables_dir / "representative_model_predictions.csv").exists()
            )
            self.assertTrue((run.tables_dir / "cluster_model_comparison.csv").exists())

            predictions = pd.read_csv(
                run.tables_dir / "representative_model_predictions.csv",
                encoding="utf-8-sig",
            )
            self.assertEqual(len(predictions), len(rows))
            self.assertIn("model_inferred_label", predictions.columns)
            self.assertIn("calibrated_probability", predictions.columns)
            self.assertIn("all_class_probabilities", predictions.columns)
            self.assertNotIn("expert_label", predictions.columns)

            comparison = pd.read_csv(
                run.tables_dir / "cluster_model_comparison.csv",
                encoding="utf-8-sig",
                dtype={"cluster_id": str},
            )
            self.assertEqual(len(comparison), 2)
            # 多个 Cluster → 同一物种是合法结果
            self.assertEqual(
                set(comparison["overall_dominant_label"]), {"Species A"}
            )
            cluster_one = comparison[comparison["cluster_id"].astype(str) == "001"].iloc[0]
            self.assertEqual(cluster_one["center_dominant_label"], "Species A")
            self.assertAlmostEqual(float(cluster_one["center_agreement"]), 1.0)
            self.assertAlmostEqual(float(cluster_one["boundary_agreement"]), 1.0)
            # cb（center|boundary）在综合统计中只算一次：13 个 unique pulse
            self.assertEqual(int(cluster_one["overall_sample_count"]), 13)
            self.assertIn(
                cluster_one["model_status"],
                {MODEL_STATUS_HIGH, MODEL_STATUS_MODERATE, MODEL_STATUS_MIXED},
            )

    def test_group_stats_per_strategy_saved_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self._make_run(root)
            rows = _rep_rows(
                "001",
                [
                    ("c1", "center"),
                    ("b1", "boundary"),
                    ("r1", "random"),
                    ("o1", "outlier"),
                ],
            )
            _write_representatives(run, rows)
            proba_map = {
                int(round(float(row["duration_ms"]))): [0.9, 0.1]
                for _, row in rows.iterrows()
            }
            result = run_model_comparison(run, self._payload(proba_map))
            self.assertTrue(result["ok"])
            groups = result["clusters"][0]["groups"]
            self.assertEqual(
                set(groups), {"center", "boundary", "random", "outlier"}
            )
            for reason in ("center", "boundary", "random", "outlier"):
                self.assertEqual(groups[reason]["sample_count"], 1, reason)

    def test_uncalibrated_model_reports_model_support_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self._make_run(root)
            rows = _rep_rows(
                "001",
                [("c1", "center"), ("c2", "center"), ("b1", "boundary")],
            )
            _write_representatives(run, rows)
            proba_map = {
                int(round(float(row["duration_ms"]))): [0.9, 0.1]
                for _, row in rows.iterrows()
            }
            payload = self._payload(proba_map, calibrated=False)
            result = run_model_comparison(run, payload)
            self.assertTrue(result["ok"])
            self.assertFalse(result["is_calibrated"])
            self.assertEqual(result["probability_kind"], "model_support")
            groups = result["clusters"][0]["groups"]
            self.assertIsNone(groups["center"]["average_calibrated_probability"])
            self.assertIsNotNone(groups["center"]["average_model_support"])
            comparison = pd.read_csv(
                run.tables_dir / "cluster_model_comparison.csv",
                encoding="utf-8-sig",
            )
            row = comparison.iloc[0]
            self.assertTrue(pd.isna(row["center_avg_calibrated_probability"]))
            self.assertFalse(pd.isna(row["center_avg_model_support"]))
            self.assertIn("UNCALIBRATED_MODEL", str(row["flags"]))

    def test_training_data_overlap_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self._make_run(root)
            rows = _rep_rows("001", [("c1", "center"), ("c2", "center")])
            _write_representatives(run, rows)
            proba_map = {
                int(round(float(row["duration_ms"]))): [0.9, 0.1]
                for _, row in rows.iterrows()
            }
            payload = self._payload(proba_map)
            payload["training_pulse_ids"] = ["c1"]
            result = run_model_comparison(run, payload)
            self.assertTrue(result["ok"])
            self.assertTrue(result["training_data_overlap"])
            self.assertIn(MODEL_OVERLAP_FLAG, result["clusters"][0]["flags"])

    def test_expert_annotations_never_touched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self._make_run(root)
            rows = _rep_rows("001", [("c1", "center"), ("b1", "boundary")])
            _write_representatives(run, rows)

            annotations = pd.DataFrame(
                [
                    {
                        "pulse_id": "c1",
                        "cluster_id": "001",
                        "source_file": "rec_c1.wav",
                        "source_path": "/data/rec_c1.wav",
                        "selection_reason": "center",
                        "expert_label": "",
                        "expert_confidence": "",
                        "review_status": "pending",
                        "notes": "",
                        "reviewed_at": "",
                    }
                ]
            )
            annotations.to_csv(
                run.tables_dir / "expert_annotations.csv",
                index=False,
                encoding="utf-8-sig",
            )

            proba_map = {
                int(round(float(row["duration_ms"]))): [0.9, 0.1]
                for _, row in rows.iterrows()
            }
            result = run_model_comparison(run, self._payload(proba_map))
            self.assertTrue(result["ok"])

            after = pd.read_csv(
                run.tables_dir / "expert_annotations.csv", encoding="utf-8-sig"
            )
            expert_label = after.iloc[0]["expert_label"]
            self.assertTrue(pd.isna(expert_label) or str(expert_label) == "")
            self.assertEqual(str(after.iloc[0]["review_status"]), "pending")

    def test_missing_model_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self._make_run(root)
            _write_representatives(run, _rep_rows("001", [("c1", "center")]))
            result = run_model_comparison(run, None)
            self.assertFalse(result["ok"])
            self.assertIn("模型", result["error"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
