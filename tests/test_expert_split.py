"""专家数据划分 / 冻结评估 / 污染断言 / 评估表 测试（⑥ 伪标签反哺）。

覆盖需求 Test：
1. 按 recording 划分（绝不拆同一录音的 pulse）；
2. Evaluation 不进入训练（训练指纹不含评估录音/脉冲）；
4. 固定 split 可复现；
5. 冻结 Evaluation（新标注不改考试卷，去向显式 parking/train）；
6. Macro-F1 手算一致；
7. 缺 class evaluation truth → N/A + INSUFFICIENT_EVAL_SUPPORT。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bat_classifier.config import (  # noqa: E402
    EXPERT_SPLIT_STATUS_INSUFFICIENT,
    SPECIES_FEATURE_COLUMNS,
)
from bat_classifier.expert_split import (  # noqa: E402
    EvalContaminationError,
    ExpertSplitStore,
    apply_expert_split,
    assert_eval_isolation,
    make_expert_split,
    new_label_destination,
    split_summary,
)
from bat_classifier.expert_split import (  # noqa: E402
    NEW_LABEL_DESTINATION_PARKED,
    NEW_LABEL_DESTINATION_TRAIN,
)
from bat_classifier.species_model import (  # noqa: E402
    INSUFFICIENT_EVAL_SUPPORT,
    classification_table,
    compare_evaluations,
    train_species_model,
)

SPECIES_PEAKS = {"Species A": 25.0, "Species B": 45.0, "Species C": 65.0}


def _make_dataset(
    species_recordings: dict[str, int],
    pulses_per_recording: int = 3,
    prefix: str = "rec",
) -> pd.DataFrame:
    """合成专家标注表（reviewed + expert_label + 录音来源 + 特征）。"""
    rows: list[dict] = []
    pulse_index = 0
    for species, recording_count in species_recordings.items():
        peak = SPECIES_PEAKS[species]
        for recording in range(recording_count):
            source_file = (
                f"{species.replace(' ', '')}_{prefix}{recording}.wav"
            )
            for pulse in range(pulses_per_recording):
                features = {column: 30.0 for column in SPECIES_FEATURE_COLUMNS}
                features["peak_freq_khz"] = peak + 0.1 * pulse
                features["duration_ms"] = 20.0 + pulse
                rows.append(
                    {
                        "pulse_id": f"P{pulse_index:05d}",
                        "expert_label": species,
                        "expert_confidence": 1.0,
                        "review_status": "reviewed",
                        "source_file": source_file,
                        "source_path": source_file,
                        "audio_hash": f"hash-{source_file}",
                        **features,
                    }
                )
                pulse_index += 1
    return pd.DataFrame(rows)


def _side_of(split: dict, recording: str) -> str:
    if recording in {str(value) for value in split.get("train_recordings") or []}:
        return "train"
    if recording in {
        str(value) for value in split.get("evaluation_recordings") or []
    }:
        return "evaluation"
    return "unknown"


class TestRecordingLevelSplit(unittest.TestCase):
    """Test 1：按 source_recording 整体拆分，绝不拆同一录音的 pulse。"""

    def test_no_recording_on_both_sides(self) -> None:
        data = _make_dataset(
            {"Species A": 6, "Species B": 6, "Species C": 6},
            pulses_per_recording=3,
        )
        split = make_expert_split(data, eval_fraction=0.20, random_state=42)

        train = {str(value) for value in split["train_recordings"]}
        evaluation = {str(value) for value in split["evaluation_recordings"]}
        self.assertTrue(train.isdisjoint(evaluation))
        self.assertEqual(
            train | evaluation, set(data["source_file"].astype(str))
        )

        # 同一录音的所有 pulse 必须在同一侧（录音级拆分的直接证明）
        for source_file, group in data.groupby("source_file"):
            sides = {
                _side_of(split, str(row["source_file"]))
                for _, row in group.iterrows()
            }
            self.assertEqual(sides, {_side_of(split, str(source_file))})
            pulse_ids = set(group["pulse_id"].astype(str))
            frozen = {
                str(value) for value in split.get("evaluation_pulse_ids") or []
            }
            if _side_of(split, str(source_file)) == "evaluation":
                self.assertTrue(pulse_ids.issubset(frozen))
            else:
                self.assertTrue(pulse_ids.isdisjoint(frozen))

    def test_insufficient_class_reported_not_split(self) -> None:
        data = _make_dataset(
            {
                "Species A": 4,
                "Species B": 4,
                "Species C": 1,  # 只有 1 条录音 → 无法两侧都有真值
            },
            pulses_per_recording=3,
        )
        split = make_expert_split(data, eval_fraction=0.20, random_state=42)

        self.assertEqual(
            split["status"], EXPERT_SPLIT_STATUS_INSUFFICIENT
        )
        self.assertIn("Species C", split["insufficient_classes"])
        self.assertIn("Species C", split["uncovered_classes"])
        # 强制留在 Train（Train 侧必须能学到该物种）
        c_recording = str(
            data[data["expert_label"] == "Species C"]["source_file"].iloc[0]
        )
        self.assertEqual(_side_of(split, c_recording), "train")
        self.assertEqual(
            split.get("per_class_evaluation_count", {}).get("Species C", 0), 0
        )
        summary = split_summary(split)
        notes = [str(row.get("note", "")) for row in summary["per_class"]]
        self.assertTrue(
            any("无法计算 Recall / F1" in note for note in notes)
        )


class TestFrozenEvaluation(unittest.TestCase):
    """Test 4 / 5：固定 split 可复现 + Evaluation Set 冻结。"""

    def test_split_reproducible(self) -> None:
        data = _make_dataset(
            {"Species A": 5, "Species B": 5, "Species C": 5},
            pulses_per_recording=2,
        )
        first = make_expert_split(data, eval_fraction=0.20, random_state=42)
        second = make_expert_split(data, eval_fraction=0.20, random_state=42)
        self.assertEqual(first["train_recordings"], second["train_recordings"])
        self.assertEqual(
            first["evaluation_recordings"], second["evaluation_recordings"]
        )
        self.assertEqual(
            first["evaluation_pulse_ids"], second["evaluation_pulse_ids"]
        )

    def test_new_labels_do_not_touch_frozen_eval(self) -> None:
        data = _make_dataset(
            {"Species A": 6, "Species B": 6}, pulses_per_recording=3
        )
        split = make_expert_split(data, eval_fraction=0.20, random_state=42)
        with tempfile.TemporaryDirectory() as directory:
            store = ExpertSplitStore.load(Path(directory))
            saved = store.save(split, replace=False)
            frozen_ids = {
                str(value) for value in saved["evaluation_pulse_ids"]
            }
            frozen_recordings = {
                str(value) for value in saved["evaluation_recordings"]
            }

            # 冻结后新增标注：一条来自 Train 录音、一条来自 Evaluation 录音
            train_recording = str(saved["train_recordings"][0])
            eval_recording = str(saved["evaluation_recordings"][0])
            new_rows = pd.DataFrame(
                [
                    {
                        "pulse_id": "NEW-TRAIN-1",
                        "expert_label": "Species A",
                        "review_status": "reviewed",
                        "source_file": train_recording,
                        "source_path": train_recording,
                    },
                    {
                        "pulse_id": "NEW-EVAL-1",
                        "expert_label": "Species B",
                        "review_status": "reviewed",
                        "source_file": eval_recording,
                        "source_path": eval_recording,
                    },
                ]
            )
            combined = pd.concat(
                [
                    data[["pulse_id", "expert_label", "review_status", "source_file", "source_path"]],
                    new_rows,
                ],
                ignore_index=True,
            )
            parts = apply_expert_split(combined, saved)

            # 冻结 Evaluation 不变
            self.assertEqual(
                set(parts["evaluation"]["pulse_id"].astype(str)), frozen_ids
            )
            # Train 录音上的新标注 → Train Pool
            self.assertIn(
                "NEW-TRAIN-1", set(parts["train"]["pulse_id"].astype(str))
            )
            # Evaluation 录音上的新标注 → parked（录音隔离优先）
            self.assertIn(
                "NEW-EVAL-1", set(parts["parked"]["pulse_id"].astype(str))
            )
            self.assertNotIn(
                "NEW-EVAL-1", set(parts["evaluation"]["pulse_id"].astype(str))
            )
            self.assertNotIn(
                "NEW-EVAL-1", set(parts["train"]["pulse_id"].astype(str))
            )

            # 去向查询接口
            self.assertEqual(
                new_label_destination(saved, train_recording),
                NEW_LABEL_DESTINATION_TRAIN,
            )
            self.assertEqual(
                new_label_destination(saved, eval_recording),
                NEW_LABEL_DESTINATION_PARKED,
            )
            self.assertTrue(frozen_recordings)

    def test_replacement_requires_explicit_flag_and_bumps_version(self) -> None:
        data = _make_dataset(
            {"Species A": 4, "Species B": 4}, pulses_per_recording=2
        )
        split = make_expert_split(data, eval_fraction=0.20, random_state=42)
        with tempfile.TemporaryDirectory() as directory:
            store = ExpertSplitStore.load(Path(directory))
            first = store.save(split, replace=False)
            self.assertEqual(first["split_version"], 1)
            with self.assertRaises(ValueError):
                store.save(split, replace=False)
            second = store.save(dict(split), replace=True)
            self.assertEqual(second["split_version"], 2)


class TestEvalContamination(unittest.TestCase):
    """Test 2：Evaluation 一条都不进入训练 / 校准 / 伪标签。"""

    def test_assert_eval_isolation_names_offending_recording(self) -> None:
        with self.assertRaises(EvalContaminationError) as context:
            assert_eval_isolation(
                ["speciesA_rec0.wav"],
                ["speciesA_rec0.wav"],
                "unit-test",
            )
        message = str(context.exception)
        self.assertIn("EVAL_CONTAMINATED", message)
        self.assertIn("speciesA_rec0.wav", message)

    def test_eval_samples_never_enter_training(self) -> None:
        data = _make_dataset(
            {"Species A": 6, "Species B": 6, "Species C": 6},
            pulses_per_recording=3,
        )
        split = make_expert_split(data, eval_fraction=0.20, random_state=42)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_csv = root / "expert_data.csv"
            data.to_csv(source_csv, index=False, encoding="utf-8-sig")
            store = ExpertSplitStore.load(root)
            store.save(split, replace=False)
            split_path = store.path

            result = train_species_model(
                [str(source_csv)],
                "split-model",
                expert_split=str(split_path),
                output_root=root / "models",
            )
            payload = result["payload"]
            eval_recordings = {
                str(value) for value in split["evaluation_recordings"]
            }
            eval_pulses = {
                str(value) for value in split["evaluation_pulse_ids"]
            }

            # 训练指纹：录音与 pulse 都不得包含 Evaluation
            self.assertTrue(
                set(payload["training_recordings"]).isdisjoint(eval_recordings)
            )
            self.assertTrue(
                set(payload["training_pulse_ids"]).isdisjoint(eval_pulses)
            )
            self.assertEqual(
                result["metadata"]["training_labels"]["pseudo"], 0
            )
            self.assertEqual(
                result["metadata"]["eval_labels"], "expert_only"
            )
            # 评估确实来自冻结 Evaluation Set
            self.assertGreater(
                result["metadata"]["training_sample_count"], 0
            )
            self.assertEqual(
                {
                    str(value)
                    for value in result["evaluation"]["evaluation_recordings"]
                },
                eval_recordings,
            )
            self.assertTrue(result["evaluation"].get("per_class"))

    def test_pseudo_on_eval_recording_rejected(self) -> None:
        data = _make_dataset(
            {"Species A": 6, "Species B": 6}, pulses_per_recording=3
        )
        split = make_expert_split(data, eval_fraction=0.20, random_state=42)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_csv = root / "expert_data.csv"
            data.to_csv(source_csv, index=False, encoding="utf-8-sig")
            store = ExpertSplitStore.load(root)
            store.save(split, replace=False)
            split_path = store.path

            eval_recording = str(split["evaluation_recordings"][0])
            poisoned = data.iloc[[0]].copy()
            poisoned["source_file"] = eval_recording
            poisoned["source_path"] = eval_recording
            poisoned["pulse_id"] = "POISON-1"
            poisoned["pseudo_label"] = poisoned["expert_label"]
            poisoned["expert_label"] = ""

            with self.assertRaises(EvalContaminationError):
                train_species_model(
                    [str(source_csv)],
                    "poisoned-model",
                    expert_split=str(split_path),
                    pseudo_source=poisoned,
                    output_root=root / "models",
                )


class TestEvaluationTable(unittest.TestCase):
    """Test 6 / 7：手算指标一致 + 缺真值物种 N/A。"""

    def test_precision_recall_f1_hand_computed(self) -> None:
        y_true = ["A", "A", "A", "A", "A", "B", "B", "B", "B"]
        y_pred = ["A", "A", "A", "A", "B", "B", "B", "B", "A"]
        table = classification_table(y_true, y_pred, ["A", "B", "C"])
        rows = {row["species"]: row for row in table["rows"]}

        # A: support 5, predicted 5, TP 4 → 0.8 / 0.8 / 0.8
        self.assertEqual(rows["A"]["support"], 5)
        self.assertAlmostEqual(rows["A"]["precision"], 4 / 5)
        self.assertAlmostEqual(rows["A"]["recall"], 4 / 5)
        self.assertAlmostEqual(rows["A"]["f1"], 0.8)
        # B: support 4, predicted 4, TP 3 → 0.75 / 0.75 / 0.75
        self.assertAlmostEqual(rows["B"]["precision"], 3 / 4)
        self.assertAlmostEqual(rows["B"]["recall"], 3 / 4)
        self.assertAlmostEqual(rows["B"]["f1"], 0.75)
        # Macro-F1 = (0.8 + 0.75) / 2 = 0.775
        self.assertAlmostEqual(table["macro_f1"], 0.775)
        self.assertEqual(set(table["macro_f1_classes"]), {"A", "B"})

    def test_zero_support_is_na_not_zero(self) -> None:
        y_true = ["A", "A", "A", "A", "A", "B", "B", "B", "B"]
        y_pred = ["A", "A", "A", "A", "B", "B", "B", "B", "A"]
        table = classification_table(y_true, y_pred, ["A", "B", "C"])
        rows = {row["species"]: row for row in table["rows"]}

        # C 的 Evaluation 真值为 0 → N/A，绝不输出误导性的 F1 = 0
        self.assertEqual(rows["C"]["support"], 0)
        self.assertIsNone(rows["C"]["precision"])
        self.assertIsNone(rows["C"]["recall"])
        self.assertIsNone(rows["C"]["f1"])
        self.assertEqual(rows["C"]["status"], INSUFFICIENT_EVAL_SUPPORT)
        self.assertIn("C", table["macro_f1_excluded"])
        self.assertAlmostEqual(table["macro_f1"], 0.775)

    def test_compare_evaluations_flags_recall_drop(self) -> None:
        old_metrics = {
            "per_class": [
                {"species": "A", "support": 4, "recall": 0.90, "f1": 0.90},
            ],
            "macro_f1_supported": 0.90,
        }
        new_metrics = {
            "per_class": [
                {"species": "A", "support": 4, "recall": 0.80, "f1": 0.85},
            ],
            "macro_f1_supported": 0.85,
        }
        comparison = compare_evaluations(old_metrics, new_metrics)
        self.assertIn("A", comparison["recall_drop_classes"])
        self.assertFalse(comparison["adoptable"])
        self.assertAlmostEqual(comparison["macro_f1_delta"], -0.05)


if __name__ == "__main__":
    unittest.main()
