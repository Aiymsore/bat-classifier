"""Evaluation 生命周期测试（Fixed Gold Evaluation，⑥ 伪标签反哺 v3）。

覆盖需求 Test：
1. Evaluation immutable：模型训练 / pseudo round / 新标注均不改 Gold Evaluation v1；
2. Pseudo cannot become Evaluation：label_source=pseudo 永远进不了 Evaluation；
3. Evaluation excluded from training：训练 / 校准 / OOF / purity / core radius /
   local expert check / pseudo dataset 全部不含 Evaluation；
4. EVAL_CONTAMINATED：评估录音混入训练侧 → 立即终止并列出 source_recording；
5. Same Evaluation comparison：新旧模型必须同时重新在同一 Gold Evaluation 上评估。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from bat_classifier.config import SPECIES_FEATURE_COLUMNS  # noqa: E402
from bat_classifier.expert_split import (  # noqa: E402
    EvalContaminationError,
    ExpertSplitStore,
    GoldEvaluationError,
    assert_no_pseudo_labels,
    create_gold_evaluation,
    gold_evaluation_path,
    gold_record_from_split,
    list_gold_evaluations,
    load_gold_evaluation,
    make_expert_split,
)
from bat_classifier.species_model import (  # noqa: E402
    compare_models_on_gold_evaluation,
    evaluate_payload_on_gold_evaluation,
    train_species_model,
)
import test_pseudo_labels as tpl  # noqa: E402  （复用合成任务构建器）


class TestGoldEvaluationImmutable(unittest.TestCase):
    """Test 1 / 2：Gold Evaluation 不可变；pseudo 永远进不了 Evaluation。"""

    def test_gold_eval_v1_survives_training_and_pseudo_rounds(self) -> None:
        experts, candidates, split = tpl._local_heterogeneity_data()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, _ = tpl._write_cluster_run(root, experts, candidates, split)
            gold_path = gold_evaluation_path(root, 1)
            self.assertTrue(gold_path.exists())
            with open(gold_path, "r", encoding="utf-8") as handle:
                before = json.load(handle)

            # ---- 破坏性活动 1：模型训练（走冻结评估路径） ----
            data = pd.DataFrame(experts).rename(
                columns={"dur": "duration_ms", "peak": "peak_freq_khz"}
            )
            for column in SPECIES_FEATURE_COLUMNS:
                if column not in data.columns:
                    data[column] = 30.0
            data["review_status"] = "reviewed"
            data["source_path"] = data["source_file"]
            source_csv = root / "expert_data.csv"
            data.to_csv(source_csv, index=False, encoding="utf-8-sig")
            train_species_model(
                [str(source_csv)],
                "lifecycle-model",
                expert_split=str(ExpertSplitStore.load(root).path),
                output_root=root / "models",
            )

            # ---- 破坏性活动 2：pseudo round ----
            tpl.run_pseudo_feedback(run, random_state=42)

            # ---- 破坏性活动 3：新标注 + 重新划分（split_version +1） ----
            extended = pd.concat(
                [data, pd.DataFrame([{
                    "pulse_id": "NEW-1",
                    "expert_label": "Species A",
                    "review_status": "reviewed",
                    "source_file": str(split["train_recordings"][0]),
                    "source_path": str(split["train_recordings"][0]),
                }])],
                ignore_index=True,
            )
            new_split = make_expert_split(extended, eval_fraction=0.20, random_state=42)
            ExpertSplitStore.load(root).save(new_split, replace=True)

            # ---- 断言：Gold Evaluation v1 完全没变 ----
            with open(gold_path, "r", encoding="utf-8") as handle:
                after = json.load(handle)
            self.assertEqual(before, after)
            self.assertEqual(
                before["truth"], gold_record_from_split(split)["truth"]
            )
            # 旧版本永远保留：v2 归档另存，不覆盖 v1
            self.assertIn(1, list_gold_evaluations(root))

    def test_pseudo_labels_can_never_enter_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pseudo_row = pd.DataFrame(
                [
                    {
                        "pulse_id": "P-PSEUDO-1",
                        "expert_label": "Species A",
                        "label_source": "pseudo",
                        "pseudo_label": "Species A",
                        "source_file": "new_rec.wav",
                        "source_path": "new_rec.wav",
                        "review_status": "",
                    }
                ]
            )
            with self.assertRaises(GoldEvaluationError) as context:
                create_gold_evaluation(
                    root, pseudo_row, description="bad v2"
                )
            self.assertIn("P-PSEUDO-1", str(context.exception))
            # 没有写入任何 gold 文件
            self.assertEqual(list_gold_evaluations(root), [])

    def test_model_confirmation_is_not_expert_truth(self) -> None:
        """'人工只是确认模型预测'的行（无 reviewed 真值）也不许进 Evaluation。"""
        fake = pd.DataFrame(
            [
                {
                    "pulse_id": "P-CONFIRM-1",
                    "expert_label": "Species A",
                    "label_source": "model",
                    "review_status": "pending",
                    "source_file": "new_rec.wav",
                    "source_path": "new_rec.wav",
                }
            ]
        )
        with self.assertRaises(GoldEvaluationError):
            assert_no_pseudo_labels(fake, "unit-test")

    def test_gold_eval_requires_human_truth_and_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = pd.DataFrame(
                [
                    {
                        "pulse_id": f"GOLD2-{index}",
                        "expert_label": "Species A",
                        "review_status": "reviewed",
                        "source_file": "gold2_rec.wav",
                        "source_path": "gold2_rec.wav",
                    }
                    for index in range(4)
                ]
            )
            # 与训练录音重叠 → EVAL_CONTAMINATED
            with self.assertRaises(EvalContaminationError):
                create_gold_evaluation(
                    root,
                    good,
                    training_recordings=["gold2_rec.wav"],
                )
            record = create_gold_evaluation(
                root,
                good,
                description="Evaluation v2（新设备/地区）",
                training_recordings=["other_rec.wav"],
            )
            self.assertEqual(record["gold_eval_version"], 1)
            # 同版本绝不覆盖
            with self.assertRaises(FileExistsError):
                create_gold_evaluation(root, good, version=1)
            self.assertEqual(len(load_gold_evaluation(root, 1)["truth"]), 4)


class TestEvaluationExcludedFromTraining(unittest.TestCase):
    """Test 3：Evaluation 一条都不进训练侧任何环节。"""

    def test_evaluation_excluded_from_all_training_side_evidence(self) -> None:
        experts, candidates, split = tpl._local_heterogeneity_data()
        # 额外在 Evaluation 录音上放 3 条 B 专家行：若泄漏进局部邻域，
        # B-CAND 的 local_expert_agreement 会从 0.6 变成 0.0
        experts = list(experts) + [
            {
                "pulse_id": f"B-EVAL-{pulse}",
                "expert_label": "Species B",
                "source_file": "a_eval_rec0.wav",
                "peak": 25.0,
                "dur": 20.0,
                "selection_reason": "center",
                "start_s": float(pulse) * 2.0,
            }
            for pulse in range(3)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, frame = tpl._write_cluster_run(root, experts, candidates, split)
            result = tpl.run_pseudo_feedback(run, random_state=42)
            gates = pd.read_csv(result["gates_csv"], encoding="utf-8-sig")
            row = gates.iloc[0]
            candidates_table = pd.read_csv(
                result["candidates_csv"], encoding="utf-8-sig"
            ).fillna("")
            dataset = pd.read_csv(
                result["dataset_csv"], encoding="utf-8-sig"
            ).fillna("")

            eval_pulses = {
                str(value) for value in split["evaluation_pulse_ids"]
            } | {"B-EVAL-0", "B-EVAL-1", "B-EVAL-2"}
            eval_recordings = {"a_eval_rec0.wav"}

            # 训练侧计数 / 纯度：只用 Train Pool
            self.assertEqual(int(row["reviewed_count_train"]), 22)
            self.assertAlmostEqual(float(row["cluster_purity"]), 20 / 22, places=3)
            self.assertNotEqual(int(row["reviewed_count_train"]), 25)

            # core radius：只用 Train Pool 一致证据（A-EXP 非 boundary）P95
            train_expert_ids = [
                f"A-EXP-{recording}-{pulse}"
                for recording in range(5)
                for pulse in range(1, 4)
            ]
            core_distances = pd.to_numeric(
                frame[frame["pulse_id"].isin(train_expert_ids)][
                    "distance_to_center"
                ],
                errors="coerce",
            ).dropna()
            expected_radius = float(np.quantile(core_distances, 0.95))
            self.assertAlmostEqual(
                float(row["core_radius"]), expected_radius, places=6
            )

            # local expert check：邻居只数 Train Pool 专家（0.6 = B B A A A），
            # Evaluation 上的 B 专家不许进邻域（否则会变成 0.0）
            b_rows = candidates_table[
                candidates_table["pulse_id"].astype(str).str.startswith("B-CAND")
            ]
            self.assertTrue(len(b_rows) > 0)
            agreements = pd.to_numeric(b_rows["local_expert_agreement"])
            self.assertTrue((agreements == 0.6).all(), set(agreements))

            # pseudo dataset / candidates：Evaluation 脉冲一条都不进
            self.assertTrue(
                eval_pulses.isdisjoint(set(dataset["pulse_id"].astype(str)))
            )
            self.assertTrue(
                eval_pulses.isdisjoint(set(candidates_table["pulse_id"].astype(str)))
            )

            # OOF 训练录音 ∩ Evaluation 录音 = ∅
            oof_meta = result["metadata"]["oof"]
            for fold in oof_meta.get("folds", []):
                self.assertTrue(
                    eval_recordings.isdisjoint(set(fold["train_recordings"]))
                )
            full = oof_meta.get("full_model") or {}
            if full:
                self.assertTrue(
                    eval_recordings.isdisjoint(set(full["train_recordings"]))
                )


class TestEvalContaminated(unittest.TestCase):
    """Test 4：评估录音混入训练侧 → EVAL_CONTAMINATED，立即终止。"""

    def test_pseudo_pipeline_halts_on_contaminated_split(self) -> None:
        experts, candidates, split = tpl._local_heterogeneity_data()
        # 故意让 evaluation_recording 同时出现在 train_recordings
        split["train_recordings"] = list(split["train_recordings"]) + [
            str(split["evaluation_recordings"][0])
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, _ = tpl._write_cluster_run(root, experts, candidates, split)
            with self.assertRaises(EvalContaminationError) as context:
                tpl.run_pseudo_feedback(run, random_state=42)
            message = str(context.exception)
            self.assertIn("EVAL_CONTAMINATED", message)
            self.assertIn("a_eval_rec0.wav", message)


class TestSameGoldEvaluationComparison(unittest.TestCase):
    """Test 5：新旧模型必须同时重新在同一 Gold Evaluation 上评估。"""

    @staticmethod
    def _payload(model, classes: list[str]) -> dict:
        return {
            "model": model,
            "feature_columns": list(SPECIES_FEATURE_COLUMNS),
            "class_labels": classes,
            "is_calibrated": False,
        }

    def test_both_models_rerun_on_same_frozen_gold(self) -> None:
        rng = np.random.default_rng(7)
        rows = []
        for index in range(40):
            species = "Species A" if index < 20 else "Species B"
            peak = 25.0 if species == "Species A" else 45.0
            features = {column: 30.0 for column in SPECIES_FEATURE_COLUMNS}
            features["peak_freq_khz"] = peak + float(rng.normal(0, 0.3))
            features["duration_ms"] = 20.0
            rows.append(
                {
                    "pulse_id": f"G-{index:03d}",
                    "expert_label": species,
                    "source_file": f"g_rec{index % 4}.wav",
                    "source_path": f"g_rec{index % 4}.wav",
                    **features,
                }
            )
        frame = pd.DataFrame(rows)
        x_values = frame[list(SPECIES_FEATURE_COLUMNS)]
        y_values = frame["expert_label"]

        # 新模型：学对；旧模型：标签打乱（故意学错）
        new_model = RandomForestClassifier(n_estimators=10, random_state=1)
        new_model.fit(x_values, y_values)
        old_model = RandomForestClassifier(n_estimators=10, random_state=1)
        old_model.fit(x_values, y_values.map({"Species A": "Species B", "Species B": "Species A"}))

        gold = {
            "gold_eval_version": 3,
            "created_at": "2026-01-01T00:00:00",
            "evaluation_recordings": sorted(set(frame["source_file"])),
            "evaluation_pulse_ids": sorted(frame["pulse_id"]),
            "truth": [
                {
                    "pulse_id": row["pulse_id"],
                    "expert_label": row["expert_label"],
                    "source_recording": row["source_file"],
                }
                for _, row in frame.iterrows()
            ],
            "eval_labels": "expert_only",
        }

        result = compare_models_on_gold_evaluation(
            self._payload(old_model, ["Species A", "Species B"]),
            self._payload(new_model, ["Species A", "Species B"]),
            frame,
            gold,
        )

        # 同一份考试卷、同一份冻结真值、双方都重新跑过
        self.assertTrue(result["same_evaluation"])
        self.assertEqual(result["gold_eval_version"], 3)
        self.assertEqual(
            result["old"]["evaluation_pulse_ids"], result["new"]["evaluation_pulse_ids"]
        )
        self.assertEqual(
            set(result["old"]["evaluation_pulse_ids"]), set(gold["evaluation_pulse_ids"])
        )
        # 旧模型学错 → 指标差；新模型 → 指标好（证明都是真跑出来的）
        self.assertLess(
            result["old"]["macro_f1_supported"], result["new"]["macro_f1_supported"]
        )
        # 新模型显著更好 → 可采纳（adoptable 只看新 vs 旧是否退步）
        self.assertTrue(result["adoptable"])
        self.assertIn("重新评估", result["notes"])

    def test_frozen_truth_overrides_live_labels(self) -> None:
        """考试卷答案来自快照：活标注被篡改也不影响指标（Test 1 的另一半）。"""
        rng = np.random.default_rng(3)
        features = {column: 30.0 for column in SPECIES_FEATURE_COLUMNS}
        features["peak_freq_khz"] = 25.0
        frame = pd.DataFrame(
            [
                {
                    "pulse_id": f"F-{index}",
                    "expert_label": "Species A",
                    "source_file": "f_rec0.wav",
                    "source_path": "f_rec0.wav",
                    **{**features, "peak_freq_khz": 25.0 + float(rng.normal(0, 0.2))},
                }
                for index in range(8)
            ]
        )
        model = RandomForestClassifier(n_estimators=5, random_state=2)
        model.fit(
            frame[list(SPECIES_FEATURE_COLUMNS)],
            ["Species A"] * 4 + ["Species B"] * 4,
        )
        gold = {
            "gold_eval_version": 1,
            "evaluation_recordings": ["f_rec0.wav"],
            "evaluation_pulse_ids": sorted(frame["pulse_id"]),
            "truth": [
                {
                    "pulse_id": row["pulse_id"],
                    "expert_label": "Species A",
                    "source_recording": "f_rec0.wav",
                }
                for _, row in frame.iterrows()
            ],
        }
        payload = self._payload(model, ["Species A", "Species B"])
        first = evaluate_payload_on_gold_evaluation(payload, frame, gold)
        tampered = frame.copy()
        tampered["expert_label"] = "Species B"  # 篡改活标注
        tampered["label_source"] = "pseudo"
        second = evaluate_payload_on_gold_evaluation(payload, tampered, gold)
        self.assertEqual(first["per_class"], second["per_class"])
        self.assertEqual(first["macro_f1_supported"], second["macro_f1_supported"])
        self.assertEqual(second["eval_labels"], "expert_only")


if __name__ == "__main__":
    unittest.main()
