"""伪标签反哺闭环测试（⑥：专家审核 → Cluster → 伪标签 → 模型反哺）。

覆盖需求 Test：
3.  Evaluation 样本不进入 pseudo gate（纯度/计数只用 Expert Train Pool）；
9.  OOF Scoring 不可用时 pseudo count = 0 并记录原因——Production Model
    即使给 0.999 的概率也改变不了结果（决策链根本不读 Production 概率）；
10. 局部异质区域：Cluster 整体 dominant_fraction >= 90%，但 B 区候选附近
    专家邻居存在明显异类 → 不进入 pseudo dataset（local_expert_conflict）。

v3 新增（Cluster-wide OOF consistency + coverage，修复代表样本偏置）：
6.  representatives 10/10 A、但全簇 OOF 300A/200B → 绝不判 HIGH；
7.  全簇 OOF 与 dominant 一致 + 覆盖充足 → 可判 HIGH；
8.  OOF 覆盖不足（如 40/500）→ INSUFFICIENT_OOF_COVERAGE，即使 40/40 全对；
9'. 覆盖不足 + Production 0.999 → 依旧零传播（绝不 fallback Production）；
10. 多录音 Cluster：每个 recording 的预测来自没见过它的 grouped OOF model。

外加：多样性抽样可复现、数据集溯源列完整。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bat_classifier.config import (  # noqa: E402
    MODEL_STATUS_HIGH,
    MODEL_STATUS_INSUFFICIENT_OOF_COVERAGE,
    PSEUDO_SAMPLE_WEIGHT,
    SPECIES_FEATURE_COLUMNS,
)
from bat_classifier.curation.pseudo_labels import (  # noqa: E402
    REJECT_DOMINANT_FRACTION,
    REJECT_EXPERT,
    REJECT_INSUFFICIENT_COVERAGE,
    REJECT_LOCAL_CONFLICT,
    REJECT_NOT_HIGH,
    REJECT_OOF_INSUFFICIENT,
    run_pseudo_feedback,
)
from bat_classifier.expert_split import ExpertSplitStore  # noqa: E402


class _FakeRun:
    """pseudo_labels 只需要 run_dir / tables_dir / metadata / save_metadata。"""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.tables_dir = run_dir / "tables"
        self.metadata: dict = {}

    def save_metadata(self) -> None:
        pass


def _write_cluster_run(
    root: Path,
    experts: list[dict],
    candidates: list[dict],
    split: dict,
) -> tuple[_FakeRun, pd.DataFrame]:
    """写出 clustered_pulses / cluster_model.joblib / expert_annotations / expert_split。

    特征只动 peak_freq_khz 与 duration_ms，其余恒为 30；
    distance_to_center = 标准化空间到全体均值中心的欧氏距离（与
    curation.clustering 同口径）。行内可用 ``cluster_id`` 覆盖默认 001。
    """
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for item in experts:
        source_file = item["source_file"]
        features = {column: 30.0 for column in SPECIES_FEATURE_COLUMNS}
        features["peak_freq_khz"] = item["peak"]
        features["duration_ms"] = item["dur"]
        rows.append(
            {
                "pulse_id": item["pulse_id"],
                "cluster_id": item.get("cluster_id", "001"),
                "raw_cluster": 0,
                "source_file": source_file,
                "source_path": f"/data/{source_file}",
                "audio_hash": f"hash-{source_file}",
                "start_s": item.get("start_s", 0.0),
                "end_s": item.get("start_s", 0.0) + 0.02,
                "snr_db": 12.0,
                **features,
            }
        )
    for item in candidates:
        source_file = item["source_file"]
        features = {column: 30.0 for column in SPECIES_FEATURE_COLUMNS}
        features["peak_freq_khz"] = item["peak"]
        features["duration_ms"] = item["dur"]
        rows.append(
            {
                "pulse_id": item["pulse_id"],
                "cluster_id": item.get("cluster_id", "001"),
                "raw_cluster": 0,
                "source_file": source_file,
                "source_path": f"/data/{source_file}",
                "audio_hash": f"hash-{source_file}",
                "start_s": item.get("start_s", 0.0),
                "end_s": item.get("start_s", 0.0) + 0.02,
                "snr_db": 12.0,
                **features,
            }
        )
    frame = pd.DataFrame(rows)
    complete = frame[list(SPECIES_FEATURE_COLUMNS)].dropna()
    scaler = StandardScaler().fit(complete)
    scaled = scaler.transform(frame[list(SPECIES_FEATURE_COLUMNS)])
    center = np.nanmean(scaled, axis=0)
    frame["distance_to_center"] = np.linalg.norm(scaled - center, axis=1)
    frame.to_csv(
        tables / "clustered_pulses.csv", index=False, encoding="utf-8-sig"
    )
    joblib.dump(
        {
            "scaler": scaler,
            "feature_columns": list(SPECIES_FEATURE_COLUMNS),
            "random_state": 42,
        },
        tables / "cluster_model.joblib",
    )

    annotations = pd.DataFrame(
        [
            {
                "pulse_id": item["pulse_id"],
                "cluster_id": item.get("cluster_id", "001"),
                "source_file": item["source_file"],
                "source_path": f"/data/{item['source_file']}",
                "selection_reason": item.get("selection_reason", "center"),
                "expert_label": item["expert_label"],
                "expert_confidence": 1.0,
                "review_status": "reviewed",
                "notes": "",
                "reviewed_at": "2024-01-01T00:00:00",
            }
            for item in experts
        ]
    )
    annotations.to_csv(
        tables / "expert_annotations.csv", index=False, encoding="utf-8-sig"
    )
    ExpertSplitStore.load(root).save(split, replace=False)
    return _FakeRun(root), frame


def _split_for(
    train_recordings: list[str],
    eval_recordings: list[str],
    eval_ids: list[str],
    eval_truth: list[dict] | None = None,
) -> dict:
    if eval_truth is None:
        eval_truth = [
            {
                "pulse_id": pulse_id,
                "expert_label": "Species A",
                "source_recording": eval_recordings[0] if eval_recordings else "",
            }
            for pulse_id in eval_ids
        ]
    return {
        "split_version": 1,
        "created_at": "2024-01-01T00:00:00",
        "random_state": 42,
        "eval_fraction_target": 0.2,
        "status": "OK",
        "insufficient_classes": [],
        "uncovered_classes": [],
        "train_recordings": train_recordings,
        "evaluation_recordings": eval_recordings,
        "evaluation_pulse_ids": eval_ids,
        "evaluation_truth": eval_truth,
        "train_recording_count": len(train_recordings),
        "evaluation_recording_count": len(eval_recordings),
        "train_pulse_count": 0,
        "evaluation_pulse_count": len(eval_ids),
    }


def _local_heterogeneity_data() -> tuple[list[dict], list[dict], dict]:
    """Test 10 场景：大簇 90%+ 主导 A，局部 B 区（专家证据与候选都在）。

    - A 专家 20 条（5 录音 × 4，peak 25±1）+ B 专家 2 条（同 1 录音）；
      Train Pool 纯度 20/22 ≈ 0.909 >= 0.90 且 reviewed 22 >= 10；
    - B 候选与 B 专家在特征空间零距离 → 最近 5 个专家邻居 = B B A A A
      （局部一致率 0.6 < 0.80）→ local_expert_conflict；
    - B 区在特征上与 A 区不可分（模型倾向判 A）：若没有局部专家检查，
      这些候选会被候选级 OOF 闸门放行——正是 §4.3.5 的价值；
    - 另有 1 条 A 专家录音进入冻结 Evaluation（不参与任何传播依据）。
    """
    rng = np.random.default_rng(20240101)
    experts: list[dict] = []
    for recording in range(5):
        for pulse in range(4):
            experts.append(
                {
                    "pulse_id": f"A-EXP-{recording}-{pulse}",
                    "expert_label": "Species A",
                    "source_file": f"a_train_rec{recording}.wav",
                    "peak": float(25.0 + rng.normal(0.0, 1.0)),
                    "dur": float(20.0 + rng.normal(0.0, 2.0)),
                    "selection_reason": "boundary" if pulse == 0 else "center",
                    "start_s": float(pulse) * 2.0,
                }
            )
    for pulse in range(2):
        experts.append(
            {
                "pulse_id": f"B-EXP-{pulse}",
                "expert_label": "Species B",
                "source_file": "b_train_rec0.wav",
                "peak": 25.0,
                "dur": 20.0,
                "selection_reason": "center",
                "start_s": float(pulse) * 2.0,
            }
        )
    for pulse in range(4):
        experts.append(
            {
                "pulse_id": f"A-EVAL-{pulse}",
                "expert_label": "Species A",
                "source_file": "a_eval_rec0.wav",
                "peak": float(25.0 + rng.normal(0.0, 1.0)),
                "dur": float(20.0 + rng.normal(0.0, 2.0)),
                "selection_reason": "center",
                "start_s": float(pulse) * 2.0,
            }
        )

    candidates: list[dict] = []
    for pulse in range(30):
        candidates.append(
            {
                "pulse_id": f"A-CAND-{pulse}",
                "source_file": f"a_cand_rec{pulse // 10}.wav",
                "peak": float(25.0 + rng.normal(0.0, 1.5)),
                "dur": float(20.0 + rng.normal(0.0, 2.0)),
                "start_s": float(pulse % 10) * 2.0,
            }
        )
    for pulse in range(3):
        candidates.append(
            {
                "pulse_id": f"B-CAND-{pulse}",
                "source_file": "b_cand_rec0.wav",
                "peak": 25.0,
                "dur": 20.0,
                "start_s": float(pulse) * 2.0,
            }
        )

    split = _split_for(
        train_recordings=[
            *[f"a_train_rec{i}.wav" for i in range(5)],
            "b_train_rec0.wav",
        ],
        eval_recordings=["a_eval_rec0.wav"],
        eval_ids=[f"A-EVAL-{pulse}" for pulse in range(4)],
    )
    return experts, candidates, split


class TestLocalHeterogeneity(unittest.TestCase):
    """Test 10：局部异质区域不得进入 pseudo dataset。"""

    def test_local_conflict_region_rejected(self) -> None:
        experts, candidates, split = _local_heterogeneity_data()
        with tempfile.TemporaryDirectory() as directory:
            run, _ = _write_cluster_run(
                Path(directory), experts, candidates, split
            )
            result = run_pseudo_feedback(run, random_state=42)

            dataset = pd.read_csv(
                result["dataset_csv"], encoding="utf-8-sig"
            ).fillna("")
            candidates_table = pd.read_csv(
                result["candidates_csv"], encoding="utf-8-sig"
            ).fillna("")
            gates = pd.read_csv(result["gates_csv"], encoding="utf-8-sig")

            # 簇门控通过（整体一致性高）——否则谈不上"局部"保护
            self.assertEqual(int(gates.iloc[0]["passed"]), 1)
            self.assertEqual(int(gates.iloc[0]["reviewed_count_train"]), 22)
            self.assertGreaterEqual(float(gates.iloc[0]["cluster_purity"]), 0.90)

            selected = set(dataset["pulse_id"].astype(str))
            b_selected = {pid for pid in selected if pid.startswith("B-CAND")}
            a_selected = {pid for pid in selected if pid.startswith("A-CAND")}
            self.assertEqual(b_selected, set(), "B 区候选不得进入 pseudo dataset")
            self.assertTrue(a_selected, "A 区候选应有安全样本被选中")

            # B 区候选全部因局部专家异类被拒
            b_rows = candidates_table[
                candidates_table["pulse_id"].astype(str).str.startswith("B-CAND")
            ]
            self.assertEqual(len(b_rows), 3)
            self.assertTrue(
                (b_rows["rejection_reason"] == REJECT_LOCAL_CONFLICT).all(),
                set(b_rows["rejection_reason"]),
            )
            self.assertTrue(
                (b_rows["local_expert_check"] == "conflict").all()
            )
            self.assertTrue(
                (pd.to_numeric(b_rows["local_expert_agreement"]) < 0.80).all()
            )

            # A 区被选中样本的局部检查必须干净
            a_pass = candidates_table[
                candidates_table["pulse_id"].astype(str).isin(a_selected)
            ]
            self.assertTrue(
                a_pass["local_expert_check"].isin(["pass", "insufficient"]).all()
            )

    def test_deterministic_diversity_sampling(self) -> None:
        experts, candidates, split = _local_heterogeneity_data()
        with tempfile.TemporaryDirectory() as directory:
            run, _ = _write_cluster_run(
                Path(directory), experts, candidates, split
            )
            first = run_pseudo_feedback(run, random_state=42)
            second = run_pseudo_feedback(run, random_state=42)

            def _stable(path: str) -> pd.DataFrame:
                frame = pd.read_csv(path, encoding="utf-8-sig")
                return frame.drop(columns=["selected_at"], errors="ignore")

            pd.testing.assert_frame_equal(
                _stable(first["dataset_csv"]), _stable(second["dataset_csv"])
            )
            pd.testing.assert_frame_equal(
                _stable(first["candidates_csv"]),
                _stable(second["candidates_csv"]),
            )

    def test_dataset_provenance_columns(self) -> None:
        experts, candidates, split = _local_heterogeneity_data()
        with tempfile.TemporaryDirectory() as directory:
            run, _ = _write_cluster_run(
                Path(directory), experts, candidates, split
            )
            result = run_pseudo_feedback(run, random_state=42)
            dataset = pd.read_csv(
                result["dataset_csv"], encoding="utf-8-sig"
            ).fillna("")
            self.assertFalse(dataset.empty)
            self.assertTrue((dataset["label_source"] == "pseudo").all())
            self.assertTrue((dataset["expert_label"] == "").all())
            self.assertTrue((dataset["review_status"] == "").all())
            self.assertTrue((dataset["propagated_from"] != "").all())
            self.assertTrue((dataset["oof_fold_id"] != "").all())
            self.assertTrue((dataset["applied_threshold"] != "").all())
            self.assertTrue(
                (dataset["pseudo_label"] == "Species A").all()
            )
            self.assertAlmostEqual(
                float(PSEUDO_SAMPLE_WEIGHT), 0.4, places=6
            )


class TestEvalNotInGates(unittest.TestCase):
    """Test 3：Evaluation 样本一条都不进 pseudo gate 证据。"""

    def test_purity_computed_from_train_pool_only(self) -> None:
        # Train：A 16 条 + B 2 条 → 纯度 16/18 ≈ 0.889 < 0.90（应拒）
        # 若错误计入 Evaluation（A 12 条）→ 28/30 ≈ 0.933 >= 0.90（会放行）
        experts: list[dict] = []
        for recording in range(2):
            for pulse in range(8):
                experts.append(
                    {
                        "pulse_id": f"A-TRAIN-{recording}-{pulse}",
                        "expert_label": "Species A",
                        "source_file": f"a_train_rec{recording}.wav",
                        "peak": 25.0,
                        "dur": 20.0,
                        "selection_reason": "center",
                        "start_s": float(pulse) * 2.0,
                    }
                )
        for pulse in range(2):
            experts.append(
                {
                    "pulse_id": f"B-TRAIN-{pulse}",
                    "expert_label": "Species B",
                    "source_file": "b_train_rec0.wav",
                    "peak": 30.0,
                    "dur": 20.0,
                    "selection_reason": "center",
                    "start_s": float(pulse) * 2.0,
                }
            )
        for pulse in range(12):
            experts.append(
                {
                    "pulse_id": f"A-EVAL-{pulse}",
                    "expert_label": "Species A",
                    "source_file": "a_eval_rec0.wav",
                    "peak": 25.0,
                    "dur": 20.0,
                    "selection_reason": "center",
                    "start_s": float(pulse) * 2.0,
                }
            )
        candidates = [
            {
                "pulse_id": f"A-CAND-{pulse}",
                "source_file": "a_cand_rec0.wav",
                "peak": 25.0,
                "dur": 20.0,
                "start_s": float(pulse) * 2.0,
            }
            for pulse in range(4)
        ]
        split = _split_for(
            train_recordings=[
                "a_train_rec0.wav",
                "a_train_rec1.wav",
                "b_train_rec0.wav",
            ],
            eval_recordings=["a_eval_rec0.wav"],
            eval_ids=[f"A-EVAL-{pulse}" for pulse in range(12)],
        )
        with tempfile.TemporaryDirectory() as directory:
            run, _ = _write_cluster_run(
                Path(directory), experts, candidates, split
            )
            result = run_pseudo_feedback(run, random_state=42)

            gates = pd.read_csv(result["gates_csv"], encoding="utf-8-sig")
            row = gates.iloc[0]
            self.assertEqual(int(row["reviewed_count_train"]), 18)
            self.assertAlmostEqual(float(row["cluster_purity"]), 16 / 18, places=3)
            self.assertEqual(int(row["passed"]), 0)
            self.assertEqual(row["gate_reason"], REJECT_DOMINANT_FRACTION)

            self.assertEqual(result["selected_count"], 0)
            candidates_table = pd.read_csv(
                result["candidates_csv"], encoding="utf-8-sig"
            ).fillna("")
            self.assertTrue(
                (
                    candidates_table["rejection_reason"]
                    == REJECT_DOMINANT_FRACTION
                ).all()
            )
            # Evaluation 行：不进候选宇宙、不进伪标签数据集（§九）
            eval_pulses = {f"A-EVAL-{pulse}" for pulse in range(12)}
            self.assertTrue(
                eval_pulses.isdisjoint(set(candidates_table["pulse_id"].astype(str)))
            )
            dataset = pd.read_csv(
                result["dataset_csv"], encoding="utf-8-sig"
            ).fillna("")
            self.assertTrue(
                eval_pulses.isdisjoint(set(dataset["pulse_id"].astype(str)))
            )


class TestNoProductionFallback(unittest.TestCase):
    """Test 9：OOF 不可用 → 零传播；Production 概率 0.999 也不改变结果。"""

    def test_zero_pseudo_when_oof_unavailable(self) -> None:
        experts: list[dict] = []
        for pulse in range(6):
            experts.append(
                {
                    "pulse_id": f"A-TRAIN-0-{pulse}",
                    "expert_label": "Species A",
                    "source_file": "a_train_rec0.wav",
                    "peak": 25.0,
                    "dur": 20.0,
                    "selection_reason": "center",
                    "start_s": float(pulse) * 2.0,
                }
            )
        for pulse in range(5):
            experts.append(
                {
                    "pulse_id": f"A-TRAIN-1-{pulse}",
                    "expert_label": "Species A",
                    "source_file": "a_train_rec1.wav",
                    "peak": 25.0,
                    "dur": 20.0,
                    "selection_reason": "center",
                    "start_s": float(pulse) * 2.0,
                }
            )
        experts.append(
            {
                "pulse_id": "B-TRAIN-0-0",
                "expert_label": "Species B",
                "source_file": "b_train_rec0.wav",
                "peak": 30.0,
                "dur": 20.0,
                "selection_reason": "center",
                "start_s": 0.0,
            }
        )
        # 候选带着 Production Model 的 0.999 概率——决策链根本不读它
        candidates = [
            {
                "pulse_id": f"A-CAND-{pulse}",
                "source_file": "a_cand_rec0.wav",
                "peak": 25.0,
                "dur": 20.0,
                "start_s": float(pulse) * 2.0,
            }
            for pulse in range(4)
        ]
        split = _split_for(
            train_recordings=[
                "a_train_rec0.wav",
                "a_train_rec1.wav",
                "b_train_rec0.wav",
            ],
            eval_recordings=[],
            eval_ids=[],
        )
        with tempfile.TemporaryDirectory() as directory:
            run, frame = _write_cluster_run(
                Path(directory), experts, candidates, split
            )
            # 伪装的 Production 预测（高置信）写入聚类表
            frame["model_inferred_label"] = "Species A"
            frame["calibrated_probability"] = 0.999
            frame.to_csv(
                run.tables_dir / "clustered_pulses.csv",
                index=False,
                encoding="utf-8-sig",
            )

            result = run_pseudo_feedback(run, random_state=42)

            self.assertFalse(result["metadata"]["oof"]["available"])
            self.assertEqual(result["selected_count"], 0)
            self.assertTrue(
                (
                    result["funnel"].get(REJECT_OOF_INSUFFICIENT, 0) > 0
                )
            )
            candidates_table = pd.read_csv(
                result["candidates_csv"], encoding="utf-8-sig"
            ).fillna("")
            self.assertTrue(
                (
                    candidates_table["rejection_reason"]
                    == REJECT_OOF_INSUFFICIENT
                ).all(),
                set(candidates_table["rejection_reason"]),
            )
            gates = pd.read_csv(result["gates_csv"], encoding="utf-8-sig")
            self.assertEqual(int(gates.iloc[0]["passed"]), 0)
            self.assertEqual(
                gates.iloc[0]["gate_reason"], REJECT_OOF_INSUFFICIENT
            )


# ---------------------------------------------------------------------------
# Test 6–10：Cluster-wide OOF consistency + coverage（修复代表样本偏置）
# ---------------------------------------------------------------------------


def _cluster_wide_data(
    a_reps: int = 10,
    a_candidates: int = 290,
    b_candidates: int = 0,
    nan_candidates: int = 0,
) -> tuple[list[dict], list[dict], dict]:
    """cluster 001：A 主导簇（representatives 10/10 Species A）。

    Train Pool 另含 cluster 002 的 Species B 专家（让模型认识 B）；
    候选都在 unseen 录音上 → 由 full-pool OOF 模型评分。
    ``nan_candidates`` 条候选缺特征（写表后为 NaN）→ 无合法 OOF 预测。
    """
    rng = np.random.default_rng(20240102)
    experts: list[dict] = []
    per_recording = a_reps // 2
    for recording in range(2):
        for pulse in range(per_recording):
            experts.append(
                {
                    "pulse_id": f"A-REP-{recording}-{pulse}",
                    "expert_label": "Species A",
                    "cluster_id": "001",
                    "source_file": f"a_train_rec{recording}.wav",
                    "peak": float(25.0 + rng.normal(0.0, 0.5)),
                    "dur": 20.0,
                    "selection_reason": "center",
                    "start_s": float(pulse) * 2.0,
                }
            )
    for recording in range(2):
        for pulse in range(5):
            experts.append(
                {
                    "pulse_id": f"B-OTHER-{recording}-{pulse}",
                    "expert_label": "Species B",
                    "cluster_id": "002",
                    "source_file": f"b_train_rec{recording}.wav",
                    "peak": float(45.0 + rng.normal(0.0, 0.5)),
                    "dur": 20.0,
                    "selection_reason": "center",
                    "start_s": float(pulse) * 2.0,
                }
            )
    for pulse in range(2):
        experts.append(
            {
                "pulse_id": f"A-EVAL-{pulse}",
                "expert_label": "Species A",
                "cluster_id": "001",
                "source_file": "a_eval_rec0.wav",
                "peak": float(25.0 + rng.normal(0.0, 0.5)),
                "dur": 20.0,
                "selection_reason": "center",
                "start_s": float(pulse) * 2.0,
            }
        )

    candidates: list[dict] = []
    for index in range(a_candidates + nan_candidates):
        missing = index >= a_candidates
        candidates.append(
            {
                "pulse_id": f"A-CAND-{index}",
                "source_file": f"a_cand_rec{index % 3}.wav",
                "peak": float("nan") if missing else float(25.0 + rng.normal(0.0, 0.5)),
                "dur": float("nan") if missing else 20.0,
                "start_s": float(index % 100) * 0.5,
            }
        )
    for index in range(b_candidates):
        candidates.append(
            {
                "pulse_id": f"B-CAND-{index}",
                "source_file": f"b_cand_rec{index % 2}.wav",
                "peak": float(45.0 + rng.normal(0.0, 0.5)),
                "dur": 20.0,
                "start_s": float(index % 100) * 0.5,
            }
        )

    eval_ids = ["A-EVAL-0", "A-EVAL-1"]
    split = _split_for(
        train_recordings=[
            "a_train_rec0.wav",
            "a_train_rec1.wav",
            "b_train_rec0.wav",
            "b_train_rec1.wav",
        ],
        eval_recordings=["a_eval_rec0.wav"],
        eval_ids=eval_ids,
        eval_truth=[
            {
                "pulse_id": pulse_id,
                "expert_label": "Species A",
                "source_recording": "a_eval_rec0.wav",
            }
            for pulse_id in eval_ids
        ],
    )
    return experts, candidates, split


def _multi_recording_data() -> tuple[list[dict], list[dict], dict]:
    """cluster 001 跨 3 个 Train 录音（a_train_rec0/1/2）+ unseen 候选录音。"""
    rng = np.random.default_rng(20240103)
    experts: list[dict] = []
    for recording in range(3):
        for pulse in range(5):
            experts.append(
                {
                    "pulse_id": f"A-REP-{recording}-{pulse}",
                    "expert_label": "Species A",
                    "cluster_id": "001",
                    "source_file": f"a_train_rec{recording}.wav",
                    "peak": float(25.0 + rng.normal(0.0, 0.5)),
                    "dur": 20.0,
                    "selection_reason": "center",
                    "start_s": float(pulse) * 2.0,
                }
            )
    for pulse in range(5):
        experts.append(
            {
                "pulse_id": f"B-OTHER-{pulse}",
                "expert_label": "Species B",
                "cluster_id": "002",
                "source_file": "b_train_rec0.wav",
                "peak": float(45.0 + rng.normal(0.0, 0.5)),
                "dur": 20.0,
                "selection_reason": "center",
                "start_s": float(pulse) * 2.0,
            }
        )
    candidates: list[dict] = [
        {
            "pulse_id": f"A-CAND-{index}",
            "source_file": "a_cand_rec0.wav",
            "peak": float(25.0 + rng.normal(0.0, 0.5)),
            "dur": 20.0,
            "start_s": float(index) * 0.5,
        }
        for index in range(20)
    ]
    eval_ids = ["A-EVAL-0", "A-EVAL-1"]
    split = _split_for(
        train_recordings=[
            "a_train_rec0.wav",
            "a_train_rec1.wav",
            "a_train_rec2.wav",
            "b_train_rec0.wav",
        ],
        eval_recordings=["a_eval_rec0.wav"],
        eval_ids=eval_ids,
        eval_truth=[
            {
                "pulse_id": pulse_id,
                "expert_label": "Species A",
                "source_recording": "a_eval_rec0.wav",
            }
            for pulse_id in eval_ids
        ],
    )
    return experts, candidates, split


class TestClusterWideOOFConsistency(unittest.TestCase):
    """第一层 Cluster-level OOF consistency + coverage（§十五–§二十一）。"""

    def _run_one(self, **kwargs) -> tuple[dict, pd.Series]:
        experts, candidates, split = _cluster_wide_data(**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            run, _ = _write_cluster_run(Path(directory), experts, candidates, split)
            result = run_pseudo_feedback(run, random_state=42)
            gates = pd.read_csv(result["gates_csv"], encoding="utf-8-sig").fillna("")
            row = gates[
                pd.to_numeric(gates["cluster_id"], errors="coerce") == 1
            ].iloc[0]
            return result, row

    # ---- Test 6：代表样本偏置 ----
    def test_representative_bias_no_high_consistency(self) -> None:
        """representatives 10/10 A、全簇 OOF 300A/200B → 绝不判 HIGH。"""
        result, row = self._run_one(
            a_reps=10, a_candidates=290, b_candidates=200
        )
        self.assertEqual(int(row["reviewed_count_train"]), 10)
        self.assertEqual(row["dominant_label"], "Species A")
        self.assertAlmostEqual(float(row["cluster_purity"]), 1.0, places=3)
        self.assertEqual(int(row["cluster_total_train_side_pulses"]), 500)
        self.assertEqual(int(row["cluster_oof_eligible_pulses"]), 500)
        agree = int(row["cluster_oof_agree_count"])
        disagree = int(row["cluster_oof_disagree_count"])
        self.assertEqual(agree + disagree, 500)
        self.assertTrue(280 <= agree <= 320, agree)  # ≈300A/200B
        self.assertTrue(
            180 <= disagree <= 220, disagree
        )
        self.assertNotEqual(row["cluster_oof_status"], MODEL_STATUS_HIGH)
        self.assertEqual(int(row["passed"]), 0)
        self.assertEqual(row["gate_reason"], REJECT_NOT_HIGH)
        self.assertEqual(result["selected_count"], 0)

    # ---- Test 7：全簇一致 → 可判 HIGH ----
    def test_high_whole_cluster_consistency(self) -> None:
        """reps 10/10 A、全簇 OOF 几乎全部 A、覆盖充足 → HIGH + 可传播。"""
        result, row = self._run_one(
            a_reps=10, a_candidates=490, b_candidates=0
        )
        self.assertEqual(int(row["cluster_total_train_side_pulses"]), 500)
        self.assertEqual(int(row["cluster_oof_eligible_pulses"]), 500)
        self.assertAlmostEqual(float(row["cluster_oof_coverage"]), 1.0, places=3)
        self.assertGreaterEqual(float(row["cluster_oof_agreement"]), 0.85)
        self.assertEqual(row["cluster_oof_status"], MODEL_STATUS_HIGH)
        self.assertEqual(int(row["passed"]), 1)
        self.assertEqual(row["gate_reason"], "")
        self.assertGreater(result["selected_count"], 0)

    # ---- Test 8：覆盖不足 ----
    def test_insufficient_oof_coverage_blocks_high(self) -> None:
        """OOF 可用 40/40 全对也不许判 HIGH：coverage < 0.80 直接拦下。"""
        result, row = self._run_one(
            a_reps=10, a_candidates=50, nan_candidates=440
        )
        self.assertEqual(int(row["cluster_total_train_side_pulses"]), 500)
        eligible = int(row["cluster_oof_eligible_pulses"])
        self.assertLessEqual(eligible, 80)  # 缺特征行无合法 OOF 预测
        self.assertLess(float(row["cluster_oof_coverage"]), 0.80)
        self.assertEqual(
            row["cluster_oof_status"], MODEL_STATUS_INSUFFICIENT_OOF_COVERAGE
        )
        self.assertEqual(int(row["passed"]), 0)
        self.assertEqual(row["gate_reason"], REJECT_INSUFFICIENT_COVERAGE)
        self.assertEqual(result["selected_count"], 0)
        self.assertTrue(
            result["funnel"].get(REJECT_INSUFFICIENT_COVERAGE, 0) > 0
        )

    # ---- Test 9'：绝不 fallback Production ----
    def test_no_production_fallback_when_coverage_low(self) -> None:
        """覆盖不足 + Production P=0.999 → 依旧零传播。"""
        experts, candidates, split = _cluster_wide_data(
            a_reps=10, a_candidates=50, nan_candidates=440
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, _ = _write_cluster_run(root, experts, candidates, split)
            table_path = root / "tables" / "clustered_pulses.csv"
            frame = pd.read_csv(table_path, encoding="utf-8-sig").fillna("")
            frame["model_inferred_label"] = "Species A"
            frame["calibrated_probability"] = 0.999
            frame["all_class_probabilities"] = (
                '{"Species A": 0.999, "Species B": 0.001}'
            )
            frame.to_csv(table_path, index=False, encoding="utf-8-sig")

            result = run_pseudo_feedback(run, random_state=42)
            self.assertEqual(result["selected_count"], 0)
            dataset = pd.read_csv(
                result["dataset_csv"], encoding="utf-8-sig"
            ).fillna("")
            self.assertTrue(dataset.empty)

    # ---- Test 10：多录音 Cluster 的 grouped OOF ----
    def test_multi_recording_cluster_grouped_oof(self) -> None:
        """每个 recording 的预测来自没见过它的 grouped OOF model。"""
        experts, candidates, split = _multi_recording_data()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, _ = _write_cluster_run(root, experts, candidates, split)
            result = run_pseudo_feedback(run, random_state=42)
            candidates_table = pd.read_csv(
                result["candidates_csv"], encoding="utf-8-sig"
            ).fillna("")
            oof_meta = result["metadata"]["oof"]
            fold_by_id = {fold["fold_id"]: fold for fold in oof_meta["folds"]}

            seen_fold_ids: dict[str, str] = {}
            for recording in ["a_train_rec0.wav", "a_train_rec1.wav", "a_train_rec2.wav"]:
                rows = candidates_table[
                    candidates_table["source_recording"].astype(str) == recording
                ]
                self.assertTrue(len(rows) > 0, recording)
                fold_ids = set(rows["oof_fold_id"].astype(str))
                fold_ids.discard("")
                self.assertEqual(len(fold_ids), 1, recording)
                fold_id = sorted(fold_ids)[0]
                self.assertIn(fold_id, fold_by_id)
                # 评分模型训练录音不含本录音（grouped cross-fit，绝不随机 KFold）
                self.assertNotIn(
                    recording, set(fold_by_id[fold_id]["train_recordings"])
                )
                # 本录音确实在该 fold 的 score 侧
                self.assertIn(
                    recording, set(fold_by_id[fold_id]["score_recordings"])
                )
                seen_fold_ids[recording] = fold_id
            # 每个录音由各自不同的 hold-out 模型评分（绝不统一用某个模型）
            self.assertEqual(len(set(seen_fold_ids.values())), 3)

            # unseen 录音 → full_train_pool 模型（训练录音同样不含它）
            unseen_rows = candidates_table[
                candidates_table["source_recording"].astype(str) == "a_cand_rec0.wav"
            ]
            self.assertTrue(len(unseen_rows) > 0)
            self.assertTrue(
                (unseen_rows["oof_fold_id"].astype(str) == "full_train_pool").all()
            )
            full = oof_meta["full_model"]
            self.assertNotIn(
                "a_cand_rec0.wav", set(full["train_recordings"])
            )
            # 绝不使用 Production Model：cluster 001 每一行的预测出处都是
            # 某个合法 OOF scoring model（非空 fold 出处）
            cluster_one = candidates_table[
                pd.to_numeric(
                    candidates_table["cluster_id"], errors="coerce"
                ) == 1
            ]
            self.assertTrue(
                (cluster_one["oof_fold_id"].astype(str) != "").all(),
                set(cluster_one.loc[cluster_one["oof_fold_id"].astype(str) == "", "pulse_id"]),
            )


if __name__ == "__main__":
    unittest.main()
