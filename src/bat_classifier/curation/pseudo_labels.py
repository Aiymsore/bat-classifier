"""伪标签反哺（⑥）：Cluster 核心区 → 物种模型自训练。

完整决策链（docs/pseudo_label_feedback.md §3；**Expert Evaluation Set
一条都不参与**——reviewed 计数 / 纯度 / core radius / 局部邻域 /
propagated_from / OOF 训练全部只用 Expert Train Pool）：

```text
Expert Train Pool 专家审核
  → reviewed >= PSEUDO_MIN_REVIEWED(10)
  → dominant_fraction >= 0.90
  → OOF Cluster Consistency == HIGH_CONSISTENCY（OOF 预测 vs 专家主导标签）
  → candidate 位于 expert-defined core radius
  → 局部 Expert Train 邻域无明显冲突
  → OOF top1 == cluster dominant species
  → OOF calibrated P(top1) >= threshold
  → top1 - top2 >= probability margin
  → Safe Candidate Pool
  → diversity sampling（时间桶去冗余 + 录音轮转/限额 + 距离分层）
  → pseudo_training_dataset.csv
```

Production 模型**完全不参与**伪标签准入（I9/I10/I11）；OOF 不可用时
零传播并记录原因，绝不回退 Production 概率。
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..config import (
    MODEL_STATUS_HIGH,
    MODEL_STATUS_INSUFFICIENT_OOF_COVERAGE,
    MODEL_STATUS_LOW,
    MODEL_STATUS_MIXED,
    MODEL_STATUS_MODERATE,
    MODEL_STATUS_NO_MODEL,
    PSEUDO_CLASS_THRESHOLDS,
    PSEUDO_CORE_DISTANCE_QUANTILE,
    PSEUDO_CORE_MIN_RADIUS_SAMPLES,
    PSEUDO_DISTANCE_STRATA,
    PSEUDO_LOCAL_EXPERT_K,
    PSEUDO_LOCAL_EXPERT_MIN_NEIGHBORS,
    PSEUDO_LOCAL_EXPERT_RADIUS_FACTOR,
    PSEUDO_MAX_PER_CLUSTER,
    PSEUDO_MAX_PER_CLUSTER_FACTOR,
    PSEUDO_MAX_PER_RECORDING_PER_CLASS,
    PSEUDO_MAX_PER_TIME_BUCKET,
    PSEUDO_MIN_DOMINANT_FRACTION,
    PSEUDO_MIN_LOCAL_EXPERT_AGREEMENT,
    PSEUDO_MIN_MODEL_PROBABILITY,
    PSEUDO_MIN_MODEL_PROBABILITY_UNCALIBRATED,
    PSEUDO_MIN_OOF_CLUSTER_COVERAGE,
    PSEUDO_MIN_PROBABILITY_MARGIN,
    PSEUDO_MIN_REVIEWED,
    PSEUDO_TIME_BUCKET_S,
    SPECIES_FEATURE_COLUMNS,
    SPECIES_UNCERTAIN_LABEL,
)
from ..expert_split import apply_expert_split, assert_eval_isolation
from ..oof_scoring import OOFScoring
from ..species_model import _clean_label, source_recording_series
from .annotations import STATUS_NOISE, STATUS_REVIEWED, STATUS_UNCERTAIN, AnnotationStore
from .model_comparison import (
    determine_cluster_oof_consistency,
    reference_agreement_stats,
)
from .pipeline import atomic_write_csv, read_table

GATES_CSV = "pseudo_cluster_gates.csv"
CANDIDATES_CSV = "pseudo_label_candidates.csv"
PSEUDO_DATASET_CSV = "pseudo_training_dataset.csv"

# ---- rejection_reason 全集（按判定顺序，每条候选记首个命中原因） ----
REJECT_INSUFFICIENT_REVIEWED = "insufficient_reviewed"
REJECT_DOMINANT_FRACTION = "dominant_fraction_too_low"
REJECT_OOF_INSUFFICIENT = "insufficient_oof_training_data"
REJECT_INSUFFICIENT_COVERAGE = "insufficient_oof_coverage"
REJECT_NOT_HIGH = "not_high_consistency"
REJECT_EXPERT = "expert_sample"
REJECT_UNCERTAIN = "uncertain_sample"
REJECT_NOISE = "noise_sample"
REJECT_BOUNDARY = "boundary_or_outlier_sample"
REJECT_OUTSIDE_CORE = "outside_core_radius"
REJECT_OOF_UNAVAILABLE = "oof_probability_unavailable"
REJECT_TOP1_MISMATCH = "model_top1_mismatch"
REJECT_LOW_PROBABILITY = "low_model_probability"
REJECT_LOW_MARGIN = "low_probability_margin"
REJECT_LOCAL_CONFLICT = "local_expert_conflict"
REJECT_TEMPORAL = "temporal_diversity_filtered"
REJECT_RECORDING_QUOTA = "recording_quota_full"
REJECT_STRATUM_QUOTA = "distance_stratum_quota_full"
REJECT_CLUSTER_QUOTA = "cluster_quota_full"

ACTION_AUTO = "auto_pseudo"
ACTION_SUGGEST_REVIEW = "suggest_review"
ACTION_MANUAL_REVIEW = "manual_review"
ACTION_NO_MODEL = "no_oof_scoring"

LOCAL_CHECK_PASS = "pass"
LOCAL_CHECK_CONFLICT = "conflict"
LOCAL_CHECK_INSUFFICIENT = "insufficient"

DATASET_COLUMNS = [
    "pulse_id",
    "cluster_id",
    "pseudo_label",
    "label_source",
    "expert_label",
    "review_status",
    "audio_hash",
    "source_file",
    "source_path",
    "source_recording",
    "start_s",
    "end_s",
    "duration_ms",
    "snr_db",
    "distance_to_center",
    "core_radius",
    "normalized_core_distance",
    "cluster_purity",
    "local_expert_check",
    "local_expert_agreement",
    "propagated_from",
    "scoring_model",
    "scoring_model_version",
    "oof_fold_id",
    "oof_train_recordings_hash",
    "model_probability",
    "probability_kind",
    "top1_label",
    "top1_probability",
    "top2_probability",
    "probability_margin",
    "applied_threshold",
    "pseudo_round",
    "selected_at",
    *SPECIES_FEATURE_COLUMNS,
]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _float(value) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


class _PseudoFeedbackRun:
    """一次伪标签筛选的内部状态（集群门控 + 候选判定 + 多样性抽样）。"""

    def __init__(self, run, random_state: int | None) -> None:
        self.run = run
        self.random_state = random_state
        self.notes: list[str] = []
        self.candidate_rows: list[dict] = []
        self.gate_rows: list[dict] = []

    # -------------------------------------------------------------- 数据装载
    def load(self) -> None:
        tables = self.run.tables_dir
        self.clustered = read_table(
            tables / "clustered_pulses.csv", key_column="pulse_id"
        )
        if self.clustered.empty:
            raise ValueError("该任务还没有聚类结果（请先完成聚类）。")
        if "distance_to_center" not in self.clustered.columns:
            raise ValueError(
                "聚类结果缺少 distance_to_center 列（请重新运行聚类以生成"
                "全量距离与 cluster_model.joblib）。"
            )
        self.clustered["cluster_id"] = self.clustered["cluster_id"].map(_clean_label)
        self.clustered["distance_to_center"] = pd.to_numeric(
            self.clustered["distance_to_center"], errors="coerce"
        )
        model_path = tables / "cluster_model.joblib"
        if not model_path.exists():
            raise ValueError(
                "缺少 cluster_model.joblib（请重新运行聚类）。"
            )
        bundle = joblib.load(model_path)
        self.scaler = bundle["scaler"]
        self.feature_columns = list(
            bundle.get("feature_columns") or SPECIES_FEATURE_COLUMNS
        )
        scaled = self.scaler.transform(
            self.clustered[self.feature_columns].astype(float)
        )
        self.clustered["_scaled"] = list(np.asarray(scaled, dtype=float))

        # ---- 专家标注（只有 Train Pool 参与任何传播依据） ----
        store = AnnotationStore.load(self.run.run_dir)
        annotations = store.to_frame()
        if annotations.empty:
            raise ValueError("还没有任何专家标注，无法生成伪标签。")
        annotations["pulse_id"] = annotations["pulse_id"].map(_clean_label)
        self.annotations = annotations
        reviewed = annotations[
            (annotations["review_status"].map(_clean_label) == STATUS_REVIEWED)
            & (annotations["expert_label"].map(_clean_label) != "")
        ].copy()
        reviewed["expert_label"] = reviewed["expert_label"].map(_clean_label)
        # 特征 / 距离 / 录音键并入（传播依据必须与聚类口径一致）
        keep_columns = [
            "pulse_id",
            "cluster_id",
            "distance_to_center",
            "source_file",
            "_scaled",
            *self.feature_columns,
        ]
        if "start_s" in self.clustered.columns:
            keep_columns.append("start_s")
        keep = self.clustered[list(dict.fromkeys(keep_columns))].copy()
        reviewed = reviewed.drop(
            columns=[c for c in ("cluster_id", "source_file") if c in reviewed.columns]
        ).merge(keep, on="pulse_id", how="inner")
        reviewed["source_recording"] = source_recording_series(reviewed).map(_clean_label)

        from ..expert_split import ExpertSplitStore

        self.split = ExpertSplitStore.load(self.run.run_dir).record
        if not self.split:
            raise ValueError(
                "尚未生成专家数据划分：请先在 ⑥「专家数据划分」生成并冻结"
                " Train Pool / Evaluation Set（伪标签传播依据只允许来自 Train Pool）。"
            )
        parts = apply_expert_split(reviewed, self.split)
        self.expert_train = parts["train"]
        self.expert_eval = parts["evaluation"]
        self.notes.append(
            f"传播依据只用 Expert Train Pool：{len(self.expert_train)} 条已审样本；"
            f"冻结 Evaluation Set {len(self.expert_eval)} 条不参与任何传播依据。"
        )
        self.eval_recordings = {
            _clean_label(value)
            for value in (self.split.get("evaluation_recordings") or [])
        }

        # ---- 候选宇宙：全部合格聚类脉冲（含已审样本，逐条给结论） ----
        self.annotated_ids = set(self.annotations["pulse_id"])
        self.annotated_status = {
            str(row["pulse_id"]): _clean_label(row["review_status"])
            for _, row in self.annotations.iterrows()
        }
        self.annotated_label = {
            str(row["pulse_id"]): _clean_label(row["expert_label"])
            for _, row in self.annotations.iterrows()
        }
        self.reason_by_pulse = {
            str(row["pulse_id"]): _clean_label(row.get("selection_reason", ""))
            for _, row in self.annotations.iterrows()
        }

    # ---------------------------------------------------------- OOF 评分体系
    def build_oof(self) -> OOFScoring:
        # §十：OOF 训练录音 ∩ Evaluation 录音必须为空（不自动修，直接终止）
        assert_eval_isolation(
            self.eval_recordings,
            source_recording_series(self.expert_train),
            "OOF scoring 训练集（伪标签反哺）",
        )
        self.oof = OOFScoring.build(
            self.expert_train,
            feature_columns=self.feature_columns,
            random_state=self.random_state,
        )
        for note in self.oof.notes:
            self.notes.append(note)
        if not self.oof.available:
            self.notes.append(
                "OOF 评分不可用：伪标签零传播（绝不回退 Production 模型概率）。"
            )
        return self.oof

    # ---------------------------------------------------------- OOF 全量评分
    def score_universe(self) -> None:
        """对 Train-side 簇脉冲**全量**打 OOF 分（Legal OOF Prediction）。

        合法 OOF prediction 定义（§十六）：评分模型训练时完全没见过该
        pulse 所属 `source_recording`（source_recording-level grouped
        cross-fitting，**禁止随机 pulse KFold**）；不同录音由不同 fold
        模型评分（§十七），汇总为整个 Cluster 的一致性证据。
        Evaluation 录音的脉冲**不进入**打分宇宙（I13/§九）。
        """
        universe = self.clustered.copy()
        universe["source_recording"] = source_recording_series(
            universe
        ).map(_clean_label)
        universe = universe[
            ~universe["source_recording"].isin(self.eval_recordings)
        ].reset_index(drop=True)
        self.scored = (
            self.oof.predict(universe) if self.oof.available else universe
        )

    # -------------------------------------------------------------- 簇级门控
    def evaluate_clusters(self) -> None:
        """**第一层：Cluster-level OOF consistency**（§二十二）。

        回答"这个 Cluster 整体是否与专家 dominant species 一致"，
        与第二层（candidate-level OOF scoring）严格分开，绝不混为
        一个 probability。

        一致性覆盖**整个可合法 OOF 评分的 Train-side Cluster**（§十五），
        不再只根据 reviewed representatives 计算（修复代表样本偏置：
        representatives 只提供 Expert Truth）。dominant species / purity
        仍只由 Expert Train 决定（§十四）。
        """
        expert_train = self.expert_train.copy()
        expert_train["cluster_id"] = expert_train["cluster_id"].map(_clean_label)

        self.cluster_info: dict[str, dict] = {}
        for cluster_id in sorted(set(self.scored["cluster_id"].map(_clean_label))):
            cluster_scored = self.scored[
                self.scored["cluster_id"].map(_clean_label) == cluster_id
            ]
            cluster_expert = expert_train[expert_train["cluster_id"] == cluster_id]

            # ---- Expert Truth：dominant species 只由 Expert Train 决定 ----
            reviewed_count = int(len(cluster_expert))
            dominant_label = ""
            dominant_fraction = 0.0
            if reviewed_count:
                counts = cluster_expert["expert_label"].value_counts()
                dominant_label = str(counts.index[0])
                dominant_fraction = float(counts.iloc[0] / reviewed_count)

            # ---- Cluster-wide OOF consistency + coverage（§十五–§二十一） ----
            total_train_side = int(len(cluster_scored))
            oof_available = cluster_scored.get(
                "oof_available", pd.Series(False, index=cluster_scored.index)
            ).astype(bool)
            top1_all = cluster_scored.get(
                "oof_top1_label", pd.Series("", index=cluster_scored.index)
            ).map(_clean_label)
            eligible_rows = cluster_scored[oof_available & (top1_all != "")]
            eligible_count = int(len(eligible_rows))
            coverage = (
                float(eligible_count / total_train_side) if total_train_side else 0.0
            )

            if dominant_label:
                stats = reference_agreement_stats(
                    eligible_rows,
                    dominant_label,
                    "cluster_wide",
                    label_column="oof_top1_label",
                    probability_json_column="oof_all_class_probabilities",
                )
            else:
                stats = {
                    "sample_count": eligible_count,
                    "agreement_count": 0,
                    "agreement_rate": float("nan"),
                    "average_calibrated_probability": None,
                }
            agree_count = int(stats.get("agreement_count", 0))
            disagree_count = int(stats.get("sample_count", 0)) - agree_count
            agreement_rate = stats.get("agreement_rate", float("nan"))

            if not self.oof.available:
                status = MODEL_STATUS_NO_MODEL
            elif coverage < float(PSEUDO_MIN_OOF_CLUSTER_COVERAGE):
                # 覆盖不足：40/40 全对也不许判 HIGH（§十九/§二十）
                status = MODEL_STATUS_INSUFFICIENT_OOF_COVERAGE
            else:
                status = determine_cluster_oof_consistency(stats)

            passed = bool(
                reviewed_count >= int(PSEUDO_MIN_REVIEWED)
                and dominant_fraction >= float(PSEUDO_MIN_DOMINANT_FRACTION)
                and dominant_label not in ("", SPECIES_UNCERTAIN_LABEL)
                and status == MODEL_STATUS_HIGH
            )

            if passed:
                gate_reason, action = "", ACTION_AUTO
            elif reviewed_count < int(PSEUDO_MIN_REVIEWED):
                gate_reason, action = REJECT_INSUFFICIENT_REVIEWED, ACTION_SUGGEST_REVIEW
            elif (
                dominant_fraction < float(PSEUDO_MIN_DOMINANT_FRACTION)
                or dominant_label in ("", SPECIES_UNCERTAIN_LABEL)
            ):
                gate_reason, action = REJECT_DOMINANT_FRACTION, ACTION_SUGGEST_REVIEW
            elif not self.oof.available:
                gate_reason, action = REJECT_OOF_INSUFFICIENT, ACTION_NO_MODEL
            elif status == MODEL_STATUS_INSUFFICIENT_OOF_COVERAGE:
                gate_reason, action = REJECT_INSUFFICIENT_COVERAGE, ACTION_MANUAL_REVIEW
            elif status == MODEL_STATUS_MODERATE:
                gate_reason, action = REJECT_NOT_HIGH, ACTION_SUGGEST_REVIEW
            elif status in (MODEL_STATUS_LOW, MODEL_STATUS_MIXED):
                gate_reason, action = REJECT_NOT_HIGH, ACTION_MANUAL_REVIEW
            else:
                gate_reason, action = REJECT_NOT_HIGH, ACTION_SUGGEST_REVIEW

            # ---- core radius（只用 Train Pool 一致证据，宁窄勿宽） ----
            members = cluster_scored  # Train-side：Evaluation 不进 core radius
            radius = float("nan")
            radius_fallback = False
            if dominant_label:
                consistent = cluster_expert[
                    cluster_expert["expert_label"] == dominant_label
                ]
                reasons = consistent.get(
                    "selection_reason", pd.Series("", index=consistent.index)
                ).map(_clean_label)
                core_rows = consistent[
                    ~reasons.str.contains("boundary", na=False)
                    & ~reasons.str.contains("outlier", na=False)
                ]
                core_distances = pd.to_numeric(
                    core_rows["distance_to_center"], errors="coerce"
                ).dropna()
                if (
                    len(core_rows) >= int(PSEUDO_CORE_MIN_RADIUS_SAMPLES)
                    and len(core_distances)
                ):
                    radius = float(
                        np.quantile(
                            core_distances,
                            float(PSEUDO_CORE_DISTANCE_QUANTILE),
                        )
                    )
                else:
                    radius_fallback = True
                    cluster_distances = pd.to_numeric(
                        members["distance_to_center"], errors="coerce"
                    ).dropna()
                    if len(cluster_distances):
                        radius = float(
                            np.quantile(cluster_distances, 0.25)
                        )

            propagated_from = ";".join(
                sorted(
                    cluster_expert.loc[
                        cluster_expert["expert_label"] == dominant_label, "pulse_id"
                    ].astype(str)
                )
            ) if dominant_label else ""

            self.cluster_info[cluster_id] = {
                "reviewed_count": reviewed_count,
                "dominant_label": dominant_label,
                "dominant_fraction": dominant_fraction,
                "core_radius": radius,
                "radius_fallback": radius_fallback,
                "cluster_total_train_side_pulses": total_train_side,
                "cluster_oof_eligible_pulses": eligible_count,
                "cluster_oof_coverage": round(coverage, 4),
                "cluster_oof_agree_count": agree_count,
                "cluster_oof_disagree_count": disagree_count,
                "cluster_oof_agreement": (
                    round(float(agreement_rate), 4)
                    if agreement_rate is not None
                    and np.isfinite(float(agreement_rate))
                    else None
                ),
                "cluster_oof_status": status,
                "oof_status": status,
                "passed": passed,
                "gate_reason": gate_reason,
                "action": action,
                "propagated_from": propagated_from,
                "member_count": int(len(members)),
            }
            self.gate_rows.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_size": int(
                        len(self.clustered[self.clustered["cluster_id"] == cluster_id])
                    ),
                    "reviewed_count_train": reviewed_count,
                    "dominant_label": dominant_label,
                    "cluster_purity": round(dominant_fraction, 4),
                    "cluster_total_train_side_pulses": total_train_side,
                    "cluster_oof_eligible_pulses": eligible_count,
                    "cluster_oof_coverage": round(coverage, 4),
                    "cluster_oof_agree_count": agree_count,
                    "cluster_oof_disagree_count": disagree_count,
                    "cluster_oof_agreement": (
                        round(float(agreement_rate), 4)
                        if agreement_rate is not None
                        and np.isfinite(float(agreement_rate))
                        else None
                    ),
                    "cluster_oof_status": status,
                    "core_radius": radius,
                    "radius_fallback": radius_fallback,
                    "passed": int(passed),
                    "gate_reason": gate_reason,
                    "recommended_action": action,
                }
            )

    # -------------------------------------------------------------- 候选判定
    def evaluate_candidates(self) -> pd.DataFrame:
        """**第二层：Candidate-level OOF scoring**（§二十二）。

        回答"这一条具体 pulse 是否值得生成 pseudo label"；
        与第一层（Cluster-level OOF consistency）严格分开。
        Evaluation 录音的脉冲不在打分宇宙中（score_universe 已排除）。
        """
        scored = self.scored

        rows: list[dict] = []
        safe_indices: list[int] = []
        for index, row in scored.iterrows():
            pulse_id = str(row["pulse_id"])
            cluster_id = str(row["cluster_id"])
            info = self.cluster_info.get(cluster_id, {})
            record = {
                "pulse_id": pulse_id,
                "cluster_id": cluster_id,
                "source_file": _clean_label(row.get("source_file", "")),
                "source_recording": _clean_label(row.get("source_recording", "")),
                "start_s": row.get("start_s", ""),
                "distance_to_center": row.get("distance_to_center", ""),
                "core_radius": info.get("core_radius", ""),
                "normalized_core_distance": "",
                "cluster_purity": info.get("dominant_fraction", ""),
                "selection_reason": self.reason_by_pulse.get(pulse_id, ""),
                "local_expert_check": "",
                "local_expert_agreement": "",
                "local_expert_neighbor_count": "",
                "applied_threshold": "",
                "oof_top1_label": row.get("oof_top1_label", ""),
                "oof_top1_probability": row.get("oof_top1_probability", ""),
                "oof_probability_margin": row.get("oof_probability_margin", ""),
                "oof_fold_id": row.get("oof_fold_id", ""),
                "oof_scoring_model_version": row.get("oof_scoring_model_version", ""),
                "rejection_reason": "",
                "selected": 0,
            }
            radius = info.get("core_radius")
            distance = _float(row.get("distance_to_center", np.nan))
            if isinstance(radius, (int, float)) and np.isfinite(radius) and radius > 0:
                record["normalized_core_distance"] = float(distance / radius)

            reason = self._candidate_reject(row, pulse_id, info, record, distance)
            if reason:
                record["rejection_reason"] = reason
                rows.append(record)
                continue
            record["rejection_reason"] = ""
            rows.append(record)
            safe_indices.append(index)

        self.candidate_rows = rows
        return scored.loc[safe_indices].copy()

    def _candidate_reject(
        self,
        row: pd.Series,
        pulse_id: str,
        info: dict,
        record: dict,
        distance: float,
    ) -> str:
        # 1) 簇门控
        if not info.get("passed"):
            return str(info.get("gate_reason") or REJECT_NOT_HIGH)
        # 2) 专家样本（任何状态都不许伪标签覆盖）
        status = self.annotated_status.get(pulse_id, "")
        if status == STATUS_NOISE:
            return REJECT_NOISE
        if status == STATUS_UNCERTAIN:
            return REJECT_UNCERTAIN
        if pulse_id in self.annotated_ids:
            return REJECT_EXPERT
        # 3) boundary / outlier 代表样本
        selection_reason = self.reason_by_pulse.get(pulse_id, "")
        if "boundary" in selection_reason or "outlier" in selection_reason:
            return REJECT_BOUNDARY
        # 4) 核心区半径
        radius = info.get("core_radius")
        if not isinstance(radius, (int, float)) or not np.isfinite(radius):
            return REJECT_OUTSIDE_CORE
        if float(distance) > float(radius):
            return REJECT_OUTSIDE_CORE
        # 5) 局部专家邻域（不足为据时不硬判：insufficient + 继续）
        local = self._local_expert_check(row, info)
        record["local_expert_check"] = local["status"]
        record["local_expert_agreement"] = local["agreement"]
        record["local_expert_neighbor_count"] = local["count"]
        if local["status"] == LOCAL_CHECK_CONFLICT:
            return REJECT_LOCAL_CONFLICT
        # 6) OOF 概率可用（绝不回退 Production 模型）
        if str(row.get("oof_available", "")).lower() not in ("true", "1"):
            unavailable = _clean_label(row.get("oof_unavailable_reason", ""))
            return unavailable or REJECT_OOF_UNAVAILABLE
        # 7) top1 == 簇主导标签
        dominant = str(info.get("dominant_label") or "")
        top1 = _clean_label(row.get("oof_top1_label", ""))
        top1_probability = _float(row.get("oof_top1_probability", np.nan))
        if top1 != dominant:
            return REJECT_TOP1_MISMATCH
        # 8) 概率阈值（per-class 覆盖预留）
        threshold = float(
            PSEUDO_CLASS_THRESHOLDS.get(dominant, float(PSEUDO_MIN_MODEL_PROBABILITY))
        )
        kind = _clean_label(row.get("oof_probability_kind", ""))
        if kind == "model_support":
            threshold = max(
                threshold, float(PSEUDO_MIN_MODEL_PROBABILITY_UNCALIBRATED)
            )
        record["applied_threshold"] = threshold
        if not np.isfinite(top1_probability) or top1_probability < threshold:
            return REJECT_LOW_PROBABILITY
        # 9) top1-top2 margin
        margin = _float(row.get("oof_probability_margin", np.nan))
        if not np.isfinite(margin) or margin < float(PSEUDO_MIN_PROBABILITY_MARGIN):
            return REJECT_LOW_MARGIN
        return ""

    def _local_expert_check(self, row: pd.Series, info: dict) -> dict:
        """局部专家邻域一致性（防大簇内部异质小区域误传播）。

        只用 **同簇 Expert Train Pool 已审样本**（Evaluation Set 一条不用）。
        邻居不足 PSEUDO_LOCAL_EXPERT_MIN_NEIGHBORS → 不硬判：
        ``insufficient`` 记录在案后继续走其余闸门（保守默认，可后续升级 hard gate）。
        """
        dominant = str(info.get("dominant_label") or "")
        cluster_id = str(row.get("cluster_id") or "")
        cluster_expert = self.expert_train[
            self.expert_train["cluster_id"].map(_clean_label) == cluster_id
        ]
        result = {
            "status": LOCAL_CHECK_INSUFFICIENT,
            "agreement": "",
            "count": 0,
        }
        if cluster_expert.empty:
            return result
        target = np.asarray(row.get("_scaled"), dtype=float).reshape(1, -1)
        neighbor_scaled = np.asarray(list(cluster_expert["_scaled"]), dtype=float)
        distances = np.sqrt(
            ((neighbor_scaled - target) ** 2).sum(axis=1)
        )
        radius = float(info.get("core_radius") or 0.0) or 1.0
        limit = float(PSEUDO_LOCAL_EXPERT_RADIUS_FACTOR) * radius
        order = sorted(
            range(len(distances)),
            key=lambda i: (float(distances[i]), str(cluster_expert["pulse_id"].iloc[i])),
        )
        near = [i for i in order if float(distances[i]) <= limit]
        near = near[: int(PSEUDO_LOCAL_EXPERT_K)]
        result["count"] = len(near)
        if len(near) < int(PSEUDO_LOCAL_EXPERT_MIN_NEIGHBORS):
            return result
        labels = [str(cluster_expert["expert_label"].iloc[i]) for i in near]
        agreement = sum(1 for label in labels if label == dominant) / len(labels)
        local_dominant = max(
            sorted(set(labels)), key=lambda label: (labels.count(label), label)
        )
        result["agreement"] = round(float(agreement), 4)
        if local_dominant != dominant or agreement < float(
            PSEUDO_MIN_LOCAL_EXPERT_AGREEMENT
        ):
            result["status"] = LOCAL_CHECK_CONFLICT
        else:
            result["status"] = LOCAL_CHECK_PASS
        return result

    # ---------------------------------------------------------- 多样性抽样
    def diversity_select(self, safe: pd.DataFrame) -> set[str]:
        """两阶段：safe first，diversity second（确定性、无 RNG）。

        时间桶去冗余 + 录音字典序轮转 + 距离三等分层配额（最大余数法，
        余数 S1→S2→S3，宁内勿外）；层满回流补满后其余记 cluster_quota_full。
        """
        if safe.empty:
            return set()
        quota_by_cluster: dict[str, int] = {}
        for cluster_id in set(safe["cluster_id"].map(_clean_label)):
            info = self.cluster_info[cluster_id]
            quota_by_cluster[cluster_id] = int(
                min(
                    int(PSEUDO_MAX_PER_CLUSTER),
                    max(1, info["reviewed_count"] * int(PSEUDO_MAX_PER_CLUSTER_FACTOR)),
                )
            )

        selected: set[str] = set()
        bucket_seen: set[tuple[str, int]] = set()
        recording_counts: dict[str, int] = defaultdict(int)
        self.rejection_overrides: dict[str, str] = {}

        def stratum_order(sub: pd.DataFrame) -> list[pd.Series]:
            by_recording: dict[str, list[pd.Series]] = defaultdict(list)
            for _, row in sub.iterrows():
                by_recording[str(row["source_recording"])].append(row)
            for recording in by_recording:
                by_recording[recording].sort(
                    key=lambda r: (
                        float(r["distance_to_center"])
                        if np.isfinite(float(r["distance_to_center"]))
                        else float("inf"),
                        str(r["pulse_id"]),
                    )
                )
            queues = {recording: rows for recording, rows in sorted(by_recording.items())}
            order: list[pd.Series] = []
            while queues:
                for recording in list(queues):
                    rows = queues[recording]
                    if rows:
                        order.append(rows.pop(0))
                    if not rows:
                        del queues[recording]
            return order

        def try_take(row: pd.Series) -> str:
            """返回 ""=选中；否则返回拒绝原因（记录但不停止）。"""
            pulse_id = str(row["pulse_id"])
            recording = str(row["source_recording"])
            start_s = _float(row.get("start_s", np.nan))
            bucket = int(float(start_s) // float(PSEUDO_TIME_BUCKET_S)) if np.isfinite(start_s) else -1
            key = (recording, bucket)
            if key in bucket_seen:
                return REJECT_TEMPORAL
            if recording_counts[recording] >= int(PSEUDO_MAX_PER_RECORDING_PER_CLASS):
                return REJECT_RECORDING_QUOTA
            bucket_seen.add(key)
            recording_counts[recording] += 1
            selected.add(pulse_id)
            return ""

        for cluster_id in sorted(set(safe["cluster_id"].map(_clean_label))):
            cluster_safe = safe[safe["cluster_id"].map(_clean_label) == cluster_id]
            quota = quota_by_cluster[cluster_id]
            radius = float(self.cluster_info[cluster_id].get("core_radius") or 0.0) or 1.0
            normalized = (
                pd.to_numeric(cluster_safe["distance_to_center"], errors="coerce")
                / radius
            ).fillna(1.0).clip(0.0, 1.0)
            strata = []
            edges = np.linspace(0.0, 1.0, int(PSEUDO_DISTANCE_STRATA) + 1)
            for stratum_index in range(int(PSEUDO_DISTANCE_STRATA)):
                lower, upper = edges[stratum_index], edges[stratum_index + 1]
                mask = (
                    (normalized > lower) & (normalized <= upper)
                    if stratum_index > 0
                    else (normalized >= lower) & (normalized <= upper)
                )
                strata.append(cluster_safe[mask])
            non_empty = [s for s in strata if not s.empty]
            base, remainder = divmod(quota, max(1, len(non_empty)))
            stratum_quota = []
            quota_left = quota
            for stratum in strata:
                if stratum.empty:
                    stratum_quota.append(0)
                    continue
                share = base + (1 if remainder > 0 else 0)
                remainder = max(0, remainder - 1)
                share = min(share, len(stratum), quota_left)
                stratum_quota.append(share)
                quota_left -= share

            picked = 0
            leftovers: list[pd.Series] = []
            for stratum, share in zip(strata, stratum_quota):
                taken_here = 0
                for row in stratum_order(stratum):
                    if taken_here >= share:
                        leftovers.append(row)
                        continue
                    reason = try_take(row)
                    if not reason:
                        taken_here += 1
                        picked += 1
                    else:
                        self.rejection_overrides[str(row["pulse_id"])] = reason
            # 回流补满（S1→S2→S3 顺序，宁内勿外）
            for row in leftovers:
                if picked >= quota:
                    self.rejection_overrides[str(row["pulse_id"])] = REJECT_CLUSTER_QUOTA
                    continue
                reason = try_take(row)
                if not reason:
                    picked += 1
                else:
                    self.rejection_overrides[str(row["pulse_id"])] = reason
        return selected

    # ---------------------------------------------------------------- 落盘
    def write_tables(self, safe: pd.DataFrame, selected: set[str]) -> dict:
        now = _now()
        dataset_rows: list[dict] = []
        for record in self.candidate_rows:
            pulse_id = str(record["pulse_id"])
            if pulse_id not in set(safe["pulse_id"].map(str)):
                continue
            if pulse_id in selected:
                record["selected"] = 1
                record["rejection_reason"] = ""
            else:
                record["selected"] = 0
                record["rejection_reason"] = self.rejection_overrides.get(
                    pulse_id, REJECT_CLUSTER_QUOTA
                )

        candidates_frame = pd.DataFrame(self.candidate_rows)
        gates_frame = pd.DataFrame(self.gate_rows)

        by_pulse = {str(row["pulse_id"]): row for _, row in safe.iterrows()}
        info_by_cluster = self.cluster_info
        for pulse_id in sorted(selected):
            row = by_pulse[pulse_id]
            cluster_id = _clean_label(row["cluster_id"])
            info = info_by_cluster[cluster_id]
            candidate_record = next(
                (
                    record
                    for record in self.candidate_rows
                    if str(record["pulse_id"]) == pulse_id
                ),
                {},
            )
            dataset_rows.append(
                {
                    "pulse_id": pulse_id,
                    "cluster_id": cluster_id,
                    "pseudo_label": info["dominant_label"],
                    "label_source": "pseudo",
                    "expert_label": "",
                    "review_status": "",
                    "audio_hash": _clean_label(row.get("audio_hash", "")),
                    "source_file": _clean_label(row.get("source_file", "")),
                    "source_path": _clean_label(row.get("source_path", "")),
                    "source_recording": _clean_label(row.get("source_recording", "")),
                    "start_s": row.get("start_s", ""),
                    "end_s": row.get("end_s", ""),
                    "duration_ms": row.get("duration_ms", ""),
                    "snr_db": row.get("snr_db", ""),
                    "distance_to_center": row.get("distance_to_center", ""),
                    "core_radius": info["core_radius"],
                    "normalized_core_distance": candidate_record.get(
                        "normalized_core_distance", ""
                    ),
                    "cluster_purity": info["dominant_fraction"],
                    "local_expert_check": candidate_record.get("local_expert_check", ""),
                    "local_expert_agreement": candidate_record.get(
                        "local_expert_agreement", ""
                    ),
                    "propagated_from": info["propagated_from"],
                    "scoring_model": _clean_label(row.get("oof_fold_id", "")),
                    "scoring_model_version": _clean_label(
                        row.get("oof_scoring_model_version", "")
                    ),
                    "oof_fold_id": _clean_label(row.get("oof_fold_id", "")),
                    "oof_train_recordings_hash": _clean_label(
                        row.get("oof_train_recordings_hash", "")
                    ),
                    "model_probability": row.get("oof_top1_probability", ""),
                    "probability_kind": _clean_label(
                        row.get("oof_probability_kind", "")
                    ),
                    "top1_label": _clean_label(row.get("oof_top1_label", "")),
                    "top1_probability": row.get("oof_top1_probability", ""),
                    "top2_probability": row.get("oof_top2_probability", ""),
                    "probability_margin": row.get("oof_probability_margin", ""),
                    "applied_threshold": candidate_record.get("applied_threshold", ""),
                    "pseudo_round": 1,
                    "selected_at": now,
                    **{
                        column: row.get(column, "")
                        for column in SPECIES_FEATURE_COLUMNS
                    },
                }
            )

        dataset_frame = pd.DataFrame(dataset_rows, columns=DATASET_COLUMNS)
        atomic_write_csv(gates_frame, self.run.tables_dir / GATES_CSV)
        atomic_write_csv(candidates_frame, self.run.tables_dir / CANDIDATES_CSV)
        atomic_write_csv(dataset_frame, self.run.tables_dir / PSEUDO_DATASET_CSV)

        funnel = (
            candidates_frame["rejection_reason"]
            .value_counts()
            .to_dict()
            if not candidates_frame.empty
            else {}
        )
        return {
            "gates_csv": str(self.run.tables_dir / GATES_CSV),
            "candidates_csv": str(self.run.tables_dir / CANDIDATES_CSV),
            "dataset_csv": str(self.run.tables_dir / PSEUDO_DATASET_CSV),
            "candidate_count": int(len(candidates_frame)),
            "selected_count": int(len(dataset_frame)),
            "funnel": {str(key): int(value) for key, value in funnel.items()},
            "cluster_count": int(len(gates_frame)),
            "auto_pseudo_clusters": int(
                sum(1 for _, row in gates_frame.iterrows() if row.get("passed"))
            )
            if not gates_frame.empty
            else 0,
        }


def run_pseudo_feedback(
    run,
    random_state: int | None = None,
) -> dict:
    """执行一轮伪标签筛选，产出 gates / candidates / dataset 三张表。

    两层判断严格分开（§二十二）：
    第一层 Cluster-level OOF consistency（整个可合法 OOF 评分的 Train-side
    Cluster）回答"簇整体是否与专家 dominant 一致"；第二层 Candidate-level
    OOF scoring 回答"这一条 pulse 是否值得生成 pseudo label"。
    不训练模型、不做采纳判断（那是 species_model 带权训练 + Gold Evaluation 的事）。
    """
    worker = _PseudoFeedbackRun(run, random_state)
    worker.load()
    worker.build_oof()
    worker.score_universe()
    worker.evaluate_clusters()
    safe = worker.evaluate_candidates()
    selected = worker.diversity_select(safe) if not safe.empty else set()
    summary = worker.write_tables(safe, selected)

    metadata = {
        "generated_at": _now(),
        "split_version": int((worker.split or {}).get("split_version") or 1),
        "oof": worker.oof.metadata(),
        "thresholds": {
            "min_reviewed": int(PSEUDO_MIN_REVIEWED),
            "min_dominant_fraction": float(PSEUDO_MIN_DOMINANT_FRACTION),
            "min_oof_cluster_coverage": float(PSEUDO_MIN_OOF_CLUSTER_COVERAGE),
            "min_model_probability": float(PSEUDO_MIN_MODEL_PROBABILITY),
            "min_probability_margin": float(PSEUDO_MIN_PROBABILITY_MARGIN),
            "class_thresholds": dict(PSEUDO_CLASS_THRESHOLDS),
        },
        "summary": summary,
        "notes": worker.notes,
    }
    try:
        run.metadata.setdefault("pseudo_feedback", {})
        run.metadata["pseudo_feedback"] = metadata
        run.save_metadata()
    except Exception:  # noqa: BLE001 - metadata 落盘失败不影响表输出
        pass

    for note in worker.notes:
        print(f"[伪标签] {note}")
    print(
        f"[伪标签] 完成：候选 {summary['candidate_count']} 条，"
        f"选中 {summary['selected_count']} 条，"
        f"自动反哺簇 {summary['auto_pseudo_clusters']} 个。"
    )
    return {
        "ok": True,
        **summary,
        "metadata": metadata,
    }
