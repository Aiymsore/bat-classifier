"""Cluster 四类 representative × 已确认物种模型 的对照统计（⑥ 页面核心新增）。

语义（务必保持）：
- Cluster ≠ 物种：多个 Cluster 可以被模型判断为同一物种，这是合法且预期的；
- Center / Boundary / Random / Outlier 四类**分别**统计（科研意义不同），
  同一 pulse 命中多种策略时在各策略统计中分别出现，但综合统计必须去重；
- agreement_rate = 预测为主导标签的样本数 / 该组有效模型预测样本数，
  绝不用 confidence 代替；
- average_calibrated_probability 对**组内全部样本**取 P(主导标签) 后平均
  （包括预测成其他标签的样本），不能只平均“预测正确”的样本；
- 模型结果只写 model_inferred_label / predicted_label，
  绝不写回 expert_label；
- 未校准模型不得输出“校准概率”，只输出模型支持度（model support）。
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..config import (
    MODEL_COMPARE_HIGH_BOUNDARY_AGREEMENT,
    MODEL_COMPARE_HIGH_CENTER_AGREEMENT,
    MODEL_COMPARE_HIGH_OVERALL_AGREEMENT,
    MODEL_COMPARE_HIGH_OVERALL_PROBABILITY,
    MODEL_COMPARE_MIN_GROUP_SAMPLES,
    MODEL_COMPARE_MIN_OVERALL_SAMPLES,
    MODEL_COMPARE_MIXED_OVERALL_AGREEMENT,
    MODEL_OVERLAP_FLAG,
    MODEL_STATUS_HIGH,
    MODEL_STATUS_INSUFFICIENT,
    MODEL_STATUS_LOW,
    MODEL_STATUS_MIXED,
    MODEL_STATUS_MODERATE,
    MODEL_STATUS_NO_MODEL,
    SPECIES_UNCERTAIN_LABEL,
    SPECIES_UNCERTAIN_THRESHOLD,
)
from ..species_model import (
    CALIBRATED_PROBABILITY,
    MODEL_SUPPORT,
    predict_pulse_frame,
)
from .pipeline import atomic_write_csv, read_table

if TYPE_CHECKING:  # pragma: no cover
    from .pipeline import CurationRun

REASON_ORDER = ("center", "boundary", "random", "outlier")
REASON_LABELS = {
    "center": "Center",
    "boundary": "Boundary",
    "random": "Random",
    "outlier": "Outlier",
}

MODEL_PREDICTIONS_CSV = "representative_model_predictions.csv"
MODEL_COMPARISON_CSV = "cluster_model_comparison.csv"

PREDICTION_COLUMNS = [
    "cluster_id",
    "pulse_id",
    "selection_reason",
    "source_file",
    "pulse_file",
    "spectrogram_file",
    "model_inferred_label",
    "predicted_label",
    "calibrated_probability",
    "raw_probability",
    "probability_kind",
    "all_class_probabilities",
    "is_uncertain",
]


def split_reasons(selection_reason: object) -> list[str]:
    """selection_reason（'center|boundary'）→ 命中的策略列表。"""
    text = str(selection_reason or "")
    parts = {part.strip() for part in text.split("|") if part.strip()}
    return [reason for reason in REASON_ORDER if reason in parts]


# ---------------------------------------------------------------------------
# 单组统计
# ---------------------------------------------------------------------------
def _probability_of(
    row: dict,
    label: str,
    key: str = "all_class_probabilities",
) -> float:
    """一行样本对指定标签的概率（从 all_class_probabilities 解析）。"""
    text = str(row.get(key, "") or "")
    if not text:
        calibrated = row.get("calibrated_probability")
        try:
            value = float(calibrated)
            return value if np.isfinite(value) else float("nan")
        except (TypeError, ValueError):
            return float("nan")
    try:
        probabilities = json.loads(text)
    except json.JSONDecodeError:
        return float("nan")
    try:
        return float(probabilities.get(label, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def compute_group_stats(rows: pd.DataFrame, representative_type: str) -> dict:
    """一类 representative 的独立统计（Center / Boundary / Random / Outlier）。"""
    records = rows.to_dict("records") if isinstance(rows, pd.DataFrame) else list(rows)
    valid = [
        record
        for record in records
        if str(record.get("model_inferred_label", "") or "").strip()
    ]
    sample_count = len(valid)
    skipped_count = len(records) - sample_count

    counts: dict[str, int] = {}
    for record in valid:
        label = str(record["model_inferred_label"]).strip()
        counts[label] = counts.get(label, 0) + 1

    concrete = {
        label: count
        for label, count in counts.items()
        if label != SPECIES_UNCERTAIN_LABEL
    }
    if concrete:
        # 并列时取字典序小的标签，保证结果确定可复现
        dominant_label = sorted(concrete.items(), key=lambda item: (-item[1], item[0]))[0][0]
        dominant_count = int(concrete[dominant_label])
    else:
        dominant_label = SPECIES_UNCERTAIN_LABEL
        dominant_count = 0

    agreement_rate = float(dominant_count / sample_count) if sample_count else float("nan")

    probabilities = (
        np.asarray([_probability_of(record, dominant_label) for record in valid], dtype=float)
        if dominant_label != SPECIES_UNCERTAIN_LABEL and sample_count
        else np.asarray([], dtype=float)
    )
    probabilities = probabilities[np.isfinite(probabilities)]

    uncertain_count = sum(
        1 for record in valid if bool(record.get("is_uncertain"))
    )

    def _stat(value: float | None) -> float | None:
        return float(value) if value is not None and np.isfinite(value) else None

    stats = {
        "representative_type": representative_type,
        "sample_count": int(sample_count),
        "skipped_count": int(skipped_count),
        "dominant_label": dominant_label,
        "dominant_count": int(dominant_count),
        "agreement_rate": agreement_rate,
        "uncertain_count": int(uncertain_count),
        "species_distribution": dict(
            sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "probability_count": int(len(probabilities)),
        "average_calibrated_probability": _stat(probabilities.mean()) if len(probabilities) else None,
        "median_calibrated_probability": _stat(np.median(probabilities)) if len(probabilities) else None,
        "min_calibrated_probability": _stat(probabilities.min()) if len(probabilities) else None,
        "max_calibrated_probability": _stat(probabilities.max()) if len(probabilities) else None,
    }
    return stats


def compute_overall_stats(rows: pd.DataFrame) -> dict:
    """综合统计：必须按 pulse_id 去重（同一 pulse 可能被多种策略选中）。"""
    if isinstance(rows, pd.DataFrame):
        frame = rows.copy()
        if "pulse_id" in frame.columns:
            frame = frame.drop_duplicates(subset=["pulse_id"], keep="first")
        records = frame
    else:
        seen: set[str] = set()
        records = []
        for record in rows:
            pulse_id = str(record.get("pulse_id", ""))
            if pulse_id in seen:
                continue
            seen.add(pulse_id)
            records.append(record)
    return compute_group_stats(records, "overall")


def reference_agreement_stats(
    rows,
    reference_label: str,
    representative_type: str,
    label_column: str = "model_inferred_label",
    probability_json_column: str = "all_class_probabilities",
) -> dict:
    """与 compute_group_stats 同构的统计，但 agreement / 概率都相对 reference_label。

    共享逻辑（供 OOF cluster consistency 复用，**绝不复制另一套阈值**——
    阈值只在 determine_cluster_status / MODEL_COMPARE_* 一处）：

    - ``agreement_rate`` = 预测 == reference_label 的比例（专家主导标签口径）；
    - ``dominant_label`` 仍取组内**预测**主导（供 determine_cluster_status 的
      MIXED 判定：center 与 boundary 的预测主导不同即混杂）；
    - 平均概率 = 组内全部样本 P(reference_label) 的平均（含预测错的样本）。
    """
    if isinstance(rows, pd.DataFrame):
        records = rows.to_dict("records")
    else:
        records = list(rows)
    reference = str(reference_label or "").strip()
    valid = [
        record
        for record in records
        if str(record.get(label_column, "") or "").strip()
    ]
    sample_count = len(valid)
    skipped_count = len(records) - sample_count

    counts: dict[str, int] = {}
    for record in valid:
        label = str(record[label_column]).strip()
        counts[label] = counts.get(label, 0) + 1
    concrete = {
        label: count
        for label, count in counts.items()
        if label != SPECIES_UNCERTAIN_LABEL
    }
    if concrete:
        dominant_label = sorted(
            concrete.items(), key=lambda item: (-item[1], item[0])
        )[0][0]
        dominant_count = int(concrete[dominant_label])
    else:
        dominant_label = SPECIES_UNCERTAIN_LABEL
        dominant_count = 0

    agreement_count = sum(
        1 for record in valid if str(record[label_column]).strip() == reference
    )
    agreement_rate = (
        float(agreement_count / sample_count) if sample_count else float("nan")
    )

    probabilities = (
        np.asarray(
            [
                _probability_of(record, reference, key=probability_json_column)
                for record in valid
            ],
            dtype=float,
        )
        if reference and sample_count
        else np.asarray([], dtype=float)
    )
    probabilities = probabilities[np.isfinite(probabilities)]
    uncertain_count = sum(
        1 for record in valid if bool(record.get("is_uncertain"))
    )

    def _stat(value: float | None) -> float | None:
        return float(value) if value is not None and np.isfinite(value) else None

    return {
        "representative_type": representative_type,
        "sample_count": int(sample_count),
        "skipped_count": int(skipped_count),
        "dominant_label": dominant_label,
        "dominant_count": int(dominant_count),
        "agreement_rate": agreement_rate,
        "reference_label": reference,
        "agreement_count": int(agreement_count),
        "uncertain_count": int(uncertain_count),
        "species_distribution": dict(
            sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "probability_count": int(len(probabilities)),
        "average_calibrated_probability": _stat(probabilities.mean())
        if len(probabilities)
        else None,
        "median_calibrated_probability": _stat(np.median(probabilities))
        if len(probabilities)
        else None,
        "min_calibrated_probability": _stat(probabilities.min())
        if len(probabilities)
        else None,
        "max_calibrated_probability": _stat(probabilities.max())
        if len(probabilities)
        else None,
    }


# ---------------------------------------------------------------------------
# Cluster 状态判定（阈值全部来自 config，UI 不得硬编码）
# ---------------------------------------------------------------------------
def determine_cluster_status(
    center_stats: dict,
    boundary_stats: dict,
    overall_stats: dict,
    has_model: bool = True,
) -> str:
    if not has_model:
        return MODEL_STATUS_NO_MODEL

    center_valid = int(center_stats.get("sample_count", 0))
    boundary_valid = int(boundary_stats.get("sample_count", 0))
    overall_valid = int(overall_stats.get("sample_count", 0))
    if (
        center_valid < MODEL_COMPARE_MIN_GROUP_SAMPLES
        or boundary_valid < MODEL_COMPARE_MIN_GROUP_SAMPLES
        or overall_valid < MODEL_COMPARE_MIN_OVERALL_SAMPLES
    ):
        return MODEL_STATUS_INSUFFICIENT

    center_label = str(center_stats.get("dominant_label", ""))
    boundary_label = str(boundary_stats.get("dominant_label", ""))
    if (
        center_label
        and boundary_label
        and center_label != SPECIES_UNCERTAIN_LABEL
        and boundary_label != SPECIES_UNCERTAIN_LABEL
        and center_label != boundary_label
    ):
        return MODEL_STATUS_MIXED

    center_agreement = float(center_stats.get("agreement_rate") or 0.0)
    boundary_agreement = float(boundary_stats.get("agreement_rate") or 0.0)
    overall_agreement = float(overall_stats.get("agreement_rate") or 0.0)
    overall_probability = float(
        overall_stats.get("average_calibrated_probability")
        if overall_stats.get("average_calibrated_probability") is not None
        else overall_stats.get("average_model_support") or 0.0
    )

    if (
        center_agreement >= MODEL_COMPARE_HIGH_CENTER_AGREEMENT
        and boundary_agreement >= MODEL_COMPARE_HIGH_BOUNDARY_AGREEMENT
        and overall_agreement >= MODEL_COMPARE_HIGH_OVERALL_AGREEMENT
        and overall_probability >= MODEL_COMPARE_HIGH_OVERALL_PROBABILITY
    ):
        return MODEL_STATUS_HIGH
    if overall_agreement >= MODEL_COMPARE_MIXED_OVERALL_AGREEMENT:
        return MODEL_STATUS_MODERATE
    return MODEL_STATUS_LOW


def determine_cluster_oof_consistency(oof_stats: dict) -> str:
    """Cluster-wide OOF 一致性判定（伪标签反哺专用）。

    与 determine_cluster_status 共用 **同一套** MODEL_COMPARE_* 阈值
    （绝不复制另一套阈值），但统计对象不同：

    - **整个可合法 OOF 评分的 Train-side Cluster 脉冲**，而不是只有
      reviewed representatives（修复代表样本偏置：representatives 只提供
      Expert Truth，不再代表整个 Cluster 的模型一致性）；
    - 合法 OOF prediction 定义：评分模型训练时完全没见过该 pulse 所属
      `source_recording`（grouped cross-fit，禁止随机 pulse KFold）；
    - ``oof_stats`` 由 reference_agreement_stats(oof_top1_label vs
      簇 dominant_species) 产出：sample_count = 合法 OOF 预测数，
      agreement_rate = OOF top1 == dominant 的比例。

    注意：这只是**模型一致性证据**，不是专家真值；coverage 门槛
    （PSEUDO_MIN_OOF_CLUSTER_COVERAGE）由调用方另行判定。
    """
    valid = int(oof_stats.get("sample_count", 0))
    if valid < MODEL_COMPARE_MIN_OVERALL_SAMPLES:
        return MODEL_STATUS_INSUFFICIENT
    agreement = oof_stats.get("agreement_rate")
    probability = oof_stats.get("average_calibrated_probability")
    if probability is None:
        probability = oof_stats.get("average_model_support")
    try:
        agreement_value = float(agreement)
        probability_value = float(probability) if probability is not None else 0.0
    except (TypeError, ValueError):
        return MODEL_STATUS_LOW
    if not np.isfinite(agreement_value):
        return MODEL_STATUS_LOW
    if not np.isfinite(probability_value):
        probability_value = 0.0
    if (
        agreement_value >= MODEL_COMPARE_HIGH_OVERALL_AGREEMENT
        and probability_value >= MODEL_COMPARE_HIGH_OVERALL_PROBABILITY
    ):
        return MODEL_STATUS_HIGH
    if agreement_value >= MODEL_COMPARE_MIXED_OVERALL_AGREEMENT:
        return MODEL_STATUS_MODERATE
    return MODEL_STATUS_LOW


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _cluster_sizes(run: "CurationRun") -> dict[str, int]:
    summary = read_table(run.tables_dir / "cluster_summary.csv")
    if not summary.empty and {"cluster_id", "pulse_count"} <= set(summary.columns):
        return {
            str(row["cluster_id"]): int(row["pulse_count"])
            for _, row in summary.iterrows()
        }
    clustered = read_table(
        run.tables_dir / "clustered_pulses.csv", key_column="pulse_id"
    )
    if not clustered.empty and "cluster_id" in clustered.columns:
        return {
            str(key): int(value)
            for key, value in clustered["cluster_id"].astype(str).value_counts().items()
        }
    return {}


def _flatten_stats(stats: dict, prefix: str, calibrated: bool) -> dict:
    row: dict = {
        f"{prefix}_dominant_label": stats.get("dominant_label", ""),
        f"{prefix}_sample_count": stats.get("sample_count", 0),
        f"{prefix}_dominant_count": stats.get("dominant_count", 0),
        f"{prefix}_agreement": stats.get("agreement_rate"),
        f"{prefix}_uncertain_count": stats.get("uncertain_count", 0),
        f"{prefix}_species_distribution": json.dumps(
            stats.get("species_distribution", {}), ensure_ascii=False
        ),
    }
    if calibrated:
        row.update(
            {
                f"{prefix}_avg_calibrated_probability": stats.get(
                    "average_calibrated_probability"
                ),
                f"{prefix}_median_calibrated_probability": stats.get(
                    "median_calibrated_probability"
                ),
                f"{prefix}_min_calibrated_probability": stats.get(
                    "min_calibrated_probability"
                ),
                f"{prefix}_max_calibrated_probability": stats.get(
                    "max_calibrated_probability"
                ),
            }
        )
    else:
        # 未校准模型：只输出模型支持度（model support），绝不冒充校准概率。
        row.update(
            {
                f"{prefix}_avg_calibrated_probability": None,
                f"{prefix}_median_calibrated_probability": None,
                f"{prefix}_min_calibrated_probability": None,
                f"{prefix}_max_calibrated_probability": None,
                f"{prefix}_avg_model_support": stats.get("average_model_support"),
                f"{prefix}_median_model_support": stats.get("median_model_support"),
                f"{prefix}_min_model_support": stats.get("min_model_support"),
                f"{prefix}_max_model_support": stats.get("max_model_support"),
            }
        )
    return row


def _rename_support_fields(stats: dict, calibrated: bool) -> dict:
    """未校准模型：概率值必须挪到 model support 字段，不得冒充校准概率。

    两组字段都保留（未填的一组为 None），方便 UI 与 CSV 统一读取。
    """
    stats = dict(stats)
    value_keys = (
        "average_calibrated_probability",
        "median_calibrated_probability",
        "min_calibrated_probability",
        "max_calibrated_probability",
    )
    support_keys = (
        "average_model_support",
        "median_model_support",
        "min_model_support",
        "max_model_support",
    )
    for value_key, support_key in zip(value_keys, support_keys):
        value = stats.get(value_key)
        if calibrated:
            stats[support_key] = None
        else:
            stats[support_key] = value
            stats[value_key] = None
    return stats


def run_model_comparison(
    run: "CurationRun",
    payload: dict,
    uncertain_threshold: float | None = None,
    model_name: str = "",
) -> dict:
    """对一个 ⑥ run 的全部 representative 运行物种模型对照并落盘两张结果表。"""
    from ..species_model import describe_species_model

    if payload is None:
        return {"ok": False, "error": "尚未加载已确认物种模型。"}
    description = describe_species_model(payload)
    if not description.get("loaded"):
        return {"ok": False, "error": "尚未加载已确认物种模型。"}

    representatives = read_table(
        run.tables_dir / "representatives.csv", key_column="pulse_id"
    )
    if representatives.empty:
        return {"ok": False, "error": "该任务还没有代表样本（请先完成聚类）。"}

    threshold = float(
        uncertain_threshold
        if uncertain_threshold is not None
        else SPECIES_UNCERTAIN_THRESHOLD
    )
    calibrated = bool(payload.get("is_calibrated"))
    probability_kind = CALIBRATED_PROBABILITY if calibrated else MODEL_SUPPORT

    try:
        predictions = predict_pulse_frame(payload, representatives, threshold)
    except ValueError as error:
        return {"ok": False, "error": str(error)}

    prediction_table = predictions.copy()
    for column in PREDICTION_COLUMNS:
        if column not in prediction_table.columns:
            prediction_table[column] = ""
    prediction_table["probability_kind"] = probability_kind
    prediction_table["model_name"] = model_name or str(description.get("name") or "")
    prediction_table["uncertain_threshold"] = float(threshold)
    atomic_write_csv(
        prediction_table[PREDICTION_COLUMNS + ["model_name", "uncertain_threshold"]],
        run.tables_dir / MODEL_PREDICTIONS_CSV,
    )

    # ---- 训练数据重合检测（TRAINING_DATA_OVERLAP） ----
    training_pulses = {
        str(value) for value in (payload.get("training_pulse_ids") or [])
    }
    training_hashes = {
        str(value) for value in (payload.get("training_audio_hashes") or [])
    }
    predicted_pulses = set(predictions["pulse_id"].astype(str)) if "pulse_id" in predictions.columns else set()
    predicted_hashes = (
        set(predictions["audio_hash"].astype(str))
        if "audio_hash" in predictions.columns
        else set()
    )
    overlap = bool(
        (training_pulses & predicted_pulses) or (training_hashes & predicted_hashes)
    )

    sizes = _cluster_sizes(run)
    cluster_ids = sorted(set(predictions["cluster_id"].astype(str)))
    comparison_rows: list[dict] = []
    cluster_results: list[dict] = []

    for cluster_id in cluster_ids:
        cluster_rows = predictions[
            predictions["cluster_id"].astype(str) == cluster_id
        ]
        per_type: dict[str, dict] = {}
        for reason in REASON_ORDER:
            members = cluster_rows[
                cluster_rows["selection_reason"].map(
                    lambda value, r=reason: r in split_reasons(value)
                )
            ]
            per_type[reason] = _rename_support_fields(
                compute_group_stats(members, reason), calibrated
            )
        overall = _rename_support_fields(
            compute_overall_stats(cluster_rows), calibrated
        )

        status = determine_cluster_status(
            per_type["center"], per_type["boundary"], overall, has_model=True
        )
        flags = []
        if overlap:
            flags.append(MODEL_OVERLAP_FLAG)
        if not calibrated:
            flags.append("UNCALIBRATED_MODEL")

        row = {
            "cluster_id": cluster_id,
            "cluster_size": int(sizes.get(cluster_id, 0)),
            "model_status": status,
            "flags": "|".join(flags),
            "model_name": model_name or str(description.get("name") or ""),
            "is_calibrated": bool(calibrated),
            "probability_kind": probability_kind,
            "uncertain_threshold": float(threshold),
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        }
        for reason in REASON_ORDER:
            row.update(_flatten_stats(per_type[reason], reason, calibrated))
        row.update(_flatten_stats(overall, "overall", calibrated))
        comparison_rows.append(row)
        cluster_results.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": int(sizes.get(cluster_id, 0)),
                "model_status": status,
                "flags": flags,
                "groups": per_type,
                "overall": overall,
                "probability_kind": probability_kind,
                "is_calibrated": bool(calibrated),
                "uncertain_threshold": float(threshold),
            }
        )

    comparison_table = pd.DataFrame(comparison_rows)
    atomic_write_csv(comparison_table, run.tables_dir / MODEL_COMPARISON_CSV)

    dominant_species = [
        {
            "cluster_id": result["cluster_id"],
            "species": result["overall"].get("dominant_label", ""),
            "model_status": result["model_status"],
        }
        for result in cluster_results
    ]
    print(
        f"[模型对照] 完成：{len(cluster_results)} 个 Cluster × "
        f"{len(predictions)} 个代表样本（{probability_kind}）。"
    )
    if not calibrated:
        print(f"[模型对照] 未校准模型：概率列为模型支持度，不显示校准概率。")
    if overlap:
        print(f"[模型对照] {MODEL_OVERLAP_FLAG}：部分代表样本与模型训练数据重合，结果不代表泛化能力。")

    return {
        "ok": True,
        "clusters": cluster_results,
        "predictions": prediction_table,
        "dominant_species": dominant_species,
        "training_data_overlap": overlap,
        "probability_kind": probability_kind,
        "is_calibrated": bool(calibrated),
        "uncertain_threshold": float(threshold),
        "model_name": model_name or str(description.get("name") or ""),
        "predictions_csv": str(run.tables_dir / MODEL_PREDICTIONS_CSV),
        "comparison_csv": str(run.tables_dir / MODEL_COMPARISON_CSV),
    }
