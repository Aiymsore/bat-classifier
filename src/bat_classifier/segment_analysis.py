"""⑦ 片段识别与叫声统计（上游：WAV 片段或整条超长录音；下游：片段级统计 CSV/JSON）。

业务语义（务必区分）：
- 统计单位是“一次声学 call / pulse”：WAV 片段内部可能包含 N 次独立叫声；
  3 小时以上（未来整夜）的整条长录音也可直接分析（流式 chunk 管线，
  内存与文件总时长无关，见 segment_stream.py）。

两个独立任务（绝不混淆）：
- Task A 分类：这一段是哪种蝙蝠？——复用**已训练好的确定模型**（pulse-level），
  不重新训练、不重新 KMeans、不用 Cluster 代替物种。
- Task B 计数：这一段内部有几次叫声？——复用 detection.py 的
  detect/refine/去重/质量链，保留的独立 pulse 数即叫声数。

因为当前“确定模型”是 pulse-level（一个 pulse → 一个 label），
所以对片段内每个 pulse 分别预测，再按片段聚合：
  每段 → 检测 N 个 pulse → 逐 pulse 分类 → 统计各类别叫声次数 → 片段汇总。

调用链（本模块完全脱离 Qt，可单独测试）：
  analyze_segment_folder
    → 每个 WAV → analyze_segment
        → segment_stream 流式 chunk 管线（chunk+overlap 覆盖全长、跨块去重、
          保守快筛、有限并行、增量落盘 + checkpoint/断点续跑）
          每 chunk：detect_pulses + refine_all_pulses + create_demo_pulse_rows
        → _finalize_chunk_rows（noise/bat 门控 + 逐 pulse 分类 + 不确定标记）
        → 按片段聚合 label_counts / major_label / purity
    → summarize_segment_results（片段汇总 + 总体类别统计）

科研语义：本功能统计的是“声学检测到的 call 数量”，
不代表蝙蝠个体数量。Species A calls = 20 表示模型识别出 20 次
Species A 类型的声学 call，不等于这里有 20 只 Species A 蝙蝠。
速率分母 = 有效监测总时长（valid_monitoring_duration_s）：
静音快筛只是计算优化，被跳过的 chunk 仍算在监测时间内。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .audio_io import find_wav_files
from .calibration import (
    CALIBRATED_PROBABILITY,
    MODEL_SUPPORT,
    describe_calibration,
    select_classifier,
)
from .config import (
    ACOUSTIC_TYPE_LABELS,
    COMBINED_DEMO_MODEL_PATH,
    SEGMENT_ANALYSIS_DIR,
    SEGMENT_CHECKPOINT_EVERY_CHUNKS,
    SEGMENT_FIGURE_SUBDIR,
    SEGMENT_LOW_CONFIDENCE_ACTION,
    SEGMENT_MAX_INMEMORY_ROWS,
    SEGMENT_MAX_PULSE_WAV_EXPORTS,
    SEGMENT_MIN_CONFIDENCE,
    SEGMENT_MIXED_PURITY_THRESHOLD,
    SEGMENT_PULSE_SUBDIR,
    SEGMENT_SEPARATED_SUBDIR,
    SEGMENT_STREAM_CHUNK_SECONDS,
    SEGMENT_STREAM_OVERLAP_SECONDS,
    SEGMENT_UNAVAILABLE_LABEL,
    SEGMENT_UNCERTAIN_LABEL,
    TARGET_SAMPLE_RATE,
)
from .features import numeric_feature_frame
from .noise_gate import apply_noise_gate


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass
class SegmentAnalysisConfig:
    """片段识别与叫声统计的可调参数（UI 可覆盖默认值）。"""

    min_confidence: float = SEGMENT_MIN_CONFIDENCE
    uncertain_label: str = SEGMENT_UNCERTAIN_LABEL
    unavailable_label: str = SEGMENT_UNAVAILABLE_LABEL
    # 低于置信度时："uncertain" 标记为不确定；"keep" 保留模型原标签
    low_confidence_action: str = SEGMENT_LOW_CONFIDENCE_ACTION
    # 是否只统计通过质量筛选（accepted）的有效 pulse
    use_quality_filter: bool = True
    # 是否把每个 pulse 切成 WAV（segments/<id>/<id>__pNNN.wav）
    save_pulse_audio: bool = True
    # 是否生成片段声谱图（figures/<id>.png）
    save_figures: bool = True
    mixed_purity_threshold: float = SEGMENT_MIXED_PURITY_THRESHOLD

    def validate(self) -> "SegmentAnalysisConfig":
        self.min_confidence = float(self.min_confidence)
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError("最低置信度必须在 0~1 之间。")
        if self.low_confidence_action not in {"uncertain", "keep"}:
            raise ValueError("低于置信度处理必须是 uncertain 或 keep。")
        self.use_quality_filter = bool(self.use_quality_filter)
        self.save_pulse_audio = bool(self.save_pulse_audio)
        self.save_figures = bool(self.save_figures)
        return self


# ---------------------------------------------------------------------------
# 模型加载与描述（复用现有模型文件，不新建模型管理系统）
# ---------------------------------------------------------------------------
def load_classification_model(
    model_path: Path | None = None,
) -> dict | None:
    """加载已训练好的确定模型（combined two-stage payload）。

    只读取已有模型文件，绝不在此训练。找不到返回 None（进入仅计数模式）。
    """
    import joblib

    path = Path(model_path) if model_path else COMBINED_DEMO_MODEL_PATH
    if not path.exists():
        return None
    payload = joblib.load(path)
    if not isinstance(payload, dict):
        return None
    return payload


def describe_model(
    combined_payload: dict | None,
    model_path: Path | str | None = None,
) -> dict:
    """返回模型状态描述，供 UI 显示与 run_metadata 记录。

    model_path：本次实际加载的模型文件（可由 ③/⑦ 的模型选择器指定）；
    不传则回退到默认的 ② 训练产物路径。

    is_temporary_acoustic_model：当前部署的是 KMeans 临时声型（A-D）
    模型而非专家确认物种模型时为 True，UI 必须显式提示。
    supports_probability：分类器是否具备 predict_proba（决定置信度是否可信可用）。
    is_calibrated / calibration_method / probability_kind / calibration_note：
    置信度是不是 CalibratedClassifierCV 的校准概率；未校准时 UI 必须明示
    “置信度未校准（模型支持度）”，绝不把原始 predict_proba 冒充校准概率。
    """
    if not combined_payload:
        return {
            "loaded": False,
            "labels": [],
            "feature_columns": [],
            "is_temporary_acoustic_model": False,
            "supports_probability": False,
            "is_calibrated": False,
            "calibration_method": "",
            "probability_kind": MODEL_SUPPORT,
            "calibration_note": "",
            "calibration_warnings": [],
            "model_path": str(model_path or COMBINED_DEMO_MODEL_PATH),
            "warning": "未检测到可用分类模型，仅可运行叫声检测与计数。",
        }

    acoustic = combined_payload.get("acoustic_types") or {}
    labels = [str(label) for label in (acoustic.get("labels") or [])]
    feature_columns = [
        str(column) for column in (acoustic.get("feature_columns") or [])
    ]
    classifier, _is_calibrated = select_classifier(acoustic)
    supports_probability = hasattr(classifier, "predict_proba")
    calibration = describe_calibration(acoustic)

    # 现有确定模型带 temporary_classifier + raw_cluster_to_type + kmeans，
    # 这是“KMeans 临时声型（默认 A-D）模型”的特征，不是专家确认物种模型。
    is_temporary = (
        "temporary_classifier" in acoustic
        or "raw_cluster_to_type" in acoustic
        or labels == list(ACOUSTIC_TYPE_LABELS)
    )

    warning = str(acoustic.get("warning") or "")
    if is_temporary:
        warning = (
            "当前模型为临时声型模型（A-D 声学相似类型），"
            "不是专家确认物种模型。"
        )

    return {
        "loaded": True,
        "labels": labels or list(ACOUSTIC_TYPE_LABELS),
        "feature_columns": feature_columns,
        "is_temporary_acoustic_model": bool(is_temporary),
        "supports_probability": bool(supports_probability),
        "is_calibrated": bool(calibration["is_calibrated"]),
        "calibration_method": str(calibration["calibration_method"] or ""),
        "probability_kind": str(calibration["probability_kind"]),
        "calibration_note": str(calibration["calibration_note"] or ""),
        "calibration_warnings": list(calibration["warnings"] or []),
        "model_path": str(model_path or COMBINED_DEMO_MODEL_PATH),
        "warning": warning,
    }


# ---------------------------------------------------------------------------
# 逐 pulse 分类（复用现有推理链：特征顺序 + StandardScaler + classifier + metadata）
# ---------------------------------------------------------------------------
def classify_pulse_rows(
    bat_df: pd.DataFrame,
    acoustic_payload: dict,
    config: SegmentAnalysisConfig,
) -> pd.DataFrame:
    """对每个 bat pulse 独立预测 label + confidence，并标记 uncertain。

    严格复用训练时的推理链，绝不自己重新拼 X：
    - 特征顺序取 acoustic_payload["feature_columns"]（= CLUSTER_FEATURE_COLUMNS）；
    - 数值清洗用 numeric_feature_frame，NaN 行剔除；
    - 分类器用**未缩放**特征 predict / predict_proba（与 prediction.py 一致）；
    - **置信度优先取校准概率**：模型带 calibrated_classifier（CalibratedClassifierCV）
      时用它出 predict / predict_proba，confidence_kind = calibrated_probability；
      否则退回临时分类器的原始 predict_proba，confidence_kind = model_support
      （未校准的模型支持度，不能当“预测正确的概率”读）；
    - StandardScaler + KMeans 仅用于对照声型（kmeans_label / model_agreement）。

    返回的 DataFrame 新增列：raw_label、predicted_label、confidence、is_uncertain、
    confidence_kind、kmeans_label、model_agreement。
    """
    features = list(acoustic_payload["feature_columns"])
    labels = [str(label) for label in (acoustic_payload.get("labels") or [])]
    scaler = acoustic_payload.get("scaler")
    kmeans = acoustic_payload.get("kmeans")
    classifier, is_calibrated = select_classifier(acoustic_payload)
    mapping = dict(acoustic_payload.get("raw_cluster_to_type") or {})
    confidence_kind = (
        CALIBRATED_PROBABILITY if is_calibrated else MODEL_SUPPORT
    )

    df = bat_df.copy()
    if df.empty:
        df["raw_label"] = []
        df["predicted_label"] = []
        df["confidence"] = []
        df["is_uncertain"] = []
        df["confidence_kind"] = []
        df["kmeans_label"] = []
        df["model_agreement"] = []
        return df

    # 1) 特征顺序严格按 metadata，数值清洗
    df[features] = numeric_feature_frame(df, features)
    valid_mask = df[features].notna().all(axis=1)
    df = df.loc[valid_mask].copy()
    if df.empty:
        for column in (
            "raw_label",
            "predicted_label",
            "confidence",
            "is_uncertain",
            "confidence_kind",
            "kmeans_label",
            "model_agreement",
        ):
            df[column] = []
        return df

    # 2) 分类器：未缩放特征（与训练/prediction 一致）；置信度优先校准概率
    raw_labels = np.asarray(classifier.predict(df[features]))

    supports_probability = hasattr(classifier, "predict_proba")
    if supports_probability:
        probabilities = classifier.predict_proba(df[features])
        class_to_index = {
            str(class_name): index
            for index, class_name in enumerate(classifier.classes_)
        }
        confidence = np.asarray(
            [
                probabilities[row, class_to_index[str(label)]]
                for row, label in enumerate(raw_labels)
            ],
            dtype=float,
        )
    else:
        confidence = np.full(len(df), np.nan)

    # 3) StandardScaler + KMeans 仅作对照（不作为最终物种标签）
    kmeans_labels = np.full(len(df), "", dtype=object)
    if scaler is not None and kmeans is not None:
        try:
            scaled = scaler.transform(df[features])
            raw_clusters = kmeans.predict(scaled)
            kmeans_labels = np.asarray(
                [
                    mapping.get(int(cluster), "")
                    for cluster in raw_clusters
                ],
                dtype=object,
            )
        except Exception:  # noqa: BLE001 - 对照失败不影响主分类
            kmeans_labels = np.full(len(df), "", dtype=object)

    # 4) 不确定标记 + 最终标签
    predicted: list[str] = []
    is_uncertain: list[bool] = []
    for index, label in enumerate(raw_labels):
        label_text = str(label)
        conf = confidence[index]
        uncertain = False
        if supports_probability and config.low_confidence_action == "uncertain":
            if not np.isfinite(conf) or conf < config.min_confidence:
                uncertain = True
        if uncertain:
            predicted.append(config.uncertain_label)
        else:
            predicted.append(label_text)
        is_uncertain.append(uncertain)

    df["raw_label"] = [str(label) for label in raw_labels]
    df["predicted_label"] = predicted
    df["confidence"] = confidence
    df["is_uncertain"] = is_uncertain
    df["confidence_kind"] = confidence_kind
    df["kmeans_label"] = kmeans_labels
    df["model_agreement"] = [
        str(raw) == str(kmeans)
        for raw, kmeans in zip(df["raw_label"], kmeans_labels)
    ]
    _ = labels  # labels 用于聚合阶段，这里仅为可读性保留
    return df


# ---------------------------------------------------------------------------
# 单片段分析
# ---------------------------------------------------------------------------
def _empty_label_counts(
    labels: list[str],
    uncertain_label: str,
) -> dict[str, int]:
    counts = {label: 0 for label in labels}
    counts[uncertain_label] = 0
    return counts


def _aggregate_label_counts(
    pulse_labels: list[str],
    labels: list[str],
    uncertain_label: str,
) -> dict[str, int]:
    """把逐 pulse 标签聚合成 {label: count}（含 uncertain，绝不把总数归给 major）。"""
    counts = _empty_label_counts(labels, uncertain_label)
    for label in pulse_labels:
        key = str(label)
        if key in counts:
            counts[key] += 1
        else:
            # 未知标签（理论上不应出现）归入 uncertain，保证总和守恒
            counts[uncertain_label] += 1
    return counts


def _checkpoint_identity(wav_path: Path) -> dict:
    from .segment_stream import file_identity

    return file_identity(wav_path)


def _replay_checkpoint_rows(
    checkpoint, keep_n: int | None
) -> tuple[list[dict], "set[tuple[float, float]]"]:
    """从 partial CSV 回放已写行（数值类型尽力还原）。

    keep_n：水位线（state.rows_total）——只认前 keep_n 行；
    崩溃时可能多写出的尾部行一律截掉，由续跑重新处理，
    绝不出现“计数一次 + 行里还留一次”的双重残留。
    返回 (rows, 时间键集合)。
    """
    import csv

    rows: list[dict] = []
    if not checkpoint.partial_csv_path.exists():
        return rows, set()
    with open(
        checkpoint.partial_csv_path, "r", newline="", encoding="utf-8-sig"
    ) as handle:
        for raw in csv.DictReader(handle):
            if keep_n is not None and len(rows) >= int(keep_n):
                break
            row: dict = {}
            for key, value in raw.items():
                if value is None or value == "":
                    row[key] = value
                    continue
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = value
            rows.append(row)
    keys = {
        (round(float(row["start_s"]), 3), round(float(row["end_s"]), 3))
        for row in rows
        if "start_s" in row and "end_s" in row
    }
    return rows, keys


def _finalize_chunk_rows(
    analysis_rows: list[dict],
    combined_payload: dict | None,
    noise_payload: dict | None,
    acoustic_payload: dict | None,
    config: "SegmentAnalysisConfig",
    unavailable_label: str,
    pulse_id_start: int,
) -> tuple[list[dict], int, int]:
    """噪声门控 + 逐 pulse 分类（按 chunk 小批量，语义与旧实现完全一致）。

    分类模型/特征/阈值不变；只是把“整段一次 predict”拆成“每 chunk 一次
    小批量 predict”（64~数百行），低内存高吞吐，结果逐行一致。

    返回 (result_rows, noise_count, next_pulse_id)。
    """
    result_rows: list[dict] = []
    if combined_payload is None:
        # 仅计数模式：无可用分类模型，标签 = unavailable
        for index, row in enumerate(analysis_rows):
            result_rows.append(
                {
                    **row,
                    "pulse_id": f"p{pulse_id_start + index:03d}",
                    "stage1_prediction": "bat",
                    "prob_bat": None,
                    "raw_label": unavailable_label,
                    "predicted_label": unavailable_label,
                    "confidence": None,
                    "is_uncertain": False,
                    "kmeans_label": "",
                    "model_agreement": False,
                }
            )
        return result_rows, 0, pulse_id_start + len(analysis_rows)

    # noise/bat 门控
    stage1_df = apply_noise_gate(pd.DataFrame(analysis_rows), noise_payload)
    noise_count = 0
    bat_df = stage1_df
    if not stage1_df.empty:
        bat_df = stage1_df.loc[
            stage1_df["stage1_prediction"] == "bat"
        ].copy()
        noise_df = stage1_df.loc[
            stage1_df["stage1_prediction"] == "noise"
        ].copy()
        noise_count = int(len(noise_df))
        # noise 行保留在明细中，但不进入物种分类
        for _, noise_row in noise_df.iterrows():
            result_rows.append(
                {
                    **noise_row.to_dict(),
                    "pulse_id": "",
                    "raw_label": "",
                    "predicted_label": "noise",
                    "confidence": (
                        float(noise_row.get("prob_bat"))
                        if noise_row.get("prob_bat") is not None
                        else None
                    ),
                    "is_uncertain": False,
                    # noise 行的 confidence 是第一层门控的 prob_bat，
                    # 既不是物种置信度也不是校准概率，明确标注来源。
                    "confidence_kind": "noise_gate_prob_bat",
                    "kmeans_label": "",
                    "model_agreement": False,
                }
            )

    classified_count = 0
    if not bat_df.empty:
        classified_df = classify_pulse_rows(
            bat_df, acoustic_payload, config
        )
        for position, (_, row) in enumerate(classified_df.iterrows()):
            confidence_value = row.get("confidence")
            result_rows.append(
                {
                    **row.to_dict(),
                    "pulse_id": f"p{pulse_id_start + position:03d}",
                    "confidence": (
                        float(confidence_value)
                        if confidence_value is not None
                        and np.isfinite(confidence_value)
                        else None
                    ),
                }
            )
        classified_count = int(len(classified_df))
    return result_rows, noise_count, pulse_id_start + classified_count


PULSE_ROW_EXTRA_KEYS = (
    "pulse_id",
    "stage1_prediction",
    "prob_bat",
    "prob_noise",
    "raw_label",
    "predicted_label",
    "confidence",
    "confidence_kind",
    "is_uncertain",
    "kmeans_label",
    "model_agreement",
)


def _csv_fieldnames(rows: list[dict]) -> list[str]:
    if not rows:
        return list(PULSE_ROW_EXTRA_KEYS)
    fieldnames = list(rows[0].keys())
    for key in PULSE_ROW_EXTRA_KEYS:
        if key not in fieldnames:
            fieldnames.append(key)
    return fieldnames


class _SegmentAccumulator:
    """流式累计：计数/标签分布/置信度均值，支持 checkpoint 序列化。

    **计数绝不被导出上限影响**：这里累计的是检测/分类的全量结果，
    导出预算只在写 WAV 时生效。
    """

    def __init__(
        self,
        labels: list[str],
        uncertain_label: str,
        config: "SegmentAnalysisConfig",
    ) -> None:
        self.labels = list(labels)
        self.uncertain_label = uncertain_label
        self.config = config
        self.raw_candidate_count = 0
        self.candidate_call_count = 0
        self.valid_call_count = 0
        self.noise_count = 0
        self.bat_call_count = 0
        self.label_counts = _empty_label_counts(self.labels, uncertain_label)
        self.confidence_sum = 0.0
        self.confidence_n = 0
        self.next_pulse_id = 1
        self.rows: list[dict] = []
        self.rows_total = 0
        self.rows_truncated = False
        self.pulse_file_map: dict[str, str] = {}
        self.processed_chunk_count = 0
        self.fast_skipped_chunk_count = 0
        self.failed_duration_s = 0.0

    # ---- 增量累计 ----
    def add_batch(
        self,
        *,
        raw_candidate_count: int,
        pulse_rows: list[dict],
        result_rows: list[dict],
        noise_count: int,
        next_pulse_id: int,
        skipped: bool,
    ) -> None:
        self.raw_candidate_count += int(raw_candidate_count)
        self.candidate_call_count += len(pulse_rows)
        self.valid_call_count += sum(
            1 for row in pulse_rows if bool(row.get("accepted"))
        )
        self.noise_count += int(noise_count)
        self.next_pulse_id = int(next_pulse_id)

        bat_rows = [
            row
            for row in result_rows
            if str(row.get("predicted_label", "")) != "noise"
        ]
        self.bat_call_count += len(bat_rows)
        batch_labels = [
            str(row.get("predicted_label", "")) for row in bat_rows
        ]
        if batch_labels:
            counts = _aggregate_label_counts(
                batch_labels, self.labels, self.uncertain_label
            )
            for key, value in counts.items():
                self.label_counts[key] = int(
                    self.label_counts.get(key, 0)
                ) + int(value)
        for row in bat_rows:
            if row.get("confidence") is not None and not row.get("is_uncertain"):
                self.confidence_sum += float(row["confidence"])
                self.confidence_n += 1

        self.rows_total += len(result_rows)
        max_rows = int(SEGMENT_MAX_INMEMORY_ROWS)
        if max_rows <= 0 or len(self.rows) < max_rows:
            room = (
                max_rows - len(self.rows) if max_rows > 0 else len(result_rows)
            )
            self.rows.extend(result_rows[:room])
        elif result_rows:
            self.rows_truncated = True

        if skipped:
            self.fast_skipped_chunk_count += 1
        else:
            self.processed_chunk_count += 1

    # ---- checkpoint ----
    def to_state(self) -> dict:
        return {
            "raw_candidate_count": self.raw_candidate_count,
            "candidate_call_count": self.candidate_call_count,
            "valid_call_count": self.valid_call_count,
            "noise_count": self.noise_count,
            "bat_call_count": self.bat_call_count,
            "label_counts": dict(self.label_counts),
            "confidence_sum": self.confidence_sum,
            "confidence_n": self.confidence_n,
            "next_pulse_id": self.next_pulse_id,
            "rows_total": self.rows_total,
            "processed_chunk_count": self.processed_chunk_count,
            "fast_skipped_chunk_count": self.fast_skipped_chunk_count,
            "failed_duration_s": self.failed_duration_s,
            "pulse_file_map": dict(self.pulse_file_map),
        }

    def from_state(self, state: dict) -> None:
        for key in (
            "raw_candidate_count",
            "candidate_call_count",
            "valid_call_count",
            "noise_count",
            "bat_call_count",
            "confidence_sum",
            "confidence_n",
            "next_pulse_id",
            "rows_total",
            "processed_chunk_count",
            "fast_skipped_chunk_count",
            "failed_duration_s",
        ):
            if key in state:
                setattr(self, key, type(getattr(self, key))(state[key]))
        if isinstance(state.get("label_counts"), dict):
            self.label_counts = {
                str(key): int(value)
                for key, value in state["label_counts"].items()
            }
        if isinstance(state.get("pulse_file_map"), dict):
            self.pulse_file_map = {
                str(key): str(value)
                for key, value in state["pulse_file_map"].items()
            }

    # ---- 输出 ----
    @property
    def uncertain_count(self) -> int:
        return int(self.label_counts.get(self.uncertain_label, 0))

    @property
    def classified_call_count(self) -> int:
        return int(self.bat_call_count - self.uncertain_count)

    @property
    def detected_call_count(self) -> int:
        return int(self.bat_call_count)

    @property
    def mean_confidence(self) -> float | None:
        return (
            float(self.confidence_sum / self.confidence_n)
            if self.confidence_n
            else None
        )


def _export_batch_pulse_audio(
    result_rows: list[dict],
    *,
    wav_path: Path,
    segment_output_dir: Path,
    read_y: np.ndarray,
    read_start_s: float,
    batch_offset: int,
    budget_remaining: int,
) -> tuple[int, bool]:
    """按批导出 pulse WAV。返回 (本批导出数, 是否因预算截断)。

    **导出上限只影响磁盘文件数量**：行已全部进入计数/分类/CSV。
    """
    from .clustering import export_pulse_audio

    stem = wav_path.stem
    exported = 0
    truncated = False
    cache: dict[str, tuple[np.ndarray, int]] = {
        str(wav_path): (read_y, TARGET_SAMPLE_RATE)
    }
    for position, row in enumerate(result_rows):
        if str(row.get("predicted_label", "")) == "noise":
            continue
        if budget_remaining <= 0:
            truncated = True
            break
        index = batch_offset + position + 1
        destination = segment_output_dir / f"{stem}__p{index:03d}.wav"
        local_row = dict(row)
        local_row["start_s"] = float(row.get("start_s", 0.0)) - read_start_s
        local_row["end_s"] = float(row.get("end_s", 0.0)) - read_start_s
        try:
            written = export_pulse_audio(local_row, destination, cache)
        except Exception:  # noqa: BLE001 - 导出失败不影响统计
            written = ""
        if written:
            row["pulse_file"] = written
            exported += 1
            budget_remaining -= 1
    return exported, truncated


def analyze_segment(
    wav_path: Path | str,
    combined_payload: dict | None,
    config: SegmentAnalysisConfig,
    *,
    segment_output_dir: Path | None = None,
    separated_dir: Path | None = None,
    figure_path: Path | None = None,
    chunk_progress: "Callable[[dict], None] | None" = None,
    should_stop: "Callable[[], bool] | None" = None,
    resume: bool | None = None,
    max_workers: int | None = None,
) -> dict:
    """分析单个 WAV（几分钟片段，或 3 小时以上整条长录音）。

    检测内部叫声 + 逐叫声分类 + 片段汇总。算法不依赖 Qt。

    流式 chunk pipeline（segment_stream）：
    - 任何时刻只驻留 1~2 个 chunk，内存与文件总时长无关（绝不整条载入）；
    - **FULL RECORDING**：coverage 连续铺满 0→总时长，绝不抽样；
    - chunk + overlap，脉冲时间统一转 global 绝对时间，跨块不漏不重；
    - 结果增量落盘 + checkpoint；stop 在 chunk 边界优雅退出；
    - resume=True 时从断点续跑（校验文件身份 + 跨块去重防重复计数）。

    chunk_progress：块级进度回调；should_stop：取消标记。
    """
    from .segment_stream import (
        ChunkCheckpoint,
        ChunkContext,
        StreamingPreprocessor,
        StreamStats,
        SpectrogramSummary,
        build_chunk_specs,
        current_rss_mb,
        dedupe_cross_chunk_rows,
        probe_audio,
        resolve_max_workers,
        stream_chunks,
    )

    config = config.validate()
    wav_path = Path(wav_path)

    acoustic_payload = (
        combined_payload.get("acoustic_types") if combined_payload else None
    )
    noise_payload = (
        combined_payload.get("noise_gate") if combined_payload else None
    )
    labels = (
        [str(label) for label in (acoustic_payload or {}).get("labels") or []]
        or (list(ACOUSTIC_TYPE_LABELS) if acoustic_payload else [config.unavailable_label])
    )
    uncertain_label = config.uncertain_label
    unavailable_label = config.unavailable_label

    # ---- 流式探测：绝不整条载入（内存与文件总长无关）----
    total_frames, orig_sr, duration_s = probe_audio(wav_path)
    recording_duration_s = float(duration_s)
    # 速率分母 = 有效监测总时长（整条原始录音）；静音快筛只是计算优化，
    # 不代表那段时间从生态监测时间里消失。只有损坏/无法解码区间才排除。
    valid_monitoring_duration_s = float(recording_duration_s)
    failed_duration_s = 0.0

    specs = build_chunk_specs(total_frames, orig_sr)
    preprocessor = StreamingPreprocessor(
        wav_path, total_frames, orig_sr, StreamStats()
    )
    stats = preprocessor.compute_stats()
    context = ChunkContext(
        wav_path=wav_path,
        preprocessor=preprocessor,
        stats=stats,
        total_samples=total_frames,
        separated_dir=separated_dir,
        save_separated_audio=True,
    )
    accumulator = _SegmentAccumulator(labels, uncertain_label, config)

    # ---- checkpoint / 断点续跑 ----
    checkpoint = None
    resumed = False
    start_at_s = 0.0
    seen_row_keys: set[tuple[float, float]] = set()
    if segment_output_dir is not None:
        checkpoint = ChunkCheckpoint(
            Path(segment_output_dir) / ".checkpoint", wav_path
        )
        state = checkpoint.load_resumable() if resume else None
        if state is not None:
            accumulator.from_state(dict(state.get("accumulator") or {}))
            start_at_s = float(state.get("processed_until_s") or 0.0)
            resumed = True
            # 回放已写行（带水位线截断）；行集即计数集，续跑绝不双算
            max_rows = int(SEGMENT_MAX_INMEMORY_ROWS)
            replayed_rows, seen_row_keys = _replay_checkpoint_rows(
                checkpoint, int(accumulator.rows_total)
            )
            accumulator.rows = (
                replayed_rows[:max_rows]
                if max_rows > 0
                else replayed_rows
            )
        else:
            checkpoint.clear()

    if start_at_s > 0:
        specs = [
            spec
            for spec in specs
            if spec.coverage_end_s > start_at_s + 1e-9
            or spec.coverage_start_s >= start_at_s - 1e-9
        ]

    # ---- 流式处理主循环 ----
    export_budget = int(SEGMENT_MAX_PULSE_WAV_EXPORTS)
    exported_count = 0
    export_truncated = False
    prev_tail_rows: list[dict] = []
    batch_offset = int(accumulator.rows_total)
    processed_until_s = float(start_at_s)
    cancelled = False
    start_clock = datetime.now()
    summary = (
        SpectrogramSummary(recording_duration_s, sr=TARGET_SAMPLE_RATE)
        if config.save_figures and figure_path is not None
        else None
    )
    pending_rows: list[dict] = []

    def _on_progress(payload: dict) -> None:
        if chunk_progress is None:
            return
        elapsed = max(
            (datetime.now() - start_clock).total_seconds(), 1e-6
        )
        processed = float(payload.get("processed_until_s", 0.0))
        chunk_progress(
            {
                "segment_file": wav_path.name,
                "recording_duration_s": recording_duration_s,
                "processed_until_s": processed,
                "progress_ratio": (
                    processed / recording_duration_s
                    if recording_duration_s > 0
                    else 0.0
                ),
                "elapsed_s": elapsed,
                "speed_realtime": recording_duration_s
                and min(processed, recording_duration_s) / elapsed,
                "chunk_index": payload.get("chunk_index"),
                "chunk_count": payload.get("chunk_count"),
                "raw_candidate_count": accumulator.raw_candidate_count
                + int(payload.get("raw_candidate_count", 0)),
                "candidate_call_count": accumulator.candidate_call_count,
                "detected_call_count": accumulator.detected_call_count,
                "label_counts": dict(accumulator.label_counts),
                "fast_skipped_chunk_count": accumulator.fast_skipped_chunk_count,
                "rss_mb": current_rss_mb(),
            }
        )

    for batch in stream_chunks(
        context,
        specs,
        max_workers=resolve_max_workers(max_workers),
        progress=lambda payload: None,
        should_stop=should_stop,
    ):
        # 跨块去重：同一真实 pulse 只计一次（边界安全网）
        rows = dedupe_cross_chunk_rows(prev_tail_rows, batch.rows)
        if seen_row_keys:
            rows = [
                row
                for row in rows
                if (
                    round(float(row["start_s"]), 3),
                    round(float(row["end_s"]), 3),
                )
                not in seen_row_keys
            ]
        prev_tail_rows = [dict(row) for row in rows[-8:]]

        accepted_rows = [row for row in rows if bool(row.get("accepted"))]
        analysis_rows = accepted_rows if config.use_quality_filter else rows

        # 小批量分类（模型语义完全不变）
        result_rows, noise_count, next_id = _finalize_chunk_rows(
            analysis_rows,
            combined_payload,
            noise_payload,
            acoustic_payload,
            config,
            unavailable_label,
            accumulator.next_pulse_id,
        )
        result_rows.sort(key=lambda r: float(r.get("start_s", 0.0)))

        # 导出 pulse WAV（导出上限只影响磁盘；计数已在上面全量累计）
        if config.save_pulse_audio and segment_output_dir is not None:
            segment_output_dir.mkdir(parents=True, exist_ok=True)
            if export_budget > 0:
                remaining = export_budget - exported_count
            else:
                remaining = -1  # 不限
            exported, truncated = _export_batch_pulse_audio(
                result_rows,
                wav_path=wav_path,
                segment_output_dir=Path(segment_output_dir),
                read_y=batch.read_y if batch.read_y is not None
                else np.zeros(0),
                read_start_s=batch.spec.read_start_s,
                batch_offset=batch_offset,
                budget_remaining=remaining if remaining != -1 else 10**9,
            )
            exported_count += exported
            export_truncated = export_truncated or truncated

        if summary is not None and batch.read_y is not None:
            try:
                summary.add(batch.read_y, batch.spec.read_start_s)
            except Exception:  # noqa: BLE001 - 概览图失败不影响统计
                pass

        accumulator.add_batch(
            raw_candidate_count=batch.raw_candidate_count,
            pulse_rows=rows,
            result_rows=result_rows,
            noise_count=noise_count,
            next_pulse_id=next_id,
            skipped=bool(batch.skipped),
        )
        batch_offset += len(result_rows)
        batch.read_y = None
        processed_until_s = batch.spec.coverage_end_s

        # 增量落盘 + checkpoint（长任务不等全跑完才第一次写结果）
        if checkpoint is not None:
            pending_rows.extend(result_rows)
            checkpoint.append_rows(pending_rows, columns=_csv_fieldnames(pending_rows))
            pending_rows = []
            checkpoint.save(
                {
                    "status": "in_progress",
                    "file_identity": _checkpoint_identity(wav_path),
                    "processed_until_s": processed_until_s,
                    "accumulator": accumulator.to_state(),
                    "recording_duration_s": recording_duration_s,
                    "valid_monitoring_duration_s": valid_monitoring_duration_s,
                }
            )
        _on_progress(
            {
                "processed_until_s": processed_until_s,
                "chunk_index": batch.spec.index,
                "chunk_count": len(specs),
                "raw_candidate_count": 0,
            }
        )

    if should_stop is not None and should_stop():
        cancelled = True

    # ---- 片段聚合（与旧实现同一口径）----
    label_counts = dict(accumulator.label_counts)
    uncertain_count = accumulator.uncertain_count
    classified_call_count = accumulator.classified_call_count
    specific_counts = {
        label: int(label_counts.get(label, 0)) for label in labels
    }
    if classified_call_count > 0:
        major_label = max(specific_counts, key=lambda k: specific_counts[k])
        if specific_counts[major_label] == 0:
            major_label = ""
        purity = float(specific_counts[major_label] / classified_call_count)
    else:
        major_label = ""
        purity = 0.0
    consistency_note = ""
    if classified_call_count > 0 and purity < config.mixed_purity_threshold:
        consistency_note = "混合 / 低一致性"

    detected_call_count = accumulator.detected_call_count
    calls_per_second = (
        float(detected_call_count / valid_monitoring_duration_s)
        if valid_monitoring_duration_s > 0
        else 0.0
    )
    mean_confidence = accumulator.mean_confidence

    # ---- 每段 pulse_results.csv（增量写入后原子替换）----
    if segment_output_dir is not None and checkpoint is not None:
        segment_output_dir = Path(segment_output_dir)
        segment_output_dir.mkdir(parents=True, exist_ok=True)
        if pending_rows:
            checkpoint.append_rows(
                pending_rows, columns=_csv_fieldnames(pending_rows)
            )
            pending_rows = []
        if cancelled:
            # 取消：partial 结果保留可见，checkpoint 保留可续跑
            checkpoint.save(
                {
                    "status": "cancelled",
                    "file_identity": _checkpoint_identity(wav_path),
                    "processed_until_s": processed_until_s,
                    "accumulator": accumulator.to_state(),
                    "recording_duration_s": recording_duration_s,
                    "valid_monitoring_duration_s": valid_monitoring_duration_s,
                }
            )
        else:
            checkpoint.promote_partial_csv(
                segment_output_dir / "pulse_results.csv"
            )
            checkpoint.clear()

    # ---- 片段声谱图（流式低分辨率摘要；仅可视化，不参与统计）----
    if config.save_figures and figure_path is not None and summary is not None:
        try:
            from .visualization import save_segment_call_overview_from_summary

            save_segment_call_overview_from_summary(
                summary,
                accumulator.rows,
                figure_path,
                title=(
                    f"{wav_path.name} | calls: {detected_call_count}"
                ),
            )
        except Exception:  # noqa: BLE001 - 绘图失败不影响统计
            pass

    return {
        "segment_id": wav_path.stem,
        "segment_file": wav_path.name,
        "segment_path": str(wav_path),
        "duration_s": recording_duration_s,
        "status": "cancelled" if cancelled else "success",
        "error_message": "",
        # 四层计数（严格区分，见 README）
        "raw_candidate_count": int(accumulator.raw_candidate_count),
        "candidate_call_count": accumulator.candidate_call_count,
        "valid_call_count": accumulator.valid_call_count,
        "noise_count": int(accumulator.noise_count),
        "bat_call_count": accumulator.bat_call_count,
        "classified_call_count": classified_call_count,
        "uncertain_count": uncertain_count,
        # 主结果：有效蝙蝠叫声次数
        "detected_call_count": detected_call_count,
        "calls_per_second": calls_per_second,
        # 片段主要类别（描述性，非新物种算法）
        "major_label": major_label,
        "predicted_label": major_label,
        "prediction_confidence": mean_confidence,
        "purity": purity,
        "consistency_note": consistency_note,
        "label_counts": label_counts,
        "pulse_rows": accumulator.rows,
        "pulse_rows_truncated": accumulator.rows_truncated,
        "pulse_file_map": accumulator.pulse_file_map,
        # ---- 长录音口径（⑦ 流式）----
        "recording_duration_s": recording_duration_s,
        "valid_monitoring_duration_s": valid_monitoring_duration_s,
        "failed_duration_s": failed_duration_s,
        "processed_chunk_count": accumulator.processed_chunk_count,
        "fast_skipped_chunk_count": accumulator.fast_skipped_chunk_count,
        "chunk_seconds": float(SEGMENT_STREAM_CHUNK_SECONDS),
        "overlap_seconds": float(SEGMENT_STREAM_OVERLAP_SECONDS),
        "exported_pulse_count": int(exported_count),
        "export_truncated": bool(export_truncated),
        "resumed": bool(resumed),
    }






# ---------------------------------------------------------------------------
# 批量分析 + 汇总
# ---------------------------------------------------------------------------
def _failed_segment_result(
    wav_path: Path, error: str
) -> dict:
    return {
        "segment_id": Path(wav_path).stem,
        "segment_file": Path(wav_path).name,
        "segment_path": str(wav_path),
        "duration_s": 0.0,
        "status": "failed",
        "error_message": str(error),
        "raw_candidate_count": 0,
        "candidate_call_count": 0,
        "valid_call_count": 0,
        "noise_count": 0,
        "bat_call_count": 0,
        "classified_call_count": 0,
        "uncertain_count": 0,
        "detected_call_count": 0,
        "calls_per_second": 0.0,
        "major_label": "",
        "predicted_label": "",
        "prediction_confidence": None,
        "purity": 0.0,
        "consistency_note": "",
        "label_counts": {},
        "pulse_rows": [],
        "pulse_rows_truncated": False,
        "pulse_file_map": {},
        # ---- 长录音口径（⑦ 流式）----
        "recording_duration_s": 0.0,
        "valid_monitoring_duration_s": 0.0,
        "failed_duration_s": 0.0,
        "processed_chunk_count": 0,
        "fast_skipped_chunk_count": 0,
        "chunk_seconds": float(SEGMENT_STREAM_CHUNK_SECONDS),
        "overlap_seconds": float(SEGMENT_STREAM_OVERLAP_SECONDS),
        "exported_pulse_count": 0,
        "export_truncated": False,
        "resumed": False,
    }


def summarize_segment_results(
    segment_results: list[dict],
    labels: list[str],
    uncertain_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把逐片段结果聚合成 (片段汇总表, 总体类别统计表)。

    统计正确性要求：一段内多类别时，各类别按**实际 call 数**累加，
    绝不把整段 total_calls 归给 major_label。
    """
    # ---- 第二层：片段汇总 ----
    summary_rows: list[dict] = []
    for result in segment_results:
        label_counts = result.get("label_counts", {}) or {}
        row = {
            "segment_id": result.get("segment_id", ""),
            "segment_file": result.get("segment_file", ""),
            "segment_path": result.get("segment_path", ""),
            "duration_s": result.get("duration_s", 0.0),
            "major_label": result.get("major_label", ""),
            "total_calls": result.get("detected_call_count", 0),
            "classified_call_count": result.get("classified_call_count", 0),
            "uncertain_count": result.get("uncertain_count", 0),
            "purity": result.get("purity", 0.0),
            "consistency_note": result.get("consistency_note", ""),
            "prediction_confidence": result.get("prediction_confidence"),
            "raw_candidate_count": result.get("raw_candidate_count", 0),
            "candidate_call_count": result.get("candidate_call_count", 0),
            "valid_call_count": result.get("valid_call_count", 0),
            "noise_count": result.get("noise_count", 0),
            "calls_per_second": result.get("calls_per_second", 0.0),
            "recording_duration_s": result.get("recording_duration_s", 0.0),
            "valid_monitoring_duration_s": result.get(
                "valid_monitoring_duration_s", 0.0
            ),
            "failed_duration_s": result.get("failed_duration_s", 0.0),
            "processed_chunk_count": result.get("processed_chunk_count", 0),
            "fast_skipped_chunk_count": result.get(
                "fast_skipped_chunk_count", 0
            ),
            "processing_status": result.get("status", ""),
            "error_message": result.get("error_message", ""),
        }
        for label in labels:
            row[label] = int(label_counts.get(label, 0))
        row[uncertain_label] = int(label_counts.get(uncertain_label, 0))
        summary_rows.append(row)

    summary_columns = [
        "segment_id",
        "segment_file",
        "segment_path",
        "duration_s",
        "major_label",
        "total_calls",
        *labels,
        uncertain_label,
        "classified_call_count",
        "uncertain_count",
        "purity",
        "consistency_note",
        "prediction_confidence",
        "raw_candidate_count",
        "candidate_call_count",
        "valid_call_count",
        "noise_count",
        "calls_per_second",
        "recording_duration_s",
        "valid_monitoring_duration_s",
        "failed_duration_s",
        "processed_chunk_count",
        "fast_skipped_chunk_count",
        "processing_status",
        "error_message",
    ]
    segment_summary_df = pd.DataFrame(summary_rows)
    for column in summary_columns:
        if column not in segment_summary_df.columns:
            segment_summary_df[column] = 0
    segment_summary_df = segment_summary_df[summary_columns]

    # ---- 第三层：总体类别统计（严格按各类别 call 数累加）----
    overall_rows: list[dict] = []
    all_labels = list(labels) + [uncertain_label]
    for label in all_labels:
        call_count = 0
        segment_count = 0
        for result in segment_results:
            label_counts = result.get("label_counts", {}) or {}
            count = int(label_counts.get(label, 0))
            call_count += count
            if count >= 1:
                segment_count += 1
        overall_rows.append(
            {
                "species": label,
                "segment_count": segment_count,
                "call_count": call_count,
            }
        )

    overall_df = pd.DataFrame(overall_rows)
    return segment_summary_df, overall_df


def analyze_segment_folder(
    input_path: Path,
    combined_payload: dict | None,
    config: SegmentAnalysisConfig,
    *,
    output_root: Path | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    chunk_progress: Callable[[dict], None] | None = None,
    on_segment_done: Callable[[dict], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    model_path: Path | str | None = None,
    resume: bool | None = None,
) -> dict:
    """批量分析文件夹/单个 WAV 的所有片段，写独立输出目录并返回汇总。

    - 逐个 WAV：流式 analyze → save → release（不一次载入全部到内存）；
    - 失败隔离：单个损坏 WAV 记 failed，继续后续；
    - chunk_progress：块级进度回调（长录音逐块上报）；
    - resume：断点续跑开关（校验文件身份，变了自动重新开始）；
    - 输出只写 results/segment_analysis/<run_id>/（或 output_root）。
    """
    config = config.validate()
    input_path = Path(input_path)

    if input_path.is_file():
        wav_files = (
            [input_path] if input_path.suffix.lower() == ".wav" else []
        )
    elif input_path.is_dir():
        wav_files = find_wav_files(input_path)
    else:
        raise FileNotFoundError(f"输入路径不存在：{input_path}")

    if not wav_files:
        raise FileNotFoundError(f"没有找到 WAV 文件：{input_path}")

    # ---- 输出目录（独立 run）----
    if output_root is None:
        base = SEGMENT_ANALYSIS_DIR
        base.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base / run_id
        suffix = 1
        while run_dir.exists():
            suffix += 1
            run_dir = base / f"{run_id}_{suffix}"
    else:
        run_dir = Path(output_root)
    segments_dir = run_dir / SEGMENT_PULSE_SUBDIR
    figures_dir = run_dir / SEGMENT_FIGURE_SUBDIR
    separated_dir = run_dir / SEGMENT_SEPARATED_SUBDIR
    for directory in (run_dir, segments_dir, figures_dir, separated_dir):
        directory.mkdir(parents=True, exist_ok=True)

    model_info = describe_model(combined_payload, model_path=model_path)
    if model_info["loaded"]:
        labels = model_info["labels"] or list(ACOUSTIC_TYPE_LABELS)
    else:
        labels = [config.unavailable_label]
    uncertain_label = config.uncertain_label

    segment_results: list[dict] = []
    total = len(wav_files)

    for index, wav_path in enumerate(wav_files, start=1):
        if should_stop is not None and should_stop():
            break
        if progress is not None:
            progress(index, total, wav_path.name)

        segment_id = wav_path.stem
        segment_output_dir = segments_dir / segment_id
        figure_path = figures_dir / f"{segment_id}.png"
        try:
            result = analyze_segment(
                wav_path,
                combined_payload,
                config,
                segment_output_dir=segment_output_dir,
                separated_dir=separated_dir,
                figure_path=figure_path,
                chunk_progress=chunk_progress,
                should_stop=should_stop,
                resume=resume,
            )
        except Exception as error:  # noqa: BLE001 - 失败隔离
            result = _failed_segment_result(wav_path, f"{type(error).__name__}: {error}")

        segment_results.append(result)
        if on_segment_done is not None:
            try:
                on_segment_done(result)
            except Exception:  # noqa: BLE001 - 回调异常不影响批处理
                pass

    segment_summary_df, overall_df = summarize_segment_results(
        segment_results, labels, uncertain_label
    )

    # ---- 逐叫声明细（第一层）----
    # 内存有界：段内行超过 SEGMENT_MAX_INMEMORY_ROWS 时，
    # 明细已增量写入 pulse_results.csv，这里改从 CSV 流式汇总。
    all_pulse_rows: list[dict] = []
    prediction_columns = [
        "pulse_id",
        "start_s",
        "end_s",
        "duration_ms",
        "predicted_label",
        "raw_label",
        "confidence",
        "is_uncertain",
        "stage1_prediction",
        "accepted",
        "peak_freq_khz",
        "start_freq_khz",
        "end_freq_khz",
        "snr_db",
        "pulse_file",
    ]
    for result in segment_results:
        rows_iter: list[dict] = list(result.get("pulse_rows", []))
        if result.get("pulse_rows_truncated"):
            csv_path = (
                Path(run_dir)
                / SEGMENT_PULSE_SUBDIR
                / str(result.get("segment_id", ""))
                / "pulse_results.csv"
            )
            if csv_path.exists():
                rows_iter = pd.read_csv(
                    csv_path, encoding="utf-8-sig"
                ).to_dict("records")
        for row in rows_iter:
            all_pulse_rows.append(
                {
                    "segment_file": result.get("segment_file", ""),
                    "segment_id": result.get("segment_id", ""),
                    **{
                        column: row.get(column, "")
                        for column in prediction_columns
                    },
                }
            )
    pulse_predictions_df = pd.DataFrame(all_pulse_rows)

    # ---- 写 CSV ----
    segment_summary_df.to_csv(
        run_dir / "segment_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pulse_predictions_df.to_csv(
        run_dir / "pulse_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    overall_df.to_csv(
        run_dir / "overall_species_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    success_count = sum(
        1 for r in segment_results if r.get("status") == "success"
    )
    failed_count = len(segment_results) - success_count

    run_metadata = {
        "run_id": run_dir.name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_path": str(input_path),
        "segment_count": len(segment_results),
        "success_count": success_count,
        "failed_count": failed_count,
        "total_bat_calls": int(
            sum(r.get("detected_call_count", 0) for r in segment_results)
        ),
        "total_recording_duration_s": round(
            float(sum(r.get("recording_duration_s", 0.0) for r in segment_results)),
            3,
        ),
        "total_valid_monitoring_duration_s": round(
            float(
                sum(
                    r.get("valid_monitoring_duration_s", 0.0)
                    for r in segment_results
                )
            ),
            3,
        ),
        "cancelled_count": sum(
            1 for r in segment_results if r.get("status") == "cancelled"
        ),
        "streaming": {
            "chunk_seconds": float(SEGMENT_STREAM_CHUNK_SECONDS),
            "overlap_seconds": float(SEGMENT_STREAM_OVERLAP_SECONDS),
            "full_recording_coverage": True,
            "checkpoint_enabled": True,
        },
        "labels": labels,
        "uncertain_label": uncertain_label,
        "config": asdict(config),
        "model": model_info,
        "warning": (
            "统计的是声学检测到的 call 数量，不是蝙蝠个体数量；"
            "标签为模型输出（当前为临时声型 A-D 时不是专家确认物种）。"
        ),
        "output_files": {
            "segment_summary": "segment_summary.csv",
            "pulse_predictions": "pulse_predictions.csv",
            "overall_species_summary": "overall_species_summary.csv",
            "segments_dir": SEGMENT_PULSE_SUBDIR,
            "figures_dir": SEGMENT_FIGURE_SUBDIR,
        },
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "run_dir": str(run_dir),
        "segment_results": segment_results,
        "segment_summary": segment_summary_df,
        "pulse_predictions": pulse_predictions_df,
        "overall_summary": overall_df,
        "success_count": success_count,
        "failed_count": failed_count,
        "segment_count": len(segment_results),
        "model": model_info,
        "labels": labels,
        "uncertain_label": uncertain_label,
    }
