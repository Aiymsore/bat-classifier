"""概率校准（CalibratedClassifierCV）：让 ⑦ 的“置信度”成为校准概率。

解决的问题：⑦ 原来的 confidence 是 RandomForest 的 `predict_proba` 原始输出
（投票占比），**未校准**，不能当“预测正确的概率”读。本模块在**训练/补校准**
阶段用 `CalibratedClassifierCV` 校准概率，⑦ 推理时优先使用校准分类器。

三个使用场景：
1. ② 训练（`clustering.cluster_acoustic_types`）：训练完临时分类器后自动校准，
   校准器随模型一起写进 `models/noise_bat_ABCD_model.joblib`；
2. “补校准”（`attach_calibration`）：给**已训练好的两级模型**补一个校准器，
   不重新训练、不覆盖原模型；
3. ⑦ 推理（`segment_analysis.classify_pulse_rows`）：优先用校准分类器输出
   `calibrated_probability`；没有校准器就退回原始 `predict_proba`，并在
   UI / `run_metadata.json` 明确标注“未校准（模型支持度）”。

科学约束（与 ⑥ 已确认物种模型一致）：
- 校准用显式 (train, calib) 切分，**同一 source_recording 的 pulse 绝不跨两侧**；
- 数据不足以安全校准 → **绝不伪造**：classifier=None，只输出未校准置信度；
- 推理路径绝不 fit；本模块只有 `calibrate_*` / `attach_calibration` 会 fit。
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd

from .config import (
    RANDOM_STATE,
    SEGMENT_CALIBRATION_FOLDS,
    SEGMENT_CALIBRATION_METHOD,
    SEGMENT_ISOTONIC_MIN_SAMPLES,
    SEGMENT_MIN_CLASS_SAMPLES,
    SEGMENT_MIN_RECORDINGS_FOR_CALIBRATION,
)

# ⑦ 输出的置信度性质标记（写进 pulse_predictions.csv 的 confidence_kind 列）
CALIBRATED_PROBABILITY = "calibrated_probability"   # 校准概率
MODEL_SUPPORT = "model_support"                     # 未校准的模型支持度

# payload 里挂校准结果的键（acoustic_types 段内）
CALIBRATED_CLASSIFIER_KEY = "calibrated_classifier"
CALIBRATION_INFO_KEY = "calibration"

INSUFFICIENT_CALIBRATION_WARNING = "当前数据不足以进行可靠概率校准"
UNGROUPED_CALIBRATION_WARNING = (
    "未提供 source_recording 分组：校准切分未按录音隔离，"
    "同一录音的脉冲可能同时进入训练侧与校准侧。"
)

# 补校准用的标注 CSV 里，真值标签列的候选名（按优先级）。
# 故意不含 predicted_label：模型自己的预测不能当真值来校准自己。
LABEL_COLUMN_CANDIDATES = (
    "acoustic_type",
    "label",
    "expert_label",
    "true_label",
    "species",
)
SOURCE_COLUMN_CANDIDATES = (
    "source_file",
    "source_recording",
    "source_path",
)


def detect_label_column(frame: pd.DataFrame) -> str:
    """在标注 CSV 里找真值标签列名；找不到返回 ""。"""
    for name in LABEL_COLUMN_CANDIDATES:
        if name in frame.columns:
            return name
    return ""


def detect_source_column(frame: pd.DataFrame) -> str:
    """在标注 CSV 里找 source_recording 分组列名；找不到返回 ""。"""
    for name in SOURCE_COLUMN_CANDIDATES:
        if name in frame.columns:
            return name
    return ""


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def resolve_calibration_method(
    requested: str | None,
    sample_count: int,
) -> tuple[str, str]:
    """决定校准方法；isotonic 只在样本量足够时使用（绝不默认用于小样本）。"""
    text = (requested or SEGMENT_CALIBRATION_METHOD).strip().lower()
    if text not in ("sigmoid", "isotonic", "auto"):
        text = SEGMENT_CALIBRATION_METHOD
    if text == "isotonic" and sample_count < SEGMENT_ISOTONIC_MIN_SAMPLES:
        return (
            "sigmoid",
            f"样本量 {sample_count} < {SEGMENT_ISOTONIC_MIN_SAMPLES}，"
            "isotonic 不可靠，已改用 sigmoid。",
        )
    if text == "auto":
        method = (
            "isotonic"
            if sample_count >= SEGMENT_ISOTONIC_MIN_SAMPLES
            else "sigmoid"
        )
        return method, ""
    return text, ""


def build_calibration_folds(
    labels: Sequence,
    groups: Sequence | None = None,
    random_state: int = RANDOM_STATE,
) -> dict:
    """产生 (train_idx, calib_idx) 切分对，尽量按 source_recording 分组隔离。

    返回：
    - folds：[(train_idx, calib_idx)]，全局索引、已排序；
    - ok：是否具备安全校准条件（folds 非空）；
    - group_isolated：本次切分是否按录音分组隔离；
    - split：录音分组明细（train/calibration 录音名单，供 metadata 断言）；
    - notes：为什么没校准 / 有什么风险的中文说明。
    """
    from sklearn.model_selection import (
        StratifiedGroupKFold,
        StratifiedKFold,
    )

    y = pd.Series(list(labels)).astype(str).reset_index(drop=True)
    index = np.arange(len(y))
    classes = sorted(set(y))
    notes: list[str] = []

    if len(y) == 0 or len(classes) < 2:
        return {
            "folds": [],
            "ok": False,
            "group_isolated": False,
            "split": {"folds": []},
            "notes": ["类别数不足 2，无法做概率校准。"],
        }

    # ---- 分组（source_recording）：决定是否能按录音隔离 ----
    groups_series: pd.Series | None = None
    if groups is not None:
        candidate = pd.Series(list(groups)).astype(str).reset_index(drop=True)
        if len(candidate) == len(y) and candidate.nunique() >= 2:
            groups_series = candidate

    group_isolated = groups_series is not None
    if not group_isolated:
        notes.append(UNGROUPED_CALIBRATION_WARNING)

    min_class = int(y.value_counts().min())
    if min_class < SEGMENT_MIN_CLASS_SAMPLES:
        notes.append(
            f"最小类别样本数 {min_class} < {SEGMENT_MIN_CLASS_SAMPLES}，"
            f"{INSUFFICIENT_CALIBRATION_WARNING}。"
        )
        return {
            "folds": [],
            "ok": False,
            "group_isolated": group_isolated,
            "split": {"folds": []},
            "notes": notes,
        }

    if group_isolated:
        unique_groups = int(groups_series.nunique())
        if unique_groups < SEGMENT_MIN_RECORDINGS_FOR_CALIBRATION:
            notes.append(
                f"独立录音只有 {unique_groups} 条"
                f"（< {SEGMENT_MIN_RECORDINGS_FOR_CALIBRATION}），"
                f"{INSUFFICIENT_CALIBRATION_WARNING}，"
                "置信度保持未校准的模型支持度。"
            )
            return {
                "folds": [],
                "ok": False,
                "group_isolated": True,
                "split": {"folds": []},
                "notes": notes,
            }
    else:
        unique_groups = 0

    n_splits = int(
        max(
            2,
            min(
                SEGMENT_CALIBRATION_FOLDS,
                min_class,
                unique_groups or min_class,
            ),
        )
    )

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    split_folds: list[dict] = []
    try:
        if group_isolated:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=int(random_state),
            )
            iterator = splitter.split(index, y, groups_series)
        else:
            splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=int(random_state),
            )
            iterator = splitter.split(index, y)

        for train_local, calib_local in iterator:
            calib_labels = set(y.iloc[calib_local])
            calib_counts = y.iloc[calib_local].value_counts()
            # 每个校准折必须覆盖全部类别且每类 ≥ SEGMENT_MIN_CLASS_SAMPLES
            if calib_labels != set(classes):
                continue
            if int(calib_counts.min()) < SEGMENT_MIN_CLASS_SAMPLES:
                continue
            train_idx = np.sort(np.asarray(train_local, dtype=int))
            calib_idx = np.sort(np.asarray(calib_local, dtype=int))
            folds.append((train_idx, calib_idx))
            if group_isolated:
                split_folds.append(
                    {
                        "train_recordings": sorted(
                            set(groups_series.iloc[train_idx])
                        ),
                        "calibration_recordings": sorted(
                            set(groups_series.iloc[calib_idx])
                        ),
                    }
                )
    except Exception as error:  # noqa: BLE001 - 切分失败不阻断训练
        notes.append(f"构造校准切分失败（{error}），不做校准。")
        folds = []
        split_folds = []

    if not folds:
        notes.append(
            f"无法构造覆盖全部类别的校准切分，{INSUFFICIENT_CALIBRATION_WARNING}，"
            "绝不伪造校准概率。"
        )
        split_folds = []

    return {
        "folds": folds,
        "ok": bool(folds),
        "group_isolated": group_isolated,
        "split": {"folds": split_folds},
        "notes": notes,
    }


def fit_calibrated_classifier(
    base,
    x_values: pd.DataFrame,
    y_values: Sequence,
    method: str,
    folds: list[tuple[np.ndarray, np.ndarray]] | None = None,
):
    """用 CalibratedClassifierCV 包装基础分类器。

    folds 为 None 时是“已拟合模型后校准”（补校准场景，FrozenEstimator）；
    其余情况使用显式 (train, calib) 索引对——同一录音绝不跨 train/calib 两侧。
    """
    from sklearn.calibration import CalibratedClassifierCV

    if folds:
        return CalibratedClassifierCV(
            estimator=base,
            method=method,
            cv=[(train_idx, calib_idx) for train_idx, calib_idx in folds],
        ).fit(x_values, y_values)

    try:
        from sklearn.frozen import FrozenEstimator
    except ImportError:  # pragma: no cover - 旧版 sklearn 回退
        return CalibratedClassifierCV(
            estimator=base, method=method, cv="prefit"
        ).fit(x_values, y_values)
    return CalibratedClassifierCV(
        estimator=FrozenEstimator(base), method=method
    ).fit(x_values, y_values)


def calibrate_classifier(
    features: pd.DataFrame,
    labels: Sequence,
    groups: Sequence | None = None,
    base=None,
    method: str | None = None,
    random_state: int = RANDOM_STATE,
) -> dict:
    """训练校准分类器；数据不足时返回 classifier=None（绝不伪造）。

    返回 ``{"classifier": CalibratedClassifierCV | None, "info": {...}}``。
    ``base`` 为 None 时用项目默认 RandomForest 作为基础分类器。
    """
    x_values = pd.DataFrame(features).reset_index(drop=True)
    y_values = pd.Series(list(labels)).astype(str).reset_index(drop=True)

    base_estimator = base
    if base_estimator is None:
        from sklearn.ensemble import RandomForestClassifier

        base_estimator = RandomForestClassifier(
            n_estimators=600,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=int(random_state),
            n_jobs=-1,
        )

    folds_info = build_calibration_folds(
        y_values, groups=groups, random_state=random_state
    )
    notes = list(folds_info["notes"])

    class_counts = {
        str(label): int(count)
        for label, count in y_values.value_counts().items()
    }
    info: dict = {
        "is_calibrated": False,
        "calibration_method": "",
        "calibrator": None,
        "sample_count": int(len(y_values)),
        "class_counts": class_counts,
        "class_labels": sorted(class_counts),
        "group_isolated": bool(folds_info["group_isolated"]),
        "recording_count": (
            int(pd.Series(list(groups)).nunique()) if groups is not None else None
        ),
        "fold_count": 0,
        "warnings": notes,
        "calibrated_at": _now(),
    }

    if not folds_info["ok"]:
        return {"classifier": None, "info": info}

    resolved, note = resolve_calibration_method(method, int(len(y_values)))
    if note:
        notes.append(note)

    try:
        calibrated = fit_calibrated_classifier(
            base_estimator,
            x_values,
            y_values,
            method=resolved,
            folds=folds_info["folds"],
        )
    except Exception as error:  # noqa: BLE001 - 校准失败退回未校准
        notes.append(f"校准拟合失败（{error}），置信度保持未校准。")
        return {"classifier": None, "info": info}

    info.update(
        {
            "is_calibrated": True,
            "calibration_method": resolved,
            "calibrator": f"CalibratedClassifierCV(method={resolved})",
            "fold_count": len(folds_info["folds"]),
            "recording_split": folds_info["split"],
        }
    )
    info["warnings"] = notes
    return {"classifier": calibrated, "info": info}


def attach_calibration(
    payload: dict,
    features: pd.DataFrame,
    labels: Sequence,
    groups: Sequence | None = None,
    method: str | None = None,
    random_state: int = RANDOM_STATE,
) -> dict:
    """给**已训练好的两级模型**补挂校准器（不重训、不覆盖原模型）。

    payload：`models/*.joblib` 里的两级模型字典（含 ``acoustic_types`` 段）；
    features/labels：带标签的脉冲特征表（列顺序自动对齐
    ``acoustic_types["feature_columns"]``）；groups：source_recording 分组。

    直接修改并返回 payload；校准失败时 payload 里只多一个
    ``acoustic_types["calibration"]`` 说明（classifier 仍缺省）。
    """
    acoustic = payload.get("acoustic_types")
    if not isinstance(acoustic, dict):
        raise ValueError("payload 缺少 acoustic_types 段，不是两级模型。")

    feature_columns = [str(c) for c in (acoustic.get("feature_columns") or [])]
    if not feature_columns:
        raise ValueError("acoustic_types 缺少 feature_columns，无法对齐特征顺序。")

    x_values = pd.DataFrame(features).copy()
    missing = [c for c in feature_columns if c not in x_values.columns]
    if missing:
        raise ValueError(f"特征表缺少列：{missing}")
    x_values = x_values[feature_columns].reset_index(drop=True)

    base = acoustic.get("temporary_classifier")
    if base is None:
        raise ValueError("acoustic_types 缺少 temporary_classifier，无法补校准。")

    result = calibrate_classifier(
        x_values,
        labels,
        groups=groups,
        base=base,
        method=method,
        random_state=random_state,
    )
    if result["classifier"] is not None:
        acoustic[CALIBRATED_CLASSIFIER_KEY] = result["classifier"]
    acoustic[CALIBRATION_INFO_KEY] = result["info"]
    return payload


def select_classifier(acoustic_payload: dict) -> tuple[object, bool]:
    """取 ⑦/③ 推理用的分类器：优先校准分类器，退回临时分类器。

    返回 (分类器, is_calibrated)。
    """
    if not isinstance(acoustic_payload, dict):
        return None, False
    calibrated = acoustic_payload.get(CALIBRATED_CLASSIFIER_KEY)
    if calibrated is not None and hasattr(calibrated, "predict_proba"):
        return calibrated, True
    return acoustic_payload.get("temporary_classifier"), False


def describe_calibration(acoustic_payload: dict | None) -> dict:
    """读侧描述：校准状态 / 方法 / 置信度性质（供 UI 与 run_metadata）。"""
    if not isinstance(acoustic_payload, dict):
        return {
            "is_calibrated": False,
            "calibration_method": "",
            "calibrator": None,
            "probability_kind": MODEL_SUPPORT,
            "calibration_note": "无模型，置信度不可用。",
            "warnings": [],
        }

    info = acoustic_payload.get(CALIBRATION_INFO_KEY)
    info = dict(info) if isinstance(info, dict) else {}
    calibrated = acoustic_payload.get(CALIBRATED_CLASSIFIER_KEY)
    is_calibrated = calibrated is not None and hasattr(
        calibrated, "predict_proba"
    )

    method = str(info.get("calibration_method") or "")
    if is_calibrated and not method:
        method = "未标注"
    warnings = [str(w) for w in (info.get("warnings") or [])]

    if is_calibrated:
        note = (
            f"置信度为校准概率（CalibratedClassifierCV, method={method}）；"
            "未再做额外的概率校准验证。"
        )
    else:
        note = (
            "置信度未校准：为分类器原始 predict_proba（模型支持度），"
            "不能当“预测正确的概率”读。"
        )
        if not warnings:
            warnings.append(INSUFFICIENT_CALIBRATION_WARNING)

    return {
        "is_calibrated": bool(is_calibrated),
        "calibration_method": method,
        "calibrator": info.get("calibrator"),
        "probability_kind": (
            CALIBRATED_PROBABILITY if is_calibrated else MODEL_SUPPORT
        ),
        "calibration_note": note,
        "warnings": warnings,
    }
