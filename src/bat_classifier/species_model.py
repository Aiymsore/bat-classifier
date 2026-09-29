"""已确认物种模型：用专家确认标签训练的监督物种分类器（⑥ 模型对照专用）。

与 A-D 临时声型模型（KMeans 伪标签）、noise/bat 门控模型完全分开：
- 训练标签只能是 expert_label（review_status == reviewed 且非空），
  绝不使用 cluster_id / A-D 聚类标签 / cluster_inferred_label；
- 模型输出写 model_inferred_label，绝不覆盖 expert_label；
- 概率必须是校准概率（CalibratedClassifierCV），未校准时明确标记
  “未校准模型”，此时只允许显示“模型支持度”；
- 训练 / 校准 / 评估按 source_recording 分组隔离（GroupShuffleSplit /
  StratifiedGroupKFold），同一录音的 pulse 绝不同时出现在不该重合的集合里；
- 数据不足以安全校准时绝不伪造 calibrated probability。

模型保存：models/species/<model_name>/{model.joblib, metadata.json}
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from .config import (
    PSEUDO_MAX_RECALL_DROP,
    PSEUDO_SAMPLE_WEIGHT,
    RANDOM_STATE,
    SPECIES_AVAILABLE_COLUMNS,
    SPECIES_CALIBRATION_FOLDS,
    SPECIES_CALIBRATION_METHOD,
    SPECIES_FEATURE_COLUMNS,
    SPECIES_GROUP_TEST_FRACTION,
    SPECIES_ISOTONIC_MIN_SAMPLES,
    SPECIES_MIN_CLASS_SAMPLES,
    SPECIES_MIN_RECORDINGS_FOR_CALIBRATION,
    SPECIES_MIN_RECORDINGS_FOR_SPLIT,
    SPECIES_MODEL_DIR,
    SPECIES_RF_CLASS_WEIGHT,
    SPECIES_RF_MIN_SAMPLES_LEAF,
    SPECIES_RF_N_ESTIMATORS,
    SPECIES_SPLIT_MAX_ATTEMPTS,
    SPECIES_UNCERTAIN_LABEL,
    SPECIES_UNCERTAIN_THRESHOLD,
    SPECIES_WARN_CLASS_SAMPLES,
)
from .features import numeric_feature_frame

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"
MODEL_VERSION = "1.0"

TRAINING_SOURCE_ALIASES = ("source_recording", "source_file", "source_path")

INSUFFICIENT_CALIBRATION_WARNING = "当前数据不足以进行可靠概率校准"
SINGLE_CLASS_WARNING = "当前数据仅包含一个物种，无法训练普通多类别物种分类器。"

CALIBRATED_PROBABILITY = "calibrated_probability"
MODEL_SUPPORT = "model_support"


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _clean_label(value: object) -> str:
    return str(value if value is not None else "").strip()


def source_recording_series(frame: pd.DataFrame) -> pd.Series:
    """按 source_recording / source_file / source_path 归一出录音分组键。

    同一原始录音的所有 pulse 必须共享同一个键——这是防数据泄漏的分组依据。
    """
    for column in TRAINING_SOURCE_ALIASES:
        if column in frame.columns:
            values = frame[column].map(_clean_label)
            if (values != "").any():
                return values
    raise ValueError(
        "训练数据缺少录音来源列：需要 source_recording / source_file / source_path 之一。"
    )


def make_species_classifier(random_state: int = RANDOM_STATE) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=SPECIES_RF_N_ESTIMATORS,
        min_samples_leaf=SPECIES_RF_MIN_SAMPLES_LEAF,
        max_features="sqrt",
        class_weight=SPECIES_RF_CLASS_WEIGHT,
        random_state=int(random_state),
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# 训练数据收集（专家确认标签）
# ---------------------------------------------------------------------------
def _frame_from_csv(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"训练数据文件为空：{path}") from error
    if frame.empty:
        raise ValueError(f"训练数据文件没有任何行：{path}")
    return frame


def _frame_from_run_dir(path: Path) -> pd.DataFrame:
    """从 ⑥ 任务目录收集专家标注（已导出数据集优先，否则现场合并）。"""
    candidates = [
        path / "tables" / "expert_training_dataset.csv",
        path / "expert_training_dataset.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return _frame_from_csv(candidate)

    tables = path / "tables"
    annotations_path = tables / "expert_annotations.csv"
    representatives_path = tables / "representatives.csv"
    if annotations_path.exists() and representatives_path.exists():
        annotations = _frame_from_csv(annotations_path)
        representatives = _frame_from_csv(representatives_path)
        if annotations.empty or representatives.empty:
            raise ValueError(f"任务目录没有可用的专家标注数据：{path}")
        merged = annotations.merge(
            representatives,
            on="pulse_id",
            how="inner",
            suffixes=("", "_rep"),
        )
        return merged
    raise ValueError(
        f"不是有效的 ⑥ 任务目录（缺少 expert_training_dataset.csv 或标注表）：{path}"
    )


def collect_expert_training_frame(sources: list) -> pd.DataFrame:
    """收集专家确认标签数据（CSV 文件或 ⑥ 任务目录，可多来源合并）。"""
    frames: list[pd.DataFrame] = []
    for source in sources:
        path = Path(source)
        if path.is_dir():
            frames.append(_frame_from_run_dir(path))
        elif path.is_file():
            frames.append(_frame_from_csv(path))
        else:
            raise FileNotFoundError(f"训练数据来源不存在：{path}")
    if not frames:
        raise ValueError("没有选择任何训练数据。")
    merged = pd.concat(frames, ignore_index=True)
    return normalize_training_frame(merged)


def normalize_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """校验并归一专家训练表：只保留 reviewed + 非空 expert_label 的行。"""
    data = frame.copy()
    if "pulse_id" not in data.columns:
        raise ValueError("训练数据缺少必需列：['pulse_id']")
    data["pulse_id"] = data["pulse_id"].map(_clean_label)
    if "expert_label" not in data.columns:
        raise ValueError("训练数据缺少必需列：['expert_label']")

    missing_features = [
        column for column in SPECIES_FEATURE_COLUMNS if column not in data.columns
    ]
    if missing_features:
        raise ValueError(f"训练数据缺少模型需要的声学特征列：{missing_features}")

    data["source_recording"] = source_recording_series(data).to_numpy()

    # 只允许专家确认（reviewed）且标签非空的数据参与训练。
    label = data["expert_label"].map(_clean_label)
    if "review_status" in data.columns:
        status = data["review_status"].map(_clean_label).str.lower()
        allowed = (status == "reviewed") & (label != "")
    else:
        # 外部数据没有 review_status 列时只能凭 expert_label 判断，
        # 在训练 metadata 中明确记录这一点。
        allowed = label != ""
    data = data.loc[allowed].copy()
    data["expert_label"] = data["expert_label"].map(_clean_label)

    if data.empty:
        raise ValueError(
            "没有可用于训练的数据：只允许 review_status == reviewed 且 expert_label 非空的样本。"
        )

    data[SPECIES_FEATURE_COLUMNS] = numeric_feature_frame(
        data, list(SPECIES_FEATURE_COLUMNS)
    )
    valid = data[SPECIES_FEATURE_COLUMNS].notna().all(axis=1)
    dropped = int((~valid).sum())
    if dropped:
        print(f"[物种模型] 剔除特征缺失/非法样本 {dropped} 个。")
    data = data.loc[valid].copy()
    if data.empty:
        raise ValueError("所有样本的声学特征都不完整，无法训练。")
    return data.reset_index(drop=True)


def validate_training_frame(data: pd.DataFrame) -> pd.DataFrame:
    """训练前的最终把关：类别数、每类样本量、录音分组完整性。"""
    counts = data["expert_label"].value_counts()
    if len(counts) < 2:
        raise ValueError(SINGLE_CLASS_WARNING)
    too_few = {
        str(label): int(count)
        for label, count in counts.items()
        if int(count) < SPECIES_MIN_CLASS_SAMPLES
    }
    if too_few:
        raise ValueError(
            f"以下物种样本过少（每类至少 {SPECIES_MIN_CLASS_SAMPLES} 条）：{too_few}"
        )
    groups = source_recording_series(data)
    if groups.nunique() < 2:
        raise ValueError("独立录音只有 1 条，无法按录音分组划分训练/校准/评估数据。")
    return data


def _warn_class_imbalance(data: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    counts = data["expert_label"].value_counts()
    scarce = {
        str(label): int(count)
        for label, count in counts.items()
        if int(count) < SPECIES_WARN_CLASS_SAMPLES
    }
    if scarce:
        notes.append(
            f"类别样本不平衡提醒：{scarce} 样本偏少，评估请重点看 Macro F1 而不是 Accuracy。"
        )
    return notes


# ---------------------------------------------------------------------------
# 录音级分组划分（训练 / 校准 / 评估全部 recording-level 隔离）
# ---------------------------------------------------------------------------
def build_calibration_folds(
    y_values: pd.Series,
    groups: pd.Series,
    classes,
    random_state: int = RANDOM_STATE,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """在样本池内按 source_recording 分组产生 (train, calib) 显式索引对（局部索引）。

    共享逻辑（build_group_splits 与 OOF cross-fitting 共用，绝不复制两套阈值）：
    同一录音绝不跨 train/calib 两侧；每个 fold 的校准侧必须覆盖全部类别且
    每类 ≥ 2 条，训练侧也必须覆盖全部类别（单类基学习器无法产生
    predict_proba 两列），否则该 fold 丢弃（不伪造校准）。索引相对传入的样本池。
    """
    y = y_values.map(_clean_label)
    classes = sorted({_clean_label(value) for value in classes})
    if len(y) == 0 or not classes or groups.nunique() < 2:
        return []
    min_class = int(y.value_counts().min())
    if min_class < 2:
        return []
    n_splits = int(
        max(2, min(SPECIES_CALIBRATION_FOLDS, min_class, int(groups.nunique())))
    )
    calibration_folds: list[tuple[np.ndarray, np.ndarray]] = []
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(random_state),
    )
    for train_local, calib_local in splitter.split(np.zeros(len(y)), y, groups):
        calib_labels = set(y.iloc[calib_local])
        calib_counts = y.iloc[calib_local].value_counts()
        if calib_labels != set(classes):
            continue
        if int(calib_counts.min()) < 2:
            continue
        if set(y.iloc[train_local]) != set(classes):
            continue
        calibration_folds.append(
            (
                np.sort(np.asarray(train_local, dtype=int)),
                np.sort(np.asarray(calib_local, dtype=int)),
            )
        )
    return calibration_folds


def build_group_splits(
    data: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> dict:
    """按 source_recording 分组产生 train / calibration / test 索引。

    返回：
    - fit_idx / test_idx：拟合池与留出评估集（不相交，录音级隔离）；
    - calibration_folds：[(train_idx, calib_idx)]（全局索引），同 fold 内
      训练录音与校准录音绝不重合；
    - calibration_ok：是否具备安全概率校准条件；
    - recording_split：录音分组明细（供测试与 metadata 断言）。
    """
    y = data["expert_label"].map(_clean_label)
    groups = source_recording_series(data)
    classes = sorted(set(y))
    index = np.arange(len(data))

    recording_split: dict = {
        "fit": sorted(set(groups)),
        "test": [],
        "folds": [],
    }
    notes: list[str] = []

    unique_groups = int(groups.nunique())
    test_idx = np.array([], dtype=int)
    fit_idx = index.copy()

    if unique_groups >= SPECIES_MIN_RECORDINGS_FOR_SPLIT:
        found = False
        for attempt in range(SPECIES_SPLIT_MAX_ATTEMPTS):
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=SPECIES_GROUP_TEST_FRACTION,
                random_state=int(random_state) + attempt,
            )
            train_index, holdout_index = next(splitter.split(index, y, groups))
            if set(y.iloc[train_index]) != set(classes):
                continue
            if set(y.iloc[holdout_index]) != set(classes):
                continue
            fit_idx = np.sort(np.asarray(train_index, dtype=int))
            test_idx = np.sort(np.asarray(holdout_index, dtype=int))
            found = True
            break
        if not found:
            notes.append(
                "无法找到覆盖全部物种的录音级留出集，本次没有独立评估集。"
            )
        else:
            recording_split["test"] = sorted(set(groups.iloc[test_idx]))
            recording_split["fit"] = sorted(set(groups.iloc[fit_idx]))
    else:
        notes.append(
            f"独立录音只有 {unique_groups} 条（< {SPECIES_MIN_RECORDINGS_FOR_SPLIT}），"
            "本次没有独立评估集。"
        )

    # ---- 校准 folds：在拟合池内按录音分组（共享逻辑 build_calibration_folds） ----
    calibration_folds: list[tuple[np.ndarray, np.ndarray]] = []
    calibration_ok = unique_groups >= SPECIES_MIN_RECORDINGS_FOR_CALIBRATION
    if calibration_ok:
        pool_y = y.iloc[fit_idx]
        pool_groups = groups.iloc[fit_idx]
        pool_folds = build_calibration_folds(
            pool_y, pool_groups, classes, random_state=random_state
        )
        for train_local, calib_local in pool_folds:
            train_global = np.sort(fit_idx[train_local])
            calib_global = np.sort(fit_idx[calib_local])
            calibration_folds.append((train_global, calib_global))
            recording_split["folds"].append(
                {
                    "train_recordings": sorted(set(pool_groups.iloc[train_local])),
                    "calibration_recordings": sorted(set(pool_groups.iloc[calib_local])),
                }
            )
        if not calibration_folds:
            calibration_ok = False
            recording_split["folds"] = []
    if not calibration_ok:
        notes.append(INSUFFICIENT_CALIBRATION_WARNING + "，输出为未校准的模型支持度。")

    return {
        "fit_idx": fit_idx,
        "test_idx": test_idx,
        "calibration_folds": calibration_folds,
        "calibration_ok": calibration_ok,
        "recording_split": recording_split,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# 评估指标（含概率校准评估）
# ---------------------------------------------------------------------------
def reliability_bins(
    y_true: list[str],
    y_pred: list[str],
    confidences: list[float],
    n_bins: int = 10,
) -> dict:
    """top-label reliability 分箱 + ECE（模型说 0.8 时实际正确率是否 ≈ 0.8）。"""
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    conf = np.asarray(confidences, dtype=float)
    correct = (y_true == y_pred).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict] = []
    ece = 0.0
    total = max(len(conf), 1)
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (conf > lower) & (conf <= upper) if index > 0 else (conf >= lower) & (conf <= upper)
        count = int(mask.sum())
        if count:
            mean_confidence = float(conf[mask].mean())
            accuracy = float(correct[mask].mean())
            ece += abs(accuracy - mean_confidence) * count / total
        else:
            mean_confidence = None
            accuracy = None
        bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    return {"ece": float(ece), "bins": bins, "sample_count": int(len(conf))}


def save_reliability_plot(calibration: dict, output_path: Path) -> bool:
    """保存 reliability / calibration plot；失败只降级为纯数值，不伪造统计。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 - 绘图环境不可用时保留数值即可
        return False

    points = [
        (bin_row["mean_confidence"], bin_row["accuracy"], bin_row["count"])
        for bin_row in calibration.get("bins", [])
        if bin_row.get("count") and bin_row.get("mean_confidence") is not None
    ]
    if not points:
        return False
    try:
        figure, axis = plt.subplots(figsize=(6, 6))
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")
        axis.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            label="model",
        )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel("predicted probability (top label)")
        axis.set_ylabel("empirical accuracy")
        axis.set_title(
            f"Reliability diagram (top-label)\nECE = {calibration.get('ece', float('nan')):.3f}"
        )
        axis.legend()
        figure.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        return True
    except Exception:  # noqa: BLE001
        return False


def evaluate_classifier(
    model,
    x_values: pd.DataFrame,
    y_true: pd.Series,
    class_labels: list[str],
    is_calibrated: bool,
) -> dict:
    """常规分类指标 + 概率指标（log loss / Brier / reliability 分箱）。"""
    predictions = [str(value) for value in model.predict(x_values)]
    truth = [str(value) for value in y_true]
    metrics: dict = {
        "sample_count": int(len(truth)),
        "class_count": int(len(class_labels)),
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_precision": float(
            precision_score(truth, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(truth, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(truth, predictions, average="macro", zero_division=0)
        ),
        "confusion_matrix": {
            "labels": list(class_labels),
            "matrix": confusion_matrix(truth, predictions, labels=class_labels).tolist(),
        },
    }

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(x_values), dtype=float)
        model_classes = [str(value) for value in model.classes_]
        clipped = np.clip(probabilities, 1e-12, 1.0)
        metrics["log_loss"] = float(
            log_loss(
                truth,
                clipped,
                labels=[str(label) for label in class_labels],
            )
        )
        one_hot = np.zeros_like(probabilities)
        for row, label in enumerate(truth):
            if label in model_classes:
                one_hot[row, model_classes.index(label)] = 1.0
        metrics["brier_score"] = float(
            np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
        )
        confidences = probabilities.max(axis=1)
        metrics["calibration"] = reliability_bins(truth, predictions, confidences)
        metrics["probability_kind"] = (
            CALIBRATED_PROBABILITY if is_calibrated else MODEL_SUPPORT
        )
    else:  # pragma: no cover - RandomForest 必有 predict_proba
        metrics["probability_kind"] = "none"
    return metrics


INSUFFICIENT_EVAL_SUPPORT = "INSUFFICIENT_EVAL_SUPPORT"


def classification_table(
    y_true,
    y_pred,
    class_labels: list[str],
) -> dict:
    """逐物种 Support / Precision / Recall / F1 表（评估表 UI 直接消费）。

    - ``support`` = 评估集中专家确认真正属于该物种的 pulse 数；
    - support == 0 的物种：Precision / Recall / F1 输出 ``None`` 并打
      ``INSUFFICIENT_EVAL_SUPPORT``——**区分"模型表现为 0"与"根本没有
      evaluation truth"**，绝不输出误导性的 F1 = 0；
    - ``macro_f1`` 只对 support ≥ 1 的物种等权平均，纳入/排除清单显式列出。
    """
    truth = [str(value) for value in y_true]
    predictions = [str(value) for value in y_pred]
    labels = [str(value) for value in class_labels]

    rows: list[dict] = []
    included: list[str] = []
    excluded: list[str] = []
    f1_values: list[float] = []
    for label in labels:
        support = sum(1 for value in truth if value == label)
        predicted_count = sum(1 for value in predictions if value == label)
        true_positive = sum(
            1 for t, p in zip(truth, predictions) if t == label and p == label
        )
        if support <= 0:
            rows.append(
                {
                    "species": label,
                    "support": 0,
                    "predicted_count": int(predicted_count),
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "status": INSUFFICIENT_EVAL_SUPPORT,
                }
            )
            excluded.append(label)
            continue
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        rows.append(
            {
                "species": label,
                "support": int(support),
                "predicted_count": int(predicted_count),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "status": "",
            }
        )
        included.append(label)
        f1_values.append(float(f1))

    return {
        "rows": rows,
        "macro_f1": float(np.mean(f1_values)) if f1_values else None,
        "macro_f1_classes": included,
        "macro_f1_excluded": excluded,
        "sample_count": int(len(truth)),
    }


def evaluate_payload_on_expert_frame(
    payload: dict,
    frame: pd.DataFrame,
    eval_labels: str = "expert_only",
) -> dict:
    """在冻结 Expert Evaluation Set 上评估一个物种模型。

    真值只认 ``expert_label``；``label_source == pseudo`` 的行在评估前
    过滤（伪标签绝不参与 Precision / Recall / F1 真值）。
    """
    data = frame.copy()
    if "label_source" in data.columns:
        data = data[
            data["label_source"].map(_clean_label).str.lower() != "pseudo"
        ]
    if "expert_label" in data.columns:
        label = data["expert_label"].map(_clean_label)
        data = data[label != ""]
        y_true = data["expert_label"].map(_clean_label)
    else:
        raise ValueError("评估数据缺少 expert_label 列（评估只认真值）。")
    if data.empty:
        raise ValueError("评估集为空，无法计算指标。")

    feature_columns = list(payload.get("feature_columns") or SPECIES_FEATURE_COLUMNS)
    x_values = numeric_feature_frame(data, feature_columns)
    class_labels = [str(value) for value in (payload.get("class_labels") or [])]
    model = payload["model"]
    is_calibrated = bool(payload.get("is_calibrated"))

    metrics = evaluate_classifier(
        model, x_values, y_true, class_labels, is_calibrated
    )
    predictions = [str(value) for value in model.predict(x_values)]
    table = classification_table(y_true, predictions, class_labels)
    metrics["per_class"] = table["rows"]
    metrics["macro_f1_supported"] = table["macro_f1"]
    metrics["macro_f1_classes"] = table["macro_f1_classes"]
    metrics["macro_f1_excluded"] = table["macro_f1_excluded"]
    metrics["eval_labels"] = eval_labels
    return metrics


def evaluate_payload_on_gold_evaluation(
    payload: dict,
    frame: pd.DataFrame,
    gold: dict,
) -> dict:
    """在固定 Gold Evaluation（**冻结真值快照**）上评估。

    - 真值一律来自 gold 快照（recording / pulse / expert truth 全部固定），
      活标注、pseudo、模型结果都改不了考试卷答案；
    - **不读任何缓存指标**：每次调用都真实 re-predict；
    - 缺特征的考题只报告（missing_feature_pulses），不静默吞掉。
    """
    from .expert_split import GoldEvaluationError

    truth_rows = gold.get("truth") or []
    if not truth_rows:
        raise GoldEvaluationError(
            "Gold Evaluation 无冻结真值快照，拒绝评估（考试卷必须固定）。"
        )
    truth = {
        str(row.get("pulse_id", "")): _clean_label(row.get("expert_label", ""))
        for row in truth_rows
    }
    data = frame.copy()
    data["pulse_id"] = data["pulse_id"].astype(str)
    data = data[data["pulse_id"].isin(set(truth))].copy()
    missing = sorted(set(truth) - set(data["pulse_id"]))
    if data.empty:
        raise GoldEvaluationError(
            "特征表中找不到 Gold Evaluation 的任何考题，无法评估。"
        )
    # 真值覆盖：答案只来自快照
    data["expert_label"] = data["pulse_id"].map(truth)
    if "label_source" in data.columns:
        data["label_source"] = "expert"

    metrics = evaluate_payload_on_expert_frame(
        payload, data, eval_labels="expert_only"
    )
    metrics["gold_eval_version"] = int(gold.get("gold_eval_version") or 1)
    metrics["evaluation_recordings"] = list(gold.get("evaluation_recordings") or [])
    metrics["evaluation_pulse_ids"] = sorted(truth)
    metrics["missing_feature_pulses"] = missing[:50]
    metrics["missing_feature_count"] = len(missing)
    metrics["same_evaluation"] = True
    return metrics


def compare_models_on_gold_evaluation(
    old_payload: dict,
    new_payload: dict,
    frame: pd.DataFrame,
    gold: dict,
) -> dict:
    """新旧模型**同时重新**在同一个 Fixed Gold Evaluation 上评估再对比。

    §十一：绝不拿旧模型以前缓存的结果与新模型当前结果直接比较——
    每轮 adoption 前，两个模型都在同一个 Gold Evaluation 上重新跑。
    """
    old_metrics = evaluate_payload_on_gold_evaluation(old_payload, frame, gold)
    new_metrics = evaluate_payload_on_gold_evaluation(new_payload, frame, gold)
    comparison = compare_evaluations(old_metrics, new_metrics)
    version = int(gold.get("gold_eval_version") or 1)
    return {
        "gold_eval_version": version,
        "same_evaluation": True,
        "evaluation_recordings": list(gold.get("evaluation_recordings") or []),
        "evaluation_pulse_ids": old_metrics["evaluation_pulse_ids"],
        "old": old_metrics,
        "new": new_metrics,
        "comparison": comparison,
        "adoptable": bool(comparison.get("adoptable")),
        "notes": (
            f"两个模型都在同一个 Gold Evaluation v{version} 上重新评估"
            "（未使用任何缓存指标）；真值为冻结专家快照。"
        ),
    }


def compare_evaluations(old_metrics: dict, new_metrics: dict) -> dict:
    """新旧模型在同一冻结 Evaluation Set 上的对比（供 UI 与采纳闸门）。"""
    old_rows = {
        str(row.get("species")): row for row in (old_metrics.get("per_class") or [])
    }
    new_rows = {
        str(row.get("species")): row for row in (new_metrics.get("per_class") or [])
    }
    per_class: list[dict] = []
    recall_drops: list[str] = []
    for species in sorted(set(old_rows) | set(new_rows)):
        old_row = old_rows.get(species, {})
        new_row = new_rows.get(species, {})
        old_recall = old_row.get("recall")
        new_recall = new_row.get("recall")
        delta = None
        if old_recall is not None and new_recall is not None:
            delta = float(new_recall) - float(old_recall)
            if delta < -float(PSEUDO_MAX_RECALL_DROP):
                recall_drops.append(species)
        per_class.append(
            {
                "species": species,
                "old_support": old_row.get("support"),
                "new_support": new_row.get("support"),
                "old_f1": old_row.get("f1"),
                "new_f1": new_row.get("f1"),
                "old_recall": old_recall,
                "new_recall": new_recall,
                "recall_delta": delta,
                "old_status": old_row.get("status", ""),
                "new_status": new_row.get("status", ""),
            }
        )
    old_macro = old_metrics.get("macro_f1_supported")
    new_macro = new_metrics.get("macro_f1_supported")
    macro_delta = (
        float(new_macro) - float(old_macro)
        if old_macro is not None and new_macro is not None
        else None
    )
    adopted = bool(
        macro_delta is not None
        and macro_delta >= 0.0
        and not recall_drops
    )
    return {
        "per_class": per_class,
        "old_macro_f1": old_macro,
        "new_macro_f1": new_macro,
        "macro_f1_delta": macro_delta,
        "recall_drop_classes": recall_drops,
        "adoptable": adopted,
        "notes": (
            "指标基于同一冻结 Expert Evaluation Set（专家真值）；"
            "Macro-F1 对 support ≥ 1 的物种等权平均。"
        ),
    }


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def _resolve_calibration_method(
    requested: str,
    sample_count: int,
) -> tuple[str, str]:
    """决定校准方法；isotonic 只在样本量足够时使用（绝不默认用于小样本）。"""
    requested = (requested or SPECIES_CALIBRATION_METHOD).strip().lower()
    if requested not in ("sigmoid", "isotonic", "auto"):
        requested = SPECIES_CALIBRATION_METHOD
    if requested == "isotonic" and sample_count < SPECIES_ISOTONIC_MIN_SAMPLES:
        return (
            "sigmoid",
            f"样本量 {sample_count} < {SPECIES_ISOTONIC_MIN_SAMPLES}，"
            "isotonic 不可靠，已改用 sigmoid。",
        )
    if requested == "auto":
        method = (
            "isotonic"
            if sample_count >= SPECIES_ISOTONIC_MIN_SAMPLES
            else "sigmoid"
        )
        return method, ""
    return requested, ""


def _fit_calibrated(
    base,
    x_values: pd.DataFrame,
    y_values: pd.Series,
    method: str,
    folds: list[tuple[np.ndarray, np.ndarray]] | None,
    sample_weight=None,
):
    """用 CalibratedClassifierCV 包装 RandomForest。

    folds 为 None 时是“已拟合模型后校准”（导入模型场景），其余情况使用
    显式 (train, calib) 索引对——同一录音绝不跨 train/calib 两侧。
    sample_weight 可选（专家 1.0 / 伪标签 0.4）。
    """
    from sklearn.calibration import CalibratedClassifierCV

    weight = {} if sample_weight is None else {"sample_weight": sample_weight}
    if folds is not None:
        return CalibratedClassifierCV(
            estimator=base,
            method=method,
            cv=[(train_idx, calib_idx) for train_idx, calib_idx in folds],
        ).fit(x_values, y_values, **weight)

    try:
        from sklearn.frozen import FrozenEstimator
    except ImportError:  # pragma: no cover - 旧版 sklearn 回退
        return CalibratedClassifierCV(
            estimator=base, method=method, cv="prefit"
        ).fit(x_values, y_values, **weight)
    return CalibratedClassifierCV(
        estimator=FrozenEstimator(base), method=method
    ).fit(x_values, y_values, **weight)


def train_species_model(
    sources: list,
    model_name: str,
    calibration_method: str = SPECIES_CALIBRATION_METHOD,
    uncertain_threshold: float = SPECIES_UNCERTAIN_THRESHOLD,
    random_state: int = RANDOM_STATE,
    output_root: Path | None = None,
    expert_split: dict | str | Path | None = None,
    pseudo_source=None,
) -> dict:
    """专家确认标签 → 录音级 group split → RandomForest → 概率校准 → 评估 → 保存。

    提供 ``expert_split``（冻结划分 dict / expert_split.json 路径）时走
    "冻结 Expert Evaluation Set" 路径（见 ``_train_with_expert_split``）：
    训练/校准只用 Expert Train Pool（+ 带权伪标签），评估只在冻结
    Evaluation Set（专家真值）上进行；``pseudo_source`` 必须配合
    ``expert_split`` 使用（伪标签只进 Train Pool）。
    """
    if expert_split is not None:
        return _train_with_expert_split(
            sources,
            model_name,
            expert_split=expert_split,
            pseudo_source=pseudo_source,
            calibration_method=calibration_method,
            uncertain_threshold=uncertain_threshold,
            random_state=random_state,
            output_root=output_root,
        )
    if pseudo_source is not None:
        raise ValueError(
            "伪标签训练必须提供 expert_split：伪标签只允许进 Expert Train Pool，"
            "且绝不参与评估（请先生成专家数据划分）。"
        )
    raw_frames = collect_expert_training_frame(sources)
    data = validate_training_frame(raw_frames)

    notes = _warn_class_imbalance(data)
    feature_columns = list(SPECIES_FEATURE_COLUMNS)
    x_values = data[feature_columns]
    y_values = data["expert_label"].map(_clean_label)
    class_labels = sorted(set(y_values))

    splits = build_group_splits(data, random_state=random_state)
    notes.extend(splits["notes"])

    fit_idx = splits["fit_idx"]
    test_idx = splits["test_idx"]
    calibration_ok = bool(splits["calibration_ok"] and splits["calibration_folds"])

    method = ""
    if calibration_ok:
        method, note = _resolve_calibration_method(
            calibration_method, len(fit_idx)
        )
        if note:
            notes.append(note)
        # folds 是全局索引对（train_idx / calib_idx 严格按录音分组隔离），
        # 未进入任何 fold 的留出评估行不会被模型看到。
        model = _fit_calibrated(
            make_species_classifier(random_state),
            x_values,
            y_values,
            method=method,
            folds=splits["calibration_folds"],
        )
    else:
        model = make_species_classifier(random_state)
        model.fit(x_values.iloc[fit_idx], y_values.iloc[fit_idx])

    # ---- 评估（录音级留出集） ----
    evaluation: dict | None = None
    if len(test_idx):
        evaluation = evaluate_classifier(
            model,
            x_values.iloc[test_idx],
            y_values.iloc[test_idx],
            class_labels,
            is_calibrated=calibration_ok,
        )
    else:
        notes.append("没有独立评估集，本次不报告验证指标。")

    class_counts = {
        str(label): int(count) for label, count in y_values.value_counts().items()
    }
    metadata = {
        "model_type": "RandomForestClassifier",
        "calibration_method": method if calibration_ok else "",
        "is_calibrated": bool(calibration_ok),
        "feature_columns": feature_columns,
        "class_labels": class_labels,
        "training_sample_count": int(len(data)),
        "recording_count": int(source_recording_series(data).nunique()),
        "class_counts": class_counts,
        "evaluation": {
            "macro_f1": evaluation["macro_f1"] if evaluation else None,
            "accuracy": evaluation["accuracy"] if evaluation else None,
        },
        "trained_at": _now(),
        "model_version": MODEL_VERSION,
        "random_state": int(random_state),
        "uncertain_threshold": float(uncertain_threshold),
        "split": {
            "fit_recordings": splits["recording_split"]["fit"],
            "test_recordings": splits["recording_split"]["test"],
            "calibration_folds": splits["recording_split"]["folds"],
        },
        "notes": notes,
        "warning": (
            "model_inferred_label 只是模型推断，绝不覆盖 expert_label；"
            "Cluster ≠ 物种，多个 Cluster 可以被判断为同一物种。"
        ),
    }
    if "review_status" not in data.columns:
        metadata["notes"].append(
            "训练来源缺少 review_status 列，仅按 expert_label 非空筛选。"
        )

    payload = {
        "model": model,
        "calibrator": f"CalibratedClassifierCV(method={method})" if calibration_ok else None,
        "feature_columns": feature_columns,
        "class_labels": class_labels,
        "is_calibrated": bool(calibration_ok),
        "calibration_method": method if calibration_ok else "",
        "model_type": "RandomForestClassifier",
        "model_version": MODEL_VERSION,
        "trained_at": metadata["trained_at"],
        "uncertainty_threshold": float(uncertain_threshold),
        "training_metadata": metadata,
        "evaluation_metrics": evaluation,
        "training_pulse_ids": [
            str(value) for value in data.get("pulse_id", pd.Series(dtype=str))
        ],
        "training_audio_hashes": sorted(
            {
                str(value)
                for value in data.get("audio_hash", pd.Series(dtype=str))
                if str(value).strip()
            }
        ),
        "training_recordings": sorted(
            set(source_recording_series(data).map(_clean_label))
        ),
        "is_imported": False,
    }

    model_dir = save_species_model(payload, model_name, output_root=output_root)
    if evaluation and evaluation.get("calibration"):
        plot_path = model_dir / "calibration_reliability.png"
        if save_reliability_plot(evaluation["calibration"], plot_path):
            evaluation["reliability_plot"] = plot_path.name

    print(f"[物种模型] 训练完成：{len(data)} 个样本，{len(class_labels)} 个物种，"
          f"{metadata['recording_count']} 条录音。")
    print(f"[物种模型] 概率校准：{'已启用 (' + method + ')' if calibration_ok else '未启用（' + INSUFFICIENT_CALIBRATION_WARNING + '）'}")
    for note in notes:
        print(f"[物种模型] {note}")
    if evaluation:
        print(f"[物种模型] 验证 Macro F1：{evaluation['macro_f1']:.3f}，"
              f"Accuracy：{evaluation['accuracy']:.3f}")
    print(f"[物种模型] 已保存：{model_dir}")

    return {
        "model_dir": str(model_dir),
        "payload": payload,
        "metadata": metadata,
        "evaluation": evaluation,
        "notes": notes,
        "is_calibrated": bool(calibration_ok),
    }


# ---------------------------------------------------------------------------
# 冻结 Expert Evaluation Set 路径（⑥ 专家数据划分 + 伪标签反哺）
# ---------------------------------------------------------------------------
def _resolve_expert_split(expert_split) -> dict:
    if isinstance(expert_split, dict):
        return expert_split
    path = Path(expert_split)
    if not path.exists():
        raise FileNotFoundError(f"专家数据划分文件不存在：{path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _collect_pseudo_frame(pseudo_source) -> pd.DataFrame:
    """载入伪标签训练集（DataFrame / CSV 路径 / 路径列表）。"""
    if pseudo_source is None:
        return pd.DataFrame()
    if isinstance(pseudo_source, pd.DataFrame):
        frame = pseudo_source.copy()
    else:
        items = (
            list(pseudo_source)
            if isinstance(pseudo_source, (list, tuple))
            else [pseudo_source]
        )
        frames = []
        for item in items:
            path = Path(item)
            try:
                frames.append(pd.read_csv(path, encoding="utf-8-sig"))
            except UnicodeDecodeError:
                frames.append(pd.read_csv(path))
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if frame.empty:
        return frame
    if "pseudo_label" not in frame.columns:
        raise ValueError("伪标签训练集缺少 pseudo_label 列。")
    frame["pseudo_label"] = frame["pseudo_label"].map(_clean_label)
    frame = frame[frame["pseudo_label"] != ""].reset_index(drop=True)
    if "source_recording" not in frame.columns:
        frame["source_recording"] = source_recording_series(frame).to_numpy()
    return frame


def _train_with_expert_split(
    sources: list,
    model_name: str,
    expert_split: dict | str | Path,
    pseudo_source=None,
    calibration_method: str = SPECIES_CALIBRATION_METHOD,
    uncertain_threshold: float = SPECIES_UNCERTAIN_THRESHOLD,
    random_state: int = RANDOM_STATE,
    output_root: Path | None = None,
) -> dict:
    """冻结 Expert Evaluation Set 路径。

    - **训练 / 校准**：只用 Expert Train Pool（+ 带权伪标签；
      校准 fold 的校准侧永远只有专家样本）；
    - **评估**：只在冻结 Evaluation Set（专家真值）上计算
      Support / Precision / Recall / F1 / Macro-F1（真值绝不含伪标签）；
    - **EVAL_CONTAMINATED 断言**：评估录音出现在训练 / 校准 / 伪标签
      数据中时立即终止并列出录音 ID；
    - 评估录音上冻结后新增的标注 parked：不参与训练也不进冻结评估集。
    """
    from .expert_split import (
        apply_expert_split,
        assert_eval_isolation,
    )

    split = _resolve_expert_split(expert_split)
    raw_frames = collect_expert_training_frame(sources)
    parts = apply_expert_split(raw_frames, split)
    train_data = parts["train"]
    eval_data = parts["evaluation"]
    parked = parts["parked"]

    notes: list[str] = []
    if not parked.empty:
        notes.append(
            f"parked {len(parked)} 条：Evaluation 录音上冻结后新增的标注，"
            "录音隔离优先——不参与训练，也不计入冻结评估集（想启用请重新生成划分）。"
        )

    eval_recordings = [
        _clean_label(value) for value in (split.get("evaluation_recordings") or [])
    ]
    assert_eval_isolation(
        eval_recordings,
        source_recording_series(train_data),
        "Species Model 训练集（Expert Train Pool）",
    )
    pseudo = _collect_pseudo_frame(pseudo_source)
    if not pseudo.empty:
        assert_eval_isolation(
            eval_recordings,
            source_recording_series(pseudo),
            "伪标签训练集",
        )
    if eval_data.empty:
        raise ValueError(
            "冻结 Evaluation Set 为空，无法计算最终评估指标"
            "（请先生成专家数据划分）。"
        )
    # §四：Evaluation 只允许真正人工专家真值——pseudo / 模型结果一律拒绝
    from .expert_split import assert_no_pseudo_labels

    assert_no_pseudo_labels(eval_data, "冻结 Evaluation Set（训练前校验）")

    train_data = validate_training_frame(train_data)
    notes.extend(_warn_class_imbalance(train_data))

    feature_columns = list(SPECIES_FEATURE_COLUMNS)
    class_labels = sorted(set(train_data["expert_label"].map(_clean_label)))
    eval_labels = sorted(set(eval_data["expert_label"].map(_clean_label)))
    unknown_eval = [label for label in eval_labels if label not in class_labels]
    if unknown_eval:
        raise ValueError(
            f"Evaluation Set 出现 Train Pool 中不存在的物种：{unknown_eval}"
            "（划分不完整，拒绝训练以免指标失真）。"
        )

    x_expert = numeric_feature_frame(train_data, feature_columns)
    y_expert = train_data["expert_label"].map(_clean_label)
    weights = [1.0] * len(train_data)

    if not pseudo.empty:
        missing = [
            column for column in feature_columns if column not in pseudo.columns
        ]
        if missing:
            raise ValueError(f"伪标签训练集缺少特征列：{missing}")
        unknown_pseudo = sorted(
            set(pseudo["pseudo_label"]) - set(class_labels)
        )
        if unknown_pseudo:
            raise ValueError(
                f"伪标签含 Train Pool 之外的物种：{unknown_pseudo}（拒绝训练）。"
            )
        x_pseudo = numeric_feature_frame(pseudo, feature_columns)
        valid_pseudo = x_pseudo.notna().all(axis=1)
        pseudo = pseudo.loc[valid_pseudo].reset_index(drop=True)
        x_pseudo = x_pseudo.loc[valid_pseudo]
        x_all = pd.concat([x_expert, x_pseudo], ignore_index=True)
        y_all = pd.concat(
            [y_expert, pseudo["pseudo_label"].map(_clean_label)], ignore_index=True
        )
        weights = weights + [float(PSEUDO_SAMPLE_WEIGHT)] * len(pseudo)
    else:
        x_all = x_expert
        y_all = y_expert

    groups_expert = source_recording_series(train_data).map(_clean_label)
    calibration_ok = (
        int(groups_expert.nunique()) >= SPECIES_MIN_RECORDINGS_FOR_CALIBRATION
    )
    method = ""
    model = None
    if calibration_ok:
        pool_folds = build_calibration_folds(
            y_expert, groups_expert, class_labels, random_state=random_state
        )
        if pool_folds:
            method, note = _resolve_calibration_method(
                calibration_method, len(train_data)
            )
            if note:
                notes.append(note)
            # 校准 fold：train 侧 = 专家 + 同录音组的伪标签；calib 侧只专家。
            pseudo_groups = (
                source_recording_series(pseudo).map(_clean_label)
                if not pseudo.empty
                else pd.Series(dtype=str)
            )
            n_expert = len(train_data)
            aug_folds: list[tuple[np.ndarray, np.ndarray]] = []
            for train_local, calib_local in pool_folds:
                fold_train_recordings = set(groups_expert.iloc[train_local])
                extra = [
                    n_expert + index
                    for index, value in enumerate(pseudo_groups)
                    if value in fold_train_recordings
                ]
                train_aug = np.sort(
                    np.concatenate(
                        [np.asarray(train_local, dtype=int), np.asarray(extra, dtype=int)]
                    )
                )
                aug_folds.append((train_aug, np.asarray(calib_local, dtype=int)))
            model = _fit_calibrated(
                make_species_classifier(random_state),
                x_all,
                y_all,
                method=method,
                folds=aug_folds,
                sample_weight=np.asarray(weights, dtype=float),
            )
        else:
            calibration_ok = False
    if model is None:
        model = make_species_classifier(random_state)
        model.fit(x_all, y_all, sample_weight=np.asarray(weights, dtype=float))

    # ---- 评估：固定 Gold Evaluation（冻结真值快照，专家-only） ----
    table_labels = sorted(set(class_labels) | set(eval_labels))
    eval_payload = {
        "model": model,
        "feature_columns": feature_columns,
        "class_labels": table_labels,
        "is_calibrated": bool(calibration_ok),
    }
    from .expert_split import gold_record_from_split

    gold = gold_record_from_split(split)
    if gold["truth"]:
        evaluation = evaluate_payload_on_gold_evaluation(
            eval_payload, eval_data, gold
        )
    else:
        # 旧版划分没有真值快照：退回活标注过滤路径并明确记录
        evaluation = evaluate_payload_on_expert_frame(
            eval_payload, eval_data, eval_labels="expert_only"
        )
        notes.append(
            "旧版划分缺少 evaluation_truth 快照：评估真值取自活标注，"
            "建议重新生成专家数据划分以冻结考试卷。"
        )
    evaluation["eval_split_version"] = int(split.get("split_version") or 1)
    evaluation["evaluation_recordings"] = eval_recordings

    recording_split = {
        "fit": sorted(set(groups_expert)),
        "test": eval_recordings,
        "folds": [],
    }
    class_counts = {
        str(label): int(count) for label, count in y_expert.value_counts().items()
    }
    metadata = {
        "model_type": "RandomForestClassifier",
        "calibration_method": method if calibration_ok else "",
        "is_calibrated": bool(calibration_ok),
        "feature_columns": feature_columns,
        "class_labels": class_labels,
        "training_sample_count": int(len(train_data)),
        "pseudo_sample_count": int(len(pseudo)),
        "recording_count": int(groups_expert.nunique()),
        "class_counts": class_counts,
        "evaluation": {
            "macro_f1": evaluation.get("macro_f1"),
            "macro_f1_supported": evaluation.get("macro_f1_supported"),
            "accuracy": evaluation.get("accuracy"),
            "eval_labels": "expert_only",
            "split_version": int(split.get("split_version") or 1),
        },
        "trained_at": _now(),
        "model_version": MODEL_VERSION,
        "random_state": int(random_state),
        "uncertain_threshold": float(uncertain_threshold),
        "split": recording_split,
        "expert_split": {
            "split_version": int(split.get("split_version") or 1),
            "status": str(split.get("status") or ""),
            "frozen": True,
            "train_recordings": recording_split["fit"],
            "evaluation_recordings": eval_recordings,
        },
        "training_labels": {
            "expert": int(len(train_data)),
            "pseudo": int(len(pseudo)),
            "pseudo_weight": float(PSEUDO_SAMPLE_WEIGHT),
        },
        "eval_labels": "expert_only",
        "notes": notes,
        "warning": (
            "model_inferred_label 只是模型推断，绝不覆盖 expert_label；"
            "Evaluation Set 是冻结考试卷，绝不参与训练/校准/伪标签传播。"
        ),
    }
    payload = {
        "model": model,
        "calibrator": f"CalibratedClassifierCV(method={method})" if calibration_ok else None,
        "feature_columns": feature_columns,
        "class_labels": class_labels,
        "is_calibrated": bool(calibration_ok),
        "calibration_method": method if calibration_ok else "",
        "model_type": "RandomForestClassifier",
        "model_version": MODEL_VERSION,
        "trained_at": metadata["trained_at"],
        "uncertainty_threshold": float(uncertain_threshold),
        "training_metadata": metadata,
        "evaluation_metrics": evaluation,
        "training_pulse_ids": [
            str(value)
            for value in list(train_data.get("pulse_id", pd.Series(dtype=str)))
            + list(pseudo.get("pulse_id", pd.Series(dtype=str)))
        ],
        "training_audio_hashes": sorted(
            {
                str(value)
                for value in list(
                    train_data.get("audio_hash", pd.Series(dtype=str))
                )
                + list(pseudo.get("audio_hash", pd.Series(dtype=str)))
                if str(value).strip()
            }
        ),
        "training_recordings": sorted(
            set(groups_expert) | set(source_recording_series(pseudo).map(_clean_label))
            if not pseudo.empty
            else set(groups_expert)
        ),
        "is_imported": False,
    }

    model_dir = save_species_model(payload, model_name, output_root=output_root)
    try:
        table_frame = pd.DataFrame(evaluation.get("per_class") or [])
        if not table_frame.empty:
            table_frame.to_csv(
                model_dir / "expert_evaluation.csv", index=False, encoding="utf-8-sig"
            )
            evaluation["evaluation_table_csv"] = str(
                model_dir / "expert_evaluation.csv"
            )
    except Exception:  # noqa: BLE001 - 评估表落盘失败不阻塞训练结果
        pass
    if evaluation and evaluation.get("calibration"):
        plot_path = model_dir / "calibration_reliability.png"
        if save_reliability_plot(evaluation["calibration"], plot_path):
            evaluation["reliability_plot"] = plot_path.name

    print(
        f"[物种模型] 训练完成（冻结评估集路径）：专家 {len(train_data)} + "
        f"伪标签 {len(pseudo)}（权重 {PSEUDO_SAMPLE_WEIGHT}），"
        f"{len(class_labels)} 个物种，Train 录音 {groups_expert.nunique()} 条。"
    )
    print(
        f"[物种模型] 评估：冻结 Evaluation Set（split_version="
        f"{split.get('split_version')}）{len(eval_data)} 条，"
        f"Macro-F1（support≥1）="
        f"{evaluation.get('macro_f1_supported') if evaluation.get('macro_f1_supported') is not None else 'N/A'}"
    )
    for note in notes:
        print(f"[物种模型] {note}")

    return {
        "model_dir": str(model_dir),
        "payload": payload,
        "metadata": metadata,
        "evaluation": evaluation,
        "notes": notes,
        "is_calibrated": bool(calibration_ok),
    }


# ---------------------------------------------------------------------------
# 保存 / 加载 / 导入
# ---------------------------------------------------------------------------
def species_model_dir(model_name: str, output_root: Path | None = None) -> Path:
    root = Path(output_root) if output_root else SPECIES_MODEL_DIR
    name = _clean_label(model_name) or datetime.now().strftime("species_%Y%m%d_%H%M%S")
    return root / name


def save_species_model(
    payload: dict,
    model_name: str,
    output_root: Path | None = None,
) -> Path:
    model_dir = species_model_dir(model_name, output_root=output_root)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, model_dir / MODEL_FILENAME)

    metadata = dict(payload.get("training_metadata") or {})
    evaluation_metrics = payload.get("evaluation_metrics") or {}
    metadata.update(
        {
            "model_name": model_dir.name,
            "is_calibrated": bool(payload.get("is_calibrated")),
            "calibration_method": str(payload.get("calibration_method") or ""),
            "is_imported": bool(payload.get("is_imported", False)),
            "class_labels": list(payload.get("class_labels") or []),
            "feature_columns": list(payload.get("feature_columns") or []),
            "evaluation": {
                "macro_f1": evaluation_metrics.get("macro_f1"),
                "accuracy": evaluation_metrics.get("accuracy"),
            },
        }
    )
    (model_dir / METADATA_FILENAME).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return model_dir


def validate_imported_payload(
    payload: object,
    available_columns: list[str] | None = None,
) -> dict:
    """导入/加载校验：feature_columns、类别列表、模型类型、是否校准。

    特征与当前流水线不一致时必须拒绝并列出具体缺失列，绝不静默改序或补值。
    """
    if not isinstance(payload, dict):
        raise ValueError("模型文件格式不正确：不是可识别的模型包（dict）。")

    if "temporary_classifier" in payload or "raw_cluster_to_type" in payload:
        raise ValueError(
            "这是 A-D 临时声型（KMeans 伪标签）模型，不是专家确认物种模型，拒绝导入。"
        )

    model = payload.get("model") or payload.get("classifier") or payload.get(
        "calibrated_classifier"
    )
    if model is None:
        raise ValueError("模型文件缺少分类器对象（model）。")
    if not hasattr(model, "predict_proba"):
        raise ValueError("模型不支持 predict_proba，无法输出概率。")

    feature_columns = payload.get("feature_columns")
    if not feature_columns:
        raise ValueError("模型缺少 feature_columns，无法验证特征一致性。")
    feature_columns = [str(column) for column in feature_columns]

    class_labels = payload.get("class_labels") or payload.get("classes")
    if not class_labels:
        class_labels = [str(value) for value in getattr(model, "classes_", [])]
    class_labels = [str(label) for label in class_labels]
    if not class_labels:
        raise ValueError("模型缺少 class_labels / classes，无法解释预测输出。")

    available = [str(column) for column in (available_columns or SPECIES_AVAILABLE_COLUMNS)]
    missing = [column for column in feature_columns if column not in available]
    if missing:
        raise ValueError(
            "模型要求的特征与当前流水线提取的特征不一致，缺失列："
            f"{missing}（当前可用列：{available}）。拒绝运行，绝不静默调整列顺序或补值。"
        )

    is_calibrated = bool(payload.get("is_calibrated", False))
    if is_calibrated and not payload.get("calibration_method"):
        raise ValueError("模型声称已校准但缺少 calibration_method，信息不可信。")
    if not is_calibrated:
        # 导入的老模型可能没写 is_calibrated，但对象本身是 CalibratedClassifierCV。
        from sklearn.calibration import CalibratedClassifierCV

        if isinstance(model, CalibratedClassifierCV):
            is_calibrated = True

    payload["feature_columns"] = feature_columns
    payload["class_labels"] = class_labels
    payload["is_calibrated"] = is_calibrated
    payload.setdefault("model_type", type(model).__name__)
    return payload


def load_species_model(path) -> dict:
    """加载模型包：接受 models/species/<name>/ 目录或 model.joblib 文件。"""
    model_path = Path(path)
    if model_path.is_dir():
        model_path = model_path / MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(f"模型不可用：{model_path} 不存在。")
    try:
        payload = joblib.load(model_path)
    except Exception as error:  # noqa: BLE001
        raise ValueError(f"模型文件无法读取：{type(error).__name__}: {error}") from error
    payload = validate_imported_payload(payload)
    payload.setdefault("source_path", str(model_path))
    return payload


def import_species_model(
    source_path,
    model_name: str | None = None,
    output_root: Path | None = None,
) -> dict:
    """导入现成监督模型到 models/species/<name>/（校验不通过直接拒绝）。"""
    source = Path(source_path)
    if source.is_dir():
        source = source / MODEL_FILENAME
    if not source.exists():
        raise FileNotFoundError(f"模型文件不存在：{source}")

    payload = load_species_model(source)
    payload["is_imported"] = True
    payload["source_path"] = str(source)

    name = _clean_label(model_name) or source.parent.name or source.stem
    model_dir = save_species_model(payload, name, output_root=output_root)
    print(f"[物种模型] 已导入：{source} → {model_dir}")
    print(
        f"[物种模型] 校准状态：{'已校准 (' + str(payload['calibration_method']) + ')' if payload['is_calibrated'] else '未校准模型（只能显示模型支持度）'}"
    )
    return {"model_dir": str(model_dir), "payload": payload}


def describe_species_model(payload: dict | None, path: str = "") -> dict:
    """UI 显示用的模型状态描述（含“未校准模型”等明确提示）。"""
    if not payload:
        return {
            "loaded": False,
            "name": "",
            "path": path,
            "warning": "未加载模型。",
        }
    metadata = payload.get("training_metadata") or {}
    evaluation = payload.get("evaluation_metrics") or {}
    is_calibrated = bool(payload.get("is_calibrated"))
    return {
        "loaded": True,
        "name": str(metadata.get("model_name") or payload.get("model_name") or ""),
        "path": path or str(payload.get("source_path") or ""),
        "model_type": str(payload.get("model_type") or "RandomForestClassifier"),
        "class_labels": list(payload.get("class_labels") or []),
        "class_count": len(payload.get("class_labels") or []),
        "feature_columns": list(payload.get("feature_columns") or []),
        "training_sample_count": int(metadata.get("training_sample_count") or 0),
        "recording_count": int(metadata.get("recording_count") or 0),
        "is_calibrated": is_calibrated,
        "calibration_method": str(payload.get("calibration_method") or ""),
        "probability_kind": CALIBRATED_PROBABILITY if is_calibrated else MODEL_SUPPORT,
        "macro_f1": evaluation.get("macro_f1"),
        "accuracy": evaluation.get("accuracy"),
        "is_imported": bool(payload.get("is_imported", False)),
        "uncertain_threshold": float(
            payload.get("uncertainty_threshold", SPECIES_UNCERTAIN_THRESHOLD)
        ),
        "calibration_note": (
            "显示的是 calibrated probability（校准后概率）。"
            if is_calibrated
            else "未校准模型：只能显示模型支持度（raw model support），不是校准概率。"
        ),
    }


def model_status_text(description: dict) -> str:
    """把 describe_species_model 的结果渲染成 UI 文本块。"""
    if not description.get("loaded"):
        return f"当前模型：未加载\n{description.get('warning', '')}".strip()
    probability_text = (
        "已启用（校准概率）"
        if description.get("is_calibrated")
        else "未启用（未校准模型，仅模型支持度）"
    )
    calibration_method = description.get("calibration_method") or "—"
    lines = [
        f"当前模型：{description.get('name') or '（未命名）'}",
        f"路径：{description.get('path', '')}",
        f"类型：{description.get('model_type')} + Calibration",
        f"物种数：{description.get('class_count')}"
        f"（{'、'.join(description.get('class_labels', []))}）",
        f"训练样本：{description.get('training_sample_count')}"
        f"（来自 {description.get('recording_count')} 条录音）",
        f"Calibration：{probability_text}"
        + (f"（method={calibration_method}）" if description.get("is_calibrated") else ""),
    ]
    if description.get("macro_f1") is not None:
        lines.append(
            f"验证 Macro F1：{float(description['macro_f1']):.3f} ｜ "
            f"Accuracy：{float(description.get('accuracy') or 0.0):.3f}"
        )
    else:
        lines.append("验证 Macro F1：未评估")
    lines.append(
        f"uncertain 阈值：{float(description.get('uncertain_threshold', SPECIES_UNCERTAIN_THRESHOLD)):.2f}"
    )
    if description.get("is_imported"):
        lines.append("来源：导入的现成模型")
    lines.append(str(description.get("calibration_note", "")))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 用标注数据校准已有（未校准）模型
# ---------------------------------------------------------------------------
def calibrate_species_model_with_labels(
    payload: dict,
    sources: list,
    calibration_method: str = SPECIES_CALIBRATION_METHOD,
    output_root: Path | None = None,
) -> dict:
    """用真实专家标签为导入的 raw RandomForest 增加概率校准。

    校准集按 source_recording 分组提供（与模型原训练数据可能不同来源）；
    校准前后都不改动 base 分类器的类别与特征。
    """
    payload = validate_imported_payload(payload)
    if payload.get("is_calibrated"):
        raise ValueError("模型已经校准，无需重复校准。")

    data = validate_training_frame(normalize_training_frame(collect_expert_training_frame(sources)))
    feature_columns = list(payload["feature_columns"])
    missing = [column for column in feature_columns if column not in data.columns]
    if missing:
        raise ValueError(f"校准数据缺少模型需要的特征列：{missing}")

    x_values = numeric_feature_frame(data, feature_columns)
    y_values = data["expert_label"].map(_clean_label)

    known_classes = set(payload["class_labels"])
    unknown = sorted(set(y_values) - known_classes)
    if unknown:
        raise ValueError(f"校准数据包含模型未知的物种：{unknown}")
    if y_values.nunique() < 2:
        raise ValueError("校准数据只包含一个物种，无法进行概率校准。")
    if source_recording_series(data).nunique() < 2:
        raise ValueError("校准数据的独立录音不足 2 条，无法安全校准。")

    method, note = _resolve_calibration_method(calibration_method, len(data))
    base = payload["model"]
    try:
        calibrated = _fit_calibrated(base, x_values, y_values, method=method, folds=None)
    except Exception as error:  # noqa: BLE001
        raise ValueError(f"概率校准失败：{type(error).__name__}: {error}") from error

    new_payload = dict(payload)
    new_payload["model"] = calibrated
    new_payload["calibrator"] = f"CalibratedClassifierCV(method={method})"
    new_payload["is_calibrated"] = True
    new_payload["calibration_method"] = method
    metadata = dict(payload.get("training_metadata") or {})
    metadata.setdefault("notes", [])
    metadata["notes"] = list(metadata.get("notes") or []) + [
        f"使用外部专家标注数据完成概率校准（method={method}，"
        f"样本 {len(data)}，录音 {source_recording_series(data).nunique()} 条）。"
    ]
    if note:
        metadata["notes"].append(note)
    metadata["calibrated_at"] = _now()
    new_payload["training_metadata"] = metadata

    model_dir = save_species_model(
        new_payload,
        f"{metadata.get('model_name') or payload.get('model_type', 'model')}_calibrated",
        output_root=output_root,
    )
    print(f"[物种模型] 已用标注数据校准（{method}）并保存：{model_dir}")
    return {"model_dir": str(model_dir), "payload": new_payload}


# ---------------------------------------------------------------------------
# 推理：pulse → model_inferred_label + 概率（绝不写回 expert_label）
# ---------------------------------------------------------------------------
def _raw_base_estimator(model):
    """拿到未校准的底层分类器（用于诊断列 raw_probability），失败返回 None。"""
    try:
        calibrated_classifiers = getattr(model, "calibrated_classifiers_", None)
        if calibrated_classifiers:
            estimator = getattr(calibrated_classifiers[0], "estimator", None)
            if estimator is None:
                estimator = getattr(calibrated_classifiers[0], "base_estimator", None)
            return estimator
    except Exception:  # noqa: BLE001 - 仅诊断用途
        return None
    return model


def predict_pulse_frame(
    payload: dict,
    frame: pd.DataFrame,
    uncertain_threshold: float | None = None,
) -> pd.DataFrame:
    """对 pulse 特征表预测物种与概率。

    - 特征严格按 payload["feature_columns"] 顺序，缺失列直接报错并列出列名；
    - max calibrated probability < threshold → predicted = uncertain（不硬分）；
    - calibrated_probability 仅在校准模型下填写；未校准模型只给 raw_probability
      （模型支持度），UI 不得称其为校准概率。
    """
    payload = validate_imported_payload(payload)
    feature_columns = list(payload["feature_columns"])
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            "当前数据的特征与模型要求不一致，缺失列："
            f"{missing}。拒绝运行，绝不静默调整列顺序或补值。"
        )

    threshold = float(
        uncertain_threshold
        if uncertain_threshold is not None
        else payload.get("uncertainty_threshold", SPECIES_UNCERTAIN_THRESHOLD)
    )
    is_calibrated = bool(payload.get("is_calibrated"))
    model = payload["model"]

    result = frame.copy()
    empty = len(result) == 0
    if empty:
        for column in (
            "model_inferred_label",
            "predicted_label",
            "calibrated_probability",
            "raw_probability",
            "probability_kind",
            "all_class_probabilities",
            "is_uncertain",
        ):
            result[column] = []
        return result

    features = numeric_feature_frame(result, feature_columns)
    valid_mask = features.notna().all(axis=1)

    labels = np.full(len(result), "", dtype=object)
    calibrated_probability = np.full(len(result), np.nan, dtype=float)
    raw_probability = np.full(len(result), np.nan, dtype=float)
    probability_json = np.full(len(result), "", dtype=object)
    is_uncertain = np.ones(len(result), dtype=bool)

    if valid_mask.any():
        x_valid = features.loc[valid_mask, feature_columns]
        probabilities = np.asarray(model.predict_proba(x_valid), dtype=float)
        model_classes = [str(value) for value in model.classes_]
        max_index = probabilities.argmax(axis=1)
        max_probability = probabilities[np.arange(len(probabilities)), max_index]
        predicted = np.asarray(
            [model_classes[index] for index in max_index], dtype=object
        )
        uncertain = max_probability < threshold
        predicted = np.where(uncertain, SPECIES_UNCERTAIN_LABEL, predicted)

        positions = np.flatnonzero(valid_mask.to_numpy())
        labels[positions] = predicted
        is_uncertain[positions] = uncertain
        if is_calibrated:
            calibrated_probability[positions] = max_probability
        probability_json[positions] = [
            json.dumps(
                {cls: float(probabilities[row, col]) for col, cls in enumerate(model_classes)},
                ensure_ascii=False,
            )
            for row in range(len(probabilities))
        ]

        base = _raw_base_estimator(model)
        if base is not None and base is not model:
            try:
                raw_probabilities = np.asarray(base.predict_proba(x_valid), dtype=float)
                raw_classes = [str(value) for value in base.classes_]
                raw_max = np.asarray(
                    [
                        raw_probabilities[row, raw_classes.index(predicted[row])]
                        if predicted[row] in raw_classes
                        else raw_probabilities[row].max()
                        for row in range(len(raw_probabilities))
                    ],
                    dtype=float,
                )
                raw_probability[positions] = raw_max
            except Exception:  # noqa: BLE001
                pass
        elif not is_calibrated:
            raw_probability[positions] = max_probability

    result["model_inferred_label"] = labels
    result["predicted_label"] = labels
    result["calibrated_probability"] = calibrated_probability
    result["raw_probability"] = raw_probability
    result["probability_kind"] = (
        CALIBRATED_PROBABILITY if is_calibrated else MODEL_SUPPORT
    )
    result["all_class_probabilities"] = probability_json
    result["is_uncertain"] = is_uncertain
    return result
