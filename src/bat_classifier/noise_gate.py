"""第一层 noise/bat 门控模型（上游：pulse_table 明细表；下游：两级预测）。

职责：用质量筛选结果生成原型伪标签（accepted→bat / rejected→noise）、
训练按录音分组的随机森林门控、对新候选批量输出 noise/bat 概率。
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit

from .features import accepted_mask, numeric_feature_frame
from .config import (
    MAX_NOISE_GATE_ROWS_PER_SOURCE_CLASS,
    MIN_NOISE_GATE_ROWS_PER_CLASS,
    NOISE_BAT_PROBABILITY_THRESHOLD,
    NOISE_GATE_FEATURE_COLUMNS,
    NOISE_GATE_MODEL_PATH,
    NOISE_GATE_PREDICTIONS_CSV,
    RANDOM_STATE,
)


def prepare_noise_gate_data(
    pulse_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    用当前检测器的质量判断生成原型标签：

    accepted=True  -> bat
    accepted=False -> noise

    这是演示用伪标签。拿到老师数据库后，应使用人工确认的
    蝙蝠脉冲和真实噪声替换。
    """
    required = {
        "source_file",
        "accepted",
        *NOISE_GATE_FEATURE_COLUMNS,
    }
    missing = required - set(pulse_df.columns)
    if missing:
        raise RuntimeError(
            f"noise/bat 数据缺少字段：{sorted(missing)}"
        )

    data = pulse_df.copy()
    data["stage1_label"] = np.where(
        accepted_mask(data["accepted"]),
        "bat",
        "noise",
    )

    data[NOISE_GATE_FEATURE_COLUMNS] = numeric_feature_frame(
        data,
        list(NOISE_GATE_FEATURE_COLUMNS),
    )
    data = (
        data.dropna(
            subset=[
                "source_file",
                "stage1_label",
                *NOISE_GATE_FEATURE_COLUMNS,
            ]
        )
        .copy()
    )

    # 限制同一个录音、同一类别的贡献，避免长录音压倒其他数据。
    # bat 与 noise 都按 snr_db、track_coverage 降序保留靠前的"难样本"：
    # noise 保留较难的误检，而不是只保留完全安静的片段。
    kept_parts: list[pd.DataFrame] = []

    for _, group in data.groupby(
        ["source_file", "stage1_label"],
        sort=False,
    ):
        sort_columns = [
            column
            for column in (
                "snr_db",
                "track_coverage",
            )
            if column in group.columns
        ]
        if sort_columns:
            group = group.sort_values(
                sort_columns,
                ascending=[False] * len(sort_columns),
            )

        kept_parts.append(
            group.head(
                MAX_NOISE_GATE_ROWS_PER_SOURCE_CLASS
            )
        )

    data = pd.concat(
        kept_parts,
        ignore_index=True,
    )

    class_counts = data["stage1_label"].value_counts()
    if set(class_counts.index) != {"bat", "noise"}:
        raise RuntimeError(
            f"noise/bat 类别不完整：{class_counts.to_dict()}"
        )

    if int(class_counts.min()) < MIN_NOISE_GATE_ROWS_PER_CLASS:
        raise RuntimeError(
            "noise或bat样本太少，无法训练第一层模型。"
            f"当前数量：{class_counts.to_dict()}"
        )

    print("\n第一层 noise / bat 数据：")
    print(class_counts.sort_index())
    print(
        f"独立原始录音：{data['source_file'].nunique()}个"
    )

    return data


def make_noise_gate_model(
    random_state: int,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=600,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )


def train_noise_gate(
    pulse_df: pd.DataFrame,
) -> dict[str, object]:
    """
    训练第一层 noise / bat 随机森林。

    优先按原始WAV分组划分验证集，避免同一录音的片段同时
    出现在训练和测试中。
    """
    data = prepare_noise_gate_data(pulse_df)

    X = data[NOISE_GATE_FEATURE_COLUMNS].copy()
    y = data["stage1_label"].astype(str)
    groups = data["source_file"].astype(str)

    model = make_noise_gate_model(RANDOM_STATE)
    evaluation_note = ""
    validation_accuracy: float | None = None
    validation_df = pd.DataFrame()

    unique_groups = groups.nunique()

    if unique_groups >= 4:
        split_found = False

        for attempt in range(30):
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=0.25,
                random_state=RANDOM_STATE + attempt,
            )
            train_index, test_index = next(
                splitter.split(X, y, groups)
            )

            if (
                y.iloc[train_index].nunique() == 2
                and y.iloc[test_index].nunique() == 2
            ):
                split_found = True
                break

        if split_found:
            validation_model = make_noise_gate_model(
                RANDOM_STATE + 100
            )
            validation_model.fit(
                X.iloc[train_index],
                y.iloc[train_index],
            )
            predictions = validation_model.predict(
                X.iloc[test_index]
            )
            probabilities = validation_model.predict_proba(
                X.iloc[test_index]
            )
            validation_accuracy = accuracy_score(
                y.iloc[test_index],
                predictions,
            )

            validation_df = data.iloc[test_index][
                [
                    "source_file",
                    "pulse_id",
                    "accepted",
                    "rejection_reason",
                    "stage1_label",
                ]
            ].copy()
            validation_df["prediction"] = predictions

            for class_index, class_name in enumerate(
                validation_model.classes_
            ):
                validation_df[
                    f"prob_{class_name}"
                ] = probabilities[:, class_index]

            evaluation_note = (
                "按原始WAV分组留出25%录音验证。"
                "此准确率反映原型伪标签的一致性，"
                "不是人工标注下的真实noise/bat准确率。"
            )

    if validation_accuracy is None:
        evaluation_note = (
            "独立录音数量或类别分布不足，"
            "没有进行可靠的分组留出验证。"
        )

    # 最终模型使用全部原型数据训练。
    model.fit(X, y)

    payload = {
        "model": model,
        "feature_columns": list(
            NOISE_GATE_FEATURE_COLUMNS
        ),
        "classes": list(model.classes_),
        "bat_probability_threshold": (
            NOISE_BAT_PROBABILITY_THRESHOLD
        ),
        "validation_accuracy": validation_accuracy,
        "evaluation_note": evaluation_note,
        "warning": (
            "当前noise/bat标签来自自动质量筛选，"
            "用于演示流程；正式版应换成人工确认标签。"
        ),
    }

    joblib.dump(payload, NOISE_GATE_MODEL_PATH)

    if not validation_df.empty:
        validation_df.to_csv(
            NOISE_GATE_PREDICTIONS_CSV,
            index=False,
            encoding="utf-8-sig",
        )

    print("\n========== 第一层 noise / bat 完成 ==========")
    if validation_accuracy is not None:
        print(
            f"分组验证一致率："
            f"{validation_accuracy:.2%}"
        )
    print(evaluation_note)
    print(
        f"第一层模型：{NOISE_GATE_MODEL_PATH}"
    )

    importance = pd.Series(
        model.feature_importances_,
        index=NOISE_GATE_FEATURE_COLUMNS,
    ).sort_values(ascending=False)
    print("\n第一层特征重要性：")
    print(importance.head(10))

    return payload


def apply_noise_gate(
    candidate_df: pd.DataFrame,
    noise_payload: dict[str, object],
) -> pd.DataFrame:
    """给每个候选片段输出 noise / bat 及概率。"""
    model: RandomForestClassifier = noise_payload["model"]
    feature_columns = list(
        noise_payload["feature_columns"]
    )
    threshold = float(
        noise_payload["bat_probability_threshold"]
    )

    data = candidate_df.copy()

    data[feature_columns] = numeric_feature_frame(
        data,
        feature_columns,
    )
    valid_mask = data[feature_columns].notna().all(axis=1)
    data = data.loc[valid_mask].copy()

    if data.empty:
        return data

    probabilities = model.predict_proba(
        data[feature_columns]
    )
    class_to_index = {
        str(class_name): index
        for index, class_name in enumerate(
            model.classes_
        )
    }

    if "bat" not in class_to_index:
        raise RuntimeError(
            "第一层模型缺少 bat 类。"
        )

    bat_probability = probabilities[
        :,
        class_to_index["bat"],
    ]
    noise_probability = (
        probabilities[
            :,
            class_to_index["noise"],
        ]
        if "noise" in class_to_index
        else 1.0 - bat_probability
    )

    data["prob_bat"] = bat_probability
    data["prob_noise"] = noise_probability
    data["stage1_prediction"] = np.where(
        bat_probability >= threshold,
        "bat",
        "noise",
    )

    return data
