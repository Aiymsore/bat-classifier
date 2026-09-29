"""A-D 声型聚类（上游：pulse_table 明细表；下游：两级预测与模型落盘）。

职责：清洗平衡聚类数据、标准化 + KMeans 聚类、把无意义聚类编号
稳定映射为声型 A-D、PCA 图与代表脉冲导出、训练临时随机森林分类器、
保存聚类模型与元数据。
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .audio_io import load_audio
from .calibration import calibrate_classifier
from .features import accepted_mask, numeric_feature_frame
from .visualization import save_pca_plot
from .config import (
    ACOUSTIC_TYPE_DESCRIPTIONS,
    ACOUSTIC_TYPE_LABELS,
    BASE_DIR,
    CLUSTER_CENTER_CSV,
    CLUSTER_EXAMPLE_DIR,
    CLUSTER_FEATURE_COLUMNS,
    CLUSTER_METADATA_JSON,
    CLUSTER_MODEL_PATH,
    CLUSTER_PCA_PNG,
    CLUSTER_REPRESENTATIVE_CSV,
    CLUSTERED_PULSE_CSV,
    ENABLE_SIMULTANEOUS_SEPARATION,
    HIGH_FREQ_HZ,
    LOW_FREQ_HZ,
    MAX_CLUSTER_PULSES_PER_RECORDING,
    MIN_PULSES_FOR_CLUSTERING,
    RANDOM_STATE,
    RECORDING_CLUSTER_CSV,
    REPRESENTATIVE_PULSES_PER_TYPE,
    SEPARATION_MAX_COMPONENTS,
    SEPARATION_MIN_FREQ_GAP_KHZ,
    SEPARATION_MIN_SECONDARY_RELATIVE_DB,
    SEPARATION_MIN_TEMPORAL_OVERLAP,
    TARGET_SAMPLE_RATE,
    validate_labels,
)


def prepare_cluster_data(
    pulse_df: pd.DataFrame,
) -> pd.DataFrame:
    """清理并平衡进入聚类的脉冲。"""
    required_columns = {
        "source_file",
        "accepted",
        *CLUSTER_FEATURE_COLUMNS,
    }
    missing_columns = (
        required_columns
        - set(pulse_df.columns)
    )
    if missing_columns:
        raise RuntimeError(
            "脉冲表缺少字段："
            f"{sorted(missing_columns)}"
        )

    data = pulse_df.loc[
        accepted_mask(pulse_df["accepted"])
    ].copy()

    data[CLUSTER_FEATURE_COLUMNS] = numeric_feature_frame(
        data,
        list(CLUSTER_FEATURE_COLUMNS),
    )
    data = data.dropna(
        subset=[
            "source_file",
            *CLUSTER_FEATURE_COLUMNS,
        ]
    )

    if len(data) < MIN_PULSES_FOR_CLUSTERING:
        raise RuntimeError(
            f"合格脉冲只有 {len(data)} 个，"
            f"至少需要 {MIN_PULSES_FOR_CLUSTERING} 个"
            "才能稳定进行声型聚类。"
        )

    # 一个长录音可能含几百个脉冲。
    # 这里限制每个原始WAV的最大贡献，避免单个文件压倒全部数据。
    sort_columns = ["source_file"]
    ascending = [True]

    if "snr_db" in data.columns:
        data["snr_db"] = pd.to_numeric(
            data["snr_db"],
            errors="coerce",
        ).fillna(-999.0)
        sort_columns.append("snr_db")
        ascending.append(False)

    if "track_coverage" in data.columns:
        sort_columns.append("track_coverage")
        ascending.append(False)

    data = (
        data.sort_values(
            sort_columns,
            ascending=ascending,
        )
        .groupby(
            "source_file",
            group_keys=False,
        )
        .head(MAX_CLUSTER_PULSES_PER_RECORDING)
        .reset_index(drop=True)
    )

    print(
        f"\n进入聚类的脉冲总数：{len(data)}"
    )
    print("各录音进入聚类的脉冲数：")
    print(
        data["source_file"]
        .value_counts()
        .sort_index()
    )

    return data


def standardize_cluster_centers(
    centers_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    只在四个聚类中心之间做标准化，
    用于稳定地给原始KMeans编号映射A-D。
    """
    standardized = centers_df.copy()

    for column in CLUSTER_FEATURE_COLUMNS:
        values = centers_df[
            column
        ].to_numpy(dtype=float)
        std = float(np.std(values))
        if std <= 1e-12:
            standardized[column] = 0.0
        else:
            standardized[column] = (
                values - float(np.mean(values))
            ) / std

    return standardized


def create_acoustic_type_mapping(
    centers_df: pd.DataFrame,
    labels: list[str] | None = None,
) -> dict[int, str]:
    """
    KMeans的原始编号没有固定含义，需要稳定映射为声型标签。

    默认 A-D 四类时沿用语义规则：
    A：下降幅度和带宽较大的FM型；
    B：高频、持续较长、较窄带的CF型；
    C：低频、较窄带、变化较缓；
    D：剩余的复杂或中间型。

    其他标签集或类数使用通用规则：
    聚类中心按峰值频率升序排列，依次分配标签列表中的名称。
    """
    if labels is None:
        labels = list(ACOUSTIC_TYPE_LABELS)

    if labels == ["A", "B", "C", "D"] and len(centers_df) == 4:
        return _semantic_abcd_mapping(centers_df)

    order = sorted(
        (int(index) for index in centers_df.index),
        key=lambda cluster: float(
            centers_df.loc[cluster, "peak_freq_khz"]
        ),
    )
    return {
        cluster: labels[position]
        for position, cluster in enumerate(order)
    }


def _semantic_abcd_mapping(
    centers_df: pd.DataFrame,
) -> dict[int, str]:
    """默认A-D四类的语义评分映射（历史实现，保持默认输出不变）。"""
    z = standardize_cluster_centers(
        centers_df
    )
    available = set(
        int(index)
        for index in centers_df.index
    )
    mapping: dict[int, str] = {}

    fm_score = (
        z["frequency_drop_khz"]
        + z["bandwidth_90_khz"]
        - 0.45 * z["duration_ms"]
        - 0.35 * z["slope_khz_per_ms"]
    )
    cluster_a = int(
        fm_score.loc[
            list(available)
        ].idxmax()
    )
    mapping[cluster_a] = "A"
    available.remove(cluster_a)

    cf_score = (
        z["peak_freq_khz"]
        + 0.65 * z["duration_ms"]
        - 0.75 * z["bandwidth_90_khz"]
        - 0.55
        * np.abs(z["frequency_drop_khz"])
        - 0.35
        * np.abs(z["slope_khz_per_ms"])
    )
    cluster_b = int(
        cf_score.loc[
            list(available)
        ].idxmax()
    )
    mapping[cluster_b] = "B"
    available.remove(cluster_b)

    low_narrow_score = (
        -z["peak_freq_khz"]
        - 0.65 * z["bandwidth_90_khz"]
        - 0.35
        * np.abs(z["frequency_drop_khz"])
    )
    cluster_c = int(
        low_narrow_score.loc[
            list(available)
        ].idxmax()
    )
    mapping[cluster_c] = "C"
    available.remove(cluster_c)

    cluster_d = int(next(iter(available)))
    mapping[cluster_d] = "D"

    return mapping


def export_pulse_audio(
    row: "pd.Series | dict",
    destination: Path,
    audio_cache: dict[str, tuple[np.ndarray, int]] | None = None,
) -> str:
    """把一个脉冲写到 destination，返回写入路径（相对 BASE_DIR，无法相对时为绝对路径）。

    代表脉冲必须沿用已分离波形（row["pulse_file"]），绝不能再从原始
    混合录音切一遍把两个声源混回去；只有未发生分离的普通脉冲才回退
    到按 start_s/end_s 从源录音切割。

    训练页 save_cluster_representatives 与批量整理（curation）的
    代表样本导出共用此逻辑。audio_cache 由调用方提供以复用录音加载。
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cache = (
        audio_cache if audio_cache is not None else {}
    )

    pulse_path_text = str(row.get("pulse_file", "") or "")
    source_path_text = str(row.get("source_path", "") or "")

    separated_source = None
    if pulse_path_text:
        separated_source = Path(pulse_path_text)
        if not separated_source.is_absolute():
            separated_source = BASE_DIR / separated_source

    if separated_source is not None and separated_source.exists():
        shutil.copy2(separated_source, destination)
    elif source_path_text:
        if source_path_text not in cache:
            cache[source_path_text] = load_audio(
                Path(source_path_text)
            )
        source_y, source_sr = cache[source_path_text]
        start_sample = max(
            0,
            int(round(float(row["start_s"]) * source_sr)),
        )
        end_sample = min(
            len(source_y),
            int(round(float(row["end_s"]) * source_sr)),
        )
        segment = source_y[start_sample:end_sample]
        if len(segment) < 32:
            return ""
        sf.write(
            destination,
            segment.astype(np.float32),
            source_sr,
            subtype="PCM_16",
        )
    else:
        return ""

    try:
        return str(destination.relative_to(BASE_DIR))
    except ValueError:
        return str(destination)


def save_cluster_representatives(
    data: pd.DataFrame,
    scaled_features: np.ndarray,
    kmeans: KMeans,
    mapping: dict[int, str],
    labels: list[str] | None = None,
    type_descriptions: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    找出距离每个聚类中心最近的代表脉冲，
    并复制到 cluster_examples/声型<标签> 目录。
    """
    if labels is None:
        labels = list(ACOUSTIC_TYPE_LABELS)
    descriptions = (
        dict(type_descriptions)
        if type_descriptions
        else dict(ACOUSTIC_TYPE_DESCRIPTIONS)
    )
    representative_rows: list[
        dict[str, float | int | str]
    ] = []
    audio_cache: dict[
        str,
        tuple[np.ndarray, int],
    ] = {}

    raw_labels = data[
        "raw_cluster"
    ].to_numpy(dtype=int)

    for raw_cluster, acoustic_type in sorted(
        mapping.items(),
        key=lambda item: (
            labels.index(item[1])
            if item[1] in labels
            else len(labels)
        ),
    ):
        indices = np.where(
            raw_labels == raw_cluster
        )[0]
        if len(indices) == 0:
            continue

        center = kmeans.cluster_centers_[
            raw_cluster
        ]
        distances = np.linalg.norm(
            scaled_features[indices] - center,
            axis=1,
        )
        order = np.argsort(distances)[
            :REPRESENTATIVE_PULSES_PER_TYPE
        ]

        type_dir = (
            CLUSTER_EXAMPLE_DIR
            / f"声型{acoustic_type}"
        )
        type_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for rank, local_order in enumerate(
            order,
            start=1,
        ):
            row_index = int(
                indices[int(local_order)]
            )
            row = data.iloc[row_index]
            pulse_path_text = str(
                row.get("pulse_file", "")
            )

            source_path_text = str(
                row.get("source_path", "")
            )
            destination = type_dir / (
                f"{rank:02d}_"
                f"{Path(source_path_text).stem}"
                f"_pulse_{int(row['pulse_id']):03d}"
                ".wav"
            )

            # 代表脉冲必须沿用已分离波形（export_pulse_audio 内部保证），
            # 未分离的普通脉冲才回退为从源录音按时间切割。
            copied_path = export_pulse_audio(
                row,
                destination,
                audio_cache,
            )

            representative_rows.append(
                {
                    "acoustic_type": acoustic_type,
                    "description": (
                        descriptions.get(
                            acoustic_type,
                            "",
                        )
                    ),
                    "rank": rank,
                    "source_file": row[
                        "source_file"
                    ],
                    "pulse_id": int(
                        row["pulse_id"]
                    ),
                    "distance_to_center": float(
                        distances[
                            int(local_order)
                        ]
                    ),
                    "original_pulse_file": (
                        pulse_path_text
                    ),
                    "copied_example_file": (
                        copied_path
                    ),
                }
            )

    representative_df = pd.DataFrame(
        representative_rows
    )
    representative_df.to_csv(
        CLUSTER_REPRESENTATIVE_CSV,
        index=False,
        encoding="utf-8-sig",
    )
    return representative_df


def create_recording_cluster_summary(
    clustered_df: pd.DataFrame,
    detection_df: pd.DataFrame,
    labels: list[str] | None = None,
    type_descriptions: dict[str, str] | None = None,
) -> pd.DataFrame:
    """统计每段录音中各声型所占比例。"""
    if labels is None:
        labels = list(ACOUSTIC_TYPE_LABELS)
    descriptions = (
        dict(type_descriptions)
        if type_descriptions
        else dict(ACOUSTIC_TYPE_DESCRIPTIONS)
    )
    count_table = pd.crosstab(
        clustered_df["source_file"],
        clustered_df["acoustic_type"],
    )

    for acoustic_type in labels:
        if acoustic_type not in count_table:
            count_table[acoustic_type] = 0

    count_table = count_table[
        labels
    ]
    total = count_table.sum(axis=1)

    summary = pd.DataFrame(
        {
            "source_file": count_table.index,
            "clustered_pulse_count": total,
        }
    ).reset_index(drop=True)

    for acoustic_type in labels:
        summary[
            f"type_{acoustic_type}_count"
        ] = count_table[
            acoustic_type
        ].to_numpy()
        summary[
            f"type_{acoustic_type}_ratio"
        ] = (
            count_table[
                acoustic_type
            ].to_numpy()
            / total.to_numpy()
        )

    summary["major_acoustic_type"] = (
        count_table.idxmax(axis=1)
        .to_numpy()
    )
    summary[
        "major_type_description"
    ] = summary[
        "major_acoustic_type"
    ].map(
        lambda label: descriptions.get(label, "")
    )
    summary["major_type_ratio"] = (
        count_table.max(axis=1).to_numpy()
        / total.to_numpy()
    )

    if not detection_df.empty:
        summary = detection_df.merge(
            summary,
            on="source_file",
            how="left",
        )

    summary.to_csv(
        RECORDING_CLUSTER_CSV,
        index=False,
        encoding="utf-8-sig",
    )
    return summary


def train_temporary_classifier(
    features: pd.DataFrame,
    labels: pd.Series,
) -> tuple[
    RandomForestClassifier,
    float | None,
    str,
]:
    """
    训练把新脉冲归入A-D的临时随机森林。

    这里得到的是“复现聚类标签的一致率”，
    不是物种鉴定准确率。
    """
    class_counts = labels.value_counts()
    can_split = (
        len(features) >= 40
        and int(class_counts.min()) >= 2
    )

    model = RandomForestClassifier(
        n_estimators=600,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    if can_split:
        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = train_test_split(
            features,
            labels,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=labels,
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        consistency = accuracy_score(
            y_test,
            predictions,
        )
        evaluation_note = (
            "随机抽取25%脉冲测试，"
            "该数值只表示临时分类器复现"
            "KMeans声型标签的一致率。"
        )
    else:
        consistency = None
        evaluation_note = (
            "某些声型样本过少，未划分测试集。"
        )

    # 最终部署模型使用全部伪标签脉冲训练
    model.fit(features, labels)

    return (
        model,
        consistency,
        evaluation_note,
    )


def _interpret_silhouette(value: float) -> str:
    """把轮廓系数翻译成一句中文评价。"""
    if value >= 0.5:
        return "组间分离明显"
    if value >= 0.25:
        return "有一定结构，但组间重叠较多"
    return "组间重叠严重，结果仅供流程演示参考"


def build_zh_summary(
    labels: list[str],
    descriptions: dict[str, str],
    pulse_count: int,
    recording_count: int,
    silhouette: float | None,
    consistency: float | None,
    pca_ratios: list[float] | None,
    separation: dict[str, object] | None,
    trained_at: str | None = None,
) -> str:
    """把训练元数据渲染成人类可读的中文摘要。

    元数据写入（cluster_acoustic_types）与界面兜底渲染
    （模型管理页对旧元数据）共用此函数。
    """
    lines: list[str] = []
    header = "【训练摘要】"
    if trained_at:
        header += f" {trained_at}"
    lines.append(header)

    lines.append(
        f"声型标签：{'、'.join(labels)}（共{len(labels)}类）"
    )
    desc_text = " ｜ ".join(
        f"{label}：{descriptions.get(label, '') or '（无描述）'}"
        for label in labels
    )
    lines.append(f"  {desc_text}")

    lines.append(
        f"训练规模：{pulse_count} 个合格脉冲，来自 {recording_count} 条录音"
    )

    if silhouette is not None:
        lines.append(
            f"聚类质量：轮廓系数 {silhouette:.3f} —— "
            f"{_interpret_silhouette(silhouette)}"
            "（取值0~1，越大组间分得越开）"
        )
    else:
        lines.append("聚类质量：未计算轮廓系数。")

    if consistency is not None:
        lines.append(
            f"临时分类器一致率：{consistency:.2%} —— "
            "复现聚类标签的能力，不是物种识别准确率"
        )
    else:
        lines.append("临时分类器一致率：样本过少，本次未评估。")

    if pca_ratios:
        detail = " + ".join(f"{value:.1%}" for value in pca_ratios)
        lines.append(
            f"PCA：前{len(pca_ratios)}个主成分共解释 "
            f"{sum(pca_ratios):.1%} 的特征差异（{detail}）"
        )

    if separation:
        enabled = "已启用" if separation.get("enabled") else "未启用"
        gap = separation.get("minimum_frequency_gap_khz")
        components = separation.get("maximum_components")
        extra = ""
        if gap is not None and components is not None:
            extra = f"（最小频率间隔 {gap} kHz，最多 {components} 分量）"
        lines.append(f"声波分离：{enabled}{extra}")

    lines.append("提示：声型标签只代表声学相似类型，不是物种标签。")
    return "\n".join(lines)


def cluster_acoustic_types(
    pulse_df: pd.DataFrame,
    detection_df: pd.DataFrame,
    labels: list[str] | None = None,
    type_descriptions: dict[str, str] | None = None,
) -> dict[str, object]:
    """执行声型聚类、生成图表并训练临时分类器。

    labels：声型标签列表（默认 A-D 四类），长度即聚类数 K；
    type_descriptions：标签到描述文字的映射（默认取 config）。
    """
    if labels is None:
        labels = list(ACOUSTIC_TYPE_LABELS)
    labels = validate_labels(labels)
    descriptions = (
        dict(type_descriptions)
        if type_descriptions
        else dict(ACOUSTIC_TYPE_DESCRIPTIONS)
    )
    for label in labels:
        descriptions.setdefault(label, "")

    data = prepare_cluster_data(pulse_df)

    feature_frame = data[
        CLUSTER_FEATURE_COLUMNS
    ].copy()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(
        feature_frame
    )

    kmeans = KMeans(
        n_clusters=len(labels),
        n_init=30,
        random_state=RANDOM_STATE,
    )
    raw_labels = kmeans.fit_predict(
        scaled_features
    )
    data["raw_cluster"] = raw_labels

    original_centers = scaler.inverse_transform(
        kmeans.cluster_centers_
    )
    centers_df = pd.DataFrame(
        original_centers,
        columns=CLUSTER_FEATURE_COLUMNS,
    )
    centers_df.index.name = "raw_cluster"

    mapping = create_acoustic_type_mapping(
        centers_df,
        labels,
    )
    data["acoustic_type"] = data[
        "raw_cluster"
    ].map(mapping)
    data["acoustic_type_description"] = (
        data["acoustic_type"].map(
            lambda label: descriptions.get(label, "")
        )
    )

    pca = PCA(
        n_components=2,
        random_state=RANDOM_STATE,
    )
    pca_coordinates = pca.fit_transform(
        scaled_features
    )
    data["pca_1"] = pca_coordinates[:, 0]
    data["pca_2"] = pca_coordinates[:, 1]

    silhouette = None
    if (
        len(data) > len(labels)
        and len(np.unique(raw_labels)) > 1
    ):
        silhouette = float(
            silhouette_score(
                scaled_features,
                raw_labels,
            )
        )

    center_output = centers_df.copy()
    center_output["acoustic_type"] = [
        mapping[int(index)]
        for index in center_output.index
    ]
    center_output["description"] = (
        center_output[
            "acoustic_type"
        ].map(
            lambda label: descriptions.get(label, "")
        )
    )
    cluster_counts = (
        data["acoustic_type"]
        .value_counts()
    )
    center_output["pulse_count"] = (
        center_output[
            "acoustic_type"
        ].map(cluster_counts)
    )
    label_order = {
        label: position
        for position, label in enumerate(labels)
    }
    center_output = (
        center_output.reset_index()
        .sort_values(
            "acoustic_type",
            key=lambda column: column.map(label_order),
        )
    )
    center_output.to_csv(
        CLUSTER_CENTER_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    data.to_csv(
        CLUSTERED_PULSE_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    save_pca_plot(data, labels)
    representative_df = (
        save_cluster_representatives(
            data,
            scaled_features,
            kmeans,
            mapping,
            labels,
            descriptions,
        )
    )
    recording_summary = (
        create_recording_cluster_summary(
            data,
            detection_df,
            labels,
            descriptions,
        )
    )

    temporary_model, consistency, note = (
        train_temporary_classifier(
            feature_frame,
            data["acoustic_type"],
        )
    )

    # 概率校准（⑦ 置信度用）：按 source_recording 分组隔离；
    # 数据不足则不校准，置信度退回未校准的模型支持度（绝不伪造）。
    calibration_result = calibrate_classifier(
        feature_frame,
        data["acoustic_type"],
        groups=(
            data["source_file"]
            if "source_file" in data.columns
            else None
        ),
        base=temporary_model,
    )
    calibrated_model = calibration_result["classifier"]
    calibration_info = calibration_result["info"]
    if calibration_info["is_calibrated"]:
        print(
            f"[概率校准] ⑦ 置信度已校准："
            f"CalibratedClassifierCV(method="
            f"{calibration_info['calibration_method']})，"
            f"{calibration_info['fold_count']} 折，录音分组隔离。"
        )
    else:
        print(
            "[概率校准] ⑦ 置信度未校准（数据不足以安全校准）："
            + "；".join(calibration_info.get("warnings") or ["未知原因"])
        )

    payload = {
        "scaler": scaler,
        "kmeans": kmeans,
        "pca": pca,
        "temporary_classifier": (
            temporary_model
        ),
        "calibrated_classifier": (
            calibrated_model
        ),
        "calibration": calibration_info,
        "raw_cluster_to_type": mapping,
        "feature_columns": (
            CLUSTER_FEATURE_COLUMNS
        ),
        "labels": list(labels),
        "type_descriptions": (
            descriptions
        ),
        "target_sample_rate": (
            TARGET_SAMPLE_RATE
        ),
        "low_freq_hz": LOW_FREQ_HZ,
        "high_freq_hz": HIGH_FREQ_HZ,
        "simultaneous_separation": {
            "enabled": (
                ENABLE_SIMULTANEOUS_SEPARATION
            ),
            "minimum_frequency_gap_khz": (
                SEPARATION_MIN_FREQ_GAP_KHZ
            ),
            "maximum_components": (
                SEPARATION_MAX_COMPONENTS
            ),
            "minimum_temporal_overlap": (
                SEPARATION_MIN_TEMPORAL_OVERLAP
            ),
            "minimum_secondary_relative_db": (
                SEPARATION_MIN_SECONDARY_RELATIVE_DB
            ),
            "method": (
                "multi_ridge_tracking_and_soft_mask"
            ),
        },
        "silhouette_score": silhouette,
        "temporary_classifier_consistency": (
            consistency
        ),
        "warning": (
            "声型标签是由当前无标签数据自动聚类得到的，"
            "不是经过物种鉴定的真实物种标签。"
        ),
    }
    joblib.dump(
        payload,
        CLUSTER_MODEL_PATH,
    )

    pca_ratios = [
        float(value)
        for value
        in pca.explained_variance_ratio_
    ]
    metadata = {
        "labels": list(labels),
        "acoustic_types": {
            acoustic_type: descriptions.get(
                acoustic_type,
                "",
            )
            for acoustic_type
            in labels
        },
        "trained_at": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "pulse_count": int(len(data)),
        "recording_count": int(
            data["source_file"].nunique()
        ),
        "silhouette_score": silhouette,
        "temporary_classifier_consistency": (
            consistency
        ),
        "calibration": calibration_info,
        "evaluation_note": note,
        "pca_explained_variance_ratio": pca_ratios,
        "model_file": str(
            CLUSTER_MODEL_PATH.name
        ),
        "simultaneous_separation": (
            payload["simultaneous_separation"]
        ),
        "summary_zh": build_zh_summary(
            labels=labels,
            descriptions=descriptions,
            pulse_count=int(len(data)),
            recording_count=int(
                data["source_file"].nunique()
            ),
            silhouette=silhouette,
            consistency=consistency,
            pca_ratios=pca_ratios,
            separation=payload[
                "simultaneous_separation"
            ],
            trained_at=datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        ),
        "warning": (
            "当前声型标签只代表声学相似类型。"
            "拿到老师的物种数据库后，"
            "再把声型标签替换为真实物种标签。"
        ),
    }
    CLUSTER_METADATA_JSON.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n========== 声型聚类完成 ==========")
    print(
        data["acoustic_type"]
        .value_counts()
        .sort_index()
    )
    print("\n聚类中心：")
    print(
        center_output[
            [
                "acoustic_type",
                "description",
                "pulse_count",
                "duration_ms",
                "peak_freq_khz",
                "frequency_drop_khz",
                "bandwidth_90_khz",
            ]
        ].to_string(index=False)
    )

    if silhouette is not None:
        print(
            f"\n轮廓系数：{silhouette:.3f}"
        )
        print(
            "越接近1表示各组分离越明显；"
            "接近0表示组间重叠较多。"
        )

    if consistency is not None:
        print(
            f"临时随机森林复现声型标签的一致率："
            f"{consistency:.2%}"
        )
        print(
            "注意：这不是物种识别准确率。"
        )

    print(
        f"\n逐脉冲声型结果："
        f"{CLUSTERED_PULSE_CSV}"
    )
    print(
        f"逐录音声型汇总："
        f"{RECORDING_CLUSTER_CSV}"
    )
    print(
        f"PCA聚类图：{CLUSTER_PCA_PNG}"
    )
    print(
        f"代表脉冲目录："
        f"{CLUSTER_EXAMPLE_DIR}"
    )
    print(
        f"临时声型模型："
        f"{CLUSTER_MODEL_PATH}"
    )

    return {
        "data": data,
        "recording_summary": (
            recording_summary
        ),
        "representatives": (
            representative_df
        ),
        "payload": payload,
    }
