"""A-D 声型聚类（上游：pulse_table 明细表；下游：两级预测与模型落盘）。

职责：清洗平衡聚类数据、标准化 + KMeans 聚类、把无意义聚类编号
稳定映射为声型 A-D、PCA 图与代表脉冲导出、训练临时随机森林分类器、
保存聚类模型与元数据。
"""

from __future__ import annotations

import json
import shutil
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
from .features import accepted_mask, numeric_feature_frame
from .visualization import save_pca_plot
from .config import (
    ACOUSTIC_TYPE_DESCRIPTIONS,
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
    N_ACOUSTIC_TYPES,
    RANDOM_STATE,
    RECORDING_CLUSTER_CSV,
    REPRESENTATIVE_PULSES_PER_TYPE,
    SEPARATION_MAX_COMPONENTS,
    SEPARATION_MIN_FREQ_GAP_KHZ,
    SEPARATION_MIN_SECONDARY_RELATIVE_DB,
    SEPARATION_MIN_TEMPORAL_OVERLAP,
    TARGET_SAMPLE_RATE,
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
            "才能稳定演示四类聚类。"
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
) -> dict[int, str]:
    """
    KMeans的0、1、2、3没有固定含义。
    用聚类中心的声学特征把它们稳定映射为A-D：

    A：下降幅度和带宽较大的FM型；
    B：高频、持续较长、较窄带的CF型；
    C：低频、较窄带、变化较缓；
    D：剩余的复杂或中间型。
    """
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


def save_cluster_representatives(
    data: pd.DataFrame,
    scaled_features: np.ndarray,
    kmeans: KMeans,
    mapping: dict[int, str],
) -> pd.DataFrame:
    """
    找出距离每个聚类中心最近的代表脉冲，
    并复制到 cluster_examples/声型A-D。
    """
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
        key=lambda item: item[1],
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

            copied_path = ""
            source_path_text = str(
                row.get("source_path", "")
            )
            destination = type_dir / (
                f"{rank:02d}_"
                f"{Path(source_path_text).stem}"
                f"_pulse_{int(row['pulse_id']):03d}"
                ".wav"
            )

            separated_source = None
            if pulse_path_text:
                separated_source = Path(
                    pulse_path_text
                )
                if not separated_source.is_absolute():
                    separated_source = (
                        BASE_DIR / separated_source
                    )

            if (
                separated_source is not None
                and separated_source.exists()
            ):
                # 代表脉冲必须沿用已分离波形，不能再从原始
                # 混合录音切一遍，否则会把两个声源重新混回去。
                shutil.copy2(
                    separated_source,
                    destination,
                )
                copied_path = str(
                    destination.relative_to(BASE_DIR)
                )
            elif source_path_text:
                if (
                    source_path_text
                    not in audio_cache
                ):
                    source_audio_path = Path(
                        source_path_text
                    )
                    audio_cache[
                        source_path_text
                    ] = load_audio(
                        source_audio_path
                    )

                source_y, source_sr = (
                    audio_cache[
                        source_path_text
                    ]
                )
                start_sample = max(
                    0,
                    int(
                        round(
                            float(row["start_s"])
                            * source_sr
                        )
                    ),
                )
                end_sample = min(
                    len(source_y),
                    int(
                        round(
                            float(row["end_s"])
                            * source_sr
                        )
                    ),
                )
                segment = source_y[
                    start_sample:end_sample
                ]

                if len(segment) >= 32:
                    sf.write(
                        destination,
                        segment.astype(np.float32),
                        source_sr,
                        subtype="PCM_16",
                    )
                    copied_path = str(
                        destination.relative_to(
                            BASE_DIR
                        )
                    )

            representative_rows.append(
                {
                    "acoustic_type": acoustic_type,
                    "description": (
                        ACOUSTIC_TYPE_DESCRIPTIONS[
                            acoustic_type
                        ]
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
) -> pd.DataFrame:
    """统计每段录音中A-D各自所占比例。"""
    count_table = pd.crosstab(
        clustered_df["source_file"],
        clustered_df["acoustic_type"],
    )

    for acoustic_type in ["A", "B", "C", "D"]:
        if acoustic_type not in count_table:
            count_table[acoustic_type] = 0

    count_table = count_table[
        ["A", "B", "C", "D"]
    ]
    total = count_table.sum(axis=1)

    summary = pd.DataFrame(
        {
            "source_file": count_table.index,
            "clustered_pulse_count": total,
        }
    ).reset_index(drop=True)

    for acoustic_type in ["A", "B", "C", "D"]:
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
        ACOUSTIC_TYPE_DESCRIPTIONS
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


def cluster_acoustic_types(
    pulse_df: pd.DataFrame,
    detection_df: pd.DataFrame,
) -> dict[str, object]:
    """执行四类聚类、生成图表并训练临时分类器。"""
    data = prepare_cluster_data(pulse_df)

    feature_frame = data[
        CLUSTER_FEATURE_COLUMNS
    ].copy()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(
        feature_frame
    )

    kmeans = KMeans(
        n_clusters=N_ACOUSTIC_TYPES,
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
        centers_df
    )
    data["acoustic_type"] = data[
        "raw_cluster"
    ].map(mapping)
    data["acoustic_type_description"] = (
        data["acoustic_type"].map(
            ACOUSTIC_TYPE_DESCRIPTIONS
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
        len(data) > N_ACOUSTIC_TYPES
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
            ACOUSTIC_TYPE_DESCRIPTIONS
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
    center_output = (
        center_output.reset_index()
        .sort_values("acoustic_type")
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

    save_pca_plot(data)
    representative_df = (
        save_cluster_representatives(
            data,
            scaled_features,
            kmeans,
            mapping,
        )
    )
    recording_summary = (
        create_recording_cluster_summary(
            data,
            detection_df,
        )
    )

    temporary_model, consistency, note = (
        train_temporary_classifier(
            feature_frame,
            data["acoustic_type"],
        )
    )

    payload = {
        "scaler": scaler,
        "kmeans": kmeans,
        "pca": pca,
        "temporary_classifier": (
            temporary_model
        ),
        "raw_cluster_to_type": mapping,
        "feature_columns": (
            CLUSTER_FEATURE_COLUMNS
        ),
        "type_descriptions": (
            ACOUSTIC_TYPE_DESCRIPTIONS
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
            "A-D是由当前无标签数据自动聚类得到的声型，"
            "不是经过物种鉴定的真实物种标签。"
        ),
    }
    joblib.dump(
        payload,
        CLUSTER_MODEL_PATH,
    )

    metadata = {
        "acoustic_types": {
            acoustic_type: (
                ACOUSTIC_TYPE_DESCRIPTIONS[
                    acoustic_type
                ]
            )
            for acoustic_type
            in ["A", "B", "C", "D"]
        },
        "pulse_count": int(len(data)),
        "recording_count": int(
            data["source_file"].nunique()
        ),
        "silhouette_score": silhouette,
        "temporary_classifier_consistency": (
            consistency
        ),
        "evaluation_note": note,
        "pca_explained_variance_ratio": [
            float(value)
            for value
            in pca.explained_variance_ratio_
        ],
        "model_file": str(
            CLUSTER_MODEL_PATH.name
        ),
        "simultaneous_separation": (
            payload["simultaneous_separation"]
        ),
        "warning": (
            "当前A-D只代表声学相似类型。"
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
            "越接近1表示四组分离越明显；"
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
        f"\n逐脉冲A-D结果："
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
