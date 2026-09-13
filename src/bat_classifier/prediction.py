"""新录音两级预测（上游：noise_gate + clustering 模型；下游：预测 CSV/JSON 输出）。

职责：对单个 WAV 或整个文件夹执行
候选检测 → noise/bat 门控 → A-D 声型分类 → 逐片段 CSV 与汇总输出。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .audio_io import find_wav_files, load_audio
from .detection import detect_and_refine_pulses_chunked
from .features import numeric_feature_frame
from .noise_gate import apply_noise_gate
from .pulse_table import create_demo_pulse_rows
from .visualization import (
    save_fast_overview_spectrogram,
    save_top_pulse_zoom_grid,
)
from .config import ACOUSTIC_TYPE_DESCRIPTIONS, DEMO_PREDICTION_DIR

if TYPE_CHECKING:
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler


def predict_one_demo_wav_two_stage(
    audio_path: Path,
    combined_payload: dict[str, object],
) -> dict[str, object]:
    """
    新WAV两级判断：

    第一层：noise / bat
    第二层：仅把bat候选分到A-D
    """
    noise_payload = combined_payload["noise_gate"]
    acoustic_payload = combined_payload["acoustic_types"]

    scaler: StandardScaler = acoustic_payload["scaler"]
    kmeans: KMeans = acoustic_payload["kmeans"]
    temporary_model: RandomForestClassifier = (
        acoustic_payload["temporary_classifier"]
    )
    mapping: dict[int, str] = (
        acoustic_payload["raw_cluster_to_type"]
    )
    acoustic_features = list(
        acoustic_payload["feature_columns"]
    )

    y, sr = load_audio(audio_path)

    (
        raw_candidate_count,
        refined_pulses,
        _,
    ) = detect_and_refine_pulses_chunked(y, sr)

    candidate_rows = create_demo_pulse_rows(
        audio_path,
        y,
        sr,
        refined_pulses,
    )
    split_candidate_count = len(
        {
            int(row["candidate_id"])
            for row in candidate_rows
            if bool(row["separation_applied"])
        }
    )

    if not candidate_rows:
        print(
            f"\n{audio_path.name}："
            "没有可提取特征的候选片段。"
        )
        return {
            "source_file": audio_path.name,
            "status": "no_candidates",
            "raw_candidate_count": raw_candidate_count,
        }

    stage1_df = apply_noise_gate(
        pd.DataFrame(candidate_rows),
        noise_payload,
    )

    if stage1_df.empty:
        return {
            "source_file": audio_path.name,
            "status": "no_valid_features",
            "raw_candidate_count": raw_candidate_count,
        }

    bat_df = stage1_df.loc[
        stage1_df["stage1_prediction"] == "bat"
    ].copy()
    noise_df = stage1_df.loc[
        stage1_df["stage1_prediction"] == "noise"
    ].copy()

    stage1_df["acoustic_type"] = ""
    stage1_df["acoustic_type_description"] = ""

    if not bat_df.empty:
        bat_df[acoustic_features] = numeric_feature_frame(
            bat_df,
            acoustic_features,
        )
        valid_bat_mask = bat_df[acoustic_features].notna().all(axis=1)
        bat_df = bat_df.loc[valid_bat_mask].copy()

    if not bat_df.empty:
        scaled = scaler.transform(
            bat_df[acoustic_features]
        )
        raw_clusters = kmeans.predict(scaled)

        kmeans_types = np.array(
            [
                mapping[int(cluster)]
                for cluster in raw_clusters
            ],
            dtype=object,
        )
        classifier_types = temporary_model.predict(
            bat_df[acoustic_features]
        )
        type_probabilities = (
            temporary_model.predict_proba(
                bat_df[acoustic_features]
            )
        )

        bat_df["kmeans_acoustic_type"] = (
            kmeans_types
        )
        bat_df["acoustic_type"] = (
            classifier_types
        )
        bat_df[
            "acoustic_type_description"
        ] = bat_df["acoustic_type"].map(
            ACOUSTIC_TYPE_DESCRIPTIONS
        )
        bat_df["ABCD_model_agreement"] = (
            kmeans_types == classifier_types
        )

        for class_index, acoustic_type in enumerate(
            temporary_model.classes_
        ):
            bat_df[
                f"prob_type_{acoustic_type}"
            ] = type_probabilities[
                :,
                class_index,
            ]

        for index in bat_df.index:
            stage1_df.loc[
                index,
                "acoustic_type",
            ] = bat_df.loc[
                index,
                "acoustic_type",
            ]
            stage1_df.loc[
                index,
                "acoustic_type_description",
            ] = bat_df.loc[
                index,
                "acoustic_type_description",
            ]

    output_path = (
        DEMO_PREDICTION_DIR
        / f"{audio_path.stem}_noise_ABCD.csv"
    )
    stage1_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    bat_count = int(len(bat_df))
    noise_count = int(len(noise_df))
    total_count = int(len(stage1_df))
    bat_ratio = (
        bat_count / total_count
        if total_count
        else 0.0
    )

    if bat_count:
        type_counts = (
            bat_df["acoustic_type"]
            .value_counts()
        )
        major_type = str(type_counts.index[0])
        major_type_ratio = float(
            type_counts.iloc[0] / bat_count
        )
    else:
        type_counts = pd.Series(dtype=int)
        major_type = ""
        major_type_ratio = 0.0

    print(
        f"\n========== {audio_path.name} =========="
    )
    print(
        f"候选片段：{total_count}，"
        f"拆分时间候选：{split_candidate_count}，"
        f"bat：{bat_count}，"
        f"noise：{noise_count}"
    )
    print(f"bat占比：{bat_ratio:.2%}")

    if bat_count:
        print(
            f"主要声型：{major_type} "
            f"（{ACOUSTIC_TYPE_DESCRIPTIONS[major_type]}）"
        )
        print(
            f"主要声型占bat脉冲："
            f"{major_type_ratio:.2%}"
        )
        print("A-D数量：")
        print(
            type_counts.reindex(
                ["A", "B", "C", "D"],
                fill_value=0,
            )
        )
    else:
        print(
            "结论：当前候选主要被判为noise，"
            "没有足够bat脉冲进入A-D分类。"
        )

    print(f"逐片段结果：{output_path}")

    save_fast_overview_spectrogram(
        audio_path,
        y,
        sr,
        bat_count,
    )
    if bat_count:
        save_top_pulse_zoom_grid(
            audio_path,
            y,
            sr,
            bat_df.to_dict("records"),
        )

    result: dict[str, object] = {
        "source_file": audio_path.name,
        "status": (
            "success"
            if bat_count
            else "noise_only"
        ),
        "candidate_count": total_count,
        "split_candidate_count": (
            split_candidate_count
        ),
        "bat_count": bat_count,
        "noise_count": noise_count,
        "bat_ratio": bat_ratio,
        "major_acoustic_type": major_type,
        "major_type_ratio": major_type_ratio,
        "prediction_csv": str(output_path),
    }

    for acoustic_type in ["A", "B", "C", "D"]:
        count = int(
            type_counts.get(acoustic_type, 0)
        )
        result[
            f"type_{acoustic_type}_count"
        ] = count
        result[
            f"type_{acoustic_type}_ratio"
        ] = (
            count / bat_count
            if bat_count
            else 0.0
        )

    return result


def predict_demo_path_two_stage(
    input_path: Path,
    combined_payload: dict[str, object],
) -> None:
    """支持单个WAV或整个文件夹的两级预测。"""
    if input_path.is_file():
        wav_files = (
            [input_path]
            if input_path.suffix.lower() == ".wav"
            else []
        )
    elif input_path.is_dir():
        wav_files = find_wav_files(input_path)
    else:
        print(f"路径不存在：{input_path}")
        return

    if not wav_files:
        print("没有找到WAV文件。")
        return

    summary_rows: list[dict[str, object]] = []

    for audio_path in wav_files:
        try:
            summary_rows.append(
                predict_one_demo_wav_two_stage(
                    audio_path,
                    combined_payload,
                )
            )
        except Exception as error:
            print(
                f"预测失败 {audio_path.name}：{error}"
            )
            summary_rows.append(
                {
                    "source_file": audio_path.name,
                    "status": "failed",
                    "error": str(error),
                }
            )

    summary_path = (
        DEMO_PREDICTION_DIR
        / "noise_ABCD_prediction_summary.csv"
    )
    pd.DataFrame(summary_rows).to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    print(
        f"\n两级预测汇总：{summary_path}"
    )
