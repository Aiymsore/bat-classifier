from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .config import (
    ACOUSTIC_TYPE_DESCRIPTIONS,
    CLUSTER_FEATURE_COLUMNS,
    MODEL_DIR,
    NOISE_BAT_PROBABILITY_THRESHOLD,
    NOISE_GATE_FEATURE_COLUMNS,
    RANDOM_STATE,
    RESULTS_DIR,
    TABLE_DIR,
    DEMO_PREDICTION_DIR,
    SPECTROGRAM_DIR,
    PULSE_ZOOM_DIR,
    SEPARATED_PULSE_DIR,
)
from .pipeline import accepted_mask, predict_one_demo_wav_two_stage

DEFAULT_MODEL_PATH = MODEL_DIR / "demo_two_stage_model.joblib"
DEFAULT_SAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "sample"
    / "demo_xianren_0_4s.wav"
)
DEFAULT_SUMMARY_PATH = RESULTS_DIR / "predictions" / "demo_summary.json"


def _numeric_frame(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = data[columns].copy()
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.replace([np.inf, -np.inf], np.nan)


def _build_noise_gate(pulse_table: pd.DataFrame) -> dict[str, Any]:
    required = {"accepted", *NOISE_GATE_FEATURE_COLUMNS}
    missing = required - set(pulse_table.columns)
    if missing:
        raise RuntimeError(
            "acoustic_pulses.csv 缺少第一层模型字段："
            f"{sorted(missing)}"
        )

    features = _numeric_frame(pulse_table, list(NOISE_GATE_FEATURE_COLUMNS))
    labels = np.where(accepted_mask(pulse_table["accepted"]), "bat", "noise")
    valid = features.notna().all(axis=1)
    features = features.loc[valid]
    labels = labels[valid.to_numpy()]

    class_counts = pd.Series(labels).value_counts()
    if set(class_counts.index) != {"bat", "noise"}:
        raise RuntimeError(f"noise/bat 类别不完整：{class_counts.to_dict()}")

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(features, labels)

    return {
        "model": model,
        "feature_columns": list(NOISE_GATE_FEATURE_COLUMNS),
        "classes": list(model.classes_),
        "bat_probability_threshold": NOISE_BAT_PROBABILITY_THRESHOLD,
        "training_rows": int(len(features)),
        "warning": (
            "当前 noise/bat 标签来自自动质量筛选，"
            "只用于展示两级推理流程。"
        ),
    }


def _unique_cluster_mapping(
    raw_clusters: np.ndarray,
    target_labels: pd.Series,
) -> dict[int, str]:
    acoustic_types = ["A", "B", "C", "D"]
    contingency = np.zeros((4, 4), dtype=int)

    for cluster_id in range(4):
        cluster_labels = target_labels.loc[raw_clusters == cluster_id]
        counts = cluster_labels.value_counts()
        for type_index, acoustic_type in enumerate(acoustic_types):
            contingency[cluster_id, type_index] = int(
                counts.get(acoustic_type, 0)
            )

    row_index, column_index = linear_sum_assignment(-contingency)
    return {
        int(row): acoustic_types[int(column)]
        for row, column in zip(row_index, column_index)
    }


def _build_acoustic_classifier(cluster_table: pd.DataFrame) -> dict[str, Any]:
    required = {"acoustic_type", *CLUSTER_FEATURE_COLUMNS}
    missing = required - set(cluster_table.columns)
    if missing:
        raise RuntimeError(
            "acoustic_clusters_ABCD.csv 缺少第二层模型字段："
            f"{sorted(missing)}"
        )

    features = _numeric_frame(cluster_table, list(CLUSTER_FEATURE_COLUMNS))
    labels = cluster_table["acoustic_type"].astype(str)
    valid = features.notna().all(axis=1) & labels.isin(["A", "B", "C", "D"])
    features = features.loc[valid]
    labels = labels.loc[valid]

    if labels.nunique() != 4:
        raise RuntimeError(
            "A-D 声型标签不完整："
            f"{labels.value_counts().to_dict()}"
        )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(
        n_clusters=4,
        n_init=30,
        random_state=RANDOM_STATE,
    )
    raw_clusters = kmeans.fit_predict(scaled)
    mapping = _unique_cluster_mapping(raw_clusters, labels)

    classifier = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    classifier.fit(features, labels)

    return {
        "scaler": scaler,
        "kmeans": kmeans,
        "temporary_classifier": classifier,
        "raw_cluster_to_type": mapping,
        "feature_columns": list(CLUSTER_FEATURE_COLUMNS),
        "type_descriptions": ACOUSTIC_TYPE_DESCRIPTIONS,
        "training_rows": int(len(features)),
        "warning": (
            "A-D 是无监督聚类形成的临时声型，"
            "不是经过专家确认的物种标签。"
        ),
    }


def build_demo_model(
    model_path: Path = DEFAULT_MODEL_PATH,
    force: bool = False,
) -> dict[str, Any]:
    """从仓库自带结果表构建可复现的两级演示模型。"""
    model_path = Path(model_path)
    if model_path.exists() and not force:
        return joblib.load(model_path)

    pulse_csv = TABLE_DIR / "acoustic_pulses.csv"
    cluster_csv = TABLE_DIR / "acoustic_clusters_ABCD.csv"
    if not pulse_csv.exists() or not cluster_csv.exists():
        raise FileNotFoundError(
            "缺少演示训练表。请确认 results/tables/ 下存在 "
            "acoustic_pulses.csv 和 acoustic_clusters_ABCD.csv。"
        )

    pulse_table = pd.read_csv(pulse_csv)
    cluster_table = pd.read_csv(cluster_csv)

    payload = {
        "noise_gate": _build_noise_gate(pulse_table),
        "acoustic_types": _build_acoustic_classifier(cluster_table),
        "pipeline": [
            "candidate_detection",
            "noise_or_bat",
            "bat_acoustic_type_A_to_D",
        ],
        "built_from": {
            "pulse_table": str(pulse_csv.relative_to(pulse_csv.parents[2])),
            "cluster_table": str(cluster_csv.relative_to(cluster_csv.parents[2])),
        },
        "warning": (
            "演示模型使用仓库结果表中的伪标签训练，"
            "用于展示代码流程，不代表真实物种识别性能。"
        ),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, model_path)
    return payload


def run_demo(
    input_path: Path = DEFAULT_SAMPLE_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    rebuild_model: bool = False,
) -> dict[str, Any]:
    input_path = Path(input_path).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"演示音频不存在：{input_path}")
    if input_path.suffix.lower() != ".wav":
        raise ValueError("演示输入必须是 WAV 文件。")

    for directory in [
        MODEL_DIR,
        DEMO_PREDICTION_DIR,
        SPECTROGRAM_DIR,
        PULSE_ZOOM_DIR,
        SEPARATED_PULSE_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    payload = build_demo_model(model_path, force=rebuild_model)
    result = predict_one_demo_wav_two_stage(input_path, payload)

    summary = {
        **result,
        "input_path": str(input_path),
        "model_path": str(model_path),
        "warning": payload["warning"],
    }
    DEFAULT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行蝙蝠声学两级分类演示。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_SAMPLE_PATH,
        help="待预测 WAV；默认使用仓库自带 4 秒样例。",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="演示模型保存路径。",
    )
    parser.add_argument(
        "--rebuild-model",
        action="store_true",
        help="忽略已有模型并从结果表重新训练。",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="运行后检查结果文件与候选片段，失败时返回非零退出码。",
    )
    return parser


def cli(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print("======= Bat Classifier Demo =======")
    print(f"输入：{args.input}")
    print("说明：A-D 是临时声型，不是真实物种。")

    try:
        summary = run_demo(
            input_path=args.input,
            model_path=args.model,
            rebuild_model=args.rebuild_model,
        )
    except Exception as error:
        print(f"Demo 运行失败：{error}")
        return 1

    print(f"\nDemo 摘要：{DEFAULT_SUMMARY_PATH}")

    if args.self_test:
        prediction_csv = Path(str(summary.get("prediction_csv", "")))
        candidate_count = int(summary.get("candidate_count", 0) or 0)
        allowed_status = {"success", "noise_only"}
        checks = {
            "status_valid": summary.get("status") in allowed_status,
            "candidate_count_positive": candidate_count > 0,
            "prediction_csv_exists": prediction_csv.exists(),
            "summary_json_exists": DEFAULT_SUMMARY_PATH.exists(),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            print("Self-test 失败：" + ", ".join(failed))
            return 2
        print("Self-test 通过：模型构建、音频检测、两级预测和结果写入均正常。")

    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
