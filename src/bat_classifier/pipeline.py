"""两级流程编排入口（兼容 run.py 与 python -m bat_classifier）。

只负责串联完整流程：
数据准备 → 脉冲建表 → noise/bat 门控训练 → A-D 声型聚类
→ 保存两级模型 → 新录音预测。算法细节见各功能模块。
"""

from __future__ import annotations

from pathlib import Path

import joblib

from .clustering import cluster_acoustic_types
from .config import BASE_DIR, COMBINED_DEMO_MODEL_PATH
from .noise_gate import train_noise_gate
from .prediction import predict_demo_path_two_stage
from .pulse_table import build_demo_pulse_table, prepare_demo_dataset


def main() -> None:
    print(
        "======= 两级蝙蝠声学原型系统 ======="
    )
    print(
        "第一层：noise / bat；"
        "第二层：bat脉冲自动归入声型A-D；"
        "检测前自动拆分同时近频声波。"
    )
    print(
        "A-D是临时声型标签，不是已确认物种。"
    )

    prepare_demo_dataset()
    pulse_df, detection_df = (
        build_demo_pulse_table()
    )

    noise_payload = train_noise_gate(pulse_df)
    cluster_result = cluster_acoustic_types(
        pulse_df,
        detection_df,
    )
    acoustic_payload = cluster_result["payload"]

    combined_payload = {
        "noise_gate": noise_payload,
        "acoustic_types": acoustic_payload,
        "pipeline": [
            "candidate_detection",
            "simultaneous_near_frequency_separation",
            "noise_or_bat",
            "bat_acoustic_type_A_to_D",
        ],
        "warning": (
            "当前第一层标签来自自动质量筛选，"
            "A-D来自无监督聚类；"
            "用于证明流程可行，不代表物种鉴定。"
        ),
    }
    joblib.dump(
        combined_payload,
        COMBINED_DEMO_MODEL_PATH,
    )

    print(
        f"\n完整两级模型："
        f"{COMBINED_DEMO_MODEL_PATH}"
    )

    raw_path = input(
        "\n输入新的WAV或文件夹路径进行"
        "noise/bat + A-D预测，直接回车结束："
    ).strip().strip('"').lstrip("\ufeff").strip()

    if not raw_path:
        return

    input_path = Path(raw_path)
    if not input_path.is_absolute():
        input_path = BASE_DIR / input_path

    predict_demo_path_two_stage(
        input_path,
        combined_payload,
    )
