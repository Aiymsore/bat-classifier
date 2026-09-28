"""演示数据准备与逐脉冲建表（上游：audio_io/detection/features/separation；下游：聚类与门控训练）。

职责：解压/检查演示录音、把每条录音的候选脉冲展开成逐行明细表
（含分离分量），并输出 recording 级检测汇总 CSV。
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .audio_io import (
    calculate_audio_content_hash,
    find_wav_files,
    load_audio,
)
from .detection import detect_and_refine_pulses_chunked
from .features import evaluate_pulse_quality, extract_pulse_features
from .separation import (
    _unchanged_component,
    clear_separated_pulses_for_source,
    save_separated_pulse,
    separate_simultaneous_pulse_components,
)
from .visualization import (
    save_fast_overview_spectrogram,
    save_top_pulse_zoom_grid,
)
from .config import (
    AUDIO_OUTPUT_DIR,
    CLUSTER_EXAMPLE_DIR,
    DEMO_PREDICTION_DIR,
    DEMO_PULSE_CSV,
    DEMO_RECORDING_CSV,
    DEMO_ZIP_CANDIDATES,
    FIGURE_DIR,
    MAX_REFINED_PULSES_PER_RECORDING,
    MODEL_DIR,
    PULSE_ZOOM_DIR,
    RAW_DATA_DIR,
    RESULTS_DIR,
    SAVE_PER_RECORDING_DIAGNOSTICS,
    SAVE_SEPARATED_PULSE_AUDIO,
    SEPARATED_PULSE_DIR,
    SPECTROGRAM_DIR,
    TABLE_DIR,
)


def _clear_raw_data_dir() -> None:
    """清空原始数据目录内容（带防越界检查），用于切换训练数据来源。"""
    if not RAW_DATA_DIR.exists():
        return

    resolved_root = RAW_DATA_DIR.resolve()
    for item in RAW_DATA_DIR.iterdir():
        resolved_item = item.resolve()
        if (
            resolved_item == resolved_root
            or resolved_root not in resolved_item.parents
        ):
            raise RuntimeError(
                f"拒绝清理异常数据路径：{resolved_item}"
            )
        if item.is_dir():
            shutil.rmtree(resolved_item)
        else:
            item.unlink()


def prepare_demo_dataset(
    source: Path | None = None,
) -> int:
    """准备训练用的演示录音数据，返回 WAV 数量。

    source 为 None 时的行为与历史版本一致：
    data/raw 已有 WAV 则直接使用，否则从 data/ 下的默认压缩包解压。

    source 指定时的行为：
    - 指向 zip 文件：清空 data/raw 后解压到其中；
    - 指向文件夹：清空 data/raw 后复制其中的全部 WAV（含子目录）。
    """
    for directory in (
        RAW_DATA_DIR,
        RESULTS_DIR,
        TABLE_DIR,
        FIGURE_DIR,
        AUDIO_OUTPUT_DIR,
        MODEL_DIR,
        SPECTROGRAM_DIR,
        PULSE_ZOOM_DIR,
        DEMO_PREDICTION_DIR,
        CLUSTER_EXAMPLE_DIR,
        SEPARATED_PULSE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    if source is not None:
        source = Path(source)
        if source.is_file() and source.suffix.lower() == ".zip":
            _clear_raw_data_dir()
            print(
                f"正在解压训练数据："
                f"{source.name} -> {RAW_DATA_DIR.name}"
            )
            with zipfile.ZipFile(source, "r") as zf:
                zf.extractall(RAW_DATA_DIR)
        elif source.is_dir():
            wav_files = find_wav_files(source)
            if not wav_files:
                raise FileNotFoundError(
                    f"{source} 中没有找到WAV文件。"
                )
            _clear_raw_data_dir()
            for wav_path in wav_files:
                destination = RAW_DATA_DIR / wav_path.name
                shutil.copy2(wav_path, destination)
            print(
                f"已从 {source.name} 导入 "
                f"{len(wav_files)} 个WAV。"
            )
        else:
            raise FileNotFoundError(
                f"训练数据来源无效（需要 zip 文件或文件夹）：{source}"
            )

        wav_files = find_wav_files(RAW_DATA_DIR)
        if not wav_files:
            raise RuntimeError(
                "数据源导入完成，但没有找到WAV文件。"
            )
        return len(wav_files)

    existing_wavs = find_wav_files(
        RAW_DATA_DIR
    )
    if existing_wavs:
        print(
            f"检测到已有演示数据："
            f"{len(existing_wavs)} 个WAV"
        )
        return len(existing_wavs)

    zip_path = next(
        (
            candidate
            for candidate
            in DEMO_ZIP_CANDIDATES
            if candidate.exists()
        ),
        None,
    )
    if zip_path is None:
        expected_names = "、".join(
            path.name
            for path in DEMO_ZIP_CANDIDATES
        )
        raise FileNotFoundError(
            f"没有找到 {expected_names}。"
            "请把压缩包放入项目的 data 目录，"
            f"或者把WAV手动放入 {RAW_DATA_DIR}。"
        )

    print(
        f"正在解压演示数据："
        f"{zip_path.name} -> {RAW_DATA_DIR.name}"
    )
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DATA_DIR)

    wav_files = find_wav_files(RAW_DATA_DIR)
    if not wav_files:
        raise RuntimeError(
            "压缩包解压成功，但没有找到WAV文件。"
        )

    print(
        f"演示数据准备完成："
        f"{len(wav_files)} 个WAV"
    )
    return len(wav_files)


def create_demo_pulse_rows(
    audio_path: Path,
    y: np.ndarray,
    sr: int,
    pulses: list[dict[str, float]],
) -> list[dict[str, float | int | str | bool]]:
    """
    Step6轻量版脉冲建表。

    普通单脉冲只提取特征；若一个时间候选中检测到多条
    同时发声轨迹，则展开成多行，并保存各自的分离WAV。
    这样既保留原来的轻量流程，又能审听分离结果。
    """
    if len(pulses) > MAX_REFINED_PULSES_PER_RECORDING:
        pulses = sorted(
            pulses,
            key=lambda pulse: float(
                pulse.get("snr_db", -999.0)
            ),
            reverse=True,
        )[:MAX_REFINED_PULSES_PER_RECORDING]
        pulses = sorted(
            pulses,
            key=lambda pulse: float(
                pulse["start_s"]
            ),
        )

    rows: list[
        dict[str, float | int | str | bool]
    ] = []

    pulse_id = 0

    for candidate_id, pulse in enumerate(
        pulses,
        start=1,
    ):
        pulse_start_sample = max(
            0,
            int(round(
                float(pulse["start_s"]) * sr
            )),
        )
        pulse_end_sample = min(
            len(y),
            int(round(
                float(pulse["end_s"]) * sr
            )),
        )
        pulse_y = y[
            pulse_start_sample:pulse_end_sample
        ]

        try:
            components = (
                separate_simultaneous_pulse_components(
                    pulse_y,
                    sr,
                )
            )
        except Exception:
            components = [
                _unchanged_component(pulse_y, 0.0)
            ]

        # 轨迹几何上可拆并不等于两个分量都有足够声学质量。
        # 若第二条轨迹通不过原有质量门槛，则回退到原混合脉冲，
        # 防止把宽带单脉冲、回声或噪声旁瓣误拆后改变旧结果。
        if len(components) > 1:
            qualified_component_count = 0
            for component in components:
                preview_audio = np.asarray(
                    component["audio"],
                    dtype=np.float32,
                )
                try:
                    preview_features = (
                        extract_pulse_features(
                            preview_audio,
                            sr,
                        )
                    )
                except Exception:
                    continue

                component["_preview_features"] = (
                    preview_features
                )

                preview_pulse = dict(pulse)
                preview_pulse["duration_ms"] = (
                    len(preview_audio) / sr * 1000
                )
                preview_pulse["snr_db"] = max(
                    0.0,
                    float(pulse["snr_db"])
                    + float(
                        component[
                            "track_relative_db"
                        ]
                    ),
                )
                preview_accepted, _ = (
                    evaluate_pulse_quality(
                        preview_pulse,
                        preview_features,
                    )
                )
                qualified_component_count += int(
                    preview_accepted
                )

            if qualified_component_count < 2:
                components = [
                    _unchanged_component(
                        pulse_y,
                        1.0,
                    )
                ]

        for component in components:
            component_audio = np.asarray(
                component["audio"],
                dtype=np.float32,
            )
            if len(component_audio) < 32:
                continue

            cached_features = component.get(
                "_preview_features"
            )
            if cached_features is not None:
                features = dict(cached_features)
            else:
                try:
                    features = extract_pulse_features(
                        component_audio,
                        sr,
                    )
                except Exception:
                    continue

            relative_db = float(
                component["track_relative_db"]
            )
            component_start_sample = (
                pulse_start_sample
                + int(component["start_sample"])
            )
            component_end_sample = (
                pulse_start_sample
                + int(component["end_sample"])
            )
            component_pulse = dict(pulse)
            component_pulse["start_s"] = (
                component_start_sample / sr
            )
            component_pulse["end_s"] = (
                component_end_sample / sr
            )
            component_pulse["duration_ms"] = (
                component_end_sample
                - component_start_sample
            ) / sr * 1000
            component_pulse["snr_db"] = max(
                0.0,
                float(pulse["snr_db"])
                + relative_db,
            )

            accepted, rejection_reason = (
                evaluate_pulse_quality(
                    component_pulse,
                    features,
                )
            )
            pulse_id += 1

            pulse_file = ""
            if (
                SAVE_SEPARATED_PULSE_AUDIO
                and accepted
                and bool(
                    component["separation_applied"]
                )
            ):
                pulse_file = save_separated_pulse(
                    audio_path,
                    candidate_id,
                    int(component["component_id"]),
                    component_audio,
                    sr,
                )

            rows.append(
                {
                    "source_file": audio_path.name,
                    "source_path": str(audio_path),
                    "species": "unlabeled",
                    "pulse_id": pulse_id,
                    "candidate_id": candidate_id,
                    "component_id": int(
                        component["component_id"]
                    ),
                    "component_count": int(
                        component["component_count"]
                    ),
                    "separation_applied": bool(
                        component["separation_applied"]
                    ),
                    "separation_confidence": float(
                        component[
                            "separation_confidence"
                        ]
                    ),
                    "track_median_freq_khz": float(
                        component[
                            "track_median_freq_khz"
                        ]
                    ),
                    "track_relative_db": relative_db,
                    "pulse_file": pulse_file,
                    "accepted": accepted,
                    "rejection_reason": (
                        rejection_reason
                    ),
                    "candidate_start_s": float(
                        pulse["candidate_start_s"]
                    ),
                    "candidate_end_s": float(
                        pulse["candidate_end_s"]
                    ),
                    "start_s": float(
                        component_pulse["start_s"]
                    ),
                    "end_s": float(
                        component_pulse["end_s"]
                    ),
                    "duration_ms": float(
                        component_pulse["duration_ms"]
                    ),
                    "snr_db": float(
                        component_pulse["snr_db"]
                    ),
                    "core_threshold_db": float(
                        pulse["core_threshold_db"]
                    ),
                    **features,
                }
            )

    return rows


def build_demo_pulse_table(
    save_diagnostics: bool | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    对全部录音自动检测、精修并切割脉冲。

    save_diagnostics：是否为每条录音生成概览声谱图与脉冲放大图；
    None 时沿用 config 的 SAVE_PER_RECORDING_DIAGNOSTICS 默认值。

    返回：
    - pulse_df：每个候选脉冲一行；
    - recording_df：每个原始录音一行的检测汇总。
    """
    use_diagnostics = (
        SAVE_PER_RECORDING_DIAGNOSTICS
        if save_diagnostics is None
        else save_diagnostics
    )
    wav_files = find_wav_files(RAW_DATA_DIR)
    if not wav_files:
        raise FileNotFoundError(
            f"{RAW_DATA_DIR} 中没有WAV文件。"
        )

    pulse_rows: list[
        dict[str, float | int | str | bool]
    ] = []
    recording_rows: list[
        dict[str, float | int | str]
    ] = []
    seen_audio_hashes: dict[str, str] = {}

    for index, audio_path in enumerate(
        wav_files,
        start=1,
    ):
        relative_source = str(
            audio_path.relative_to(
                RAW_DATA_DIR
            )
        )

        try:
            y, sr = load_audio(audio_path)
            audio_hash = calculate_audio_content_hash(
                y,
                sr,
            )

            if audio_hash in seen_audio_hashes:
                print(
                    f"[{index:02d}/{len(wav_files)}] "
                    f"跳过重复音频：{relative_source} "
                    f"== {seen_audio_hashes[audio_hash]}"
                )
                continue

            seen_audio_hashes[
                audio_hash
            ] = relative_source

            (
                raw_candidate_count,
                refined_pulses,
                threshold_db,
            ) = detect_and_refine_pulses_chunked(
                y,
                sr,
            )

            clear_separated_pulses_for_source(
                audio_path
            )
            current_rows = create_demo_pulse_rows(
                audio_path,
                y,
                sr,
                refined_pulses,
            )

            for row in current_rows:
                row["source_file"] = relative_source
                row["group_id"] = relative_source
                row["source_path"] = str(audio_path)
                row["temporary_label"] = ""

            pulse_rows.extend(current_rows)

            accepted_rows = [
                row
                for row in current_rows
                if bool(row["accepted"])
            ]
            split_candidate_ids = {
                int(row["candidate_id"])
                for row in current_rows
                if bool(row["separation_applied"])
            }
            separated_component_count = sum(
                bool(row["separation_applied"])
                for row in current_rows
            )

            if use_diagnostics:
                save_fast_overview_spectrogram(
                    audio_path,
                    y,
                    sr,
                    len(accepted_rows),
                )
                save_top_pulse_zoom_grid(
                    audio_path,
                    y,
                    sr,
                    accepted_rows,
                )

            recording_rows.append(
                {
                    "source_file": relative_source,
                    "sample_rate_hz": sr,
                    "duration_s": len(y) / sr,
                    "pulse_threshold_db": threshold_db,
                    "raw_candidate_count": (
                        raw_candidate_count
                    ),
                    "refined_candidate_count": len(
                        refined_pulses
                    ),
                    "split_candidate_count": len(
                        split_candidate_ids
                    ),
                    "separated_component_count": (
                        separated_component_count
                    ),
                    "accepted_pulse_count": len(
                        accepted_rows
                    ),
                    "rejected_pulse_count": (
                        len(current_rows)
                        - len(accepted_rows)
                    ),
                }
            )

            print(
                f"[{index:02d}/{len(wav_files)}] "
                f"{relative_source}："
                f"原始候选 {raw_candidate_count}，"
                f"精修 {len(refined_pulses)}，"
                f"拆分候选 {len(split_candidate_ids)}，"
                f"合格分量 {len(accepted_rows)}"
            )

        except Exception as error:
            print(
                f"[{index:02d}/{len(wav_files)}] "
                f"跳过 {relative_source}：{error}"
            )

    pulse_df = pd.DataFrame(pulse_rows)
    recording_df = pd.DataFrame(
        recording_rows
    )

    if pulse_df.empty:
        raise RuntimeError(
            "没有成功提取任何候选脉冲。"
        )

    pulse_df.to_csv(
        DEMO_PULSE_CSV,
        index=False,
        encoding="utf-8-sig",
    )
    recording_df.to_csv(
        DEMO_RECORDING_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\n脉冲明细已保存：{DEMO_PULSE_CSV}"
    )
    print(
        f"录音检测汇总：{DEMO_RECORDING_CSV}"
    )

    return pulse_df, recording_df
