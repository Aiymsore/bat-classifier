"""候选脉冲检测与边界精修（上游：audio_io；下游：pulse_table、prediction）。

三节内容：
1. 粗检：按超声频段能量阈值找出候选脉冲（detect_pulses）；
2. 精修：在候选区内重找高信噪比脉冲核心（refine_pulse_boundary 等）；
3. 分块检测：长录音抽窗分析并去重（detect_and_refine_pulses_chunked）。
"""

from __future__ import annotations

import librosa
import numpy as np

from .audio_io import (
    adaptive_stft_params,
    build_spectrogram,
    interval_overlap_ratio,
    ultrasonic_band_mask,
)
from .config import (
    CORE_BACKGROUND_MARGIN_DB,
    CORE_CONTEXT_MS,
    CORE_MAX_GAP_MS,
    CORE_MAX_THRESHOLD_DB,
    CORE_PADDING_MS,
    CORE_THRESHOLD_DB,
    DETECTION_CHUNK_OVERLAP_MS,
    DETECTION_CHUNK_SECONDS,
    ENERGY_THRESHOLD_DB,
    MAX_ANALYSIS_WINDOWS_PER_RECORDING,
    MAX_PULSE_MS,
    MERGE_GAP_MS,
    MIN_PULSE_MS,
    PULSE_PADDING_MS,
)


def merge_active_frames(
    active_mask: np.ndarray,
    max_gap_frames: int,
) -> list[tuple[int, int]]:
    """把相邻或间隔很短的活动帧合并。"""
    active_indices = np.flatnonzero(active_mask)
    if len(active_indices) == 0:
        return []

    groups: list[tuple[int, int]] = []
    start = int(active_indices[0])
    previous = start

    for current_value in active_indices[1:]:
        current = int(current_value)

        if current - previous <= max_gap_frames + 1:
            previous = current
            continue

        groups.append((start, previous))
        start = current
        previous = current

    groups.append((start, previous))
    return groups


def detect_pulses(
    y: np.ndarray,
    sr: int,
) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray, np.ndarray, float]:
    """
    从声谱图能量中检测疑似脉冲。

    返回：
    - 脉冲时间范围
    - 功率谱
    - 频率轴
    - 时间轴
    - 检测阈值
    """
    power, frequencies, times, hop_length = build_spectrogram(y, sr)

    frequency_mask = ultrasonic_band_mask(frequencies, sr)

    ultrasonic_power = np.sum(power[frequency_mask, :], axis=0)
    energy_db = librosa.power_to_db(
        ultrasonic_power + 1e-18,
        ref=np.max,
    )

    # 轻微平滑，减少孤立噪点
    if len(energy_db) >= 5:
        energy_db = np.convolve(
            energy_db,
            np.ones(5) / 5,
            mode="same",
        )

    # 同时使用固定相对阈值和背景自适应阈值
    background_db = float(np.median(energy_db))
    adaptive_threshold = background_db + 10.0
    threshold_db = max(ENERGY_THRESHOLD_DB, adaptive_threshold)

    active_mask = energy_db >= threshold_db

    frame_ms = hop_length / sr * 1000
    max_gap_frames = max(
        1,
        int(round(MERGE_GAP_MS / frame_ms)),
    )

    frame_groups = merge_active_frames(
        active_mask,
        max_gap_frames,
    )

    pulses: list[dict[str, float]] = []
    audio_duration_s = len(y) / sr
    padding_s = PULSE_PADDING_MS / 1000

    for start_frame, end_frame in frame_groups:
        start_s = max(
            0.0,
            float(times[start_frame]) - padding_s,
        )
        end_s = min(
            audio_duration_s,
            float(times[end_frame]) + frame_ms / 1000 + padding_s,
        )
        duration_ms = (end_s - start_s) * 1000

        if not (MIN_PULSE_MS <= duration_ms <= MAX_PULSE_MS):
            continue

        pulses.append(
            {
                "start_s": start_s,
                "end_s": end_s,
                "duration_ms": duration_ms,
            }
        )

    return pulses, power, frequencies, times, threshold_db


def select_group_containing_peak(
    groups: list[tuple[int, int]],
    peak_frame: int,
) -> tuple[int, int] | None:
    """优先选择包含最高能量帧的连续活动区。"""
    for start_frame, end_frame in groups:
        if start_frame <= peak_frame <= end_frame:
            return start_frame, end_frame

    if not groups:
        return None

    return min(
        groups,
        key=lambda group: min(
            abs(peak_frame - group[0]),
            abs(peak_frame - group[1]),
        ),
    )


def refine_pulse_boundary(
    y: np.ndarray,
    sr: int,
    pulse: dict[str, float],
) -> dict[str, float] | None:
    """
    在候选区内部重新寻找高信噪比的脉冲核心。

    这一步用于去掉候选窗口前后的背景拖尾，使切出的WAV更接近单个真实脉冲。
    """
    candidate_start_sample = max(
        0,
        int(round(pulse["start_s"] * sr)),
    )
    candidate_end_sample = min(
        len(y),
        int(round(pulse["end_s"] * sr)),
    )

    context_samples = int(round(CORE_CONTEXT_MS / 1000 * sr))
    analysis_start_sample = max(
        0,
        candidate_start_sample - context_samples,
    )
    analysis_end_sample = min(
        len(y),
        candidate_end_sample + context_samples,
    )
    candidate_y = y[analysis_start_sample:analysis_end_sample]

    if len(candidate_y) < 64:
        return None

    n_fft, hop_length = adaptive_stft_params(
        len(candidate_y),
        max_n_fft=256,
        min_n_fft=64,
        min_hop=4,
    )

    stft = librosa.stft(
        candidate_y,
        n_fft=n_fft,
        hop_length=hop_length,
        window="hann",
        center=True,
    )
    power = np.abs(stft) ** 2
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    frequency_mask = ultrasonic_band_mask(frequencies, sr)
    band_power = np.sum(power[frequency_mask, :], axis=0)

    if band_power.size == 0 or float(np.max(band_power)) <= 0:
        return None

    energy_db = librosa.power_to_db(
        band_power + 1e-18,
        ref=np.max,
    )
    background_db = float(np.percentile(energy_db, 20))
    threshold_db = min(
        CORE_MAX_THRESHOLD_DB,
        max(
            CORE_THRESHOLD_DB,
            background_db + CORE_BACKGROUND_MARGIN_DB,
        ),
    )
    active_mask = energy_db >= threshold_db

    frame_ms = hop_length / sr * 1000
    max_gap_frames = max(
        1,
        int(round(CORE_MAX_GAP_MS / frame_ms)),
    )
    groups = merge_active_frames(active_mask, max_gap_frames)
    peak_frame = int(np.argmax(band_power))
    selected_group = select_group_containing_peak(groups, peak_frame)

    if selected_group is None:
        return None

    start_frame, end_frame = selected_group
    core_start_sample = (
        analysis_start_sample
        + max(0, start_frame * hop_length - n_fft // 2)
    )
    core_end_sample = (
        analysis_start_sample
        + min(
            len(candidate_y),
            end_frame * hop_length + n_fft // 2,
        )
    )

    padding_samples = int(round(CORE_PADDING_MS / 1000 * sr))
    core_start_sample = max(0, core_start_sample - padding_samples)
    core_end_sample = min(len(y), core_end_sample + padding_samples)

    duration_ms = (
        core_end_sample - core_start_sample
    ) / sr * 1000

    if not (MIN_PULSE_MS <= duration_ms <= MAX_PULSE_MS):
        return None

    core_frame_values = energy_db[start_frame : end_frame + 1]
    outside_values = np.concatenate(
        [
            energy_db[:start_frame],
            energy_db[end_frame + 1 :],
        ]
    )

    signal_level_db = float(np.median(core_frame_values))
    if len(outside_values):
        noise_level_db = float(np.median(outside_values))
    else:
        noise_level_db = background_db

    snr_db = signal_level_db - noise_level_db

    return {
        "candidate_start_s": pulse["start_s"],
        "candidate_end_s": pulse["end_s"],
        "start_s": core_start_sample / sr,
        "end_s": core_end_sample / sr,
        "duration_ms": duration_ms,
        "snr_db": snr_db,
        "core_threshold_db": threshold_db,
    }


def refine_all_pulses(
    y: np.ndarray,
    sr: int,
    pulses: list[dict[str, float]],
) -> list[dict[str, float]]:
    """对所有候选脉冲进行边界精修，并删除重叠的重复候选。"""
    refined: list[dict[str, float]] = []

    for pulse in pulses:
        result = refine_pulse_boundary(y, sr, pulse)
        if result is not None:
            refined.append(result)

    refined.sort(key=lambda item: item["start_s"])

    deduplicated: list[dict[str, float]] = []
    for pulse in refined:
        if not deduplicated:
            deduplicated.append(pulse)
            continue

        previous = deduplicated[-1]
        overlap_ratio = interval_overlap_ratio(
            previous["start_s"],
            previous["end_s"],
            pulse["start_s"],
            pulse["end_s"],
        )

        if overlap_ratio >= 0.60:
            if pulse["snr_db"] > previous["snr_db"]:
                deduplicated[-1] = pulse
            continue

        deduplicated.append(pulse)

    return deduplicated


def remove_duplicate_boundary_pulses(
    pulses: list[dict[str, float]],
) -> list[dict[str, float]]:
    """删除分块重叠区域中重复检测到的同一脉冲。"""
    if not pulses:
        return []

    pulses = sorted(
        pulses,
        key=lambda pulse: (
            float(pulse["start_s"]),
            float(pulse["end_s"]),
        ),
    )
    kept: list[dict[str, float]] = []

    for pulse in pulses:
        if not kept:
            kept.append(pulse)
            continue

        previous = kept[-1]
        overlap_ratio = interval_overlap_ratio(
            float(previous["start_s"]),
            float(previous["end_s"]),
            float(pulse["start_s"]),
            float(pulse["end_s"]),
        )
        center_distance_ms = abs(
            (
                float(previous["start_s"])
                + float(previous["end_s"])
            )
            / 2
            - (
                float(pulse["start_s"])
                + float(pulse["end_s"])
            )
            / 2
        ) * 1000

        if (
            overlap_ratio >= 0.50
            or center_distance_ms <= 1.5
        ):
            previous_snr = float(
                previous.get("snr_db", -999.0)
            )
            current_snr = float(
                pulse.get("snr_db", -999.0)
            )
            if current_snr > previous_snr:
                kept[-1] = pulse
            continue

        kept.append(pulse)

    return kept


def detect_and_refine_pulses_chunked(
    y: np.ndarray,
    sr: int,
) -> tuple[int, list[dict[str, float]], float]:
    """
    为原型演示抽取录音的开始、中间、结尾三个窗口进行检测。

    每个窗口默认6秒。这样60秒录音只分析约18秒，
    能显著缩短演示运行时间。拿到正式数据库后，可以把
    MAX_ANALYSIS_WINDOWS_PER_RECORDING 调大，或改为扫描全部录音。
    """
    chunk_samples = max(
        int(round(DETECTION_CHUNK_SECONDS * sr)),
        1,
    )
    audio_length = len(y)

    if audio_length <= chunk_samples:
        window_starts = [0]
    elif (
        audio_length
        <= chunk_samples
        * MAX_ANALYSIS_WINDOWS_PER_RECORDING
    ):
        overlap_samples = max(
            int(
                round(
                    DETECTION_CHUNK_OVERLAP_MS
                    / 1000
                    * sr
                )
            ),
            1,
        )
        step_samples = max(
            chunk_samples - overlap_samples,
            1,
        )
        window_starts = list(
            range(
                0,
                audio_length,
                step_samples,
            )
        )
        if (
            window_starts
            and window_starts[-1]
            + chunk_samples
            < audio_length
        ):
            window_starts.append(
                max(
                    0,
                    audio_length
                    - chunk_samples,
                )
            )
    else:
        last_start = max(
            0,
            audio_length - chunk_samples,
        )
        window_starts = [
            int(round(value))
            for value in np.linspace(
                0,
                last_start,
                num=MAX_ANALYSIS_WINDOWS_PER_RECORDING,
            )
        ]

    window_starts = sorted(
        set(int(value) for value in window_starts)
    )

    all_refined: list[dict[str, float]] = []
    raw_candidate_count = 0
    thresholds: list[float] = []

    for start_sample in window_starts:
        end_sample = min(
            audio_length,
            start_sample + chunk_samples,
        )
        chunk_y = y[start_sample:end_sample]

        (
            raw_pulses,
            _,
            _,
            _,
            threshold_db,
        ) = detect_pulses(chunk_y, sr)
        refined = refine_all_pulses(
            chunk_y,
            sr,
            raw_pulses,
        )

        raw_candidate_count += len(raw_pulses)
        thresholds.append(float(threshold_db))
        offset_s = start_sample / sr

        for pulse in refined:
            adjusted = dict(pulse)
            for key in (
                "candidate_start_s",
                "candidate_end_s",
                "start_s",
                "end_s",
            ):
                adjusted[key] = (
                    float(adjusted[key])
                    + offset_s
                )
            all_refined.append(adjusted)

    refined_pulses = (
        remove_duplicate_boundary_pulses(
            all_refined
        )
    )
    mean_threshold = (
        float(np.mean(thresholds))
        if thresholds
        else float("nan")
    )
    return (
        raw_candidate_count,
        refined_pulses,
        mean_threshold,
    )
