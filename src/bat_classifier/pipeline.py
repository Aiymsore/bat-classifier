from __future__ import annotations

import hashlib
import re
import warnings
import zipfile
from pathlib import Path
import joblib
import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import (
    butter,
    find_peaks,
    medfilt,
    sosfilt,
    sosfiltfilt,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
import json
import shutil
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

matplotlib.use("Agg")

from .config import (
    BASE_DIR,
    DATA_DIR,
    RAW_DATA_DIR,
    RESULTS_DIR,
    TABLE_DIR,
    FIGURE_DIR,
    AUDIO_OUTPUT_DIR,
    MODEL_DIR,
    DEMO_ZIP_CANDIDATES,
    DEMO_DATA_DIR,
    DEMO_OUTPUT_DIR,
    DEMO_PULSE_CSV,
    DEMO_RECORDING_CSV,
    CLUSTERED_PULSE_CSV,
    RECORDING_CLUSTER_CSV,
    CLUSTER_CENTER_CSV,
    CLUSTER_REPRESENTATIVE_CSV,
    CLUSTER_PCA_PNG,
    CLUSTER_MODEL_PATH,
    CLUSTER_METADATA_JSON,
    CLUSTER_EXAMPLE_DIR,
    DEMO_PREDICTION_DIR,
    SPECTROGRAM_DIR,
    PULSE_ZOOM_DIR,
    CUT_PULSE_DIR,
    REJECTED_PULSE_DIR,
    SEPARATED_PULSE_DIR,
    NOISE_GATE_MODEL_PATH,
    COMBINED_DEMO_MODEL_PATH,
    NOISE_GATE_PREDICTIONS_CSV,
    TARGET_SAMPLE_RATE,
    LOW_FREQ_HZ,
    HIGH_FREQ_HZ,
    ENERGY_THRESHOLD_DB,
    MIN_PULSE_MS,
    MAX_PULSE_MS,
    MERGE_GAP_MS,
    PULSE_PADDING_MS,
    CORE_THRESHOLD_DB,
    CORE_BACKGROUND_MARGIN_DB,
    CORE_MAX_THRESHOLD_DB,
    CORE_CONTEXT_MS,
    CORE_PADDING_MS,
    CORE_MAX_GAP_MS,
    MIN_PULSE_SNR_DB,
    MIN_TRACK_COVERAGE,
    MAX_TRACK_JUMP_KHZ,
    TOP_PULSES_PER_FILE,
    ZOOM_CONTEXT_MS,
    CONTOUR_MIN_DB,
    RANDOM_STATE,
    CLUSTER_FEATURE_COLUMNS,
    N_ACOUSTIC_TYPES,
    MAX_CLUSTER_PULSES_PER_RECORDING,
    REPRESENTATIVE_PULSES_PER_TYPE,
    MIN_PULSES_FOR_CLUSTERING,
    ACOUSTIC_TYPE_DESCRIPTIONS,
    DETECTION_CHUNK_SECONDS,
    DETECTION_CHUNK_OVERLAP_MS,
    MAX_ANALYSIS_WINDOWS_PER_RECORDING,
    SAVE_PER_RECORDING_DIAGNOSTICS,
    DEMO_DETECTION_HOP_LENGTH,
    OVERVIEW_MAX_TIME_BINS,
    MAX_REFINED_PULSES_PER_RECORDING,
    ENABLE_SIMULTANEOUS_SEPARATION,
    SEPARATION_N_FFT,
    SEPARATION_WIN_LENGTH,
    SEPARATION_HOP_LENGTH,
    SEPARATION_MAX_COMPONENTS,
    SEPARATION_MAX_PEAKS_PER_FRAME,
    SEPARATION_MIN_FREQ_GAP_KHZ,
    SEPARATION_PEAK_PROMINENCE_DB,
    SEPARATION_FRAME_DYNAMIC_RANGE_DB,
    SEPARATION_GLOBAL_MIN_DB,
    SEPARATION_MAX_TRACK_JUMP_KHZ,
    SEPARATION_MAX_TRACK_GAP_FRAMES,
    SEPARATION_MIN_TRACK_MS,
    SEPARATION_MIN_TRACK_COVERAGE,
    SEPARATION_MIN_TEMPORAL_OVERLAP,
    SEPARATION_MIN_SECONDARY_RELATIVE_DB,
    SEPARATION_MASK_BANDWIDTH_KHZ,
    SEPARATION_HARMONIC_TOLERANCE,
    SAVE_SEPARATED_PULSE_AUDIO,
    NOISE_GATE_FEATURE_COLUMNS,
    MAX_NOISE_GATE_ROWS_PER_SOURCE_CLASS,
    MIN_NOISE_GATE_ROWS_PER_CLASS,
    NOISE_BAT_PROBABILITY_THRESHOLD,
)

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

def load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    """读取并统一采样率。"""
    y, sr = librosa.load(
        audio_path,
        sr=TARGET_SAMPLE_RATE,
        mono=True,
    )
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]

    if len(y) < 64:
        raise ValueError("音频过短或为空")

    y = y - np.mean(y)
    y = highpass_filter(y, sr)

    peak = float(np.max(np.abs(y)))
    if peak <= 0:
        raise ValueError("音频没有有效信号")

    return y / peak, sr


def highpass_filter(
    y: np.ndarray,
    sr: int,
    cutoff_hz: float = LOW_FREQ_HZ,
) -> np.ndarray:
    """滤除低频背景噪声。"""
    if len(y) < 32 or sr / 2 <= cutoff_hz:
        return y

    sos = butter(
        4,
        cutoff_hz,
        btype="highpass",
        fs=sr,
        output="sos",
    )

    try:
        return sosfiltfilt(sos, y)
    except ValueError:
        return sosfilt(sos, y)


def calculate_audio_content_hash(y: np.ndarray, sr: int) -> str:
    """
    对解码、统一采样率后的实际声音计算哈希。

    即使两个WAV的文件头或元数据不同，只要解码后的声音采样值相同，
    audio_hash仍会相同。
    """
    clean = np.asarray(y, dtype=np.float64)
    clean = np.nan_to_num(clean, nan=0.0, posinf=0.0, neginf=0.0)
    clean = np.clip(clean, -1.0, 1.0)
    pcm16 = np.round(clean * 32767.0).astype("<i2")

    digest = hashlib.sha256()
    digest.update(str(int(sr)).encode("ascii"))
    digest.update(pcm16.tobytes())
    return digest.hexdigest()


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

    frequency_mask = (
        (frequencies >= LOW_FREQ_HZ)
        & (frequencies <= min(HIGH_FREQ_HZ, sr / 2))
    )

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

    n_fft = min(
        256,
        2 ** int(np.floor(np.log2(len(candidate_y)))),
    )
    n_fft = max(64, n_fft)
    hop_length = max(4, n_fft // 16)

    stft = librosa.stft(
        candidate_y,
        n_fft=n_fft,
        hop_length=hop_length,
        window="hann",
        center=True,
    )
    power = np.abs(stft) ** 2
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    frequency_mask = (
        (frequencies >= LOW_FREQ_HZ)
        & (frequencies <= min(HIGH_FREQ_HZ, sr / 2))
    )
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
        overlap_s = min(
            previous["end_s"],
            pulse["end_s"],
        ) - max(
            previous["start_s"],
            pulse["start_s"],
        )

        shorter_duration_s = min(
            previous["end_s"] - previous["start_s"],
            pulse["end_s"] - pulse["start_s"],
        )

        overlap_ratio = (
            overlap_s / shorter_duration_s
            if shorter_duration_s > 0 and overlap_s > 0
            else 0.0
        )

        if overlap_ratio >= 0.60:
            if pulse["snr_db"] > previous["snr_db"]:
                deduplicated[-1] = pulse
            continue

        deduplicated.append(pulse)

    return deduplicated


def weighted_frequency_quantile(
    frequencies: np.ndarray,
    power: np.ndarray,
    quantile: float,
) -> float:
    total = float(np.sum(power))
    if total <= 0:
        return 0.0

    cumulative = np.cumsum(power) / total
    index = int(np.searchsorted(cumulative, quantile, side="left"))
    index = min(index, len(frequencies) - 1)
    return float(frequencies[index])


def extract_pulse_features(
    pulse_y: np.ndarray,
    sr: int,
) -> dict[str, float]:
    """只在可靠高能量帧中提取单脉冲特征。"""
    if len(pulse_y) < 32:
        raise ValueError("脉冲过短")

    n_fft = min(
        512,
        2 ** int(np.floor(np.log2(len(pulse_y)))),
    )
    n_fft = max(n_fft, 128)
    hop_length = max(8, n_fft // 16)

    stft = librosa.stft(
        pulse_y,
        n_fft=n_fft,
        hop_length=hop_length,
        window="hann",
        center=True,
    )
    power = np.abs(stft) ** 2
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    frequency_mask = (
        (frequencies >= LOW_FREQ_HZ)
        & (frequencies <= min(HIGH_FREQ_HZ, sr / 2))
    )
    frequencies = frequencies[frequency_mask]
    power = power[frequency_mask, :]

    if power.size == 0 or np.max(power) <= 0:
        raise ValueError("脉冲中没有有效超声频谱")

    frame_peak_power = np.max(power, axis=0)
    frame_peak_db = librosa.power_to_db(
        frame_peak_power + 1e-18,
        ref=np.max,
    )
    reliable_mask = frame_peak_db >= CONTOUR_MIN_DB

    if np.sum(reliable_mask) < 2:
        strongest_count = min(3, power.shape[1])
        strongest_indices = np.argsort(frame_peak_power)[-strongest_count:]
        reliable_mask = np.zeros(power.shape[1], dtype=bool)
        reliable_mask[strongest_indices] = True

    reliable_power = power[:, reliable_mask]
    mean_spectrum = np.mean(reliable_power, axis=1)
    peak_freq_hz = float(
        frequencies[int(np.argmax(mean_spectrum))]
    )

    f05_hz = weighted_frequency_quantile(
        frequencies,
        mean_spectrum,
        0.05,
    )
    f95_hz = weighted_frequency_quantile(
        frequencies,
        mean_spectrum,
        0.95,
    )

    dominant_indices = np.argmax(power, axis=0)
    dominant_freqs_hz = frequencies[dominant_indices]

    if len(dominant_freqs_hz) >= 5:
        dominant_freqs_hz = medfilt(
            dominant_freqs_hz,
            kernel_size=5,
        )

    all_frame_times_s = librosa.frames_to_time(
        np.arange(power.shape[1]),
        sr=sr,
        hop_length=hop_length,
    )
    reliable_times_s = all_frame_times_s[reliable_mask]
    reliable_freqs_hz = dominant_freqs_hz[reliable_mask]

    if len(reliable_times_s) >= 2:
        slope_hz_per_s = float(
            np.polyfit(
                reliable_times_s,
                reliable_freqs_hz,
                1,
            )[0]
        )
        edge_count = min(3, len(reliable_freqs_hz))
        start_freq_hz = float(
            np.median(reliable_freqs_hz[:edge_count])
        )
        end_freq_hz = float(
            np.median(reliable_freqs_hz[-edge_count:])
        )
        frequency_jumps_khz = (
            np.abs(np.diff(reliable_freqs_hz)) / 1000
        )
        max_track_jump_khz = (
            float(np.max(frequency_jumps_khz))
            if len(frequency_jumps_khz)
            else 0.0
        )
    else:
        slope_hz_per_s = 0.0
        start_freq_hz = peak_freq_hz
        end_freq_hz = peak_freq_hz
        max_track_jump_khz = 0.0

    return {
        "peak_freq_khz": peak_freq_hz / 1000,
        "start_freq_khz": start_freq_hz / 1000,
        "end_freq_khz": end_freq_hz / 1000,
        "frequency_drop_khz": (
            start_freq_hz - end_freq_hz
        )
        / 1000,
        "f05_khz": f05_hz / 1000,
        "f95_khz": f95_hz / 1000,
        "bandwidth_90_khz": (f95_hz - f05_hz) / 1000,
        "slope_khz_per_ms": slope_hz_per_s / 1_000_000,
        "track_coverage": float(np.mean(reliable_mask)),
        "max_track_jump_khz": max_track_jump_khz,
        "rms": float(np.sqrt(np.mean(pulse_y**2))),
    }


def evaluate_pulse_quality(
    pulse: dict[str, float],
    features: dict[str, float],
) -> tuple[bool, str]:
    """进行不依赖具体物种的通用质量筛选。"""
    reasons: list[str] = []

    if pulse["snr_db"] < MIN_PULSE_SNR_DB:
        reasons.append("low_snr")

    if features["track_coverage"] < MIN_TRACK_COVERAGE:
        reasons.append("low_track_coverage")

    if features["max_track_jump_khz"] > MAX_TRACK_JUMP_KHZ:
        reasons.append("unstable_frequency_track")

    if not (
        LOW_FREQ_HZ / 1000
        <= features["peak_freq_khz"]
        <= min(HIGH_FREQ_HZ, TARGET_SAMPLE_RATE / 2) / 1000
    ):
        reasons.append("peak_frequency_out_of_range")

    return len(reasons) == 0, ";".join(reasons)


def save_top_pulse_zoom_grid(
    audio_path: Path,
    y: np.ndarray,
    sr: int,
    pulse_rows: list[dict[str, float | int | str]],
) -> None:
    """
    把当前WAV中能量最强的若干候选脉冲放大，并叠加主导频率轮廓线。

    每个小图：
    - 横轴为相对时间（毫秒）
    - 纵轴为频率（kHz）
    - 白线为每个时间帧中能量最强的频率轨迹
    """
    if not pulse_rows:
        return

    strongest_rows = sorted(
        pulse_rows,
        key=lambda row: float(row.get("rms", 0.0)),
        reverse=True,
    )[:TOP_PULSES_PER_FILE]

    columns = 2
    rows_count = int(np.ceil(len(strongest_rows) / columns))

    figure, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(14, max(4, rows_count * 3.4)),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    context_samples = int(round(ZOOM_CONTEXT_MS / 1000 * sr))

    for plot_index, row in enumerate(strongest_rows):
        axis = axes_flat[plot_index]

        start_sample = max(
            0,
            int(round(float(row["start_s"]) * sr)) - context_samples,
        )
        end_sample = min(
            len(y),
            int(round(float(row["end_s"]) * sr)) + context_samples,
        )
        segment = y[start_sample:end_sample]

        if len(segment) < 64:
            axis.set_visible(False)
            continue

        # 放大图优先保证时间分辨率
        n_fft = min(
            512,
            2 ** int(np.floor(np.log2(len(segment)))),
        )
        n_fft = max(128, n_fft)
        hop_length = max(8, n_fft // 16)

        stft = librosa.stft(
            segment,
            n_fft=n_fft,
            hop_length=hop_length,
            window="hann",
            center=True,
        )
        pulse_power = np.abs(stft) ** 2
        pulse_db = librosa.power_to_db(
            pulse_power + 1e-18,
            ref=np.max,
        )

        pulse_frequencies = librosa.fft_frequencies(
            sr=sr,
            n_fft=n_fft,
        )
        pulse_times_ms = (
            librosa.frames_to_time(
                np.arange(pulse_power.shape[1]),
                sr=sr,
                hop_length=hop_length,
            )
            * 1000
        )

        frequency_mask = (
            (pulse_frequencies >= LOW_FREQ_HZ)
            & (
                pulse_frequencies
                <= min(HIGH_FREQ_HZ, sr / 2)
            )
        )

        selected_frequencies = pulse_frequencies[frequency_mask]
        selected_power = pulse_power[frequency_mask, :]
        selected_db = pulse_db[frequency_mask, :]

        axis.pcolormesh(
            pulse_times_ms,
            selected_frequencies / 1000,
            selected_db,
            shading="auto",
            vmin=-60,
            vmax=0,
        )

        # 每个时间帧找能量最高的频率，形成轮廓线
        dominant_indices = np.argmax(selected_power, axis=0)
        contour_khz = (
            selected_frequencies[dominant_indices] / 1000
        )

        if len(contour_khz) >= 5:
            contour_khz = medfilt(contour_khz, kernel_size=5)

        frame_peak_db = np.max(selected_db, axis=0)
        reliable_mask = frame_peak_db >= CONTOUR_MIN_DB

        axis.plot(
            pulse_times_ms[reliable_mask],
            contour_khz[reliable_mask],
            color="white",
            linewidth=1.4,
        )

        pulse_id = int(row["pulse_id"])
        duration_ms = float(row["duration_ms"])
        peak_freq_khz = float(row["peak_freq_khz"])
        start_freq_khz = float(row["start_freq_khz"])
        end_freq_khz = float(row["end_freq_khz"])

        axis.set_title(
            f"Pulse {pulse_id} | "
            f"duration={duration_ms:.2f} ms | "
            f"peak={peak_freq_khz:.1f} kHz\n"
            f"start={start_freq_khz:.1f}, "
            f"end={end_freq_khz:.1f} kHz"
        )
        axis.set_ylim(
            LOW_FREQ_HZ / 1000,
            min(HIGH_FREQ_HZ, sr / 2) / 1000,
        )
        axis.set_xlabel("Relative time (ms)")
        axis.set_ylabel("Frequency (kHz)")

    for unused_axis in axes_flat[len(strongest_rows):]:
        unused_axis.set_visible(False)

    figure.suptitle(
        f"{audio_path.name} | "
        f"top {len(strongest_rows)} strongest candidate pulses",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))

    output_path = (
        PULSE_ZOOM_DIR
        / f"{audio_path.stem}_top_pulses.png"
    )
    figure.savefig(
        output_path,
        dpi=180,
    )
    plt.close(figure)


def accepted_mask(series: pd.Series) -> pd.Series:
    """兼容布尔值和从CSV读取后的字符串布尔值。"""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def _track_frequency_at_frames(
    track: dict[str, object],
    frames: np.ndarray,
) -> np.ndarray:
    """把稀疏轨迹线性插值到指定帧。"""
    track_frames = np.asarray(
        track["frames"],
        dtype=float,
    )
    track_frequencies = np.asarray(
        track["frequencies_hz"],
        dtype=float,
    )
    return np.interp(
        frames.astype(float),
        track_frames,
        track_frequencies,
    )


def _tracks_are_harmonic_duplicates(
    first: dict[str, object],
    second: dict[str, object],
) -> bool:
    """避免把同一叫声的二、三次谐波误当作两只蝙蝠。"""
    first_median = float(
        np.median(first["frequencies_hz"])
    )
    second_median = float(
        np.median(second["frequencies_hz"])
    )
    lower = min(first_median, second_median)
    upper = max(first_median, second_median)
    if lower <= 0:
        return False

    ratio = upper / lower
    return any(
        abs(ratio - harmonic)
        <= SEPARATION_HARMONIC_TOLERANCE * harmonic
        for harmonic in (2.0, 3.0)
    )


def _compare_simultaneous_tracks(
    primary: dict[str, object],
    candidate: dict[str, object],
) -> tuple[bool, float, float]:
    """
    判断两条轨迹是否真正在时间上重叠且频率可分辨。

    返回：(是否独立、重叠比例、中位频差kHz)。
    """
    primary_frames = np.asarray(
        primary["frames"],
        dtype=int,
    )
    candidate_frames = np.asarray(
        candidate["frames"],
        dtype=int,
    )
    common_frames = np.intersect1d(
        primary_frames,
        candidate_frames,
    )
    shorter_length = min(
        len(primary_frames),
        len(candidate_frames),
    )
    overlap_ratio = (
        len(common_frames) / shorter_length
        if shorter_length
        else 0.0
    )
    if (
        len(common_frames) < 2
        or overlap_ratio
        < SEPARATION_MIN_TEMPORAL_OVERLAP
    ):
        return False, overlap_ratio, 0.0

    primary_frequencies = _track_frequency_at_frames(
        primary,
        common_frames,
    )
    candidate_frequencies = _track_frequency_at_frames(
        candidate,
        common_frames,
    )
    gaps_khz = (
        np.abs(
            primary_frequencies
            - candidate_frequencies
        )
        / 1000
    )
    median_gap_khz = float(np.median(gaps_khz))
    resolvable_ratio = float(
        np.mean(
            gaps_khz
            >= SEPARATION_MIN_FREQ_GAP_KHZ
        )
    )
    independent = (
        median_gap_khz
        >= SEPARATION_MIN_FREQ_GAP_KHZ
        and resolvable_ratio >= 0.60
        and not _tracks_are_harmonic_duplicates(
            primary,
            candidate,
        )
    )
    return independent, overlap_ratio, median_gap_khz


def _link_spectral_peak_tracks(
    band_power: np.ndarray,
    band_frequencies: np.ndarray,
    sr: int,
) -> list[dict[str, object]]:
    """逐帧寻找局部峰，并用带速度预测的贪心匹配连成轨迹。"""
    if band_power.size == 0:
        return []

    global_peak = float(np.max(band_power))
    if global_peak <= 0:
        return []

    power_db = librosa.power_to_db(
        band_power + 1e-18,
        ref=global_peak,
    )
    frequency_step_hz = float(
        np.median(np.diff(band_frequencies))
    )
    minimum_peak_distance = max(
        1,
        int(
            round(
                SEPARATION_MIN_FREQ_GAP_KHZ
                * 1000
                / max(frequency_step_hz, 1.0)
            )
        ),
    )

    tracks: list[dict[str, object]] = []

    for frame_index in range(band_power.shape[1]):
        frame_db = power_db[:, frame_index]
        frame_max_db = float(np.max(frame_db))
        if frame_max_db < SEPARATION_GLOBAL_MIN_DB:
            continue

        minimum_height = max(
            SEPARATION_GLOBAL_MIN_DB,
            frame_max_db
            - SEPARATION_FRAME_DYNAMIC_RANGE_DB,
        )
        peak_indices, properties = find_peaks(
            frame_db,
            height=minimum_height,
            prominence=SEPARATION_PEAK_PROMINENCE_DB,
            distance=minimum_peak_distance,
        )
        if len(peak_indices) == 0:
            # 极窄带信号偶尔位于频带边缘；至少保留帧内最强峰。
            peak_indices = np.array(
                [int(np.argmax(frame_db))],
                dtype=int,
            )

        peak_order = np.argsort(
            band_power[peak_indices, frame_index]
        )[::-1][
            :SEPARATION_MAX_PEAKS_PER_FRAME
        ]
        peak_indices = peak_indices[peak_order]
        peak_frequencies = band_frequencies[
            peak_indices
        ]

        active_track_indices = [
            index
            for index, track in enumerate(tracks)
            if frame_index
            - int(track["frames"][-1])
            <= SEPARATION_MAX_TRACK_GAP_FRAMES + 1
        ]

        possible_matches: list[
            tuple[float, int, int]
        ] = []
        for track_index in active_track_indices:
            track = tracks[track_index]
            track_frames = track["frames"]
            track_frequencies = track[
                "frequencies_hz"
            ]
            last_frame = int(track_frames[-1])
            frame_delta = frame_index - last_frame
            predicted_frequency = float(
                track_frequencies[-1]
            )

            if len(track_frames) >= 2:
                previous_delta = max(
                    1,
                    last_frame
                    - int(track_frames[-2]),
                )
                velocity = (
                    float(track_frequencies[-1])
                    - float(track_frequencies[-2])
                ) / previous_delta
                predicted_frequency += (
                    velocity * frame_delta
                )

            maximum_jump_hz = (
                SEPARATION_MAX_TRACK_JUMP_KHZ
                * 1000
                * max(frame_delta, 1)
            )
            for peak_order_index, frequency in enumerate(
                peak_frequencies
            ):
                cost = abs(
                    float(frequency)
                    - predicted_frequency
                )
                if cost <= maximum_jump_hz:
                    possible_matches.append(
                        (
                            cost,
                            track_index,
                            peak_order_index,
                        )
                    )

        assigned_tracks: set[int] = set()
        assigned_peaks: set[int] = set()
        for _, track_index, peak_order_index in sorted(
            possible_matches,
            key=lambda item: item[0],
        ):
            if (
                track_index in assigned_tracks
                or peak_order_index in assigned_peaks
            ):
                continue

            peak_index = int(
                peak_indices[peak_order_index]
            )
            track = tracks[track_index]
            track["frames"].append(frame_index)
            track["frequency_bins"].append(
                peak_index
            )
            track["frequencies_hz"].append(
                float(band_frequencies[peak_index])
            )
            track["powers"].append(
                float(
                    band_power[
                        peak_index,
                        frame_index,
                    ]
                )
            )
            track["peak_db"].append(
                float(
                    power_db[
                        peak_index,
                        frame_index,
                    ]
                )
            )
            assigned_tracks.add(track_index)
            assigned_peaks.add(peak_order_index)

        for peak_order_index, peak_index_value in enumerate(
            peak_indices
        ):
            if peak_order_index in assigned_peaks:
                continue

            peak_index = int(peak_index_value)
            tracks.append(
                {
                    "frames": [frame_index],
                    "frequency_bins": [peak_index],
                    "frequencies_hz": [
                        float(
                            band_frequencies[
                                peak_index
                            ]
                        )
                    ],
                    "powers": [
                        float(
                            band_power[
                                peak_index,
                                frame_index,
                            ]
                        )
                    ],
                    "peak_db": [
                        float(
                            power_db[
                                peak_index,
                                frame_index,
                            ]
                        )
                    ],
                }
            )

    hop_ms = SEPARATION_HOP_LENGTH / sr * 1000
    minimum_frames = max(
        3,
        int(
            np.ceil(
                SEPARATION_MIN_TRACK_MS
                / max(hop_ms, 1e-9)
            )
        ),
    )
    valid_tracks: list[dict[str, object]] = []

    for track in tracks:
        frames = np.asarray(
            track["frames"],
            dtype=int,
        )
        if len(frames) < minimum_frames:
            continue

        span_frames = int(frames[-1] - frames[0] + 1)
        coverage = len(frames) / max(span_frames, 1)
        if coverage < SEPARATION_MIN_TRACK_COVERAGE:
            continue

        track["coverage"] = float(coverage)
        track["total_power"] = float(
            np.sum(track["powers"])
        )
        track["median_frequency_hz"] = float(
            np.median(track["frequencies_hz"])
        )
        valid_tracks.append(track)

    return sorted(
        valid_tracks,
        key=lambda track: float(track["total_power"]),
        reverse=True,
    )


def separate_simultaneous_pulse_components(
    pulse_y: np.ndarray,
    sr: int,
) -> list[dict[str, object]]:
    """
    将一个时间候选展开为一条或多条同时发声的近频脉冲。

    只有检测到至少两条可靠、重叠且非谐波的轨迹时才修改
    波形；否则直接返回原波形，避免对普通单脉冲造成失真。
    """
    unchanged = {
        "audio": pulse_y,
        "start_sample": 0,
        "end_sample": len(pulse_y),
        "component_id": 1,
        "component_count": 1,
        "separation_applied": False,
        "track_median_freq_khz": float("nan"),
        "track_relative_db": 0.0,
        "separation_confidence": 1.0,
    }
    if (
        not ENABLE_SIMULTANEOUS_SEPARATION
        or len(pulse_y) < 64
    ):
        return [unchanged]

    win_length = min(
        SEPARATION_WIN_LENGTH,
        len(pulse_y),
    )
    if win_length < 64:
        return [unchanged]

    # 对不足4.1 ms的短叫声仍使用零填充细采样频率轴；
    # 这是有意设计，屏蔽librosa针对该情况的提示即可。
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="n_fft=.*is too large for input signal.*",
            category=UserWarning,
        )
        stft = librosa.stft(
            pulse_y,
            n_fft=SEPARATION_N_FFT,
            hop_length=SEPARATION_HOP_LENGTH,
            win_length=win_length,
            window="hann",
            center=True,
        )
    power = np.abs(stft) ** 2
    frequencies = librosa.fft_frequencies(
        sr=sr,
        n_fft=SEPARATION_N_FFT,
    )
    frequency_mask = (
        (frequencies >= LOW_FREQ_HZ)
        & (
            frequencies
            <= min(HIGH_FREQ_HZ, sr / 2)
        )
    )
    band_frequencies = frequencies[
        frequency_mask
    ]
    band_power = power[frequency_mask, :]

    tracks = _link_spectral_peak_tracks(
        band_power,
        band_frequencies,
        sr,
    )
    if len(tracks) < 2:
        return [unchanged]

    primary = tracks[0]
    primary_power = max(
        float(primary["total_power"]),
        1e-18,
    )
    selected_tracks = [primary]
    track_metrics: dict[
        int,
        tuple[float, float, float],
    ] = {
        id(primary): (1.0, 0.0, 0.0),
    }

    for candidate in tracks[1:]:
        relative_db = float(
            10
            * np.log10(
                max(
                    float(candidate["total_power"]),
                    1e-18,
                )
                / primary_power
            )
        )
        if (
            relative_db
            < SEPARATION_MIN_SECONDARY_RELATIVE_DB
        ):
            continue

        independent, overlap_ratio, median_gap_khz = (
            _compare_simultaneous_tracks(
                primary,
                candidate,
            )
        )
        if not independent:
            continue

        distinct_from_selected = all(
            _compare_simultaneous_tracks(
                selected,
                candidate,
            )[0]
            for selected in selected_tracks[1:]
        )
        if not distinct_from_selected:
            continue

        selected_tracks.append(candidate)
        track_metrics[id(candidate)] = (
            overlap_ratio,
            median_gap_khz,
            relative_db,
        )
        if (
            len(selected_tracks)
            >= SEPARATION_MAX_COMPONENTS
        ):
            break

    if len(selected_tracks) < 2:
        return [unchanged]

    frame_numbers = np.arange(
        stft.shape[1],
        dtype=int,
    )
    component_weights = np.zeros(
        (
            len(selected_tracks),
            len(frequencies),
            stft.shape[1],
        ),
        dtype=np.float32,
    )
    sigma_hz = max(
        SEPARATION_MASK_BANDWIDTH_KHZ * 1000,
        float(
            np.median(np.diff(frequencies))
        ),
    )

    for component_index, track in enumerate(
        selected_tracks
    ):
        track_frames = np.asarray(
            track["frames"],
            dtype=int,
        )
        active_frames = (
            (frame_numbers >= int(track_frames[0]))
            & (frame_numbers <= int(track_frames[-1]))
        )
        interpolated_frequency = (
            _track_frequency_at_frames(
                track,
                frame_numbers,
            )
        )
        distance = (
            frequencies[:, None]
            - interpolated_frequency[None, :]
        )
        gaussian = np.exp(
            -0.5 * (distance / sigma_hz) ** 2
        ).astype(np.float32)

        track_power = np.interp(
            frame_numbers.astype(float),
            track_frames.astype(float),
            np.asarray(track["powers"], dtype=float),
        )
        track_power /= max(
            float(np.max(track_power)),
            1e-18,
        )
        gaussian *= np.sqrt(
            np.maximum(track_power, 1e-6)
        )[None, :].astype(np.float32)
        gaussian[:, ~active_frames] = 0.0
        gaussian[~frequency_mask, :] = 0.0
        component_weights[component_index] = gaussian

    weight_sum = np.sum(
        component_weights,
        axis=0,
    )
    valid_weight = weight_sum > 1e-12
    components: list[dict[str, object]] = []
    padding_samples = int(
        round(CORE_PADDING_MS / 1000 * sr)
    )
    minimum_samples = max(
        32,
        int(round(MIN_PULSE_MS / 1000 * sr)),
    )

    for component_index, track in enumerate(
        selected_tracks
    ):
        soft_mask = np.zeros_like(
            weight_sum,
            dtype=np.float32,
        )
        np.divide(
            component_weights[component_index],
            weight_sum,
            out=soft_mask,
            where=valid_weight,
        )
        component_stft = stft * soft_mask
        component_audio = librosa.istft(
            component_stft,
            hop_length=SEPARATION_HOP_LENGTH,
            win_length=win_length,
            window="hann",
            center=True,
            length=len(pulse_y),
        ).astype(np.float32)

        track_frames = np.asarray(
            track["frames"],
            dtype=int,
        )
        start_sample = max(
            0,
            int(track_frames[0])
            * SEPARATION_HOP_LENGTH
            - win_length // 2
            - padding_samples,
        )
        end_sample = min(
            len(pulse_y),
            int(track_frames[-1])
            * SEPARATION_HOP_LENGTH
            + win_length // 2
            + padding_samples,
        )
        if end_sample - start_sample < minimum_samples:
            missing = (
                minimum_samples
                - (end_sample - start_sample)
            )
            start_sample = max(
                0,
                start_sample - missing // 2,
            )
            end_sample = min(
                len(pulse_y),
                start_sample + minimum_samples,
            )
            start_sample = max(
                0,
                end_sample - minimum_samples,
            )

        overlap_ratio, median_gap_khz, relative_db = (
            track_metrics[id(track)]
        )
        if component_index == 0:
            comparisons = [
                _compare_simultaneous_tracks(
                    track,
                    other,
                )
                for other in selected_tracks[1:]
            ]
            overlap_ratio = float(
                max(value[1] for value in comparisons)
            )
            median_gap_khz = float(
                min(value[2] for value in comparisons)
            )

        energy_confidence = float(
            np.clip(
                (
                    relative_db
                    - SEPARATION_MIN_SECONDARY_RELATIVE_DB
                )
                / max(
                    -SEPARATION_MIN_SECONDARY_RELATIVE_DB,
                    1e-9,
                ),
                0.0,
                1.0,
            )
        )
        if component_index == 0:
            energy_confidence = 1.0
        gap_confidence = float(
            np.clip(
                median_gap_khz
                / max(
                    2 * SEPARATION_MIN_FREQ_GAP_KHZ,
                    1e-9,
                ),
                0.0,
                1.0,
            )
        )
        separation_confidence = float(
            np.clip(
                0.45 * overlap_ratio
                + 0.35 * gap_confidence
                + 0.20 * energy_confidence,
                0.0,
                1.0,
            )
        )

        components.append(
            {
                "audio": component_audio[
                    start_sample:end_sample
                ],
                "start_sample": start_sample,
                "end_sample": end_sample,
                "component_id": component_index + 1,
                "component_count": len(
                    selected_tracks
                ),
                "separation_applied": True,
                "track_median_freq_khz": (
                    float(
                        track[
                            "median_frequency_hz"
                        ]
                    )
                    / 1000
                ),
                "track_relative_db": float(relative_db),
                "separation_confidence": (
                    separation_confidence
                ),
            }
        )

    return components


def save_separated_pulse(
    audio_path: Path,
    candidate_id: int,
    component_id: int,
    component_audio: np.ndarray,
    sr: int,
) -> str:
    """只把确实发生拆分的分量写盘，并返回相对路径。"""
    output_dir = (
        SEPARATED_PULSE_DIR / audio_path.stem
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path = output_dir / (
        f"{audio_path.stem}_candidate_"
        f"{candidate_id:04d}_component_"
        f"{component_id:02d}.wav"
    )
    sf.write(
        output_path,
        component_audio.astype(np.float32),
        sr,
        subtype="PCM_16",
    )
    return str(output_path.relative_to(BASE_DIR))


def clear_separated_pulses_for_source(
    audio_path: Path,
) -> None:
    """重建训练表前清理该录音旧分量，避免参数变化后残留文件。"""
    output_dir = (
        SEPARATED_PULSE_DIR / audio_path.stem
    )
    if not output_dir.exists():
        return

    resolved_root = SEPARATED_PULSE_DIR.resolve()
    resolved_output = output_dir.resolve()
    if (
        resolved_output == resolved_root
        or resolved_root
        not in resolved_output.parents
    ):
        raise RuntimeError(
            f"拒绝清理异常分离目录：{resolved_output}"
        )
    shutil.rmtree(resolved_output)


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
                {
                    "audio": pulse_y,
                    "start_sample": 0,
                    "end_sample": len(pulse_y),
                    "component_id": 1,
                    "component_count": 1,
                    "separation_applied": False,
                    "track_median_freq_khz": (
                        float("nan")
                    ),
                    "track_relative_db": 0.0,
                    "separation_confidence": 0.0,
                }
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
                    {
                        "audio": pulse_y,
                        "start_sample": 0,
                        "end_sample": len(pulse_y),
                        "component_id": 1,
                        "component_count": 1,
                        "separation_applied": False,
                        "track_median_freq_khz": (
                            float("nan")
                        ),
                        "track_relative_db": 0.0,
                        "separation_confidence": 1.0,
                    }
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


def build_spectrogram(
    y: np.ndarray,
    sr: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Step6覆盖版声谱图：把检测步长从128提高到256，
    在保留约1毫秒时间分辨率的同时减少一半计算量。
    """
    n_fft = 1024
    hop_length = DEMO_DETECTION_HOP_LENGTH

    stft = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        window="hann",
        center=True,
    )
    power = np.abs(stft) ** 2
    frequencies = librosa.fft_frequencies(
        sr=sr,
        n_fft=n_fft,
    )
    times = librosa.frames_to_time(
        np.arange(power.shape[1]),
        sr=sr,
        hop_length=hop_length,
    )
    return power, frequencies, times, hop_length


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
        overlap_s = max(
            0.0,
            min(
                float(previous["end_s"]),
                float(pulse["end_s"]),
            )
            - max(
                float(previous["start_s"]),
                float(pulse["start_s"]),
            ),
        )
        shorter_duration_s = min(
            float(previous["end_s"])
            - float(previous["start_s"]),
            float(pulse["end_s"])
            - float(pulse["start_s"]),
        )
        overlap_ratio = (
            overlap_s / shorter_duration_s
            if shorter_duration_s > 0
            else 0.0
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


def save_fast_overview_spectrogram(
    audio_path: Path,
    y: np.ndarray,
    sr: int,
    pulse_count: int,
) -> None:
    """
    保存压缩版整段声谱图。
    时间帧最多约4000个，因此几十秒录音也不会占用巨大内存。
    """
    n_fft = 1024
    dynamic_hop = int(
        np.ceil(
            len(y) / OVERVIEW_MAX_TIME_BINS
        )
    )
    hop_length = max(
        512,
        dynamic_hop,
    )

    stft = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        window="hann",
        center=True,
    )
    power = np.abs(stft) ** 2
    frequencies = librosa.fft_frequencies(
        sr=sr,
        n_fft=n_fft,
    )
    times = librosa.frames_to_time(
        np.arange(power.shape[1]),
        sr=sr,
        hop_length=hop_length,
    )

    mask = (
        (frequencies >= LOW_FREQ_HZ)
        & (
            frequencies
            <= min(HIGH_FREQ_HZ, sr / 2)
        )
    )
    selected_power = power[mask, :]
    selected_frequencies = frequencies[mask]

    if selected_power.size == 0:
        return

    power_db = librosa.power_to_db(
        selected_power + 1e-18,
        ref=np.max,
    )

    figure, axis = plt.subplots(
        figsize=(14, 5)
    )
    mesh = axis.pcolormesh(
        times,
        selected_frequencies / 1000,
        power_db,
        shading="auto",
        vmin=-80,
        vmax=0,
    )
    axis.set_title(
        f"{audio_path.name} | "
        f"accepted pulses: {pulse_count}"
    )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Frequency (kHz)")
    figure.colorbar(
        mesh,
        ax=axis,
        label="Relative power (dB)",
    )
    figure.tight_layout()
    output_path = (
        SPECTROGRAM_DIR
        / f"{audio_path.stem}_overview.png"
    )
    figure.savefig(
        output_path,
        dpi=150,
    )
    plt.close(figure)


def find_wav_files(root: Path) -> list[Path]:
    """同时识别 .wav 和 .WAV。"""
    if not root.exists():
        return []

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".wav"
    )


def prepare_demo_dataset() -> None:
    """准备用户自己的演示录音数据。"""
    for directory in (
        RAW_DATA_DIR,
        TABLE_DIR,
        FIGURE_DIR,
        AUDIO_OUTPUT_DIR,
        MODEL_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    DEMO_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    DEMO_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    SPECTROGRAM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    PULSE_ZOOM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    CUT_PULSE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    REJECTED_PULSE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    DEMO_PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    CLUSTER_EXAMPLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    SEPARATED_PULSE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_wavs = find_wav_files(
        DEMO_DATA_DIR
    )
    if existing_wavs:
        print(
            f"检测到已有演示数据："
            f"{len(existing_wavs)} 个WAV"
        )
        return

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
            f"或者把WAV手动放入 {DEMO_DATA_DIR}。"
        )

    print(
        f"正在解压演示数据："
        f"{zip_path.name} -> {DEMO_DATA_DIR.name}"
    )
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DEMO_DATA_DIR)

    wav_files = find_wav_files(DEMO_DATA_DIR)
    if not wav_files:
        raise RuntimeError(
            "压缩包解压成功，但没有找到WAV文件。"
        )

    print(
        f"演示数据准备完成："
        f"{len(wav_files)} 个WAV"
    )


def build_demo_pulse_table() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    对全部录音自动检测、精修并切割脉冲。

    返回：
    - pulse_df：每个候选脉冲一行；
    - recording_df：每个原始录音一行的检测汇总。
    """
    wav_files = find_wav_files(DEMO_DATA_DIR)
    if not wav_files:
        raise FileNotFoundError(
            f"{DEMO_DATA_DIR} 中没有WAV文件。"
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
                DEMO_DATA_DIR
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

            if SAVE_PER_RECORDING_DIAGNOSTICS:
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

    for column in CLUSTER_FEATURE_COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna(
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


def save_pca_plot(
    data: pd.DataFrame,
) -> None:
    """保存每个点代表一个脉冲的二维PCA聚类图。"""
    figure, axis = plt.subplots(
        figsize=(10, 7)
    )

    for acoustic_type in ["A", "B", "C", "D"]:
        group = data[
            data["acoustic_type"]
            == acoustic_type
        ]
        if group.empty:
            continue

        axis.scatter(
            group["pca_1"],
            group["pca_2"],
            s=20,
            alpha=0.65,
            label=(
                f"声型{acoustic_type} "
                f"({len(group)})"
            ),
        )

    axis.set_title(
        "Bat acoustic pulse clusters A-D (PCA)"
    )
    axis.set_xlabel("PCA component 1")
    axis.set_ylabel("PCA component 2")
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        CLUSTER_PCA_PNG,
        dpi=180,
    )
    plt.close(figure)


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

    for column in NOISE_GATE_FEATURE_COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = (
        data.replace([np.inf, -np.inf], np.nan)
        .dropna(
            subset=[
                "source_file",
                "stage1_label",
                *NOISE_GATE_FEATURE_COLUMNS,
            ]
        )
        .copy()
    )

    # 限制同一个录音、同一类别的贡献，避免长录音压倒其他数据。
    kept_parts: list[pd.DataFrame] = []

    for (_, label), group in data.groupby(
        ["source_file", "stage1_label"],
        sort=False,
    ):
        if label == "bat":
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
        else:
            # noise保留较难的误检，而不是只保留完全安静的片段。
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

    for column in feature_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    valid_mask = (
        data[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .all(axis=1)
    )
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
        for column in acoustic_features:
            bat_df[column] = pd.to_numeric(
                bat_df[column],
                errors="coerce",
            )

        valid_bat_mask = (
            bat_df[acoustic_features]
            .replace([np.inf, -np.inf], np.nan)
            .notna()
            .all(axis=1)
        )
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
