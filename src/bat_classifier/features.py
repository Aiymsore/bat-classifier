"""单脉冲特征提取与质量筛选（上游：detection 切出的脉冲波形；下游：聚类/门控/建表）。

职责：频谱特征（峰值频率、带宽、斜率等）计算、通用质量门槛判断、
accepted 布尔列兼容解析，以及训练表通用的数值列清洗工具。
"""

from __future__ import annotations

import librosa
import numpy as np
import pandas as pd
from scipy.signal import medfilt

from .audio_io import adaptive_stft_params, ultrasonic_band_mask
from .config import (
    CONTOUR_MIN_DB,
    HIGH_FREQ_HZ,
    LOW_FREQ_HZ,
    MAX_TRACK_JUMP_KHZ,
    MIN_PULSE_SNR_DB,
    MIN_TRACK_COVERAGE,
    TARGET_SAMPLE_RATE,
)


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

    n_fft, hop_length = adaptive_stft_params(
        len(pulse_y),
        max_n_fft=512,
        min_n_fft=128,
        min_hop=8,
    )

    stft = librosa.stft(
        pulse_y,
        n_fft=n_fft,
        hop_length=hop_length,
        window="hann",
        center=True,
    )
    power = np.abs(stft) ** 2
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    frequency_mask = ultrasonic_band_mask(frequencies, sr)
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


def numeric_feature_frame(
    data: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """把特征列统一转为数值并把 ±inf 替换为 NaN，便于后续过滤。"""
    frame = data[columns].copy()
    for column in columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )
    return frame.replace([np.inf, -np.inf], np.nan)
