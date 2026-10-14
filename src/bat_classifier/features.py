"""单脉冲频谱与轨迹特征提取，供模板匹配和声波建表使用。
"""

from __future__ import annotations

import librosa
import numpy as np
from scipy.signal import medfilt

from .audio_io import adaptive_stft_params, ultrasonic_band_mask
from .config import CONTOUR_MIN_DB


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
