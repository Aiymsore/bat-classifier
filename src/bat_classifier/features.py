"""单脉冲频谱与轨迹特征提取，供模板匹配和声波建表使用。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import librosa
import numpy as np
from scipy.signal import medfilt

from .audio_io import adaptive_stft_params, ultrasonic_band_mask
from .config import (
    CONTOUR_MIN_DB,
    PLATEAU_ACTIVE_MIN_DB,
    PLATEAU_MAX_GAP_MS,
    PLATEAU_REFERENCE_MIN_DB,
    PLATEAU_TOLERANCE_KHZ,
    TAIL_ACTIVE_MIN_DB,
    TAIL_BAND_ABOVE_KHZ,
    TAIL_BAND_BELOW_KHZ,
    TAIL_BIN_DYNAMIC_RANGE_DB,
    TAIL_LOW_OFFSET_KHZ,
    TAIL_MAX_FRACTION,
    TAIL_MAX_MS,
    TAIL_MIN_MS,
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


@dataclass(frozen=True)
class PulseSpectrum:
    """同一段音频的一次频谱计算；只共享原始材料，不共享平滑或筛帧规则。"""

    sample_count: int
    sample_rate: int
    hop_length: int
    power: np.ndarray
    frequencies: np.ndarray
    frame_peak_power: np.ndarray
    dominant_freqs_hz: np.ndarray
    supported_frames: np.ndarray | None = None


def with_observed_support(spectrum: PulseSpectrum, track: dict | None) -> PulseSpectrum:
    """Map directly observed mixture-ridge frames onto the shared feature STFT."""
    if track is None or len(track.get('times_s', [])) < 2:
        return spectrum
    source=np.asarray(track['times_s'],dtype=float)
    times=np.arange(spectrum.power.shape[1])*spectrum.hop_length/spectrum.sample_rate
    index=np.searchsorted(source,times)
    left=np.clip(index-1,0,len(source)-1);right=np.clip(index,0,len(source)-1)
    distance=np.minimum(abs(times-source[left]),abs(times-source[right]))
    step=max(float(np.median(np.diff(source))),spectrum.hop_length/spectrum.sample_rate)
    mask=(distance<=step*1.05)&(times>=source[0])&(times<=source[-1])
    mask &= times <= (spectrum.sample_count-1)/spectrum.sample_rate
    mask.setflags(write=False)
    return replace(spectrum,supported_frames=mask)


def compute_pulse_spectrum(
    pulse_y: np.ndarray,
    sr: int,
) -> PulseSpectrum:
    """计算建表和模板共用的 Hann/STFT、超声频段功率和未平滑主频。"""
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
    dominant_freqs_hz = frequencies[np.argmax(power, axis=0)]
    for values in (power, frequencies, frame_peak_power, dominant_freqs_hz):
        values.setflags(write=False)
    return PulseSpectrum(
        sample_count=len(pulse_y), sample_rate=sr, hop_length=hop_length,
        power=power, frequencies=frequencies, frame_peak_power=frame_peak_power,
        dominant_freqs_hz=dominant_freqs_hz,
    )


def mask_concentrations(
    power: np.ndarray,
    near_mask: np.ndarray,
    local_mask: np.ndarray,
) -> tuple[float, float]:
    """返回 (全带集中度, 局部带集中度)：近场能量占全带 / 占局部带的比例。

    两列共用同一次频谱与同一个近场掩膜。全带值会随 15–120 kHz 内的噪声总量
    稀释，适合做“有无宽带能量”的下限；局部带值的平谱基线约为
    近场带宽/局部带宽（默认 3 kHz / 12 kHz ≈ 0.25），用于判断轨迹处是否
    存在真正的窄带峰，不随整段频带的噪声底变化。
    """
    if power.size == 0 or power.shape[1] == 0:
        return 0.0, 0.0
    near_energy = np.sum(np.where(near_mask, power, 0.0), axis=0)
    local_energy = np.sum(np.where(local_mask, power, 0.0), axis=0)
    total_energy = np.maximum(np.sum(power, axis=0), 1e-18)
    full = float(np.mean(near_energy / total_energy))
    local = float(np.mean(near_energy / np.maximum(local_energy, 1e-18)))
    return full, local


def extract_tail_features(spectrum: PulseSpectrum) -> dict[str, float]:
    """描述叫声末段在 CF 主频下方出现的能量（CF-FM 的 FM 尾巴）。

    实测 CF-FM 模板的尾巴常与 CF 主线同时存在、能量更弱，每帧最强频率
    仍停在 CF 上，只看峰值轨迹会把它完全丢掉。这里改看尾段能量分布：

    - ``tail_low_energy_ratio``：尾段局部带内，低于 CF 主频
      ``TAIL_LOW_OFFSET_KHZ`` 以上的能量占比；
    - ``tail_drop_khz``：主体段与尾段累计谱 10 分位频率之差（尾段低频边缘下移量）。

    主体段取活动帧 20%～70%，用主体段未平滑主频的中位数作为 CF 主频；尾段为
    活动帧末尾 ``TAIL_MIN_MS``～``TAIL_MAX_MS``（且不超过 ``TAIL_MAX_FRACTION``）。
    只统计局部带 [CF-``TAIL_BAND_BELOW_KHZ``, CF+``TAIL_BAND_ABOVE_KHZ``]，并丢弃
    低于该段最强点 ``TAIL_BIN_DYNAMIC_RANGE_DB`` 的格点，避免平坦底噪抬高占比。
    材料不足时两项均为 0。
    """
    empty = {"tail_low_energy_ratio": 0.0, "tail_drop_khz": 0.0}
    power, frequencies = spectrum.power, spectrum.frequencies
    if power.size == 0 or power.shape[1] < 4:
        return empty
    peak_db = 10.0 * np.log10(
        spectrum.frame_peak_power / max(float(np.max(spectrum.frame_peak_power)), 1e-30) + 1e-30
    )
    active = np.flatnonzero(peak_db >= TAIL_ACTIVE_MIN_DB)
    if active.size < 4:
        return empty
    first, last = int(active[0]), int(active[-1])
    span = last - first + 1
    frame_ms = spectrum.hop_length / spectrum.sample_rate * 1000.0
    tail_frames = int(round(min(TAIL_MAX_MS / frame_ms, TAIL_MAX_FRACTION * span)))
    tail_frames = max(tail_frames, int(round(TAIL_MIN_MS / frame_ms)), 2)
    body_start = first + int(0.2 * span)
    body_end = min(first + int(0.7 * span), last - tail_frames)
    if body_end <= body_start or tail_frames >= span:
        return empty

    cf_khz = float(np.median(spectrum.dominant_freqs_hz[body_start:body_end + 1])) / 1000.0
    freqs_khz = frequencies / 1000.0
    band = (freqs_khz >= cf_khz - TAIL_BAND_BELOW_KHZ) & (freqs_khz <= cf_khz + TAIL_BAND_ABOVE_KHZ)
    if np.count_nonzero(band) < 3:
        return empty
    band_freqs = freqs_khz[band]
    floor = 10 ** (-TAIL_BIN_DYNAMIC_RANGE_DB / 10.0)

    def _clean_sum(start: int, end: int) -> np.ndarray:
        block = power[band, start:end + 1]
        block = np.where(block >= float(np.max(block)) * floor, block, 0.0)
        return np.sum(block, axis=1)

    tail_spec = _clean_sum(last - tail_frames + 1, last)
    body_spec = _clean_sum(body_start, body_end)
    if float(np.sum(tail_spec)) <= 0 or float(np.sum(body_spec)) <= 0:
        return empty
    bin_khz = float(np.median(np.diff(band_freqs)))
    low_offset = max(TAIL_LOW_OFFSET_KHZ, 3.0 * bin_khz)
    ratio = float(np.sum(tail_spec[band_freqs < cf_khz - low_offset]) / np.sum(tail_spec))
    drop = (
        weighted_frequency_quantile(band_freqs, body_spec, 0.10)
        - weighted_frequency_quantile(band_freqs, tail_spec, 0.10)
    )
    return {"tail_low_energy_ratio": ratio, "tail_drop_khz": float(max(drop, 0.0))}


def extract_plateau_features(spectrum: PulseSpectrum) -> dict[str, float]:
    """CF 平台（横频段）特征，不受首尾 FM 影响，供 CF 族模板匹配和判族。

    - ``cf_plateau_khz``：平台段主频中位数；
    - ``cf_plateau_ms``：主频保持在参考频率 ±容差内的最长连续时长；
    - ``cf_plateau_spread_khz``：平台段主频 95 与 5 分位之差。

    参考频率取强帧（不低于最强 ``PLATEAU_REFERENCE_MIN_DB``）主频的中位数；
    容差为 max(``PLATEAU_TOLERANCE_KHZ``, 2 个频率格点)；连续段允许不超过
    ``PLATEAU_MAX_GAP_MS`` 的缺帧。没有平台时时长为 0、频率取强帧主频。
    """
    power, frequencies = spectrum.power, spectrum.frequencies
    freqs_khz = spectrum.dominant_freqs_hz / 1000.0
    peak_db = 10.0 * np.log10(
        spectrum.frame_peak_power / max(float(np.max(spectrum.frame_peak_power)), 1e-30) + 1e-30
    )
    strong = peak_db >= PLATEAU_REFERENCE_MIN_DB
    reference = float(np.median(freqs_khz[strong])) if np.any(strong) else float(freqs_khz[np.argmax(peak_db)])
    empty = {"cf_plateau_khz": reference, "cf_plateau_ms": 0.0, "cf_plateau_spread_khz": 0.0}
    if power.shape[1] < 3 or len(frequencies) < 2:
        return empty
    smoothed = medfilt(freqs_khz, kernel_size=3)
    bin_khz = float(np.median(np.diff(frequencies))) / 1000.0
    tolerance = max(PLATEAU_TOLERANCE_KHZ, 2.0 * bin_khz)
    activity = spectrum.supported_frames if spectrum.supported_frames is not None else peak_db >= PLATEAU_ACTIVE_MIN_DB
    on = (np.abs(smoothed - reference) <= tolerance) & activity
    frame_ms = spectrum.hop_length / spectrum.sample_rate * 1000.0
    max_gap = int(PLATEAU_MAX_GAP_MS / frame_ms)
    best: tuple[int, int] | None = None
    start = previous = None
    for index in np.flatnonzero(on):
        index = int(index)
        if start is not None and index - previous - 1 <= max_gap:
            previous = index
            continue
        if start is not None and (best is None or previous - start > best[1] - best[0]):
            best = (start, previous)
        start = previous = index
    if start is not None and (best is None or previous - start > best[1] - best[0]):
        best = (start, previous)
    if best is None:
        return empty
    run = smoothed[best[0]:best[1] + 1][on[best[0]:best[1] + 1]]
    return {
        "cf_plateau_khz": float(np.median(run)),
        "cf_plateau_ms": float((best[1] - best[0] + 1) * frame_ms),
        "cf_plateau_spread_khz": float(np.percentile(run, 95) - np.percentile(run, 5)),
    }


def extract_pulse_features(
    pulse_y: np.ndarray,
    sr: int,
    *,
    spectrum: PulseSpectrum | None = None,
) -> dict[str, float]:
    """按建表规则筛选可靠帧、平滑和提取特征，可复用同一音频的频谱。"""
    if spectrum is None:
        spectrum = compute_pulse_spectrum(pulse_y, sr)
    if spectrum.sample_count != len(pulse_y) or spectrum.sample_rate != sr:
        raise ValueError("复用频谱的音频长度或采样率不一致。")
    power, frequencies = spectrum.power, spectrum.frequencies
    hop_length = spectrum.hop_length
    frame_peak_power = spectrum.frame_peak_power
    frame_peak_db = librosa.power_to_db(
        frame_peak_power + 1e-18,
        ref=np.max,
    )
    reliable_mask = (spectrum.supported_frames.copy() if spectrum.supported_frames is not None
                     else frame_peak_db >= CONTOUR_MIN_DB)

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

    dominant_freqs_hz = spectrum.dominant_freqs_hz

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
    } | extract_tail_features(spectrum) | extract_plateau_features(spectrum)
