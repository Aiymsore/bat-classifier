"""音频读取与频谱基础工具（流程最底层，被所有模块依赖）。

职责：WAV 读取与采样率统一、高通滤波、音频内容哈希去重、
基础声谱图计算，以及被检测/特征/分离/可视化复用的三个小工具：
超声频段掩码、自适应 STFT 参数、区间重叠比例。
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

from .config import (
    DETECTION_HOP_LENGTH,
    HIGH_FREQ_HZ,
    LOW_FREQ_HZ,
    TARGET_SAMPLE_RATE,
)


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


def load_audio_interval(
    audio_path: Path,
    start_s: float,
    end_s: float,
    *,
    preprocessed: bool = False,
) -> tuple[np.ndarray, int, float]:
    """定位读取 WAV 小区间；滤波/重采样留边后裁回选区，不载入整条录音。"""
    import soundfile as sf

    if not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
        raise ValueError("时间选区无效")
    with sf.SoundFile(str(audio_path), "r") as handle:
        source_sr = int(handle.samplerate)
        duration_s = len(handle) / float(source_sr)
        start_s = max(0.0, float(start_s))
        end_s = min(duration_s, float(end_s))
        if end_s <= start_s:
            raise ValueError("选区超出 WAV 的时间范围")
        # 两侧仅用于抑制边缘瞬态，不增加最终显示区间。
        margin = int(round(0.05 * source_sr))
        read_start = max(0, int(np.floor(start_s * source_sr)) - margin)
        read_end = min(len(handle), int(np.ceil(end_s * source_sr)) + margin)
        handle.seek(read_start)
        raw = handle.read(read_end - read_start, dtype="float64", always_2d=True)
    y = np.asarray(raw.mean(axis=1), dtype=np.float64)
    y = np.where(np.isfinite(y), y, 0.0)
    if source_sr != TARGET_SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=source_sr, target_sr=TARGET_SAMPLE_RATE)
    if not preprocessed:
        y = highpass_filter(y - np.mean(y), TARGET_SAMPLE_RATE)
    read_offset_s = read_start / float(source_sr)
    first = max(0, int(round((start_s - read_offset_s) * TARGET_SAMPLE_RATE)))
    last = min(len(y), int(round((end_s - read_offset_s) * TARGET_SAMPLE_RATE)))
    selected = y[first:last]
    if len(selected) < 64:
        raise ValueError("选区过短，请选择至少 0.3 毫秒的声音")
    return selected, TARGET_SAMPLE_RATE, read_offset_s + first / float(TARGET_SAMPLE_RATE)


def build_spectrogram(
    y: np.ndarray,
    sr: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """演示检测用声谱图：步长256，保留约1毫秒时间分辨率并减少计算量。"""
    n_fft = 1024
    hop_length = DETECTION_HOP_LENGTH

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


def ultrasonic_band_mask(
    frequencies: np.ndarray,
    sr: int,
) -> np.ndarray:
    """返回落在超声分析频段 [LOW_FREQ_HZ, min(HIGH_FREQ_HZ, sr/2)] 内的掩码。"""
    return (
        (frequencies >= LOW_FREQ_HZ)
        & (frequencies <= min(HIGH_FREQ_HZ, sr / 2))
    )


def adaptive_stft_params(
    sample_count: int,
    max_n_fft: int,
    min_n_fft: int,
    min_hop: int,
) -> tuple[int, int]:
    """按片段长度自适应选择 n_fft 与 hop_length。"""
    n_fft = min(
        max_n_fft,
        2 ** int(np.floor(np.log2(sample_count))),
    )
    n_fft = max(min_n_fft, n_fft)
    hop_length = max(min_hop, n_fft // 16)
    return n_fft, hop_length


def interval_overlap_ratio(
    first_start: float,
    first_end: float,
    second_start: float,
    second_end: float,
) -> float:
    """两个时间区间的重叠长度相对较短区间的比例（无重叠时为 0）。"""
    overlap_s = max(
        0.0,
        min(first_end, second_end)
        - max(first_start, second_start),
    )
    shorter_duration_s = min(
        first_end - first_start,
        second_end - second_start,
    )
    if shorter_duration_s <= 0 or overlap_s <= 0:
        return 0.0
    return overlap_s / shorter_duration_s
