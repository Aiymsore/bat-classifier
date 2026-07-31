from __future__ import annotations

import hashlib
import re
import warnings
import zipfile
from pathlib import Path

import joblib
import librosa
import matplotlib

matplotlib.use("Agg")

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

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# =========================================================
# 1. 可修改配置
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
ZIP_PATH = BASE_DIR / "4287212.zip"
DATA_DIR = BASE_DIR / "bat_data"

FEATURE_CSV = BASE_DIR / "bat_features.csv"
PULSE_CSV = BASE_DIR / "bat_pulses.csv"
MODEL_PATH = BASE_DIR / "bat_random_forest.joblib"  # 旧整段WAV模型路径，保留兼容
PULSE_MODEL_PATH = BASE_DIR / "bat_pulse_random_forest.joblib"
PULSE_CV_PREDICTIONS_CSV = BASE_DIR / "pulse_cv_predictions.csv"
RECORDING_CV_PREDICTIONS_CSV = BASE_DIR / "recording_cv_predictions.csv"
PREDICTION_DIR = BASE_DIR / "predictions"
SPECTROGRAM_DIR = BASE_DIR / "spectrograms"
PULSE_ZOOM_DIR = BASE_DIR / "pulse_zooms"
CUT_PULSE_DIR = BASE_DIR / "cut_pulses"
REJECTED_PULSE_DIR = BASE_DIR / "rejected_pulses"
DUPLICATE_CSV = BASE_DIR / "duplicate_audio.csv"

SELECTED_SPECIES = [
    "Malbescens",
    "Mnigricans",
    "Pparnelli",
]

TARGET_SAMPLE_RATE = 250_000
LOW_FREQ_HZ = 15_000
HIGH_FREQ_HZ = 120_000

# 脉冲检测参数：以后可按声谱图效果微调
ENERGY_THRESHOLD_DB = -28.0       # 相对最强能量的阈值
MIN_PULSE_MS = 0.8
MAX_PULSE_MS = 40.0
MERGE_GAP_MS = 1.5
PULSE_PADDING_MS = 0.4

# 精确边界与质量筛选参数
CORE_THRESHOLD_DB = -24.0
CORE_BACKGROUND_MARGIN_DB = 6.0
CORE_MAX_THRESHOLD_DB = -10.0
CORE_CONTEXT_MS = 2.0
CORE_PADDING_MS = 0.20
CORE_MAX_GAP_MS = 0.35
MIN_PULSE_SNR_DB = 4.0
MIN_TRACK_COVERAGE = 0.55
MAX_TRACK_JUMP_KHZ = 35.0
SKIP_DUPLICATE_AUDIO = True
SAVE_REJECTED_PULSES = True

TOP_PULSES_PER_FILE = 10
ZOOM_CONTEXT_MS = 3.0
CONTOUR_MIN_DB = -35.0

# 脉冲级随机森林配置
PULSE_FEATURE_COLUMNS = [
    "duration_ms",
    "peak_freq_khz",
    "start_freq_khz",
    "end_freq_khz",
    "frequency_drop_khz",
    "f05_khz",
    "f95_khz",
    "bandwidth_90_khz",
    "slope_khz_per_ms",
]

MAX_PULSES_PER_RECORDING = 50
MIN_ACCEPTED_PULSES_PER_RECORDING = 3
MIN_PULSES_FOR_PREDICTION = 3
MAX_GROUP_FOLDS = 5
RECORDING_PROBABILITY_THRESHOLD = 0.60
RECORDING_AGREEMENT_THRESHOLD = 0.60

RANDOM_STATE = 42


# =========================================================
# 2. 数据准备
# =========================================================
def prepare_dataset() -> None:
    """如果 bat_data 中没有 WAV，则自动解压 4287212.zip。"""
    DATA_DIR.mkdir(exist_ok=True)
    SPECTROGRAM_DIR.mkdir(exist_ok=True)
    PULSE_ZOOM_DIR.mkdir(exist_ok=True)
    CUT_PULSE_DIR.mkdir(exist_ok=True)
    REJECTED_PULSE_DIR.mkdir(exist_ok=True)
    PREDICTION_DIR.mkdir(exist_ok=True)

    if any(DATA_DIR.rglob("*.wav")):
        return

    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"没有找到 {ZIP_PATH.name}。请把压缩包放到脚本同目录，"
            f"或者手动把 WAV 解压到 {DATA_DIR}。"
        )

    print(f"正在解压：{ZIP_PATH.name} -> {DATA_DIR.name}")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)


def parse_species(filename: str) -> str:
    """从文件名中提取物种标签。"""
    match = re.match(r"^(.+?)_20\d{2}", Path(filename).stem)
    if not match:
        raise ValueError(f"无法从文件名识别物种：{filename}")
    return match.group(1)


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


def calculate_file_sha256(audio_path: Path) -> str:
    """计算整个文件的SHA-256，用于判断字节级完全重复。"""
    digest = hashlib.sha256()
    with audio_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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



# =========================================================
# 3. 自动检测疑似蝙蝠脉冲
# =========================================================
def build_spectrogram(
    y: np.ndarray,
    sr: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """计算功率声谱图。"""
    n_fft = 1024
    hop_length = 128

    stft = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        window="hann",
        center=True,
    )
    power = np.abs(stft) ** 2
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(
        np.arange(power.shape[1]),
        sr=sr,
        hop_length=hop_length,
    )

    return power, frequencies, times, hop_length


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



# =========================================================
# 4. 单个脉冲特征
# =========================================================
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


def save_cut_pulse(
    audio_path: Path,
    species: str,
    pulse_id: int,
    pulse_y: np.ndarray,
    sr: int,
    accepted: bool,
) -> Path:
    """把单个脉冲保存成独立WAV。"""
    root_dir = (
        CUT_PULSE_DIR
        if accepted
        else REJECTED_PULSE_DIR
    )
    safe_species = species or "unknown"
    output_dir = root_dir / safe_species / audio_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / (
        f"{audio_path.stem}_pulse_{pulse_id:03d}.wav"
    )
    sf.write(
        output_path,
        pulse_y.astype(np.float32),
        sr,
        subtype="PCM_16",
    )
    return output_path


def create_pulse_rows(
    audio_path: Path,
    species: str,
    y: np.ndarray,
    sr: int,
    pulses: list[dict[str, float]],
) -> list[dict[str, float | int | str | bool]]:
    """提取特征、质量筛选，并自动保存单脉冲WAV。"""
    rows: list[dict[str, float | int | str | bool]] = []

    for pulse_id, pulse in enumerate(pulses, start=1):
        start_sample = max(
            0,
            int(round(pulse["start_s"] * sr)),
        )
        end_sample = min(
            len(y),
            int(round(pulse["end_s"] * sr)),
        )
        pulse_y = y[start_sample:end_sample]

        try:
            features = extract_pulse_features(pulse_y, sr)
        except Exception as error:
            print(f"  跳过脉冲 {pulse_id}：{error}")
            continue

        accepted, rejection_reason = evaluate_pulse_quality(
            pulse,
            features,
        )

        pulse_file = ""
        if accepted or SAVE_REJECTED_PULSES:
            saved_path = save_cut_pulse(
                audio_path,
                species,
                pulse_id,
                pulse_y,
                sr,
                accepted,
            )
            pulse_file = str(
                saved_path.relative_to(BASE_DIR)
            )

        rows.append(
            {
                "source_file": audio_path.name,
                "species": species,
                "pulse_id": pulse_id,
                "pulse_file": pulse_file,
                "accepted": accepted,
                "rejection_reason": rejection_reason,
                "candidate_start_s": pulse["candidate_start_s"],
                "candidate_end_s": pulse["candidate_end_s"],
                "start_s": pulse["start_s"],
                "end_s": pulse["end_s"],
                "duration_ms": pulse["duration_ms"],
                "snr_db": pulse["snr_db"],
                "core_threshold_db": pulse["core_threshold_db"],
                **features,
            }
        )

    return rows


# =========================================================
# 5. 保存带脉冲标记的声谱图
# =========================================================
def save_annotated_spectrogram(
    audio_path: Path,
    power: np.ndarray,
    frequencies: np.ndarray,
    times: np.ndarray,
    pulses: list[dict[str, float]],
) -> None:
    spectrogram_db = librosa.power_to_db(
        power + 1e-18,
        ref=np.max,
    )

    plt.figure(figsize=(14, 6))
    plt.pcolormesh(
        times,
        frequencies / 1000,
        spectrogram_db,
        shading="auto",
    )

    for pulse in pulses:
        plt.axvspan(
            pulse["start_s"],
            pulse["end_s"],
            alpha=0.22,
        )

    plt.ylim(
        LOW_FREQ_HZ / 1000,
        min(HIGH_FREQ_HZ, frequencies[-1]) / 1000,
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (kHz)")
    plt.title(
        f"{audio_path.name} | candidate pulses: {len(pulses)}"
    )
    plt.colorbar(label="Relative power (dB)")
    plt.tight_layout()

    output_path = SPECTROGRAM_DIR / f"{audio_path.stem}.png"
    plt.savefig(
        output_path,
        dpi=160,
    )
    plt.close()


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



# =========================================================
# 6. 原有的整段 WAV 特征
# =========================================================
def extract_recording_features(
    y: np.ndarray,
    sr: int,
    pulses: list[dict[str, float]],
) -> dict[str, float | int]:
    power, frequencies, _, hop_length = build_spectrogram(y, sr)

    frequency_mask = (
        (frequencies >= LOW_FREQ_HZ)
        & (frequencies <= min(HIGH_FREQ_HZ, sr / 2))
    )
    frequencies = frequencies[frequency_mask]
    power = power[frequency_mask, :]

    rms = np.sqrt(np.mean(power, axis=0))
    rms_db = librosa.amplitude_to_db(
        rms + 1e-12,
        ref=np.max,
    )
    active_mask = rms_db >= -25.0

    if np.sum(active_mask) < 2:
        keep = min(5, power.shape[1])
        top_indices = np.argsort(rms)[-keep:]
        active_mask = np.zeros(power.shape[1], dtype=bool)
        active_mask[top_indices] = True

    active_power = power[:, active_mask]
    mean_spectrum = np.mean(active_power, axis=1)

    dominant_indices = np.argmax(active_power, axis=0)
    dominant_freqs = frequencies[dominant_indices]

    spectral_sum = np.sum(active_power, axis=0) + 1e-12
    centroids = (
        np.sum(frequencies[:, None] * active_power, axis=0)
        / spectral_sum
    )

    f05 = weighted_frequency_quantile(
        frequencies,
        mean_spectrum,
        0.05,
    )
    f95 = weighted_frequency_quantile(
        frequencies,
        mean_spectrum,
        0.95,
    )
    peak_freq = float(
        frequencies[int(np.argmax(mean_spectrum))]
    )

    active_times = librosa.frames_to_time(
        np.flatnonzero(active_mask),
        sr=sr,
        hop_length=hop_length,
    )

    if len(active_times) >= 2:
        slope_hz_per_s = float(
            np.polyfit(
                active_times,
                dominant_freqs,
                1,
            )[0]
        )
    else:
        slope_hz_per_s = 0.0

    pulse_durations = np.array(
        [pulse["duration_ms"] for pulse in pulses],
        dtype=float,
    )

    pulse_intervals_ms = np.array(
        [
            (pulses[i]["start_s"] - pulses[i - 1]["start_s"])
            * 1000
            for i in range(1, len(pulses))
        ],
        dtype=float,
    )

    return {
        "sample_rate": int(sr),
        "duration_s": float(len(y) / sr),
        "active_duration_s": float(
            np.sum(active_mask) * hop_length / sr
        ),
        "active_ratio": float(np.mean(active_mask)),
        "peak_freq_khz": peak_freq / 1000,
        "dominant_freq_mean_khz": float(
            np.mean(dominant_freqs)
        )
        / 1000,
        "dominant_freq_std_khz": float(
            np.std(dominant_freqs)
        )
        / 1000,
        "dominant_freq_min_khz": float(
            np.min(dominant_freqs)
        )
        / 1000,
        "dominant_freq_max_khz": float(
            np.max(dominant_freqs)
        )
        / 1000,
        "centroid_mean_khz": float(
            np.mean(centroids)
        )
        / 1000,
        "centroid_std_khz": float(
            np.std(centroids)
        )
        / 1000,
        "f05_khz": f05 / 1000,
        "f95_khz": f95 / 1000,
        "bandwidth_90_khz": (f95 - f05) / 1000,
        "dominant_slope_khz_per_s": slope_hz_per_s / 1000,
        "rms_mean": float(np.mean(rms[active_mask])),
        "rms_std": float(np.std(rms[active_mask])),
        "pulse_count": int(len(pulses)),
        "pulse_duration_mean_ms": (
            float(np.mean(pulse_durations))
            if len(pulse_durations)
            else 0.0
        ),
        "pulse_duration_std_ms": (
            float(np.std(pulse_durations))
            if len(pulse_durations)
            else 0.0
        ),
        "pulse_interval_mean_ms": (
            float(np.mean(pulse_intervals_ms))
            if len(pulse_intervals_ms)
            else 0.0
        ),
        "pulse_interval_std_ms": (
            float(np.std(pulse_intervals_ms))
            if len(pulse_intervals_ms)
            else 0.0
        ),
    }


# =========================================================
# 7. 建表：整段录音表 + 脉冲表 + 声谱图
# =========================================================
def build_feature_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    recording_rows: list[dict[str, float | int | str]] = []
    pulse_rows: list[dict[str, float | int | str | bool]] = []
    duplicate_rows: list[dict[str, str]] = []
    seen_audio_hashes: dict[str, Path] = {}

    wav_files = sorted(DATA_DIR.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(
            f"{DATA_DIR} 中没有找到 WAV 文件"
        )

    for index, audio_path in enumerate(wav_files, start=1):
        try:
            species = parse_species(audio_path.name)

            if species not in SELECTED_SPECIES:
                continue

            y, sr = load_audio(audio_path)

            file_hash = calculate_file_sha256(audio_path)
            audio_hash = calculate_audio_content_hash(y, sr)
            duplicate_of = seen_audio_hashes.get(audio_hash)

            if duplicate_of is not None:
                duplicate_rows.append(
                    {
                        "duplicate_file": audio_path.name,
                        "duplicate_of": duplicate_of.name,
                        "file_sha256": file_hash,
                        "audio_sha256": audio_hash,
                    }
                )
                print(
                    f"[{index:02d}/{len(wav_files)}] 重复音频："
                    f"{audio_path.name} == {duplicate_of.name}"
                )
                if SKIP_DUPLICATE_AUDIO:
                    continue
            else:
                seen_audio_hashes[audio_hash] = audio_path

            raw_pulses, power, frequencies, times, threshold_db = detect_pulses(
                y,
                sr,
            )
            refined_pulses = refine_all_pulses(
                y,
                sr,
                raw_pulses,
            )

            current_pulse_rows = create_pulse_rows(
                audio_path,
                species,
                y,
                sr,
                refined_pulses,
            )
            pulse_rows.extend(current_pulse_rows)

            accepted_rows = [
                row
                for row in current_pulse_rows
                if bool(row["accepted"])
            ]
            accepted_pulses = [
                {
                    "start_s": float(row["start_s"]),
                    "end_s": float(row["end_s"]),
                    "duration_ms": float(row["duration_ms"]),
                }
                for row in accepted_rows
            ]

            recording_features = extract_recording_features(
                y,
                sr,
                accepted_pulses,
            )
            recording_features["file"] = audio_path.name
            recording_features["species"] = species
            recording_features["pulse_threshold_db"] = threshold_db
            recording_features["raw_candidate_count"] = len(raw_pulses)
            recording_features["refined_pulse_count"] = len(refined_pulses)
            recording_features["accepted_pulse_count"] = len(accepted_rows)
            recording_rows.append(recording_features)

            save_annotated_spectrogram(
                audio_path,
                power,
                frequencies,
                times,
                accepted_pulses,
            )

            save_top_pulse_zoom_grid(
                audio_path,
                y,
                sr,
                accepted_rows,
            )

            print(
                f"[{index:02d}/{len(wav_files)}] 完成："
                f"{audio_path.name}，"
                f"原始候选 {raw_candidate_count}，"
                f"精修 {len(refined_pulses)}，"
                f"通过质量筛选 {len(accepted_rows)}"
            )

        except Exception as error:
            print(f"跳过 {audio_path.name}：{error}")

    recording_df = pd.DataFrame(recording_rows)
    pulse_df = pd.DataFrame(pulse_rows)
    duplicate_df = pd.DataFrame(duplicate_rows)

    if recording_df.empty:
        raise RuntimeError("没有成功提取任何录音特征")

    recording_df.to_csv(
        FEATURE_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    pulse_df.to_csv(
        PULSE_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    duplicate_df.to_csv(
        DUPLICATE_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n各物种录音样本数：")
    print(
        recording_df["species"]
        .value_counts()
        .sort_index()
    )

    if not pulse_df.empty:
        print("\n各物种候选脉冲数：")
        print(
            pulse_df["species"]
            .value_counts()
            .sort_index()
        )

    print(f"\n整段录音特征表：{FEATURE_CSV}")
    print(f"脉冲特征表：{PULSE_CSV}")
    print(f"标记声谱图目录：{SPECTROGRAM_DIR}")
    print(f"最强脉冲放大图目录：{PULSE_ZOOM_DIR}")
    print(f"通过筛选的单脉冲目录：{CUT_PULSE_DIR}")
    print(f"未通过筛选的单脉冲目录：{REJECTED_PULSE_DIR}")
    print(f"重复音频报告：{DUPLICATE_CSV}")

    return recording_df, pulse_df


# =========================================================
# 8. 脉冲级随机森林：按原始WAV分组验证
# =========================================================
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


def prepare_pulse_training_data(
    pulse_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    清理脉冲表，并限制每个原始WAV最多使用的脉冲数。

    注意：
    - 一个脉冲是一行训练样本；
    - source_file 是分组字段；
    - 同一个 source_file 的脉冲不会跨越训练集和测试集。
    """
    required_columns = {
        "source_file",
        "species",
        "accepted",
        *PULSE_FEATURE_COLUMNS,
    }
    missing_columns = required_columns - set(pulse_df.columns)
    if missing_columns:
        raise RuntimeError(
            f"脉冲表缺少字段：{sorted(missing_columns)}"
        )

    data = pulse_df.loc[
        accepted_mask(pulse_df["accepted"])
    ].copy()

    if data.empty:
        raise RuntimeError("没有通过质量筛选的脉冲，无法训练。")

    for column in PULSE_FEATURE_COLUMNS:
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
            "species",
            *PULSE_FEATURE_COLUMNS,
        ]
    )

    # 至少含有若干合格脉冲的原始录音才进入训练
    pulse_count_by_recording = (
        data.groupby("source_file")
        .size()
    )
    valid_recordings = pulse_count_by_recording[
        pulse_count_by_recording
        >= MIN_ACCEPTED_PULSES_PER_RECORDING
    ].index
    data = data[
        data["source_file"].isin(valid_recordings)
    ].copy()

    if data.empty:
        raise RuntimeError(
            "没有原始WAV达到最低合格脉冲数量要求。"
        )

    # 防止某个录音含大量脉冲，从而压倒其他录音
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
        data["track_coverage"] = pd.to_numeric(
            data["track_coverage"],
            errors="coerce",
        ).fillna(0.0)
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
        .head(MAX_PULSES_PER_RECORDING)
        .reset_index(drop=True)
    )

    recording_species = (
        data[["source_file", "species"]]
        .drop_duplicates()
    )
    species_group_counts = (
        recording_species["species"]
        .value_counts()
        .sort_index()
    )

    if (species_group_counts < 2).any():
        insufficient = species_group_counts[
            species_group_counts < 2
        ].to_dict()
        raise RuntimeError(
            "每个物种至少需要2个独立原始WAV才能做分组验证。"
            f"不足的物种：{insufficient}"
        )

    print("\n进入脉冲级训练的数据：")
    print(f"  合格脉冲总数：{len(data)}")
    print(
        f"  独立原始WAV数："
        f"{data['source_file'].nunique()}"
    )
    print("\n每个物种的独立WAV数：")
    print(species_group_counts)
    print("\n每个物种使用的脉冲数：")
    print(
        data["species"]
        .value_counts()
        .sort_index()
    )

    return data


def make_pulse_model(
    random_state: int,
) -> RandomForestClassifier:
    """创建脉冲级随机森林。"""
    return RandomForestClassifier(
        n_estimators=600,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )


def aggregate_recording_predictions(
    prediction_df: pd.DataFrame,
    classes: np.ndarray,
) -> pd.DataFrame:
    """把同一个原始WAV的多个脉冲预测汇总成一次录音预测。"""
    rows: list[dict[str, float | int | str]] = []

    probability_columns = [
        f"prob_{species}"
        for species in classes
    ]

    for source_file, group in prediction_df.groupby(
        "source_file",
        sort=True,
    ):
        true_species_values = group["species"].unique()
        if len(true_species_values) != 1:
            raise RuntimeError(
                f"{source_file} 出现多个物种标签："
                f"{true_species_values}"
            )

        mean_probabilities = (
            group[probability_columns]
            .mean()
            .to_numpy(dtype=float)
        )
        order = np.argsort(mean_probabilities)[::-1]
        best_index = int(order[0])
        second_index = (
            int(order[1])
            if len(order) > 1
            else best_index
        )

        predicted_species = str(classes[best_index])
        pulse_votes = group["pulse_prediction"].astype(str)
        agreement = float(
            np.mean(
                pulse_votes.to_numpy()
                == predicted_species
            )
        )

        row: dict[str, float | int | str] = {
            "source_file": source_file,
            "species": str(true_species_values[0]),
            "recording_prediction": predicted_species,
            "accepted_pulse_count": int(len(group)),
            "mean_top_probability": float(
                mean_probabilities[best_index]
            ),
            "probability_margin": float(
                mean_probabilities[best_index]
                - mean_probabilities[second_index]
            ),
            "pulse_vote_agreement": agreement,
        }

        for species, probability in zip(
            classes,
            mean_probabilities,
        ):
            row[f"mean_prob_{species}"] = float(
                probability
            )

        rows.append(row)

    return pd.DataFrame(rows)


def train_pulse_model(
    pulse_df: pd.DataFrame,
) -> tuple[RandomForestClassifier, list[str]]:
    """
    使用合格单脉冲训练随机森林。

    交叉验证严格按 source_file 分组：
    同一个原始WAV中的所有脉冲只能在同一折中。
    """
    data = prepare_pulse_training_data(pulse_df)

    X = data[PULSE_FEATURE_COLUMNS].copy()
    y = data["species"].astype(str)
    groups = data["source_file"].astype(str)
    classes = np.array(
        sorted(y.unique()),
        dtype=object,
    )

    group_species = (
        data[["source_file", "species"]]
        .drop_duplicates()
    )
    min_groups_per_species = int(
        group_species["species"]
        .value_counts()
        .min()
    )
    n_splits = min(
        MAX_GROUP_FOLDS,
        min_groups_per_species,
    )

    if n_splits < 2:
        raise RuntimeError(
            "独立原始WAV数量不足，无法进行分组交叉验证。"
        )

    print(
        f"\n开始 {n_splits} 折分层分组交叉验证："
        "同一原始WAV不会同时进入训练集和测试集。"
    )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    out_of_fold_probabilities = np.full(
        (len(data), len(classes)),
        np.nan,
        dtype=float,
    )
    out_of_fold_predictions = np.empty(
        len(data),
        dtype=object,
    )
    fold_ids = np.zeros(
        len(data),
        dtype=int,
    )

    for fold_id, (train_indices, test_indices) in enumerate(
        splitter.split(
            X,
            y,
            groups,
        ),
        start=1,
    ):
        train_groups = set(
            groups.iloc[train_indices]
        )
        test_groups = set(
            groups.iloc[test_indices]
        )

        overlap = train_groups & test_groups
        if overlap:
            raise RuntimeError(
                f"第{fold_id}折发生分组泄漏：{overlap}"
            )

        model = make_pulse_model(
            RANDOM_STATE + fold_id,
        )
        model.fit(
            X.iloc[train_indices],
            y.iloc[train_indices],
        )

        fold_probabilities = model.predict_proba(
            X.iloc[test_indices]
        )
        fold_predictions = model.predict(
            X.iloc[test_indices]
        )

        # 按全局类别顺序写入概率
        for local_class_index, species in enumerate(
            model.classes_
        ):
            global_class_index = int(
                np.where(classes == species)[0][0]
            )
            out_of_fold_probabilities[
                test_indices,
                global_class_index,
            ] = fold_probabilities[
                :,
                local_class_index,
            ]

        out_of_fold_predictions[
            test_indices
        ] = fold_predictions
        fold_ids[test_indices] = fold_id

        print(
            f"  第{fold_id}折："
            f"训练WAV {len(train_groups)}个，"
            f"测试WAV {len(test_groups)}个，"
            f"测试脉冲 {len(test_indices)}个"
        )

    if np.isnan(out_of_fold_probabilities).any():
        raise RuntimeError(
            "交叉验证概率存在缺失，可能有类别未进入某一折训练集。"
        )

    pulse_prediction_df = data[
        [
            "source_file",
            "species",
            "pulse_id",
            "pulse_file",
        ]
    ].copy()
    pulse_prediction_df["fold"] = fold_ids
    pulse_prediction_df[
        "pulse_prediction"
    ] = out_of_fold_predictions

    for class_index, species in enumerate(classes):
        pulse_prediction_df[
            f"prob_{species}"
        ] = out_of_fold_probabilities[
            :,
            class_index,
        ]

    pulse_prediction_df.to_csv(
        PULSE_CV_PREDICTIONS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    pulse_accuracy = accuracy_score(
        y,
        out_of_fold_predictions,
    )
    print(
        f"\n脉冲级交叉验证准确率："
        f"{pulse_accuracy:.2%}"
    )
    print("\n脉冲级混淆矩阵：")
    print(
        confusion_matrix(
            y,
            out_of_fold_predictions,
            labels=classes,
        )
    )
    print("\n脉冲级分类报告：")
    print(
        classification_report(
            y,
            out_of_fold_predictions,
            labels=classes,
            zero_division=0,
        )
    )

    recording_prediction_df = (
        aggregate_recording_predictions(
            pulse_prediction_df,
            classes,
        )
    )
    recording_prediction_df.to_csv(
        RECORDING_CV_PREDICTIONS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    recording_accuracy = accuracy_score(
        recording_prediction_df["species"],
        recording_prediction_df[
            "recording_prediction"
        ],
    )
    print(
        f"\n整段录音级交叉验证准确率："
        f"{recording_accuracy:.2%}"
    )
    print("\n整段录音级混淆矩阵：")
    print(
        confusion_matrix(
            recording_prediction_df["species"],
            recording_prediction_df[
                "recording_prediction"
            ],
            labels=classes,
        )
    )
    print("\n整段录音级结果：")
    print(
        recording_prediction_df[
            [
                "source_file",
                "species",
                "recording_prediction",
                "accepted_pulse_count",
                "mean_top_probability",
                "pulse_vote_agreement",
            ]
        ].to_string(index=False)
    )

    # 最终模型使用全部合格脉冲训练，供以后识别新WAV
    final_model = make_pulse_model(
        RANDOM_STATE,
    )
    final_model.fit(X, y)

    payload = {
        "model": final_model,
        "feature_columns": PULSE_FEATURE_COLUMNS,
        "selected_species": list(classes),
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "low_freq_hz": LOW_FREQ_HZ,
        "high_freq_hz": HIGH_FREQ_HZ,
        "group_field": "source_file",
        "training_pulse_count": int(len(data)),
        "training_recording_count": int(
            data["source_file"].nunique()
        ),
    }
    joblib.dump(
        payload,
        PULSE_MODEL_PATH,
    )

    print(
        f"\n脉冲级模型已保存：{PULSE_MODEL_PATH}"
    )
    print(
        f"脉冲交叉验证明细："
        f"{PULSE_CV_PREDICTIONS_CSV}"
    )
    print(
        f"整段录音交叉验证明细："
        f"{RECORDING_CV_PREDICTIONS_CSV}"
    )

    importance = pd.Series(
        final_model.feature_importances_,
        index=PULSE_FEATURE_COLUMNS,
    ).sort_values(ascending=False)
    print("\n脉冲模型特征重要性：")
    print(importance)

    return final_model, list(PULSE_FEATURE_COLUMNS)


# =========================================================
# 9. 新WAV：自动切脉冲后，多脉冲投票识别
# =========================================================
def predict_file(
    audio_path: Path,
    model: RandomForestClassifier,
    feature_columns: list[str],
) -> None:
    y, sr = load_audio(audio_path)
    raw_pulses, power, frequencies, times, _ = detect_pulses(
        y,
        sr,
    )
    refined_pulses = refine_all_pulses(
        y,
        sr,
        raw_pulses,
    )
    current_pulse_rows = create_pulse_rows(
        audio_path,
        "unknown",
        y,
        sr,
        refined_pulses,
    )
    accepted_rows = [
        row
        for row in current_pulse_rows
        if bool(row["accepted"])
    ]

    accepted_pulses = [
        {
            "start_s": float(row["start_s"]),
            "end_s": float(row["end_s"]),
            "duration_ms": float(row["duration_ms"]),
        }
        for row in accepted_rows
    ]

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

    print(f"\n待预测文件：{audio_path.name}")
    print(f"原始候选脉冲：{len(raw_pulses)} 个")
    print(f"精修后脉冲：{len(refined_pulses)} 个")
    print(f"通过质量筛选：{len(accepted_rows)} 个")

    if len(accepted_rows) < MIN_PULSES_FOR_PREDICTION:
        print(
            f"有效脉冲少于"
            f"{MIN_PULSES_FOR_PREDICTION}个，"
            "无法可靠判断。"
        )
        return

    pulse_input_df = pd.DataFrame(
        accepted_rows
    )

    for column in feature_columns:
        pulse_input_df[column] = pd.to_numeric(
            pulse_input_df[column],
            errors="coerce",
        )

    valid_mask = (
        pulse_input_df[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .all(axis=1)
    )
    pulse_input_df = pulse_input_df.loc[
        valid_mask
    ].copy()

    if len(pulse_input_df) < MIN_PULSES_FOR_PREDICTION:
        print(
            "清理无效特征后可用脉冲不足，"
            "无法可靠判断。"
        )
        return

    if len(pulse_input_df) > MAX_PULSES_PER_RECORDING:
        sort_columns: list[str] = []
        ascending: list[bool] = []

        if "snr_db" in pulse_input_df.columns:
            sort_columns.append("snr_db")
            ascending.append(False)

        if "track_coverage" in pulse_input_df.columns:
            sort_columns.append("track_coverage")
            ascending.append(False)

        if sort_columns:
            pulse_input_df = (
                pulse_input_df
                .sort_values(
                    sort_columns,
                    ascending=ascending,
                )
                .head(MAX_PULSES_PER_RECORDING)
                .copy()
            )
        else:
            pulse_input_df = (
                pulse_input_df
                .head(MAX_PULSES_PER_RECORDING)
                .copy()
            )

    pulse_probabilities = model.predict_proba(
        pulse_input_df[feature_columns]
    )
    pulse_predictions = model.predict(
        pulse_input_df[feature_columns]
    )

    mean_probabilities = np.mean(
        pulse_probabilities,
        axis=0,
    )
    order = np.argsort(mean_probabilities)[::-1]
    best_index = int(order[0])
    predicted_species = str(
        model.classes_[best_index]
    )
    agreement = float(
        np.mean(
            pulse_predictions == predicted_species
        )
    )

    print("\n整段录音候选物种：")
    for class_index in order[:3]:
        print(
            f"  {model.classes_[class_index]:<15} "
            f"{mean_probabilities[class_index]:.2%}"
        )

    print(
        f"\n用于投票的有效脉冲："
        f"{len(pulse_input_df)} 个"
    )
    print(
        f"脉冲投票一致率：{agreement:.2%}"
    )

    needs_review = (
        mean_probabilities[best_index]
        < RECORDING_PROBABILITY_THRESHOLD
        or agreement
        < RECORDING_AGREEMENT_THRESHOLD
    )

    if needs_review:
        print(
            "结论：需要人工复核。"
            "最高平均概率或脉冲一致率不足。"
        )
    else:
        print(
            f"结论：当前模型判断为 "
            f"{predicted_species}。"
        )

    output_df = pulse_input_df[
        [
            "pulse_id",
            "pulse_file",
            "start_s",
            "end_s",
            "duration_ms",
            *feature_columns,
        ]
    ].copy()
    output_df[
        "pulse_prediction"
    ] = pulse_predictions

    for class_index, species in enumerate(
        model.classes_
    ):
        output_df[
            f"prob_{species}"
        ] = pulse_probabilities[
            :,
            class_index,
        ]

    prediction_output_path = (
        PREDICTION_DIR
        / f"{audio_path.stem}_pulse_predictions.csv"
    )
    output_df.to_csv(
        prediction_output_path,
        index=False,
        encoding="utf-8-sig",
    )
    print(
        f"逐脉冲预测结果："
        f"{prediction_output_path}"
    )


def main() -> None:
    prepare_dataset()
    _, pulse_df = build_feature_tables()
    model, feature_columns = train_pulse_model(
        pulse_df
    )

    raw_path = input(
        "\n输入要预测的WAV路径，直接回车可跳过："
    ).strip().strip('"').lstrip("\ufeff").strip()

    if not raw_path:
        return

    audio_path = Path(raw_path)

    if not audio_path.is_absolute():
        audio_path = BASE_DIR / audio_path

    if not audio_path.exists():
        print(f"找不到文件：{audio_path}")
        return

    predict_file(
        audio_path,
        model,
        feature_columns,
    )



# =========================================================
# Step5：加入 noise 与 unknown_bat 的开放集分类
# =========================================================

TARGET_SPECIES = list(SELECTED_SPECIES)
UNKNOWN_BAT_CLASS = "unknown_bat"
NOISE_CLASS = "noise"

OPEN_SET_MODEL_PATH = (
    BASE_DIR / "bat_open_set_random_forest.joblib"
)
OPEN_SET_PULSE_CV_CSV = (
    BASE_DIR / "open_set_pulse_cv_predictions.csv"
)
OPEN_SET_RECORDING_CV_CSV = (
    BASE_DIR / "open_set_recording_cv_predictions.csv"
)
OPEN_SET_PULSE_CSV = BASE_DIR / "bat_pulses_open_set.csv"
OPEN_SET_FEATURE_CSV = BASE_DIR / "bat_features_open_set.csv"
OPEN_SET_DUPLICATE_CSV = (
    BASE_DIR / "duplicate_audio_open_set.csv"
)

NOISE_PULSE_DIR = BASE_DIR / "noise_pulses"
BACKGROUND_NOISE_DIR = NOISE_PULSE_DIR / "background"

# 其他所有已知蝙蝠物种合并成 unknown_bat
USE_OTHER_SPECIES_AS_UNKNOWN = True

# rejected 中只有比较明确的错误候选才用于 noise。
# 仅仅 low_snr 的片段可能仍是真实弱蝙蝠声，因此不直接当作 noise。
USE_HARD_REJECTS_AS_NOISE = True
HARD_NOISE_REASONS = {
    "low_track_coverage",
    "unstable_frequency_track",
    "peak_frequency_out_of_range",
}

# 额外从脉冲之外抽取安静背景，作为 noise 样本
USE_BACKGROUND_AS_NOISE = True
BACKGROUND_WINDOWS_PER_RECORDING = 8
BACKGROUND_CANDIDATES_MULTIPLIER = 12
BACKGROUND_WINDOW_DURATIONS_MS = (
    4.0,
    8.0,
    12.0,
    20.0,
)
BACKGROUND_EXCLUSION_MARGIN_MS = 3.0
BACKGROUND_MIN_RMS = 1e-7

MAX_NOISE_PULSES_PER_RECORDING = 20
MAX_UNKNOWN_PULSES_PER_RECORDING = 40

# 新录音判断规则
NOISE_PULSE_PROBABILITY_THRESHOLD = 0.55
MAX_RECORDING_MEAN_NOISE_PROBABILITY = 0.70
MIN_NON_NOISE_PULSES_FOR_PREDICTION = 3

# 为避免一次生成过多图片，默认只给目标物种生成完整诊断图。
SAVE_UNKNOWN_DIAGNOSTICS = False


def prepare_dataset() -> None:
    """准备目录和原始数据。"""
    DATA_DIR.mkdir(exist_ok=True)
    SPECTROGRAM_DIR.mkdir(exist_ok=True)
    PULSE_ZOOM_DIR.mkdir(exist_ok=True)
    CUT_PULSE_DIR.mkdir(exist_ok=True)
    REJECTED_PULSE_DIR.mkdir(exist_ok=True)
    PREDICTION_DIR.mkdir(exist_ok=True)
    NOISE_PULSE_DIR.mkdir(exist_ok=True)
    BACKGROUND_NOISE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if any(DATA_DIR.rglob("*.wav")):
        return

    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"没有找到 {ZIP_PATH.name}。"
            f"请把压缩包放到脚本同目录，"
            f"或者手动把 WAV 解压到 {DATA_DIR}。"
        )

    print(f"正在解压：{ZIP_PATH.name} -> {DATA_DIR.name}")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)


def map_recording_label(
    original_species: str,
) -> str:
    """
    把原始物种映射为模型类别。

    三个目标物种保留原名，其他蝙蝠物种合并为 unknown_bat。
    """
    if original_species in TARGET_SPECIES:
        return original_species

    if USE_OTHER_SPECIES_AS_UNKNOWN:
        return UNKNOWN_BAT_CLASS

    return ""


def is_hard_noise_reject(
    rejection_reason: str,
) -> bool:
    """判断一个 rejected 候选是否足够明确，可用于 noise 训练。"""
    reasons = {
        item.strip()
        for item in str(rejection_reason).split(";")
        if item.strip()
    }

    return bool(reasons & HARD_NOISE_REASONS)


def intervals_overlap(
    start_s: float,
    end_s: float,
    excluded_intervals: list[tuple[float, float]],
) -> bool:
    for excluded_start, excluded_end in excluded_intervals:
        if (
            start_s < excluded_end
            and end_s > excluded_start
        ):
            return True
    return False


def save_background_noise_pulse(
    audio_path: Path,
    noise_id: int,
    segment: np.ndarray,
    sr: int,
) -> Path:
    output_dir = (
        BACKGROUND_NOISE_DIR
        / audio_path.stem
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / (
        f"{audio_path.stem}_background_"
        f"{noise_id:03d}.wav"
    )
    sf.write(
        output_path,
        segment.astype(np.float32),
        sr,
        subtype="PCM_16",
    )
    return output_path


def create_background_noise_rows(
    audio_path: Path,
    original_species: str,
    recording_label: str,
    y: np.ndarray,
    sr: int,
    excluded_pulses: list[dict[str, float]],
) -> list[dict[str, float | int | str | bool]]:
    """
    从蝙蝠脉冲以外的安静区间抽取背景片段，作为 noise。

    为防止把真实叫声切进 noise：
    - 候选窗口不得与任何精修脉冲重叠；
    - 脉冲前后额外留出排除边界；
    - 从大量候选里选择 RMS 最低的一部分。
    """
    if not USE_BACKGROUND_AS_NOISE:
        return []

    duration_s = len(y) / sr
    if duration_s <= 0.05:
        return []

    margin_s = (
        BACKGROUND_EXCLUSION_MARGIN_MS / 1000
    )
    excluded_intervals = [
        (
            max(0.0, float(pulse["start_s"]) - margin_s),
            min(
                duration_s,
                float(pulse["end_s"]) + margin_s,
            ),
        )
        for pulse in excluded_pulses
    ]

    seed = int(
        hashlib.sha256(
            audio_path.name.encode("utf-8")
        ).hexdigest()[:8],
        16,
    )
    rng = np.random.default_rng(seed)

    candidate_count = (
        BACKGROUND_WINDOWS_PER_RECORDING
        * BACKGROUND_CANDIDATES_MULTIPLIER
    )
    candidate_segments: list[
        tuple[float, float, float, np.ndarray]
    ] = []

    for _ in range(candidate_count):
        window_ms = float(
            rng.choice(
                BACKGROUND_WINDOW_DURATIONS_MS
            )
        )
        window_s = window_ms / 1000

        if duration_s <= window_s:
            continue

        start_s = float(
            rng.uniform(
                0.0,
                duration_s - window_s,
            )
        )
        end_s = start_s + window_s

        if intervals_overlap(
            start_s,
            end_s,
            excluded_intervals,
        ):
            continue

        start_sample = int(round(start_s * sr))
        end_sample = int(round(end_s * sr))
        segment = y[start_sample:end_sample]

        if len(segment) < 32:
            continue

        rms = float(
            np.sqrt(np.mean(segment**2))
        )
        if not np.isfinite(rms):
            continue
        if rms < BACKGROUND_MIN_RMS:
            continue

        candidate_segments.append(
            (
                rms,
                start_s,
                end_s,
                segment.copy(),
            )
        )

    candidate_segments.sort(
        key=lambda item: item[0]
    )
    selected_segments = candidate_segments[
        :BACKGROUND_WINDOWS_PER_RECORDING
    ]

    rows: list[
        dict[str, float | int | str | bool]
    ] = []

    for noise_id, (
        rms,
        start_s,
        end_s,
        segment,
    ) in enumerate(
        selected_segments,
        start=1,
    ):
        try:
            features = extract_pulse_features(
                segment,
                sr,
            )
        except Exception:
            continue

        saved_path = save_background_noise_pulse(
            audio_path,
            noise_id,
            segment,
            sr,
        )

        rows.append(
            {
                "source_file": audio_path.name,
                "group_id": audio_path.name,
                "original_species": original_species,
                "recording_species": recording_label,
                "species": NOISE_CLASS,
                "training_label": NOISE_CLASS,
                "trainable": True,
                "pulse_source": "background",
                "pulse_id": 100000 + noise_id,
                "pulse_file": str(
                    saved_path.relative_to(BASE_DIR)
                ),
                "accepted": False,
                "detector_accepted": False,
                "rejection_reason": "background_window",
                "candidate_start_s": start_s,
                "candidate_end_s": end_s,
                "start_s": start_s,
                "end_s": end_s,
                "duration_ms": (
                    end_s - start_s
                )
                * 1000,
                "snr_db": 0.0,
                "core_threshold_db": np.nan,
                **features,
            }
        )

    return rows


def convert_candidate_rows_to_open_set(
    candidate_rows: list[
        dict[str, float | int | str | bool]
    ],
    original_species: str,
    recording_label: str,
) -> list[dict[str, float | int | str | bool]]:
    """
    把 Step4 的候选脉冲转换为开放集训练标签。

    - accepted=True：
      目标种保持原名，非目标蝙蝠标为 unknown_bat。
    - 明确的 rejected：
      标为 noise。
    - 仅因低 SNR 等模糊原因被拒绝：
      保留在 CSV 中，但不参加训练。
    """
    rows: list[
        dict[str, float | int | str | bool]
    ] = []

    for row in candidate_rows:
        new_row = dict(row)
        detector_accepted = bool(
            row.get("accepted", False)
        )

        new_row["group_id"] = row["source_file"]
        new_row["original_species"] = original_species
        new_row["recording_species"] = (
            recording_label
        )
        new_row["detector_accepted"] = (
            detector_accepted
        )
        new_row["pulse_source"] = (
            "accepted_bat"
            if detector_accepted
            else "rejected_candidate"
        )

        if detector_accepted:
            new_row["species"] = recording_label
            new_row["training_label"] = (
                recording_label
            )
            new_row["trainable"] = True

        elif (
            USE_HARD_REJECTS_AS_NOISE
            and is_hard_noise_reject(
                str(
                    row.get(
                        "rejection_reason",
                        "",
                    )
                )
            )
        ):
            new_row["species"] = NOISE_CLASS
            new_row["training_label"] = NOISE_CLASS
            new_row["trainable"] = True

        else:
            new_row["species"] = ""
            new_row["training_label"] = ""
            new_row["trainable"] = False

        rows.append(new_row)

    return rows


def build_feature_tables() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    建立五类训练数据：

    目标物种1、目标物种2、目标物种3、unknown_bat、noise。
    """
    recording_rows: list[
        dict[str, float | int | str]
    ] = []
    pulse_rows: list[
        dict[str, float | int | str | bool]
    ] = []
    duplicate_rows: list[dict[str, str]] = []
    seen_audio_hashes: dict[str, Path] = {}

    wav_files = sorted(DATA_DIR.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(
            f"{DATA_DIR} 中没有找到 WAV 文件"
        )

    for index, audio_path in enumerate(
        wav_files,
        start=1,
    ):
        try:
            original_species = parse_species(
                audio_path.name
            )
            recording_label = map_recording_label(
                original_species
            )

            if not recording_label:
                continue

            y, sr = load_audio(audio_path)

            file_hash = calculate_file_sha256(
                audio_path
            )
            audio_hash = calculate_audio_content_hash(
                y,
                sr,
            )
            duplicate_of = seen_audio_hashes.get(
                audio_hash
            )

            if duplicate_of is not None:
                duplicate_rows.append(
                    {
                        "duplicate_file": (
                            audio_path.name
                        ),
                        "duplicate_of": (
                            duplicate_of.name
                        ),
                        "file_sha256": file_hash,
                        "audio_sha256": audio_hash,
                    }
                )
                print(
                    f"[{index:02d}/{len(wav_files)}] "
                    f"重复音频：{audio_path.name} "
                    f"== {duplicate_of.name}"
                )
                if SKIP_DUPLICATE_AUDIO:
                    continue
            else:
                seen_audio_hashes[
                    audio_hash
                ] = audio_path

            (
                raw_candidate_count,
                refined_pulses,
                threshold_db,
            ) = detect_and_refine_pulses_chunked(
                y,
                sr,
            )

            base_candidate_rows = create_pulse_rows(
                audio_path,
                recording_label,
                y,
                sr,
                refined_pulses,
            )
            current_rows = (
                convert_candidate_rows_to_open_set(
                    base_candidate_rows,
                    original_species,
                    recording_label,
                )
            )

            background_rows = (
                create_background_noise_rows(
                    audio_path,
                    original_species,
                    recording_label,
                    y,
                    sr,
                    refined_pulses,
                )
            )
            current_rows.extend(background_rows)
            pulse_rows.extend(current_rows)

            accepted_bat_rows = [
                row
                for row in current_rows
                if (
                    bool(
                        row.get(
                            "detector_accepted",
                            False,
                        )
                    )
                    and row.get("species")
                    != NOISE_CLASS
                )
            ]
            accepted_pulses = [
                {
                    "start_s": float(row["start_s"]),
                    "end_s": float(row["end_s"]),
                    "duration_ms": float(
                        row["duration_ms"]
                    ),
                }
                for row in accepted_bat_rows
            ]

            recording_features = (
                extract_recording_features(
                    y,
                    sr,
                    accepted_pulses,
                )
            )
            recording_features[
                "file"
            ] = audio_path.name
            recording_features[
                "original_species"
            ] = original_species
            recording_features[
                "species"
            ] = recording_label
            recording_features[
                "pulse_threshold_db"
            ] = threshold_db
            recording_features[
                "raw_candidate_count"
            ] = len(raw_pulses)
            recording_features[
                "refined_pulse_count"
            ] = len(refined_pulses)
            recording_features[
                "accepted_bat_pulse_count"
            ] = len(accepted_bat_rows)
            recording_features[
                "noise_training_count"
            ] = sum(
                row.get("species") == NOISE_CLASS
                and bool(
                    row.get(
                        "trainable",
                        False,
                    )
                )
                for row in current_rows
            )
            recording_rows.append(
                recording_features
            )

            should_save_diagnostics = (
                original_species in TARGET_SPECIES
                or SAVE_UNKNOWN_DIAGNOSTICS
            )
            if should_save_diagnostics:
                save_annotated_spectrogram(
                    audio_path,
                    power,
                    frequencies,
                    times,
                    accepted_pulses,
                )
                save_top_pulse_zoom_grid(
                    audio_path,
                    y,
                    sr,
                    accepted_bat_rows,
                )

            target_text = (
                recording_label
                if original_species in TARGET_SPECIES
                else (
                    f"{UNKNOWN_BAT_CLASS}"
                    f"（原始：{original_species}）"
                )
            )
            print(
                f"[{index:02d}/{len(wav_files)}] "
                f"完成：{audio_path.name} -> "
                f"{target_text}，"
                f"蝙蝠脉冲 "
                f"{len(accepted_bat_rows)}，"
                f"noise训练片段 "
                f"{recording_features['noise_training_count']}"
            )

        except Exception as error:
            print(
                f"跳过 {audio_path.name}：{error}"
            )

    recording_df = pd.DataFrame(
        recording_rows
    )
    pulse_df = pd.DataFrame(pulse_rows)
    duplicate_df = pd.DataFrame(
        duplicate_rows
    )

    if recording_df.empty:
        raise RuntimeError(
            "没有成功提取任何录音特征"
        )
    if pulse_df.empty:
        raise RuntimeError(
            "没有成功提取任何脉冲特征"
        )

    recording_df.to_csv(
        OPEN_SET_FEATURE_CSV,
        index=False,
        encoding="utf-8-sig",
    )
    pulse_df.to_csv(
        OPEN_SET_PULSE_CSV,
        index=False,
        encoding="utf-8-sig",
    )
    duplicate_df.to_csv(
        OPEN_SET_DUPLICATE_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    trainable_df = pulse_df.loc[
        pulse_df["trainable"]
        .astype(str)
        .str.lower()
        .isin({"true", "1"})
    ]

    print("\n开放集训练类别脉冲数：")
    print(
        trainable_df["species"]
        .value_counts()
        .sort_index()
    )

    recording_group_table = (
        trainable_df[
            [
                "group_id",
                "species",
            ]
        ]
        .drop_duplicates()
    )
    print("\n开放集训练类别独立WAV数：")
    print(
        recording_group_table["species"]
        .value_counts()
        .sort_index()
    )

    print(
        f"\n开放集整段特征表："
        f"{OPEN_SET_FEATURE_CSV}"
    )
    print(
        f"开放集脉冲表："
        f"{OPEN_SET_PULSE_CSV}"
    )
    print(
        f"开放集重复音频报告："
        f"{OPEN_SET_DUPLICATE_CSV}"
    )

    return recording_df, pulse_df


def trainable_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def prepare_pulse_training_data(
    pulse_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    准备五类训练数据。

    同一个原始WAV中的目标脉冲、unknown脉冲和noise片段
    共享同一个 group_id，因此不会跨越训练折和测试折。
    """
    required_columns = {
        "source_file",
        "group_id",
        "recording_species",
        "species",
        "trainable",
        *PULSE_FEATURE_COLUMNS,
    }
    missing_columns = (
        required_columns - set(pulse_df.columns)
    )
    if missing_columns:
        raise RuntimeError(
            f"脉冲表缺少字段："
            f"{sorted(missing_columns)}"
        )

    data = pulse_df.loc[
        trainable_mask(pulse_df["trainable"])
    ].copy()

    if data.empty:
        raise RuntimeError(
            "没有可用于开放集训练的脉冲。"
        )

    for column in PULSE_FEATURE_COLUMNS:
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
            "group_id",
            "recording_species",
            "species",
            *PULSE_FEATURE_COLUMNS,
        ]
    )

    # 每个原始WAV至少要有一定数量的非noise蝙蝠脉冲，
    # noise本身则只要求能提取出有效特征。
    non_noise = data[
        data["species"] != NOISE_CLASS
    ]
    valid_bat_groups = (
        non_noise.groupby("group_id")
        .size()
    )
    valid_bat_groups = valid_bat_groups[
        valid_bat_groups
        >= MIN_ACCEPTED_PULSES_PER_RECORDING
    ].index

    data = data[
        data["group_id"].isin(
            valid_bat_groups
        )
    ].copy()

    if data.empty:
        raise RuntimeError(
            "没有原始WAV达到最低合格蝙蝠脉冲数量。"
        )

    if "snr_db" in data.columns:
        data["snr_db"] = pd.to_numeric(
            data["snr_db"],
            errors="coerce",
        ).fillna(-999.0)

    if "track_coverage" in data.columns:
        data["track_coverage"] = pd.to_numeric(
            data["track_coverage"],
            errors="coerce",
        ).fillna(0.0)

    kept_parts: list[pd.DataFrame] = []

    for (
        group_id,
        class_name,
    ), group in data.groupby(
        [
            "group_id",
            "species",
        ],
        sort=False,
    ):
        if class_name == NOISE_CLASS:
            limit = (
                MAX_NOISE_PULSES_PER_RECORDING
            )
        elif class_name == UNKNOWN_BAT_CLASS:
            limit = (
                MAX_UNKNOWN_PULSES_PER_RECORDING
            )
        else:
            limit = MAX_PULSES_PER_RECORDING

        sort_columns: list[str] = []
        ascending: list[bool] = []

        if class_name == NOISE_CLASS:
            # noise优先保留更接近误检的困难样本，
            # 而不是只保留完全安静的片段。
            if "pulse_source" in group.columns:
                group = group.assign(
                    _noise_priority=(
                        group["pulse_source"]
                        == "rejected_candidate"
                    ).astype(int)
                )
                sort_columns.append(
                    "_noise_priority"
                )
                ascending.append(False)

            if "rms" in group.columns:
                sort_columns.append("rms")
                ascending.append(False)

        else:
            if "snr_db" in group.columns:
                sort_columns.append("snr_db")
                ascending.append(False)
            if "track_coverage" in group.columns:
                sort_columns.append(
                    "track_coverage"
                )
                ascending.append(False)

        if sort_columns:
            group = group.sort_values(
                sort_columns,
                ascending=ascending,
            )

        kept_parts.append(
            group.head(limit)
        )

    data = pd.concat(
        kept_parts,
        ignore_index=True,
    )

    group_class_table = (
        data[
            [
                "group_id",
                "species",
            ]
        ]
        .drop_duplicates()
    )
    class_group_counts = (
        group_class_table["species"]
        .value_counts()
        .sort_index()
    )

    required_classes = {
        *TARGET_SPECIES,
        UNKNOWN_BAT_CLASS,
        NOISE_CLASS,
    }
    missing_classes = (
        required_classes
        - set(data["species"].unique())
    )
    if missing_classes:
        raise RuntimeError(
            "以下类别没有训练数据："
            f"{sorted(missing_classes)}"
        )

    insufficient = class_group_counts[
        class_group_counts < 2
    ]
    if not insufficient.empty:
        raise RuntimeError(
            "每个类别至少需要2个独立原始WAV。"
            f"不足类别：{insufficient.to_dict()}"
        )

    print("\n进入开放集脉冲模型的数据：")
    print(f"  训练脉冲总数：{len(data)}")
    print(
        f"  独立原始WAV数："
        f"{data['group_id'].nunique()}"
    )
    print("\n各类别独立WAV数：")
    print(class_group_counts)
    print("\n各类别训练脉冲数：")
    print(
        data["species"]
        .value_counts()
        .sort_index()
    )

    return data


def make_pulse_model(
    random_state: int,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=800,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )


def aggregate_open_set_recording_predictions(
    prediction_df: pd.DataFrame,
    classes: np.ndarray,
) -> pd.DataFrame:
    """
    把同一个原始WAV的所有候选脉冲汇总成录音结果。

    noise不参与具体物种排名：
    - 先根据 P(noise) 过滤噪声候选；
    - 再在目标物种和 unknown_bat 之间汇总概率。
    """
    rows: list[
        dict[str, float | int | str]
    ] = []

    probability_columns = [
        f"prob_{species}"
        for species in classes
    ]
    non_noise_classes = [
        str(species)
        for species in classes
        if str(species) != NOISE_CLASS
    ]

    for source_file, group in (
        prediction_df.groupby(
            "source_file",
            sort=True,
        )
    ):
        recording_labels = (
            group["recording_species"]
            .dropna()
            .astype(str)
            .unique()
        )
        if len(recording_labels) != 1:
            raise RuntimeError(
                f"{source_file} 的整段标签异常："
                f"{recording_labels}"
            )
        true_recording_label = str(
            recording_labels[0]
        )

        noise_probability = (
            group[f"prob_{NOISE_CLASS}"]
            .to_numpy(dtype=float)
        )
        usable_mask = (
            noise_probability
            < NOISE_PULSE_PROBABILITY_THRESHOLD
        )
        usable_group = group.loc[
            usable_mask
        ].copy()

        mean_noise_probability = float(
            np.mean(noise_probability)
        )

        if usable_group.empty:
            row = {
                "source_file": source_file,
                "species": true_recording_label,
                "recording_prediction": NOISE_CLASS,
                "candidate_pulse_count": int(
                    len(group)
                ),
                "usable_bat_pulse_count": 0,
                "mean_noise_probability": (
                    mean_noise_probability
                ),
                "mean_top_probability": 0.0,
                "probability_margin": 0.0,
                "pulse_vote_agreement": 0.0,
            }
            rows.append(row)
            continue

        non_noise_probability_matrix = (
            usable_group[
                [
                    f"prob_{species}"
                    for species
                    in non_noise_classes
                ]
            ]
            .to_numpy(dtype=float)
        )

        row_sums = (
            non_noise_probability_matrix.sum(
                axis=1,
                keepdims=True,
            )
        )
        row_sums[row_sums <= 0] = 1.0
        normalized_non_noise = (
            non_noise_probability_matrix
            / row_sums
        )
        mean_probabilities = (
            normalized_non_noise.mean(axis=0)
        )

        order = np.argsort(
            mean_probabilities
        )[::-1]
        best_index = int(order[0])
        second_index = (
            int(order[1])
            if len(order) > 1
            else best_index
        )
        predicted_species = (
            non_noise_classes[best_index]
        )

        usable_predictions = (
            usable_group["pulse_prediction"]
            .astype(str)
        )
        agreement = float(
            np.mean(
                usable_predictions.to_numpy()
                == predicted_species
            )
        )

        row = {
            "source_file": source_file,
            "species": true_recording_label,
            "recording_prediction": (
                predicted_species
            ),
            "candidate_pulse_count": int(
                len(group)
            ),
            "usable_bat_pulse_count": int(
                len(usable_group)
            ),
            "mean_noise_probability": (
                mean_noise_probability
            ),
            "mean_top_probability": float(
                mean_probabilities[best_index]
            ),
            "probability_margin": float(
                mean_probabilities[best_index]
                - mean_probabilities[second_index]
            ),
            "pulse_vote_agreement": agreement,
        }

        for species, probability in zip(
            non_noise_classes,
            mean_probabilities,
        ):
            row[
                f"mean_prob_{species}"
            ] = float(probability)

        rows.append(row)

    return pd.DataFrame(rows)


def train_pulse_model(
    pulse_df: pd.DataFrame,
) -> tuple[
    RandomForestClassifier,
    list[str],
]:
    """
    训练五类开放集模型：

    三个目标物种 + unknown_bat + noise。
    """
    data = prepare_pulse_training_data(
        pulse_df
    )

    X = data[PULSE_FEATURE_COLUMNS].copy()
    y = data["species"].astype(str)
    groups = data["group_id"].astype(str)
    classes = np.array(
        sorted(y.unique()),
        dtype=object,
    )

    group_class_table = (
        data[
            [
                "group_id",
                "species",
            ]
        ]
        .drop_duplicates()
    )
    min_groups_per_class = int(
        group_class_table["species"]
        .value_counts()
        .min()
    )
    n_splits = min(
        MAX_GROUP_FOLDS,
        min_groups_per_class,
    )

    if n_splits < 2:
        raise RuntimeError(
            "独立原始WAV数量不足，"
            "无法进行分组交叉验证。"
        )

    print(
        f"\n开始 {n_splits} 折开放集分组交叉验证。"
    )
    print(
        "同一个原始WAV中的蝙蝠脉冲与noise片段"
        "会始终进入同一折。"
    )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    out_of_fold_probabilities = np.full(
        (len(data), len(classes)),
        np.nan,
        dtype=float,
    )
    out_of_fold_predictions = np.empty(
        len(data),
        dtype=object,
    )
    fold_ids = np.zeros(
        len(data),
        dtype=int,
    )

    for fold_id, (
        train_indices,
        test_indices,
    ) in enumerate(
        splitter.split(
            X,
            y,
            groups,
        ),
        start=1,
    ):
        train_groups = set(
            groups.iloc[train_indices]
        )
        test_groups = set(
            groups.iloc[test_indices]
        )

        overlap = train_groups & test_groups
        if overlap:
            raise RuntimeError(
                f"第{fold_id}折发生分组泄漏："
                f"{overlap}"
            )

        model = make_pulse_model(
            RANDOM_STATE + fold_id
        )
        model.fit(
            X.iloc[train_indices],
            y.iloc[train_indices],
        )

        fold_probabilities = (
            model.predict_proba(
                X.iloc[test_indices]
            )
        )
        fold_predictions = model.predict(
            X.iloc[test_indices]
        )

        for local_index, species in enumerate(
            model.classes_
        ):
            global_index = int(
                np.where(
                    classes == species
                )[0][0]
            )
            out_of_fold_probabilities[
                test_indices,
                global_index,
            ] = fold_probabilities[
                :,
                local_index,
            ]

        out_of_fold_predictions[
            test_indices
        ] = fold_predictions
        fold_ids[test_indices] = fold_id

        print(
            f"  第{fold_id}折："
            f"训练WAV {len(train_groups)}个，"
            f"测试WAV {len(test_groups)}个，"
            f"测试脉冲 {len(test_indices)}个"
        )

    if np.isnan(
        out_of_fold_probabilities
    ).any():
        raise RuntimeError(
            "交叉验证概率存在缺失。"
            "某一折可能缺少某个类别。"
        )

    pulse_prediction_df = data[
        [
            "source_file",
            "group_id",
            "original_species",
            "recording_species",
            "species",
            "pulse_id",
            "pulse_file",
            "pulse_source",
        ]
    ].copy()
    pulse_prediction_df["fold"] = fold_ids
    pulse_prediction_df[
        "pulse_prediction"
    ] = out_of_fold_predictions

    for class_index, species in enumerate(
        classes
    ):
        pulse_prediction_df[
            f"prob_{species}"
        ] = out_of_fold_probabilities[
            :,
            class_index,
        ]

    pulse_prediction_df.to_csv(
        OPEN_SET_PULSE_CV_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    pulse_accuracy = accuracy_score(
        y,
        out_of_fold_predictions,
    )
    print(
        f"\n开放集脉冲级准确率："
        f"{pulse_accuracy:.2%}"
    )
    print("\n开放集脉冲级混淆矩阵：")
    print(
        confusion_matrix(
            y,
            out_of_fold_predictions,
            labels=classes,
        )
    )
    print("\n开放集脉冲级分类报告：")
    print(
        classification_report(
            y,
            out_of_fold_predictions,
            labels=classes,
            zero_division=0,
        )
    )

    recording_prediction_df = (
        aggregate_open_set_recording_predictions(
            pulse_prediction_df,
            classes,
        )
    )
    recording_prediction_df.to_csv(
        OPEN_SET_RECORDING_CV_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    recording_accuracy = accuracy_score(
        recording_prediction_df["species"],
        recording_prediction_df[
            "recording_prediction"
        ],
    )
    print(
        f"\n开放集整段录音级准确率："
        f"{recording_accuracy:.2%}"
    )
    print("\n开放集整段录音级混淆矩阵：")
    recording_classes = np.array(
        [
            *TARGET_SPECIES,
            UNKNOWN_BAT_CLASS,
            NOISE_CLASS,
        ],
        dtype=object,
    )
    print(
        confusion_matrix(
            recording_prediction_df["species"],
            recording_prediction_df[
                "recording_prediction"
            ],
            labels=recording_classes,
        )
    )
    print("\n整段录音级结果：")
    display_columns = [
        "source_file",
        "species",
        "recording_prediction",
        "candidate_pulse_count",
        "usable_bat_pulse_count",
        "mean_noise_probability",
        "mean_top_probability",
        "pulse_vote_agreement",
    ]
    print(
        recording_prediction_df[
            display_columns
        ].to_string(index=False)
    )

    final_model = make_pulse_model(
        RANDOM_STATE
    )
    final_model.fit(X, y)

    payload = {
        "model": final_model,
        "feature_columns": (
            PULSE_FEATURE_COLUMNS
        ),
        "target_species": TARGET_SPECIES,
        "unknown_class": UNKNOWN_BAT_CLASS,
        "noise_class": NOISE_CLASS,
        "target_sample_rate": (
            TARGET_SAMPLE_RATE
        ),
        "low_freq_hz": LOW_FREQ_HZ,
        "high_freq_hz": HIGH_FREQ_HZ,
        "group_field": "group_id",
        "training_pulse_count": int(
            len(data)
        ),
        "training_recording_count": int(
            data["group_id"].nunique()
        ),
    }
    joblib.dump(
        payload,
        OPEN_SET_MODEL_PATH,
    )

    print(
        f"\n开放集模型已保存："
        f"{OPEN_SET_MODEL_PATH}"
    )
    print(
        f"开放集脉冲CV："
        f"{OPEN_SET_PULSE_CV_CSV}"
    )
    print(
        f"开放集录音CV："
        f"{OPEN_SET_RECORDING_CV_CSV}"
    )

    importance = pd.Series(
        final_model.feature_importances_,
        index=PULSE_FEATURE_COLUMNS,
    ).sort_values(ascending=False)
    print("\n开放集模型特征重要性：")
    print(importance)

    return final_model, list(
        PULSE_FEATURE_COLUMNS
    )


def summarize_new_recording(
    candidate_df: pd.DataFrame,
    model: RandomForestClassifier,
    feature_columns: list[str],
) -> dict[str, object]:
    """
    对新录音的候选脉冲进行开放集预测和整段汇总。
    """
    for column in feature_columns:
        candidate_df[column] = pd.to_numeric(
            candidate_df[column],
            errors="coerce",
        )

    valid_mask = (
        candidate_df[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .notna()
        .all(axis=1)
    )
    candidate_df = candidate_df.loc[
        valid_mask
    ].copy()

    if candidate_df.empty:
        return {
            "status": "no_valid_candidates",
            "candidate_df": candidate_df,
        }

    probabilities = model.predict_proba(
        candidate_df[feature_columns]
    )
    predictions = model.predict(
        candidate_df[feature_columns]
    )

    candidate_df[
        "pulse_prediction"
    ] = predictions

    for class_index, species in enumerate(
        model.classes_
    ):
        candidate_df[
            f"prob_{species}"
        ] = probabilities[
            :,
            class_index,
        ]

    noise_index_values = np.where(
        model.classes_ == NOISE_CLASS
    )[0]
    if len(noise_index_values) != 1:
        raise RuntimeError(
            "模型中没有唯一的 noise 类别。"
        )
    noise_index = int(
        noise_index_values[0]
    )
    noise_probabilities = probabilities[
        :,
        noise_index,
    ]

    usable_mask = (
        noise_probabilities
        < NOISE_PULSE_PROBABILITY_THRESHOLD
    )
    usable_df = candidate_df.loc[
        usable_mask
    ].copy()

    mean_noise_probability = float(
        np.mean(noise_probabilities)
    )

    if usable_df.empty:
        return {
            "status": "noise_only",
            "candidate_df": candidate_df,
            "usable_df": usable_df,
            "mean_noise_probability": (
                mean_noise_probability
            ),
        }

    non_noise_classes = [
        str(species)
        for species in model.classes_
        if str(species) != NOISE_CLASS
    ]
    non_noise_probability_matrix = (
        usable_df[
            [
                f"prob_{species}"
                for species
                in non_noise_classes
            ]
        ]
        .to_numpy(dtype=float)
    )

    row_sums = (
        non_noise_probability_matrix.sum(
            axis=1,
            keepdims=True,
        )
    )
    row_sums[row_sums <= 0] = 1.0
    normalized = (
        non_noise_probability_matrix
        / row_sums
    )
    mean_probabilities = normalized.mean(
        axis=0
    )

    order = np.argsort(
        mean_probabilities
    )[::-1]
    best_index = int(order[0])
    predicted_species = (
        non_noise_classes[best_index]
    )
    agreement = float(
        np.mean(
            usable_df[
                "pulse_prediction"
            ].astype(str).to_numpy()
            == predicted_species
        )
    )

    ranking = [
        (
            non_noise_classes[
                int(class_index)
            ],
            float(
                mean_probabilities[
                    int(class_index)
                ]
            ),
        )
        for class_index in order
    ]

    return {
        "status": "success",
        "candidate_df": candidate_df,
        "usable_df": usable_df,
        "mean_noise_probability": (
            mean_noise_probability
        ),
        "predicted_species": predicted_species,
        "agreement": agreement,
        "ranking": ranking,
    }


def predict_file(
    audio_path: Path,
    model: RandomForestClassifier,
    feature_columns: list[str],
) -> None:
    """
    新WAV预测：

    候选脉冲全部进入开放集模型，
    noise由随机森林作为第二层过滤。
    """
    y, sr = load_audio(audio_path)

    (
        raw_pulses,
        power,
        frequencies,
        times,
        _,
    ) = detect_pulses(y, sr)

    refined_pulses = refine_all_pulses(
        y,
        sr,
        raw_pulses,
    )

    base_rows = create_pulse_rows(
        audio_path,
        "prediction_input",
        y,
        sr,
        refined_pulses,
    )

    if not base_rows:
        print(
            "\n没有检测到可提取特征的候选脉冲。"
        )
        return

    candidate_df = pd.DataFrame(base_rows)
    result = summarize_new_recording(
        candidate_df,
        model,
        feature_columns,
    )

    save_annotated_spectrogram(
        audio_path,
        power,
        frequencies,
        times,
        refined_pulses,
    )
    save_top_pulse_zoom_grid(
        audio_path,
        y,
        sr,
        base_rows,
    )

    print(f"\n待预测文件：{audio_path.name}")
    print(
        f"原始候选脉冲："
        f"{len(raw_pulses)} 个"
    )
    print(
        f"精修后候选脉冲："
        f"{len(refined_pulses)} 个"
    )

    status = str(result["status"])
    output_df = result["candidate_df"]

    if status == "no_valid_candidates":
        print(
            "没有具有完整数值特征的候选脉冲。"
        )
        return

    prediction_output_path = (
        PREDICTION_DIR
        / (
            f"{audio_path.stem}"
            "_open_set_predictions.csv"
        )
    )
    output_df.to_csv(
        prediction_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    if status == "noise_only":
        print(
            "结论：候选片段主要被判断为 noise，"
            "当前没有足够证据确认蝙蝠。"
        )
        print(
            f"平均 noise 概率："
            f"{result['mean_noise_probability']:.2%}"
        )
        print(
            f"逐脉冲结果："
            f"{prediction_output_path}"
        )
        return

    usable_df = result["usable_df"]
    ranking = result["ranking"]
    predicted_species = str(
        result["predicted_species"]
    )
    agreement = float(result["agreement"])
    mean_noise_probability = float(
        result["mean_noise_probability"]
    )

    print(
        f"模型认为不是noise的候选："
        f"{len(usable_df)} 个"
    )
    print(
        f"全部候选平均noise概率："
        f"{mean_noise_probability:.2%}"
    )
    print("\n非noise候选类别：")

    for species, probability in ranking[:4]:
        print(
            f"  {species:<15} "
            f"{probability:.2%}"
        )

    print(
        f"\n脉冲投票一致率："
        f"{agreement:.2%}"
    )

    needs_review = (
        len(usable_df)
        < MIN_NON_NOISE_PULSES_FOR_PREDICTION
        or mean_noise_probability
        > MAX_RECORDING_MEAN_NOISE_PROBABILITY
        or ranking[0][1]
        < RECORDING_PROBABILITY_THRESHOLD
        or agreement
        < RECORDING_AGREEMENT_THRESHOLD
    )

    if predicted_species == UNKNOWN_BAT_CLASS:
        print(
            "结论：检测到更像蝙蝠的脉冲，"
            "但不符合当前目标物种，"
            "暂定为 unknown_bat。"
        )
    elif needs_review:
        print(
            "结论：候选物种结果不够稳定，"
            "需要人工复核。"
        )
    else:
        print(
            f"结论：当前开放集模型判断为 "
            f"{predicted_species}。"
        )

    print(
        f"逐脉冲结果："
        f"{prediction_output_path}"
    )


def main() -> None:
    prepare_dataset()
    _, pulse_df = build_feature_tables()
    model, feature_columns = train_pulse_model(
        pulse_df
    )

    raw_path = input(
        "\n输入要预测的WAV路径，"
        "直接回车可跳过："
    ).strip().strip('"').lstrip("\ufeff").strip()

    if not raw_path:
        return

    audio_path = Path(raw_path)
    if not audio_path.is_absolute():
        audio_path = BASE_DIR / audio_path

    if not audio_path.exists():
        print(f"找不到文件：{audio_path}")
        return

    predict_file(
        audio_path,
        model,
        feature_columns,
    )



# =========================================================
# Step6：无监督声型聚类 A/B/C/D + 临时分类器
# =========================================================
#
# 本阶段目标不是声称识别出真实物种，而是先证明：
# 原始WAV -> 自动切脉冲 -> 提取特征 -> 聚类为声型A-D
# -> 训练临时分类器 -> 新录音自动归类。
#
# 以后拿到老师的可靠物种数据库后，只需用真实物种标签
# 替换 acoustic_type 标签，不必重写前面的脉冲处理流程。
# =========================================================

import json
import shutil

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Step6 独立输入、输出位置
# -----------------------------
DEMO_ZIP_CANDIDATES = [
    BASE_DIR / "声波文件(1).zip",
    BASE_DIR / "声波文件.zip",
]
DEMO_DATA_DIR = BASE_DIR / "acoustic_demo_data"
DEMO_OUTPUT_DIR = BASE_DIR / "acoustic_demo_output"

DEMO_PULSE_CSV = DEMO_OUTPUT_DIR / "acoustic_pulses.csv"
DEMO_RECORDING_CSV = (
    DEMO_OUTPUT_DIR / "recording_detection_summary.csv"
)
CLUSTERED_PULSE_CSV = (
    DEMO_OUTPUT_DIR / "acoustic_clusters_ABCD.csv"
)
RECORDING_CLUSTER_CSV = (
    DEMO_OUTPUT_DIR / "recording_cluster_summary.csv"
)
CLUSTER_CENTER_CSV = (
    DEMO_OUTPUT_DIR / "cluster_centers_ABCD.csv"
)
CLUSTER_REPRESENTATIVE_CSV = (
    DEMO_OUTPUT_DIR / "cluster_representatives.csv"
)
CLUSTER_PCA_PNG = (
    DEMO_OUTPUT_DIR / "cluster_pca_ABCD.png"
)
CLUSTER_MODEL_PATH = (
    DEMO_OUTPUT_DIR / "acoustic_type_ABCD_model.joblib"
)
CLUSTER_METADATA_JSON = (
    DEMO_OUTPUT_DIR / "acoustic_type_ABCD_metadata.json"
)
CLUSTER_EXAMPLE_DIR = (
    DEMO_OUTPUT_DIR / "cluster_examples"
)
DEMO_PREDICTION_DIR = (
    DEMO_OUTPUT_DIR / "predictions"
)

# 覆盖前面步骤使用的目录，让切片和图片统一进入演示输出目录
DATA_DIR = DEMO_DATA_DIR
SPECTROGRAM_DIR = DEMO_OUTPUT_DIR / "spectrograms"
PULSE_ZOOM_DIR = DEMO_OUTPUT_DIR / "pulse_zooms"
CUT_PULSE_DIR = DEMO_OUTPUT_DIR / "cut_pulses"
REJECTED_PULSE_DIR = DEMO_OUTPUT_DIR / "rejected_pulses"
PREDICTION_DIR = DEMO_PREDICTION_DIR

# 聚类使用的特征。避免使用RMS和SNR作为分类特征，
# 因为它们很容易受到设备、距离和录音增益影响。
CLUSTER_FEATURE_COLUMNS = [
    "duration_ms",
    "peak_freq_khz",
    "start_freq_khz",
    "end_freq_khz",
    "frequency_drop_khz",
    "f05_khz",
    "f95_khz",
    "bandwidth_90_khz",
    "slope_khz_per_ms",
    "track_coverage",
    "max_track_jump_khz",
]

N_ACOUSTIC_TYPES = 4
MAX_CLUSTER_PULSES_PER_RECORDING = 150
REPRESENTATIVE_PULSES_PER_TYPE = 10
MIN_PULSES_FOR_CLUSTERING = 40

ACOUSTIC_TYPE_DESCRIPTIONS = {
    "A": "短促宽带下降FM型",
    "B": "高频窄带或CF型",
    "C": "低频窄带或缓变型",
    "D": "复杂、中间或混合声型",
}

# 长时间野外录音按块检测，避免一次构建数百MB声谱图
DETECTION_CHUNK_SECONDS = 4.0
DETECTION_CHUNK_OVERLAP_MS = 50.0
MAX_ANALYSIS_WINDOWS_PER_RECORDING = 2
SAVE_PER_RECORDING_DIAGNOSTICS = False
DEMO_DETECTION_HOP_LENGTH = 256
OVERVIEW_MAX_TIME_BINS = 4000
MAX_REFINED_PULSES_PER_RECORDING = 350

# 同时发声分离参数。
#
# 原检测器只沿时间检测候选，随后每帧只取一个最强频率，
# 因而同一时刻的两条近频声波会被合成一个脉冲。这里在候选
# 内追踪多条局部频谱峰，并用维纳式软掩膜重建各条声波。
# 单通道录音中完全同频的两个源在物理上不可辨识；下面的
# MIN_FREQ_GAP 是当前250 kHz采样率和短时窗下的保守分辨下限。
ENABLE_SIMULTANEOUS_SEPARATION = True
SEPARATION_N_FFT = 1024
SEPARATION_WIN_LENGTH = 256
SEPARATION_HOP_LENGTH = 32
SEPARATION_MAX_COMPONENTS = 3
SEPARATION_MAX_PEAKS_PER_FRAME = 5
SEPARATION_MIN_FREQ_GAP_KHZ = 1.5
SEPARATION_PEAK_PROMINENCE_DB = 2.5
SEPARATION_FRAME_DYNAMIC_RANGE_DB = 22.0
SEPARATION_GLOBAL_MIN_DB = -38.0
SEPARATION_MAX_TRACK_JUMP_KHZ = 5.0
SEPARATION_MAX_TRACK_GAP_FRAMES = 2
SEPARATION_MIN_TRACK_MS = 0.55
SEPARATION_MIN_TRACK_COVERAGE = 0.50
SEPARATION_MIN_TEMPORAL_OVERLAP = 0.45
SEPARATION_MIN_SECONDARY_RELATIVE_DB = -18.0
SEPARATION_MASK_BANDWIDTH_KHZ = 1.2
SEPARATION_HARMONIC_TOLERANCE = 0.06
SAVE_SEPARATED_PULSE_AUDIO = True
SEPARATED_PULSE_DIR = (
    DEMO_OUTPUT_DIR / "separated_pulses"
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
            "请把压缩包放到脚本同目录，"
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


def predict_one_demo_wav(
    audio_path: Path,
    model_payload: dict[str, object],
) -> dict[str, object]:
    """把一个新WAV自动归入声型A-D。"""
    scaler: StandardScaler = (
        model_payload["scaler"]
    )
    kmeans: KMeans = model_payload["kmeans"]
    temporary_model: RandomForestClassifier = (
        model_payload[
            "temporary_classifier"
        ]
    )
    mapping: dict[int, str] = (
        model_payload[
            "raw_cluster_to_type"
        ]
    )
    feature_columns: list[str] = list(
        model_payload["feature_columns"]
    )

    y, sr = load_audio(audio_path)
    (
        raw_candidate_count,
        refined_pulses,
        _,
    ) = detect_and_refine_pulses_chunked(
        y,
        sr,
    )
    rows = create_demo_pulse_rows(
        audio_path,
        y,
        sr,
        refined_pulses,
    )
    accepted_rows = [
        row
        for row in rows
        if bool(row["accepted"])
    ]
    split_candidate_count = len(
        {
            int(row["candidate_id"])
            for row in rows
            if bool(row["separation_applied"])
        }
    )

    print(
        f"\n预测：{audio_path.name}，"
        f"原始候选 {raw_candidate_count}，"
        f"拆分候选 {split_candidate_count}，"
        f"合格脉冲分量 {len(accepted_rows)}"
    )

    if not accepted_rows:
        return {
            "source_file": audio_path.name,
            "status": "no_accepted_pulses",
            "accepted_pulse_count": 0,
        }

    prediction_df = pd.DataFrame(
        accepted_rows
    )
    for column in feature_columns:
        prediction_df[column] = pd.to_numeric(
            prediction_df[column],
            errors="coerce",
        )

    valid_mask = (
        prediction_df[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .notna()
        .all(axis=1)
    )
    prediction_df = prediction_df.loc[
        valid_mask
    ].copy()

    if prediction_df.empty:
        return {
            "source_file": audio_path.name,
            "status": "invalid_features",
            "accepted_pulse_count": 0,
        }

    scaled = scaler.transform(
        prediction_df[feature_columns]
    )
    raw_cluster = kmeans.predict(scaled)
    kmeans_types = np.array(
        [
            mapping[int(value)]
            for value in raw_cluster
        ],
        dtype=object,
    )
    classifier_types = (
        temporary_model.predict(
            prediction_df[feature_columns]
        )
    )

    prediction_df[
        "kmeans_acoustic_type"
    ] = kmeans_types
    prediction_df[
        "predicted_acoustic_type"
    ] = classifier_types
    prediction_df[
        "model_agreement"
    ] = (
        prediction_df[
            "kmeans_acoustic_type"
        ]
        == prediction_df[
            "predicted_acoustic_type"
        ]
    )

    probabilities = (
        temporary_model.predict_proba(
            prediction_df[
                feature_columns
            ]
        )
    )
    for class_index, acoustic_type in enumerate(
        temporary_model.classes_
    ):
        prediction_df[
            f"prob_type_{acoustic_type}"
        ] = probabilities[
            :,
            class_index,
        ]

    counts = (
        prediction_df[
            "predicted_acoustic_type"
        ]
        .value_counts()
    )
    major_type = str(counts.index[0])
    major_ratio = float(
        counts.iloc[0] / len(prediction_df)
    )
    model_agreement = float(
        prediction_df[
            "model_agreement"
        ].mean()
    )

    output_path = (
        DEMO_PREDICTION_DIR
        / (
            f"{audio_path.stem}"
            "_ABCD_predictions.csv"
        )
    )
    prediction_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    accepted_pulses = [
        {
            "start_s": float(row["start_s"]),
            "end_s": float(row["end_s"]),
            "duration_ms": float(
                row["duration_ms"]
            ),
        }
        for row in accepted_rows
    ]
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

    print(
        f"主要声型：{major_type} "
        f"（{ACOUSTIC_TYPE_DESCRIPTIONS[major_type]}）"
    )
    print(
        f"主要声型占比：{major_ratio:.2%}"
    )
    print(
        f"KMeans与临时分类器一致率："
        f"{model_agreement:.2%}"
    )
    print("各声型脉冲数：")
    print(
        counts.reindex(
            ["A", "B", "C", "D"],
            fill_value=0,
        )
    )
    print(
        f"逐脉冲预测已保存：{output_path}"
    )

    result: dict[str, object] = {
        "source_file": audio_path.name,
        "status": "success",
        "accepted_pulse_count": int(
            len(prediction_df)
        ),
        "split_candidate_count": (
            split_candidate_count
        ),
        "major_acoustic_type": major_type,
        "major_type_description": (
            ACOUSTIC_TYPE_DESCRIPTIONS[
                major_type
            ]
        ),
        "major_type_ratio": major_ratio,
        "kmeans_classifier_agreement": (
            model_agreement
        ),
        "prediction_csv": str(output_path),
    }
    for acoustic_type in ["A", "B", "C", "D"]:
        result[
            f"type_{acoustic_type}_count"
        ] = int(counts.get(acoustic_type, 0))
        result[
            f"type_{acoustic_type}_ratio"
        ] = float(
            counts.get(acoustic_type, 0)
            / len(prediction_df)
        )

    return result


def predict_demo_path(
    input_path: Path,
    model_payload: dict[str, object],
) -> None:
    """支持输入单个WAV，也支持输入整个文件夹。"""
    if input_path.is_file():
        if input_path.suffix.lower() != ".wav":
            print(
                f"不是WAV文件：{input_path}"
            )
            return
        wav_files = [input_path]
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
            result = predict_one_demo_wav(
                audio_path,
                model_payload,
            )
            summary_rows.append(result)
        except Exception as error:
            print(
                f"预测失败 {audio_path.name}："
                f"{error}"
            )
            summary_rows.append(
                {
                    "source_file": (
                        audio_path.name
                    ),
                    "status": "failed",
                    "error": str(error),
                }
            )

    summary_df = pd.DataFrame(
        summary_rows
    )
    summary_path = (
        DEMO_PREDICTION_DIR
        / "prediction_summary.csv"
    )
    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    print(
        f"\n批量预测汇总：{summary_path}"
    )


def main() -> None:
    print(
        "========== 蝙蝠声型A-D原型系统 =========="
    )
    print(
        "A-D是自动聚类得到的临时声型，"
        "不是已经确认的物种名称。"
    )

    prepare_demo_dataset()
    pulse_df, detection_df = (
        build_demo_pulse_table()
    )
    cluster_result = cluster_acoustic_types(
        pulse_df,
        detection_df,
    )
    model_payload = cluster_result["payload"]

    raw_path = input(
        "\n输入新的WAV或文件夹路径进行A-D预测，"
        "直接回车可结束："
    ).strip().strip('"').lstrip("\ufeff").strip()

    if not raw_path:
        return

    input_path = Path(raw_path)
    if not input_path.is_absolute():
        input_path = BASE_DIR / input_path

    predict_demo_path(
        input_path,
        model_payload,
    )



# =========================================================
# Step6.1：第一层 noise / bat，第二层声型 A-D
# =========================================================
from sklearn.model_selection import GroupShuffleSplit

NOISE_GATE_MODEL_PATH = (
    DEMO_OUTPUT_DIR / "noise_bat_gate_model.joblib"
)
COMBINED_DEMO_MODEL_PATH = (
    DEMO_OUTPUT_DIR / "noise_bat_ABCD_model.joblib"
)
NOISE_GATE_PREDICTIONS_CSV = (
    DEMO_OUTPUT_DIR / "noise_gate_validation.csv"
)

NOISE_GATE_FEATURE_COLUMNS = [
    "duration_ms",
    "peak_freq_khz",
    "start_freq_khz",
    "end_freq_khz",
    "frequency_drop_khz",
    "f05_khz",
    "f95_khz",
    "bandwidth_90_khz",
    "slope_khz_per_ms",
    "snr_db",
    "track_coverage",
    "max_track_jump_khz",
]

MAX_NOISE_GATE_ROWS_PER_SOURCE_CLASS = 120
MIN_NOISE_GATE_ROWS_PER_CLASS = 20
NOISE_BAT_PROBABILITY_THRESHOLD = 0.55


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


if __name__ == "__main__":
    main()
