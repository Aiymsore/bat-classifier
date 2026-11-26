"""候选脉冲检测与边界精修（上游：audio_io；下游：pulse_table、prediction）。

三节内容：
1. 粗检：按超声频段能量阈值找出候选脉冲（detect_pulses）；
2. 精修：在候选区内重找高信噪比脉冲核心（refine_pulse_boundary 等）；
长录音的连续分块与跨块去重由 segment_stream.py 负责。
"""

from __future__ import annotations

import librosa
import numpy as np
from scipy.signal import find_peaks

from .audio_io import (
    adaptive_stft_params,
    build_spectrogram,
    interval_overlap_ratio,
    ultrasonic_band_mask,
)
from .config import (
    BOUNDARY_EXTEND_BAND_ABOVE_KHZ,
    BOUNDARY_EXTEND_BAND_BELOW_KHZ,
    BOUNDARY_EXTEND_CONTEXT_MS,
    BOUNDARY_BACKGROUND_CONTEXT_MS,
    BOUNDARY_TRACK_STEP_KHZ,
    BOUNDARY_EXTEND_MARGIN_DB,
    BOUNDARY_EXTEND_MAX_GAP_MS,
    CORE_BACKGROUND_MARGIN_DB,
    CORE_CONTEXT_MS,
    CORE_MAX_GAP_MS,
    CORE_MAX_THRESHOLD_DB,
    CORE_PADDING_MS,
    CORE_THRESHOLD_DB,
    DETECTION_ADAPTIVE_MARGIN_DB,
    FRAGMENT_EDGE_MAX_SHIFT_KHZ,
    FRAGMENT_EDGE_MS,
    FRAGMENT_MERGE_FREQ_KHZ,
    FRAGMENT_MERGE_GAP_MS,
    FRAGMENT_PEAK_COUNT,
    FRAGMENT_PEAK_PROMINENCE_DB,
    DETECTION_DIAGNOSTIC_BANDS_KHZ,
    ENERGY_THRESHOLD_DB,
    HIGH_FREQ_HZ,
    LOW_FREQ_HZ,
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


def _detect_from_band_power(
    band_power: np.ndarray,
    times: np.ndarray,
    hop_length: int,
    sr: int,
    audio_duration_s: float,
    max_duration_ms: float,
) -> tuple[list[dict[str, float]], float]:
    """对一条频带能量包络执行同一套判据：相对本带最大值 + 本带背景自适应阈值。"""
    energy_db = librosa.power_to_db(
        band_power + 1e-18,
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
    adaptive_threshold = background_db + DETECTION_ADAPTIVE_MARGIN_DB
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

        if not (MIN_PULSE_MS <= duration_ms <= max_duration_ms):
            continue

        pulses.append(
            {
                "start_s": start_s,
                "end_s": end_s,
                "duration_ms": duration_ms,
            }
        )

    return pulses, threshold_db


def normalize_detection_bands(
    bands_khz,
    sr: int,
) -> list[tuple[float, float]]:
    """清洗模板子带：限制在超声分析频段内，丢弃无效或过窄的频带。"""
    cleaned: list[tuple[float, float]] = []
    upper_khz = min(HIGH_FREQ_HZ, sr / 2) / 1000
    lower_khz = LOW_FREQ_HZ / 1000
    for band in bands_khz or ():
        low, high = float(band[0]), float(band[1])
        if not (np.isfinite(low) and np.isfinite(high)):
            continue
        low, high = max(low, lower_khz), min(high, upper_khz)
        if high - low >= 1.0:
            cleaned.append((round(low, 3), round(high, 3)))
    return cleaned


def band_label(band_khz) -> str:
    """检测来源标签：fullband 或 band:<low>-<high>（kHz）。"""
    if not band_khz:
        return "fullband"
    return f"band:{float(band_khz[0]):g}-{float(band_khz[1]):g}"


def detect_pulses(
    y: np.ndarray,
    sr: int,
    *,
    max_duration_ms: float = MAX_PULSE_MS,
    bands_khz=None,
) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray, np.ndarray, float]:
    """
    从声谱图能量中检测疑似脉冲。

    默认只在 15–120 kHz 全带求和检测（原行为）。传入 ``bands_khz``
    （模板频带列表，单位 kHz）时，另在每个子带内独立求和、按本带背景
    设阈值再检测一次：窄带叫声不再被其他频段的宽带底噪稀释。
    子带候选带有 ``detect_band_khz``，精修也只在该子带内计算能量。

    返回：
    - 脉冲时间范围（全带候选在前，子带候选在后，未去重）
    - 功率谱
    - 频率轴
    - 时间轴
    - 全带检测阈值
    """
    power, frequencies, times, hop_length = build_spectrogram(y, sr)
    audio_duration_s = len(y) / sr

    frequency_mask = ultrasonic_band_mask(frequencies, sr)
    ultrasonic_power = np.sum(power[frequency_mask, :], axis=0)
    pulses, threshold_db = _detect_from_band_power(
        ultrasonic_power,
        times,
        hop_length,
        sr,
        audio_duration_s,
        max_duration_ms,
    )

    for band in normalize_detection_bands(bands_khz, sr):
        band_mask = (
            (frequencies >= band[0] * 1000)
            & (frequencies <= band[1] * 1000)
        )
        if not np.any(band_mask):
            continue
        band_pulses, _band_threshold = _detect_from_band_power(
            np.sum(power[band_mask, :], axis=0),
            times,
            hop_length,
            sr,
            audio_duration_s,
            max_duration_ms,
        )
        for pulse in band_pulses:
            pulse["detect_band_khz"] = band
        pulses.extend(band_pulses)

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


def salient_peak_freqs_khz(
    power: np.ndarray,
    frequencies: np.ndarray,
    frequency_mask: np.ndarray,
    start_frame: int,
    end_frame: int,
) -> list[float]:
    """核心段平均谱中最显著的窄带峰频率（kHz），按峰高降序，最多 FRAGMENT_PEAK_COUNT 个。"""
    spectrum = np.mean(power[:, start_frame:end_frame + 1], axis=1)
    freqs = frequencies[frequency_mask]
    db = 10 * np.log10(spectrum[frequency_mask] + 1e-30)
    if len(db) < 5:
        return []
    bin_hz = float(np.median(np.diff(freqs)))
    half = max(2, int(round(5000.0 / bin_hz)))
    padded = np.pad(db, half, mode="edge")
    local_median = np.array([np.median(padded[i:i + 2 * half + 1]) for i in range(len(db))])
    peaks, _ = find_peaks(db)
    peaks = [int(i) for i in peaks if db[i] - local_median[i] >= FRAGMENT_PEAK_PROMINENCE_DB]
    peaks.sort(key=lambda i: -db[i])
    return [float(freqs[i]) / 1000.0 for i in peaks[:FRAGMENT_PEAK_COUNT]]


def extend_pulse_boundary(
    y: np.ndarray,
    sr: int,
    start_sample: int,
    end_sample: int,
    peak_freq_hz: float,
    *,
    max_duration_ms: float = MAX_PULSE_MS,
    report_edges: bool = False,
):
    """滞回延伸：在核心峰值频率附近的频带内，从核心两端向外延伸。

    ``report_edges=True`` 时额外返回 (起点是否延伸到窗口边缘, 终点是否延伸到窗口边缘)，
    调用方只在碰到窗口边缘时才需要再看更远处。

    一次读取较宽上下文，在核心之外估计每个频率格点的10分位背景；
    跟随相邻帧的局部谱峰，要求高于固定背景和邻频背景各6dB。
    允许不超过 ``BOUNDARY_EXTEND_MAX_GAP_MS`` 的短缺口；不重估噪声来反复蹭进背景。
    """
    # A single fixed context supplies the noise reference for the whole walk.
    # Do not repeatedly estimate "background" inside a short CF plateau.
    context = int(round(max(BOUNDARY_BACKGROUND_CONTEXT_MS, max_duration_ms) / 1000 * sr))
    window_start = max(0, start_sample - context)
    window_end = min(len(y), end_sample + context)
    segment = y[window_start:window_end]
    hop = 32
    n_fft = 512
    unchanged = (start_sample, end_sample, False, False) if report_edges else (start_sample, end_sample)
    if len(segment) < n_fft or not np.isfinite(peak_freq_hz) or peak_freq_hz <= 0:
        return unchanged
    power = np.abs(librosa.stft(segment, n_fft=n_fft, hop_length=hop, window="hann", center=True)) ** 2
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band = (
        (frequencies >= peak_freq_hz - BOUNDARY_EXTEND_BAND_BELOW_KHZ * 1000)
        & (frequencies <= peak_freq_hz + BOUNDARY_EXTEND_BAND_ABOVE_KHZ * 1000)
        & ultrasonic_band_mask(frequencies, sr)
    )
    if not np.any(band):
        return unchanged
    n_frames = power.shape[1]
    frame_samples = window_start + np.arange(n_frames) * hop
    external = ((frame_samples < start_sample - int(.002 * sr))
                | (frame_samples > end_sample + int(.002 * sr)))
    # Ignore reflected STFT edge frames when estimating quiet background.
    external &= (frame_samples >= window_start+n_fft//2) & (frame_samples < window_end-n_fft//2)
    # Frequency-specific background: a broad-band maximum is dominated by
    # unrelated noise bins and can erase the weak continuation of a CF ridge.
    reference_power = power[:, external] if np.any(external) else power
    bin_threshold = np.percentile(reference_power, 10, axis=1) * 10**(BOUNDARY_EXTEND_MARGIN_DB/10)
    frame_of = lambda sample: int(np.clip(round((sample - window_start) / hop), 0, n_frames - 1))
    first, last = frame_of(start_sample), frame_of(end_sample)
    max_gap = max(1, int(round(BOUNDARY_EXTEND_MAX_GAP_MS / 1000 * sr / hop)))
    max_frames = int(max_duration_ms / 1000 * sr / hop)

    def walk(origin: int, step: int, limit: int) -> tuple[int, bool]:
        """返回 (最远的高于阈值帧, 是否一直延伸到窗口边缘而未遇到足够长的凹陷)。"""
        best, gap, index = origin, 0, origin + step
        previous_hz = peak_freq_hz
        while 0 <= index < n_frames and abs(index - origin) <= limit:
            search = band & (np.abs(frequencies-previous_hz) <= max(BOUNDARY_TRACK_STEP_KHZ*1000, 3*sr/n_fft))
            ids = np.flatnonzero(search)
            chosen = ids[np.argmax(power[ids,index])] if len(ids) else None
            supported = False
            if chosen is not None:
                value = power[chosen,index]
                local_peak = ((chosen == 0 or value >= power[chosen-1,index])
                              and (chosen == len(frequencies)-1 or value >= power[chosen+1,index]))
                distance = np.abs(frequencies-frequencies[chosen])
                neighbors = (distance >= 1500) & (distance <= 6000)
                floor = float(np.median(power[neighbors,index])) if np.any(neighbors) else value
                supported = (local_peak and value >= bin_threshold[chosen]
                             and value >= max(floor,1e-30)*10**(BOUNDARY_EXTEND_MARGIN_DB/10))
            if supported:
                best, gap = index, 0
                previous_hz = float(frequencies[chosen])
            else:
                gap += 1
                if gap > max_gap:
                    return best, False
            index += step
        at_window_edge = not (0 <= index < n_frames) and gap == 0
        return best, at_window_edge

    budget = max(0, max_frames - (last - first))
    new_last, end_edge = walk(last, 1, budget)
    new_first, start_edge = walk(first, -1, max(0, budget - (new_last - last)))
    new_start = window_start + new_first * hop
    new_end = window_start + new_last * hop
    start_out = min(start_sample, max(0, new_start))
    end_out = max(end_sample, min(len(y), new_end))
    # 录音（块）本身的两端不算“窗口边缘”：再往外已经没有音频。
    start_edge = start_edge and window_start > 0
    end_edge = end_edge and window_end < len(y)
    return (start_out, end_out, start_edge, end_edge) if report_edges else (start_out, end_out)


def refine_pulse_cores(
    y: np.ndarray,
    sr: int,
    pulse: dict[str, float],
    *,
    max_duration_ms: float = MAX_PULSE_MS,
    all_groups: bool = True,
) -> list[dict[str, float]]:
    """
    在候选区内部重新寻找高信噪比的脉冲核心，每个核心各自做滞回延伸。

    去掉候选窗口前后的背景拖尾。一个候选里可能有不止一条叫声（检测的平滑会跨过
    几毫秒的间隙），因此默认返回全部核心组；是否是同一叫声的碎片由
    ``merge_adjacent_fragments`` 判断。``all_groups=False`` 只保留包含最强帧的核心。
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
        return []

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

    detect_band = pulse.get("detect_band_khz")
    if detect_band:
        # 子带候选：只在该子带内找核心，避免其他频段的强声/宽带噪声主导边界。
        frequency_mask = (
            (frequencies >= float(detect_band[0]) * 1000)
            & (frequencies <= float(detect_band[1]) * 1000)
        )
        if not np.any(frequency_mask):
            frequency_mask = ultrasonic_band_mask(frequencies, sr)
    else:
        frequency_mask = ultrasonic_band_mask(frequencies, sr)
    band_power = np.sum(power[frequency_mask, :], axis=0)

    if band_power.size == 0 or float(np.max(band_power)) <= 0:
        return []

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
    if not groups:
        return []
    if not all_groups:
        selected = select_group_containing_peak(groups, int(np.argmax(band_power)))
        groups = [selected] if selected is not None else []

    # 噪声电平取所有核心组之外的帧；全部是核心时退回背景分位数。
    inside = np.zeros(len(energy_db), dtype=bool)
    for first, last in merge_active_frames(active_mask, max_gap_frames):
        inside[first:last + 1] = True
    noise_level_db = float(np.median(energy_db[~inside])) if np.any(~inside) else background_db
    padding_samples = int(round(CORE_PADDING_MS / 1000 * sr))

    cores: list[dict[str, float]] = []
    for start_frame, end_frame in groups:
        peak_frame = start_frame + int(np.argmax(band_power[start_frame:end_frame + 1]))
        core_start_sample = analysis_start_sample + max(0, start_frame * hop_length - n_fft // 2)
        core_end_sample = analysis_start_sample + min(len(candidate_y), end_frame * hop_length + n_fft // 2)
        seed_start_sample, seed_end_sample = core_start_sample, core_end_sample

        # 核心峰值频率：用于在叫声自身频带内做滞回延伸。
        peak_bin = int(np.argmax(np.where(frequency_mask, power[:, peak_frame], -np.inf)))
        # 每次最多向外看 BOUNDARY_EXTEND_CONTEXT_MS；只有延伸一直到达窗口边缘（叫声可能还在继续）
        # 时才再看更远处。不碰边缘就停，避免每次重估噪声底后一点点蹭进噪声。
        for _ in range(int(np.ceil(max_duration_ms / BOUNDARY_EXTEND_CONTEXT_MS))):
            new_start, new_end, start_edge, end_edge = extend_pulse_boundary(
                y, sr, core_start_sample, core_end_sample, float(frequencies[peak_bin]),
                max_duration_ms=max_duration_ms, report_edges=True,
            )
            grew = (new_start, new_end) != (core_start_sample, core_end_sample)
            core_start_sample, core_end_sample = new_start, new_end
            if not grew or not (start_edge or end_edge):
                break

        core_start_sample = max(0, core_start_sample - padding_samples)
        core_end_sample = min(len(y), core_end_sample + padding_samples)
        duration_ms = (core_end_sample - core_start_sample) / sr * 1000
        if not (MIN_PULSE_MS <= duration_ms <= max_duration_ms):
            continue

        signal_level_db = float(np.median(energy_db[start_frame:end_frame + 1]))
        cores.append({
            "candidate_start_s": pulse["start_s"],
            "candidate_end_s": pulse["end_s"],
            "core_start_s": seed_start_sample / sr,
            "core_end_s": seed_end_sample / sr,
            "boundary_start_s": core_start_sample / sr,
            "boundary_end_s": core_end_sample / sr,
            "boundary_method": "fixed_background_track",
            "start_s": core_start_sample / sr,
            "end_s": core_end_sample / sr,
            "duration_ms": duration_ms,
            "snr_db": signal_level_db - noise_level_db,
            "core_threshold_db": threshold_db,
            "detect_band_khz": detect_band if detect_band else None,
            "detect_source": band_label(detect_band),
            "peak_freq_khz": float(frequencies[peak_bin]) / 1000.0,
            "peak_freqs_khz": salient_peak_freqs_khz(power, frequencies, frequency_mask, start_frame, end_frame)
            or [float(frequencies[peak_bin]) / 1000.0],
            "fragment_count": 1,
        })
    return cores


def refine_pulse_boundary(
    y: np.ndarray,
    sr: int,
    pulse: dict[str, float],
    *,
    max_duration_ms: float = MAX_PULSE_MS,
) -> dict[str, float] | None:
    """只返回包含最强帧的那个核心（旧接口）；分析流程用 ``refine_pulse_cores`` 取全部核心。"""
    cores = refine_pulse_cores(y, sr, pulse, max_duration_ms=max_duration_ms, all_groups=False)
    return cores[0] if cores else None


def _merge_overlapping_pulses(
    first: dict,
    second: dict,
    max_duration_ms: float,
) -> dict | None:
    """来自不同频带的重叠候选取时间并集，保证两个频带的声波都留在区间内，
    由后续分离按频率拆分；并集超长时返回 None。"""
    start_s = min(float(first["start_s"]), float(second["start_s"]))
    end_s = max(float(first["end_s"]), float(second["end_s"]))
    duration_ms = (end_s - start_s) * 1000
    if duration_ms > max_duration_ms:
        return None
    stronger = first if float(first["snr_db"]) >= float(second["snr_db"]) else second
    merged = dict(stronger)
    merged.update(
        {
            "candidate_start_s": min(
                float(first["candidate_start_s"]), float(second["candidate_start_s"])
            ),
            "candidate_end_s": max(
                float(first["candidate_end_s"]), float(second["candidate_end_s"])
            ),
            "start_s": start_s,
            "end_s": end_s,
            "duration_ms": duration_ms,
        }
    )
    if 'boundary_start_s' in first or 'boundary_start_s' in second:
        merged['boundary_start_s']=start_s
        merged['boundary_end_s']=end_s
        merged['core_start_s']=min(float(p.get('core_start_s',p['start_s'])) for p in (first,second))
        merged['core_end_s']=max(float(p.get('core_end_s',p['end_s'])) for p in (first,second))
    sources = []
    for item in (first, second):
        for source in str(item.get("detect_source", "fullband")).split("|"):
            if source and source not in sources:
                sources.append(source)
    sources.sort(key=lambda source: (source != "fullband", source))
    merged["detect_source"] = "|".join(sources)
    # 不同频带的候选合并时保留双方的显著峰，后续同频碎片合并才不会只认“更强”那一方的频率。
    merged["peak_freqs_khz"] = _union_peaks(first, second)
    return merged


def refine_all_pulses(
    y: np.ndarray,
    sr: int,
    pulses: list[dict[str, float]],
    *,
    max_duration_ms: float = MAX_PULSE_MS,
) -> list[dict[str, float]]:
    """对所有候选脉冲进行边界精修，并删除重叠的重复候选。"""
    refined: list[dict[str, float]] = []

    for pulse in pulses:
        refined.extend(refine_pulse_cores(y, sr, pulse, max_duration_ms=max_duration_ms))

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
            if (
                previous.get("detect_band_khz") is None
                and pulse.get("detect_band_khz") is None
            ):
                # 原全带行为：保留信噪比较高的一条。
                if pulse["snr_db"] > previous["snr_db"]:
                    deduplicated[-1] = pulse
                continue
            merged = _merge_overlapping_pulses(previous, pulse, max_duration_ms)
            if merged is not None:
                deduplicated[-1] = merged
            elif pulse["snr_db"] > previous["snr_db"]:
                deduplicated[-1] = pulse
            continue

        deduplicated.append(pulse)

    return merge_adjacent_fragments(deduplicated, max_duration_ms=max_duration_ms, y=y, sr=sr)


def _peaks_of(pulse: dict) -> list[float]:
    peaks = pulse.get("peak_freqs_khz")
    if peaks:
        return [float(value) for value in peaks]
    value = pulse.get("peak_freq_khz")
    return [float(value)] if value is not None and np.isfinite(float(value)) else []


def _shared_peak(first: dict, second: dict) -> float | None:
    """两个候选显著峰中最接近的一对（相差不超过 FRAGMENT_MERGE_FREQ_KHZ）的平均频率；没有则 None。"""
    pairs = [(abs(a - b), (a + b) / 2) for a in _peaks_of(first) for b in _peaks_of(second)
             if abs(a - b) <= FRAGMENT_MERGE_FREQ_KHZ]
    return min(pairs)[1] if pairs else None


def _is_call_boundary(y: np.ndarray, sr: int, end_s: float, start_s: float, freq_khz: float) -> bool:
    """间隙两侧主频是否离开共同频率：前段结尾下扫或后段起点上扫说明是两条叫声。"""
    edge = FRAGMENT_EDGE_MS / 1000.0
    lo = max(0, int((end_s - edge - 0.002) * sr))
    hi = min(len(y), int((start_s + edge + 0.002) * sr))
    segment = y[lo:hi]
    if len(segment) < 256:
        return False
    hop = 16
    power = np.abs(librosa.stft(segment, n_fft=256, hop_length=hop, window="hann", center=True)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=256) / 1000.0
    band = (freqs >= freq_khz - BOUNDARY_EXTEND_BAND_BELOW_KHZ) & (freqs <= freq_khz + BOUNDARY_EXTEND_BAND_ABOVE_KHZ)
    if not np.any(band):
        return False
    dominant = freqs[band][np.argmax(power[band], axis=0)]
    times = lo / sr + np.arange(power.shape[1]) * hop / sr
    # 只看间隙两侧各 FRAGMENT_EDGE_MS；间隙本身常是短暂掉线（主频落在噪声上），不作依据。
    for mask in ((times >= end_s - edge) & (times <= end_s), (times >= start_s) & (times <= start_s + edge)):
        if np.any(mask) and abs(freq_khz - float(np.median(dominant[mask]))) > FRAGMENT_EDGE_MAX_SHIFT_KHZ:
            return True
    return False


def _union_peaks(first: dict, second: dict) -> list[float]:
    peaks: list[float] = []
    for value in _peaks_of(first) + _peaks_of(second):
        if all(abs(value - kept) > FRAGMENT_MERGE_FREQ_KHZ for kept in peaks):
            peaks.append(value)
    return peaks


def merge_adjacent_fragments(
    pulses: list[dict[str, float]],
    *,
    max_duration_ms: float = MAX_PULSE_MS,
    y: np.ndarray | None = None,
    sr: int | None = None,
) -> list[dict[str, float]]:
    """把同一叫声被截成的同频碎片合并为一个候选。

    正式流程提供原始音频，在50ms以内搜索同频碎片；正间隙必须有实际连续轨迹支持，
    并且合并后不超过时长上限。纯元数据旧接口仅保留旧的短间隙规则。
    不同频率的重叠候选保持分开。有间隙的两段还要通过 ``_is_call_boundary``：间隙任一侧主频偏离共同频率
    超过 ``FRAGMENT_EDGE_MAX_SHIFT_KHZ``（结尾下扫或起点上扫）视为两条叫声，不合并。
    """
    merged: list[dict[str, float]] = []
    gap_s = .050 if y is not None and sr is not None else FRAGMENT_MERGE_GAP_MS / 1000.0
    for pulse in sorted(pulses, key=lambda item: float(item["start_s"])):
        target = None
        # 只回看仍可能相邻的候选；中间夹着其他频率的候选也能接上。
        for index in range(len(merged) - 1, -1, -1):
            previous = merged[index]
            if float(previous["end_s"]) + gap_s < float(pulse["start_s"]):
                if float(previous["end_s"]) + gap_s + max_duration_ms / 1000.0 < float(pulse["start_s"]):
                    break
                continue
            shared = _shared_peak(previous, pulse)
            if shared is None:
                continue
            gap_exists = float(pulse["start_s"]) > float(previous["end_s"])
            if gap_exists and y is not None and sr is not None:
                from .boundary_reliability import gap_decision
                end, start = float(previous['end_s']), float(pulse['start_s'])
                lo = max(0, int((end-.100)*sr))
                hi = min(len(y), int((start+.100)*sr))
                if gap_decision(y[lo:hi], sr, lo/sr, end, start, shared*1000) != 'continuous':
                    continue
            if gap_exists and y is not None and sr is not None and _is_call_boundary(
                y, sr, float(previous["end_s"]), float(pulse["start_s"]), shared,
            ):
                continue
            target = index
            break
        if target is None:
            merged.append(pulse)
            continue
        previous = merged[target]
        union = _merge_overlapping_pulses(previous, pulse, max_duration_ms)
        if union is None:
            merged.append(pulse)
            continue
        union["peak_freqs_khz"] = _union_peaks(previous, pulse)
        union["fragment_count"] = int(previous.get("fragment_count", 1)) + int(pulse.get("fragment_count", 1))
        merged[target] = union
    merged.sort(key=lambda item: float(item["start_s"]))
    return merged


# ---------------------------------------------------------------------------
# 检测诊断（只读）：逐频带复现检测器判据，用于解释"为什么这个块没有候选"
# ---------------------------------------------------------------------------
def diagnose_band_detection(
    block: np.ndarray,
    sr: int,
    *,
    bands_khz: "list[tuple[float, float]] | None" = None,
    include_refined: bool = True,
) -> list[dict]:
    """逐频带给出检测器看到的东西；不改变任何检测结果。

    每行对应一个频带。`contrast_db` = 平滑后峰值 − 背景中位数，检测器要求它
    不低于自适应余量（`DETECTION_ADAPTIVE_MARGIN_DB`，默认 10 dB），所以
    `contrast_db` 小于余量的频带，在这个块里**不可能**产生候选。
    全带行（`band_khz="all"`）与 `detect_pulses` 完全一致，可用于核对。
    """
    if block.size == 0:
        return []

    power, frequencies, times, hop_length = build_spectrogram(block, sr)
    ultrasonic = ultrasonic_band_mask(frequencies, sr)
    frame_ms = hop_length / sr * 1000
    max_gap_frames = max(1, int(round(MERGE_GAP_MS / frame_ms)))
    padding_s = PULSE_PADDING_MS / 1000
    block_duration_s = len(block) / sr

    definitions = [
        ("all", float(LOW_FREQ_HZ) / 1000.0,
         float(min(HIGH_FREQ_HZ, sr / 2)) / 1000.0),
    ]
    definitions.extend(
        (f"{low:g}-{high:g}", float(low), float(high))
        for low, high in (bands_khz or DETECTION_DIAGNOSTIC_BANDS_KHZ)
    )

    rows: list[dict] = []
    for label, low_khz, high_khz in definitions:
        mask = (
            ultrasonic
            & (frequencies >= low_khz * 1000.0)
            & (frequencies <= high_khz * 1000.0)
        )
        if not np.any(mask):
            continue
        band_power = np.sum(power[mask, :], axis=0)
        energy_db = librosa.power_to_db(band_power + 1e-18, ref=np.max)
        if len(energy_db) >= 5:
            energy_db = np.convolve(energy_db, np.ones(5) / 5, mode="same")
        background_db = float(np.median(energy_db))
        peak_db = float(np.max(energy_db))
        threshold_db = max(
            ENERGY_THRESHOLD_DB, background_db + DETECTION_ADAPTIVE_MARGIN_DB,
        )
        active = energy_db >= threshold_db
        groups = merge_active_frames(active, max_gap_frames)
        pulses: list[dict[str, float]] = []
        for start_frame, end_frame in groups:
            start_s = max(0.0, float(times[start_frame]) - padding_s)
            end_s = min(
                block_duration_s,
                float(times[end_frame]) + frame_ms / 1000 + padding_s,
            )
            pulse_duration_ms = (end_s - start_s) * 1000
            if MIN_PULSE_MS <= pulse_duration_ms <= MAX_PULSE_MS:
                pulses.append(
                    {"start_s": start_s, "end_s": end_s,
                     "duration_ms": pulse_duration_ms}
                )
        refined = len(refine_all_pulses(block, sr, pulses)) if include_refined else None
        rows.append({
            "band_khz": label,
            "background_median_db": round(background_db, 3),
            "peak_db": round(peak_db, 3),
            "contrast_db": round(peak_db - background_db, 3),
            "threshold_db": round(threshold_db, 3),
            "peak_over_threshold_db": round(peak_db - threshold_db, 3),
            "active_frames": int(np.count_nonzero(active)),
            "total_frames": int(len(energy_db)),
            "groups": len(groups),
            "raw_candidates": len(pulses),
            "refined_candidates": refined,
        })
    return rows
