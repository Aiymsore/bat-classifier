"""同时近频声波分离（实验性功能；上游：pulse_table 的时间候选；下游：分离 WAV 输出）。

职责：在单个时间候选内逐帧追踪多个频谱峰轨迹，判断是否为
真正同时发声的独立声源（非谐波、时间重叠、频率可分辨），
并用软掩码把混合波形拆成多条分量音频。
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

from .audio_io import ultrasonic_band_mask
from .features import mask_concentrations
from .harmonics import group_harmonic_tracks, harmonic_component_metadata
from .config import (
    BOUNDARY_EXTEND_MARGIN_DB,
    BOUNDARY_EXTEND_MAX_GAP_MS,
    BOUNDARY_TRACK_STEP_KHZ,
    BASE_DIR,
    CORE_PADDING_MS,
    ENABLE_SIMULTANEOUS_SEPARATION,
    GATE_LOCAL_CONCENTRATION_MIN_HALF_WIDTH_KHZ,
    GATE_LOCAL_CONCENTRATION_RADIUS_FACTOR,
    MIN_PULSE_MS,
    SEPARATED_PULSE_DIR,
    SEPARATION_FRAME_DYNAMIC_RANGE_DB,
    SEPARATION_GLOBAL_MIN_DB,
    SEPARATION_HARMONIC_TOLERANCE,
    SEPARATION_HOP_LENGTH,
    SEPARATION_MASK_BANDWIDTH_KHZ,
    SEPARATION_MAX_COMPONENTS,
    SEPARATION_MAX_PEAKS_PER_FRAME,
    SEPARATION_MAX_TRACK_GAP_FRAMES,
    SEPARATION_MAX_TRACK_JUMP_KHZ,
    SEPARATION_MAX_WIN_LENGTH,
    SEPARATION_MIN_FREQ_GAP_KHZ,
    SEPARATION_MIN_SECONDARY_RELATIVE_DB,
    SEPARATION_MIN_TEMPORAL_OVERLAP,
    SEPARATION_MIN_TRACK_COVERAGE,
    SEPARATION_MIN_TRACK_MS,
    SEPARATION_N_FFT,
    SEPARATION_PEAK_PROMINENCE_DB,
    SEPARATION_WIN_LENGTH,
)


def _extend_observed_track_edges(track, power, frequencies, sr):
    """Recover observed weak ridge ends within the refined parent only.

    Peak contrast is measured in the original mixture. No zero padding or
    unobserved frequency interpolation is used to invent extension frames.
    """
    original_frames=np.asarray(track['frames'],dtype=int)
    original_freqs=np.asarray(track['frequencies_hz'],dtype=float)
    if not len(original_frames):
        return
    reference=float(np.median(original_freqs))
    stable=float(np.percentile(original_freqs,90)-np.percentile(original_freqs,10))<=1500.
    max_gap=max(1,int(round(BOUNDARY_EXTEND_MAX_GAP_MS/1000*sr/SEPARATION_HOP_LENGTH)))
    found={int(i):(float(f),float(p)) for i,f,p in zip(original_frames,original_freqs,track['powers'])}
    for origin,step,previous in [(int(original_frames[0]),-1,float(original_freqs[0])),
                               (int(original_frames[-1]),1,float(original_freqs[-1]))]:
        gap=0;frame=origin+step
        while 0<=frame<power.shape[1]:
            search=np.abs(frequencies-previous)<=max(BOUNDARY_TRACK_STEP_KHZ*1000,2*np.median(np.diff(frequencies)))
            if stable:
                search &= np.abs(frequencies-reference)<=1500.
            bins=np.flatnonzero(search)
            if not len(bins):break
            k=int(bins[np.argmax(power[bins,frame])]);value=float(power[k,frame])
            peak=(k==0 or value>=power[k-1,frame]) and (k==len(frequencies)-1 or value>=power[k+1,frame])
            distances=np.abs(frequencies-frequencies[k])
            neighbors=(distances>=1500.)&(distances<=6000.)
            noise=float(np.median(power[neighbors,frame])) if neighbors.any() else value
            if peak and value>max(noise,1e-30)*10**(BOUNDARY_EXTEND_MARGIN_DB/10):
                found[frame]=(float(frequencies[k]),value);previous=float(frequencies[k]);gap=0
            else:
                gap+=1
                if gap>max_gap:break
            frame+=step
    frames=sorted(found)
    track['frames']=np.asarray(frames,dtype=int)
    track['frequencies_hz']=np.asarray([found[i][0] for i in frames])
    track['powers']=np.asarray([found[i][1] for i in frames])
    track['coverage']=len(frames)/max(1,frames[-1]-frames[0]+1)


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
    *,
    reject_basic_harmonics: bool = True,
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
        and (
            not reject_basic_harmonics
            or not _tracks_are_harmonic_duplicates(primary, candidate)
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
    *,
    window_mode: str = "fixed",
    harmonic_grouping_mode: str = "basic",
) -> list[dict[str, object]]:
    """
    将一个时间候选展开为一条或多条同时发声的近频脉冲。

    只有检测到至少两条可靠、重叠且非谐波的轨迹时才修改
    波形；否则直接返回原波形，避免对普通单脉冲造成失真。
    """
    if window_mode not in {"fixed", "adaptive"}:
        raise ValueError("分离窗模式必须是 fixed 或 adaptive。")
    if harmonic_grouping_mode not in {"basic", "enhanced"}:
        raise ValueError("谐波分组模式必须是 basic 或 enhanced。")
    unchanged = _unchanged_component(pulse_y, 1.0)
    if (
        not ENABLE_SIMULTANEOUS_SEPARATION
        or len(pulse_y) < 64
    ):
        return [unchanged]

    base_win_length = min(SEPARATION_WIN_LENGTH, len(pulse_y))
    if window_mode == "adaptive" and len(pulse_y) > SEPARATION_WIN_LENGTH:
        # 保留至少约 3 个分析窗，避免把整条短叫声压成一帧；
        # 根据候选长度逐级延长窗，最多到 1024 点。
        target = max(1, len(pulse_y) // 3)
        adaptive_win_length = 2 ** int(np.floor(np.log2(target)))
        win_length = min(
            len(pulse_y),
            SEPARATION_MAX_WIN_LENGTH,
            SEPARATION_N_FFT,
            max(SEPARATION_WIN_LENGTH, adaptive_win_length),
        )
    else:
        win_length = base_win_length
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
    frequency_mask = ultrasonic_band_mask(
        frequencies,
        sr,
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
        if tracks:
            unchanged["_gate_track"] = _gate_track_material(tracks[0], sr, power=band_power, frequencies=band_frequencies)
        return [unchanged]

    if harmonic_grouping_mode == "enhanced":
        tracks = group_harmonic_tracks(tracks)
        if len(tracks) == 1:
            unchanged.update(harmonic_component_metadata(tracks[0]))
            unchanged["_gate_track"] = _gate_track_material(tracks[0], sr, power=band_power, frequencies=band_frequencies)
            return [unchanged]
    reject_basic_harmonics = harmonic_grouping_mode == "basic"
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
                reject_basic_harmonics=reject_basic_harmonics,
            )
        )
        if not independent:
            continue

        distinct_from_selected = all(
            _compare_simultaneous_tracks(
                selected,
                candidate,
                reject_basic_harmonics=reject_basic_harmonics,
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

    quality_tracks={id(track):deepcopy(track) for track in selected_tracks}
    for track in selected_tracks:
        members=track.get('_harmonic_members',[])
        for member in members:
            if member is not track:
                _extend_observed_track_edges(member,band_power,band_frequencies,sr)
        _extend_observed_track_edges(track,band_power,band_frequencies,sr)

    if len(selected_tracks) < 2:
        unchanged["_gate_track"] = _gate_track_material(quality_tracks[id(primary)], sr, power=band_power, frequencies=band_frequencies)
        unchanged["_boundary_track"] = _gate_track_material(primary, sr, power=band_power, frequencies=band_frequencies)
        unchanged.update(harmonic_component_metadata(primary))
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

    for component_index, track in enumerate(selected_tracks):
        # 谐波与主轨迹共用一个分量掩码；基本模式仅保留原轨迹。
        for member in track.get("_harmonic_members", [track]):
            track_frames = np.asarray(member["frames"], dtype=int)
            active_frames = (
                (frame_numbers >= int(track_frames[0]))
                & (frame_numbers <= int(track_frames[-1]))
            )
            interpolated_frequency = _track_frequency_at_frames(member, frame_numbers)
            distance = frequencies[:, None] - interpolated_frequency[None, :]
            gaussian = np.exp(-0.5 * (distance / sigma_hz) ** 2).astype(np.float32)
            track_power = np.interp(
                frame_numbers.astype(float),
                track_frames.astype(float),
                np.asarray(member["powers"], dtype=float),
            )
            track_power /= max(float(np.max(track_power)), 1e-18)
            gaussian *= np.sqrt(np.maximum(track_power, 1e-6))[None, :].astype(np.float32)
            gaussian[:, ~active_frames] = 0.0
            gaussian[~frequency_mask, :] = 0.0
            np.maximum(component_weights[component_index], gaussian,
                       out=component_weights[component_index])

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

        track_frames = np.unique(np.concatenate([
            np.asarray(member["frames"], dtype=int)
            for member in track.get("_harmonic_members", [track])
        ]))
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
                    reject_basic_harmonics=reject_basic_harmonics,
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
                **harmonic_component_metadata(track),
                "_gate_track": _gate_track_material(quality_tracks[id(track)], sr, start_sample, power=band_power, frequencies=band_frequencies),
                "_boundary_track": _gate_track_material(track, sr, start_sample, power=band_power, frequencies=band_frequencies),
            }
        )

    return components


def _gate_track_material(
    track: dict, sr: int, offset_samples: int = 0,
    *, power: np.ndarray | None = None, frequencies: np.ndarray | None = None,
) -> dict:
    """仅传递已观测的连续轨迹，随当前分量释放，不写入结果或模板。"""
    frames = np.asarray(track["frames"], dtype=int)
    span = max(1, int(frames[-1] - frames[0] + 1))
    material = {
        "times_s": (frames * SEPARATION_HOP_LENGTH - offset_samples) / float(sr),
        "frequencies_hz": np.asarray(track["frequencies_hz"], dtype=float),
        "coverage": len(frames) / span,
    }
    if power is not None and frequencies is not None:
        # 用分离前已有功率谱验证支持度，不能让窄带掩膜把宽带噪声变成高质量轨迹。
        selected_power = power[:, frames]
        near_bands = np.zeros(selected_power.shape, dtype=bool)
        local_bands = np.zeros(selected_power.shape, dtype=bool)
        local_radius = max(
            GATE_LOCAL_CONCENTRATION_MIN_HALF_WIDTH_KHZ * 1000.0,
            GATE_LOCAL_CONCENTRATION_RADIUS_FACTOR * 1500.0,
        )
        for member in track.get("_harmonic_members", [track]):
            curve = _track_frequency_at_frames(member, frames)
            in_time = (frames >= member["frames"][0]) & (frames <= member["frames"][-1])
            offsets = np.abs(frequencies[:, None] - curve[None, :])
            near_bands |= (offsets <= 1500.0) & in_time
            local_bands |= (offsets <= local_radius) & in_time
        concentration, concentration_local = mask_concentrations(
            selected_power, near_bands, local_bands,
        )
        material["original_concentration"] = concentration
        material["original_concentration_local"] = concentration_local
        totals = np.maximum(np.sum(selected_power, axis=0), 1e-18)
        probabilities = selected_power / totals
        material["original_entropy"] = float(np.mean(-np.sum(
            probabilities * np.log(np.maximum(probabilities, 1e-18)), axis=0)
            / np.log(max(len(frequencies), 2))))
    return material


def save_separated_pulse(
    audio_path: Path,
    candidate_id: int,
    component_id: int,
    component_audio: np.ndarray,
    sr: int,
    output_dir: Path | None = None,
) -> str:
    """只把确实发生拆分的分量写盘，并返回相对路径。

    output_dir 为 None 时保持历史行为（写入 results/audio/separated_pulses/）；
    分析时传入本次 run 目录，以隔离不同运行的音频输出。
    """
    if output_dir is None:
        output_dir = SEPARATED_PULSE_DIR / audio_path.stem
    output_dir = Path(output_dir)
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
    try:
        return str(output_path.relative_to(BASE_DIR))
    except ValueError:
        return str(output_path)


def _unchanged_component(
    pulse_y: np.ndarray,
    separation_confidence: float,
) -> dict[str, object]:
    """未发生拆分时返回的单一分量包装（原混合波形）。"""
    return {
        "audio": pulse_y,
        "start_sample": 0,
        "end_sample": len(pulse_y),
        "component_id": 1,
        "component_count": 1,
        "separation_applied": False,
        "track_median_freq_khz": float("nan"),
        "track_relative_db": 0.0,
        "separation_confidence": separation_confidence,
    }
