"""可解释的 CF / CF-FM / FM 形态门控；评分是规则分数，不是概率。

只读建表共用频谱和分离阶段已观测轨迹。没有谐波不扣分，也不推断缺失基频。
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from .features import (
    PulseSpectrum,
    extract_plateau_features,
    mask_concentrations,
    with_observed_support,
)
from .config import (
    GATE_LOCAL_CONCENTRATION_MIN_HALF_WIDTH_KHZ,
    GATE_LOCAL_CONCENTRATION_RADIUS_FACTOR,
    TEMPLATE_CF_MIN_PLATEAU_MS,
)

GATE_VERSION = "morphology-v6"  # 门控使用原混合谱中已找回的有效轨迹
BAT_THRESHOLD = 0.72
QUALITY_THRESHOLD = 0.55

# 局部带集中度：近场（轨迹 ±radius）能量占局部带（±max(6kHz, 4×radius)）的比例。
# 平谱噪声的理论基线约为 近场带宽/局部带宽 ≈ 3kHz/12kHz ≈ 0.25，与整段
# 15–120 kHz 的噪声总量无关。它不替代全带集中度（实测区分度更低），而是：
#   1) 全带集中度不足、但局部带仍有一条干净窄带峰时，救回真叫声；
#   2) 全带与局部带都没有峰时，判为宽带瞬态而不是留进待复核。
CONCENTRATION_LOCAL_MIN_HALF_WIDTH_KHZ = GATE_LOCAL_CONCENTRATION_MIN_HALF_WIDTH_KHZ
CONCENTRATION_LOCAL_RADIUS_FACTOR = GATE_LOCAL_CONCENTRATION_RADIUS_FACTOR
# 以下阈值由 20261020_125835 运行标定（见 docs/morphology_gate.md）；
# 它们不是由人工标注集校准出来的概率，门控分数依旧只是规则支持度。
BAT_CONCENTRATION = 0.25
BAT_RESCUE_CONCENTRATION_LOCAL = 0.90
BAT_RESCUE_SHAPE = 0.80
BAT_COVERAGE = 0.55
BAT_CONTINUITY = 0.65
BAT_TRACK_DURATION_MS = 0.5
# 宽带瞬态：全带与局部带都没有窄带峰、形状证据弱、无谐波支持。
IMPULSE_LOCAL_MAX = 0.40
IMPULSE_SHAPE_MAX = 0.70
IMPULSE_ENTROPY_MIN = 0.45

# 形状分与质量分继续使用全带集中度，避免一次改动同时移动所有分数；
# 局部带集中度只参与判定的两条旁路。
CONCENTRATION_SCORE_RAMP = (0.15, 0.65)

# 权重之和各为 1；CF-FM 还需同时具备可测的平台和扫频段。
CF_WEIGHTS = {"platform": 0.30, "stability": 0.25, "concentration": 0.20,
              "continuity": 0.15, "duration": 0.10}
FM_WEIGHTS = {"extent": 0.30, "direction": 0.25, "smoothness": 0.20,
              "concentration": 0.15, "duration": 0.10}
CFFM_WEIGHTS = {"platform": 0.25, "sweep": 0.25, "junction": 0.25,
                "concentration": 0.15, "continuity": 0.10}

# 与分类判族一致：横频平台超过 CF 最短平台时长后，CF-FM 轨迹通道
# 逐渐失去支持，带首尾 FM 的长 CF 叫声在门控里归 CF 形态。
PLATEAU_TRANSITION_SOFTNESS_MS = 2.0

# 判定用到的硬门槛，统一列表驱动 gate_reason 与 gate_fail_* 布尔列。
GATE_FAIL_KEYS = ("score", "quality", "concentration_local", "concentration",
                  "coverage", "continuity", "duration")


def _ramp(value: float, low: float, high: float) -> float:
    return float(np.clip((value - low) / max(high - low, 1e-12), 0.0, 1.0))


def _weighted(weights: dict, **values: float) -> float:
    return float(sum(weight * values[key] for key, weight in weights.items()))


def unavailable_gate(reason: str = "没有足够的可用轨迹") -> dict:
    """保持所有候选的 CSV 字段一致；缺少证据保留为待定。"""
    return {
        "gate_version": GATE_VERSION, "gate_status": "uncertain",
        "gate_reason": reason, "gate_reason_code": "insufficient_evidence",
        "gate_shape": "unresolved", "gate_score": 0.0,
        "gate_cf_score": 0.0, "gate_cffm_score": 0.0, "gate_fm_score": 0.0,
        "gate_cffm_source": "", "gate_tail_score": 0.0,
        "gate_quality": 0.0, "gate_continuity": 0.0, "gate_coverage": 0.0,
        "gate_concentration": 0.0, "gate_concentration_local": 0.0,
        "gate_entropy": 0.0,
        "gate_track_duration_ms": 0.0, "gate_sweep_khz": 0.0,
        "gate_platform_fraction": 0.0, "gate_platform_ms": 0.0,
        "gate_frequency_tolerance_khz": 0.0, "gate_local_band_hz": 0.0,
        "gate_context_contrast_db": None, "gate_background_available": False,
        "gate_persistent_tone": False, "gate_envelope_cv": 0.0,
        "gate_harmonic_support": 0.0, "gate_track_source": "unavailable",
        "gate_pass_path": "", "gate_promoted": False, "gate_fail_checks": "",
        **{f"gate_fail_{key}": False for key in GATE_FAIL_KEYS},
    }


GATE_COLUMNS = tuple(unavailable_gate())


def _sweep(f: np.ndarray, tolerance: float) -> tuple[float, float, float]:
    if len(f) < 3:
        return 0.0, 0.0, 0.0
    differences = np.diff(f)
    extent = abs(float(np.median(f[-2:]) - np.median(f[:2])))
    # 去除频率格点抖动后，单向弯曲扫频同样可以得到高分。
    direction = extent / max(float(np.sum(np.abs(differences))), extent, 1e-12)
    smoothness = 1.0 - _ramp(float(np.percentile(np.abs(np.diff(f, n=2)), 90)),
                             tolerance, 4.0 * tolerance)
    return _ramp(extent, max(1.5, 2 * tolerance), max(6.0, 6 * tolerance)), direction, smoothness


def _longest_platform(f: np.ndarray, times: np.ndarray, tolerance: float) -> tuple[int, int]:
    """最长连续窄频率区间；按观测格点给容差，不要求几十 Hz 的虚假精度。"""
    left = 0
    best = (0, 0)
    for right in range(len(f)):
        while left < right and float(np.ptp(f[left:right + 1])) > tolerance:
            left += 1
        if times[right] - times[left] > times[best[1]] - times[best[0]]:
            best = left, right
    return best


def _context_evidence(
    audio: np.ndarray, sr: int, start: int, end: int, frequency_hz: float,
) -> dict:
    """原混合音频前后各 4ms：RMS 对照及同频正弦投影，不另做 STFT。

    同频投影只用于稳定 CF 干扰证据；原始背景绝不与分离后波形直接比强弱。
    """
    length = max(1, int(round(0.004 * sr)))
    before = audio[max(0, start - length):start]
    after = audio[end:min(len(audio), end + length)]
    inside = audio[start:end]
    result = {"gate_context_contrast_db": None, "gate_background_available": False,
              "gate_persistent_tone": False, "gate_envelope_cv": 0.0}
    if len(inside) < 32:
        return result
    frame = max(8, int(sr * 0.00025))
    count = len(inside) // frame
    if count >= 3:
        envelope = np.sqrt(np.mean(inside[:count * frame].reshape(count, frame) ** 2, axis=1))
        result["gate_envelope_cv"] = float(np.std(envelope) / max(np.mean(envelope), 1e-12))
    if min(len(before), len(after)) < max(32, length // 2):
        return result
    result["gate_background_available"] = True
    signal = float(np.sqrt(np.mean(inside ** 2)))
    background = float(np.sqrt((np.mean(before ** 2) + np.mean(after ** 2)) / 2))
    result["gate_context_contrast_db"] = float(np.clip(
        20 * np.log10(max(signal, 1e-12) / max(background, 1e-12)), -120, 120))

    def tone_amplitude(samples: np.ndarray) -> float:
        phase = np.arange(len(samples)) * (2 * np.pi * frequency_hz / sr)
        return float(2 * abs(np.dot(samples, np.exp(-1j * phase))) / len(samples))

    # 长 CF 的分辨率误差会令长窗投影相消，三个等长短窗取中位数。
    probe = min(len(before), len(after), len(inside), max(32, int(sr * 0.001)))
    def probes(samples: np.ndarray) -> float:
        starts = np.linspace(0, len(samples) - probe, 3).astype(int)
        return float(np.median([tone_amplitude(samples[i:i + probe]) for i in starts]))
    amplitude = probes(inside)
    result["gate_persistent_tone"] = bool(
        amplitude > max(signal * 0.3, 1e-10)
        and min(probes(before), probes(after)) >= amplitude * 0.70
        and result["gate_envelope_cv"] < 0.15
    )
    return result


def score_morphology(
    spectrum: PulseSpectrum, *, component: dict, context_audio: np.ndarray,
    start_sample: int, end_sample: int,
) -> dict:
    """三个并列形态通道 + 信号质量、背景和可信谐波证据。"""
    result = unavailable_gate()
    power, frequencies = spectrum.power, spectrum.frequencies
    sr = spectrum.sample_rate
    frame_times = np.arange(power.shape[1]) * spectrum.hop_length / sr
    boundary_track = component.get("_boundary_track")
    if boundary_track is not None:
        # The same observed ridge supports duration and gate evaluation. Keep
        # raw-mixture concentration checks below; a separation mask alone is
        # never evidence that weak frames are clean signal.
        spectrum = with_observed_support(spectrum, boundary_track)
    if boundary_track is not None and spectrum.supported_frames is not None:
        active = spectrum.supported_frames.copy()
    else:
        active = spectrum.frame_peak_power >= np.max(spectrum.frame_peak_power) * 10 ** (-24 / 10)
    active &= frame_times <= (spectrum.sample_count - 1) / sr
    if np.count_nonzero(active) < 3:
        return result
    track = boundary_track if boundary_track is not None else component.get("_gate_track")
    track_source = "spectrum"
    valid = active.copy()
    f_all = spectrum.dominant_freqs_hz.copy() / 1000
    if track is not None and len(track["times_s"]) >= 3:
        source_times = np.asarray(track["times_s"])
        valid &= (frame_times >= source_times[0]) & (frame_times <= source_times[-1])
        f_all = np.interp(frame_times, source_times, track["frequencies_hz"]) / 1000
        track_source = "observed_boundary" if boundary_track is not None else "separation"
    if np.count_nonzero(valid) < 3:
        return result
    times = frame_times[valid] * 1000
    raw_f = f_all[valid]
    f = median_filter(raw_f, size=3, mode="nearest")
    delta_f = float(np.median(np.diff(frequencies))) / 1000
    tolerance = max(1.0, 2 * delta_f)
    duration = float(times[-1] - times[0])
    # 大幅跳频不通过插值或平滑变成看似连续的 FM。
    gaps = np.diff(times)
    nominal_step = spectrum.hop_length / sr * 1000
    continuity = float(np.mean((np.abs(np.diff(raw_f)) <= np.maximum(3.0, gaps * 12.0))
                               & (gaps <= nominal_step * 2.1)))
    coverage = float(np.count_nonzero(valid) / max(np.count_nonzero(active), 1))
    if track is not None:
        coverage *= float(track.get("coverage", 1.0))
        continuity *= float(track.get("coverage", 1.0))

    # 每一帧围绕轨迹统计；FM 的总带宽很大也可以有很高的瞬时集中度。
    # 同时统计局部带：平谱噪声的局部基线约 0.25，不随整段频带噪声总量漂移。
    radius = max(1500.0, delta_f * 1000 * 2)
    local_radius = max(CONCENTRATION_LOCAL_MIN_HALF_WIDTH_KHZ * 1000.0,
                       CONCENTRATION_LOCAL_RADIUS_FACTOR * radius)
    centers_hz = f_all[None, :] * 1000.0
    bands = np.abs(frequencies[:, None] - centers_hz) <= radius
    local_bands = np.abs(frequencies[:, None] - centers_hz) <= local_radius
    evidence = component.get("harmonic_evidence", [])
    harmonic_support = 0.0
    if evidence and component.get("harmonic_group_size", 1) > 1:
        harmonic_support = float(np.clip(min(
            item.get("ratio_coverage", 0.0) * item.get("envelope_correlation", 0.0)
            for item in evidence), 0, 1))
        members = component.get("harmonic_frequencies_khz", [])
        reference = float(np.median(raw_f))
        for frequency in members:
            ratio = float(frequency) / max(reference, 1e-12)
            offsets = np.abs(frequencies[:, None] - centers_hz * ratio)
            bands |= offsets <= radius
            local_bands |= offsets <= local_radius
    active_power = power[:, valid]
    totals = np.maximum(np.sum(active_power, axis=0), 1e-18)
    concentration, concentration_local = mask_concentrations(
        active_power, bands[:, valid], local_bands[:, valid],
    )
    probabilities = active_power / totals
    entropy = float(np.mean(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-18)), axis=0)
                            / np.log(max(len(frequencies), 2))))
    if track is not None:
        concentration = min(concentration, float(track.get("original_concentration", concentration)))
        concentration_local = min(
            concentration_local,
            float(track.get("original_concentration_local", concentration_local)),
        )
        entropy = max(entropy, float(track.get("original_entropy", entropy)))
    concentration_score = _ramp(concentration, *CONCENTRATION_SCORE_RAMP)
    duration_score = _ramp(duration, 0.25, 0.8) * (1 - _ramp(duration, 80, 150))
    first, last = _longest_platform(f, times, tolerance)
    platform_ms = float(times[last] - times[first])
    platform_fraction = platform_ms / max(duration, 1e-12)
    stability = 1 - _ramp(float(np.percentile(f, 95) - np.percentile(f, 5)), tolerance, 3 * tolerance)
    cf = _weighted(CF_WEIGHTS, platform=platform_fraction, stability=stability,
                   concentration=concentration_score, continuity=continuity, duration=duration_score)
    extent, direction, smoothness = _sweep(f, tolerance)
    fm = extent * _weighted(FM_WEIGHTS, extent=extent, direction=direction, smoothness=smoothness,
                            concentration=concentration_score, duration=duration_score)

    # 搜索相邻的 CF→FM / FM→CF；也检查最长平台两边，覆盖 FM→CF→FM。
    cffm = 0.0
    splits = np.unique(np.linspace(2, max(2, len(f) - 3), min(24, max(0, len(f) - 4))).astype(int))
    partitions = [(0, split, split, len(f) - 1) for split in splits]
    partitions += [(split, len(f) - 1, 0, split) for split in splits]
    partitions += [(first, last, 0, first), (first, last, last, len(f) - 1)]
    for a, b, c, d in partitions:
        if min(b - a, d - c) < 2:
            continue
        cf_ms = float(times[b] - times[a])
        fm_ms = float(times[d] - times[c])
        flatness = 1 - _ramp(float(np.ptp(f[a:b + 1])), tolerance, 2 * tolerance)
        platform_score = flatness * _ramp(cf_ms, 0.3, 0.8) * _ramp(cf_ms / max(duration, 1e-12), 0.10, 0.30)
        sweep_extent, sweep_direction, sweep_smooth = _sweep(f[c:d + 1], tolerance)
        sweep_score = sweep_extent * (0.6 * sweep_direction + 0.4 * sweep_smooth) * _ramp(fm_ms, 0.2, 0.5)
        # 两段接点共享同一观测点；另外检验未平滑轨迹接点附近是否跳频。
        junction_index = b if b == c else a
        junction_jump = float(np.max(np.abs(np.diff(raw_f[max(0, junction_index - 1):junction_index + 2]))))
        junction = 1 - _ramp(junction_jump, tolerance, 4 * tolerance)
        value = min(platform_score, sweep_score) * _weighted(
            CFFM_WEIGHTS, platform=platform_score, sweep=sweep_score, junction=junction,
            concentration=concentration_score, continuity=continuity)
        cffm = max(cffm, value)
    plateau_ms = extract_plateau_features(spectrum)["cf_plateau_ms"]
    short_support = 1 - _ramp(plateau_ms, TEMPLATE_CF_MIN_PLATEAU_MS - PLATEAU_TRANSITION_SOFTNESS_MS,
                              TEMPLATE_CF_MIN_PLATEAU_MS + PLATEAU_TRANSITION_SOFTNESS_MS)
    cffm *= short_support
    cffm_source = "track" if cffm > 0 else ""

    tail_score = 0.0  # 旧导出字段占位；不再提取或使用尾段能量证据
    track_best = max(cf, cffm, fm)

    quality = (0.30 * continuity + 0.30 * concentration_score + 0.20 * coverage + 0.20 * duration_score)
    context = _context_evidence(context_audio, sr, start_sample, end_sample, float(np.median(f)) * 1000)
    shapes = {"CF": cf, "CF-FM": cffm, "FM": fm}
    # 组合形态有充分证据且接近最优时优先标为 CF-FM，避免长平台掩盖短尾巴。
    shape = max(shapes, key=shapes.get)
    if cffm >= 0.65 and cffm >= shapes[shape] - 0.08:
        shape = "CF-FM"
    best = max(shapes.values())
    contrast = context["gate_context_contrast_db"]
    background_bonus = 0.0 if contrast is None else 0.04 * _ramp(contrast, 3, 12)
    persistent_penalty = 0.35 if context["gate_persistent_tone"] and stability >= 0.8 else 0.0
    combined = float(np.clip(0.8 * best + 0.2 * quality + 0.05 * harmonic_support
                             + background_bonus - persistent_penalty, 0, 1))
    checks = (
        ("score", combined, BAT_THRESHOLD, "综合分数"),
        ("quality", quality, QUALITY_THRESHOLD, "质量分数"),
        ("coverage", coverage, BAT_COVERAGE, "轨迹覆盖率"),
        ("continuity", continuity, BAT_CONTINUITY, "连续性"),
        ("duration", duration, BAT_TRACK_DURATION_MS, "轨迹时长(ms)"),
        ("concentration", concentration, BAT_CONCENTRATION, "全带集中度"),
        ("concentration_local", concentration_local, BAT_RESCUE_CONCENTRATION_LOCAL, "局部集中度"),
    )
    required_keys = {"score", "quality", "coverage", "continuity", "duration"}
    core_ok = all(
        value >= threshold
        for key, value, threshold, _ in checks if key in required_keys
    )
    # 两条并列的能量证据：全带集中度达标，或全带不足但局部带仍有一条干净窄带峰。
    if core_ok and concentration >= BAT_CONCENTRATION:
        pass_path = "band"
    elif core_ok and concentration_local >= BAT_RESCUE_CONCENTRATION_LOCAL and best >= BAT_RESCUE_SHAPE:
        pass_path = "rescue"
    else:
        pass_path = ""

    status, reason_code, reason = "uncertain", "ambiguous_shape", "形态或信号质量证据不足，保留复核"
    if context["gate_persistent_tone"] and stability >= 0.8 and platform_fraction >= 0.85:
        status, reason_code, reason = "noise", "persistent_tone", "稳定窄带线在前后背景均持续存在，且包络近似恒定"
    elif concentration < 0.18 and entropy > 0.80 and track_best < 0.60 and harmonic_support == 0:
        status, reason_code, reason = "noise", "diffuse_broadband", "能量在同一时刻宽频分散，缺少充分的叫声形态证据"
    elif (concentration < BAT_CONCENTRATION
          and concentration_local < IMPULSE_LOCAL_MAX and track_best < IMPULSE_SHAPE_MAX
          and harmonic_support == 0.0 and entropy > IMPULSE_ENTROPY_MIN):
        # 只在“全带集中度已经不达标”的候选中使用：全带与局部带都没有窄带峰时，
        # 轨迹只是宽带瞬态里的一个起伏，按噪声处理，不占人工复核。
        status, reason_code, reason = (
            "noise", "broadband_impulse",
            "全带与局部带均无窄带峰，形态证据弱且无谐波支持，判为宽带瞬态",
        )
    elif pass_path:
        status, reason_code = "bat", "supported_shape"
        reason = (f"{shape} 形态、轨迹连续性与能量集中度满足门控条件" if pass_path == "band"
                  else f"{shape} 形态清晰，局部带集中度达标，按局部带救回（全带能量偏散）")
    failed_keys = {key for key, value, threshold, _ in checks if value < threshold}
    if status == "bat":
        failed_keys = set()
    elif status == "uncertain":
        reason = "；".join(
            f"{name} {value:.3f}<{threshold:g}"
            for _, value, threshold, name in checks if value < threshold
        ) + "；保留复核"
    result.update(context)
    result.update({
        "gate_status": status, "gate_reason_code": reason_code, "gate_reason": reason,
        "gate_shape": shape, "gate_score": combined, "gate_pass_path": pass_path,
        "gate_promoted": False,
        "gate_cf_score": float(cf), "gate_cffm_score": float(cffm), "gate_fm_score": float(fm),
        "gate_cffm_source": cffm_source, "gate_tail_score": float(tail_score),
        "gate_quality": float(quality), "gate_continuity": continuity, "gate_coverage": coverage,
        "gate_concentration": concentration, "gate_concentration_local": concentration_local,
        "gate_entropy": entropy,
        "gate_track_duration_ms": duration, "gate_sweep_khz": float(np.ptp(f)),
        "gate_platform_fraction": platform_fraction, "gate_platform_ms": platform_ms,
        "gate_frequency_tolerance_khz": tolerance, "gate_local_band_hz": local_radius,
        "gate_harmonic_support": harmonic_support,
        "gate_track_source": track_source,
        "gate_fail_checks": "；".join(sorted(failed_keys)),
        **{f"gate_fail_{key}": key in failed_keys for key in GATE_FAIL_KEYS},
    })
    return result
