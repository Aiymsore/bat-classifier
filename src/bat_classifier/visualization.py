"""逐 pulse 波形/声谱图与整段录音概览。"""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

from .audio_io import adaptive_stft_params, ultrasonic_band_mask
from .config import HIGH_FREQ_HZ, LOW_FREQ_HZ, OVERVIEW_MAX_TIME_BINS


# 结果图需要支持放大查看细节，统一提高 PNG 输出密度。
OUTPUT_DPI = 300

# [可选显示实验区，可整体删除/回退]
# 这些参数只影响结果图的显示，不参与检测、特征或分类。
AUTO_FOCUS_THRESHOLD_DB = 24.0
AUTO_FOCUS_PADDING_HZ = 8_000.0
AUTO_FOCUS_MIN_SPAN_HZ = 20_000.0
AUTO_GAIN_RANGE_DB = 45.0
WAVEFORM_TRACE_MAX_POINTS = 24_000


def save_representative_spectrogram(
    pulse_y: np.ndarray,
    sr: int,
    output_path: Path,
    title_lines: list[str],
    auto_focus: bool = False,
    show_waveform: bool = False,
    frequency_limits_hz: tuple[float, float] | None = None,
    component_annotations: list[tuple[float, str]] | None = None,
) -> bool:
    """为代表样本保存统一尺度的单脉冲声谱图（专家审核用）。

    统一原则（避免不同图片尺度造成视觉误导）：
    - 默认频率范围固定为超声分析频段；逐 pulse 浏览器可传入 auto_focus 聚焦主频；
    - 颜色尺度固定 -60..0 dB；
    - 图尺寸与 DPI 固定；时间轴为脉冲自身区间并留 5% 边距。

    title_lines 直接显示在标题中（Cluster、Pulse ID、duration、
    peak/start/end frequency、SNR 等），不写“预测物种”。
    ``show_waveform`` 用于第七模块的逐声波图：在彩色频谱上方增加
    放大的真实 WAV 波形，便于按线性形状先做人工判断。
    """
    if len(pulse_y) < 32:
        return False

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
    frequencies = librosa.fft_frequencies(
        sr=sr,
        n_fft=n_fft,
    )
    power_db = librosa.power_to_db(
        power + 1e-18,
        ref=np.max,
    )

    mask = ultrasonic_band_mask(frequencies, sr)
    selected_frequencies = frequencies[mask]
    selected_db = power_db[mask, :]
    if selected_db.size == 0:
        return False

    times_ms = (
        librosa.frames_to_time(
            np.arange(power.shape[1]),
            sr=sr,
            hop_length=hop_length,
        )
        * 1000
    )

    if show_waveform:
        # 第七模块的逐声波图采用 Avisoft 类似的上下布局：顶部 pulse
        # 时域波形占更大高度，下面保留现有彩色频谱与自动聚焦频率轴。
        figure = plt.figure(figsize=(10, 7))
        grid = figure.add_gridspec(
            2,
            1,
            height_ratios=(2.3, 4.7),
            hspace=0.08,
        )
        wave_axis = figure.add_subplot(grid[0])
        axis = figure.add_subplot(grid[1], sharex=wave_axis)

        trace = _waveform_trace(pulse_y, sr)
        if trace is not None:
            trace_seconds, trace_values = trace
            trace_ms = trace_seconds * 1000.0
            wave_axis.fill_between(
                trace_ms,
                0.0,
                trace_values,
                color="#ffad00",
                linewidth=0,
                alpha=0.98,
            )
            wave_axis.plot(
                trace_ms,
                trace_values,
                color="#ffc14d",
                linewidth=0.35,
                antialiased=True,
            )
        wave_axis.axhline(0.0, color="#777777", linewidth=0.5)
        wave_axis.set_facecolor("black")
        wave_axis.set_ylim(-1.08, 1.08)
        wave_axis.set_yticks([])
        wave_axis.tick_params(
            axis="x", which="both", labelbottom=False, length=0
        )
        for spine in wave_axis.spines.values():
            spine.set_color("#404040")
        figure.suptitle(
            "\n".join(title_lines),
            fontsize=10,
            y=0.995,
            linespacing=1.15,
        )
    else:
        figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.pcolormesh(
        times_ms,
        selected_frequencies / 1000,
        selected_db,
        shading="auto",
        vmin=-60,
        vmax=0,
        cmap="magma",
    )
    if frequency_limits_hz is not None:
        focus_low, focus_high = frequency_limits_hz
    elif auto_focus:
        focus_low, focus_high, _ = _auto_frequency_limits(
            selected_db,
            selected_frequencies,
            sr,
        )
    else:
        focus_low = float(LOW_FREQ_HZ)
        focus_high = float(min(HIGH_FREQ_HZ, sr / 2))
    axis.set_ylim(focus_low / 1000, focus_high / 1000)
    for frequency_khz, label in component_annotations or []:
        axis.axhline(frequency_khz, color="cyan", linewidth=0.5, alpha=0.6)
        axis.text(0.99, frequency_khz, label, transform=axis.get_yaxis_transform(),
                  ha="right", va="bottom", color="cyan", fontsize=8)
    if len(times_ms):
        span = max(float(times_ms[-1]), 1e-6)
        if show_waveform:
            # 与顶部真实波形共用 pulse 自身时间范围，不把 STFT 的边缘
            # padding 误显示成额外叫声。
            pulse_duration_ms = max(
                float(len(pulse_y)) / max(float(sr), 1.0) * 1000.0,
                span,
            )
            axis.set_xlim(0.0, pulse_duration_ms)
        else:
            axis.set_xlim(-0.05 * span, 1.05 * span)
    axis.set_xlabel("Time (ms)")
    axis.set_ylabel("Frequency (kHz)")
    if not show_waveform:
        axis.set_title("\n".join(title_lines), fontsize=10)

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    if show_waveform:
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    else:
        figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=OUTPUT_DPI,
    )
    plt.close(figure)
    return True


def save_pulse_spectrogram(
    pulse_y: np.ndarray,
    sr: int,
    output_path: Path,
    title_lines: list[str],
    *,
    frequency_limits_hz: tuple[float, float] | None = None,
    component_annotations: list[tuple[float, str]] | None = None,
) -> bool:
    """保存一个已检测/分类 pulse 的独立声谱图。

    逐 pulse 图片与代表样本使用同一套 STFT、频率范围和颜色尺度，
    但输出路径由第七模块按最终类别分目录管理，便于逐类浏览。
    """
    return save_representative_spectrogram(
        pulse_y,
        sr,
        output_path,
        title_lines,
        auto_focus=True,
        show_waveform=True,
        frequency_limits_hz=frequency_limits_hz,
        component_annotations=component_annotations,
    )


def save_segment_call_overview(
    audio_path: Path,
    y: np.ndarray,
    sr: int,
    pulse_rows: list[dict],
    output_path: Path,
    title: str | None = None,
    auto_focus: bool = True,
) -> bool:
    """保存整段片段线性声谱图（Avisoft 风格，不叠加叫声标注）。

    用于 ① 片段识别与叫声统计：一段 WAV 的整体时频外观。
    只扩展现有 matplotlib 绘图逻辑，不另起一套。
    """
    if len(y) < 64:
        return False

    n_fft = 1024
    dynamic_hop = int(np.ceil(len(y) / OVERVIEW_MAX_TIME_BINS))
    hop_length = max(256, dynamic_hop)

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

    mask = ultrasonic_band_mask(frequencies, sr)
    selected_power = power[mask, :]
    selected_frequencies = frequencies[mask]
    if selected_power.size == 0:
        return False

    power_db = librosa.power_to_db(selected_power + 1e-18, ref=np.max)
    return _plot_segment_call_overview(
        power_db,
        times,
        selected_frequencies,
        sr,
        pulse_rows,
        output_path,
        title=title,
        audio_name=audio_path.name,
        waveform=y,
        auto_focus=auto_focus,
    )


def save_segment_call_overview_from_summary(
    summary,
    pulse_rows: list[dict],
    output_path: Path,
    title: str | None = None,
    auto_focus: bool = True,
    waveform: np.ndarray | None = None,
) -> bool:
    """① 长录音版 overview：用流式累计的低分辨率摘要绘图。

    与 save_segment_call_overview 同一绘图逻辑，区别只是声谱来自
    segment_stream.SpectrogramSummary 的逐块累计——
    **绝不为 overview 生成整条长录音的完整 STFT**。
    """
    if summary is None or summary.power is None:
        return False

    power = summary.power
    frequencies = summary.frequencies
    times = summary.times
    mask = ultrasonic_band_mask(frequencies, summary.sr)
    selected_power = power[mask, :]
    selected_frequencies = frequencies[mask]
    if selected_power.size == 0:
        return False

    power_db = librosa.power_to_db(selected_power + 1e-18, ref=np.max)
    return _plot_segment_call_overview(
        power_db,
        times,
        selected_frequencies,
        summary.sr,
        pulse_rows,
        output_path,
        title=title,
        audio_name="",
        waveform=waveform,
        auto_focus=auto_focus,
    )


def save_quick_overview(
    audio_path: Path,
    output_path: Path,
    title: str | None = None,
) -> bool:
    """① 直接查看入口（WAV）：不跑检测/分类，只出线性声谱图 + 波形条。

    与 save_segment_call_overview 同一绘图逻辑（含 sidecar 光标几何），
    区别只是 pulse_rows 为空、不依赖任何分析结果。长录音走流式低分辨率
    摘要路径，绝不为出图整条 STFT。
    """
    from .audio_io import load_audio
    from .segment_stream import SpectrogramSummary

    audio_path = Path(audio_path)
    y, sr = load_audio(audio_path)
    duration_s = len(y) / float(sr)

    if duration_s <= 30.0:
        return save_segment_call_overview(
            audio_path,
            y,
            sr,
            [],
            output_path,
            title=title or f"{audio_path.name} | 直接查看",
        )

    summary = SpectrogramSummary(duration_s, sr=sr)
    block = max(int(sr * 1.0), 4096)
    for start in range(0, len(y), block):
        summary.add(y[start : start + block], start / float(sr))
    return save_segment_call_overview_from_summary(
        summary,
        [],
        output_path,
        title=title or f"{audio_path.name} | 直接查看",
        auto_focus=True,
        waveform=y,
    )


def save_son_overview(
    son_path: Path,
    output_path: Path,
    title: str | None = None,
) -> bool:
    """① 直接查看入口（SON）：用 .son 缓存的逐帧 FFT 幅度谱出线谱图。

    底部不重算 STFT，数据源是 SASLab 预先算好的频谱缓存，坐标轴来自
    .son 头部（采样率/帧移/频率分辨率）；同名 WAV 只作为顶部真实波形的可选数据源。
    """
    from .son_reader import read_son

    son_path = Path(son_path)
    snapshot = read_son(son_path)
    power_db = snapshot.power_db()  # (bins, frames)
    frequencies = snapshot.frequencies()
    times = snapshot.times()

    mask = ultrasonic_band_mask(frequencies, snapshot.sample_rate)
    selected_power = power_db[mask, :]
    selected_frequencies = frequencies[mask]
    if selected_power.size == 0:
        return False

    # [可选真实波形显示，可删除]
    # SON 只保存频谱帧；若同目录存在同名 WAV，则用 WAV 的时域采样值
    # 绘制顶部波形，底部仍使用 SON 的原始频谱缓存。没有 WAV 时回退到
    # 频谱能量包络，保证单独分发 SON 文件时仍然可以查看。
    waveform = _load_companion_waveform(son_path, snapshot.sample_rate)

    return _plot_segment_call_overview(
        selected_power,
        times,
        selected_frequencies,
        snapshot.sample_rate,
        [],
        output_path,
        title=title or f"{son_path.name} | 直接查看",
        audio_name=son_path.name,
        waveform=waveform,
        auto_focus=True,
    )


def save_direct_overview(
    source_path: Path,
    output_path: Path,
    title: str | None = None,
) -> bool:
    """① 直接查看统一入口：按扩展名分派 WAV / SON。"""
    source_path = Path(source_path)
    suffix = source_path.suffix.lower()
    if suffix == ".son":
        return save_son_overview(source_path, output_path, title=title)
    if suffix != ".wav":
        raise ValueError(f"不支持的文件类型：{source_path.suffix}（仅支持 .wav / .son）")
    return save_quick_overview(source_path, output_path, title=title)


def _band_envelope(power_db: np.ndarray) -> np.ndarray:
    """由声谱 dB 反推带内能量包络，归一化到 0..1。

    顶部波形条用（≈ Avisoft 的示波条）：流式摘要路径拿不到原始波形，
    用每个时间帧的带内总能量做包络，脉冲处自然形成尖峰。
    动态范围过小（近乎静音）时返回全 0，避免画出一条满幅假信号。
    """
    linear = np.power(10.0, np.asarray(power_db, dtype=np.float64) / 10.0)
    band_power = linear.sum(axis=0)
    if band_power.size == 0:
        return np.zeros(0)
    peak = float(band_power.max())
    if peak <= 0.0:
        return np.zeros(band_power.shape[0])
    if float(np.ptp(band_power)) / peak < 0.5:
        return np.zeros(band_power.shape[0])
    return np.sqrt(band_power / peak)


def _load_companion_waveform(
    source_path: Path,
    target_sr: int,
) -> np.ndarray | None:
    """读取与 SON 同名的 WAV，作为顶部真实时域波形的可选数据源。"""
    source_path = Path(source_path)
    candidates = [
        source_path.with_suffix(".wav"),
        source_path.with_suffix(".WAV"),
    ]
    wav_path = next((path for path in candidates if path.exists()), None)
    if wav_path is None:
        return None

    try:
        waveform, sample_rate = librosa.load(
            wav_path,
            sr=None,
            mono=True,
        )
        waveform = np.asarray(waveform, dtype=np.float64)
        waveform = waveform[np.isfinite(waveform)]
        if waveform.size < 2:
            return None
        if int(sample_rate) != int(target_sr):
            waveform = librosa.resample(
                waveform,
                orig_sr=int(sample_rate),
                target_sr=int(target_sr),
            )
        peak = float(np.max(np.abs(waveform)))
        if peak <= 0.0:
            return None
        return (waveform / peak).astype(np.float32, copy=False)
    except Exception:  # noqa: BLE001 - 真实波形失败时回退频谱包络
        return None


def _waveform_trace(
    waveform: np.ndarray,
    sr: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """把 WAV 采样值压缩为可绘制的真实波形轨迹。"""
    values = np.asarray(waveform, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None
    peak = float(np.max(np.abs(values)))
    if peak <= 0.0:
        return None
    values = values / peak

    if values.size <= WAVEFORM_TRACE_MAX_POINTS:
        times = np.arange(values.size, dtype=np.float64) / float(sr)
        return times, values

    # 长 WAV 不逐点写入 PNG；每个小块保留 min/max，避免尖峰被抽样丢掉。
    bucket = int(np.ceil(values.size / WAVEFORM_TRACE_MAX_POINTS))
    trace_times: list[float] = []
    trace_values: list[float] = []
    for start in range(0, values.size, bucket):
        end = min(values.size, start + bucket)
        chunk = values[start:end]
        if chunk.size == 0:
            continue
        low_index = int(np.argmin(chunk))
        high_index = int(np.argmax(chunk))
        ordered = sorted(
            ((low_index, float(chunk[low_index])),
             (high_index, float(chunk[high_index])))
        )
        for offset, value in ordered:
            trace_times.append((start + offset) / float(sr))
            trace_values.append(value)
    if not trace_values:
        return None
    return (
        np.asarray(trace_times, dtype=np.float64),
        np.asarray(trace_values, dtype=np.float64),
    )


def _auto_frequency_limits(
    power_db: np.ndarray,
    selected_frequencies: np.ndarray,
    sr: int,
) -> tuple[float, float, str]:
    """根据可见频谱能量自动选择频率范围，失败时返回完整超声频段。"""
    full_low = float(LOW_FREQ_HZ)
    full_high = float(min(HIGH_FREQ_HZ, sr / 2.0))
    frequencies = np.asarray(selected_frequencies, dtype=np.float64)
    matrix = np.asarray(power_db, dtype=np.float64)
    if frequencies.size < 4 or matrix.ndim != 2:
        return full_low, full_high, "full-fallback"

    safe_matrix = np.where(np.isfinite(matrix), matrix, -np.inf)
    profile = np.max(safe_matrix, axis=1)
    valid = np.isfinite(profile) & np.isfinite(frequencies)
    if int(valid.sum()) < 4:
        return full_low, full_high, "full-fallback"

    valid_profile = profile[valid]
    peak = float(np.max(valid_profile))
    active = valid & (profile >= peak - AUTO_FOCUS_THRESHOLD_DB)
    active_indices = np.flatnonzero(active)
    if active_indices.size < 3:
        return full_low, full_high, "full-fallback"

    # 把不连续的频率活动分成几段，优先选择峰值最高的主叫声频带。
    gaps = np.flatnonzero(np.diff(active_indices) > 2) + 1
    groups = np.split(active_indices, gaps)
    group = max(
        groups,
        key=lambda indices: (
            float(np.max(profile[indices])),
            float(np.sum(np.maximum(profile[indices] - peak + AUTO_FOCUS_THRESHOLD_DB, 0.0))),
        ),
    )
    low = float(frequencies[group[0]]) - AUTO_FOCUS_PADDING_HZ
    high = float(frequencies[group[-1]]) + AUTO_FOCUS_PADDING_HZ
    center = (low + high) / 2.0
    span = max(high - low, AUTO_FOCUS_MIN_SPAN_HZ)
    low = center - span / 2.0
    high = center + span / 2.0
    low = max(full_low, low)
    high = min(full_high, high)
    if high <= low or (high - low) >= 0.85 * (full_high - full_low):
        return full_low, full_high, "full-fallback"
    return low, high, "auto"


def _plot_segment_call_overview(
    power_db: np.ndarray,
    times: np.ndarray,
    selected_frequencies: np.ndarray,
    sr: int,
    pulse_rows: list[dict],
    output_path: Path,
    title: str | None = None,
    audio_name: str = "",
    waveform: np.ndarray | None = None,
    auto_focus: bool = True,
) -> bool:
    """片段线性声谱图 + 波形条的公共绘图（旧/流式两路共用）。

    ① 片段详情的最终输出图，按 Avisoft SASLab 的线性图风格绘制：
    - 顶部窄条：优先使用同源 WAV 的真实时域波形，没有 WAV 时使用能量包络；
    - 主图：线性频率轴（Hz）× 线性时间轴（s），白底灰度，
      能量越强越黑，无色标；
    - 不叠加叫声区间/标签标注（纯观察视图，pulse_rows 只用于
      统计标题里的 call 数）。
    """
    # 固定版面（不能用 bbox_inches="tight"：① 查看器靠这里记录的
    # 坐标区分数位置把 PNG 像素换算回频率，裁剪会让几何失效）
    figure = plt.figure(figsize=(14, 6))
    figure.subplots_adjust(
        left=0.08, right=0.985, top=0.93, bottom=0.11
    )
    grid = figure.add_gridspec(
        2, 1, height_ratios=(1, 4), hspace=0.06
    )
    wave_axis = figure.add_subplot(grid[0])
    axis = figure.add_subplot(grid[1], sharex=wave_axis)

    # ---- 顶部：真实 WAV 波形；无 WAV 时回退为频谱能量包络 ----
    waveform_mode = "spectral-envelope"
    trace = _waveform_trace(waveform, sr) if waveform is not None else None
    if trace is not None:
        waveform_mode = "wav-waveform"
        trace_times, trace_values = trace
        wave_axis.plot(
            trace_times,
            trace_values,
            color="#ff9500",
            linewidth=0.45,
            antialiased=True,
        )
    else:
        envelope = _band_envelope(power_db)
        if envelope.size and times.size:
            count = min(envelope.size, times.size)
            wave_axis.fill_between(
                times[:count],
                -envelope[:count],
                envelope[:count],
                color="#ff9500",
                linewidth=0,
            )
    if (trace is not None) or (envelope.size and times.size):
        wave_axis.axhline(0, color="#808080", linewidth=0.5)
    wave_axis.set_facecolor("black")
    wave_axis.set_ylim(-1.08, 1.08)
    wave_axis.set_yticks([])
    wave_axis.tick_params(
        axis="x", which="both", labelbottom=False, length=0
    )
    for spine in wave_axis.spines.values():
        spine.set_color("#404040")

    # ---- 主图：白底灰度线性声谱图（灰度图：值越大越黑）----
    if auto_focus:
        focus_low, focus_high, focus_mode = _auto_frequency_limits(
            power_db,
            selected_frequencies,
            sr,
        )
    else:
        focus_low = float(LOW_FREQ_HZ)
        focus_high = float(min(HIGH_FREQ_HZ, sr / 2.0))
        focus_mode = "full"

    with np.errstate(all="ignore"):
        finite_db = np.asarray(power_db, dtype=np.float64)
        finite_db = finite_db[np.isfinite(finite_db)]
    if finite_db.size:
        display_vmax = float(np.max(finite_db))
        display_vmin = max(
            -80.0,
            display_vmax - AUTO_GAIN_RANGE_DB,
        )
    else:
        display_vmax = 0.0
        display_vmin = -80.0

    axis.pcolormesh(
        times,
        selected_frequencies,
        power_db,
        shading="auto",
        vmin=display_vmin,
        vmax=display_vmax,
        cmap="gray_r",
    )
    axis.set_ylim(focus_low, focus_high)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Frequency (Hz)")
    axis.text(
        0.995,
        0.01,
        f"灰度：越黑能量越强（{display_vmin:.0f}…{display_vmax:.0f} dB） | {focus_mode}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#666666",
    )

    # 不叠加红色叫声标注（纯观察视图）：pulse_rows 只用于标题里的 call 数
    call_index = sum(
        1
        for row in pulse_rows
        if str(row.get("predicted_label", "")) != "noise"
    )

    if title is None:
        title = (
            f"{audio_name} | calls: {call_index}"
        )
    wave_axis.set_title(title, fontsize=11, pad=6)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- sidecar：坐标区几何 + 频率范围，供 ① 查看器叠加频率光标 ----
    # 查看窗口读 PNG 时按这份数值把像素 y 换算回 Hz（等价 Avisoft 的游标）。
    position = axis.get_position()
    peak_hz: float | None = None
    if power_db.size and selected_frequencies.size:
        peak_index = int(np.argmax(power_db.max(axis=1)))
        peak_hz = float(selected_frequencies[peak_index])
    meta = {
        "axes_bbox": [
            float(position.x0),
            float(position.y0),
            float(position.width),
            float(position.height),
        ],
        "freq_ylim": [float(v) for v in axis.get_ylim()],
        "time_xlim": [float(v) for v in axis.get_xlim()],
        "figure_size": [float(v) for v in figure.get_size_inches()],
        "dpi": OUTPUT_DPI,
        "cursor_init_hz": peak_hz,
        "waveform_mode": waveform_mode,
        "frequency_focus_mode": focus_mode,
        "display_db_range": [display_vmin, display_vmax],
    }
    try:
        output_path.with_suffix(".json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:  # 绘图本身不受 sidecar 失败影响
        pass

    figure.savefig(output_path, dpi=OUTPUT_DPI)
    plt.close(figure)
    return True
