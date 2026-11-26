"""直接查看、逐 pulse 和录音概览共用的波形与彩色声谱图。"""

from __future__ import annotations

import json
from functools import wraps
from pathlib import Path
import threading

import librosa
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy.ndimage import binary_opening

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
# 按每个频率自身的背景检查短时活动，不要求叫声占整段录音的固定比例。
# 至少连续 2 帧、相邻 3 个频点共同突出，排除孤立尖峰；宽带冲击帧不参与聚焦。
AUTO_FOCUS_PROMINENCE_DB = 10.0
AUTO_FOCUS_MIN_FRAMES = 50
AUTO_FOCUS_MAX_BROADBAND_FRACTION = 0.25
AUTO_GAIN_RANGE_DB = 60.0
DISPLAY_PEAK_PERCENTILE = 99.9
MAX_DETAIL_SECONDS = 2.0
_PLOT_LOCK = threading.RLock()
WAVEFORM_TRACE_MAX_POINTS = 24_000
WAVEFORM_COLOR = "#ffad00"
PANEL_BACKGROUND = "#080b19"
SPECTROGRAM_CMAP = LinearSegmentedColormap.from_list(
    "avisoft_blue",
    [(0.0, "#030715"), (0.12, "#061343"), (0.30, "#0831a4"),
     (0.48, "#0064e3"), (0.65, "#00c6eb"), (0.80, "#21e69c"),
     (0.92, "#c0f13c"), (1.0, "#fff36b")],
)


def _serialized_plot(function):
    @wraps(function)
    def render(*args, **kwargs):
        # 局部重算与分析后台可能同时绘图，避免 pyplot 的全局状态互相干扰。
        with _PLOT_LOCK:
            return function(*args, **kwargs)
    return render


def save_representative_spectrogram(
    pulse_y: np.ndarray,
    sr: int,
    output_path: Path,
    title_lines: list[str],
    auto_focus: bool = False,
    show_waveform: bool = False,
    frequency_limits_hz: tuple[float, float] | None = None,
    component_annotations: list[tuple[float, str]] | None = None,
    source_audio: dict | None = None,
) -> bool:
    """代表样本和逐 pulse 与直接查看共用 STFT、布局、配色与坐标。

    标题中的类别、时长和频率数据原样保留；show_waveform 保留原接口。
    """
    return save_segment_call_overview(
        Path(), pulse_y, sr, [], output_path,
        title="\n".join(title_lines), auto_focus=auto_focus,
        frequency_limits_hz=frequency_limits_hz,
        component_annotations=component_annotations,
        show_waveform=show_waveform,
        source_audio=source_audio,
    )


def save_pulse_spectrogram(
    pulse_y: np.ndarray,
    sr: int,
    output_path: Path,
    title_lines: list[str],
    *,
    frequency_limits_hz: tuple[float, float] | None = None,
    component_annotations: list[tuple[float, str]] | None = None,
    source_audio: dict | None = None,
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
        source_audio=source_audio,
    )


def compute_overview_spectrum(y: np.ndarray, sr: int, max_time_bins: int = OVERVIEW_MAX_TIME_BINS):
    """导出与交互查看共用同一套 STFT，返回 dB、时间、频率、功率及参数。"""
    n_fft, base_hop = adaptive_stft_params(
        len(y), max_n_fft=1024, min_n_fft=32, min_hop=8,
    )
    hop_length = max(base_hop, int(np.ceil(len(y) / max_time_bins)))
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window="hann", center=True)
    power = np.abs(stft) ** 2
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(power.shape[1]), sr=sr, hop_length=hop_length)
    mask = ultrasonic_band_mask(frequencies, sr)
    selected_power, selected_frequencies = power[mask, :], frequencies[mask]
    power_db = (librosa.power_to_db(selected_power + 1e-18, ref=np.max)
                if selected_power.size else np.empty_like(selected_power))
    return power_db, times, selected_frequencies, selected_power, n_fft, hop_length


def save_segment_call_overview(
    audio_path: Path,
    y: np.ndarray,
    sr: int,
    pulse_rows: list[dict],
    output_path: Path,
    title: str | None = None,
    auto_focus: bool = True,
    *,
    frequency_limits_hz: tuple[float, float] | None = None,
    component_annotations: list[tuple[float, str]] | None = None,
    show_waveform: bool = True,
    source_audio: dict | None = None,
    time_offset_s: float = 0.0,
    max_time_bins: int = OVERVIEW_MAX_TIME_BINS,
) -> bool:
    """保存整段片段线性声谱图（Avisoft 风格，不叠加叫声标注）。

    用于 ① 片段识别与叫声统计：一段 WAV 的整体时频外观。
    只扩展现有 matplotlib 绘图逻辑，不另起一套。
    """
    if len(y) < 32:
        return False

    power_db, times, selected_frequencies, selected_power, n_fft, hop_length = compute_overview_spectrum(
        y, sr, max_time_bins,
    )
    if selected_power.size == 0:
        return False

    if source_audio is None and Path(audio_path).is_file():
        source_audio = {"path": str(Path(audio_path).resolve()), "offset_s": 0.0, "preprocessed": False}
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
        spectral_power=selected_power,
        auto_focus=auto_focus,
        frequency_limits_hz=frequency_limits_hz,
        component_annotations=component_annotations,
        show_waveform=show_waveform,
        source_audio=source_audio,
        time_offset_s=time_offset_s,
        render_metadata={"stft_n_fft": n_fft, "stft_hop_length": hop_length,
                         "time_step_s": hop_length / float(sr), "time_bins": len(times)},
    )


def save_segment_call_overview_from_summary(
    summary,
    pulse_rows: list[dict],
    output_path: Path,
    title: str | None = None,
    auto_focus: bool = True,
    waveform: np.ndarray | None = None,
    source_audio: dict | None = None,
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
        waveform_trace=summary.waveform_trace if waveform is None else None,
        duration_s=summary.total_duration_s,
        spectral_power=selected_power,
        auto_focus=auto_focus,
        source_audio=source_audio,
        render_metadata={"stft_n_fft": summary.n_fft, "stft_hop_length": summary.hop_length,
                         "time_step_s": summary.hop_length / float(summary.sr), "time_bins": len(times)},
    )


def save_quick_overview(
    audio_path: Path,
    output_path: Path,
    title: str | None = None,
    full_band: bool = False,
) -> bool:
    """① 直接查看入口（WAV）：与分析输出共用波形和蓝底彩色声谱图。

    与 save_segment_call_overview 同一绘图逻辑（含 sidecar 光标几何），
    区别只是 pulse_rows 为空、不依赖任何分析结果。长录音走流式低分辨率
    摘要路径，绝不为出图整条 STFT。``full_band=True`` 时显示完整超声频段（15–120 kHz），
    不做自动频率聚焦，便于数叫声时不漏掉偶尔出现的频段。
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
            auto_focus=not full_band,
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
        auto_focus=not full_band,
        waveform=y,
        source_audio={"path": str(audio_path.resolve()), "offset_s": 0.0, "preprocessed": False},
    )


def save_wav_interval_overview(
    source_audio: dict,
    start_s: float,
    end_s: float,
    output_path: Path,
    *,
    frequency_limits_hz: tuple[float, float] | None = None,
) -> bool:
    """从源 WAV 重新计算选区，图上仍使用原查看窗口的时间坐标。"""
    from .audio_io import load_audio_interval

    if end_s - start_s > MAX_DETAIL_SECONDS + 1e-9:
        raise ValueError(f"局部精细查看请选择 {MAX_DETAIL_SECONDS:g} 秒以内的范围")
    path = Path(source_audio["path"])
    offset_s = float(source_audio.get("offset_s", 0.0))
    y, sr, actual_start_s = load_audio_interval(
        path, start_s + offset_s, end_s + offset_s,
        preprocessed=bool(source_audio.get("preprocessed", False)),
    )
    return save_segment_call_overview(
        path, y, sr, [], output_path,
        title=f"{path.name} | 局部声谱 | {start_s:.6f}–{end_s:.6f} s",
        frequency_limits_hz=frequency_limits_hz, source_audio=source_audio,
        time_offset_s=actual_start_s - offset_s, max_time_bins=8000,
    )


def save_son_overview(
    son_path: Path,
    output_path: Path,
    title: str | None = None,
    full_band: bool = False,
) -> bool:
    """① 直接查看入口（SON）：用 .son 缓存的逐帧 FFT 幅度谱出线谱图。

    底部不重算 STFT，数据源是 SASLab 预先算好的频谱缓存，坐标轴来自
    .son 头部（采样率/帧移/频率分辨率）；同名 WAV 用于顶部真实波形。
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

    # SON 只保存频谱帧；若同目录存在同名 WAV，则用 WAV 的时域采样值
    # 绘制顶部波形。没有 WAV 时显示并标明频谱幅度包络。
    waveform = _load_companion_waveform(son_path, snapshot.sample_rate)
    companion = next((p for p in (son_path.with_suffix(".wav"), son_path.with_suffix(".WAV")) if p.is_file()), None)

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
        spectral_power=snapshot.magnitudes.T[mask, :] ** 2,
        auto_focus=not full_band,
        source_audio={"path": str(companion.resolve()), "offset_s": 0.0, "preprocessed": False} if companion else None,
    )


def save_direct_overview(
    source_path: Path,
    output_path: Path,
    title: str | None = None,
    full_band: bool = False,
) -> bool:
    """① 直接查看统一入口：按扩展名分派 WAV / SON；``full_band`` 显示完整超声频段。"""
    source_path = Path(source_path)
    suffix = source_path.suffix.lower()
    if suffix == ".son":
        return save_son_overview(source_path, output_path, title=title, full_band=full_band)
    if suffix != ".wav":
        raise ValueError(f"不支持的文件类型：{source_path.suffix}（仅支持 .wav / .son）")
    return save_quick_overview(source_path, output_path, title=title, full_band=full_band)


def _band_envelope(power: np.ndarray) -> np.ndarray:
    """没有 WAV 的 SON 文件用频谱估算幅度包络，明确标注数据来源。"""
    band_power = np.asarray(power, dtype=np.float64).sum(axis=0)
    if band_power.size == 0:
        return np.zeros(0)
    peak = float(band_power.max())
    if peak <= 0.0:
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
    """幅度峰值归一化后保留正负波形；长 WAV 保留每块的真实峰谷。"""
    values = np.asarray(waveform, dtype=np.float64)
    values = np.where(np.isfinite(values), values, 0.0)
    if values.size < 2:
        return None
    peak = float(np.max(np.abs(values)))
    if peak > 0.0:
        values = values / peak

    if values.size <= WAVEFORM_TRACE_MAX_POINTS:
        times = np.arange(values.size, dtype=np.float64) / float(sr)
        return times, values

    # 每块最多保留两个点，正负峰都不丢失。
    bucket = int(np.ceil(values.size / (WAVEFORM_TRACE_MAX_POINTS // 2)))
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
    """合并有时频连续性的活动频带；不确定时显示完整超声频段。"""
    full_low = float(LOW_FREQ_HZ)
    full_high = float(min(HIGH_FREQ_HZ, sr / 2.0))
    frequencies = np.asarray(selected_frequencies, dtype=np.float64)
    matrix = np.asarray(power_db, dtype=np.float64)
    if (frequencies.size < 4 or matrix.ndim != 2
            or matrix.shape[0] != frequencies.size or matrix.shape[1] < 2):
        return full_low, full_high, "full-fallback"

    safe_matrix = np.where(np.isfinite(matrix), matrix, -np.inf)
    profile = np.max(safe_matrix, axis=1)
    valid = np.isfinite(profile) & np.isfinite(frequencies)
    if int(valid.sum()) < 4:
        return full_low, full_high, "full-fallback"

    # 用各频点的中位背景逐帧判断，罕见叫声不会被整段 97% 分位数抹掉。
    # 无效值不计入背景，也不提供连续性支持。
    background = np.array([
        np.median(row[np.isfinite(row)]) if np.isfinite(row).any() else np.inf
        for row in matrix
    ])
    elevated = valid[:, None] & (safe_matrix >= background[:, None] + AUTO_FOCUS_PROMINENCE_DB)
    broadband = np.mean(elevated[valid], axis=0) > AUTO_FOCUS_MAX_BROADBAND_FRACTION
    elevated[:, broadband] = False
    supported = binary_opening(elevated, structure=np.ones((3, 2), dtype=bool))
    indices = np.flatnonzero(np.any(supported, axis=1))
    mode = "auto-all-bands"

    if indices.size < 3:
        # 短片段的叫声可能占据大多数帧，此时自身中位数不再代表底噪。
        # 长片段没有可靠活动时保守显示全带，不再只选一个最强频带。
        if matrix.shape[1] >= AUTO_FOCUS_MIN_FRAMES:
            return full_low, full_high, "full-fallback"
        peak = float(np.max(profile[valid]))
        active = valid[:, None] & (safe_matrix >= peak - AUTO_FOCUS_THRESHOLD_DB)
        active[:, broadband] = False
        supported = binary_opening(active, structure=np.ones((3, 2), dtype=bool))
        indices = np.flatnonzero(np.any(supported, axis=1))
        if indices.size < 3:
            return full_low, full_high, "full-fallback"
        mode = "auto"

    low = float(frequencies[indices[0]]) - AUTO_FOCUS_PADDING_HZ
    high = float(frequencies[indices[-1]]) + AUTO_FOCUS_PADDING_HZ
    center = (low + high) / 2.0
    span = max(high - low, AUTO_FOCUS_MIN_SPAN_HZ)
    low = center - span / 2.0
    high = center + span / 2.0
    low = max(full_low, low)
    high = min(full_high, high)
    if high <= low or (high - low) >= 0.85 * (full_high - full_low):
        return full_low, full_high, "full-fallback"
    return low, high, mode


def _spectrogram_display_limits(
    power_db: np.ndarray,
    frequencies: np.ndarray,
    frequency_limits_hz: tuple[float, float],
) -> tuple[float, float]:
    """用可见频段的参考峰值设置显示增益，避免极少数尖峰压暗整图。

    顶部 0.1% 的频谱点饱和为最高色，其余保持线性 dB 色阶；不改音频。
    """
    low, high = frequency_limits_hz
    visible = (frequencies >= low) & (frequencies <= high)
    visible_db = np.asarray(power_db, dtype=np.float64)[visible, :]
    finite_db = visible_db[np.isfinite(visible_db)]
    if not finite_db.size:
        return -80.0, 0.0
    reference_db = float(np.percentile(finite_db, DISPLAY_PEAK_PERCENTILE))
    return reference_db - AUTO_GAIN_RANGE_DB, reference_db


@_serialized_plot
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
    spectral_power: np.ndarray | None = None,
    waveform_trace: tuple[np.ndarray, np.ndarray] | None = None,
    frequency_limits_hz: tuple[float, float] | None = None,
    component_annotations: list[tuple[float, str]] | None = None,
    show_waveform: bool = True,
    duration_s: float | None = None,
    source_audio: dict | None = None,
    time_offset_s: float = 0.0,
    render_metadata: dict | None = None,
) -> bool:
    """统一出图：橙色正负波形在上，深蓝→蓝→青→绿→黄声谱在下。

    所有入口共用布局、显示动态范围及 sidecar；底层坐标始终为秒和 Hz。
    """
    if title is None:
        call_count = sum(row.get("gate_status", "bat") == "bat"
                         and str(row.get("predicted_label", "")) != "noise" for row in pulse_rows)
        title = f"{audio_name} | calls: {call_count}"
    title_lines = title.count("\n") + 1
    figure_height = 8.0 + max(0, title_lines - 1) * 0.2
    # 固定版面（不能用 bbox_inches="tight"：① 查看器靠这里记录的
    # 坐标区分数位置把 PNG 像素换算回频率，裁剪会让几何失效）
    figure = plt.figure(figsize=(14, figure_height))
    figure.subplots_adjust(
        left=0.08, right=0.985,
        top=1.0 - (0.4 + 0.16 * title_lines) / figure_height, bottom=0.09,
    )
    if show_waveform:
        grid = figure.add_gridspec(2, 1, height_ratios=(1.15, 1), hspace=0.045)
        wave_axis = figure.add_subplot(grid[0])
        axis = figure.add_subplot(grid[1], sharex=wave_axis)
    else:
        wave_axis = None
        axis = figure.add_subplot(111)

    # WAV 概览来自原始采样的正负峰值；只有无 WAV 的 SON 才用频谱包络。
    trace = _waveform_trace(waveform, sr) if waveform is not None else waveform_trace
    times = np.asarray(times) + time_offset_s
    waveform_mode = "hidden" if not show_waveform else "unavailable"
    if wave_axis is not None:
        waveform_label = "无波形数据"
        if trace is not None:
            waveform_mode = "wav-waveform"
            waveform_label = "波形（相对振幅）"
            trace_times, trace_values = trace
            trace_times = trace_times + time_offset_s
            wave_axis.fill_between(
                trace_times, 0.0, trace_values, color=WAVEFORM_COLOR, linewidth=0,
            )
            wave_axis.plot(
                trace_times, trace_values, color=WAVEFORM_COLOR, linewidth=0.45,
                antialiased=True,
            )
        elif spectral_power is not None:
            waveform_mode = "spectral-envelope"
            waveform_label = "频谱幅度包络（无原始 WAV）"
            envelope = _band_envelope(spectral_power)
            count = min(envelope.size, times.size)
            if count:
                wave_axis.fill_between(
                    times[:count], -envelope[:count], envelope[:count],
                    color=WAVEFORM_COLOR, linewidth=0,
                )
        wave_axis.axhline(0, color="#808080", linewidth=0.5)
        wave_axis.set_facecolor(PANEL_BACKGROUND)
        wave_axis.set_ylim(-1.08, 1.08)
        wave_axis.set_yticks([-1.0, 0.0, 1.0])
        wave_axis.set_ylabel("相对振幅")
        wave_axis.text(
            0.99, 0.96, waveform_label,
            transform=wave_axis.transAxes, ha="right", va="top",
            color="#dddddd", fontsize=8,
        )
        wave_axis.tick_params(
            axis="x", which="both", labelbottom=False, length=0,
        )

    if frequency_limits_hz is not None:
        focus_low, focus_high = frequency_limits_hz
        focus_mode = "explicit"
    elif auto_focus:
        focus_low, focus_high, focus_mode = _auto_frequency_limits(
            power_db,
            selected_frequencies,
            sr,
        )
    else:
        focus_low = float(LOW_FREQ_HZ)
        focus_high = float(min(HIGH_FREQ_HZ, sr / 2.0))
        focus_mode = "full"

    if spectral_power is not None and not np.any(np.asarray(spectral_power) > 0.0):
        # 静音在相对 dB 转换后可能整幅为 0；不能被自动增益画成黄色。
        display_db = np.full_like(power_db, -80.0, dtype=np.float64)
        display_vmin, display_vmax = -80.0, 0.0
    else:
        display_db = power_db
        display_vmin, display_vmax = _spectrogram_display_limits(
            power_db, selected_frequencies, (focus_low, focus_high),
        )

    axis.pcolormesh(
        times,
        selected_frequencies,
        display_db,
        shading="auto",
        vmin=display_vmin,
        vmax=display_vmax,
        cmap=SPECTROGRAM_CMAP,
    )
    axis.set_facecolor(SPECTROGRAM_CMAP(0.0))
    axis.set_ylim(focus_low, focus_high)
    # 绘图坐标与 sidecar 保留秒/Hz，刻度仅转换显示单位。
    if duration_s is None:
        duration_s = float(times[-1]) - time_offset_s if times.size else 1.0 / sr
    if waveform is not None:
        duration_s = len(waveform) / float(sr)
    axis.set_xlim(time_offset_s, time_offset_s + max(duration_s, 1.0 / sr))
    time_unit = "ms" if duration_s <= 0.25 else "s"
    if time_unit == "ms":
        axis.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 1000:g}"))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}"))
    axis.set_xlabel(f"Time ({time_unit})")
    axis.set_ylabel("Frequency (kHz)")
    for frequency_khz, label in component_annotations or []:
        frequency_hz = frequency_khz * 1000.0
        axis.axhline(frequency_hz, color="#ffffff", linewidth=0.5, alpha=0.65)
        axis.text(0.99, frequency_hz, label, transform=axis.get_yaxis_transform(),
                  ha="right", va="bottom", color="white", fontsize=8)
    for panel in (wave_axis, axis):
        if panel is not None:
            for spine in panel.spines.values():
                spine.set_color("#555566")
    (wave_axis if wave_axis is not None else axis).set_title(title, fontsize=10, pad=8)

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
        "waveform_normalization": "per-plot-amplitude-peak",
        "waveform_ylim": [float(v) for v in wave_axis.get_ylim()] if wave_axis is not None else None,
        "spectrogram_colormap": SPECTROGRAM_CMAP.name,
        "time_axis_unit": time_unit,
        "frequency_axis_unit": "kHz",
        "frequency_focus_mode": focus_mode,
        "display_db_range": [display_vmin, display_vmax],
        "display_scale_mode": "visible-percentile",
        "display_peak_percentile": DISPLAY_PEAK_PERCENTILE,
        "source_audio": source_audio,
    }
    meta.update(render_metadata or {})
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
