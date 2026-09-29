"""结果可视化输出（被 pulse_table、clustering、prediction 调用）。

职责：整段录音概览声谱图、最强脉冲放大网格图、聚类 PCA 散点图。
导入本模块即完成 matplotlib 后端与中文字体设置。
"""

from __future__ import annotations

from pathlib import Path

import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import medfilt

matplotlib.use("Agg")

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

from .audio_io import adaptive_stft_params, ultrasonic_band_mask
from .config import (
    ACOUSTIC_TYPE_LABELS,
    CLUSTER_PCA_PNG,
    CONTOUR_MIN_DB,
    HIGH_FREQ_HZ,
    LOW_FREQ_HZ,
    OVERVIEW_MAX_TIME_BINS,
    PULSE_ZOOM_DIR,
    SPECTROGRAM_DIR,
    TOP_PULSES_PER_FILE,
    ZOOM_CONTEXT_MS,
)


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
        n_fft, hop_length = adaptive_stft_params(
            len(segment),
            max_n_fft=512,
            min_n_fft=128,
            min_hop=8,
        )

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

        frequency_mask = ultrasonic_band_mask(
            pulse_frequencies,
            sr,
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

    mask = ultrasonic_band_mask(frequencies, sr)
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


def save_pca_plot(
    data: pd.DataFrame,
    labels: list[str] | None = None,
    output_path: Path | None = None,
    label_prefix: str = "声型",
    title: str | None = None,
    label_column: str = "acoustic_type",
) -> None:
    """保存每个点代表一个脉冲的二维PCA聚类图。

    output_path/label_prefix/title/label_column 默认保持历史行为
    （A-D 声型图）；批量整理（curation）传入自己的 run 目录、
    "Cluster " 前缀与 cluster_id 列，复用同一套绘图逻辑而不复制代码。
    """
    if labels is None:
        labels = list(ACOUSTIC_TYPE_LABELS)
    if output_path is None:
        output_path = CLUSTER_PCA_PNG
    figure, axis = plt.subplots(
        figsize=(10, 7)
    )

    for acoustic_type in labels:
        group = data[
            data[label_column]
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
                f"{label_prefix}{acoustic_type} "
                f"({len(group)})"
            ),
        )

    if title is None:
        if labels == ["A", "B", "C", "D"]:
            title = (
                "Bat acoustic pulse clusters A-D (PCA)"
            )
        else:
            title = (
                "Bat acoustic pulse clusters (PCA)"
            )
    axis.set_title(title)
    axis.set_xlabel("PCA component 1")
    axis.set_ylabel("PCA component 2")
    axis.legend()
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=180,
    )
    plt.close(figure)


def save_representative_spectrogram(
    pulse_y: np.ndarray,
    sr: int,
    output_path: Path,
    title_lines: list[str],
) -> bool:
    """为代表样本保存统一尺度的单脉冲声谱图（专家审核用）。

    统一原则（避免不同图片尺度造成视觉误导）：
    - 频率范围固定为超声分析频段 [LOW_FREQ_HZ, min(HIGH_FREQ_HZ, sr/2)]；
    - 颜色尺度固定 -60..0 dB；
    - 图尺寸与 DPI 固定；时间轴为脉冲自身区间并留 5% 边距。

    title_lines 直接显示在标题中（Cluster、Pulse ID、duration、
    peak/start/end frequency、SNR 等），不写“预测物种”。
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

    figure, axis = plt.subplots(
        figsize=(9, 4.5)
    )
    axis.pcolormesh(
        times_ms,
        selected_frequencies / 1000,
        selected_db,
        shading="auto",
        vmin=-60,
        vmax=0,
        cmap="magma",
    )
    axis.set_ylim(
        LOW_FREQ_HZ / 1000,
        min(HIGH_FREQ_HZ, sr / 2) / 1000,
    )
    if len(times_ms):
        span = max(float(times_ms[-1]), 1e-6)
        axis.set_xlim(-0.05 * span, 1.05 * span)
    axis.set_xlabel("Time (ms)")
    axis.set_ylabel("Frequency (kHz)")
    axis.set_title("\n".join(title_lines), fontsize=10)

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=150,
    )
    plt.close(figure)
    return True


def save_segment_call_overview(
    audio_path: Path,
    y: np.ndarray,
    sr: int,
    pulse_rows: list[dict],
    output_path: Path,
    title: str | None = None,
) -> bool:
    """保存整段片段声谱图，并在每个检测叫声区上标注编号与预测标签。

    用于 ⑦ 片段识别与叫声统计：一张图看清
    “这一段 WAV 里有几次叫声、每次是什么标签”。
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
    )


def save_segment_call_overview_from_summary(
    summary,
    pulse_rows: list[dict],
    output_path: Path,
    title: str | None = None,
) -> bool:
    """⑦ 长录音版 overview：用流式累计的低分辨率摘要绘图。

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
    )


def _plot_segment_call_overview(
    power_db: np.ndarray,
    times: np.ndarray,
    selected_frequencies: np.ndarray,
    sr: int,
    pulse_rows: list[dict],
    output_path: Path,
    title: str | None = None,
    audio_name: str = "",
) -> bool:
    """片段声谱图 + 叫声标注的公共绘图（旧/流式两路共用）。"""
    figure, axis = plt.subplots(figsize=(14, 5))
    mesh = axis.pcolormesh(
        times,
        selected_frequencies / 1000,
        power_db,
        shading="auto",
        vmin=-80,
        vmax=0,
    )
    axis.set_ylim(
        LOW_FREQ_HZ / 1000,
        min(HIGH_FREQ_HZ, sr / 2) / 1000,
    )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Frequency (kHz)")

    # 标注每个检测到的叫声区
    call_index = 0
    for row in pulse_rows:
        if str(row.get("predicted_label", "")) == "noise":
            continue
        call_index += 1
        start_s = float(row.get("start_s", 0.0))
        end_s = float(row.get("end_s", start_s))
        label = str(row.get("predicted_label", ""))
        confidence = row.get("confidence")
        pulse_id = str(row.get("pulse_id", call_index))

        axis.axvspan(start_s, end_s, color="red", alpha=0.18)
        axis.axvline(start_s, color="red", linewidth=0.8, alpha=0.7)
        axis.axvline(end_s, color="red", linewidth=0.8, alpha=0.7)

        tag = f"[{label}]" if label else f"[{pulse_id}]"
        if confidence is not None:
            try:
                tag += f" {float(confidence):.2f}"
            except (TypeError, ValueError):
                pass
        axis.text(
            (start_s + end_s) / 2,
            0.97,
            tag,
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            color="red",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                alpha=0.7,
                edgecolor="red",
            ),
        )

    if title is None:
        title = (
            f"{audio_name} | calls: {call_index}"
        )
    axis.set_title(title)

    figure.colorbar(mesh, ax=axis, label="Relative power (dB)")
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return True
