"""① 超长 WAV 流式分析引擎（segment_analysis 的长录音后端）。

设计原则（本轮优化的 A/B/C/D/E/Resume 都在这里落地）：

- **内存与文件总长无关**：任何时刻只驻留 1~2 个 chunk 的音频数组；
  禁止整条 `librosa.load`，禁止整条 STFT，禁止长期保存 pulse waveform。
- **chunk + overlap**：真实 pulse 可能跨边界，读块带重叠余量（覆盖
  最大脉冲时长/padding/refinement 上下文/滤波瞬态），脉冲时间统一转
  **global 绝对时间**（chunk_start + local）。
- **不漏不重**：脉冲按“中心时间落在哪个 chunk 的 coverage 内”归属，
  相邻 chunk 的跨块重复再用 `remove_duplicate_boundary_pulses` 的规则
  （区间重叠 ≥0.50 或中心距 ≤1.5 ms，保留高 snr）兜底去重。
- **全量统计（FULL RECORDING）**：coverage 连续铺满 0→总时长，
  不做 MAX_ANALYSIS_WINDOWS 抽样；静音快筛只是计算优化，
  **不改变统计口径**（分母永远是有效监测总时长）。
- **保守快筛（recall 优先）**：只跳过“明显没有任何可能的超声事件”的块；
  宁可多留空 chunk，绝不误删真实叫声。
- **有限并行**：bounded worker pool（默认 2，0=auto→min(4, cpu)），
  结果按 chunk 序归并，计数与单线程完全一致。
- **长任务可操作**：增量落盘 + checkpoint + 断点续跑（校验文件身份），
  stop 在 chunk 边界优雅退出（绝不强杀线程）。

分类模型/特征/阈值完全不变：本模块只决定“音频怎么送进现有算法”。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .audio_io import highpass_filter
from .config import (
    CORE_CONTEXT_MS,
    CORE_MAX_GAP_MS,
    CORE_PADDING_MS,
    MAX_PULSE_MS,
    MERGE_GAP_MS,
    PULSE_PADDING_MS,
    SEGMENT_CHECKPOINT_EVERY_CHUNKS,
    SEGMENT_FASTSKIP_ABS_CAP,
    SEGMENT_FASTSKIP_ENABLED,
    SEGMENT_FASTSKIP_MARGIN_DB,
    SEGMENT_FASTSKIP_REL_FACTOR,
    SEGMENT_STREAM_CHUNK_SECONDS,
    SEGMENT_STREAM_MAX_WORKERS,
    SEGMENT_STREAM_OVERLAP_SECONDS,
    TARGET_SAMPLE_RATE,
)

# 跨块重复判定（与 detection.remove_duplicate_boundary_pulses 同规则）
DEDUPE_OVERLAP_RATIO = 0.50
DEDUPE_CENTER_DISTANCE_MS = 1.5

STATE_FILENAME = "run_state.json"
PARTIAL_CSV_FILENAME = "pulse_results.partial.csv"


# ---------------------------------------------------------------------------
# overlap / chunk 网格
# ---------------------------------------------------------------------------
def minimum_overlap_seconds() -> float:
    """由现有 pulse/padding/refinement 参数推导的最小重叠（秒）。

    覆盖：最大脉冲时长 + 候选 padding×2 + 精修上下文×2 + 精修 padding×2
    + 合并间隔 + 滤波/重采样瞬态余量。绝不随意猜一个小值。
    """
    required_ms = (
        MAX_PULSE_MS
        + 2 * PULSE_PADDING_MS
        + 2 * CORE_CONTEXT_MS
        + 2 * CORE_PADDING_MS
        + CORE_MAX_GAP_MS
        + MERGE_GAP_MS
    )
    transient_s = 0.25  # sosfiltfilt / 重采样核的边界瞬态余量
    return required_ms / 1000.0 + transient_s


@dataclass(frozen=True)
class ChunkSpec:
    """一个 chunk 的时间范围（秒，global 域）。

    coverage：本块“负责”的连续区间（各块 coverage 拼起来 = 整条录音）；
    read：实际读取区间 = coverage ± overlap（pulse 跨边界时有完整上下文）。
    """

    index: int
    coverage_start_s: float
    coverage_end_s: float
    read_start_s: float
    read_end_s: float



def build_chunk_specs(
    total_samples: int,
    sr: int,
    chunk_seconds: float | None = None,
    overlap_seconds: float | None = None,
) -> list[ChunkSpec]:
    """把整条录音切成 chunk + overlap 的网格（coverage 连续铺满全长）。"""
    chunk_seconds = float(
        SEGMENT_STREAM_CHUNK_SECONDS if chunk_seconds is None else chunk_seconds
    )
    overlap_seconds = float(
        SEGMENT_STREAM_OVERLAP_SECONDS
        if overlap_seconds is None
        else overlap_seconds
    )
    if chunk_seconds <= 0:
        raise ValueError("SEGMENT_STREAM_CHUNK_SECONDS 必须 > 0")
    min_overlap = minimum_overlap_seconds()
    if overlap_seconds < min_overlap:
        raise ValueError(
            f"SEGMENT_STREAM_OVERLAP_SECONDS={overlap_seconds}s 不足："
            f"覆盖跨边界 pulse 至少需要 {min_overlap:.3f}s"
            f"（最大脉冲 {MAX_PULSE_MS}ms + padding/context + 滤波瞬态）。"
        )

    total_s = float(total_samples) / float(sr)
    if total_s <= 0:
        return []

    specs: list[ChunkSpec] = []
    index = 0
    start = 0.0
    while start < total_s - 1e-9:
        end = min(start + chunk_seconds, total_s)
        specs.append(
            ChunkSpec(
                index=index,
                coverage_start_s=start,
                coverage_end_s=end,
                read_start_s=max(0.0, start - overlap_seconds),
                read_end_s=min(total_s, end + overlap_seconds),
            )
        )
        index += 1
        start = end
    return specs


def probe_audio(path: Path) -> tuple[int, int, float]:
    """用 soundfile 探测 (total_frames, sr, duration_s)，失败给明确报错。"""
    import soundfile as sf

    try:
        info = sf.info(str(path))
    except Exception as error:  # noqa: BLE001
        raise ValueError(
            f"无法读取 WAV（可能损坏或为不支持的格式，"
            f"超过 4GB 的 RIFF WAV 请转成 RF64/W64）：{path}：{error}"
        ) from error
    if info.frames <= 0:
        raise ValueError(f"WAV 为空：{path}")
    return int(info.frames), int(info.samplerate), float(info.frames) / float(info.samplerate)


# ---------------------------------------------------------------------------
# 流式预处理（与 audio_io.load_audio 同口径：重采样→去均值→高通→峰值归一）
# ---------------------------------------------------------------------------
@dataclass
class StreamStats:
    """一遍扫描得到的全局统计（决定预处理口径，与 load_audio 一致）。"""

    mean: float = 0.0
    peak: float = 0.0
    block_peak_median: float = 0.0   # 各块 1ms 包络峰值的中位数（快筛参考）
    block_count: int = 0


class StreamingPreprocessor:
    """按块读取 + 预处理；绝不整条驻留内存。"""

    def __init__(
        self,
        wav_path: Path,
        total_frames: int,
        orig_sr: int,
        stats: StreamStats,
    ) -> None:
        self._wav_path = Path(wav_path)
        self._total_frames = int(total_frames)
        self._orig_sr = int(orig_sr)
        self._stats = stats

    @property
    def orig_sr(self) -> int:
        return self._orig_sr

    @property
    def target_sr(self) -> int:
        return TARGET_SAMPLE_RATE

    def _read_raw(self, start_s: float, end_s: float) -> np.ndarray:
        import soundfile as sf

        start_frame = max(0, int(round(start_s * self._orig_sr)))
        end_frame = min(self._total_frames, int(round(end_s * self._orig_sr)))
        frames = max(0, end_frame - start_frame)
        if frames == 0:
            return np.zeros(0, dtype=np.float32)
        with sf.SoundFile(str(self._wav_path), "r") as handle:
            handle.seek(start_frame)
            data = handle.read(frames, dtype="float32", always_2d=False)
        return np.asarray(data, dtype=np.float32)

    def _resample(self, raw: np.ndarray) -> np.ndarray:
        if raw.size == 0:
            return np.zeros(0, dtype=np.float64)
        if self._orig_sr == TARGET_SAMPLE_RATE:
            return raw.astype(np.float64)
        import librosa

        return np.asarray(
            librosa.resample(
                raw,
                orig_sr=self._orig_sr,
                target_sr=TARGET_SAMPLE_RATE,
            ),
            dtype=np.float64,
        )

    def compute_stats(self, block_seconds: float = 30.0) -> StreamStats:
        """两遍流式扫描：均值（重采样域）→ 峰值（去均值+高通后）。

        内存 O(block)，与文件总长无关。
        """
        import soundfile as sf

        # Pass 1：重采样域均值
        total_sum = 0.0
        total_count = 0
        with sf.SoundFile(str(self._wav_path), "r") as handle:
            while True:
                raw = handle.read(
                    int(block_seconds * self._orig_sr),
                    dtype="float32",
                    always_2d=False,
                )
                if len(raw) == 0:
                    break
                block = self._resample(np.asarray(raw, dtype=np.float32))
                total_sum += float(np.sum(block))
                total_count += int(block.size)
        mean = total_sum / max(total_count, 1)

        # Pass 2：去均值 + 高通后的峰值（interior，忽略块边缘瞬态）
        peak = 0.0
        block_peaks: list[float] = []
        margin_s = 0.25
        start_s = 0.0
        total_s = self._total_frames / float(self._orig_sr)
        while start_s < total_s - 1e-9:
            end_s = min(start_s + block_seconds, total_s)
            raw = self._read_raw(max(0.0, start_s - margin_s),
                                 min(total_s, end_s + margin_s))
            block = self._resample(raw) - mean
            block = highpass_filter(block, TARGET_SAMPLE_RATE)
            if block.size == 0:
                start_s = end_s
                continue
            margin = int(round(margin_s * TARGET_SAMPLE_RATE))
            interior = block[margin: block.size - margin] if block.size > 2 * margin else block
            if interior.size:
                peak = max(peak, float(np.max(np.abs(interior))))
                block_peaks.append(float(np.max(np.abs(interior))))
            start_s = end_s

        self._stats = StreamStats(
            mean=mean,
            peak=peak if peak > 0 else 1.0,
            block_peak_median=float(np.median(block_peaks)) if block_peaks else 0.0,
            block_count=len(block_peaks),
        )
        return self._stats

    def block(self, read_start_s: float, read_end_s: float) -> np.ndarray:
        """读取并预处理 [read_start_s, read_end_s)（global 秒域）。"""
        raw = self._read_raw(read_start_s, read_end_s)
        block = self._resample(raw) - self._stats.mean
        block = highpass_filter(block, TARGET_SAMPLE_RATE)
        if self._stats.peak > 0:
            block = block / self._stats.peak
        return block


# ---------------------------------------------------------------------------
# 静音/无效 chunk 快筛（保守，recall 优先）
# ---------------------------------------------------------------------------
def chunk_may_contain_ultrasound(
    block: np.ndarray,
    sr: int,
    stats: StreamStats | None = None,
) -> bool:
    """快速判断“这个 chunk 是否明显没有任何可能的超声事件”。

    只有**同时**满足下面三条才允许跳过昂贵的 STFT/精修/特征/推理：

    1. 块内 1ms 包络峰值 < 绝对上限（默认 -34 dBFS，相对归一化满幅）；
    2. 包络峰值相对块内背景中位数不足 SEGMENT_FASTSKIP_MARGIN_DB
       （默认 +6 dB，远低于检测器自身的 +10 dB 门限——
       只要检测器有可能看见的事件，这里绝不跳过）；
    3. 包络峰值不超过整条录音“典型块”峰值的 SEGMENT_FASTSKIP_REL_FACTOR 倍。

    即：宁可多留空 chunk，也绝不误删真实蝙蝠叫声。
    """
    if not SEGMENT_FASTSKIP_ENABLED:
        return True
    if block.size == 0:
        return True

    frame_len = max(1, int(round(sr * 0.001)))  # 1 ms 帧
    n_frames = int(block.size // frame_len)
    if n_frames <= 0:
        return True
    frames = np.abs(block[: n_frames * frame_len]).reshape(n_frames, frame_len)
    envelope = frames.max(axis=1)
    peak = float(envelope.max())
    if peak <= 0.0:
        return False

    floor = float(np.median(envelope))
    margin_ratio = 10.0 ** (SEGMENT_FASTSKIP_MARGIN_DB / 20.0)
    if peak >= SEGMENT_FASTSKIP_ABS_CAP:
        return True          # 绝对幅度不低：照常处理
    if peak >= margin_ratio * max(floor, 1e-12):
        return True          # 包络有峰状结构：照常处理

    # 第三道保险：只有比整条录音的典型块还安静得多的块才有资格跳过。
    # 三个条件同时满足才跳过（recall 优先，越保守越安全）。
    if stats is not None and stats.block_peak_median > 0:
        if peak >= SEGMENT_FASTSKIP_REL_FACTOR * stats.block_peak_median:
            return True
    return False


# ---------------------------------------------------------------------------
# 跨块去重（扩展 detection.remove_duplicate_boundary_pulses 的规则）
# ---------------------------------------------------------------------------
def _interval_overlap_ratio(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    overlap = min(a_end, b_end) - max(a_start, b_start)
    if overlap <= 0:
        return 0.0
    shortest = min(a_end - a_start, b_end - b_start)
    return float(overlap / shortest) if shortest > 0 else 0.0


def _same_pulse(a: dict, b: dict) -> bool:
    overlap_ratio = _interval_overlap_ratio(
        float(a["start_s"]), float(a["end_s"]),
        float(b["start_s"]), float(b["end_s"]),
    )
    center_distance_ms = abs(
        (float(a["start_s"]) + float(a["end_s"])) / 2
        - (float(b["start_s"]) + float(b["end_s"])) / 2
    ) * 1000.0
    return overlap_ratio >= DEDUPE_OVERLAP_RATIO or center_distance_ms <= DEDUPE_CENTER_DISTANCE_MS


def dedupe_cross_chunk_rows(
    previous_rows: list[dict],
    new_rows: list[dict],
) -> list[dict]:
    """跨 chunk 边界去重：同一真实 pulse 只能计数一次。

    只在**不同 chunk** 的行之间去重（同 chunk 内同一候选可能合法地拆成
    多条同时发声轨迹行，绝不能互并）。判定规则与
    detection.remove_duplicate_boundary_pulses 一致：区间重叠 ≥0.50 或
    中心距 ≤1.5ms，保留 snr_db 高的一条。
    """
    kept: list[dict] = []
    for row in new_rows:
        duplicate = None
        for prev in previous_rows:
            if prev.get("_chunk_index") == row.get("_chunk_index"):
                continue
            if _same_pulse(prev, row):
                duplicate = prev
                break
        if duplicate is None:
            kept.append(row)
            continue
        prev_snr = float(duplicate.get("snr_db", -999.0))
        row_snr = float(row.get("snr_db", -999.0))
        if row_snr > prev_snr:
            duplicate.update({k: v for k, v in row.items() if k != "_chunk_index"})
    return kept


# ---------------------------------------------------------------------------
# checkpoint / 断点续跑
# ---------------------------------------------------------------------------
def file_identity(path: Path) -> dict:
    """文件身份：path/size/mtime + 头尾 1MB 内容哈希（防换文件错误续跑）。"""
    path = Path(path)
    stat = path.stat()
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read(1024 * 1024))
        if stat.st_size > 2 * 1024 * 1024:
            handle.seek(-1024 * 1024, os.SEEK_END)
            digest.update(handle.read(1024 * 1024))
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime": round(stat.st_mtime, 3),
        "content_hash": digest.hexdigest(),
    }


class ChunkCheckpoint:
    """增量落盘 + 断点续跑状态（每录音一个目录）。"""

    def __init__(self, directory: Path, wav_path: Path) -> None:
        self.directory = Path(directory)
        self.wav_path = Path(wav_path)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.directory / STATE_FILENAME
        self.partial_csv_path = self.directory / PARTIAL_CSV_FILENAME
        self._lock = threading.Lock()
        self._csv_header_written = self.partial_csv_path.exists()

    # ---- 身份校验 ----
    def load_resumable(self) -> dict | None:
        if not self.state_path.exists():
            return None
        try:
            state = json.loads(
                self.state_path.read_text(encoding="utf-8")
            )
        except Exception:  # noqa: BLE001
            return None
        if state.get("status") not in ("in_progress", "cancelled", "failed"):
            return None
        identity = state.get("file_identity") or {}
        current = file_identity(self.wav_path)
        for key in ("size", "content_hash"):
            if identity.get(key) != current.get(key):
                return None
        return state

    def save(self, state: dict) -> None:
        with self._lock:
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(state, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp, self.state_path)

    def append_rows(self, rows: list[dict], columns: list[str] | None = None) -> None:
        """增量写行（partial CSV）。最终由调用方原子替换为正式文件。"""
        if not rows:
            return
        import csv

        with self._lock:
            new_file = not self.partial_csv_path.exists()
            with open(self.partial_csv_path, "a", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=columns or sorted(rows[0].keys()),
                    extrasaction="ignore",
                )
                if new_file:
                    writer.writeheader()
                for row in rows:
                    writer.writerow(row)

    def promote_partial_csv(self, destination: Path) -> None:
        with self._lock:
            if self.partial_csv_path.exists():
                os.replace(self.partial_csv_path, Path(destination))

    def clear(self) -> None:
        with self._lock:
            for path in (self.state_path, self.partial_csv_path):
                if path.exists():
                    path.unlink()


def find_resumable_checkpoints(run_dir: Path) -> list[Path]:
    """查找 run 目录下可续跑的 checkpoint（UI 断点续跑入口用）。"""
    run_dir = Path(run_dir)
    found: list[Path] = []
    if not run_dir.exists():
        return found
    for state_path in sorted(run_dir.glob(f"**/{STATE_FILENAME}")):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if state.get("status") in ("in_progress", "cancelled", "failed"):
            found.append(state_path)
    return found


# ---------------------------------------------------------------------------
# 单 chunk 处理（read → 预处理 → 检测 → 精修 → 建表 → 归属筛选）
# ---------------------------------------------------------------------------
@dataclass
class ChunkContext:
    wav_path: Path
    preprocessor: StreamingPreprocessor
    stats: StreamStats
    total_samples: int
    separated_dir: Path | None = None
    save_separated_audio: bool = True
    template_matching: bool = False
    separation_window_mode: str = "fixed"
    harmonic_grouping_mode: str = "basic"
    retain_component_audio: bool = False


@dataclass
class ChunkResult:
    spec: ChunkSpec
    rows: list[dict] = field(default_factory=list)   # 已转 global 时间
    raw_candidate_count: int = 0
    threshold_db: float = float("nan")
    skipped: bool = False
    error: str = ""
    read_y: np.ndarray | None = None   # 供导出 WAV 用，导出后释放
    component_audio: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)


def process_chunk(spec: ChunkSpec, context: ChunkContext) -> ChunkResult:
    """处理一个 chunk：任何时刻只持有本块音频。"""
    from .detection import detect_pulses, refine_all_pulses
    from .pulse_table import create_pulse_rows

    read_y = context.preprocessor.block(spec.read_start_s, spec.read_end_s)
    result = ChunkResult(spec=spec)
    if read_y.size == 0:
        result.skipped = True
        return result

    sr = TARGET_SAMPLE_RATE
    if not chunk_may_contain_ultrasound(read_y, sr, context.stats):
        result.skipped = True
        return result

    raw_pulses, _power, _freqs, _times, threshold_db = detect_pulses(read_y, sr)
    refined = refine_all_pulses(read_y, sr, raw_pulses)
    result.raw_candidate_count = len(raw_pulses)
    result.threshold_db = float(threshold_db)

    # ---- 归属：中心时间落在本块 coverage 内的 pulse 才归本块 ----
    owned = []
    for pulse in refined:
        center = (float(pulse["start_s"]) + float(pulse["end_s"])) / 2.0
        global_center = center + spec.read_start_s
        if spec.coverage_start_s <= global_center < spec.coverage_end_s:
            owned.append(pulse)
    if not owned:
        return result

    rows = create_pulse_rows(
        context.wav_path,
        read_y,
        sr,
        owned,
        separated_dir=context.separated_dir,
        max_pulses=None,          # 计数绝不被截断（导出上限另算）
        save_separated_audio=context.save_separated_audio,
        include_template_track=context.template_matching,
        separation_window_mode=context.separation_window_mode,
        harmonic_grouping_mode=context.harmonic_grouping_mode,
        component_audio_sink=result.component_audio if context.retain_component_audio else None,
    )
    # 块内 local 时间 → global 绝对时间；事件编号包含录音身份，跨录音/块不冲突。
    recording_key = hashlib.sha256(str(context.wav_path.resolve()).encode("utf-8")).hexdigest()[:12]
    for row in rows:
        row.pop("accepted", None)
        row.pop("rejection_reason", None)
        for key in ("candidate_start_s", "candidate_end_s", "start_s", "end_s"):
            if key in row:
                row[key] = float(row[key]) + spec.read_start_s
        row["_chunk_index"] = spec.index
        row["event_id"] = f"{recording_key}_chunk{spec.index:06d}_candidate{int(row['candidate_id']):06d}"

    result.rows = rows
    result.read_y = read_y
    return result


# ---------------------------------------------------------------------------
# 流式编排（顺序/并行 + 断点续跑 + 优雅停止 + 进度）
# ---------------------------------------------------------------------------
def resolve_max_workers(requested: int | None = None) -> int:
    """有限并行：绝不默认开满 os.cpu_count()。"""
    if requested is not None and int(requested) > 0:
        return int(requested)
    if SEGMENT_STREAM_MAX_WORKERS and int(SEGMENT_STREAM_MAX_WORKERS) > 0:
        return int(SEGMENT_STREAM_MAX_WORKERS)
    cpu = os.cpu_count() or 1
    return max(1, min(4, cpu // 2 if cpu > 2 else 1))


def stream_chunks(
    context: ChunkContext,
    specs: list[ChunkSpec],
    *,
    max_workers: int | None = None,
    progress: Callable[[dict], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    on_batch: Callable[[ChunkResult], None] | None = None,
) -> Iterator[ChunkResult]:
    """按 chunk 序产出结果（即使并行执行，也按 global 时间序归并）。

    - bounded worker pool（max_workers，默认 2）；
    - should_stop 在 chunk 边界生效（优雅停止，绝不强杀）；
    - progress 每块回调一次。
    """
    workers = resolve_max_workers(max_workers)
    stop_requested = False

    if workers <= 1:
        for spec in specs:
            if should_stop is not None and should_stop():
                stop_requested = True
                break
            result = process_chunk(spec, context)
            if on_batch is not None:
                on_batch(result)
            if progress is not None:
                progress(_progress_payload(result, specs))
            yield result
        return

    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        pending: dict = {}
        next_submit = 0
        next_yield = 0
        lookahead = workers * 2
        while next_yield < len(specs):
            while (
                next_submit < len(specs)
                and next_submit < next_yield + lookahead
                and not stop_requested
            ):
                if should_stop is not None and should_stop():
                    stop_requested = True
                    break
                spec = specs[next_submit]
                pending[next_submit] = executor.submit(process_chunk, spec, context)
                next_submit += 1
            if next_yield not in pending:
                break
            result = pending.pop(next_yield).result()
            next_yield += 1
            if on_batch is not None:
                on_batch(result)
            if progress is not None:
                progress(_progress_payload(result, specs))
            yield result
    finally:
        for future in pending.values():
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def _progress_payload(result: ChunkResult, specs: list[ChunkSpec]) -> dict:
    return {
        "chunk_index": result.spec.index,
        "chunk_count": len(specs),
        "processed_until_s": result.spec.coverage_end_s,
        "coverage_start_s": result.spec.coverage_start_s,
        "skipped": bool(result.skipped),
        "raw_candidate_count": int(result.raw_candidate_count),
    }


# ---------------------------------------------------------------------------
# 进程内存（尽力而为；UI 显示用，不参与统计）
# ---------------------------------------------------------------------------
def current_rss_mb() -> float:
    try:
        import ctypes

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_uint32),
                ("PageFaultCount", ctypes.c_uint32),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.c_void_p(handle), ctypes.byref(pmc), pmc.cb
        )
        if ok:
            return round(pmc.WorkingSetSize / 1e6, 1)
    except Exception:  # noqa: BLE001 - 非 Windows/失败则退回 resource
        pass
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(usage / 1024.0, 1)  # Linux: KB
    except Exception:  # noqa: BLE001
        return -1.0


# ---------------------------------------------------------------------------
# 流式低分辨率 overview（§15：只用于可视化，绝不参与 call 统计）
# ---------------------------------------------------------------------------
class SpectrogramSummary:
    """流式累计 overview 声谱摘要（严格限制 time bins，禁止整条 STFT）。"""

    def __init__(
        self,
        total_duration_s: float,
        sr: int = TARGET_SAMPLE_RATE,
        max_time_bins: int = 4000,
        n_fft: int = 1024,
    ) -> None:
        self.sr = int(sr)
        self.n_fft = int(n_fft)
        hop = int(np.ceil(max(total_duration_s, 1e-6) * self.sr / max_time_bins))
        self.hop_length = max(self.n_fft // 4, hop)
        self.n_frames = int(
            np.ceil(max(total_duration_s, 1e-6) * self.sr / self.hop_length)
        ) + 1
        self._power: np.ndarray | None = None

    def add(self, block: np.ndarray, read_start_s: float) -> None:
        import librosa

        if block is None or len(block) < self.n_fft:
            return
        stft = librosa.stft(
            np.asarray(block, dtype=np.float64),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window="hann",
            center=True,
        )
        power = np.abs(stft) ** 2
        if self._power is None:
            self._power = np.zeros(
                (power.shape[0], self.n_frames), dtype=np.float32
            )
        frame_times = read_start_s + librosa.frames_to_time(
            np.arange(power.shape[1]), sr=self.sr, hop_length=self.hop_length
        )
        indices = np.round(frame_times * self.sr / self.hop_length).astype(int)
        valid = (indices >= 0) & (indices < self.n_frames)
        for column, index in zip(np.flatnonzero(valid), indices[valid]):
            np.maximum(
                self._power[:, index], power[:, column].astype(np.float32),
                out=self._power[:, index],
            )

    @property
    def power(self) -> np.ndarray | None:
        return self._power

    @property
    def frequencies(self) -> np.ndarray:
        import librosa

        return librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.n_frames) * self.hop_length / float(self.sr)
