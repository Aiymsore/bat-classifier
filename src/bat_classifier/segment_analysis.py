"""① 流式模板匹配与逐 pulse/片段统计；统计声学叫声数，不代表个体数。"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .audio_io import find_wav_files
from .morphology_gate import GATE_COLUMNS, GATE_VERSION, unavailable_gate
from .sequence_analysis import SEQUENCE_COLUMNS, SequenceConfig
from .boundary_reliability import BOUNDARY_COLUMNS, assess_boundaries
from .config import (
    MAX_PULSE_MS,
    SEGMENT_ANALYSIS_DIR,
    SEGMENT_FIGURE_SUBDIR,
    SEGMENT_MAX_INMEMORY_ROWS,
    SEGMENT_MAX_PULSE_WAV_EXPORTS,
    SEGMENT_MIN_CONFIDENCE,
    SEGMENT_MIXED_PURITY_THRESHOLD,
    SEGMENT_PULSE_SUBDIR,
    SEGMENT_SEPARATED_SUBDIR,
    SEGMENT_STREAM_CHUNK_SECONDS,
    SEGMENT_STREAM_OVERLAP_SECONDS,
    TARGET_SAMPLE_RATE,
    TEMPLATE_CF_MIN_PLATEAU_MS,
)
from .template_matching import (
    TEMPLATE_FAMILY_RESULT_COLUMNS,
    PreparedTemplateModel,
    classify_template_rows,
    describe_template_model,
    template_detection_bands,
    is_template_model,
    prepare_template_model,
)


# 断点格式与计数口径版本：改动门控、模板放行或导出语义时递增，旧断点必须重跑。
PULSE_PROCESSING_VERSION = 15  # 门控复用原混合谱支持的完整轨迹，旧断点重跑


def _finalize_segment_sequences(accumulator, checkpoint, wav_path, config, pulse_spectrogram_root=None):
    """Finish sequences across chunk boundaries before publishing final counts.

    Read all metadata from the disk ledger, never just the capped preview rows.
    Audio remains local/chunked. The atomic replacement is safe to reapply after
    an interrupted finalization because single-call labels are preserved.
    """
    from .sequence_analysis import SequenceConfig, classify_sequence_rows
    path = checkpoint.partial_csv_path if checkpoint is not None else None
    if path is not None and path.exists() and path.stat().st_size:
        frame = pd.read_csv(path, encoding='utf-8-sig', keep_default_na=False)
    else:
        frame = pd.DataFrame(accumulator.rows)
    if len(frame) != accumulator.rows_total:
        raise ValueError('连串判族需要完整声波记录；请提供输出目录以保存完整记录。')
    if frame.empty:
        return 0
    frame['segment_file'] = wav_path.name
    # CSV replay may contain blanks in unavailable fields.
    for col in ('start_s', 'end_s', 'cf_plateau_khz', 'cf_plateau_ms',
                'template_cffm_score', 'confidence', 'single_confidence'):
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors='coerce')
    # Restore original gates before auditing a replayed ledger.
    if 'single_gate_status' in frame:
        frame['gate_status'] = frame.single_gate_status
    frame = assess_boundaries(frame, wav_path)
    final, sequences = classify_sequence_rows(frame, SequenceConfig(
        plateau_threshold_ms=config.cf_min_plateau_ms,
        min_similarity=config.min_confidence))
    records = final.to_dict('records')
    if pulse_spectrogram_root is not None and final.sequence_changed.any():
        _refresh_sequence_images(records, wav_path, config, pulse_spectrogram_root)
        final = pd.DataFrame(records)
    # Duration review is a final admission decision, so reconcile all gate
    # counts as well as labels. Unknown remains a counted bat as before.
    accumulator.valid_call_count = int(final.gate_status.eq('bat').sum())
    accumulator.bat_call_count = accumulator.valid_call_count
    accumulator.gate_uncertain_count = int(final.gate_status.eq('uncertain').sum())
    accumulator.noise_count = int(final.gate_status.eq('noise').sum())
    accumulator.template_promoted_count = int(final.gate_promoted.eq(True).sum())
    accumulator.label_counts = _aggregate_label_counts(
        [str(r['predicted_label']) for r in records if r['gate_status'] == 'bat'],
        accumulator.labels, accumulator.uncertain_label)
    scored = final[final.gate_status.eq('bat') & ~final.predicted_label.eq(accumulator.uncertain_label)]
    accumulator.confidence_sum = float(scored.confidence.sum())
    accumulator.confidence_n = int(scored.confidence.notna().sum())
    if path is not None:
        temporary = path.with_suffix('.sequence.tmp.csv')
        final.to_csv(temporary, index=False, encoding='utf-8-sig')
        temporary.replace(path)
        sequences.to_csv(path.parent.parent / 'sequence_summary.csv', index=False, encoding='utf-8-sig')
    max_rows = int(SEGMENT_MAX_INMEMORY_ROWS)
    accumulator.rows = records[:max_rows] if max_rows > 0 else records
    accumulator.rows_truncated = len(accumulator.rows) < len(records)
    return len({str(r.get('pulse_spectrogram_file') or '') for r in records} - {''})


def _refresh_sequence_images(rows, wav_path, config, root):
    """Regenerate changed event images using saved component WAVs/local audio."""
    from .audio_io import load_audio_interval
    from .config import BASE_DIR
    import soundfile as sf
    root = Path(root).resolve()
    changed_paths = {str(r.get('pulse_spectrogram_file') or '') for r in rows
                     if r.get('sequence_changed')}
    for old_text in changed_paths - {''}:
        group = [r for r in rows if str(r.get('pulse_spectrogram_file') or '') == old_text]
        old = Path(old_text)
        if not old.is_absolute():
            old = root.parent / old  # 图片路径相对于本次 run，而非项目根目录
        old = old.resolve()
        if not old.is_relative_to(root):
            raise ValueError('声谱图路径超出本次输出目录')
        enabled = []
        for r in group:
            requested = (config.save_uncertain_spectrograms if r.get('gate_status') == 'uncertain'
                else config.save_unknown_spectrograms if r.get('predicted_label') == config.template_unknown_label
                else config.save_pulse_spectrograms)
            if requested:
                enabled.append(r)
            else:
                for key in ('pulse_spectrogram_file', 'merged_spectrogram_file', 'component_spectrogram_file'):
                    r[key] = ''
                r['pulse_spectrogram_status'] = 'not_requested_after_sequence'
        if not enabled:
            old.unlink(missing_ok=True)
            old.with_suffix('.json').unlink(missing_ok=True)
            continue
        start = max(0., min(float(r.get('candidate_start_s') or r['start_s']) for r in group)
                    - config.pulse_spectrogram_context_ms / 1000)
        end = max(float(r.get('candidate_end_s') or r['end_s']) for r in group) + config.pulse_spectrogram_context_ms / 1000
        audio, sr, offset = load_audio_interval(wav_path, start, end)
        components = {}
        for r in group:
            file = str(r.get('pulse_file') or '')
            if file:
                p = Path(file)
                if not p.is_absolute():
                    p = BASE_DIR / p
                if p.exists():
                    y, _ = sf.read(p)
                    components[(int(r['candidate_id']), int(r['component_id']))] = y
        count = _export_batch_pulse_spectrograms(group, segment_id=wav_path.stem,
            spectrogram_root=root, read_y=audio, read_start_s=offset,
            context_ms=config.pulse_spectrogram_context_ms, mode=config.pulse_spectrogram_mode,
            component_audio=components, save_regular=config.save_pulse_spectrograms,
            save_uncertain=config.save_uncertain_spectrograms, save_unknown=config.save_unknown_spectrograms,
            unknown_label=config.template_unknown_label)
        if not count:
            raise RuntimeError('连串改类后的声谱图生成失败')
        new_paths = {str(r.get('pulse_spectrogram_file') or '') for r in group}
        if old_text not in new_paths:
            old.unlink(missing_ok=True)
            old.with_suffix('.json').unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass
class SegmentAnalysisConfig:
    """片段识别与叫声统计的可调参数（UI 可覆盖默认值）。"""

    min_confidence: float = SEGMENT_MIN_CONFIDENCE
    # 是否把每个 pulse 切成 WAV（segments/<id>/<id>__pNNN.wav）
    save_pulse_audio: bool = True
    # 是否生成片段声谱图（figures/<id>.png）
    save_figures: bool = True
    # 是否为每个分类结果（含 unknown）生成独立声谱图
    save_pulse_spectrograms: bool = True
    # 独立调试开关：常规图片关闭时，也可仅输出门控 uncertain 图片。
    save_uncertain_spectrograms: bool = False
    # unknown（模板未匹配）的独立图片开关：默认不输出，避免与已分类叫声混在一起。
    save_unknown_spectrograms: bool = False
    # 实验：仅缺能量证据时，允许模板相似度达标的待定声波按模板放行（默认关闭）。
    allow_template_promotion: bool = False
    # CF 与 CF-FM 族同时达标时，横频平台不短于该值（ms）判 CF，否则判 CF-FM。
    cf_min_plateau_ms: float = TEMPLATE_CF_MIN_PLATEAU_MS
    # 诊断：逐块输出全带/子带的能量对比度与检测阈值（不影响任何结果，默认关闭）。
    save_detection_diagnostics: bool = False
    # 模板子带检测：在全带检测之外，按模板频率轨迹的子带再独立检测一次，
    # 找回被宽带底噪稀释的弱叫声（默认开启；计数会依赖模板库的频带）。
    template_band_detection: bool = True
    # merged: 同一候选事件共图；separate: 每个独立分量一图（谐波随分量）。
    pulse_spectrogram_mode: str = "merged"
    # 同时分量分离窗：fixed 保持原 256 点；adaptive 对较长 pulse 延长到 512/1024 点。
    separation_window_mode: str = "fixed"
    # 谐波判定：basic 仅使用旧的 2/3 倍率；enhanced 加入逐帧轨迹和包络一致性。
    harmonic_grouping_mode: str = "enhanced"
    # 独立声谱图在叫声前后保留的上下文（毫秒）。默认严格按 pulse 边界，
    # 避免把相邻叫声混进当前 pulse；需要背景时再显式调大这个参数。
    pulse_spectrogram_context_ms: float = 0.0
    mixed_purity_threshold: float = SEGMENT_MIXED_PURITY_THRESHOLD
    # 低于最低相似度时的未知类别名称。
    template_unknown_label: str = "unknown"

    def validate(self) -> "SegmentAnalysisConfig":
        self.min_confidence = float(self.min_confidence)
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError("最低相似度必须在 0~1 之间。")
        self.save_pulse_audio = bool(self.save_pulse_audio)
        self.save_figures = bool(self.save_figures)
        self.save_pulse_spectrograms = bool(self.save_pulse_spectrograms)
        self.save_uncertain_spectrograms = bool(self.save_uncertain_spectrograms)
        self.save_unknown_spectrograms = bool(self.save_unknown_spectrograms)
        self.allow_template_promotion = bool(self.allow_template_promotion)
        self.cf_min_plateau_ms = float(self.cf_min_plateau_ms)
        if not np.isfinite(self.cf_min_plateau_ms) or self.cf_min_plateau_ms < 0.0:
            raise ValueError("CF 最短平台时长不能小于 0 毫秒。")
        self.save_detection_diagnostics = bool(self.save_detection_diagnostics)
        self.template_band_detection = bool(self.template_band_detection)
        if self.pulse_spectrogram_mode not in {"merged", "separate"}:
            raise ValueError("声谱图输出方式必须是 merged 或 separate。")
        self.separation_window_mode = str(self.separation_window_mode).strip().lower()
        if self.separation_window_mode not in {"fixed", "adaptive"}:
            raise ValueError("分离窗模式必须是 fixed 或 adaptive。")
        self.harmonic_grouping_mode = str(self.harmonic_grouping_mode).strip().lower()
        if self.harmonic_grouping_mode not in {"basic", "enhanced"}:
            raise ValueError("谐波分组模式必须是 basic 或 enhanced。")
        self.pulse_spectrogram_context_ms = float(self.pulse_spectrogram_context_ms)
        if self.pulse_spectrogram_context_ms < 0.0:
            raise ValueError("逐声波声谱图上下文不能小于 0 毫秒。")
        self.template_unknown_label = str(self.template_unknown_label).strip()
        if not self.template_unknown_label:
            raise ValueError("模板未知类别名称不能为空。")
        return self


# ---------------------------------------------------------------------------
# 模板匹配统计
# ---------------------------------------------------------------------------
def _image_outputs(config: SegmentAnalysisConfig) -> dict[str, bool]:
    return {"figures": config.save_figures, "pulse_spectrograms": config.save_pulse_spectrograms,
            "uncertain_spectrograms": config.save_uncertain_spectrograms,
            "unknown_spectrograms": config.save_unknown_spectrograms}


def _validate_checkpoint_image_outputs(state: dict, config: SegmentAnalysisConfig) -> None:
    # 旧界面固定输出两类图片；切换开关后必须新建任务，避免混入旧图和旧路径。
    previous = state.get("image_outputs", {"figures": True, "pulse_spectrograms": True})
    previous = {"uncertain_spectrograms": False, "unknown_spectrograms": False, **previous}
    if previous != _image_outputs(config):
        raise ValueError("图片输出开关已改变，请重新开始分析，避免混用旧断点结果。")


def _validate_checkpoint_promotion(state: dict, config: SegmentAnalysisConfig) -> None:
    # 模板放行会改变有效叫声的计数口径，整段必须保持一致。
    previous = bool(state.get("template_promotion", False))
    if previous != bool(config.allow_template_promotion):
        raise ValueError(
            "模板放行开关已改变，请重新开始分析，避免混用旧断点的计数口径。"
        )
    # 判族阈值改变会改变 CF / CF-FM 的类别计数。
    previous_plateau = state.get("cf_min_plateau_ms")
    if previous_plateau is None or float(previous_plateau) != float(config.cf_min_plateau_ms):
        raise ValueError(
            "CF 最短平台时长已改变，请重新开始分析，避免混用旧断点的类别口径。"
        )


def _empty_label_counts(
    labels: list[str],
    uncertain_label: str,
) -> dict[str, int]:
    counts = {label: 0 for label in labels}
    counts[uncertain_label] = 0
    return counts


def _aggregate_label_counts(
    pulse_labels: list[str],
    labels: list[str],
    uncertain_label: str,
) -> dict[str, int]:
    """把逐 pulse 标签聚合成 {label: count}（含 uncertain，绝不把总数归给 major）。"""
    counts = _empty_label_counts(labels, uncertain_label)
    for label in pulse_labels:
        key = str(label)
        if key in counts:
            counts[key] += 1
        else:
            # 未知标签（理论上不应出现）归入 uncertain，保证总和守恒
            counts[uncertain_label] += 1
    return counts


def _checkpoint_identity(wav_path: Path) -> dict:
    from .segment_stream import file_identity

    return file_identity(wav_path)


def _replay_checkpoint_rows(
    checkpoint, keep_n: int | None
) -> tuple[list[dict], "set[tuple[float, float]]"]:
    """从 partial CSV 回放已写行（数值类型尽力还原）。

    keep_n：水位线（state.rows_total）——只认前 keep_n 行；
    崩溃时可能多写出的尾部行一律截掉，由续跑重新处理，
    绝不出现“计数一次 + 行里还留一次”的双重残留。
    返回 (rows, 时间键集合)。
    """
    import csv

    rows: list[dict] = []
    if not checkpoint.partial_csv_path.exists():
        return rows, set()
    with open(
        checkpoint.partial_csv_path, "r", newline="", encoding="utf-8-sig"
    ) as handle:
        for raw in csv.DictReader(handle):
            if keep_n is not None and len(rows) >= int(keep_n):
                break
            row: dict = {}
            for key, value in raw.items():
                if value is None or value == "":
                    row[key] = value
                    continue
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = value
            rows.append(row)
    keys = {
        (round(float(row["start_s"]), 3), round(float(row["end_s"]), 3))
        for row in rows
        if "start_s" in row and "end_s" in row
    }
    return rows, keys


def _template_shape(label: str) -> str:
    """把模板类别名映射到门控的形态通道，用于"两种证据是否互相印证"。"""
    text = str(label or "")
    if text.startswith("CFFM"):
        return "CF-FM"
    if text.startswith("CF"):
        return "CF"
    if text.startswith("FM"):
        return "FM"
    return ""


def _promotable_by_template(row: dict, config: SegmentAnalysisConfig) -> bool:
    """仅缺能量证据时，模板相似度达标且形态一致的待定行可按模板放行（V1.5）。

    允许被放行的缺失项只有：全带集中度、局部带集中度、综合分。覆盖率、
    连续性、轨迹时长、质量属于"信号完整性"证据，缺任何一项都不放行；
    形态与模板类别不一致（例如纯 CF 片段被归到 CFFM 模板）也不放行。
    """
    if not config.allow_template_promotion:
        return False
    if row.get("gate_status") != "uncertain":
        return False
    # 信号完整性证据必须齐全，且至少要有一项能量/综合分缺失，才谈得上"放行"。
    if any(bool(row.get(f"gate_fail_{key}"))
           for key in ("coverage", "continuity", "duration", "quality")):
        return False
    if not any(bool(row.get(f"gate_fail_{key}"))
               for key in ("score", "concentration", "concentration_local")):
        return False
    score = row.get("confidence")
    try:
        value = float(score)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(value) or value < float(config.min_confidence):
        return False
    label = str(row.get("template_predicted_label") or "")
    if not label or label == config.template_unknown_label:
        return False
    return _template_shape(label) == str(row.get("gate_shape") or "")


def _finalize_chunk_rows(
    analysis_rows: list[dict],
    config: SegmentAnalysisConfig,
    pulse_id_start: int,
    template_payload: dict,
    prepared_model: PreparedTemplateModel | None = None,
) -> tuple[list[dict], int, int]:
    """明确噪声跳过匹配；bat 和门控待定匹配模板，待定仍保持独立状态。"""
    if not analysis_rows:
        return [], 0, pulse_id_start
    gated_rows = []
    for position, row in enumerate(analysis_rows):
        gated = dict(row)
        if gated.get("gate_status") not in {"bat", "noise", "uncertain"}:
            gated.update(unavailable_gate("缺少门控结果，保留复核"))
        gated["_gate_position"] = position
        gated_rows.append(gated)
    match_rows = [row for row in gated_rows if row["gate_status"] != "noise"]
    frame = classify_template_rows(
        pd.DataFrame(match_rows), template_payload,
        min_similarity=config.min_confidence,
        unknown_label=config.template_unknown_label,
        prepared_model=prepared_model,
        cf_min_plateau_ms=config.cf_min_plateau_ms,
    )
    matched = {int(row["_gate_position"]): row.to_dict() for _, row in frame.iterrows()}
    rows = []
    for position, gated in enumerate(gated_rows):
        row = matched.get(position, {
            **gated, "raw_label": "", "predicted_label": "noise", "confidence": None,
            "is_uncertain": False, "confidence_kind": "template_similarity",
            "template_label": "", "template_match_score": None,
            "template_match_distance": None, "template_file": "",
        })
        row.pop("_gate_position", None)
        row["template_predicted_label"] = row["predicted_label"] if row["gate_status"] != "noise" else ""
        row["template_is_uncertain"] = bool(row["is_uncertain"])
        if row["gate_status"] == "uncertain":
            row["predicted_label"] = "uncertain"
            row["is_uncertain"] = True
        row["pulse_id"] = f"p{pulse_id_start + position:03d}"
        value = row.get("confidence")
        row["confidence"] = float(value) if value is not None and np.isfinite(value) else None
        if _promotable_by_template(row, config):
            # 按模板放行：计数归入 bat，但保留原始未达标项与放行标记，便于事后剔除重算。
            row["gate_promoted"] = True
            row["gate_status"] = "bat"
            row["gate_pass_path"] = "template"
            row["gate_reason_code"] = "template_promoted"
            row["gate_reason"] = (
                f"仅缺能量证据；模板 {row['template_predicted_label']} 相似度 "
                f"{row['confidence']:.3f} ≥ {float(config.min_confidence):.2f}，按模板放行"
            )
            row["predicted_label"] = str(row["template_predicted_label"])
            row["is_uncertain"] = False
        rows.append(row)
    return rows, sum(row["gate_status"] == "noise" for row in rows), pulse_id_start + len(rows)


def _diagnose_chunk(batch, wav_path: Path, chunk_count: int) -> list[dict]:
    """逐块诊断行（仅当开关打开时调用）；任何异常都不影响分析。"""
    from .detection import diagnose_band_detection

    if batch.read_y is None or batch.read_y.size == 0:
        return []
    prefix = {
        "segment_file": wav_path.name,
        "chunk_index": int(batch.spec.index),
        "chunk_count": int(chunk_count),
        "coverage_start_s": round(float(batch.spec.coverage_start_s), 3),
        "coverage_end_s": round(float(batch.spec.coverage_end_s), 3),
        "skipped": bool(batch.skipped),
        "chunk_raw_candidates": int(batch.raw_candidate_count),
    }
    return [{**prefix, **row}
            for row in diagnose_band_detection(batch.read_y, TARGET_SAMPLE_RATE)]


def _max_subband_contrast(rows: list[dict]) -> float | None:
    values = [float(row["contrast_db"]) for row in rows
              if row.get("band_khz") != "all" and not row.get("skipped")]
    return round(max(values), 2) if values else None


def _subband_only_candidates(rows: list[dict]) -> int:
    """全带 0 候选、但子带有候选的块：累加各块子带候选数的最大值（跨带不去重）。"""
    per_chunk: dict[int, list[dict]] = {}
    for row in rows:
        per_chunk.setdefault(int(row.get("chunk_index", -1)), []).append(row)
    total = 0
    for chunk_rows in per_chunk.values():
        full = next((r for r in chunk_rows if r.get("band_khz") == "all"), None)
        if full is None or int(full.get("raw_candidates") or 0) > 0:
            continue
        total += max((int(r.get("raw_candidates") or 0) for r in chunk_rows
                      if r.get("band_khz") != "all"), default=0)
    return total


PULSE_ROW_EXTRA_KEYS = (
    *GATE_COLUMNS,
    "template_predicted_label",
    "template_is_uncertain",
    "template_track_khz",
    "pulse_id",
    "raw_label",
    "predicted_label",
    "confidence",
    "confidence_kind",
    "is_uncertain",
    "template_label",
    "template_match_score",
    "template_match_distance",
    "template_file",
    "pulse_spectrogram_file",
    "event_id",
    "pulse_spectrogram_mode",
    "pulse_spectrogram_status",
    "merged_spectrogram_file",
    "component_spectrogram_file",
    *TEMPLATE_FAMILY_RESULT_COLUMNS,
)


def _csv_fieldnames(rows: list[dict]) -> list[str]:
    if not rows:
        return list(PULSE_ROW_EXTRA_KEYS)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    for key in PULSE_ROW_EXTRA_KEYS:
        if key not in fieldnames:
            fieldnames.append(key)
    return fieldnames


class _SegmentAccumulator:
    """流式累计：计数/标签分布/置信度均值，支持 checkpoint 序列化。

    **计数绝不被导出上限影响**：这里累计的是检测/分类的全量结果，
    导出预算只在写 WAV 时生效。
    """

    def __init__(
        self,
        labels: list[str],
        uncertain_label: str,
        config: "SegmentAnalysisConfig",
    ) -> None:
        self.labels = list(labels)
        self.uncertain_label = uncertain_label
        self.config = config
        self.raw_candidate_count = 0
        self.candidate_call_count = 0
        self.valid_call_count = 0
        self.noise_count = 0
        self.bat_call_count = 0
        self.gate_uncertain_count = 0
        self.template_promoted_count = 0
        self.label_counts = _empty_label_counts(self.labels, uncertain_label)
        self.confidence_sum = 0.0
        self.confidence_n = 0
        self.next_pulse_id = 1
        self.rows: list[dict] = []
        self.rows_total = 0
        self.rows_truncated = False
        self.pulse_file_map: dict[str, str] = {}
        self.processed_chunk_count = 0
        self.fast_skipped_chunk_count = 0
        self.failed_duration_s = 0.0

    # ---- 增量累计 ----
    def add_batch(
        self,
        *,
        raw_candidate_count: int,
        pulse_rows: list[dict],
        result_rows: list[dict],
        noise_count: int,
        next_pulse_id: int,
        skipped: bool,
    ) -> None:
        self.raw_candidate_count += int(raw_candidate_count)
        self.candidate_call_count += len(pulse_rows)
        # 有效叫声只统计门控 bat（含模板放行），与 total_calls 口径一致。
        self.valid_call_count += sum(
            row.get("gate_status") == "bat" for row in result_rows
        )
        self.noise_count += int(noise_count)
        self.gate_uncertain_count += sum(row.get("gate_status") == "uncertain" for row in result_rows)
        self.template_promoted_count += sum(bool(row.get("gate_promoted")) for row in result_rows)
        self.next_pulse_id = int(next_pulse_id)

        bat_rows = [
            row
            for row in result_rows
            if row.get("gate_status") == "bat"
        ]
        self.bat_call_count += len(bat_rows)
        batch_labels = [
            str(row.get("predicted_label", "")) for row in bat_rows
        ]
        if batch_labels:
            counts = _aggregate_label_counts(
                batch_labels, self.labels, self.uncertain_label
            )
            for key, value in counts.items():
                self.label_counts[key] = int(
                    self.label_counts.get(key, 0)
                ) + int(value)
        for row in bat_rows:
            if row.get("confidence") is not None and not row.get("is_uncertain"):
                self.confidence_sum += float(row["confidence"])
                self.confidence_n += 1

        self.rows_total += len(result_rows)
        max_rows = int(SEGMENT_MAX_INMEMORY_ROWS)
        if max_rows <= 0 or len(self.rows) < max_rows:
            room = (
                max_rows - len(self.rows) if max_rows > 0 else len(result_rows)
            )
            self.rows.extend(result_rows[:room])
        elif result_rows:
            self.rows_truncated = True

        if skipped:
            self.fast_skipped_chunk_count += 1
        else:
            self.processed_chunk_count += 1

    # ---- checkpoint ----
    def to_state(self) -> dict:
        return {
            "raw_candidate_count": self.raw_candidate_count,
            "candidate_call_count": self.candidate_call_count,
            "valid_call_count": self.valid_call_count,
            "noise_count": self.noise_count,
            "bat_call_count": self.bat_call_count,
            "gate_uncertain_count": self.gate_uncertain_count,
            "template_promoted_count": self.template_promoted_count,
            "label_counts": dict(self.label_counts),
            "confidence_sum": self.confidence_sum,
            "confidence_n": self.confidence_n,
            "next_pulse_id": self.next_pulse_id,
            "rows_total": self.rows_total,
            "processed_chunk_count": self.processed_chunk_count,
            "fast_skipped_chunk_count": self.fast_skipped_chunk_count,
            "failed_duration_s": self.failed_duration_s,
            "pulse_file_map": dict(self.pulse_file_map),
        }

    def from_state(self, state: dict) -> None:
        for key in (
            "raw_candidate_count",
            "candidate_call_count",
            "valid_call_count",
            "noise_count",
            "bat_call_count",
            "gate_uncertain_count",
            "template_promoted_count",
            "confidence_sum",
            "confidence_n",
            "next_pulse_id",
            "rows_total",
            "processed_chunk_count",
            "fast_skipped_chunk_count",
            "failed_duration_s",
        ):
            if key in state:
                setattr(self, key, type(getattr(self, key))(state[key]))
        if isinstance(state.get("label_counts"), dict):
            self.label_counts = {
                str(key): int(value)
                for key, value in state["label_counts"].items()
            }
        if isinstance(state.get("pulse_file_map"), dict):
            self.pulse_file_map = {
                str(key): str(value)
                for key, value in state["pulse_file_map"].items()
            }

    # ---- 输出 ----
    @property
    def uncertain_count(self) -> int:
        return int(self.label_counts.get(self.uncertain_label, 0))

    @property
    def classified_call_count(self) -> int:
        return int(self.bat_call_count - self.uncertain_count)

    @property
    def detected_call_count(self) -> int:
        return int(self.bat_call_count)

    @property
    def mean_confidence(self) -> float | None:
        return (
            float(self.confidence_sum / self.confidence_n)
            if self.confidence_n
            else None
        )


def _export_batch_pulse_audio(
    result_rows: list[dict],
    *,
    wav_path: Path,
    segment_output_dir: Path,
    read_y: np.ndarray,
    read_start_s: float,
    batch_offset: int,
    budget_remaining: int,
) -> tuple[int, bool]:
    """按批导出 pulse WAV。返回 (本批导出数, 是否因预算截断)。

    **导出上限只影响磁盘文件数量**：行已全部进入计数/分类/CSV。
    """
    from .pulse_export import export_pulse_audio

    stem = wav_path.stem
    exported = 0
    truncated = False
    cache: dict[str, tuple[np.ndarray, int]] = {
        str(wav_path): (read_y, TARGET_SAMPLE_RATE)
    }
    for position, row in enumerate(result_rows):
        if str(row.get("predicted_label", "")) == "noise":
            continue
        if budget_remaining <= 0:
            truncated = True
            break
        index = batch_offset + position + 1
        destination = segment_output_dir / f"{stem}__p{index:03d}.wav"
        local_row = dict(row)
        local_row["start_s"] = float(row.get("start_s", 0.0)) - read_start_s
        local_row["end_s"] = float(row.get("end_s", 0.0)) - read_start_s
        try:
            written = export_pulse_audio(local_row, destination, cache)
        except Exception:  # noqa: BLE001 - 导出失败不影响统计
            written = ""
        if written:
            row["pulse_file"] = written
            exported += 1
            budget_remaining -= 1
    return exported, truncated


def _safe_output_component(value: str, fallback: str = "unknown") -> str:
    """把类别/片段名转换为可跨平台使用的目录或文件名片段。"""
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(value)).strip(" .")
    return text or fallback


def _export_batch_pulse_spectrograms(
    result_rows: list[dict],
    *,
    segment_id: str,
    spectrogram_root: Path,
    read_y: np.ndarray,
    read_start_s: float,
    context_ms: float = 0.0,
    mode: str = "merged",
    component_audio: dict[tuple[int, int], np.ndarray] | None = None,
    save_regular: bool = True,
    save_uncertain: bool = False,
    save_unknown: bool = True,
    unknown_label: str = "unknown",
) -> int:
    """按事件合并或按独立分量保存 PNG，返回实际成功图片张数。

    ``read_y`` 是当前流式 chunk 的预处理音频，行中的时间是整段 WAV 的
    全局秒数，因此这里先转换为 chunk 内索引。噪声不进入逐类图片；
    门控 bat 中模板未匹配的 ``unknown`` 由 ``save_unknown`` 单独控制。
    分量音频仅在当前 chunk 存活，不放入特征表；可信谐波随所属分量共图。
    """
    if not result_rows or len(read_y) < 32:
        return 0

    from .visualization import save_pulse_spectrogram

    root = Path(spectrogram_root)
    root.mkdir(parents=True, exist_ok=True)
    segment_name = _safe_output_component(segment_id, "segment")
    exported = 0
    sample_rate = int(TARGET_SAMPLE_RATE)
    context_samples = max(
        0,
        int(round(max(0.0, float(context_ms)) / 1000.0 * sample_rate)),
    )

    unknown_label = str(unknown_label or "unknown")
    if mode not in {"merged", "separate"}:
        raise ValueError("无效的声谱图输出方式。")
    groups: dict[str, list[dict]] = {}
    for row in result_rows:
        label = str(row.get("predicted_label", "")).strip()
        if not label or label == "noise":
            continue
        is_gate_uncertain = row.get("gate_status") == "uncertain"
        is_unknown = (not is_gate_uncertain) and label == unknown_label
        enabled = (save_uncertain if is_gate_uncertain
                   else save_unknown if is_unknown else save_regular)
        if not enabled:
            continue
        event_id = str(row.get("event_id") or row.get("pulse_id"))
        # 同事件中确定、unknown、待定分量分组输出；关闭开关时绝不回填未输出的图路径。
        suffix = "__uncertain" if is_gate_uncertain else "__unknown" if is_unknown else "__bat"
        key = (event_id if mode == "merged" else str(row.get("pulse_id"))) + suffix
        groups.setdefault(key, []).append(row)
        row["pulse_spectrogram_mode"] = mode
        row["pulse_spectrogram_status"] = "pending"
        row["merged_spectrogram_file"] = ""
        row["component_spectrogram_file"] = ""

    for group in groups.values():
        row = group[0]
        label = str(row["predicted_label"])
        try:
            start_s = min(float(item.get("candidate_start_s", item["start_s"]))
                          if mode == "merged" else float(item["start_s"]) for item in group)
            end_s = max(float(item.get("candidate_end_s", item["end_s"]))
                        if mode == "merged" else float(item["end_s"]) for item in group)
            pulse_id_text = str(row.get("pulse_id", "")).strip()
            pulse_id = int(
                pulse_id_text[1:]
                if pulse_id_text.lower().startswith("p")
                else (pulse_id_text or exported + 1)
            )
        except (TypeError, ValueError):
            continue

        start_sample = max(
            0,
            int(round((start_s - float(read_start_s)) * sample_rate))
            - context_samples,
        )
        end_sample = min(
            len(read_y),
            int(round((end_s - float(read_start_s)) * sample_rate))
            + context_samples,
        )
        if end_sample - start_sample < 32:
            continue
        pulse_y = np.asarray(read_y[start_sample:end_sample], dtype=np.float32)

        status = "merged" if mode == "merged" else "single_or_unresolved"
        if mode == "separate" and bool(row.get("separation_applied")):
            audio_key = (int(row["candidate_id"]), int(row["component_id"]))
            separated_audio = (component_audio or {}).get(audio_key)
            if separated_audio is None or len(separated_audio) < 32:
                row["pulse_spectrogram_status"] = "component_audio_unavailable"
                continue
            pulse_y = np.asarray(separated_audio, dtype=np.float32)
            status = "separated"

        # 合并图覆盖所有有效分量；分开图保留所属分量的谐波频率范围。
        frequency_values: list[float] = []
        for item in group:
            for field in ("f05_khz", "f95_khz", "start_freq_khz", "end_freq_khz", "peak_freq_khz"):
                value = float(item.get(field, float("nan")))
                if np.isfinite(value):
                    frequency_values.append(value)
            harmonic_frequencies = item.get("harmonic_frequencies_khz", [])
            if isinstance(harmonic_frequencies, str):
                harmonic_frequencies = json.loads(harmonic_frequencies or "[]")
            frequency_values.extend(float(value) for value in harmonic_frequencies
                                    if np.isfinite(float(value)))
        limits = None
        if frequency_values and (mode == "merged" or int(row.get("harmonic_group_size", 1)) > 1):
            from .visualization import AUTO_FOCUS_PADDING_HZ, AUTO_FOCUS_MIN_SPAN_HZ
            from .config import LOW_FREQ_HZ, HIGH_FREQ_HZ
            low = min(frequency_values) * 1000 - AUTO_FOCUS_PADDING_HZ
            high = max(frequency_values) * 1000 + AUTO_FOCUS_PADDING_HZ
            center = (low + high) / 2
            span = max(high - low, AUTO_FOCUS_MIN_SPAN_HZ)
            limits = (max(LOW_FREQ_HZ, center - span / 2),
                      min(HIGH_FREQ_HZ, sample_rate / 2, center + span / 2))

        labels = {str(item["predicted_label"]) for item in group}
        safe_label = ("uncertain" if row.get("gate_status") == "uncertain" else
                      _safe_output_component(label) if len(labels) == 1 else "_merged")
        suffix = "__merged" if mode == "merged" else ""
        filename = f"{segment_name}__p{pulse_id:05d}{suffix}.png"
        destination = root / safe_label / filename
        title_lines = [
            f"{segment_id} | {'Merged event' if mode == 'merged' else 'Pulse'} | t={start_s:.3f}-{end_s:.3f} s",
        ]
        for item in group:
            title_lines.append(
                f"{item['pulse_id']} | {item['predicted_label']} | "
                f"duration={float(item.get('duration_ms', 0)):.2f} ms | "
                f"peak={float(item.get('peak_freq_khz', 0)):.1f} kHz | "
                f"start/end={float(item.get('start_freq_khz', 0)):.1f}/{float(item.get('end_freq_khz', 0)):.1f} kHz"
            )
            # 模板分数单独显示，待定与 unknown 都能一眼看出差多少。
            template_label = str(item.get("template_predicted_label", "") or "")
            template_score = item.get("template_match_score")
            try:
                score_text = (
                    f" ({float(template_score):.2f})"
                    if template_score is not None and np.isfinite(float(template_score))
                    else ""
                )
            except (TypeError, ValueError):
                score_text = ""
            title_lines.append(
                f"gate={item.get('gate_status', '')} / {item.get('gate_shape', '')} | "
                f"CF={float(item.get('gate_cf_score', 0)):.2f} "
                f"CF-FM={float(item.get('gate_cffm_score', 0)):.2f} "
                f"FM={float(item.get('gate_fm_score', 0)):.2f} | "
                f"C_loc={float(item.get('gate_concentration_local', 0)):.2f} | "
                f"{item.get('gate_reason_code', '')} | "
                f"template={template_label}{score_text}"
            )
        if mode == "separate" and status == "single_or_unresolved":
            harmonic_count = int(row.get("harmonic_group_size", 1))
            if harmonic_count > 1:
                status = "harmonic_family"
                title_lines.append(f"Harmonic family ({harmonic_count} bands, one call)")
            else:
                title_lines.append("Original pulse (single call or unresolved overlap)")
        source_audio = None
        if status == "separated":
            # 分开图只关联已保存的分量 WAV，不能局部放大后重新混入其他声源。
            source_text = str(row.get("pulse_file", "") or "")
            preprocessed = True
            source_offset_s = 0.0
        else:
            source_text = str(row.get("source_path", "") or "")
            preprocessed = False
            source_offset_s = float(read_start_s) + start_sample / float(sample_rate)
        if source_text:
            from .config import BASE_DIR
            source_path = Path(source_text)
            if not source_path.is_absolute():
                source_path = BASE_DIR / source_path
            if source_path.is_file():
                source_audio = {"path": str(source_path.resolve()), "offset_s": source_offset_s,
                                "preprocessed": preprocessed}
        try:
            ok = save_pulse_spectrogram(
                pulse_y,
                sample_rate,
                destination,
                title_lines,
                frequency_limits_hz=limits,
                source_audio=source_audio,
                component_annotations=[(float(item.get("peak_freq_khz", 0)),
                                        f"{item['pulse_id']} {item['predicted_label']}")
                                       for item in group] if len(group) > 1 else None,
            )
        except Exception:  # noqa: BLE001 - 单个图片失败不影响统计
            ok = False
        if ok:
            relative = (Path(root.name) / safe_label / filename).as_posix()
            for item in group:
                item["pulse_spectrogram_file"] = relative
                item["merged_spectrogram_file" if mode == "merged" else "component_spectrogram_file"] = relative
                item["pulse_spectrogram_status"] = status
            exported += 1
        else:
            for item in group:
                item["pulse_spectrogram_status"] = "render_failed"
    return exported


def analyze_segment(
    wav_path: Path | str,
    config: SegmentAnalysisConfig,
    *,
    segment_output_dir: Path | None = None,
    separated_dir: Path | None = None,
    figure_path: Path | None = None,
    pulse_spectrogram_root: Path | None = None,
    chunk_progress: "Callable[[dict], None] | None" = None,
    should_stop: "Callable[[], bool] | None" = None,
    resume: bool | None = None,
    max_workers: int | None = None,
    template_payload: dict | None = None,
    prepared_model: PreparedTemplateModel | None = None,
) -> dict:
    """分析单个 WAV（几分钟片段，或 3 小时以上整条长录音）。

    检测内部叫声 + 逐叫声分类 + 片段汇总。算法不依赖 Qt。

    流式 chunk pipeline（segment_stream）：
    - 任何时刻只驻留 1~2 个 chunk，内存与文件总时长无关（绝不整条载入）；
    - **FULL RECORDING**：coverage 连续铺满 0→总时长，绝不抽样；
    - chunk + overlap，脉冲时间统一转 global 绝对时间，跨块不漏不重；
    - 结果增量落盘 + checkpoint；stop 在 chunk 边界优雅退出；
    - resume=True 时从断点续跑（校验文件身份 + 跨块去重防重复计数）。

    chunk_progress：块级进度回调；should_stop：取消标记。
    """
    from .segment_stream import (
        ChunkCheckpoint,
        ChunkContext,
        StreamingPreprocessor,
        StreamStats,
        SpectrogramSummary,
        build_chunk_specs,
        current_rss_mb,
        dedupe_cross_chunk_rows,
        probe_audio,
        resolve_max_workers,
        stream_chunks,
    )

    config = config.validate()
    wav_path = Path(wav_path)

    if not is_template_model(template_payload):
        raise ValueError("请提供有效的模板声波模型。")
    labels = [str(label) for label in template_payload.get("labels") or []]
    if not labels:
        raise ValueError("模板模型没有可用类别。")
    uncertain_label = config.template_unknown_label

    if prepared_model is None:
        prepared_model = prepare_template_model(template_payload)

    # ---- 流式探测：绝不整条载入（内存与文件总长无关）----
    total_frames, orig_sr, duration_s = probe_audio(wav_path)
    recording_duration_s = float(duration_s)
    # 速率分母 = 有效监测总时长（整条原始录音）；静音快筛只是计算优化，
    # 不代表那段时间从生态监测时间里消失。只有损坏/无法解码区间才排除。
    valid_monitoring_duration_s = float(recording_duration_s)
    failed_duration_s = 0.0

    # 有模板时用模板快照的候选时长上限，长 CF 叫声不被 40 ms 默认上限整段丢弃。
    max_pulse_ms = float((template_payload or {}).get("max_pulse_ms") or MAX_PULSE_MS)
    specs = build_chunk_specs(total_frames, orig_sr, max_pulse_ms=max_pulse_ms)
    chunk_total = len(specs)
    preprocessor = StreamingPreprocessor(
        wav_path, total_frames, orig_sr, StreamStats()
    )
    stats = preprocessor.compute_stats()
    detection_bands = (
        tuple(
            (float(band["low_khz"]), float(band["high_khz"]))
            for band in template_detection_bands(template_payload)
        )
        if config.template_band_detection
        else ()
    )
    context = ChunkContext(
        detection_bands_khz=detection_bands,
        max_pulse_ms=max_pulse_ms,
        wav_path=wav_path,
        preprocessor=preprocessor,
        stats=stats,
        total_samples=total_frames,
        separated_dir=separated_dir,
        save_separated_audio=True,
        template_matching=template_payload is not None,
        separation_window_mode=config.separation_window_mode,
        harmonic_grouping_mode=config.harmonic_grouping_mode,
        retain_component_audio=(
            (config.save_pulse_spectrograms or config.save_uncertain_spectrograms
             or config.save_unknown_spectrograms)
            and config.pulse_spectrogram_mode == "separate"
        ),
    )
    accumulator = _SegmentAccumulator(labels, uncertain_label, config)

    # ---- checkpoint / 断点续跑 ----
    checkpoint = None
    state = None
    resumed = False
    start_at_s = 0.0
    seen_row_keys: set[tuple[float, float]] = set()
    if segment_output_dir is not None:
        checkpoint = ChunkCheckpoint(
            Path(segment_output_dir) / ".checkpoint", wav_path
        )
        state = checkpoint.load_resumable() if resume else None
        if state is not None:
            _validate_checkpoint_image_outputs(state, config)
            _validate_checkpoint_promotion(state, config)
            if (state.get("pulse_processing_version") != PULSE_PROCESSING_VERSION
                    or state.get("gate_version") != GATE_VERSION):
                raise ValueError("脉冲处理方式已更新，请重新开始分析。")
            expected_modes = {
                "window": config.separation_window_mode,
                "harmonics": config.harmonic_grouping_mode,
            }
            previous_modes = state.get(
                "separation_modes", {"window": "fixed", "harmonics": "basic"}
            )
            if previous_modes != expected_modes:
                raise ValueError(
                    "分离窗或谐波处理模式已改变，不能混用旧断点结果。"
                    "请重新开始分析以生成独立的对照结果。"
                )
            if state.get("pulse_spectrogram_mode", "legacy") != config.pulse_spectrogram_mode:
                raise ValueError("声谱图输出方式已改变，请重新开始分析，避免混用旧断点图片。")
            accumulator.from_state(dict(state.get("accumulator") or {}))
            start_at_s = float(state.get("processed_until_s") or 0.0)
            resumed = True
            # 回放已写行（带水位线截断）；行集即计数集，续跑绝不双算
            max_rows = int(SEGMENT_MAX_INMEMORY_ROWS)
            replayed_rows, seen_row_keys = _replay_checkpoint_rows(
                checkpoint, int(accumulator.rows_total)
            )
            accumulator.rows = (
                replayed_rows[:max_rows]
                if max_rows > 0
                else replayed_rows
            )
        else:
            checkpoint.clear()

    if start_at_s > 0:
        specs = [
            spec
            for spec in specs
            if spec.coverage_end_s > start_at_s + 1e-9
            or spec.coverage_start_s >= start_at_s - 1e-9
        ]

    # ---- 流式处理主循环 ----
    export_budget = int(SEGMENT_MAX_PULSE_WAV_EXPORTS)
    exported_count = 0
    export_truncated = False
    exported_pulse_spectrogram_count = int(state.get("exported_pulse_spectrogram_count", 0)) if state else 0
    prev_tail_rows: list[dict] = []
    detection_diagnostic_rows: list[dict] = []
    batch_offset = int(accumulator.rows_total)
    processed_until_s = float(start_at_s)
    cancelled = False
    start_clock = datetime.now()
    summary = (
        SpectrogramSummary(recording_duration_s, sr=TARGET_SAMPLE_RATE)
        if config.save_figures and figure_path is not None
        else None
    )
    pending_rows: list[dict] = []

    def _on_progress(payload: dict) -> None:
        if chunk_progress is None:
            return
        elapsed = max(
            (datetime.now() - start_clock).total_seconds(), 1e-6
        )
        processed = float(payload.get("processed_until_s", 0.0))
        chunk_progress(
            {
                "segment_file": wav_path.name,
                "recording_duration_s": recording_duration_s,
                "processed_until_s": processed,
                "progress_ratio": (
                    processed / recording_duration_s
                    if recording_duration_s > 0
                    else 0.0
                ),
                "elapsed_s": elapsed,
                "speed_realtime": recording_duration_s
                and min(processed, recording_duration_s) / elapsed,
                "chunk_index": payload.get("chunk_index"),
                "chunk_count": payload.get("chunk_count"),
                "raw_candidate_count": accumulator.raw_candidate_count
                + int(payload.get("raw_candidate_count", 0)),
                "candidate_call_count": accumulator.candidate_call_count,
                "detected_call_count": accumulator.detected_call_count,
                "label_counts": dict(accumulator.label_counts),
                "fast_skipped_chunk_count": accumulator.fast_skipped_chunk_count,
                "rss_mb": current_rss_mb(),
            }
        )

    for batch in stream_chunks(
        context,
        specs,
        max_workers=resolve_max_workers(max_workers),
        progress=lambda payload: None,
        should_stop=should_stop,
    ):
        # 跨块去重：同一真实 pulse 只计一次（边界安全网）
        rows = dedupe_cross_chunk_rows(prev_tail_rows, batch.rows)
        if seen_row_keys:
            rows = [
                row
                for row in rows
                if (
                    round(float(row["start_s"]), 3),
                    round(float(row["end_s"]), 3),
                )
                not in seen_row_keys
            ]
        prev_tail_rows = [dict(row) for row in rows[-8:]]

        # 按块进行模板匹配
        result_rows, noise_count, next_id = _finalize_chunk_rows(
            rows,
            config,
            accumulator.next_pulse_id,
            template_payload,
            prepared_model,
        )
        result_rows.sort(key=lambda r: float(r.get("start_s", 0.0)))

        if (
            (config.save_pulse_spectrograms or config.save_uncertain_spectrograms
             or config.save_unknown_spectrograms)
            and pulse_spectrogram_root is not None
            and batch.read_y is not None
        ):
            exported_pulse_spectrogram_count += _export_batch_pulse_spectrograms(
                result_rows,
                segment_id=wav_path.stem,
                spectrogram_root=pulse_spectrogram_root,
                read_y=batch.read_y,
                read_start_s=batch.spec.read_start_s,
                context_ms=config.pulse_spectrogram_context_ms,
                mode=config.pulse_spectrogram_mode,
                component_audio=batch.component_audio,
                save_regular=config.save_pulse_spectrograms,
                save_uncertain=config.save_uncertain_spectrograms,
                save_unknown=config.save_unknown_spectrograms,
                unknown_label=config.template_unknown_label,
            )

        # 导出 pulse WAV（导出上限只影响磁盘；计数已在上面全量累计）
        if config.save_pulse_audio and segment_output_dir is not None:
            segment_output_dir.mkdir(parents=True, exist_ok=True)
            if export_budget > 0:
                remaining = export_budget - exported_count
            else:
                remaining = -1  # 不限
            exported, truncated = _export_batch_pulse_audio(
                result_rows,
                wav_path=wav_path,
                segment_output_dir=Path(segment_output_dir),
                read_y=batch.read_y if batch.read_y is not None
                else np.zeros(0),
                read_start_s=batch.spec.read_start_s,
                batch_offset=batch_offset,
                budget_remaining=remaining if remaining != -1 else 10**9,
            )
            exported_count += exported
            export_truncated = export_truncated or truncated

        if summary is not None and batch.read_y is not None:
            try:
                summary.add(batch.read_y, batch.spec.read_start_s)
            except Exception:  # noqa: BLE001 - 概览图失败不影响统计
                pass

        accumulator.add_batch(
            raw_candidate_count=batch.raw_candidate_count,
            pulse_rows=rows,
            result_rows=result_rows,
            noise_count=noise_count,
            next_pulse_id=next_id,
            skipped=bool(batch.skipped),
        )
        if config.save_detection_diagnostics and batch.read_y is not None:
            # 只读诊断：即使失败也不能影响分析结果。
            try:
                detection_diagnostic_rows.extend(
                    _diagnose_chunk(batch, wav_path, chunk_total)
                )
            except Exception:  # noqa: BLE001 - 诊断失败不影响分析
                pass
        batch_offset += len(result_rows)
        batch.read_y = None
        batch.component_audio.clear()
        processed_until_s = batch.spec.coverage_end_s

        # 增量落盘 + checkpoint（长任务不等全跑完才第一次写结果）
        if checkpoint is not None:
            pending_rows.extend(result_rows)
            checkpoint.append_rows(pending_rows, columns=_csv_fieldnames(pending_rows))
            pending_rows = []
            checkpoint.save(
                {
                    "status": "in_progress",
                    "pulse_processing_version": PULSE_PROCESSING_VERSION,
                    "gate_version": GATE_VERSION,
                    "pulse_spectrogram_mode": config.pulse_spectrogram_mode,
                    "image_outputs": _image_outputs(config),
                    "template_promotion": bool(config.allow_template_promotion),
                    "cf_min_plateau_ms": float(config.cf_min_plateau_ms),
                    "exported_pulse_spectrogram_count": exported_pulse_spectrogram_count,
                    "file_identity": _checkpoint_identity(wav_path),
                    "separation_modes": {
                        "window": config.separation_window_mode,
                        "harmonics": config.harmonic_grouping_mode,
                    },
                    "processed_until_s": processed_until_s,
                    "accumulator": accumulator.to_state(),
                    "recording_duration_s": recording_duration_s,
                    "valid_monitoring_duration_s": valid_monitoring_duration_s,
                }
            )
        _on_progress(
            {
                "processed_until_s": processed_until_s,
                "chunk_index": batch.spec.index,
                "chunk_count": len(specs),
                "raw_candidate_count": 0,
            }
        )

    if should_stop is not None and should_stop():
        cancelled = True

    if not cancelled:
        exported_pulse_spectrogram_count = _finalize_segment_sequences(
            accumulator, checkpoint, wav_path, config, pulse_spectrogram_root)

    # ---- 片段聚合（与旧实现同一口径）----
    label_counts = dict(accumulator.label_counts)
    uncertain_count = accumulator.uncertain_count
    classified_call_count = accumulator.classified_call_count
    specific_counts = {
        label: int(label_counts.get(label, 0)) for label in labels
    }
    if classified_call_count > 0:
        major_label = max(specific_counts, key=lambda k: specific_counts[k])
        if specific_counts[major_label] == 0:
            major_label = ""
        purity = float(specific_counts[major_label] / classified_call_count)
    else:
        major_label = ""
        purity = 0.0
    consistency_note = ""
    if classified_call_count > 0 and purity < config.mixed_purity_threshold:
        consistency_note = "混合 / 低一致性"

    detected_call_count = accumulator.detected_call_count
    calls_per_second = (
        float(detected_call_count / valid_monitoring_duration_s)
        if valid_monitoring_duration_s > 0
        else 0.0
    )
    mean_confidence = accumulator.mean_confidence

    # ---- 每段 pulse_results.csv（增量写入后原子替换）----
    if segment_output_dir is not None and checkpoint is not None:
        segment_output_dir = Path(segment_output_dir)
        segment_output_dir.mkdir(parents=True, exist_ok=True)
        if pending_rows:
            checkpoint.append_rows(
                pending_rows, columns=_csv_fieldnames(pending_rows)
            )
            pending_rows = []
        if cancelled:
            # 取消：partial 结果保留可见，checkpoint 保留可续跑
            checkpoint.save(
                {
                    "status": "cancelled",
                    "pulse_processing_version": PULSE_PROCESSING_VERSION,
                    "gate_version": GATE_VERSION,
                    "pulse_spectrogram_mode": config.pulse_spectrogram_mode,
                    "image_outputs": _image_outputs(config),
                    "template_promotion": bool(config.allow_template_promotion),
                    "cf_min_plateau_ms": float(config.cf_min_plateau_ms),
                    "exported_pulse_spectrogram_count": exported_pulse_spectrogram_count,
                    "file_identity": _checkpoint_identity(wav_path),
                    "separation_modes": {
                        "window": config.separation_window_mode,
                        "harmonics": config.harmonic_grouping_mode,
                    },
                    "processed_until_s": processed_until_s,
                    "accumulator": accumulator.to_state(),
                    "recording_duration_s": recording_duration_s,
                    "valid_monitoring_duration_s": valid_monitoring_duration_s,
                }
            )
        else:
            checkpoint.promote_partial_csv(
                segment_output_dir / "pulse_results.csv"
            )
            checkpoint.clear()

    # ---- 片段声谱图（流式低分辨率摘要；仅可视化，不参与统计）----
    if config.save_figures and figure_path is not None and summary is not None:
        try:
            from .visualization import save_segment_call_overview_from_summary

            save_segment_call_overview_from_summary(
                summary,
                accumulator.rows,
                figure_path,
                title=(
                    f"{wav_path.name} | calls: {detected_call_count}"
                ),
                source_audio={"path": str(wav_path.resolve()), "offset_s": 0.0, "preprocessed": False},
            )
        except Exception:  # noqa: BLE001 - 绘图失败不影响统计
            pass

    return {
        "segment_id": wav_path.stem,
        "segment_file": wav_path.name,
        "segment_path": str(wav_path),
        "duration_s": recording_duration_s,
        "status": "cancelled" if cancelled else "success",
        "error_message": "",
        # 四层计数（严格区分，见 README）
        "raw_candidate_count": int(accumulator.raw_candidate_count),
        "candidate_call_count": accumulator.candidate_call_count,
        "valid_call_count": accumulator.valid_call_count,
        "noise_count": int(accumulator.noise_count),
        "bat_call_count": accumulator.bat_call_count,
        "gate_uncertain_count": accumulator.gate_uncertain_count,
        "template_promoted_count": accumulator.template_promoted_count,
        "template_unknown_count": uncertain_count,
        "classified_call_count": classified_call_count,
        "uncertain_count": uncertain_count,
        # 主结果：有效蝙蝠叫声次数
        "detected_call_count": detected_call_count,
        "calls_per_second": calls_per_second,
        # 片段主要类别（描述性，非新物种算法）
        "major_label": major_label,
        "predicted_label": major_label,
        "prediction_confidence": mean_confidence,
        "purity": purity,
        "consistency_note": consistency_note,
        "label_counts": label_counts,
        "pulse_rows": accumulator.rows,
        "pulse_rows_truncated": accumulator.rows_truncated,
        "pulse_file_map": accumulator.pulse_file_map,
        # ---- 长录音口径（① 流式）----
        "recording_duration_s": recording_duration_s,
        "valid_monitoring_duration_s": valid_monitoring_duration_s,
        "failed_duration_s": failed_duration_s,
        "processed_chunk_count": accumulator.processed_chunk_count,
        "fast_skipped_chunk_count": accumulator.fast_skipped_chunk_count,
        "chunk_seconds": float(SEGMENT_STREAM_CHUNK_SECONDS),
        "overlap_seconds": float(SEGMENT_STREAM_OVERLAP_SECONDS),
        "exported_pulse_count": int(exported_count),
        "export_truncated": bool(export_truncated),
        "exported_pulse_spectrogram_count": int(
            exported_pulse_spectrogram_count
        ),
        "resumed": bool(resumed),
        "pulse_spectrogram_mode": config.pulse_spectrogram_mode,
        # ---- 检测诊断（开关打开时才有内容）----
        "detection_diagnostics": detection_diagnostic_rows,
        "max_subband_contrast_db": _max_subband_contrast(detection_diagnostic_rows),
        "subband_only_candidates": _subband_only_candidates(detection_diagnostic_rows),
    }


# ---------------------------------------------------------------------------
# 批量分析 + 汇总
# ---------------------------------------------------------------------------
def _failed_segment_result(
    wav_path: Path, error: str
) -> dict:
    return {
        "segment_id": Path(wav_path).stem,
        "segment_file": Path(wav_path).name,
        "segment_path": str(wav_path),
        "duration_s": 0.0,
        "status": "failed",
        "error_message": str(error),
        "raw_candidate_count": 0,
        "candidate_call_count": 0,
        "valid_call_count": 0,
        "noise_count": 0,
        "bat_call_count": 0,
        "gate_uncertain_count": 0,
        "template_promoted_count": 0,
        "template_unknown_count": 0,
        "classified_call_count": 0,
        "uncertain_count": 0,
        "detected_call_count": 0,
        "calls_per_second": 0.0,
        "major_label": "",
        "predicted_label": "",
        "prediction_confidence": None,
        "purity": 0.0,
        "consistency_note": "",
        "label_counts": {},
        "pulse_rows": [],
        "pulse_rows_truncated": False,
        "pulse_file_map": {},
        # ---- 长录音口径（① 流式）----
        "recording_duration_s": 0.0,
        "valid_monitoring_duration_s": 0.0,
        "failed_duration_s": 0.0,
        "processed_chunk_count": 0,
        "fast_skipped_chunk_count": 0,
        "chunk_seconds": float(SEGMENT_STREAM_CHUNK_SECONDS),
        "overlap_seconds": float(SEGMENT_STREAM_OVERLAP_SECONDS),
        "exported_pulse_count": 0,
        "export_truncated": False,
        "exported_pulse_spectrogram_count": 0,
        "resumed": False,
        "detection_diagnostics": [],
        "max_subband_contrast_db": None,
        "subband_only_candidates": 0,
    }


def summarize_segment_results(
    segment_results: list[dict],
    labels: list[str],
    uncertain_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把逐片段结果聚合成 (片段汇总表, 总体类别统计表)。

    统计正确性要求：一段内多类别时，各类别按**实际 call 数**累加，
    绝不把整段 total_calls 归给 major_label。
    """
    # ---- 第二层：片段汇总 ----
    summary_rows: list[dict] = []
    for result in segment_results:
        label_counts = result.get("label_counts", {}) or {}
        row = {
            "segment_id": result.get("segment_id", ""),
            "segment_file": result.get("segment_file", ""),
            "segment_path": result.get("segment_path", ""),
            "duration_s": result.get("duration_s", 0.0),
            "major_label": result.get("major_label", ""),
            "total_calls": result.get("detected_call_count", 0),
            "classified_call_count": result.get("classified_call_count", 0),
            "uncertain_count": result.get("uncertain_count", 0),
            "gate_uncertain_count": result.get("gate_uncertain_count", 0),
            "template_promoted_count": result.get("template_promoted_count", 0),
            "template_unknown_count": result.get("template_unknown_count", 0),
            "purity": result.get("purity", 0.0),
            "consistency_note": result.get("consistency_note", ""),
            "prediction_confidence": result.get("prediction_confidence"),
            "raw_candidate_count": result.get("raw_candidate_count", 0),
            "candidate_call_count": result.get("candidate_call_count", 0),
            "valid_call_count": result.get("valid_call_count", 0),
            "noise_count": result.get("noise_count", 0),
            "calls_per_second": result.get("calls_per_second", 0.0),
            "recording_duration_s": result.get("recording_duration_s", 0.0),
            "valid_monitoring_duration_s": result.get(
                "valid_monitoring_duration_s", 0.0
            ),
            "failed_duration_s": result.get("failed_duration_s", 0.0),
            "processed_chunk_count": result.get("processed_chunk_count", 0),
            "fast_skipped_chunk_count": result.get(
                "fast_skipped_chunk_count", 0
            ),
            "max_subband_contrast_db": result.get("max_subband_contrast_db"),
            "subband_only_candidates": result.get("subband_only_candidates", 0),
            "processing_status": result.get("status", ""),
            "error_message": result.get("error_message", ""),
        }
        for label in labels:
            row[label] = int(label_counts.get(label, 0))
        row[uncertain_label] = int(label_counts.get(uncertain_label, 0))
        summary_rows.append(row)

    summary_columns = [
        "segment_id",
        "segment_file",
        "segment_path",
        "duration_s",
        "major_label",
        "total_calls",
        *labels,
        uncertain_label,
        "classified_call_count",
        "uncertain_count",
        "gate_uncertain_count",
        "template_promoted_count",
        "template_unknown_count",
        "purity",
        "consistency_note",
        "prediction_confidence",
        "raw_candidate_count",
        "candidate_call_count",
        "valid_call_count",
        "noise_count",
        "calls_per_second",
        "recording_duration_s",
        "valid_monitoring_duration_s",
        "failed_duration_s",
        "processed_chunk_count",
        "fast_skipped_chunk_count",
        "max_subband_contrast_db",
        "subband_only_candidates",
        "processing_status",
        "error_message",
    ]
    segment_summary_df = pd.DataFrame(summary_rows)
    for column in summary_columns:
        if column not in segment_summary_df.columns:
            segment_summary_df[column] = 0
    segment_summary_df = segment_summary_df[summary_columns]

    # ---- 第三层：总体类别统计（严格按各类别 call 数累加）----
    overall_rows: list[dict] = []
    all_labels = list(labels) + [uncertain_label]
    for label in all_labels:
        call_count = 0
        segment_count = 0
        for result in segment_results:
            label_counts = result.get("label_counts", {}) or {}
            count = int(label_counts.get(label, 0))
            call_count += count
            if count >= 1:
                segment_count += 1
        overall_rows.append(
            {
                "species": label,
                "segment_count": segment_count,
                "call_count": call_count,
            }
        )

    overall_df = pd.DataFrame(overall_rows)
    return segment_summary_df, overall_df


def analyze_segment_folder(
    input_path: Path,
    config: SegmentAnalysisConfig,
    *,
    output_root: Path | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    chunk_progress: Callable[[dict], None] | None = None,
    on_segment_done: Callable[[dict], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    resume: bool | None = None,
    template_payload: dict | None = None,
) -> dict:
    """批量分析文件夹/单个 WAV 的所有片段，写独立输出目录并返回汇总。

    - 逐个 WAV：流式 analyze → save → release（不一次载入全部到内存）；
    - 失败隔离：单个损坏 WAV 记 failed，继续后续；
    - chunk_progress：块级进度回调（长录音逐块上报）；
    - resume：断点续跑开关（校验文件身份，变了自动重新开始）；
    - 输出只写 results/segment_analysis/<run_id>/（或 output_root）。
    """
    config = config.validate()
    input_path = Path(input_path)
    if not is_template_model(template_payload):
        raise ValueError("请提供有效的模板声波模型。")
    if resume and output_root is not None:
        from .segment_stream import find_resumable_checkpoints

        expected_modes = {
            "window": config.separation_window_mode,
            "harmonics": config.harmonic_grouping_mode,
        }
        for state_path in find_resumable_checkpoints(Path(output_root)):
            state = json.loads(state_path.read_text(encoding="utf-8"))
            _validate_checkpoint_image_outputs(state, config)
            _validate_checkpoint_promotion(state, config)
            if (state.get("pulse_processing_version") != PULSE_PROCESSING_VERSION
                    or state.get("gate_version") != GATE_VERSION):
                raise ValueError("脉冲处理方式已更新，请重新开始分析。")
            previous_modes = state.get(
                "separation_modes", {"window": "fixed", "harmonics": "basic"}
            )
            if previous_modes != expected_modes:
                raise ValueError(
                    "分离窗或谐波处理模式已改变，不能混用旧断点结果。"
                    "请重新开始分析以生成独立的对照结果。"
                )
            if state.get("pulse_spectrogram_mode", "legacy") != config.pulse_spectrogram_mode:
                raise ValueError("声谱图输出方式已改变，请重新开始分析，避免混用旧断点图片。")

    if input_path.is_file():
        wav_files = (
            [input_path] if input_path.suffix.lower() == ".wav" else []
        )
    elif input_path.is_dir():
        wav_files = find_wav_files(input_path)
    else:
        raise FileNotFoundError(f"输入路径不存在：{input_path}")

    if not wav_files:
        raise FileNotFoundError(f"没有找到 WAV 文件：{input_path}")

    # ---- 输出目录（独立 run）----
    if output_root is None:
        base = SEGMENT_ANALYSIS_DIR
        base.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base / run_id
        suffix = 1
        while run_dir.exists():
            suffix += 1
            run_dir = base / f"{run_id}_{suffix}"
    else:
        run_dir = Path(output_root)
    segments_dir = run_dir / SEGMENT_PULSE_SUBDIR
    figures_dir = run_dir / SEGMENT_FIGURE_SUBDIR
    pulse_spectrogram_dir = run_dir / "pulse_spectrograms"
    separated_dir = run_dir / SEGMENT_SEPARATED_SUBDIR
    for directory in (
        run_dir,
        segments_dir,
        figures_dir,
        pulse_spectrogram_dir,
        separated_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    model_info = describe_template_model(template_payload)
    labels = model_info["labels"]
    uncertain_label = config.template_unknown_label
    prepared_model = prepare_template_model(template_payload)
    from .template_matching import save_template_model
    template_model_path = run_dir / "template_model.json"
    save_template_model(template_payload, template_model_path)

    segment_results: list[dict] = []
    total = len(wav_files)

    for index, wav_path in enumerate(wav_files, start=1):
        if should_stop is not None and should_stop():
            break
        if progress is not None:
            progress(index, total, wav_path.name)

        segment_id = wav_path.stem
        segment_output_dir = segments_dir / segment_id
        figure_path = figures_dir / f"{segment_id}.png"
        try:
            result = analyze_segment(
                wav_path,
                    config,
                segment_output_dir=segment_output_dir,
                separated_dir=separated_dir,
                figure_path=figure_path,
                pulse_spectrogram_root=pulse_spectrogram_dir,
                chunk_progress=chunk_progress,
                should_stop=should_stop,
                resume=resume,
                template_payload=template_payload,
                prepared_model=prepared_model,
            )
        except Exception as error:  # noqa: BLE001 - 失败隔离
            result = _failed_segment_result(wav_path, f"{type(error).__name__}: {error}")

        segment_results.append(result)
        if on_segment_done is not None:
            try:
                on_segment_done(result)
            except Exception:  # noqa: BLE001 - 回调异常不影响批处理
                pass

    segment_summary_df, overall_df = summarize_segment_results(
        segment_results, labels, uncertain_label
    )

    # ---- 逐叫声明细（第一层）----
    # 内存有界：段内行超过 SEGMENT_MAX_INMEMORY_ROWS 时，
    # 明细已增量写入 pulse_results.csv，这里改从 CSV 流式汇总。
    all_pulse_rows: list[dict] = []
    prediction_columns = [
        "pulse_id",
        "start_s",
        "end_s",
        "duration_ms",
        "core_start_s", "core_end_s", "boundary_start_s", "boundary_end_s", "boundary_method",
        "predicted_label",
        "raw_label",
        "confidence",
        "is_uncertain",
            "peak_freq_khz",
        "start_freq_khz",
        "end_freq_khz",
        "snr_db",
        "detect_source",
        "pulse_file",
        "template_label",
        "template_match_score",
        "template_match_distance",
        "template_file",
        "pulse_spectrogram_file",
        "event_id",
        "pulse_spectrogram_mode",
        "pulse_spectrogram_status",
        "merged_spectrogram_file",
        "component_spectrogram_file",
        "harmonic_grouping_mode",
        "harmonic_group_size",
        "harmonic_frequencies_khz",
        "harmonic_evidence",
        *GATE_COLUMNS,
        "template_predicted_label",
        "template_is_uncertain",
        *TEMPLATE_FAMILY_RESULT_COLUMNS,
        "cf_plateau_khz",
        "cf_plateau_ms",
        "cf_plateau_spread_khz",
        "tail_low_energy_ratio",
        "tail_drop_khz",
        *SEQUENCE_COLUMNS,
        *BOUNDARY_COLUMNS,
    ]
    for result in segment_results:
        rows_iter: list[dict] = list(result.get("pulse_rows", []))
        if result.get("pulse_rows_truncated"):
            csv_path = (
                Path(run_dir)
                / SEGMENT_PULSE_SUBDIR
                / str(result.get("segment_id", ""))
                / "pulse_results.csv"
            )
            if csv_path.exists():
                rows_iter = pd.read_csv(
                    csv_path, encoding="utf-8-sig"
                ).to_dict("records")
        for row in rows_iter:
            all_pulse_rows.append(
                {
                    "segment_file": result.get("segment_file", ""),
                    "segment_id": result.get("segment_id", ""),
                    **{
                        column: row.get(column, "")
                        for column in prediction_columns
                    },
                }
            )
    pulse_predictions_df = pd.DataFrame(all_pulse_rows)

    # ---- 写 CSV ----
    segment_summary_df.to_csv(
        run_dir / "segment_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pulse_predictions_df.to_csv(
        run_dir / "pulse_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    overall_df.to_csv(
        run_dir / "overall_species_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ---- 检测诊断（仅在开关打开且确有内容时写出；不影响任何结果）----
    diagnostic_rows: list[dict] = []
    for result in segment_results:
        diagnostic_rows.extend(result.get("detection_diagnostics") or [])
    diagnostics_path = ""
    if diagnostic_rows:
        diagnostics_path = str(run_dir / "detection_diagnostics.csv")
        pd.DataFrame(diagnostic_rows).to_csv(
            diagnostics_path,
            index=False,
            encoding="utf-8-sig",
        )

    success_count = sum(
        1 for r in segment_results if r.get("status") == "success"
    )
    failed_count = len(segment_results) - success_count

    run_metadata = {
        "run_id": run_dir.name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_path": str(input_path),
        "segment_count": len(segment_results),
        "success_count": success_count,
        "failed_count": failed_count,
        "total_bat_calls": int(
            sum(r.get("detected_call_count", 0) for r in segment_results)
        ),
        "total_pulse_spectrograms": int(
            sum(
                r.get("exported_pulse_spectrogram_count", 0)
                for r in segment_results
            )
        ),
        "total_recording_duration_s": round(
            float(sum(r.get("recording_duration_s", 0.0) for r in segment_results)),
            3,
        ),
        "total_valid_monitoring_duration_s": round(
            float(
                sum(
                    r.get("valid_monitoring_duration_s", 0.0)
                    for r in segment_results
                )
            ),
            3,
        ),
        "cancelled_count": sum(
            1 for r in segment_results if r.get("status") == "cancelled"
        ),
        "streaming": {
            "chunk_seconds": float(SEGMENT_STREAM_CHUNK_SECONDS),
            "overlap_seconds": float(SEGMENT_STREAM_OVERLAP_SECONDS),
            "full_recording_coverage": True,
            "checkpoint_enabled": True,
        },
        "labels": labels,
        "uncertain_label": uncertain_label,
        "config": asdict(config),
        "gate_version": GATE_VERSION,
        "pulse_processing_version": PULSE_PROCESSING_VERSION,
        "sequence_classification": {"enabled": True, "mode": "reliable_reference_duration_median",
            "reference_boundary_rule": "complete; median SNR>=10dB; quality>=0.65; coverage>=0.8; no segmentation review",
            "boundary_support_rule": "512sample STFT,32sample hop; SNR and local contrast>=6dB; fixed P10/-ln(0.9) noise power",
            "fragment_bridge_rule": "gap<=50ms; exclude half-window edges; support>=90%; unsupported run<=0.5ms; no FM onset",
            "cf_admission_rule": "single_platform_or_trusted_long_sequence",
            **{k: v for k, v in asdict(SequenceConfig(plateau_threshold_ms=config.cf_min_plateau_ms,
                min_similarity=config.min_confidence)).items()
               if k not in {"short_requires_tail", "tail_ratio_threshold"}}},
        "template_detection_bands": (
            template_detection_bands(template_payload)
            if config.template_band_detection
            else []
        ),
        "total_gate_uncertain": int(sum(r.get("gate_uncertain_count", 0) for r in segment_results)),
        "total_template_promoted": int(
            sum(r.get("template_promoted_count", 0) for r in segment_results)
        ),
        "total_noise": int(sum(r.get("noise_count", 0) for r in segment_results)),
        "model": model_info,
        "warning": (
            "统计的是声学检测到的 call 数量，不是蝙蝠个体数量；"
            "模板类别来自人工确认的模板文件名。"
        ),
        "output_files": {
            "segment_summary": "segment_summary.csv",
            "pulse_predictions": "pulse_predictions.csv",
            **(
                {"detection_diagnostics": "detection_diagnostics.csv"}
                if diagnostics_path
                else {}
            ),
            "overall_species_summary": "overall_species_summary.csv",
            "segments_dir": SEGMENT_PULSE_SUBDIR,
            "figures_dir": SEGMENT_FIGURE_SUBDIR,
            "pulse_spectrograms_dir": "pulse_spectrograms",
            **(
                {"template_model": "template_model.json"}
                if template_model_path is not None
                else {}
            ),
        },
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "run_dir": str(run_dir),
        "segment_results": segment_results,
        "segment_summary": segment_summary_df,
        "pulse_predictions": pulse_predictions_df,
        "overall_summary": overall_df,
        "success_count": success_count,
        "failed_count": failed_count,
        "segment_count": len(segment_results),
        "model": model_info,
        "labels": labels,
        "uncertain_label": uncertain_label,
        "template_model_path": (
            str(template_model_path) if template_model_path is not None else ""
        ),
        "detection_diagnostics_path": diagnostics_path,
    }
