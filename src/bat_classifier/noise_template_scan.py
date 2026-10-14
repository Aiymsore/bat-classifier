"""② 低相似度 pulse 扫描与人工候选选择；模板类别在导入①时确定。"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio_io import find_wav_files
from .config import RESULTS_DIR, TARGET_SAMPLE_RATE
from .segment_analysis import SegmentAnalysisConfig, analyze_segment, _export_batch_pulse_spectrograms
from .template_matching import build_template_model, save_template_model

SCAN_KIND = "low_similarity_pulse_scan"
THRESHOLD_NAME = "蝙蝠声波模板相似度"
PULSE_COLUMNS = [
    "pulse_key", "segment_id", "segment_file", "source_path", "pulse_id", "event_id",
    "start_s", "end_s", "duration_ms", "confidence", "confidence_kind", "template_match_score",
    "template_label", "template_file", "predicted_label", "template_filename",
    "snr_db", "peak_freq_khz", "start_freq_khz", "end_freq_khz", "component_id",
    "component_count", "separation_applied", "harmonic_group_size", "harmonic_frequencies_khz",
    "pulse_file", "pulse_spectrogram_file", "pulse_spectrogram_status", "audio_export_error",
]


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _threshold(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{THRESHOLD_NAME}必须在 0~1 之间。")
    return value


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*PULSE_COLUMNS, "review_status"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _asset_path(run_dir: Path, relative: str) -> Path:
    path = (run_dir / relative).resolve()
    if not path.is_relative_to(run_dir.resolve()):
        raise ValueError("pulse 文件路径超出本次结果目录。")
    return path


def _recover_candidate_move(root: Path, state: dict) -> None:
    journal_path = root / ".candidate_move.json"
    if not journal_path.exists():
        return
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    key = journal["pulse_key"]
    committed = (key in state.get("candidate_keys", [])
                 and state.get("file_locations", {}).get(key) == journal["locations"])
    if not committed:
        for pair in reversed(journal["moves"]):
            source = _asset_path(root, pair["source"])
            destination = _asset_path(root, pair["destination"])
            if destination.exists():
                if source.exists():
                    raise FileExistsError(f"文件位置冲突，不能覆盖：{source.name}")
                source.parent.mkdir(parents=True, exist_ok=True)
                destination.rename(source)
            elif not source.exists():
                raise FileNotFoundError(f"找不到待恢复文件：{source.name}")
    journal_path.unlink()


def load_scan_pulses(run_dir: Path | str) -> list[dict]:
    """重新打开低分输出与候选状态；原预测标签保留。"""
    root = Path(run_dir)
    metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("kind") != SCAN_KIND:
        raise ValueError("请选择第②模块生成的结果目录。")
    state = json.loads((root / "review_state.json").read_text(encoding="utf-8"))
    _recover_candidate_move(root, state)
    selected = set(state.get("candidate_keys", []))
    locations = state.get("file_locations", {})
    with (root / "pulse_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for index, row in enumerate(rows, 1):
        row["template_filename"] = row.get("template_filename") or f"npulse{index:06d}.wav"
        row.update(locations.get(row["pulse_key"], {}))
        row["review_status"] = "candidate" if row["pulse_key"] in selected else "unselected"
        for field in ("pulse_file", "pulse_spectrogram_file"):
            if row.get(field):
                _asset_path(root, row[field])
    return rows


def move_to_candidates(run_dir: Path | str, pulse_keys: list[str]) -> int:
    """移入②候选区，文件改用 pulse 编号，供①按可疑声波导入。"""
    root = Path(run_dir)
    rows = load_scan_pulses(root)
    known = {row["pulse_key"] for row in rows}
    requested = set(pulse_keys)
    if not requested <= known:
        raise ValueError("所选 pulse 不属于本次扫描结果。")
    state_path = root / "review_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    previous = set(state.get("candidate_keys", []))
    initial_candidates = set(previous)
    locations = state.setdefault("file_locations", {})
    for row in rows:
        key = row["pulse_key"]
        if key not in requested:
            continue
        filename = Path(row["template_filename"]).name
        match = re.fullmatch(r"npulse(\d+)\.wav", filename, flags=re.IGNORECASE)
        if match:
            filename = f"pulse{match.group(1)}.wav"
        targets = {
            "pulse_file": f"candidates/templates/{filename}",
            "pulse_spectrogram_file": f"candidates/pulse_spectrograms/{Path(filename).stem}.png",
        }
        pairs = []
        updated = dict(locations.get(key, {}))
        updated["template_filename"] = filename
        for field, relative in targets.items():
            if not row.get(field):
                continue
            source = _asset_path(root, row[field])
            destination = _asset_path(root, relative)
            if source != destination:
                if not source.is_file():
                    raise FileNotFoundError(f"找不到待移动文件：{source.name}")
                if destination.exists():
                    raise FileExistsError(f"候选文件已存在，不能覆盖：{destination.name}")
                pairs.append((source, destination))
            updated[field] = relative
        if not pairs and key in previous and locations.get(key) == updated:
            row["review_status"] = "candidate"
            continue
        committed = []
        journal_path = root / ".candidate_move.json"
        _atomic_json(journal_path, {
            "pulse_key": key, "locations": updated,
            "moves": [{"source": source.relative_to(root.resolve()).as_posix(),
                       "destination": destination.relative_to(root.resolve()).as_posix()}
                      for source, destination in pairs],
        })
        try:
            for source, destination in pairs:
                destination.parent.mkdir(parents=True, exist_ok=True)
                source.rename(destination)
                committed.append((source, destination))
            locations[key] = updated
            previous.add(key)
            state["candidate_keys"] = sorted(previous)
            state["updated_at"] = datetime.now().isoformat(timespec="seconds")
            _atomic_json(state_path, state)
        except Exception:
            # 同一个 pulse 的两个文件一起移动；发生异常时回退本条文件移动。
            for source, destination in reversed(committed):
                destination.rename(source)
            journal_path.unlink()
            raise
        journal_path.unlink()
        row.update(updated)
        row["review_status"] = "candidate"
    _atomic_csv(root / "pulse_predictions.csv", rows)
    _atomic_csv(root / "candidates" / "pulse_predictions.csv",
                [row for row in rows if row["review_status"] == "candidate"])
    return len(requested - initial_candidates)


def scan_low_similarity_pulses(
    input_path: Path | str,
    template_dir: Path | str,
    *,
    similarity_threshold: float = 0.2,
    analysis_config: SegmentAnalysisConfig | None = None,
    output_root: Path | None = None,
    progress=None,
    chunk_progress=None,
    should_stop=None,
    max_workers: int | None = None,
) -> dict:
    """完整扫描并且只导出 score < 阈值的 pulse WAV/PNG/明细。"""
    threshold = _threshold(similarity_threshold)
    source = Path(input_path)
    if source.is_file() and source.suffix.lower() == ".wav":
        files = [source]
    elif source.is_dir():
        files = find_wav_files(source)
    else:
        raise FileNotFoundError("请选择 WAV 文件或包含 WAV 的文件夹。")
    if not files:
        raise FileNotFoundError("没有找到 WAV 文件。")
    payload = build_template_model(template_dir, min_similarity=threshold)
    base_config = analysis_config or SegmentAnalysisConfig(
        separation_window_mode="adaptive", harmonic_grouping_mode="enhanced")
    # 只改变输出开关与本次阈值，预处理/检测/分离/分类沿用①。
    config = replace(base_config, min_confidence=threshold, save_pulse_audio=False,
                     save_figures=False, save_pulse_spectrograms=False,
                     pulse_spectrogram_mode="separate").validate()
    if output_root is None:
        base = RESULTS_DIR / "noise_template_scan"
        base.mkdir(parents=True, exist_ok=True)
        root = base / datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = 1
        while root.exists():
            suffix += 1
            root = base / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}"
    else:
        root = Path(output_root)
        if root.exists() and any(root.iterdir()):
            raise ValueError("输出目录已有文件，请选择新的空目录。")
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "review_state.json", {"version": 1, "candidate_keys": []})
    save_template_model(payload, root / "template_model.json")
    metadata = {
        "kind": SCAN_KIND, "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_path": str(source.resolve()), "threshold_name": THRESHOLD_NAME,
        "similarity_threshold": threshold, "comparison": "strictly_less_than",
        "status": "running", "selected_pulse_count": 0, "processed_pulse_count": 0,
        "template_count": payload["template_count"], "config": asdict(config), "segments": [],
    }
    _atomic_json(root / "run_metadata.json", metadata)
    selected_count = 0
    exported_images = 0
    cancelled = False
    with (root / "pulse_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PULSE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        handle.flush()
        for index, wav in enumerate(files, 1):
            if should_stop and should_stop():
                cancelled = True
                break
            if progress:
                progress(index, len(files), wav.name)
            recording_key = hashlib.sha256(str(wav.resolve()).encode("utf-8")).hexdigest()[:12]
            segment_id = f"{wav.stem}__{recording_key}"

            def output_batch(rows, batch):
                nonlocal selected_count, exported_images
                low = [row for row in rows if row.get("confidence") is not None
                       and math.isfinite(float(row["confidence"])) and float(row["confidence"]) < threshold]
                if not low:
                    return
                for offset, row in enumerate(low, 1):
                    filename = f"npulse{selected_count + offset:06d}.wav"
                    row.update(pulse_key=f"{recording_key}_{row['pulse_id']}", segment_id=segment_id,
                               segment_file=wav.name, confidence_kind="template_similarity",
                               source_path=str(wav.resolve()), audio_export_error="", template_filename=filename)
                    destination = root / "templates" / filename
                    try:
                        if row.get("separation_applied"):
                            audio = batch.component_audio.get((int(row["candidate_id"]), int(row["component_id"])))
                            if audio is None:
                                raise ValueError("缺少分量音频，不能用混合音频代替。")
                        else:
                            start = max(0, round((float(row["start_s"]) - batch.spec.read_start_s) * TARGET_SAMPLE_RATE))
                            end = min(len(batch.read_y), round((float(row["end_s"]) - batch.spec.read_start_s) * TARGET_SAMPLE_RATE))
                            audio = batch.read_y[start:end]
                        if len(audio) < 32:
                            raise ValueError("pulse 音频过短。")
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if destination.exists():
                            raise FileExistsError(f"WAV 编号重复，不能覆盖：{destination.name}")
                        sf.write(destination, np.asarray(audio, dtype=np.float32), TARGET_SAMPLE_RATE, subtype="PCM_16")
                        row["pulse_file"] = destination.relative_to(root).as_posix()
                    except Exception as error:
                        row["pulse_file"] = ""
                        row["audio_export_error"] = f"{type(error).__name__}: {error}"
                exported_images += _export_batch_pulse_spectrograms(
                    low, segment_id=segment_id, spectrogram_root=root / "pulse_spectrograms",
                    read_y=batch.read_y, read_start_s=batch.spec.read_start_s, mode="separate",
                    component_audio=batch.component_audio, context_ms=config.pulse_spectrogram_context_ms)
                for row in low:
                    if row.get("pulse_spectrogram_file"):
                        try:
                            image_source = _asset_path(root, row["pulse_spectrogram_file"])
                            image_relative = f"pulse_spectrograms/{Path(row['template_filename']).stem}.png"
                            image_destination = _asset_path(root, image_relative)
                            if image_destination.exists():
                                raise FileExistsError(f"图片编号重复：{image_destination.name}")
                            image_source.rename(image_destination)
                            row["pulse_spectrogram_file"] = image_relative
                        except Exception:
                            row["pulse_spectrogram_status"] = "image_rename_failed"
                writer.writerows(low)
                handle.flush()
                selected_count += len(low)

            def emit_chunk(payload):
                if chunk_progress:
                    chunk_progress({**payload, "selected_pulse_count": selected_count})

            try:
                result = analyze_segment(wav, config, template_payload=payload,
                    on_pulse_batch=output_batch, save_intermediate_audio=False,
                    chunk_progress=emit_chunk, should_stop=should_stop, max_workers=max_workers)
                metadata["processed_pulse_count"] += result["detected_call_count"]
                metadata["segments"].append({"file": str(wav), "status": result["status"],
                    "processed_pulse_count": result["detected_call_count"]})
                cancelled = cancelled or result["status"] == "cancelled"
            except Exception as error:
                metadata["segments"].append({"file": str(wav), "status": "failed",
                    "error": f"{type(error).__name__}: {error}"})
            metadata["selected_pulse_count"] = selected_count
            metadata["exported_image_count"] = exported_images
            _atomic_json(root / "run_metadata.json", metadata)
            if cancelled:
                break
    failed = sum(item["status"] == "failed" for item in metadata["segments"])
    metadata["status"] = "cancelled" if cancelled else ("partial_failure" if failed else "success")
    metadata["failed_count"] = failed
    _atomic_json(root / "run_metadata.json", metadata)
    return {**metadata, "run_dir": str(root)}
