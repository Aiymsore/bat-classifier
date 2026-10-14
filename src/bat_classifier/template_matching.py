"""① 模板声波匹配。

模板目录中的文件名
已经代表可信类别，末尾的数字只是同一类别的样本序号。这里不做聚类，
也不删除样本；每个类别保留全部模板，并把一次检测到的 pulse 只归到
一个最终类别（或 ``unknown``）。
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from .audio_io import adaptive_stft_params, find_wav_files, load_audio, ultrasonic_band_mask
from .config import TEMPLATE_FEATURE_COLUMNS, MAX_PULSE_MS
from .detection import detect_pulses, refine_all_pulses
from .features import extract_pulse_features


TEMPLATE_MODEL_KIND = "bat_template_matching"
TEMPLATE_MODEL_VERSION = 1
TEMPLATE_CONFIDENCE_KIND = "template_similarity"
DEFAULT_TEMPLATE_MIN_SIMILARITY = 0.60

# 这些是特征的“允许变化量”下限。类别内部仍会根据实际模板样本计算
# 标准差，但不会因为样本少而把一个类别的容差缩成 0。
_FEATURE_FLOORS = {
    "duration_ms": 2.0,
    "peak_freq_khz": 4.0,
    "start_freq_khz": 6.0,
    "end_freq_khz": 6.0,
    "frequency_drop_khz": 5.0,
    "f05_khz": 5.0,
    "f95_khz": 5.0,
    "bandwidth_90_khz": 5.0,
    "slope_khz_per_ms": 4.0,
    "track_coverage": 0.15,
    "max_track_jump_khz": 5.0,
}

_FEATURE_WEIGHTS = {
    "duration_ms": 1.0,
    "peak_freq_khz": 1.5,
    "start_freq_khz": 1.25,
    "end_freq_khz": 1.25,
    "frequency_drop_khz": 1.25,
    "f05_khz": 1.0,
    "f95_khz": 1.0,
    "bandwidth_90_khz": 0.75,
    "slope_khz_per_ms": 1.25,
    "track_coverage": 0.5,
    "max_track_jump_khz": 0.5,
}


def template_class_from_stem(stem: str) -> str:
    """去掉模板末尾样本序号，保留可信类别名称。

    例如 ``CF1_47K3`` → ``CF1_47K``，``CFFM1_67K08`` →
    ``CFFM1_67K``。没有符合频段+序号形式的文件名保持原 stem，避免
    把任意录音名误合并。
    """
    text = str(stem).strip()
    # ② 的 npulse 输出与人工移入候选区的 pulse 在①中作为两个模板类别。
    if re.fullmatch(r"npulse\d+", text, flags=re.IGNORECASE):
        return "noise pulse"
    if re.fullmatch(r"(?:pulse|p)\d+", text, flags=re.IGNORECASE):
        return "可疑声波"
    match = re.match(
        r"^(?P<label>.+\d(?:\.\d+)?[kK])[_-]?(?P<rep>\d+)$",
        text,
    )
    if match:
        return match.group("label")
    return text


def discover_template_groups(template_dir: Path | str) -> dict[str, list[Path]]:
    """递归发现 WAV 模板，并按可信文件名归组。"""
    root = Path(template_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"模板目录不存在：{root}")
    groups: dict[str, list[Path]] = {}
    for wav_path in find_wav_files(root):
        label = template_class_from_stem(wav_path.stem)
        groups.setdefault(label, []).append(wav_path)
    groups = {
        label: sorted(paths)
        for label, paths in sorted(groups.items())
    }
    if not groups:
        raise FileNotFoundError(f"模板目录中没有 WAV 文件：{root}")
    return groups


def _best_template_pulse(y: np.ndarray, sr: int) -> np.ndarray:
    """模板文件通常是一条叫声，若含静音边缘则优先截取主 pulse。"""
    # 已切短的模板缺少背景帧，原检测器的自适应背景阈值会把它全部
    # 当作背景；加静音上下文，再用同一检测/精修边界。
    padded = np.pad(y, (int(sr * 0.05), int(sr * 0.05)))
    max_ms = max(MAX_PULSE_MS, len(y) / sr * 1000 + 10.0)
    try:
        pulses, *_rest, _threshold = detect_pulses(padded, sr, max_duration_ms=max_ms)
        refined = refine_all_pulses(padded, sr, pulses, max_duration_ms=max_ms)
    except Exception:  # noqa: BLE001 - 退回整个模板，仍允许导入
        refined = []
    if not refined:
        return y
    best = max(
        refined,
        key=lambda pulse: float(pulse.get("snr_db", -999.0)),
    )
    start = max(0, int(round(float(best["start_s"]) * sr)))
    end = min(len(padded), int(round(float(best["end_s"]) * sr)))
    return padded[start:end] if end - start >= 32 else y


def extract_template_descriptor(audio: np.ndarray, sr: int) -> dict:
    """模板与待匹配 pulse 共用的频率轨迹/声学特征提取。

    保留绝对 kHz，避免不同频段因轨迹形状相近而被合并；时间归一到
    32 点，后续允许 DTW 局部伸缩。边缘低能量帧不参与轨迹。
    """
    features = dict(extract_pulse_features(audio, sr))
    n_fft, hop = adaptive_stft_params(len(audio), 512, 128, 8)
    power = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop)) ** 2
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mask = ultrasonic_band_mask(frequencies, sr)
    power, frequencies = power[mask], frequencies[mask]
    peaks = np.max(power, axis=0)
    active = np.flatnonzero(peaks >= float(np.max(peaks)) * 10 ** (-24 / 10))
    if active.size < 2:
        raise ValueError("无法提取清晰超声频率轨迹。")
    track = frequencies[np.argmax(power, axis=0)][active] / 1000.0
    track = median_filter(track, size=min(3, len(track)), mode="nearest")
    # 仅压缩时间轴，频率轴绝不归一或平移。
    track = np.interp(np.linspace(0, 1, 32), np.linspace(0, 1, len(track)), track)
    start, end = float(np.median(track[:3])), float(np.median(track[-3:]))
    duration_ms = float(len(audio) / sr * 1000)
    features.update({
        "duration_ms": duration_ms,
        "start_freq_khz": start,
        "end_freq_khz": end,
        "frequency_drop_khz": start - end,
        "slope_khz_per_ms": float(np.polyfit(np.linspace(0, duration_ms, 32), track, 1)[0]),
        "max_track_jump_khz": float(np.max(np.abs(np.diff(track)))),
        "template_track_khz": track.tolist(),
    })
    return features


def _dtw_distance(first: np.ndarray, second: np.ndarray) -> float:
    """有限时间伸缩的频率轨迹距离（kHz），不会消除绝对频率差。"""
    n, m = len(first), len(second)
    costs = np.full((n + 1, m + 1), np.inf)
    costs[0, 0] = 0.0
    lengths = np.zeros((n + 1, m + 1), dtype=int)
    radius = max(4, abs(n - m))
    for i in range(1, n + 1):
        for j in range(max(1, i - radius), min(m, i + radius) + 1):
            previous = [(i - 1, j - 1), (i - 1, j), (i, j - 1)]
            pi, pj = min(previous, key=lambda position: costs[position])
            costs[i, j] = costs[pi, pj] + abs(float(first[i - 1] - second[j - 1]))
            lengths[i, j] = lengths[pi, pj] + 1
    return float(costs[n, m] / max(lengths[n, m], 1))


def _template_features(path: Path) -> dict[str, float]:
    y, sr = load_audio(path)
    pulse_y = _best_template_pulse(y, sr)
    features = extract_template_descriptor(pulse_y, sr)
    # 只落盘/参与匹配的字段必须是有限数字。
    for column in TEMPLATE_FEATURE_COLUMNS:
        value = float(features.get(column, np.nan))
        if not np.isfinite(value):
            raise ValueError(f"模板特征无效：{path.name} / {column}")
        features[column] = value
    return {column: features[column] for column in TEMPLATE_FEATURE_COLUMNS} | {
        "template_track_khz": features["template_track_khz"]
    }


def _group_scales(template_rows: list[dict[str, float]]) -> dict[str, float]:
    frame = pd.DataFrame(template_rows, columns=TEMPLATE_FEATURE_COLUMNS)
    scales: dict[str, float] = {}
    for column in TEMPLATE_FEATURE_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        spread = float(np.ptp(values)) if values.size else 0.0
        # 极差对重复样本不敏感。相同模板多放几份，不改变类别容差或分数。
        scales[column] = max(float(_FEATURE_FLOORS[column]), spread * 0.5)
    return scales


def build_template_model(
    template_dir: Path | str,
    *,
    min_similarity: float = DEFAULT_TEMPLATE_MIN_SIMILARITY,
) -> dict:
    """从可信模板目录构建可序列化的模板模型 payload。

    所有可读取的模板都保留；单个损坏模板会记录在 ``errors``，只有在
    某个类别完全没有可用模板时才使整个导入失败。
    """
    root = Path(template_dir).resolve()
    groups = discover_template_groups(root)
    model_groups: dict[str, dict] = {}
    errors: list[str] = []
    for label, paths in groups.items():
        if label.lower() in {"unknown", "uncertain", "noise", "unavailable"}:
            raise ValueError(f"模板类别名为保留名称：{label}")
        templates: list[dict] = []
        for path in paths:
            try:
                templates.append(
                    {
                        "file": str(path),
                        "name": path.name,
                        "features": _template_features(path),
                    }
                )
            except Exception as error:  # noqa: BLE001 - 单模板失败隔离
                errors.append(f"{path.name}: {type(error).__name__}: {error}")
        if templates:
            model_groups[label] = {
                "templates": templates,
                "scales": _group_scales(
                    [template["features"] for template in templates]
                ),
            }
        else:
            raise ValueError(f"模板类别 {label} 没有可用样本。" + "；".join(errors[-5:]))
    if not model_groups:
        detail = "；".join(errors[:5])
        raise ValueError(f"模板目录中没有可用模板。{detail}")

    threshold = float(min_similarity)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("模板最低相似度必须在 0~1 之间。")
    payload = {
        "kind": TEMPLATE_MODEL_KIND,
        "version": TEMPLATE_MODEL_VERSION,
        "source_dir": str(root),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "labels": list(model_groups),
        "unknown_label": "unknown",
        "min_similarity": threshold,
        "feature_columns": list(TEMPLATE_FEATURE_COLUMNS),
        "groups": model_groups,
        "template_count": int(
            sum(
                len(group["templates"])
                for group in model_groups.values()
            )
        ),
        "errors": errors,
    }
    payload["max_pulse_ms"] = max(
        MAX_PULSE_MS,
        max(template["features"]["duration_ms"] for group in model_groups.values()
            for template in group["templates"]) * 1.5 + 10.0,
    )
    payload["fingerprint"] = template_model_fingerprint(payload)
    return payload


def template_model_fingerprint(payload: dict) -> str:
    data = {key: payload[key] for key in ("version", "labels", "groups", "max_pulse_ms")}
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def save_template_model(payload: dict, path: Path | str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return destination




def is_template_model(payload: dict | None) -> bool:
    return bool(
        isinstance(payload, dict)
        and payload.get("kind") == TEMPLATE_MODEL_KIND
    )


def describe_template_model(payload: dict | None) -> dict:
    if not is_template_model(payload):
        return {
            "loaded": False,
            "labels": [],
            "model_path": "",
            "warning": "未加载模板匹配模型。",
        }
    labels = [str(label) for label in payload.get("labels") or []]
    return {
        "loaded": bool(labels),
        "kind": TEMPLATE_MODEL_KIND,
        "labels": labels,
        "template_count": int(payload.get("template_count") or 0),
        "source_dir": str(payload.get("source_dir") or ""),
        "min_similarity": float(payload.get("min_similarity") or 0.0),
        "errors": list(payload.get("errors") or []),
        "warning": (
            "模板类别来自人工确认的文件名；每个类别包含多个模板样本，"
            "统计的是声学 call 数量，不代表蝙蝠个体数量。"
        ),
    }


def _similarity_to_template(
    values: np.ndarray,
    template: np.ndarray,
    scales: np.ndarray,
    weights: np.ndarray,
    track: np.ndarray | None = None,
    template_track: np.ndarray | None = None,
) -> float:
    normalized = (values - template) / np.maximum(scales, 1e-6)
    distance = float(
        np.sqrt(
            np.sum(weights * normalized * normalized)
            / max(float(np.sum(weights)), 1e-9)
        )
    )
    feature_score = float(np.exp(-distance))
    if track is None or template_track is None:
        return feature_score
    # 轨迹距离单位为 kHz；6 kHz 是宽松的初始容差，最终阈值由页面
    # 的最低相似度控制。轨迹和摘要特征各占一半，避免单个峰值主导。
    contour_score = float(np.exp(-_dtw_distance(track, template_track) / 6.0))
    return float(0.5 * feature_score + 0.5 * contour_score)


def classify_template_rows(
    rows: pd.DataFrame,
    template_payload: dict,
    *,
    min_similarity: float | None = None,
    unknown_label: str = "unknown",
) -> pd.DataFrame:
    """把 pulse rows 匹配到模板类别；每行只产生一个最终类别。"""
    if not is_template_model(template_payload):
        raise ValueError("传入的不是模板匹配模型。")
    columns = list(template_payload.get("feature_columns") or TEMPLATE_FEATURE_COLUMNS)
    threshold = (
        float(template_payload.get("min_similarity", DEFAULT_TEMPLATE_MIN_SIMILARITY))
        if min_similarity is None
        else float(min_similarity)
    )
    result = rows.copy()
    if result.empty:
        for column in (
            "raw_label",
            "predicted_label",
            "confidence",
            "is_uncertain",
            "confidence_kind",
            "template_label",
            "template_match_score",
            "template_match_distance",
            "template_file",
        ):
            result[column] = []
        return result

    matrix = result[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    finite_rows = np.isfinite(matrix).all(axis=1)
    weights = np.asarray(
        [float(_FEATURE_WEIGHTS.get(column, 1.0)) for column in columns],
        dtype=float,
    )
    labels = [str(label) for label in template_payload.get("labels") or []]
    raw_labels: list[str] = []
    predicted_labels: list[str] = []
    scores: list[float] = []
    distances: list[float] = []
    files: list[str] = []
    uncertain: list[bool] = []

    for row_values, row_valid in zip(matrix, finite_rows):
        row_index = len(raw_labels)
        row_track_value = result.iloc[row_index].get("template_track_khz")
        row_track = (
            np.asarray(row_track_value, dtype=float)
            if isinstance(row_track_value, (list, tuple, np.ndarray))
            else None
        )
        if row_track is not None and (
            row_track.ndim != 1
            or len(row_track) < 2
            or not np.isfinite(row_track).all()
        ):
            row_track = None
        best_label = unknown_label
        best_file = ""
        best_score = 0.0
        best_distance = float("inf")
        if row_valid:
            for label in labels:
                group = (template_payload.get("groups") or {}).get(label) or {}
                group_scales = np.asarray(
                    [
                        float(
                            (group.get("scales") or {}).get(
                                column,
                                _FEATURE_FLOORS.get(column, 1.0),
                            )
                        )
                        for column in columns
                    ],
                    dtype=float,
                )
                for template in group.get("templates") or []:
                    template_features = template.get("features") or {}
                    template_values = np.asarray(
                        [float(template_features.get(column, np.nan)) for column in columns],
                        dtype=float,
                    )
                    if not np.isfinite(template_values).all():
                        continue
                    template_track_value = template_features.get("template_track_khz")
                    template_track = (
                        np.asarray(template_track_value, dtype=float)
                        if isinstance(template_track_value, (list, tuple, np.ndarray))
                        else None
                    )
                    score = _similarity_to_template(
                        row_values,
                        template_values,
                        group_scales,
                        weights,
                        row_track,
                        template_track,
                    )
                    distance = float(-np.log(max(score, 1e-12)))
                    if score > best_score:
                        best_label = label
                        best_file = str(template.get("file") or template.get("name") or "")
                        best_score = score
                        best_distance = distance
        matched = best_label if best_score >= threshold else unknown_label
        is_unknown = matched == unknown_label
        raw_labels.append(best_label)
        predicted_labels.append(matched)
        scores.append(float(best_score))
        distances.append(float(best_distance) if np.isfinite(best_distance) else np.nan)
        files.append(best_file)
        uncertain.append(is_unknown)

    result["raw_label"] = raw_labels
    result["predicted_label"] = predicted_labels
    result["confidence"] = scores
    result["is_uncertain"] = uncertain
    result["confidence_kind"] = TEMPLATE_CONFIDENCE_KIND
    result["template_label"] = raw_labels
    result["template_match_score"] = scores
    result["template_match_distance"] = distances
    result["template_file"] = files
    return result
