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
from concurrent.futures import CancelledError
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.ndimage import median_filter

from .audio_io import find_wav_files, load_audio
from .config import (
    MAX_PULSE_MS,
    TEMPLATE_BAND_HIGH_PERCENTILE,
    TEMPLATE_BAND_LOW_PERCENTILE,
    TEMPLATE_BAND_MARGIN_KHZ,
    TEMPLATE_CF_FREQUENCY_MARGIN_KHZ,
    TEMPLATE_CF_MIN_ELIGIBLE_PLATEAU_MS,
    TEMPLATE_CF_MIN_PLATEAU_MS,
    TEMPLATE_FAMILY_FEATURE_COLUMNS,
    TEMPLATE_FEATURE_COLUMNS,
    TEMPLATE_MODEL_COLUMNS,
)
from .detection import detect_pulses, refine_all_pulses
from .features import PulseSpectrum, compute_pulse_spectrum, extract_pulse_features


TEMPLATE_MODEL_KIND = "bat_template_matching"
TEMPLATE_MODEL_VERSION = 5  # 尾段能量从模板比较和判族中移除
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
    "cf_plateau_khz": 2.0,
    "cf_plateau_ms": 3.0,
    "cf_plateau_spread_khz": 1.0,
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
    "cf_plateau_khz": 2.0,
    "cf_plateau_ms": 0.5,
    "cf_plateau_spread_khz": 0.5,
}


def template_family(label: str) -> str:
    """按模板类别名前缀归族：CFFM* → CF-FM，CF* → CF，FM* → FM，其余为 other。"""
    text = str(label or "")
    if text.startswith("CFFM"):
        return "CF-FM"
    if text.startswith("CF"):
        return "CF"
    if text.startswith("FM"):
        return "FM"
    return "other"


def template_class_from_stem(stem: str) -> str:
    """去掉模板末尾样本序号，保留可信类别名称。

    例如 ``CF1_47K3`` → ``CF1_47K``，``CFFM1_67K08`` →
    ``CFFM1_67K``。没有符合频段+序号形式的文件名保持原 stem，避免
    把任意录音名误合并。
    """
    text = str(stem).strip()
    # 保留已有 npulse/pulse 模板的类别识别，兼容历史模板文件。
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


def extract_template_descriptor(
    audio: np.ndarray,
    sr: int,
    *,
    spectrum: PulseSpectrum | None = None,
    pulse_features: dict[str, float] | None = None,
) -> dict:
    """模板与待匹配 pulse 共用的频率轨迹/声学特征提取。

    保留绝对 kHz，避免不同频段因轨迹形状相近而被合并；时间归一到
    32 点，后续允许 DTW 局部伸缩。边缘低能量帧不参与轨迹。
    """
    if spectrum is None:
        spectrum = compute_pulse_spectrum(audio, sr)
    if spectrum.sample_count != len(audio) or spectrum.sample_rate != sr:
        raise ValueError("复用频谱的音频长度或采样率不一致。")
    # 复制建表规则的公共特征，再按模板规则覆盖自己的字段；不修改建表字典。
    features = dict(pulse_features) if pulse_features is not None else extract_pulse_features(
        audio, sr, spectrum=spectrum,
    )
    peaks = spectrum.frame_peak_power
    active = np.flatnonzero(spectrum.supported_frames if spectrum.supported_frames is not None
                           else peaks >= float(np.max(peaks)) * 10 ** (-24 / 10))
    if active.size < 2:
        raise ValueError("无法提取清晰超声频率轨迹。")
    track = spectrum.dominant_freqs_hz[active] / 1000.0
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


@njit(cache=True, nogil=True, fastmath=False)
def _dtw_distance(first: np.ndarray, second: np.ndarray) -> float:
    """有限时间伸缩的频率轨迹距离（kHz），不会消除绝对频率差。"""
    n, m = len(first), len(second)
    costs = np.full((n + 1, m + 1), np.inf)
    costs[0, 0] = 0.0
    lengths = np.zeros((n + 1, m + 1), dtype=np.int64)
    radius = max(4, abs(n - m))
    for i in range(1, n + 1):
        for j in range(max(1, i - radius), min(m, i + radius) + 1):
            # 严格小于才替换，保持旧 min 的同分优先序：对角、上、左。
            pi, pj = i - 1, j - 1
            if costs[i - 1, j] < costs[pi, pj]:
                pi, pj = i - 1, j
            if costs[i, j - 1] < costs[pi, pj]:
                pi, pj = i, j - 1
            costs[i, j] = costs[pi, pj] + abs(float(first[i - 1] - second[j - 1]))
            lengths[i, j] = lengths[pi, pj] + 1
    return float(costs[n, m] / max(lengths[n, m], 1))


def _template_features(path: Path) -> dict[str, float]:
    y, sr = load_audio(path)
    pulse_y = _best_template_pulse(y, sr)
    features = extract_template_descriptor(pulse_y, sr)
    # 只落盘/参与匹配的字段必须是有限数字。
    for column in TEMPLATE_MODEL_COLUMNS:
        value = float(features.get(column, np.nan))
        if not np.isfinite(value):
            raise ValueError(f"模板特征无效：{path.name} / {column}")
        features[column] = value
    return {column: features[column] for column in TEMPLATE_MODEL_COLUMNS} | {
        "template_track_khz": features["template_track_khz"]
    }


def _group_scales(template_rows: list[dict[str, float]]) -> dict[str, float]:
    frame = pd.DataFrame(template_rows, columns=TEMPLATE_MODEL_COLUMNS)
    scales: dict[str, float] = {}
    for column in TEMPLATE_MODEL_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        spread = float(np.ptp(values)) if values.size else 0.0
        # 极差对重复样本不敏感。相同模板多放几份，不改变类别容差或分数。
        scales[column] = max(float(_FEATURE_FLOORS[column]), spread * 0.5)
    return scales


def build_template_model(
    template_dir: Path | str,
    *,
    min_similarity: float = DEFAULT_TEMPLATE_MIN_SIMILARITY,
    progress=None,
    should_stop=None,
) -> dict:
    """从可信模板目录构建可序列化的模板模型 payload。

    所有可读取的模板都保留；单个损坏模板会记录在 ``errors``，只有在
    某个类别完全没有可用模板时才使整个导入失败。
    """
    root = Path(template_dir).resolve()
    groups = discover_template_groups(root)
    model_groups: dict[str, dict] = {}
    errors: list[str] = []
    total = sum(len(paths) for paths in groups.values())
    current = 0
    for label, paths in groups.items():
        if label.lower() in {"unknown", "uncertain", "noise", "unavailable"}:
            raise ValueError(f"模板类别名为保留名称：{label}")
        templates: list[dict] = []
        for path in paths:
            if should_stop and should_stop():
                raise CancelledError("模板加载已停止。")
            current += 1
            if progress:
                progress(current, total, path.name)
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
            if should_stop and should_stop():
                raise CancelledError("模板加载已停止。")
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
        "feature_columns": list(TEMPLATE_MODEL_COLUMNS),
        # 各族参与打分的特征列；未列出的族用完整列 TEMPLATE_FEATURE_COLUMNS 加频率轨迹。
        "family_columns": {family: list(columns) for family, columns in TEMPLATE_FAMILY_FEATURE_COLUMNS.items()},
        "default_family_columns": list(TEMPLATE_FEATURE_COLUMNS),
        "cf_min_plateau_ms": float(TEMPLATE_CF_MIN_PLATEAU_MS),
        "cf_frequency_margin_khz": float(TEMPLATE_CF_FREQUENCY_MARGIN_KHZ),
        "cf_min_eligible_plateau_ms": float(TEMPLATE_CF_MIN_ELIGIBLE_PLATEAU_MS),
        "cffm_min_tail_ratio": None,  # 尾段判据已停用，旧快照亦不启用
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




_NON_BAT_TEMPLATE_LABELS = {"noise pulse", "可疑声波"}


def template_detection_bands(
    payload: dict | None,
    *,
    margin_khz: float = TEMPLATE_BAND_MARGIN_KHZ,
) -> list[dict]:
    """从模板频率轨迹得到子带检测用的频带（kHz）。

    每个类别取其全部模板轨迹的 10–90 分位频率的并集，再两侧加余量；
    重叠或相接的频带合并成一个，记录覆盖的类别。噪声类模板不参与。
    """
    if not is_template_model(payload):
        return []
    raw: list[tuple[float, float, str]] = []
    for label, group in (payload.get("groups") or {}).items():
        if str(label).strip().lower() in _NON_BAT_TEMPLATE_LABELS or str(label) in _NON_BAT_TEMPLATE_LABELS:
            continue
        lows: list[float] = []
        highs: list[float] = []
        for template in group.get("templates") or []:
            track = np.asarray(
                (template.get("features") or {}).get("template_track_khz") or [],
                dtype=float,
            )
            track = track[np.isfinite(track)]
            if track.size == 0:
                continue
            lows.append(float(np.percentile(track, TEMPLATE_BAND_LOW_PERCENTILE)))
            highs.append(float(np.percentile(track, TEMPLATE_BAND_HIGH_PERCENTILE)))
        if lows:
            raw.append((min(lows) - margin_khz, max(highs) + margin_khz, str(label)))
    raw.sort()
    bands: list[dict] = []
    for low, high, label in raw:
        if bands and low <= bands[-1]["high_khz"]:
            bands[-1]["high_khz"] = max(bands[-1]["high_khz"], high)
            bands[-1]["labels"].append(label)
        else:
            bands.append({"low_khz": low, "high_khz": high, "labels": [label]})
    for band in bands:
        band["low_khz"] = round(float(band["low_khz"]), 2)
        band["high_khz"] = round(float(band["high_khz"]), 2)
    return bands


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


@dataclass(frozen=True)
class _PreparedTemplate:
    label: str
    family: str
    file: str
    index: np.ndarray
    values: np.ndarray
    scales: np.ndarray
    weights: np.ndarray
    track: np.ndarray | None


@dataclass(frozen=True)
class PreparedTemplateModel:
    """一次任务内复用的只读数值数据，不放入 JSON 模板快照。"""

    columns: tuple[str, ...]
    weights: np.ndarray
    templates: tuple[_PreparedTemplate, ...]
    # 快照带族特征配置（version 2）时才按平台时长判族；旧快照保持单一最佳匹配。
    family_rule: bool = False
    cf_min_plateau_ms: float = TEMPLATE_CF_MIN_PLATEAU_MS
    # CF 族频率区间：(类别, 下限, 上限, 各模板平台主频, 各模板文件)；空表示按相似度匹配 CF 族。
    cf_intervals: tuple[tuple[str, float, float, np.ndarray, tuple[str, ...]], ...] = ()
    cf_min_eligible_plateau_ms: float = TEMPLATE_CF_MIN_ELIGIBLE_PLATEAU_MS
    # CF-FM 族的尾巴证据下限；None 表示不要求（旧快照）。
    cffm_min_tail_ratio: float | None = None


def _readonly_array(values) -> np.ndarray:
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def prepare_template_model(payload: dict) -> PreparedTemplateModel:
    """按原类别/样本顺序整理全部模板；后续音频块不再重复建数组。

    每个模板按所属族只保留参与打分的特征列；CF 族不使用频率轨迹。
    """
    if not is_template_model(payload):
        raise ValueError("传入的不是模板匹配模型。")
    columns = tuple(c for c in (payload.get("feature_columns") or TEMPLATE_FEATURE_COLUMNS)
                    if c not in {"tail_low_energy_ratio", "tail_drop_khz"})
    weights = _readonly_array([_FEATURE_WEIGHTS.get(column, 1.0) for column in columns])
    family_columns = payload.get("family_columns")
    family_rule = isinstance(family_columns, dict)
    default_columns = list(payload.get("default_family_columns") or columns)
    templates = []
    for label in (str(value) for value in payload.get("labels") or []):
        group = (payload.get("groups") or {}).get(label) or {}
        family = template_family(label)
        scoring = (family_columns or {}).get(family) if family_rule else None
        uses_track = scoring is None
        scoring = [c for c in (scoring or (default_columns if family_rule else columns))
                   if c not in {"tail_low_energy_ratio", "tail_drop_khz"}]
        missing = [column for column in scoring if column not in columns]
        if missing:
            raise ValueError(f"模板快照缺少 {family} 族特征列：{missing}")
        index = np.array([columns.index(column) for column in scoring], dtype=np.intp)
        index.setflags(write=False)
        scales = _readonly_array([
            (group.get("scales") or {}).get(column, _FEATURE_FLOORS.get(column, 1.0))
            for column in scoring
        ])
        for template in group.get("templates") or []:
            features = template.get("features") or {}
            values = _readonly_array([features.get(column, np.nan) for column in scoring])
            # 与旧匹配循环相同，仅跳过数值特征无效的样本。
            if not np.isfinite(values).all():
                continue
            track = features.get("template_track_khz") if uses_track else None
            templates.append(_PreparedTemplate(
                label=label,
                family=family,
                file=str(template.get("file") or template.get("name") or ""),
                index=index,
                values=values,
                scales=scales,
                weights=_readonly_array(weights[index]),
                track=_readonly_array(track) if isinstance(track, (list, tuple, np.ndarray)) else None,
            ))
    cf_intervals = []
    margin = payload.get("cf_frequency_margin_khz")
    if family_rule and margin is not None:
        for label in (str(value) for value in payload.get("labels") or []):
            if template_family(label) != "CF":
                continue
            members = [
                (float((template.get("features") or {}).get("cf_plateau_khz", np.nan)),
                 str(template.get("file") or template.get("name") or ""))
                for template in ((payload.get("groups") or {}).get(label) or {}).get("templates") or []
            ]
            members = [(freq, file) for freq, file in members if np.isfinite(freq)]
            if not members:
                continue
            freqs = _readonly_array([freq for freq, _ in members])
            cf_intervals.append((
                label, float(np.min(freqs)) - float(margin), float(np.max(freqs)) + float(margin),
                freqs, tuple(file for _, file in members),
            ))
    return PreparedTemplateModel(
        columns, weights, tuple(templates), family_rule=family_rule,
        cf_min_plateau_ms=float(payload.get("cf_min_plateau_ms", TEMPLATE_CF_MIN_PLATEAU_MS)),
        cf_intervals=tuple(cf_intervals),
        cf_min_eligible_plateau_ms=float(
            payload.get("cf_min_eligible_plateau_ms", TEMPLATE_CF_MIN_ELIGIBLE_PLATEAU_MS)
        ),
        cffm_min_tail_ratio=None,
    )


def _match_cf_interval(
    prepared: PreparedTemplateModel,
    plateau_khz: float,
    plateau_ms: float,
) -> tuple[str, str, float, str] | None:
    """CF 族区间识别：返回 (类别, 最近模板文件, 与最近模板频率之差, 区间文本)。

    平台不足 ``cf_min_eligible_plateau_ms`` 或不在任何区间内时返回 None。
    这里只产生候选；正式流程还须通过连串后的统一 CF 时长准入检查。
    多个区间都包含时，取离最近模板平台主频最近的类别。
    """
    if not prepared.cf_intervals or not np.isfinite(plateau_khz):
        return None
    if not np.isfinite(plateau_ms) or plateau_ms < prepared.cf_min_eligible_plateau_ms:
        return None
    best = None
    for label, low, high, freqs, files in prepared.cf_intervals:
        if not low <= plateau_khz <= high:
            continue
        nearest = int(np.argmin(np.abs(freqs - plateau_khz)))
        offset = float(abs(freqs[nearest] - plateau_khz))
        if best is None or offset < best[2]:
            best = (label, files[nearest], offset, f"{low:.1f}-{high:.1f}")
    return best


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


TEMPLATE_FAMILY_RESULT_COLUMNS = (
    "template_family",
    "template_family_rule",
    "template_cf_label",
    "template_cf_score",
    "template_cf_offset_khz",
    "template_cf_interval_khz",
    "template_cffm_label",
    "template_cffm_score",
    "template_cffm_tail_ok",
)


def _resolve_family(
    best: dict[str, tuple[str, str, float]],
    threshold: float,
    plateau_ms: float,
    prepared: PreparedTemplateModel,
) -> tuple[str | None, str]:
    """在达到最低相似度的族里选出最终族，返回 (族, 判定依据)。

    CF 与 CF-FM 同时达标时按平台时长判族；之后与其他达标族比较相似度。
    只有一个族达标时直接采用；都不达标返回 (None, "none")。
    """
    passed = {family for family, (_, _, score) in best.items() if score >= threshold}
    if not passed:
        return None, "none"
    rule = "single" if len(passed) == 1 else "score"
    if prepared.family_rule and {"CF", "CF-FM"} <= passed:
        if np.isfinite(plateau_ms) and plateau_ms >= prepared.cf_min_plateau_ms:
            passed.discard("CF-FM")
            rule = "plateau_long"
        else:
            passed.discard("CF")
            rule = "plateau_short"
    family = max(passed, key=lambda name: best[name][2])
    return family, rule


def classify_template_rows(
    rows: pd.DataFrame,
    template_payload: dict,
    *,
    min_similarity: float | None = None,
    unknown_label: str = "unknown",
    prepared_model: PreparedTemplateModel | None = None,
    cf_min_plateau_ms: float | None = None,
) -> pd.DataFrame:
    """把 pulse rows 匹配到单条模板候选类别。

    正式输出还须经过连串判族及统一CF时长准入，不能直接用此结果计最终CF数。

    每个族先各自取最佳模板（CF 族只比平台特征）；再由 ``_resolve_family``
    选族。``cf_min_plateau_ms`` 覆盖快照里的判族阈值。
    """
    if not is_template_model(template_payload):
        raise ValueError("传入的不是模板匹配模型。")
    columns = [c for c in (template_payload.get("feature_columns") or TEMPLATE_FEATURE_COLUMNS)
               if c not in {"tail_low_energy_ratio", "tail_drop_khz"}]
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
            *TEMPLATE_FAMILY_RESULT_COLUMNS,
        ):
            result[column] = []
        return result

    matrix = result[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    finite_rows = np.isfinite(matrix).all(axis=1)
    prepared = prepared_model if prepared_model is not None else prepare_template_model(template_payload)
    if prepared.columns != tuple(columns):
        raise ValueError("预整理模板的特征列与模板模型不一致。")
    if cf_min_plateau_ms is not None:
        from dataclasses import replace

        prepared = replace(prepared, cf_min_plateau_ms=float(cf_min_plateau_ms))
    plateau_column = columns.index("cf_plateau_ms") if "cf_plateau_ms" in columns else None
    plateau_khz_column = columns.index("cf_plateau_khz") if "cf_plateau_khz" in columns else None
    if plateau_khz_column is None and prepared.cf_intervals:
        raise ValueError("模板快照缺少 cf_plateau_khz，无法按频率区间识别 CF 族。")
    raw_labels: list[str] = []
    predicted_labels: list[str] = []
    scores: list[float] = []
    distances: list[float] = []
    files: list[str] = []
    uncertain: list[bool] = []
    family_out: dict[str, list] = {column: [] for column in TEMPLATE_FAMILY_RESULT_COLUMNS}

    tracks = result.get("template_track_khz")
    for row_values, row_valid in zip(matrix, finite_rows):
        row_index = len(raw_labels)
        row_track_value = tracks.iloc[row_index] if tracks is not None else None
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
        # 每个族的最佳 (类别, 模板文件, 相似度)；同分保留先出现的模板。
        best: dict[str, tuple[str, str, float]] = {}
        if row_valid:
            for template in prepared.templates:
                score = _similarity_to_template(
                    row_values[template.index], template.values, template.scales, template.weights,
                    row_track, template.track,
                )
                if score > best.get(template.family, ("", "", 0.0))[2]:
                    best[template.family] = (template.label, template.file, score)
        plateau_ms = float(row_values[plateau_column]) if plateau_column is not None else float("nan")
        cf_similarity = best.get("CF")
        cffm_similarity = best.get("CF-FM")
        # 旧导出字段保留为空；不再检查尾段能量。
        interval = None
        if prepared.cf_intervals:
            # CF 族改由频率区间判定：区间内记 1.0，区间外不参与达标比较。
            best.pop("CF", None)
            if row_valid:
                interval = _match_cf_interval(
                    prepared, float(row_values[plateau_khz_column]), plateau_ms,
                )
            if interval is not None:
                best["CF"] = (interval[0], interval[1], 1.0)
        family, rule = _resolve_family(best, threshold, plateau_ms, prepared)
        fallback = dict(best)
        if cf_similarity is not None and "CF" not in fallback:
            fallback["CF"] = cf_similarity
        if cffm_similarity is not None and "CF-FM" not in fallback:
            fallback["CF-FM"] = cffm_similarity
        if family is None and fallback:
            # 都不达标：沿用旧口径，记录相似度最高的模板，最终类别为 unknown。
            family = max(fallback, key=lambda name: fallback[name][2])
        best_label, best_file, best_score = fallback.get(family, (unknown_label, "", 0.0)) if family else (unknown_label, "", 0.0)
        matched = best_label if rule != "none" else unknown_label
        raw_labels.append(best_label)
        predicted_labels.append(matched)
        scores.append(float(best_score))
        distances.append(float(-np.log(max(best_score, 1e-12))) if best_score > 0 else np.nan)
        files.append(best_file)
        uncertain.append(matched == unknown_label)
        family_out["template_family"].append(family or "")
        family_out["template_family_rule"].append(rule)
        if prepared.cf_intervals:
            # 区间识别时：类别来自区间，分数列给出平台相似度，仅供参考。
            family_out["template_cf_label"].append(interval[0] if interval else "")
            family_out["template_cf_score"].append(float(cf_similarity[2]) if cf_similarity else np.nan)
        else:
            label, _, score = best.get("CF", ("", "", np.nan))
            family_out["template_cf_label"].append(label)
            family_out["template_cf_score"].append(float(score))
        family_out["template_cf_offset_khz"].append(interval[2] if interval else np.nan)
        family_out["template_cf_interval_khz"].append(interval[3] if interval else "")
        label, _, score = cffm_similarity or ("", "", np.nan)
        family_out["template_cffm_label"].append(label)
        family_out["template_cffm_score"].append(float(score))
        family_out["template_cffm_tail_ok"].append("")

    result["raw_label"] = raw_labels
    result["predicted_label"] = predicted_labels
    result["confidence"] = scores
    result["is_uncertain"] = uncertain
    result["confidence_kind"] = TEMPLATE_CONFIDENCE_KIND
    result["template_label"] = raw_labels
    result["template_match_score"] = scores
    result["template_match_distance"] = distances
    result["template_file"] = files
    for column, values in family_out.items():
        result[column] = values
    return result
