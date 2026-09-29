"""代表样本抽样与导出（⑥ 批量整理与专家标注 的核心功能）。

基础逻辑来自 clustering.save_cluster_representatives（距中心最近的真实
pulse + 沿用已分离 WAV），这里把它扩展为四种抽样策略，每个 Cluster：

- center:   距本 Cluster centroid 最近的真实样本（不是合成质心音频）；
- boundary: boundary_margin = d2 - d1 最小的真实样本（最容易落入相邻
            Cluster，d1 = 距自己 centroid，d2 = 距第二近 centroid）；
- random:   Cluster 内固定随机种子抽样（RANDOM_STATE，保证可复现）；
- outlier:  距本 Cluster centroid 最远的真实样本（检查异常/混杂/误检）。

同一 pulse 即使命中多种策略也只保留一行，selection_reason 用 "|" 连接。
导出时必须复用已分离分量 WAV（export_pulse_audio），绝不从混合录音重切。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.preprocessing import StandardScaler

from ..clustering import export_pulse_audio
from ..config import CLUSTER_FEATURE_COLUMNS, RANDOM_STATE
from ..visualization import save_representative_spectrogram
from .pipeline import (
    atomic_write_csv,
    read_table,
    resolve_run_path,
    run_relative,
)

if TYPE_CHECKING:
    from .pipeline import CurationParams, CurationRun

REASON_ORDER = ("center", "boundary", "random", "outlier")

REPRESENTATIVE_COLUMNS = [
    "pulse_id",
    "cluster_id",
    "cluster_label",
    "selection_reason",
    "distance_to_center",
    "second_center_distance",
    "boundary_margin",
    "source_file",
    "source_path",
    "audio_hash",
    "start_s",
    "end_s",
    "duration_ms",
    "snr_db",
    "pulse_file",
    "spectrogram_file",
    "original_pulse_file",
    *CLUSTER_FEATURE_COLUMNS,
]


def _distance_matrix(
    points: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """points (n,d) 到 centroids (K,d) 的欧氏距离矩阵，用矩阵乘避免广播爆炸。"""
    points = np.asarray(points, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    distances_sq = (
        np.sum(points**2, axis=1)[:, None]
        + np.sum(centroids**2, axis=1)[None, :]
        - 2.0 * points @ centroids.T
    )
    return np.sqrt(np.maximum(distances_sq, 0.0))


def _join_reasons(reasons: set[str]) -> str:
    return "|".join(reason for reason in REASON_ORDER if reason in reasons)


def _format(value: float, digits: int = 1, missing: str = "—") -> str:
    try:
        if value is None or not np.isfinite(float(value)):
            return missing
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return missing


def select_representatives(
    frame: pd.DataFrame,
    scaled: np.ndarray,
    centroids: np.ndarray,
    id_mapping: dict[int, str],
    params: "CurationParams",
    exclude_pulse_ids: set[str] | None = None,
    only_cluster: str | None = None,
) -> pd.DataFrame:
    """按四种策略抽样代表 pulse（同一 pulse 去重后只保留一行）。"""
    if frame.empty:
        return pd.DataFrame(columns=REPRESENTATIVE_COLUMNS)

    exclude = set(exclude_pulse_ids or ())
    raw_labels = frame["raw_cluster"].to_numpy(dtype=int)
    pulse_ids = frame["pulse_id"].astype(str).to_numpy()
    cluster_ids = frame["cluster_id"].astype(str).to_numpy()

    feature_columns = [
        column for column in CLUSTER_FEATURE_COLUMNS if column in frame.columns
    ]

    records: list[dict] = []
    unique_clusters = sorted(set(id_mapping.values()))
    for cluster_id in unique_clusters:
        if only_cluster is not None and cluster_id != only_cluster:
            continue
        raw_cluster = next(
            (raw for raw, label in id_mapping.items() if label == cluster_id),
            None,
        )
        if raw_cluster is None or raw_cluster >= len(centroids):
            continue

        positions = np.flatnonzero(
            (cluster_ids == cluster_id)
            & np.array(
                [pid not in exclude for pid in pulse_ids], dtype=bool
            )
        )
        if len(positions) == 0:
            continue

        distances = _distance_matrix(
            scaled[positions], centroids[[raw_cluster]]
        )[:, 0]
        all_distances = _distance_matrix(scaled[positions], centroids)
        own_mask = np.ones(all_distances.shape[1], dtype=bool)
        own_mask[raw_cluster] = False
        if own_mask.any():
            second_distances = all_distances[:, own_mask].min(axis=1)
        else:  # pragma: no cover - k>=2 时必有其他 centroid
            second_distances = all_distances[
                :, np.argmax(all_distances.shape[1] - 1)
            ]
        margins = second_distances - distances

        ids_here = pulse_ids[positions]
        order_center = np.lexsort((ids_here, distances))
        order_boundary = np.lexsort((ids_here, margins))
        order_outlier = np.lexsort((ids_here, -distances))

        reasons: dict[int, set[str]] = {}

        def _mark(order: np.ndarray, reason: str, count: int) -> None:
            if count <= 0:
                return
            for local_index in order[:count]:
                reasons.setdefault(int(local_index), set()).add(reason)

        _mark(order_center, "center", params.n_center)
        _mark(order_boundary, "boundary", params.n_boundary)
        _mark(order_outlier, "outlier", params.n_outlier)

        if params.n_random > 0:
            rng = np.random.default_rng(
                [int(params.random_state), int(raw_cluster)]
            )
            picked = rng.choice(
                len(positions),
                size=min(int(params.n_random), len(positions)),
                replace=False,
            )
            for local_index in picked:
                reasons.setdefault(int(local_index), set()).add("random")

        for local_index, reason_set in sorted(reasons.items()):
            row = frame.iloc[int(positions[local_index])]
            record = {
                "pulse_id": str(row["pulse_id"]),
                "cluster_id": cluster_id,
                "cluster_label": f"Cluster {cluster_id}",
                "selection_reason": _join_reasons(reason_set),
                "distance_to_center": float(distances[local_index]),
                "second_center_distance": float(second_distances[local_index]),
                "boundary_margin": float(margins[local_index]),
                "source_file": row.get("source_file", ""),
                "source_path": row.get("source_path", ""),
                "audio_hash": row.get("audio_hash", ""),
                "start_s": float(row.get("start_s", 0.0)),
                "end_s": float(row.get("end_s", 0.0)),
                "duration_ms": float(row.get("duration_ms", 0.0)),
                "snr_db": float(row.get("snr_db", float("nan"))),
                "pulse_file": row.get("pulse_file", ""),
                "spectrogram_file": "",
                "original_pulse_file": row.get("pulse_file", ""),
            }
            for column in feature_columns:
                record[column] = row.get(column, np.nan)
            records.append(record)

    if not records:
        return pd.DataFrame(columns=REPRESENTATIVE_COLUMNS)

    result = pd.DataFrame(records)
    ordered = [
        column for column in REPRESENTATIVE_COLUMNS if column in result.columns
    ]
    result = result[ordered].sort_values(
        ["cluster_id", "distance_to_center", "pulse_id"]
    ).reset_index(drop=True)
    return result


def export_representative_media(
    run: "CurationRun",
    representatives: pd.DataFrame,
) -> pd.DataFrame:
    """为代表样本导出真实 WAV 与统一尺度声谱图，回填两个路径列。"""
    if representatives.empty:
        return representatives

    representatives = representatives.copy()
    audio_cache: dict[str, tuple[np.ndarray, int]] = {}

    for index, row in representatives.iterrows():
        cluster_id = str(row["cluster_id"])
        pulse_id = str(row["pulse_id"])
        cluster_dir = run.clusters_dir / f"cluster_{cluster_id}"
        destination = cluster_dir / "audio" / f"{pulse_id}.wav"

        # 录音加载缓存容量 1：避免把整批源录音读进内存。
        source_path = str(row.get("source_path", "") or "")
        if source_path and source_path not in audio_cache:
            audio_cache.clear()

        exported = ""
        try:
            exported = export_pulse_audio(row, destination, audio_cache)
        except Exception as error:  # noqa: BLE001 - 单个样本失败不中断导出
            print(f"[代表样本] 导出音频失败 {pulse_id}：{error}")

        representatives.at[index, "pulse_file"] = (
            run_relative(exported, run.run_dir) if exported else ""
        )

        audio_path = resolve_run_path(
            representatives.at[index, "pulse_file"], run.run_dir
        )
        spectrogram_path = ""
        if audio_path is not None:
            try:
                pulse_y, sample_rate = sf.read(audio_path, dtype="float32")
                title_lines = [
                    f"{row.get('cluster_label', f'Cluster {cluster_id}')}  |  {pulse_id}",
                    (
                        f"duration {_format(row.get('duration_ms'), 2)} ms"
                        f"  |  SNR {_format(row.get('snr_db'))} dB"
                        f"  |  peak {_format(row.get('peak_freq_khz'))} kHz"
                    ),
                    (
                        f"start {_format(row.get('start_freq_khz'))}"
                        f" → end {_format(row.get('end_freq_khz'))} kHz"
                    ),
                ]
                ok = save_representative_spectrogram(
                    pulse_y,
                    int(sample_rate),
                    cluster_dir / "spectrograms" / f"{pulse_id}.png",
                    title_lines,
                )
                if ok:
                    spectrogram_path = run_relative(
                        str(cluster_dir / "spectrograms" / f"{pulse_id}.png"),
                        run.run_dir,
                    )
            except Exception as error:  # noqa: BLE001
                print(f"[代表样本] 生成声谱图失败 {pulse_id}：{error}")
        representatives.at[index, "spectrogram_file"] = spectrogram_path

    return representatives


def select_and_export(
    run: "CurationRun",
    frame: pd.DataFrame,
    scaled: np.ndarray,
    centroids: np.ndarray,
    id_mapping: dict[int, str],
    params: "CurationParams",
    exclude_pulse_ids: set[str] | None = None,
) -> pd.DataFrame:
    """抽样 → 导出真实音频与声谱图 → 原子写 representatives.csv。"""
    representatives = select_representatives(
        frame,
        scaled,
        centroids,
        id_mapping,
        params,
        exclude_pulse_ids=exclude_pulse_ids,
    )
    representatives = export_representative_media(run, representatives)
    ordered = [
        column for column in REPRESENTATIVE_COLUMNS if column in representatives.columns
    ]
    representatives = representatives[ordered]
    atomic_write_csv(representatives, run.tables_dir / "representatives.csv")
    return representatives


# ---------------------------------------------------------------------------
# 追加代表样本（混杂 Cluster 的 review hint 后续动作）
# ---------------------------------------------------------------------------
def _load_cluster_state(
    run: "CurationRun",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[int, str]]:
    """读取 clustered 表与质心，并用同一数据重算标准化（结果确定一致）。"""
    frame = read_table(
        run.tables_dir / "clustered_pulses.csv",
        key_column="pulse_id",
    )
    centers = read_table(run.tables_dir / "cluster_centers.csv")
    if frame.empty or centers.empty:
        raise RuntimeError("缺少聚类结果，请先完成聚类阶段。")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(
        frame[CLUSTER_FEATURE_COLUMNS].to_numpy(dtype=float)
    )
    ordered_centers = centers.sort_values("raw_cluster").reset_index(drop=True)
    centroids = scaler.transform(
        ordered_centers[CLUSTER_FEATURE_COLUMNS].to_numpy(dtype=float)
    )
    id_mapping = {
        int(row["raw_cluster"]): str(row["cluster_id"])
        for _, row in ordered_centers.iterrows()
    }
    return frame, scaled, centroids, id_mapping


def add_cluster_representatives(
    run: "CurationRun",
    cluster_id: str,
    params: "CurationParams",
    count: int | None = None,
) -> pd.DataFrame:
    """为指定 Cluster 追加代表样本。

    优先级：未被选过 + 边界区域（boundary_margin 最小），
    数量不足时用离群（距质心最远）补足——用于混杂 Cluster 的复核抽样。
    """
    from ..config import CURATION_ADD_SAMPLES

    count = int(count or CURATION_ADD_SAMPLES)
    frame, scaled, centroids, id_mapping = _load_cluster_state(run)
    existing = read_table(run.tables_dir / "representatives.csv", key_column="pulse_id")
    excluded = (
        set(existing["pulse_id"].astype(str)) if not existing.empty else set()
    )

    fresh = select_representatives(
        frame,
        scaled,
        centroids,
        id_mapping,
        params,
        exclude_pulse_ids=excluded,
        only_cluster=cluster_id,
    )
    # select_representatives 按策略取值；这里重新排序：
    # 边界优先（margin 升序），其次离群（distance 降序）。
    if not fresh.empty:
        fresh = fresh.assign(
            _boundary=fresh["boundary_margin"],
            _distance=fresh["distance_to_center"],
        )
        boundary_part = fresh.sort_values(["_boundary", "pulse_id"]).head(count)
        remaining = fresh.drop(index=boundary_part.index)
        outlier_part = remaining.sort_values(
            ["_distance", "pulse_id"], ascending=[False, True]
        ).head(max(0, count - len(boundary_part)))
        fresh = (
            pd.concat([boundary_part, outlier_part])
            .drop(columns=["_boundary", "_distance"])
            .reset_index(drop=True)
        )

    if fresh.empty:
        return fresh

    fresh = export_representative_media(run, fresh)
    merged = (
        pd.concat([existing, fresh], ignore_index=True)
        if not existing.empty
        else fresh
    )
    ordered = [column for column in REPRESENTATIVE_COLUMNS if column in merged.columns]
    merged = merged[ordered]
    atomic_write_csv(merged, run.tables_dir / "representatives.csv")

    _refresh_representative_counts(run)
    from .annotations import AnnotationStore

    store = AnnotationStore.load(run.run_dir)
    store.initialize_from_representatives(merged)
    return fresh


def _refresh_representative_counts(run: "CurationRun") -> None:
    """追加代表样本后同步 cluster_summary.csv 的 representative_count。"""
    summary_path = run.tables_dir / "cluster_summary.csv"
    summary = read_table(summary_path)
    representatives = read_table(
        run.tables_dir / "representatives.csv", key_column="pulse_id"
    )
    if summary.empty or "cluster_id" not in summary.columns:
        return
    counts = (
        representatives["cluster_id"].astype(str).value_counts().to_dict()
        if not representatives.empty
        else {}
    )
    summary["representative_count"] = (
        summary["cluster_id"].astype(str).map(counts).fillna(0).astype(int)
    )
    atomic_write_csv(summary, summary_path)
