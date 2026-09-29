"""批量整理的聚类阶段（上游：pipeline 的 accepted 特征表；下游：代表样本与审核数据）。

复用现有算法链：CLUSTER_FEATURE_COLUMNS 特征 → StandardScaler → KMeans，
PCA 只用于可视化（不参与聚类，避免为降维损失信息）。

重要科研语义：
- 这里的 Cluster 只是“声学相似组”，编号为 Cluster 001/002/…；
- Cluster ≠ 物种，也 ≠ A-D 声型模型类别；
- 最佳 silhouette 的 K 并不等于真实物种数量；
- 自动聚类只用于减少专家需要人工查看的声波数量。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from ..config import (
    CLUSTER_FEATURE_COLUMNS,
    CURATION_AUTO_K_MAX,
    CURATION_K_MAX,
    CURATION_K_MIN,
    CURATION_MAX_FIT_SAMPLES,
    MAX_CLUSTER_PULSES_PER_RECORDING,
    RANDOM_STATE,
)
from ..features import accepted_mask, numeric_feature_frame
from ..visualization import save_pca_plot
from .pipeline import atomic_write_csv, read_table

if TYPE_CHECKING:
    from .pipeline import CurationParams, CurationRun

CLUSTER_WARNING = (
    "Cluster 只是声学相似组：同一种蝙蝠可能进入多个 Cluster，"
    "同一个 Cluster 也可能包含多个物种；最佳 silhouette K 不等于真实物种数量。"
)


# ---------------------------------------------------------------------------
# 特征表准备
# ---------------------------------------------------------------------------
def prepare_cluster_frame(pulses: pd.DataFrame) -> pd.DataFrame:
    """只取合格（accepted）脉冲，清洗成聚类特征表（保留全部合格脉冲）。"""
    required = {
        "accepted",
        "pulse_id",
        "source_file",
        *CLUSTER_FEATURE_COLUMNS,
    }
    missing = required - set(pulses.columns)
    if missing:
        raise ValueError(f"脉冲表缺少字段：{sorted(missing)}")

    data = pulses.loc[accepted_mask(pulses["accepted"])].copy()
    data[CLUSTER_FEATURE_COLUMNS] = numeric_feature_frame(
        data,
        list(CLUSTER_FEATURE_COLUMNS),
    )
    data = data.dropna(
        subset=["pulse_id", "source_file", *CLUSTER_FEATURE_COLUMNS]
    )
    return data.reset_index(drop=True)


def select_fit_indices(
    frame: pd.DataFrame,
    max_samples: int,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """为 KMeans 拟合抽样：先按录音限量（防单文件压倒），再随机补足上限。

    只影响拟合样本，不丢弃任何进入 clustered_pulses.csv 的数据
    （全部合格脉冲都会被 predict 归入某个 Cluster）。
    """
    total = len(frame)
    if total <= max_samples:
        return np.arange(total)

    rng = np.random.default_rng(random_state)
    groups: list[np.ndarray] = []
    for _, group in frame.groupby("source_file"):
        indices = group.index.to_numpy()
        if len(indices) > MAX_CLUSTER_PULSES_PER_RECORDING:
            indices = rng.choice(
                indices,
                size=MAX_CLUSTER_PULSES_PER_RECORDING,
                replace=False,
            )
        groups.append(indices)
    pool = np.concatenate(groups)
    if len(pool) > max_samples:
        pool = rng.choice(pool, size=max_samples, replace=False)
    return np.sort(pool)


# ---------------------------------------------------------------------------
# K 选择（Auto / 手动）
# ---------------------------------------------------------------------------
def evaluate_k(
    scaled: np.ndarray,
    k: int,
    random_state: int = RANDOM_STATE,
) -> dict:
    """计算单个候选 K 的 silhouette 与 inertia；单个 K 失败不影响整批。"""
    record = {"k": int(k), "silhouette": None, "inertia": None, "error": ""}
    try:
        model = KMeans(
            n_clusters=int(k),
            n_init=10,
            random_state=random_state,
        )
        labels = model.fit_predict(scaled)
        record["inertia"] = float(model.inertia_)
        unique_labels = set(int(value) for value in labels)
        if len(scaled) > k and len(unique_labels) > 1:
            record["silhouette"] = float(silhouette_score(scaled, labels))
    except Exception as error:  # noqa: BLE001 - 单个 K 失败应被记录而不是中断
        record["error"] = f"{type(error).__name__}: {error}"
    return record


def candidate_k_values(scaled: np.ndarray, params: "CurationParams") -> list[int]:
    """按样本量给出候选 K 区间；绝不返回 K > 样本数。"""
    sample_count = len(scaled)
    if sample_count < 2:
        return []
    upper = min(CURATION_K_MAX, sample_count)
    if params.k_mode == "manual":
        wanted = min(max(int(params.k), CURATION_K_MIN), upper)
        # silhouette 需要 K < n，尽量保证可评估
        if sample_count - 1 >= CURATION_K_MIN:
            wanted = min(wanted, sample_count - 1)
        return [max(wanted, CURATION_K_MIN)]
    auto_upper = min(CURATION_AUTO_K_MAX, upper)
    if sample_count - 1 >= CURATION_K_MIN:
        auto_upper = min(auto_upper, sample_count - 1)
    return list(range(CURATION_K_MIN, max(auto_upper, CURATION_K_MIN) + 1))


def select_best_k(records: list[dict]) -> int | None:
    """silhouette 最大者胜（并列取更小的 K）；无 silhouette 时取可拟合的最小 K。"""
    usable = [record for record in records if not record["error"]]
    if not usable:
        return None
    with_silhouette = [
        record for record in usable if record["silhouette"] is not None
    ]
    if with_silhouette:
        best = max(
            with_silhouette,
            key=lambda record: (record["silhouette"], -record["k"]),
        )
        return int(best["k"])
    fitted = [
        record
        for record in usable
        if record["inertia"] is not None
    ]
    if fitted:
        return int(min(fitted, key=lambda record: record["inertia"])["k"])
    return int(usable[0]["k"])


def save_k_selection_plot(
    records: list[dict],
    selected_k: int | None,
    output_path: Path,
) -> bool:
    """保存候选 K 的 silhouette / inertia 曲线（cluster_quality.png）。"""
    usable = [
        record
        for record in records
        if record["silhouette"] is not None or record["inertia"] is not None
    ]
    if not usable:
        return False

    ks = [record["k"] for record in usable]
    silhouettes = [
        record["silhouette"] if record["silhouette"] is not None else np.nan
        for record in usable
    ]
    inertias = [
        record["inertia"] if record["inertia"] is not None else np.nan
        for record in usable
    ]

    figure, axis_silhouette = plt.subplots(figsize=(9, 5))
    axis_silhouette.plot(
        ks, silhouettes, marker="o", color="#1f77b4", label="silhouette"
    )
    axis_silhouette.set_xlabel("K (number of clusters)")
    axis_silhouette.set_ylabel("silhouette score", color="#1f77b4")
    axis_silhouette.tick_params(axis="y", labelcolor="#1f77b4")

    axis_inertia = axis_silhouette.twinx()
    axis_inertia.plot(
        ks, inertias, marker="s", color="#ff7f0e", label="inertia"
    )
    axis_inertia.set_ylabel("inertia", color="#ff7f0e")
    axis_inertia.tick_params(axis="y", labelcolor="#ff7f0e")

    if selected_k is not None:
        axis_silhouette.axvline(
            selected_k, color="green", linestyle="--", linewidth=1.2
        )
        axis_silhouette.annotate(
            f"selected K = {selected_k}",
            xy=(selected_k, axis_silhouette.get_ylim()[1]),
            xytext=(4, -12),
            textcoords="offset points",
            color="green",
            fontsize=9,
        )

    axis_silhouette.set_title(
        "K selection: silhouette vs inertia\n"
        "best-silhouette K is an acoustic-similarity count, NOT a species count"
    )
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return True


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _cluster_ids_by_peak_frequency(
    centers_original: np.ndarray,
    feature_columns: list[str],
) -> dict[int, str]:
    """按中心峰值频率从低到高把 raw 编号稳定映射为 001/002/…。"""
    peak_index = feature_columns.index("peak_freq_khz")
    order = np.argsort(centers_original[:, peak_index], kind="stable")
    mapping: dict[int, str] = {}
    for rank, raw_cluster in enumerate(order, start=1):
        mapping[int(raw_cluster)] = f"{rank:03d}"
    return mapping


def cluster_run(
    run: "CurationRun",
    params: "CurationParams",
    stats=None,
) -> dict:
    """对一个 run 执行聚类、图表、代表样本与审核数据初始化。"""
    pulses = read_table(
        run.tables_dir / "all_pulses.csv",
        key_column="pulse_id",
    )
    if pulses.empty:
        return {
            "ok": False,
            "error": "还没有可用脉冲：请先完成扫描与处理阶段。",
        }

    try:
        frame = prepare_cluster_frame(pulses)
    except ValueError as error:
        return {"ok": False, "error": str(error)}

    if len(frame) < 2:
        return {
            "ok": False,
            "error": f"合格脉冲只有 {len(frame)} 个，至少需要 2 个才能聚类。",
        }

    print(f"\n[聚类] 合格脉冲 {len(frame)} 个，开始标准化与 K 选择…")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(
        frame[CLUSTER_FEATURE_COLUMNS].to_numpy(dtype=float)
    )

    fit_indices = select_fit_indices(
        frame,
        CURATION_MAX_FIT_SAMPLES,
        params.random_state,
    )
    fit_scaled = scaled[fit_indices]

    # ---- 候选 K 评估（单个 K 失败只记录，不中断） ----
    records = [
        evaluate_k(fit_scaled, k, params.random_state)
        for k in candidate_k_values(fit_scaled, params)
    ]
    selected_k = select_best_k(records)
    if selected_k is None:
        detail = "; ".join(
            f"K={record['k']} {record['error']}" for record in records[:5]
        )
        return {
            "ok": False,
            "error": f"所有候选 K 拟合失败：{detail}",
        }

    mode_text = "手动" if params.k_mode == "manual" else "Auto"
    best_silhouette = next(
        (
            record["silhouette"]
            for record in records
            if record["k"] == selected_k and record["silhouette"] is not None
        ),
        None,
    )
    print(
        f"[聚类] {mode_text} 选定 K={selected_k}"
        + (
            f"（候选内最佳 silhouette={best_silhouette:.3f}）"
            if best_silhouette is not None
            else ""
        )
    )
    print("[聚类] 注意：最佳 silhouette K 是声学相似组数量，不是物种数量。")

    # ---- 最终拟合并对全量 predict ----
    try:
        kmeans = KMeans(
            n_clusters=selected_k,
            n_init=30,
            random_state=params.random_state,
        )
        kmeans.fit(fit_scaled)
        raw_labels = kmeans.predict(scaled)
    except Exception as error:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"KMeans 拟合失败：{type(error).__name__}: {error}",
        }

    centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
    id_mapping = _cluster_ids_by_peak_frequency(
        centers_original,
        list(CLUSTER_FEATURE_COLUMNS),
    )
    frame["raw_cluster"] = [int(value) for value in raw_labels]
    frame["cluster_id"] = [id_mapping[int(value)] for value in raw_labels]

    # ---- 全量 distance_to_center（标准化空间，与 representatives 同口径） ----
    # 伪标签反哺（curation.pseudo_labels）需要全部合格脉冲的距离，
    # 而不只是代表样本；口径 = StandardScaler 空间欧氏距离。
    assigned_distances = np.asarray(kmeans.transform(scaled), dtype=float)
    frame["distance_to_center"] = assigned_distances[
        np.arange(len(frame)), np.asarray(raw_labels, dtype=int)
    ]

    # ---- 聚类模型落盘（scaler + kmeans，供伪标签邻域/复算距离使用） ----
    import joblib

    joblib.dump(
        {
            "scaler": scaler,
            "kmeans": kmeans,
            "feature_columns": list(CLUSTER_FEATURE_COLUMNS),
            "random_state": int(params.random_state),
        },
        run.tables_dir / "cluster_model.joblib",
    )

    # ---- PCA 仅用于可视化 ----
    if len(frame) >= 2:
        pca = PCA(n_components=2, random_state=params.random_state)
        coordinates = pca.fit_transform(scaled)
        frame["pca_1"] = coordinates[:, 0]
        frame["pca_2"] = coordinates[:, 1]
    else:  # pragma: no cover - len>=2 已保证
        frame["pca_1"] = np.nan
        frame["pca_2"] = np.nan

    silhouette_value = None
    if len(frame) > selected_k and len(set(frame["raw_cluster"])) > 1:
        silhouette_value = float(silhouette_score(scaled, frame["raw_cluster"]))

    atomic_write_csv(frame, run.tables_dir / "clustered_pulses.csv")
    atomic_write_csv(
        pd.DataFrame(records),
        run.tables_dir / "k_selection.csv",
    )
    centers_frame = pd.DataFrame(
        centers_original,
        columns=CLUSTER_FEATURE_COLUMNS,
    )
    centers_frame.insert(
        0,
        "cluster_id",
        [id_mapping[index] for index in range(len(centers_original))],
    )
    centers_frame.insert(
        0,
        "raw_cluster",
        list(range(len(centers_original))),
    )
    atomic_write_csv(centers_frame, run.tables_dir / "cluster_centers.csv")

    # ---- 代表样本（真实 WAV + 统一声谱图） ----
    from .representatives import select_and_export

    representatives = select_and_export(
        run,
        frame,
        scaled,
        kmeans.cluster_centers_,
        id_mapping,
        params,
    )

    # ---- cluster 汇总表 ----
    cluster_ids = sorted(frame["cluster_id"].unique().tolist())
    rep_counts = (
        representatives["cluster_id"].value_counts().to_dict()
        if not representatives.empty
        else {}
    )
    summary_rows = []
    for cluster_id in cluster_ids:
        group = frame[frame["cluster_id"] == cluster_id]
        raw_cluster = int(group["raw_cluster"].iloc[0])
        row = {
            "cluster_id": cluster_id,
            "cluster_label": f"Cluster {cluster_id}",
            "pulse_count": int(len(group)),
            "pulse_fraction": float(len(group) / len(frame)),
            "recording_count": int(group["source_file"].nunique()),
            "representative_count": int(rep_counts.get(cluster_id, 0)),
        }
        for column_index, column in enumerate(CLUSTER_FEATURE_COLUMNS):
            row[f"center_{column}"] = float(
                centers_original[raw_cluster, column_index]
            )
        summary_rows.append(row)
    summary_frame = pd.DataFrame(summary_rows)
    atomic_write_csv(summary_frame, run.tables_dir / "cluster_summary.csv")

    # ---- 图表 ----
    labels = sorted(set(id_mapping.values()))
    save_pca_plot(
        frame,
        labels=labels,
        output_path=run.figures_dir / "cluster_pca.png",
        label_prefix="Cluster ",
        label_column="cluster_id",
        title="Curation clusters (PCA) — Cluster ≠ species",
    )
    save_k_selection_plot(
        records,
        selected_k,
        run.figures_dir / "cluster_quality.png",
    )

    # ---- 审核数据初始化（保留已有专家标注） ----
    from .annotations import AnnotationStore

    store = AnnotationStore.load(run.run_dir)
    store.initialize_from_representatives(representatives)

    # ---- run metadata ----
    run.metadata.update(
        {
            "cluster_count": len(cluster_ids),
            "selected_k": int(selected_k),
            "silhouette_score": silhouette_value,
            "random_state": int(params.random_state),
            "representatives_per_strategy": {
                "center": params.n_center,
                "boundary": params.n_boundary,
                "random": params.n_random,
                "outlier": params.n_outlier,
            },
            "params": params.to_dict(),
            "cluster_warning": CLUSTER_WARNING,
        }
    )

    print(
        f"[聚类] 完成：{len(cluster_ids)} 个 Cluster，"
        f"轮廓系数 "
        + (f"{silhouette_value:.3f}" if silhouette_value is not None else "未计算")
        + f"，代表样本 {len(representatives)} 个"
    )

    return {
        "ok": True,
        "selected_k": int(selected_k),
        "cluster_count": len(cluster_ids),
        "cluster_ids": cluster_ids,
        "silhouette": silhouette_value,
        "accepted_pulses": int(len(frame)),
        "representative_count": int(len(representatives)),
        "k_records": records,
        "warning": CLUSTER_WARNING,
    }
