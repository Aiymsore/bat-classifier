"""专家标注持久化、Cluster 纯度辅助与监督训练数据集导出。

设计原则（科研语义，务必保持）：
- 每保存一条立即原子写盘（临时文件 → replace），绝不把 300 条标注只留在内存；
- 只有专家实际审核过的 pulse 才有 expert_label，未审核一律留空；
- 绝不因为“同 Cluster”自动把一个物种标签传播给整个 Cluster——
  纯度统计只是给专家的 review hint，不是自动结论。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from ..config import (
    CLUSTER_FEATURE_COLUMNS,
    CURATION_MIXED_FRACTION,
    CURATION_MIXED_MIN_REVIEWED,
)
from .pipeline import atomic_write_csv, read_table

ANNOTATIONS_CSV = "expert_annotations.csv"
TRAINING_DATASET_CSV = "expert_training_dataset.csv"

STATUS_PENDING = "pending"
STATUS_REVIEWED = "reviewed"
STATUS_UNCERTAIN = "uncertain"
STATUS_NOISE = "noise"

REVIEW_STATUSES = (
    STATUS_PENDING,
    STATUS_REVIEWED,
    STATUS_UNCERTAIN,
    STATUS_NOISE,
)
CONFIDENCE_OPTIONS = ("high", "medium", "low", "")

ANNOTATION_COLUMNS = [
    "pulse_id",
    "cluster_id",
    "source_file",
    "source_path",
    "selection_reason",
    "expert_label",
    "expert_confidence",
    "review_status",
    "notes",
    "reviewed_at",
]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class AnnotationStore:
    """expert_annotations.csv 的读写封装（原子写、可反复加载）。"""

    def __init__(self, run_dir: Path, frame: pd.DataFrame) -> None:
        self.run_dir = Path(run_dir)
        self.frame = frame.reset_index(drop=True)

    # ---------------- IO ----------------
    @property
    def path(self) -> Path:
        return self.run_dir / "tables" / ANNOTATIONS_CSV

    @classmethod
    def load(cls, run_dir: Path) -> "AnnotationStore":
        run_dir = Path(run_dir)
        frame = read_table(
            run_dir / "tables" / ANNOTATIONS_CSV,
            key_column="pulse_id",
        )
        if frame.empty:
            frame = pd.DataFrame(columns=ANNOTATION_COLUMNS)
        else:
            for column in ANNOTATION_COLUMNS:
                if column not in frame.columns:
                    frame[column] = ""
                # CSV 空单元格读回来是 NaN，统一归一成空字符串
                frame[column] = frame[column].fillna("")
            frame = frame[ANNOTATION_COLUMNS]
            frame["pulse_id"] = frame["pulse_id"].astype(str)
        return cls(run_dir, frame)

    def save(self) -> None:
        ordered = [
            column
            for column in ANNOTATION_COLUMNS
            if column in self.frame.columns
        ]
        atomic_write_csv(self.frame[ordered], self.path)

    def to_frame(self) -> pd.DataFrame:
        return self.frame.copy()

    # ---------------- 初始化 / 更新 ----------------
    def initialize_from_representatives(
        self,
        representatives: pd.DataFrame,
        save: bool = True,
    ) -> int:
        """把最新代表样本并入标注表：缺失的追加为 pending，已有的刷新来源信息。

        绝不覆盖专家已填写的 label/confidence/status/notes；
        不再是代表样本的旧行也保留（不销毁已有标注）。
        返回新增行数。
        """
        if representatives is None or representatives.empty:
            return 0

        reps = representatives.copy()
        reps["pulse_id"] = reps["pulse_id"].astype(str)
        existing_ids = set(self.frame["pulse_id"].astype(str))

        refresh_columns = [
            column
            for column in ("cluster_id", "source_file", "source_path", "selection_reason")
            if column in reps.columns
        ]
        if not self.frame.empty:
            lookup = reps.set_index("pulse_id")
            for index, row in self.frame.iterrows():
                pulse_id = str(row["pulse_id"])
                if pulse_id in lookup.index:
                    source_row = lookup.loc[pulse_id]
                    for column in refresh_columns:
                        value = source_row[column]
                        self.frame.at[index, column] = (
                            value if isinstance(value, str) else str(value)
                        )

        new_rows: list[dict] = []
        for _, row in reps.iterrows():
            pulse_id = str(row["pulse_id"])
            if pulse_id in existing_ids:
                continue
            record = {column: "" for column in ANNOTATION_COLUMNS}
            record["pulse_id"] = pulse_id
            for column in refresh_columns:
                record[column] = row.get(column, "")
            record["expert_label"] = ""
            record["expert_confidence"] = ""
            record["review_status"] = STATUS_PENDING
            record["notes"] = ""
            record["reviewed_at"] = ""
            new_rows.append(record)

        if new_rows:
            additions = pd.DataFrame(new_rows)[ANNOTATION_COLUMNS]
            if self.frame.empty:
                self.frame = additions
            else:
                self.frame = pd.concat(
                    [self.frame, additions],
                    ignore_index=True,
                )
        if save:
            self.save()
        return len(new_rows)

    def update(
        self,
        pulse_id: str,
        expert_label: str,
        expert_confidence: str,
        review_status: str,
        notes: str,
        save: bool = True,
    ) -> dict:
        """更新一条标注并立即落盘；返回更新后的行。"""
        if review_status not in REVIEW_STATUSES:
            raise ValueError(f"未知审核状态：{review_status}")
        if expert_confidence not in CONFIDENCE_OPTIONS:
            raise ValueError(f"未知置信度：{expert_confidence}")

        matches = self.frame.index[
            self.frame["pulse_id"].astype(str) == str(pulse_id)
        ].tolist()
        if not matches:
            raise KeyError(f"标注表中没有该代表样本：{pulse_id}")

        index = matches[0]
        self.frame.at[index, "expert_label"] = str(expert_label).strip()
        self.frame.at[index, "expert_confidence"] = str(expert_confidence).strip()
        self.frame.at[index, "review_status"] = review_status
        self.frame.at[index, "notes"] = str(notes).strip()
        self.frame.at[index, "reviewed_at"] = (
            "" if review_status == STATUS_PENDING else _now()
        )
        if save:
            self.save()
        return self.frame.loc[index].to_dict()

    # ---------------- 统计 ----------------
    def counts(self) -> dict:
        if self.frame.empty:
            return {
                "total": 0,
                "reviewed": 0,
                "pending": 0,
                "uncertain": 0,
                "noise": 0,
            }
        status_counts = self.frame["review_status"].value_counts().to_dict()
        return {
            "total": int(len(self.frame)),
            "reviewed": int(status_counts.get(STATUS_REVIEWED, 0)),
            "pending": int(status_counts.get(STATUS_PENDING, 0)),
            "uncertain": int(status_counts.get(STATUS_UNCERTAIN, 0)),
            "noise": int(status_counts.get(STATUS_NOISE, 0)),
        }

    def cluster_stats(self, cluster_id: str) -> dict:
        """一个 Cluster 的标注统计（review hint，不用于自动传播标签）。"""
        empty = {
            "cluster_id": str(cluster_id),
            "reviewed_count": 0,
            "pending_count": 0,
            "uncertain_count": 0,
            "noise_count": 0,
            "label_counts": {},
            "dominant_label": "",
            "dominant_fraction": 0.0,
            "mixed": False,
        }
        if self.frame.empty or "cluster_id" not in self.frame.columns:
            return empty

        rows = self.frame[
            self.frame["cluster_id"].astype(str) == str(cluster_id)
        ]
        if rows.empty:
            return empty

        label_counts: dict[str, int] = {}
        reviewed_count = pending_count = uncertain_count = noise_count = 0
        for _, row in rows.iterrows():
            status = str(row.get("review_status", STATUS_PENDING))
            label = str(row.get("expert_label", "") or "").strip()
            if status == STATUS_REVIEWED:
                reviewed_count += 1
                key = label or "（未填写标签）"
                label_counts[key] = label_counts.get(key, 0) + 1
            elif status == STATUS_NOISE:
                noise_count += 1
                key = label or "Noise"
                label_counts[key] = label_counts.get(key, 0) + 1
            elif status == STATUS_UNCERTAIN:
                uncertain_count += 1
                key = label or "Uncertain"
                label_counts[key] = label_counts.get(key, 0) + 1
            else:
                pending_count += 1

        dominant_label = ""
        dominant_fraction = 0.0
        reviewed_labels = {
            key: value
            for key, value in label_counts.items()
            if key != "（未填写标签）"
        }
        if reviewed_labels and reviewed_count:
            dominant_label = max(reviewed_labels, key=reviewed_labels.get)
            dominant_fraction = reviewed_labels[dominant_label] / reviewed_count

        mixed = (
            reviewed_count >= CURATION_MIXED_MIN_REVIEWED
            and bool(dominant_label)
            and dominant_fraction < CURATION_MIXED_FRACTION
        )

        return {
            "cluster_id": str(cluster_id),
            "reviewed_count": reviewed_count,
            "pending_count": pending_count,
            "uncertain_count": uncertain_count,
            "noise_count": noise_count,
            "label_counts": dict(
                sorted(label_counts.items(), key=lambda item: -item[1])
            ),
            "dominant_label": dominant_label,
            "dominant_fraction": float(dominant_fraction),
            "mixed": mixed,
        }


def export_training_dataset(
    run_dir: Path,
    include_noise: bool = False,
) -> Path:
    """导出监督训练数据集（只有 reviewed + 非空标签；noise 可选）。

    保留 source_file / source_path / audio_hash / start_s / end_s，
    未来必须按原始录音分组划分 train/valid/test，避免数据泄漏。
    """
    run_dir = Path(run_dir)
    representatives = read_table(
        run_dir / "tables" / "representatives.csv",
        key_column="pulse_id",
    )
    store = AnnotationStore.load(run_dir)
    annotations = store.to_frame()

    if annotations.empty or representatives.empty:
        output = pd.DataFrame(
            columns=[
                "pulse_id",
                "cluster_id",
                "audio_hash",
                "source_file",
                "source_path",
                "start_s",
                "end_s",
                "duration_ms",
                "selection_reason",
                "expert_label",
                "expert_confidence",
                "review_status",
                "notes",
                "reviewed_at",
                *CLUSTER_FEATURE_COLUMNS,
                "snr_db",
            ]
        )
        path = run_dir / "tables" / TRAINING_DATASET_CSV
        atomic_write_csv(output, path)
        return path

    representatives = representatives.copy()
    representatives["pulse_id"] = representatives["pulse_id"].astype(str)
    annotations["pulse_id"] = annotations["pulse_id"].astype(str)

    merged = annotations.merge(
        representatives,
        on="pulse_id",
        how="inner",
        suffixes=("", "_rep"),
    )

    label = merged["expert_label"].astype(str).str.strip()
    status = merged["review_status"].astype(str)
    mask = (status == STATUS_REVIEWED) & (label != "")
    if include_noise:
        noise_mask = status == STATUS_NOISE
        merged.loc[noise_mask & (label == ""), "expert_label"] = "noise"
        label = merged["expert_label"].astype(str).str.strip()
        mask = mask | (noise_mask & (label != ""))

    exported = merged.loc[mask].copy()

    wanted = [
        "pulse_id",
        "cluster_id",
        "audio_hash",
        "source_file",
        "source_path",
        "start_s",
        "end_s",
        "duration_ms",
        "selection_reason",
        "expert_label",
        "expert_confidence",
        "review_status",
        "notes",
        "reviewed_at",
        *CLUSTER_FEATURE_COLUMNS,
        "snr_db",
    ]
    for column in wanted:
        if column not in exported.columns:
            exported[column] = ""
    exported = exported[wanted]

    path = run_dir / "tables" / TRAINING_DATASET_CSV
    atomic_write_csv(exported, path)
    return path
