"""recording-level OOF / cross-fit 评分（⑥ 伪标签反哺专用，模型角色 B）。

三条安全不变式（docs/pseudo_label_feedback.md §1 / §5）：

- **I9**：决定某个 pulse 是否成为伪标签的模型概率，必须来自训练时
  完全没见过该 pulse 所属 ``source_recording`` 的评分模型；
- **I10**：OOF 评分模型只以 **Expert Train Pool** 为训练真值（任何轮次都
  不喂伪标签），且只用于评分，绝不作为 Production 模型、绝不参与
  评估/校准真值；
- **I11**：OOF 概率不可用时零传播（``oof_probability_unavailable`` /
  ``insufficient_oof_training_data``），**绝不回退** Production 模型概率。

评分覆盖规则（两条都满足 I9）：

1. 候选所属录音 **在 Expert Train Pool 中** → 由“把该录音留出”的
   fold 模型评分（训练集不含该录音的任何 pulse）；
2. 候选所属录音 **不在 Expert Train Pool 中**（该录音从未贡献专家标注）
   → 由“全 Train Pool 评分模型”评分（训练集只有别的录音，同样没见过
   该录音）；
3. 录音在 Train Pool 中但其 fold 无效（类别不足等）→ 不可用，零传播。

分组恒为 ``source_recording``（复用 ``source_recording_series``），绝不退化
成普通随机 KFold。
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from .config import (
    PSEUDO_OOF_FOLDS,
    PSEUDO_OOF_MIN_RECORDINGS,
    SPECIES_CALIBRATION_METHOD,
    SPECIES_FEATURE_COLUMNS,
    SPECIES_MIN_CLASS_SAMPLES,
    SPECIES_MIN_RECORDINGS_FOR_CALIBRATION,
)
from .features import numeric_feature_frame
from .species_model import (
    _clean_label,
    _fit_calibrated,
    _resolve_calibration_method,
    build_calibration_folds,
    make_species_classifier,
    source_recording_series,
)

FULL_POOL_FOLD_ID = "full_train_pool"

OOF_UNAVAILABLE_MISMATCH = "oof_probability_unavailable"
OOF_UNAVAILABLE_INSUFFICIENT = "insufficient_oof_training_data"

OOF_COLUMNS = [
    "oof_top1_label",
    "oof_top1_probability",
    "oof_top2_probability",
    "oof_probability_margin",
    "oof_all_class_probabilities",
    "oof_probability_kind",
    "oof_fold_id",
    "oof_scoring_model_version",
    "oof_train_recordings_hash",
    "oof_available",
    "oof_unavailable_reason",
]


def _scoring_version(train_pulse_ids, feature_columns, random_state: int) -> str:
    payload = json.dumps(
        {
            "train_pulse_ids": sorted(str(value) for value in train_pulse_ids),
            "feature_columns": list(feature_columns),
            "random_state": int(random_state),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _recordings_hash(recordings) -> str:
    payload = json.dumps(
        sorted({_clean_label(value) for value in recordings}), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


class OOFScoring:
    """由 Expert Train Pool 构造的 fold 评分模型组（一次性评分仪器）。"""

    def __init__(
        self,
        feature_columns: list[str],
        folds: list[dict],
        record_to_fold: dict[str, str],
        full_model: dict | None,
        random_state: int,
        notes: list[str],
    ) -> None:
        self.feature_columns = list(feature_columns)
        self.folds = list(folds)
        self.record_to_fold = dict(record_to_fold)
        self.full_model = full_model
        self.random_state = int(random_state)
        self.notes = list(notes)

    # ------------------------------------------------------------------ 构造
    @classmethod
    def build(
        cls,
        train_frame: pd.DataFrame,
        feature_columns: list[str] | None = None,
        random_state: int | None = None,
    ) -> "OOFScoring":
        """在 Expert Train Pool 上做 recording-level cross-fitting。"""
        from .config import EXPERT_SPLIT_RANDOM_STATE

        feature_columns = list(feature_columns or SPECIES_FEATURE_COLUMNS)
        random_state = (
            int(random_state) if random_state is not None
            else int(EXPERT_SPLIT_RANDOM_STATE)
        )
        notes: list[str] = []

        if train_frame is None or train_frame.empty:
            notes.append("Expert Train Pool 为空，OOF 评分不可用。")
            return cls(feature_columns, [], {}, None, random_state, notes)
        if "expert_label" not in train_frame.columns:
            raise ValueError("OOF 评分训练数据缺少 expert_label 列。")
        missing = [c for c in feature_columns if c not in train_frame.columns]
        if missing:
            raise ValueError(f"OOF 评分训练数据缺少特征列：{missing}")

        data = train_frame.copy()
        data["expert_label"] = data["expert_label"].map(_clean_label)
        data = data[data["expert_label"] != ""].reset_index(drop=True)
        groups = source_recording_series(data).map(_clean_label)
        y = data["expert_label"]

        n_recordings = int(groups.nunique())
        if n_recordings < int(PSEUDO_OOF_MIN_RECORDINGS) or y.nunique() < 2:
            notes.append(
                f"独立录音 {n_recordings} 条（< {PSEUDO_OOF_MIN_RECORDINGS}）"
                "或类别不足，OOF 评分不可用：伪标签零传播。"
            )
            return cls(feature_columns, [], {}, None, random_state, notes)

        x_values = numeric_feature_frame(data, feature_columns)
        classes = sorted(set(y))

        def _fit_one(frame_x, frame_y, frame_groups, seed: int):
            """训练一个评分模型（可校准则校准，绝不伪造校准概率）。"""
            calibrated = False
            method = ""
            model = None
            if (
                int(frame_groups.nunique()) >= SPECIES_MIN_RECORDINGS_FOR_CALIBRATION
            ):
                calib_folds = build_calibration_folds(
                    frame_y, frame_groups, classes, random_state=seed
                )
                if calib_folds:
                    method, _note = _resolve_calibration_method(
                        SPECIES_CALIBRATION_METHOD, len(frame_y)
                    )
                    model = _fit_calibrated(
                        make_species_classifier(seed),
                        frame_x,
                        frame_y,
                        method=method,
                        folds=calib_folds,
                    )
                    calibrated = True
            if model is None:
                model = make_species_classifier(seed)
                model.fit(frame_x, frame_y)
            return model, calibrated, method

        # 全 Train Pool 评分模型：给“从未贡献专家标注的录音”打分（I9 仍成立）
        full_model, full_calibrated, full_method = _fit_one(
            x_values, y, groups, random_state
        )
        full_model_entry = {
            "fold_id": FULL_POOL_FOLD_ID,
            "model": full_model,
            "valid": True,
            "is_calibrated": full_calibrated,
            "calibration_method": full_method,
            "train_recordings": sorted(set(groups)),
            "score_recordings": [],
            "scoring_model_version": _scoring_version(
                data.get("pulse_id", pd.Series(dtype=str)),
                feature_columns,
                random_state,
            ),
            "train_recordings_hash": _recordings_hash(set(groups)),
        }

        k = int(max(2, min(int(PSEUDO_OOF_FOLDS), n_recordings)))
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(
            n_splits=k, shuffle=True, random_state=int(random_state)
        )
        folds: list[dict] = []
        record_to_fold: dict[str, str] = {}
        for fold_index, (train_idx, score_idx) in enumerate(
            splitter.split(np.zeros(len(data)), y, groups)
        ):
            train_y = y.iloc[train_idx]
            train_groups = groups.iloc[train_idx]
            score_recordings = sorted(set(groups.iloc[score_idx]))
            fold_id = f"oof_fold_{fold_index + 1}_of_{k}"
            min_class = int(train_y.value_counts().min()) if len(train_y) else 0
            valid = bool(train_y.nunique() >= 2 and min_class >= SPECIES_MIN_CLASS_SAMPLES)
            entry = {
                "fold_id": fold_id,
                "model": None,
                "valid": valid,
                "is_calibrated": False,
                "calibration_method": "",
                "train_recordings": sorted(set(train_groups)),
                "score_recordings": score_recordings,
                "scoring_model_version": "",
                "train_recordings_hash": _recordings_hash(set(train_groups)),
            }
            if valid:
                model, calibrated, method = _fit_one(
                    x_values.iloc[train_idx],
                    train_y,
                    train_groups,
                    random_state + fold_index,
                )
                entry["model"] = model
                entry["is_calibrated"] = calibrated
                entry["calibration_method"] = method
                entry["scoring_model_version"] = _scoring_version(
                    data["pulse_id"].iloc[train_idx]
                    if "pulse_id" in data.columns
                    else pd.Series(dtype=str),
                    feature_columns,
                    random_state + fold_index,
                )
            else:
                notes.append(
                    f"{fold_id} 训练侧类别不足，其名下录音的候选"
                    f"记 {OOF_UNAVAILABLE_INSUFFICIENT}（绝不回退 Production 模型）。"
                )
            folds.append(entry)
            for recording in score_recordings:
                record_to_fold[recording] = fold_id

        return cls(
            feature_columns, folds, record_to_fold, full_model_entry, random_state, notes
        )

    # ------------------------------------------------------------------ 状态
    @property
    def available(self) -> bool:
        return bool(self.full_model is not None)

    @property
    def unavailable_reason(self) -> str:
        return "" if self.available else OOF_UNAVAILABLE_INSUFFICIENT

    def metadata(self) -> dict:
        """OOF 出处链（fold → 训练录音），供测试断言 I9 与落盘审计。"""
        return {
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "random_state": self.random_state,
            "feature_columns": list(self.feature_columns),
            "notes": list(self.notes),
            "folds": [
                {
                    "fold_id": entry["fold_id"],
                    "valid": bool(entry["valid"]),
                    "is_calibrated": bool(entry["is_calibrated"]),
                    "calibration_method": entry["calibration_method"],
                    "train_recordings": list(entry["train_recordings"]),
                    "score_recordings": list(entry["score_recordings"]),
                    "scoring_model_version": entry["scoring_model_version"],
                    "train_recordings_hash": entry["train_recordings_hash"],
                }
                for entry in self.folds
            ],
            "full_model": (
                {
                    "fold_id": self.full_model["fold_id"],
                    "is_calibrated": bool(self.full_model["is_calibrated"]),
                    "train_recordings": list(self.full_model["train_recordings"]),
                    "scoring_model_version": self.full_model["scoring_model_version"],
                    "train_recordings_hash": self.full_model["train_recordings_hash"],
                }
                if self.full_model
                else None
            ),
        }

    def scoring_entry_for(self, recording: str) -> dict | None:
        """该录音应由哪个评分模型打分（I9：训练集不含该录音）。"""
        fold_id = self.record_to_fold.get(_clean_label(recording))
        if fold_id:
            for entry in self.folds:
                if entry["fold_id"] == fold_id:
                    return entry
            return None
        return self.full_model

    # ------------------------------------------------------------------ 评分
    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """对候选/专家样本表打 OOF 概率（绝不写 expert_label）。"""
        result = frame.copy()
        for column in OOF_COLUMNS:
            result[column] = "" if column.endswith(
                ("_label", "_probabilities", "_kind", "_id", "_version", "_hash", "_reason")
            ) else np.nan
        result["oof_available"] = False
        if result.empty:
            return result
        if not self.available:
            result["oof_unavailable_reason"] = self.unavailable_reason
            return result

        groups = source_recording_series(result).map(_clean_label)
        features = numeric_feature_frame(result, self.feature_columns)
        valid_mask = features.notna().all(axis=1)

        for recording, row_positions in groups.groupby(groups).groups.items():
            entry = self.scoring_entry_for(recording)
            positions = list(row_positions)
            if entry is None or entry.get("model") is None or not entry.get("valid"):
                result.loc[positions, "oof_unavailable_reason"] = (
                    OOF_UNAVAILABLE_INSUFFICIENT
                )
                continue
            sub_valid = [p for p in positions if bool(valid_mask.loc[p])]
            skipped = [p for p in positions if not bool(valid_mask.loc[p])]
            if skipped:
                result.loc[skipped, "oof_unavailable_reason"] = OOF_UNAVAILABLE_MISMATCH
            if not sub_valid:
                continue
            x_rows = features.loc[sub_valid, self.feature_columns]
            probabilities = np.asarray(entry["model"].predict_proba(x_rows), dtype=float)
            model_classes = [str(value) for value in entry["model"].classes_]
            order = np.argsort(-probabilities, axis=1, kind="stable")
            top1_index = order[:, 0]
            top2_index = order[:, 1] if probabilities.shape[1] >= 2 else order[:, 0]
            rows = np.arange(probabilities.shape[0])
            top1_prob = probabilities[rows, top1_index]
            top2_prob = probabilities[rows, top2_index]
            kind = (
                "calibrated_probability"
                if entry.get("is_calibrated")
                else "model_support"
            )
            for row_position, row_index in enumerate(sub_valid):
                result.at[row_index, "oof_top1_label"] = model_classes[
                    top1_index[row_position]
                ]
                result.at[row_index, "oof_top1_probability"] = float(top1_prob[row_position])
                result.at[row_index, "oof_top2_probability"] = float(top2_prob[row_position])
                result.at[row_index, "oof_probability_margin"] = float(
                    top1_prob[row_position] - top2_prob[row_position]
                )
                result.at[row_index, "oof_all_class_probabilities"] = json.dumps(
                    {
                        cls: float(probabilities[row_position, col])
                        for col, cls in enumerate(model_classes)
                    },
                    ensure_ascii=False,
                )
                result.at[row_index, "oof_probability_kind"] = kind
                result.at[row_index, "oof_fold_id"] = entry["fold_id"]
                result.at[row_index, "oof_scoring_model_version"] = entry[
                    "scoring_model_version"
                ]
                result.at[row_index, "oof_train_recordings_hash"] = entry[
                    "train_recordings_hash"
                ]
                result.at[row_index, "oof_available"] = True
                result.at[row_index, "oof_unavailable_reason"] = ""
        return result
