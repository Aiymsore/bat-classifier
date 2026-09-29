"""专家真值划分（⑥）：Expert Train Pool / Expert Evaluation Set。

三类数据角色（见 docs/pseudo_label_feedback.md）：

- Expert Train Pool：训练 / 校准 / OOF scoring / 伪标签传播依据；
- Expert Evaluation Set：**冻结的考试卷**，只做 Precision / Recall / F1 /
  Macro-F1 等最终评估，永久禁止参与训练、校准、OOF scoring、
  Cluster purity 传播依据、propagated_from；
- Pseudo Training Dataset：非专家真值，只以低权重进训练，绝不进评估。

硬约束：

- 拆分单位是 ``source_recording``：整条录音只能属于 Train 或 Evaluation
  之一，**绝不拆同一录音的 pulse**（否则背景噪声/设备/个体/环境会同时
  出现在训练与评估中，使 F1 虚高）；
- 划分必须 group-aware + class-aware：两侧尽量都有每个物种的专家真值；
  做不到时明确报 ``INSUFFICIENT_EVAL_COVERAGE``，**绝不为了凑 F1 破坏
  recording 隔离**；
- 冻结后 Evaluation recording IDs 不随新增标注变化；重新生成划分必须
  显式触发并递增 ``split_version``（新旧模型指标不再完全可比）。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    EVAL_CONTAMINATED_FLAG,
    EXPERT_SPLIT_EVAL_FRACTION,
    EXPERT_SPLIT_MIN_EVAL_PER_CLASS,
    EXPERT_SPLIT_MIN_TRAIN_PER_CLASS,
    EXPERT_SPLIT_RANDOM_STATE,
    EXPERT_SPLIT_STATUS_INSUFFICIENT,
    EXPERT_SPLIT_STATUS_OK,
)
from .species_model import _clean_label, source_recording_series

SPLIT_FILENAME = "expert_split.json"

# 新标注默认去向：Train 录音上的新标注 → Train Pool；
# Evaluation 录音上的新标注 → parked（录音隔离优先，见 module docstring）。
NEW_LABEL_DESTINATION_TRAIN = "train_pool"
NEW_LABEL_DESTINATION_PARKED = "parked_on_evaluation_recording"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# 污染断言（EVAL_CONTAMINATED）
# ---------------------------------------------------------------------------
class EvalContaminationError(ValueError):
    """评估集录音泄漏进训练/传播数据：立即终止，绝不静默降级。"""

    def __init__(self, context: str, offending: list[str]) -> None:
        self.context = str(context)
        self.offending = sorted({str(value) for value in offending})
        super().__init__(
            f"{EVAL_CONTAMINATED_FLAG}：{self.context} 中出现评估集录音 "
            f"{self.offending}（Evaluation Set 绝不参与训练/校准/OOF/传播）。"
        )


def check_eval_isolation(
    evaluation_recordings,
    other_recordings,
) -> list[str]:
    """返回混入的评估录音 ID 列表（空列表 = 干净）。"""
    eval_set = {_clean_label(value) for value in evaluation_recordings}
    eval_set.discard("")
    other = {_clean_label(value) for value in other_recordings}
    other.discard("")
    return sorted(eval_set & other)


def assert_eval_isolation(
    evaluation_recordings,
    other_recordings,
    context: str,
) -> None:
    """训练 / 校准 / OOF scoring / cluster gate / 伪标签生成前统一调用。"""
    offending = check_eval_isolation(evaluation_recordings, other_recordings)
    if offending:
        raise EvalContaminationError(context, offending)


# ---------------------------------------------------------------------------
# 划分算法（group-aware + class-aware，确定性可复现）
# ---------------------------------------------------------------------------
def _recording_class_map(data: pd.DataFrame) -> dict[str, set[str]]:
    groups = source_recording_series(data).map(_clean_label)
    labels = data["expert_label"].map(_clean_label)
    mapping: dict[str, set[str]] = {}
    for recording, label in zip(groups, labels):
        if not recording or not label:
            continue
        mapping.setdefault(recording, set()).add(label)
    return mapping


def make_expert_split(
    frame: pd.DataFrame,
    eval_fraction: float = EXPERT_SPLIT_EVAL_FRACTION,
    random_state: int = EXPERT_SPLIT_RANDOM_STATE,
) -> dict:
    """把专家真值按 source_recording 拆成 Train Pool / Evaluation Set。

    算法（确定性，random_state 只用于打破平局的固定置换）：

    1. 统计每个物种的录音数；录音数不足以两侧都覆盖的物种
       （< MIN_TRAIN + MIN_EVAL）记入 ``insufficient_classes``，
       其录音**强制留在 Train**（Train 侧必须能学到该物种）；
    2. Phase 1（覆盖）：贪心选录音进 Evaluation，优先覆盖尚未获得
       Evaluation 真值的物种（集合覆盖贪心，平局按固定置换序）；
    3. Phase 2（补量）：按固定置换序补入录音，直到逼近目标比例，
       全程保证“每个物种 Train 侧至少保留 MIN_TRAIN_PER_CLASS 条录音”；
    4. 输出两侧录音清单、逐类 pulse 计数、状态与原因。

    **绝不拆同一录音的 pulse**：单位是整条 source_recording。
    """
    if frame is None or frame.empty:
        raise ValueError("没有可用于划分的专家标注数据。")
    if "expert_label" not in frame.columns:
        raise ValueError("专家数据缺少 expert_label 列。")

    data = frame.copy()
    data["expert_label"] = data["expert_label"].map(_clean_label)
    data = data[data["expert_label"] != ""].reset_index(drop=True)
    if data.empty:
        raise ValueError("没有可用于划分的专家标注数据（expert_label 全为空）。")

    rec_classes = _recording_class_map(data)
    if not rec_classes:
        raise ValueError("专家数据缺少录音来源列（source_recording / source_file / source_path）。")

    class_recs: dict[str, set[str]] = {}
    for recording, classes in rec_classes.items():
        for label in classes:
            class_recs.setdefault(label, set()).add(recording)

    need = int(EXPERT_SPLIT_MIN_TRAIN_PER_CLASS) + int(EXPERT_SPLIT_MIN_EVAL_PER_CLASS)
    insufficient_classes = sorted(
        label for label, recs in class_recs.items() if len(recs) < need
    )

    recordings = sorted(rec_classes)
    rng = np.random.default_rng(int(random_state))
    perm = list(rng.permutation(len(recordings)))
    perm_rank = {recordings[perm_index]: rank for rank, perm_index in enumerate(perm)}

    def feasible(eval_set: set[str], recording: str) -> bool:
        """把 recording 放进 Evaluation 后，每个物种 Train 侧仍够 MIN_TRAIN。"""
        for label in rec_classes[recording]:
            remaining_train = len(class_recs[label] - eval_set - {recording})
            if remaining_train < int(EXPERT_SPLIT_MIN_TRAIN_PER_CLASS):
                return False
        return True

    total = len(recordings)
    target = int(round(total * float(eval_fraction)))
    target = max(1, min(target, total - 1))

    eval_set: set[str] = set()
    # Phase 1：物种覆盖（只有 ≥ need 条录音的物种才可能两侧都有）
    uncovered = {
        label for label, recs in class_recs.items() if len(recs) >= need
    }
    while uncovered:
        candidates = [
            recording
            for recording in recordings
            if recording not in eval_set
            and (rec_classes[recording] & uncovered)
            and feasible(eval_set, recording)
        ]
        if not candidates:
            break
        pick = max(
            candidates,
            key=lambda rec: (len(rec_classes[rec] & uncovered), -perm_rank[rec]),
        )
        eval_set.add(pick)
        uncovered -= rec_classes[pick]

    # Phase 2：补到目标比例（录音级，不拆 pulse）
    for recording in sorted(recordings, key=lambda rec: perm_rank[rec]):
        if len(eval_set) >= target:
            break
        if recording in eval_set or not feasible(eval_set, recording):
            continue
        eval_set.add(recording)

    train_set = set(recordings) - eval_set

    groups = source_recording_series(data).map(_clean_label)
    is_eval = groups.isin(eval_set)
    eval_rows = data.loc[is_eval]
    train_rows = data.loc[~is_eval]

    per_class_train = {
        str(label): int(count)
        for label, count in train_rows["expert_label"].value_counts().items()
    }
    per_class_eval = {
        str(label): int(count)
        for label, count in eval_rows["expert_label"].value_counts().items()
    }
    uncovered_classes = sorted(
        label
        for label, recs in class_recs.items()
        if (recs & train_set) and not (recs & eval_set)
    )

    status = EXPERT_SPLIT_STATUS_OK
    if insufficient_classes or uncovered_classes or not eval_set:
        status = EXPERT_SPLIT_STATUS_INSUFFICIENT

    # 冻结真值快照：Gold Evaluation 的 recording / pulse / expert truth 全部
    # 固定在这一刻，之后标注文件再怎么增长/修改都不影响考试卷。
    evaluation_truth = [
        {
            "pulse_id": str(row["pulse_id"]),
            "expert_label": _clean_label(row["expert_label"]),
            "source_recording": _clean_label(recording),
        }
        for (_, row), recording in zip(
            eval_rows.iterrows(), groups.loc[eval_rows.index]
        )
    ]

    return {
        "split_version": 1,
        "created_at": _now(),
        "random_state": int(random_state),
        "eval_fraction_target": float(eval_fraction),
        "status": status,
        "insufficient_classes": insufficient_classes,
        "uncovered_classes": uncovered_classes,
        "train_recordings": sorted(train_set),
        "evaluation_recordings": sorted(eval_set),
        "evaluation_pulse_ids": sorted(eval_rows.get("pulse_id", pd.Series(dtype=str)).astype(str)),
        "evaluation_truth": evaluation_truth,
        "train_recording_count": len(train_set),
        "evaluation_recording_count": len(eval_set),
        "train_pulse_count": int(len(train_rows)),
        "evaluation_pulse_count": int(len(eval_rows)),
        "per_class_train_count": dict(sorted(per_class_train.items())),
        "per_class_evaluation_count": dict(sorted(per_class_eval.items())),
        "warning": (
            "Evaluation Set 是冻结考试卷：绝不参与训练/校准/OOF/伪标签传播；"
            "模型预测 / pseudo / 高置信结果绝不允许进入 Evaluation。"
        ),
    }


def apply_expert_split(
    frame: pd.DataFrame,
    split: dict,
) -> dict:
    """按冻结划分把专家表拆成 train / evaluation / parked 三部分。

    - train：Train 录音上的专家行（含冻结后新增标注——新标注默认进 Train Pool）；
    - evaluation：冻结的 Evaluation pulse（``evaluation_pulse_ids`` 优先；
      旧版划分文件没有该字段时按录音整体取）；
    - parked：Evaluation 录音上**冻结之后**新增的标注——录音隔离优先，
      既不进训练也不进冻结评估集（想启用须显式重新生成划分）。
    """
    if frame is None or frame.empty:
        return {"train": frame, "evaluation": frame, "parked": frame}
    data = frame.copy()
    groups = source_recording_series(data).map(_clean_label)
    train_recordings = {
        _clean_label(value) for value in (split.get("train_recordings") or [])
    }
    eval_recordings = {
        _clean_label(value) for value in (split.get("evaluation_recordings") or [])
    }

    on_train = groups.isin(train_recordings)
    on_eval = groups.isin(eval_recordings)

    frozen_ids = {
        _clean_label(value) for value in (split.get("evaluation_pulse_ids") or [])
    }
    if frozen_ids and "pulse_id" in data.columns:
        pulse_ids = data["pulse_id"].map(_clean_label)
        eval_rows = data.loc[on_eval & pulse_ids.isin(frozen_ids)]
        parked_rows = data.loc[on_eval & ~pulse_ids.isin(frozen_ids)]
    else:
        eval_rows = data.loc[on_eval]
        parked_rows = data.loc[on_eval & False]  # 旧划分：无冻结 pulse 清单

    train_rows = data.loc[on_train]
    unknown = data.loc[~on_train & ~on_eval]
    if not unknown.empty:
        # 划分文件未覆盖的录音：保守处理，绝不进评估（考试卷不许长大）。
        parked_rows = pd.concat([parked_rows, unknown], ignore_index=True)

    return {
        "train": train_rows.reset_index(drop=True),
        "evaluation": eval_rows.reset_index(drop=True),
        "parked": parked_rows.reset_index(drop=True),
    }


# ---------------------------------------------------------------------------
# 存储（expert_split.json：冻结 + 版本）
# ---------------------------------------------------------------------------
def expert_split_path(run_dir: Path) -> Path:
    return Path(run_dir) / "tables" / SPLIT_FILENAME


class ExpertSplitStore:
    """``tables/expert_split.json`` 的读写封装（原子写、冻结语义）。"""

    def __init__(self, run_dir: Path, record: dict | None) -> None:
        self.run_dir = Path(run_dir)
        self.record = record

    @property
    def path(self) -> Path:
        return expert_split_path(self.run_dir)

    @property
    def frozen(self) -> bool:
        return bool(self.record)

    @property
    def evaluation_recordings(self) -> list[str]:
        return list((self.record or {}).get("evaluation_recordings") or [])

    @classmethod
    def load(cls, run_dir: Path) -> "ExpertSplitStore":
        path = expert_split_path(run_dir)
        record = None
        if path.exists():
            import json

            with open(path, "r", encoding="utf-8") as handle:
                record = json.load(handle)
        return cls(run_dir, record)

    def save(self, record: dict, replace: bool = False) -> dict:
        """写入划分。已存在时必须显式 ``replace=True``（递增 split_version）。"""
        import json

        from .curation.pipeline import atomic_write_json

        if self.record is not None and not replace:
            raise ValueError(
                "专家数据划分已冻结；如需重新划分请显式选择“重新生成划分”"
                "（新旧模型指标将不再完全可比）。"
            )
        record = dict(record)
        if self.record is not None:
            record["split_version"] = int(self.record.get("split_version") or 1) + 1
        else:
            record.setdefault("split_version", 1)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.path, record)
        self.record = record
        # 考卷存档：Gold Evaluation v{N} 不可变归档（绝不覆盖旧版本）
        try:
            archive_gold_evaluation(self.run_dir, gold_record_from_split(record))
        except FileExistsError:
            pass  # 同版本已归档：考试卷不许被改写
        return record


def new_label_destination(
    split: dict | None,
    recording: str,
) -> str:
    """冻结后新增专家标注的去向（见 module docstring）。"""
    if not split:
        return NEW_LABEL_DESTINATION_TRAIN
    eval_recordings = {
        _clean_label(value) for value in (split.get("evaluation_recordings") or [])
    }
    if _clean_label(recording) in eval_recordings:
        return NEW_LABEL_DESTINATION_PARKED
    return NEW_LABEL_DESTINATION_TRAIN


def split_summary(split: dict | None, frame: pd.DataFrame | None = None) -> dict:
    """UI 展示用摘要（总量 / 两侧 / 逐类 / 录音数 / 状态）。"""
    if not split:
        return {
            "frozen": False,
            "status": "",
            "total": int(len(frame)) if frame is not None else 0,
        }
    per_class_train = dict(split.get("per_class_train_count") or {})
    per_class_eval = dict(split.get("per_class_evaluation_count") or {})
    rows = []
    for label in sorted(set(per_class_train) | set(per_class_eval)):
        train_count = int(per_class_train.get(label, 0))
        eval_count = int(per_class_eval.get(label, 0))
        rows.append(
            {
                "species": label,
                "train_count": train_count,
                "evaluation_count": eval_count,
                "note": (
                    ""
                    if eval_count > 0
                    else "该物种当前无法计算 Recall / F1（Evaluation 真值为 0）"
                ),
            }
        )
    return {
        "frozen": True,
        "split_version": int(split.get("split_version") or 1),
        "status": str(split.get("status") or ""),
        "insufficient_classes": list(split.get("insufficient_classes") or []),
        "uncovered_classes": list(split.get("uncovered_classes") or []),
        "train_pulse_count": int(split.get("train_pulse_count") or 0),
        "evaluation_pulse_count": int(split.get("evaluation_pulse_count") or 0),
        "total": int(split.get("train_pulse_count") or 0)
        + int(split.get("evaluation_pulse_count") or 0),
        "train_recording_count": int(split.get("train_recording_count") or 0),
        "evaluation_recording_count": int(split.get("evaluation_recording_count") or 0),
        "per_class": rows,
        "created_at": str(split.get("created_at") or ""),
    }


# ---------------------------------------------------------------------------
# Gold Evaluation（固定专家真值考试集，v1/v2/... 各自独立、绝不覆盖）
# ---------------------------------------------------------------------------
GOLD_EVAL_DIRNAME = "evaluations"
GOLD_EVAL_FILENAME_PREFIX = "gold_eval_v"


class GoldEvaluationError(ValueError):
    """Evaluation 遭遇伪标签 / 模型结果 / 非人工真值混入。"""


def gold_evaluation_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "tables" / GOLD_EVAL_DIRNAME


def gold_evaluation_path(run_dir: Path, version: int) -> Path:
    return gold_evaluation_dir(run_dir) / f"{GOLD_EVAL_FILENAME_PREFIX}{int(version)}.json"


def list_gold_evaluations(run_dir: Path) -> list[int]:
    directory = gold_evaluation_dir(run_dir)
    if not directory.exists():
        return []
    versions = []
    for path in directory.glob(f"{GOLD_EVAL_FILENAME_PREFIX}*.json"):
        suffix = path.stem.replace(GOLD_EVAL_FILENAME_PREFIX, "")
        if suffix.isdigit():
            versions.append(int(suffix))
    return sorted(versions)


def assert_no_pseudo_labels(frame, context: str) -> None:
    """Evaluation 真值只允许**真正独立的人工专家真值**。

    以下一律拒绝（§四）：label_source=pseudo、pseudo_label 非空、
    review_status 非 reviewed、expert_label 为空。
    不自动修——直接终止并列出违规 pulse。
    """
    if frame is None or len(frame) == 0:
        return
    data = pd.DataFrame(frame)
    pulse_ids = (
        data["pulse_id"].astype(str).tolist()
        if "pulse_id" in data.columns
        else [str(index) for index in data.index]
    )
    offending: list[str] = []

    def _bad_ids(mask: pd.Series) -> list[str]:
        if "pulse_id" in data.columns:
            return [str(value) for value in data.loc[mask, "pulse_id"].astype(str)]
        return [str(value) for value in data.index[mask]]

    if "label_source" in data.columns:
        offending.extend(
            _bad_ids(data["label_source"].map(_clean_label).str.lower() == "pseudo")
        )
    if "pseudo_label" in data.columns:
        offending.extend(_bad_ids(data["pseudo_label"].map(_clean_label) != ""))
    if "review_status" in data.columns:
        offending.extend(
            _bad_ids(data["review_status"].map(_clean_label) != "reviewed")
        )
    if "expert_label" in data.columns:
        offending.extend(_bad_ids(data["expert_label"].map(_clean_label) == ""))
    offending = sorted(set(offending))
    if offending:
        raise GoldEvaluationError(
            f"{context}：只有真正独立的人工专家真值可以进入 Evaluation；"
            f"pseudo / 模型预测 / 未完成人工审核的 pulse 一律禁止。"
            f"违规 pulse：{offending[:20]}"
            + ("..." if len(offending) > 20 else "")
        )
    if not pulse_ids:
        raise GoldEvaluationError(f"{context}：Evaluation 真值为空。")


def gold_record_from_split(split: dict) -> dict:
    """从冻结划分生成 Gold Evaluation 记录（真值取快照，绝不取活标注）。"""
    truth = [
        {
            "pulse_id": str(row.get("pulse_id", "")),
            "expert_label": _clean_label(row.get("expert_label", "")),
            "source_recording": _clean_label(row.get("source_recording", "")),
        }
        for row in (split.get("evaluation_truth") or [])
    ]
    per_class: dict[str, int] = {}
    for row in truth:
        per_class[row["expert_label"]] = per_class.get(row["expert_label"], 0) + 1
    return {
        "gold_eval_version": int(split.get("split_version") or 1),
        "created_at": str(split.get("created_at") or _now()),
        "description": "专家数据划分冻结的 Evaluation Set（专家真值快照）",
        "evaluation_recordings": [
            _clean_label(value) for value in (split.get("evaluation_recordings") or [])
        ],
        "evaluation_pulse_ids": [
            _clean_label(value) for value in (split.get("evaluation_pulse_ids") or [])
        ],
        "truth": truth,
        "per_class_count": dict(sorted(per_class.items())),
        "eval_labels": "expert_only",
        "immutable": True,
        "warning": (
            "Fixed Gold Evaluation：self-training 系统之外的外部锚点。"
            "只回答'模型是真的进步了，还是只是越来越相信自己生成的 pseudo label'。"
            "绝不自动扩张；模型预测 / pseudo / 高置信结果绝不允许进入。"
        ),
    }


def archive_gold_evaluation(run_dir: Path, record: dict) -> Path:
    """不可变归档 Gold Evaluation：已存在同版本 → 拒绝覆盖。"""
    from .curation.pipeline import atomic_write_json

    version = int(record.get("gold_eval_version") or 1)
    path = gold_evaluation_path(run_dir, version)
    if path.exists():
        raise FileExistsError(
            f"Gold Evaluation v{version} 已存在，绝不覆盖旧版本：{path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, record)
    return path


def load_gold_evaluation(run_dir: Path, version: int) -> dict:
    import json

    path = gold_evaluation_path(run_dir, version)
    if not path.exists():
        raise FileNotFoundError(f"Gold Evaluation v{version} 不存在：{path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def create_gold_evaluation(
    run_dir: Path,
    frame: pd.DataFrame,
    description: str = "",
    training_recordings=None,
    version: int | None = None,
) -> dict:
    """创建 Gold Evaluation v{n}（未来 v2/... 流程，§七/§二十八）。

    - 只接受**真正独立的人工专家真值**（伪标签 / 模型预测一律拒绝）；
    - 与现有训练数据隔离（training_recordings ∩ Evaluation = ∅）；
    - **绝不覆盖旧版本**：v{n} 已存在 → 拒绝；v1 永远保留；
    - 传入的 frame 是"新一批完整 recording 的专家审核结果"，
      绝不允许从 pseudo pool 抽取或按模型高置信生成。
    """
    if frame is None or len(frame) == 0:
        raise GoldEvaluationError("没有可用于 Gold Evaluation 的专家真值。")
    data = pd.DataFrame(frame).copy()
    if "pulse_id" not in data.columns:
        raise GoldEvaluationError("Gold Evaluation 缺少 pulse_id 列。")
    if "expert_label" not in data.columns:
        raise GoldEvaluationError(
            "Gold Evaluation 缺少 expert_label 列（只接受人工专家真值）。"
        )
    assert_no_pseudo_labels(data, "创建 Gold Evaluation")

    recordings = [
        _clean_label(value) for value in source_recording_series(data).tolist()
    ]
    if training_recordings is not None:
        assert_eval_isolation(
            sorted(set(recordings)),
            training_recordings,
            "创建 Gold Evaluation（与训练数据隔离）",
        )

    versions = list_gold_evaluations(run_dir)
    if version is None:
        version = (max(versions) + 1) if versions else 1
    version = int(version)
    if gold_evaluation_path(run_dir, version).exists():
        raise FileExistsError(
            f"Gold Evaluation v{version} 已存在，绝不覆盖旧版本。"
            "如需新考试集请创建下一个版本（旧版本永远保留）。"
        )

    truth = [
        {
            "pulse_id": str(row["pulse_id"]),
            "expert_label": _clean_label(row["expert_label"]),
            "source_recording": _clean_label(recording),
        }
        for (_, row), recording in zip(data.iterrows(), recordings)
    ]
    per_class: dict[str, int] = {}
    for row in truth:
        per_class[row["expert_label"]] = per_class.get(row["expert_label"], 0) + 1

    record = {
        "gold_eval_version": version,
        "created_at": _now(),
        "description": str(description or f"Gold Evaluation v{version}"),
        "evaluation_recordings": sorted(set(recordings)),
        "evaluation_pulse_ids": sorted(str(value) for value in data["pulse_id"]),
        "truth": truth,
        "per_class_count": dict(sorted(per_class.items())),
        "eval_labels": "expert_only",
        "immutable": True,
        "warning": (
            "Fixed Gold Evaluation：外部锚点，不是 self-training pool。"
            "创建后 recording / pulse / expert truth 全部固定。"
        ),
    }
    archive_gold_evaluation(run_dir, record)
    return record
