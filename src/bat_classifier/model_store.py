"""两级模型文件仓库：列出 / 校验 / 导入 ③ 预测与 ⑦ 片段识别可用的模型。

设计约束（与 README「各模块功能」一致）：

- ②「模型训练」页训练出的模型固定写入 ``models/noise_bat_ABCD_model.joblib``，
  永远是默认模型，**任何导入操作都不会覆盖它**；
- 用户导入的模型统一复制到 ``models/imported/<文件名>.joblib``，与 ② 的训练产物
  分开存放、互不覆盖——因此 ② 训练的模型可以被使用，但不是唯一可用模型；
- 中间件（``acoustic_type_ABCD_model.joblib`` / ``noise_bat_gate_model.joblib``）
  不是完整的两级模型，不进入可用列表；
- 本模块只做文件与结构校验，**绝不训练模型**（训练只在 ② 页面 / run.py）。
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import COMBINED_DEMO_MODEL_PATH, MODEL_DIR

# ② 训练产物（默认模型）
TRAINED_MODEL_PATH = COMBINED_DEMO_MODEL_PATH

# 用户导入模型的存放目录（models/imported/）
IMPORTED_MODEL_DIR = MODEL_DIR / "imported"

MODEL_SUFFIX = ".joblib"

# ② 训练时同步写出的中间件，不是可直接推理的完整两级模型
AUX_MODEL_NAMES = frozenset(
    {
        "acoustic_type_ABCD_model.joblib",
        "noise_bat_gate_model.joblib",
    }
)

ORIGIN_TRAINED = "trained"
ORIGIN_IMPORTED = "imported"
ORIGIN_OTHER = "other"

_ORIGIN_TEXT = {
    ORIGIN_TRAINED: "② 训练",
    ORIGIN_IMPORTED: "导入",
    ORIGIN_OTHER: "其他",
}


@dataclass(frozen=True)
class ModelEntry:
    """一个可（或本应可）被 ③/⑦ 加载的模型文件。"""

    path: Path
    origin: str = ORIGIN_OTHER
    valid: bool = False
    error: str = ""
    labels: tuple[str, ...] = ()

    @property
    def size_mb(self) -> float:
        try:
            return self.path.stat().st_size / (1024 * 1024)
        except OSError:
            return 0.0

    @property
    def origin_text(self) -> str:
        return _ORIGIN_TEXT.get(self.origin, "其他")

    @property
    def display_name(self) -> str:
        if self.origin == ORIGIN_TRAINED:
            head = "② 训练模型（默认）"
        elif self.origin == ORIGIN_IMPORTED:
            head = "导入模型"
        else:
            head = "其他模型"
        return f"{head} · {self.path.name}"


def payload_error(payload: object) -> str | None:
    """校验 payload 是否能被 ③ 预测 / ⑦ 逐叫声分类使用。

    返回 None 表示可用；否则返回人类可读的失败原因。
    只看结构，不重新训练、不修改任何内容。
    """
    if not isinstance(payload, dict):
        return "文件内容不是模型字典（dict），无法加载。"

    acoustic = payload.get("acoustic_types")
    if not isinstance(acoustic, dict) or not acoustic:
        return "缺少 acoustic_types 段：这不是本项目的两级预测模型。"

    # 校准分类器（CalibratedClassifierCV）与临时分类器二选一即可推理
    if (
        "temporary_classifier" not in acoustic
        and "calibrated_classifier" not in acoustic
    ):
        return (
            "缺少 acoustic_types.temporary_classifier / calibrated_classifier："
            "无法逐叫声分类。"
        )

    feature_columns = acoustic.get("feature_columns")
    if not feature_columns:
        return "缺少 acoustic_types.feature_columns：特征顺序未知，无法推理。"

    return None


def inspect_model(
    path: Path | str,
    origin: str | None = None,
) -> ModelEntry:
    """加载并校验单个模型文件，返回 ModelEntry（不抛异常）。"""
    import joblib

    model_path = Path(path)
    if origin is None:
        origin = (
            ORIGIN_TRAINED
            if model_path == TRAINED_MODEL_PATH
            else ORIGIN_OTHER
        )

    if not model_path.exists():
        return ModelEntry(model_path, origin, False, "文件不存在。")

    try:
        payload = joblib.load(model_path)
    except Exception as error:  # noqa: BLE001 - 损坏/不兼容文件都转成可读信息
        return ModelEntry(model_path, origin, False, f"加载失败：{error}")

    error = payload_error(payload)
    if error:
        return ModelEntry(model_path, origin, False, error)

    acoustic = payload.get("acoustic_types") or {}
    labels = tuple(str(label) for label in (acoustic.get("labels") or []))
    return ModelEntry(model_path, origin, True, "", labels)


def list_models(model_dir: Path | str | None = None) -> list[ModelEntry]:
    """列出可用模型：② 训练模型排第一，其后是导入模型，最后是其他。

    只扫描 ``*.joblib``，跳过中间件文件；逐个做结构校验，
    不可用的条目也会返回（valid=False / error），好让 UI 显示原因。
    """
    root = Path(model_dir) if model_dir else MODEL_DIR
    imported_dir = root / IMPORTED_MODEL_DIR.name

    candidates: list[tuple[Path, str]] = []
    if root.exists():
        for path in sorted(root.glob(f"*{MODEL_SUFFIX}")):
            if path.name in AUX_MODEL_NAMES:
                continue
            origin = (
                ORIGIN_TRAINED if path == TRAINED_MODEL_PATH else ORIGIN_OTHER
            )
            candidates.append((path, origin))
    if imported_dir.exists():
        for path in sorted(imported_dir.glob(f"*{MODEL_SUFFIX}")):
            candidates.append((path, ORIGIN_IMPORTED))

    order = {ORIGIN_TRAINED: 0, ORIGIN_IMPORTED: 1, ORIGIN_OTHER: 2}
    entries = [inspect_model(path, origin) for path, origin in candidates]
    entries.sort(
        key=lambda entry: (
            order.get(entry.origin, 3),
            entry.path.name,
        )
    )
    return entries


def import_model_file(
    source: Path | str,
    model_dir: Path | str | None = None,
) -> Path:
    """把用户选择的模型文件复制进模型仓库，返回复制后的目标路径。

    - 结构不合法（不能被 ③/⑦ 使用）→ ValueError，不写任何文件；
    - 目标名冲突 → 自动加 ``_1`` / ``_2`` 后缀，**绝不覆盖已有模型**，
      尤其不会覆盖 ② 训练出的 ``noise_bat_ABCD_model.joblib``。
    """
    source_path = Path(source).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"模型文件不存在：{source_path}")
    if source_path.suffix.lower() != MODEL_SUFFIX:
        raise ValueError(f"只支持 {MODEL_SUFFIX} 模型文件。")

    entry = inspect_model(source_path, origin=ORIGIN_IMPORTED)
    if not entry.valid:
        raise ValueError(f"模型文件不可用：{entry.error}")

    root = Path(model_dir) if model_dir else MODEL_DIR
    imported_dir = root / IMPORTED_MODEL_DIR.name
    imported_dir.mkdir(parents=True, exist_ok=True)

    target = imported_dir / source_path.name
    if source_path.resolve() == target.resolve():
        return target

    index = 1
    while target.exists():
        target = imported_dir / f"{source_path.stem}_{index}{source_path.suffix}"
        index += 1

    shutil.copy2(source_path, target)
    return target
