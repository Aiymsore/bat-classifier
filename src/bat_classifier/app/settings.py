"""桌面 App 设置：工作区、声型标签、数据来源的持久化。

注意：本模块不得 import bat_classifier.config——
config 的路径在导入期由 BAT_WORKSPACE 环境变量决定，
必须先由 settings.apply() 写好环境变量，再导入库代码。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

SETTINGS_FILENAME = "app_settings.json"

DEFAULT_LABELS = ["A", "B", "C", "D"]
DEFAULT_DESCRIPTIONS = {
    "A": "短促宽带下降FM型",
    "B": "高频窄带或CF型",
    "C": "低频窄带或缓变型",
    "D": "复杂、中间或混合声型",
}

# ⑥ 已确认物种模型的 uncertain 阈值默认值。
# 与 config.SPECIES_UNCERTAIN_THRESHOLD 保持一致——本模块不得 import
# bat_classifier.config（路径在导入期由 BAT_WORKSPACE 决定），故在此重复常量。
DEFAULT_SPECIES_UNCERTAIN_THRESHOLD = 0.60


def default_workspace() -> Path:
    """默认工作区：打包态为 exe 旁的 workspace/，源码态为仓库根目录。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "workspace"
    return Path(__file__).resolve().parents[3]


@dataclass
class AppSettings:
    workspace: Path = field(default_factory=default_workspace)
    labels: list[str] = field(default_factory=lambda: list(DEFAULT_LABELS))
    type_descriptions: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_DESCRIPTIONS)
    )
    data_source: str = ""
    # ③/④/⑦ 共用：当前选中的两级模型文件路径（models/ 下的某个 .joblib）。
    # 留空表示用默认模型（② 训练产物 models/noise_bat_ABCD_model.joblib）。
    active_model_path: str = ""
    # ⑥ 已确认物种模型：当前选择的模型目录（models/species/<name>）
    species_model_path: str = ""
    species_uncertain_threshold: float = DEFAULT_SPECIES_UNCERTAIN_THRESHOLD
    # 主窗口几何：0 / -1 表示未保存，启动时按屏幕可用区域自适应
    window_width: int = 0
    window_height: int = 0
    window_x: int = -1
    window_y: int = -1
    # 底部运行日志面板是否收起
    log_collapsed: bool = False

    @property
    def settings_path(self) -> Path:
        return self.workspace / SETTINGS_FILENAME

    def apply(self) -> None:
        """把工作区写入环境变量（必须在导入 bat_classifier 之前调用）。"""
        import os

        self.workspace.mkdir(parents=True, exist_ok=True)
        os.environ["BAT_WORKSPACE"] = str(self.workspace)

    def save(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        payload = {
            "workspace": str(self.workspace),
            "labels": list(self.labels),
            "type_descriptions": dict(self.type_descriptions),
            "data_source": self.data_source,
            "active_model_path": self.active_model_path,
            "species_model_path": self.species_model_path,
            "species_uncertain_threshold": float(
                self.species_uncertain_threshold
            ),
            "window_width": int(self.window_width),
            "window_height": int(self.window_height),
            "window_x": int(self.window_x),
            "window_y": int(self.window_y),
            "log_collapsed": bool(self.log_collapsed),
        }
        self.settings_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, workspace: Path | None = None) -> "AppSettings":
        """从工作区读取设置；workspace 为 None 时用默认工作区。"""
        settings = cls()
        if workspace is not None:
            settings.workspace = Path(workspace).resolve()

        settings_path = settings.settings_path
        if settings_path.exists():
            try:
                payload = json.loads(
                    settings_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                return settings

            if isinstance(payload.get("workspace"), str) and workspace is None:
                settings.workspace = Path(payload["workspace"])
            labels = payload.get("labels")
            if isinstance(labels, list) and all(
                isinstance(item, str) for item in labels
            ):
                settings.labels = list(labels)
            descriptions = payload.get("type_descriptions")
            if isinstance(descriptions, dict):
                settings.type_descriptions = {
                    str(key): str(value)
                    for key, value in descriptions.items()
                }
            if isinstance(payload.get("data_source"), str):
                settings.data_source = payload["data_source"]
            if isinstance(payload.get("active_model_path"), str):
                settings.active_model_path = payload["active_model_path"]
            if isinstance(payload.get("species_model_path"), str):
                settings.species_model_path = payload["species_model_path"]
            threshold = payload.get("species_uncertain_threshold")
            if isinstance(threshold, (int, float)) and 0.0 < float(threshold) < 1.0:
                settings.species_uncertain_threshold = float(threshold)
            for key, attr in (
                ("window_width", "window_width"),
                ("window_height", "window_height"),
                ("window_x", "window_x"),
                ("window_y", "window_y"),
            ):
                value = payload.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    setattr(settings, attr, value)
            if isinstance(payload.get("log_collapsed"), bool):
                settings.log_collapsed = payload["log_collapsed"]

        return settings
