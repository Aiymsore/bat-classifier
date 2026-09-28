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

        return settings
