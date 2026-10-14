"""桌面 App 设置：工作区、窗口与日志显示的持久化。

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

def default_workspace() -> Path:
    """默认工作区：打包态为 exe 旁的 workspace/，源码态为仓库根目录。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "workspace"
    return Path(__file__).resolve().parents[3]


@dataclass
class AppSettings:
    workspace: Path = field(default_factory=default_workspace)
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
