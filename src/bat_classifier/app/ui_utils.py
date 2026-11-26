"""布局小工具：给长页面/长表单加“按需出现”的滚动条。

只做布局包装，不含任何业务逻辑：
- `make_scrollable`：把一整块内容包进 QScrollArea，内容超出可视高度时
  出现垂直滚动条（滚轮直接可用），超宽时出现水平滚动条；
- 表格、列表这类自带滚动条的部件留在各自面板内部即可，外层滚动条只负责
  “页面级”滚动，两者互不冲突。

典型用法：
- ① 页面：上半部分表单包进滚动区，下半部分结果区固定占满剩余高度，
  两者用纵向 QSplitter 分栏。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QScrollArea, QWidget


def open_directory(path: Path) -> None:
    """Open a directory explicitly in Explorer instead of using folder associations."""
    target = Path(path).resolve()
    if not target.is_dir():
        raise FileNotFoundError(f"输出目录不存在：{target}")
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetDllDirectoryW.argtypes = [wintypes.DWORD, wintypes.LPWSTR]
        kernel32.GetDllDirectoryW.restype = wintypes.DWORD
        kernel32.SetDllDirectoryW.argtypes = [wintypes.LPCWSTR]
        kernel32.SetDllDirectoryW.restype = wintypes.BOOL
        buffer = ctypes.create_unicode_buffer(32768)
        length = kernel32.GetDllDirectoryW(len(buffer), buffer)
        if length >= len(buffer):
            raise OSError("无法读取 DLL 搜索目录。")
        previous = buffer.value if length else None
        environment = os.environ.copy()
        bundle = getattr(sys, "_MEIPASS", None)
        if bundle:
            bundle = Path(bundle).resolve()
            environment["PATH"] = os.pathsep.join(
                entry for entry in environment.get("PATH", "").split(os.pathsep)
                if entry and not Path(entry.strip('"')).resolve().is_relative_to(bundle)
            )
        # Frozen apps alter DLL lookup for their own libraries. Explorer must
        # inherit Windows' lookup rules, then the app needs its original rules.
        if not kernel32.SetDllDirectoryW(None):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            subprocess.Popen(
                [str(system_root / "explorer.exe"), str(target)],
                env=environment, cwd=str(system_root), close_fds=True,
            )
        finally:
            if not kernel32.SetDllDirectoryW(previous):
                raise ctypes.WinError(ctypes.get_last_error())
    elif not QDesktopServices.openUrl(QUrl.fromLocalFile(str(target))):
        raise OSError("系统未能打开文件管理器。")


def make_scrollable(content: QWidget) -> QScrollArea:
    """把 content 包进滚动区：超高出垂直滚动条，超宽出水平滚动条（按需出现）。"""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setWidget(content)
    return scroll
