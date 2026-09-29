"""布局小工具：给长表单区加“按需出现”的滚动条。

只做布局包装，不含任何业务逻辑：
- `make_scrollable`：把一整块表单/分组框包进 QScrollArea，内容超出可视高度时
  出现垂直滚动条（滚轮直接可用），超宽时出现水平滚动条；
- 表格、审核工作台这类自带滚动条的主体不要包进来，避免双层滚动条。

典型用法（⑥/⑦ 页面）：上半部分表单包进滚动区，下半部分结果区固定占满剩余高度，
两者用纵向 QSplitter 分栏，小屏幕上表单区自动出滚动条而不是把界面裁掉。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QScrollArea, QWidget


def make_scrollable(content: QWidget) -> QScrollArea:
    """把 content 包进滚动区：超高出垂直滚动条，超宽出水平滚动条（按需出现）。"""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setWidget(content)
    return scroll
