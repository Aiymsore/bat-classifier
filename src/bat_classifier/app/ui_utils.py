"""布局小工具：给长页面/长表单加“按需出现”的滚动条。

只做布局包装，不含任何业务逻辑：
- `make_scrollable`：把一整块内容包进 QScrollArea，内容超出可视高度时
  出现垂直滚动条（滚轮直接可用），超宽时出现水平滚动条；
- 表格、列表这类自带滚动条的部件留在各自面板内部即可，外层滚动条只负责
  “页面级”滚动，两者互不冲突。

典型用法：
- ⑥ 页面：**整页滚动**——表单 + 审核工作台 + 模型对照结果全部进同一个滚动区，
  小屏幕一路滚到底，下半部分永远不会被挤出可视区域；
- ⑦ 页面：上半部分表单包进滚动区，下半部分结果区固定占满剩余高度，
  两者用纵向 QSplitter 分栏。
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
