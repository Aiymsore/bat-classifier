"""结果图像查看组件：大图对话框 + 列表/预览画廊。

被主窗口的“⑤ 结果图像”页签使用；纯 Qt 控件，无新打包依赖。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

MAX_GALLERY_FILES = 200


class ImagePreviewDialog(QDialog):
    """模态大图查看器：适应窗口与 1:1 原始大小切换。"""

    def __init__(self, path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"图像查看 - {path.name}")
        self.resize(1100, 780)

        self._pixmap = QPixmap(str(path))
        self._fit = True

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet("background-color: #202020;")
        self._scroll = QScrollArea()
        self._scroll.setWidget(self._label)
        self._scroll.setWidgetResizable(False)

        toggle_button = QPushButton("1:1 原始大小")
        toggle_button.clicked.connect(self._toggle_zoom)
        open_button = QPushButton("打开所在文件夹")
        open_button.clicked.connect(
            lambda: QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(path.parent))
            )
        )
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(toggle_button)
        button_row.addWidget(open_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self._scroll)
        layout.addLayout(button_row)

        self._apply_scale()

    def _toggle_zoom(self) -> None:
        self._fit = not self._fit
        self._apply_scale()

    def _apply_scale(self) -> None:
        if self._pixmap.isNull():
            self._label.setText("无法加载图像")
            self._label.adjustSize()
            return

        if self._fit:
            scaled = self._pixmap.scaled(
                self._scroll.viewport().size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self._label.setPixmap(scaled)
        else:
            self._label.setPixmap(self._pixmap)
        self._label.adjustSize()


class ImageGalleryPane(QWidget):
    """左侧文件列表 + 右侧等比预览的图像浏览区。"""

    def __init__(
        self,
        empty_hint: str = "没有找到图像。",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._paths: list[Path] = []

        self._list = QListWidget()
        self._list.setMaximumWidth(360)
        self._list.currentRowChanged.connect(self._on_row_changed)

        self._preview = QLabel(empty_hint)
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setMinimumSize(320, 240)
        self._preview.setStyleSheet("background-color: #202020; color: #c0c0c0;")
        self._preview_scroll = QScrollArea()
        self._preview_scroll.setWidget(self._preview)
        self._preview_scroll.setWidgetResizable(False)

        view_button = QPushButton("查看大图")
        open_button = QPushButton("打开所在文件夹")
        view_button.clicked.connect(self._open_large_view)
        open_button.clicked.connect(self._open_folder)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(view_button)
        button_row.addWidget(open_button)

        layout = QHBoxLayout(self)
        layout.addWidget(self._list)
        right_column = QVBoxLayout()
        right_column.addWidget(self._preview_scroll)
        right_column.addLayout(button_row)
        layout.addLayout(right_column, stretch=1)

    # ---------- 数据加载 ----------

    def load_directory(self, directory: Path) -> int:
        """按修改时间倒序加载目录（含子目录）下的 PNG，返回数量。"""
        if not directory.exists():
            self.set_paths([])
            return 0

        paths = sorted(
            (p for p in directory.rglob("*.png") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:MAX_GALLERY_FILES]
        self.set_paths(paths)
        return len(paths)

    def set_paths(self, paths: list[Path]) -> None:
        self._paths = list(paths)
        self._list.clear()
        for path in self._paths:
            try:
                relative = path.relative_to(path.parents[2])
            except (IndexError, ValueError):
                relative = path
            self._list.addItem(str(relative))
        if self._paths:
            self._list.setCurrentRow(0)
        else:
            self._show_placeholder("没有找到图像。")

    # ---------- 内部行为 ----------

    def _on_row_changed(self, row: int) -> None:
        if not (0 <= row < len(self._paths)):
            return
        path = self._paths[row]
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._show_placeholder(f"无法加载：{path.name}")
            return
        scaled = pixmap.scaled(
            self._preview_scroll.viewport().size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._preview.setPixmap(scaled)
        self._preview.adjustSize()

    def _show_placeholder(self, text: str) -> None:
        self._preview.setPixmap(QPixmap())
        self._preview.setText(text)
        self._preview.adjustSize()

    def _current_path(self) -> Path | None:
        row = self._list.currentRow()
        if 0 <= row < len(self._paths):
            return self._paths[row]
        return None

    def _open_large_view(self) -> None:
        path = self._current_path()
        if path is not None:
            ImagePreviewDialog(path, self).exec()

    def _open_folder(self) -> None:
        path = self._current_path()
        target = path.parent if path is not None else Path.cwd()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))
