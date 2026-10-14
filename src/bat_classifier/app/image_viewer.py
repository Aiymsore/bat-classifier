"""结果图像查看组件：大图对话框 + 列表/预览画廊。

供第①、第②模块浏览声谱图使用。
"""

from __future__ import annotations

import csv
from pathlib import Path

from PySide6.QtCore import QEvent, QPoint, QSize, Qt, QUrl, QTimer
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

ZOOM_STEP = 1.15
MIN_ZOOM = 0.25
MAX_ZOOM = 8.0


def _zoomed_size(size: QSize, factor: float) -> QSize:
    """按比例计算 Qt 图片尺寸，避免 QSize 对浮点乘法的版本差异。"""
    return QSize(
        max(1, round(size.width() * factor)),
        max(1, round(size.height() * factor)),
    )


class ZoomableImageScrollArea(QScrollArea):
    """带 Ctrl+滚轮缩放钩子的图片滚动区域。

    普通滚轮仍交给 QScrollArea 处理，用于上下/左右滚动；只有按住 Ctrl
    时才触发缩放回调。把事件过滤器同时装到 viewport 和图片控件上，保证
    鼠标停在图片本身或空白区域时都能使用同一快捷键。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._zoom_handler = None
        self._dragging = False
        self._last_drag_pos = QPoint()
        self.viewport().installEventFilter(self)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def set_zoom_handler(self, handler) -> None:
        self._zoom_handler = handler

    def setWidget(self, widget: QWidget) -> None:  # noqa: N802 - Qt API
        old_widget = self.widget()
        if old_widget is not None:
            old_widget.removeEventFilter(self)
        super().setWidget(widget)
        widget.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        if event.type() == QEvent.Type.Wheel and self._handle_zoom_wheel(event):
            return True
        if event.type() == QEvent.Type.MouseButtonPress and self._start_drag(event):
            return True
        if event.type() == QEvent.Type.MouseMove and self._drag_move(event):
            return True
        if event.type() == QEvent.Type.MouseButtonRelease and self._end_drag(event):
            return True
        return super().eventFilter(watched, event)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._handle_zoom_wheel(event):
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._start_drag(event):
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._drag_move(event):
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._end_drag(event):
            return
        super().mouseReleaseEvent(event)

    def _handle_zoom_wheel(self, event) -> bool:
        if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            return False
        delta = event.angleDelta().y()
        if delta == 0:
            delta = event.pixelDelta().y()
        if delta == 0 or self._zoom_handler is None:
            return False
        self._zoom_handler(ZOOM_STEP if delta > 0 else 1.0 / ZOOM_STEP)
        event.accept()
        return True

    def _start_drag(self, event) -> bool:
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        self._dragging = True
        self._last_drag_pos = event.position().toPoint()
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        event.accept()
        return True

    def _drag_move(self, event) -> bool:
        if not self._dragging:
            return False
        position = event.position().toPoint()
        delta = position - self._last_drag_pos
        self._last_drag_pos = position
        self.horizontalScrollBar().setValue(
            self.horizontalScrollBar().value() - delta.x()
        )
        self.verticalScrollBar().setValue(
            self.verticalScrollBar().value() - delta.y()
        )
        event.accept()
        return True

    def _end_drag(self, event) -> bool:
        if event.button() != Qt.MouseButton.LeftButton or not self._dragging:
            return False
        self._dragging = False
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        event.accept()
        return True


class ImagePreviewDialog(QDialog):
    """模态大图查看器：适应窗口与 1:1 原始大小切换。"""

    def __init__(self, path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"图像查看 - {path.name}")
        self.resize(1100, 780)

        self._pixmap = QPixmap(str(path))
        self._fit = True
        self._zoom = 1.0

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet("background-color: #202020;")
        self._scroll = ZoomableImageScrollArea()
        self._scroll.setWidget(self._label)
        self._scroll.setWidgetResizable(False)
        self._scroll.set_zoom_handler(self._zoom_by)
        self._scroll.setToolTip("按住 Ctrl 滚动鼠标滚轮可放大或缩小；普通滚轮用于滚动图片")

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

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if self._fit:
            self._apply_scale()

    def _toggle_zoom(self) -> None:
        self._fit = not self._fit
        self._zoom = 1.0
        self._apply_scale()

    def _zoom_by(self, factor: float) -> None:
        self._zoom = min(MAX_ZOOM, max(MIN_ZOOM, self._zoom * factor))
        self._apply_scale()

    def _apply_scale(self) -> None:
        if self._pixmap.isNull():
            self._label.setText("无法加载图像")
            self._label.adjustSize()
            return

        if self._fit:
            base = self._pixmap.scaled(
                self._scroll.viewport().size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        else:
            base = self._pixmap
        target_size = _zoomed_size(base.size(), self._zoom)
        scaled = self._pixmap.scaled(
            target_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._label.setPixmap(scaled)
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
        self._preview_scroll = ZoomableImageScrollArea()
        self._preview_scroll.setWidget(self._preview)
        self._preview_scroll.setWidgetResizable(False)
        self._preview_scroll.set_zoom_handler(self._zoom_preview_by)
        self._preview_scroll.setToolTip("按住 Ctrl 滚动鼠标滚轮可放大或缩小；普通滚轮用于滚动图片")
        self._preview_pixmap = QPixmap()
        self._preview_zoom = 1.0

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

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if self._preview_zoom == 1.0:
            self._apply_preview_scale()

    # ---------- 数据加载 ----------


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
        self._preview_pixmap = pixmap
        self._preview_zoom = 1.0
        self._apply_preview_scale()

    def _zoom_preview_by(self, factor: float) -> None:
        if self._preview_pixmap.isNull():
            return
        self._preview_zoom = min(
            MAX_ZOOM, max(MIN_ZOOM, self._preview_zoom * factor)
        )
        self._apply_preview_scale()

    def _apply_preview_scale(self) -> None:
        if self._preview_pixmap.isNull():
            return
        base = self._preview_pixmap.scaled(
            self._preview_scroll.viewport().size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        scaled = self._preview_pixmap.scaled(
            _zoomed_size(base.size(), self._preview_zoom),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._preview.setPixmap(scaled)
        self._preview.adjustSize()

    def _show_placeholder(self, text: str) -> None:
        self._preview_pixmap = QPixmap()
        self._preview_zoom = 1.0
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


class PulseImagePane(QWidget):
    """单张 pulse 大图：默认适应窗口，缩放与滚动不依赖缩略图库。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = QPixmap()
        self._mode = "fit"
        self._zoom = 1.0
        self._label = QLabel("选中一个 pulse 查看波形与声谱图。")
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet("background:#202020; color:#c0c0c0;")
        self._scroll = ZoomableImageScrollArea()
        self._scroll.setWidget(self._label)
        self._scroll.setWidgetResizable(False)
        self._scroll.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self._scroll.setMinimumHeight(280)
        self._scroll.set_zoom_handler(self._zoom_by)
        self._scroll.viewport().installEventFilter(self)
        self._scroll.setToolTip("Ctrl+滚轮缩放；拖动平移；滚动查看完整图片。")
        controls = QHBoxLayout()
        for text, callback in (
            ("适应宽度", lambda: self._set_mode("width")),
            ("适应窗口", lambda: self._set_mode("fit")),
            ("1:1", lambda: self._set_mode("original")),
            ("放大", lambda: self._zoom_by(ZOOM_STEP)),
            ("缩小", lambda: self._zoom_by(1 / ZOOM_STEP)),
        ):
            button = QPushButton(text)
            button.clicked.connect(callback)
            controls.addWidget(button)
        controls.addStretch(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._scroll, 1)
        layout.addLayout(controls)

    def show_image(self, path: Path | None):
        self._pixmap = QPixmap(str(path)) if path is not None else QPixmap()
        self._mode = "fit"
        self._zoom = 1.0
        self._scroll.horizontalScrollBar().setValue(0)
        self._scroll.verticalScrollBar().setValue(0)
        if self._pixmap.isNull():
            self._label.setPixmap(QPixmap())
            self._label.setText("未生成图片。" if path else "选中一个 pulse 查看波形与声谱图。")
            self._label.adjustSize()
        else:
            self._label.clear()
            self._render()
            QTimer.singleShot(0, self._render)

    def eventFilter(self, watched, event):
        if watched is self._scroll.viewport() and event.type() == QEvent.Type.Resize:
            QTimer.singleShot(0, self._render)
        return super().eventFilter(watched, event)

    def _set_mode(self, mode):
        self._mode = mode
        self._zoom = 1.0
        self._render()

    def _zoom_by(self, factor):
        self._zoom = min(MAX_ZOOM, max(MIN_ZOOM, self._zoom * factor))
        self._render()

    def _render(self):
        if self._pixmap.isNull():
            return
        viewport = self._scroll.viewport().size()
        if self._mode == "original":
            base = 1.0
        elif self._mode == "fit":
            base = min(viewport.width() / self._pixmap.width(), viewport.height() / self._pixmap.height())
        else:
            base = max(1, viewport.width() - 4) / self._pixmap.width()
        size = _zoomed_size(self._pixmap.size(), max(base, 0.001) * self._zoom)
        scaled = self._pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._label.setPixmap(scaled)
        self._label.resize(scaled.size())


class PulseSpectrogramDialog(QDialog):
    """按最终类别浏览第七模块生成的逐 pulse 声谱图。"""

    def __init__(
        self,
        run_dir: Path,
        *,
        labels: list[str] | None = None,
        unknown_label: str = "unknown",
        segment_id: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._run_dir = Path(run_dir)
        self._segment_id = str(segment_id or "")
        self._unknown_label = str(unknown_label or "unknown")
        self._provided_labels = [str(label) for label in (labels or [])]
        self._rows: list[dict[str, str]] = []
        self._labels: list[str] = []

        self.setWindowTitle("逐声波声谱图（按类别查看）")
        self.resize(1320, 820)

        self._category_combo = QComboBox()
        self._scope_check = QCheckBox("显示全部片段")
        self._scope_check.setChecked(not bool(self._segment_id))
        self._scope_check.setEnabled(bool(self._segment_id))
        self._scope_check.toggled.connect(self._reload_gallery)
        self._count_label = QLabel("")
        self._count_label.setStyleSheet("color:#555555;")
        self._category_combo.currentIndexChanged.connect(self._reload_gallery)

        control_row = QHBoxLayout()
        control_row.addWidget(QLabel("蝙蝠类别"))
        control_row.addWidget(self._category_combo, stretch=1)
        control_row.addWidget(self._scope_check)
        control_row.addWidget(self._count_label)

        hint = QLabel(
            "图片按类别关联：合并图可包含多个叫声，分开图对应独立分量；unknown 也会列出。"
            "图像上方为放大的时域波形，下方为彩色频谱和频率数据。"
            "预览区支持 Ctrl+滚轮缩放、按住鼠标左键拖动。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#666666;")

        self._gallery = ImageGalleryPane(
            empty_hint="该类别没有生成逐声波声谱图。"
        )
        layout = QVBoxLayout(self)
        layout.addLayout(control_row)
        layout.addWidget(hint)
        layout.addWidget(self._gallery, stretch=1)

        self._load_index()

    def _load_index(self) -> None:
        csv_path = self._run_dir / "pulse_predictions.csv"
        if csv_path.exists():
            try:
                with csv_path.open(
                    "r", encoding="utf-8-sig", newline=""
                ) as handle:
                    for row in csv.DictReader(handle):
                        label = str(row.get("predicted_label", "")).strip()
                        relative = str(
                            row.get("pulse_spectrogram_file", "")
                        ).strip()
                        if not label or not relative:
                            continue
                        path = self._run_dir / Path(relative)
                        if path.is_file():
                            self._rows.append(
                                {
                                    "label": label,
                                    "segment_id": str(
                                        row.get("segment_id", "")
                                    ),
                                    "pulse_id": str(row.get("pulse_id", "")),
                                    "path": str(path),
                                }
                            )
            except (OSError, UnicodeError, csv.Error):
                self._rows = []

        # 以调用方传入的模板/模型类别为主，再补充 CSV 中实际出现的类别。
        ordered = []
        for label in self._provided_labels + [self._unknown_label]:
            label = str(label).strip()
            if label and label not in ordered:
                ordered.append(label)
        for row in self._rows:
            label = row["label"]
            if label not in ordered:
                ordered.append(label)
        self._labels = ordered
        self._category_combo.blockSignals(True)
        self._category_combo.clear()
        self._category_combo.addItems(self._labels)
        self._category_combo.blockSignals(False)
        self._reload_gallery()

    def _reload_gallery(self) -> None:
        label = self._category_combo.currentText().strip()
        show_all = self._scope_check.isChecked()
        rows = [row for row in self._rows if row["label"] == label]
        if self._segment_id and not show_all:
            rows = [
                row for row in rows
                if row.get("segment_id", "") == self._segment_id
            ]
        rows.sort(
            key=lambda row: (
                row.get("segment_id", ""),
                self._pulse_order(row.get("pulse_id", "0")),
                row.get("path", ""),
            )
        )
        # 同一合并图片关联多个 pulse；在当前类别下只展示一次。
        paths = list(dict.fromkeys(row["path"] for row in rows))
        self._gallery.set_paths([Path(path) for path in paths])
        scope = "全部片段" if show_all else "当前片段"
        self._count_label.setText(f"{len(paths)} 张（{scope}）")

    @staticmethod
    def _pulse_order(value: object) -> int:
        text = str(value or "0").strip()
        if text.lower().startswith("p"):
            text = text[1:]
        try:
            return int(text or 0)
        except (TypeError, ValueError):
            return 0
