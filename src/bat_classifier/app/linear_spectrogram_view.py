"""① 片段线性声谱图查看器：PNG + 可移动的频率光标（Avisoft 风格）。

原理：出图时（visualization._plot_segment_call_overview）会把坐标区的
figure 分数位置、频率范围写到同名 `.json` sidecar；这里读回来后把
PNG 画到控件上，再按鼠标 y 坐标换算出频率，画一条横线并在右端给出
Hz 读数——与 Avisoft SASLab 的频率游标等价。

不重算声谱、不读 WAV：看到的像素与保存的 PNG 完全一致。
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPaintEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


def load_spectrogram_meta(png_path: Path) -> dict | None:
    """读取出图时记录的几何 sidecar；缺失/损坏返回 None（回退静态查看）。"""
    sidecar = Path(png_path).with_suffix(".json")
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    bbox = payload.get("axes_bbox")
    limits = payload.get("freq_ylim")
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or not isinstance(limits, list)
        or len(limits) != 2
    ):
        return None
    return payload


class LinearSpectrogramView(QWidget):
    """显示 PNG，并叠加随鼠标移动的频率横线与 Hz 读数。"""

    def __init__(self, pixmap: QPixmap, meta: dict, parent: QWidget | None = None):
        super().__init__(parent)
        self._pixmap = pixmap
        self._meta = meta
        limits = [float(v) for v in meta.get("freq_ylim") or (0.0, 1.0)]
        initial = meta.get("cursor_init_hz")
        try:
            initial = float(initial) if initial is not None else None
        except (TypeError, ValueError):
            initial = None
        if initial is None:
            initial = (limits[0] + limits[1]) / 2.0
        self._cursor_hz: float = initial
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._dragging = False
        self._last_drag_pos = QPointF()
        self.setMouseTracking(True)
        self.setMinimumSize(520, 340)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def sizeHint(self) -> QSize:
        return QSize(1140, 700)

    # ---------------------------------------------------------- 几何换算
    def _scaled_pixmap(self) -> QPixmap:
        base = self._pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        if self._zoom == 1.0:
            return base
        return self._pixmap.scaled(
            QSize(
                max(1, round(base.width() * self._zoom)),
                max(1, round(base.height() * self._zoom)),
            ),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

    def _zoom_by(self, factor: float) -> None:
        self._zoom = min(8.0, max(0.25, self._zoom * factor))
        self._clamp_pan(self._scaled_pixmap())
        self._update_cursor_shape()
        self.update()

    def _image_origin(self, scaled: QPixmap) -> QPointF:
        return QPointF(
            (self.width() - scaled.width()) / 2.0 + self._pan.x(),
            (self.height() - scaled.height()) / 2.0 + self._pan.y(),
        )

    def _clamp_pan(self, scaled: QPixmap) -> None:
        limit_x = max(0.0, (scaled.width() - self.width()) / 2.0)
        limit_y = max(0.0, (scaled.height() - self.height()) / 2.0)
        self._pan.setX(min(limit_x, max(-limit_x, self._pan.x())))
        self._pan.setY(min(limit_y, max(-limit_y, self._pan.y())))

    def _update_cursor_shape(self) -> None:
        scaled = self._scaled_pixmap()
        if scaled.width() > self.width() or scaled.height() > self.height():
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def _axes_rect(self, scaled: QPixmap) -> QRectF | None:
        """坐标区（声谱图面板）在控件像素里的矩形。"""
        if self._pixmap.isNull() or scaled.width() <= 0 or scaled.height() <= 0:
            return None
        left, bottom, width, height = (
            float(value) for value in self._meta["axes_bbox"]
        )
        origin = self._image_origin(scaled)
        return QRectF(
            origin.x() + left * scaled.width(),
            # matplotlib 的 y 从下往上，控件像素从上往下
            origin.y() + (1.0 - bottom - height) * scaled.height(),
            width * scaled.width(),
            height * scaled.height(),
        )

    def _hz_at(self, y: float, rect: QRectF) -> float | None:
        low, high = (float(v) for v in self._meta["freq_ylim"])
        if high <= low or rect.height() <= 0:
            return None
        ratio = (y - rect.top()) / rect.height()
        return high - min(max(ratio, 0.0), 1.0) * (high - low)

    # ---------------------------------------------------------- 绘制
    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(255, 255, 255))

        if self._pixmap.isNull():
            painter.drawText(
                self.rect(), Qt.AlignCenter, "图像无法显示（PNG 缺失或损坏）"
            )
            return

        scaled = self._scaled_pixmap()
        origin = self._image_origin(scaled)
        painter.drawPixmap(origin.toPoint(), scaled)

        rect = self._axes_rect(scaled)
        limits = self._meta.get("freq_ylim") or []
        if rect is None or len(limits) != 2:
            return
        low, high = float(limits[0]), float(limits[1])
        if high <= low or rect.height() <= 0:
            return

        # 横线（Avisoft 游标样式：细黑线贯穿坐标区）
        cursor = min(max(self._cursor_hz, low), high)
        y = rect.bottom() - (cursor - low) / (high - low) * rect.height()
        painter.setPen(QPen(QColor(15, 15, 15), 1.1))
        painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))

        # 右端频率读数
        text = f"{self._cursor_hz:.0f} Hz"
        metrics = painter.fontMetrics()
        box_width = metrics.horizontalAdvance(text) + 14
        box_height = metrics.height() + 6
        box_x = rect.right() - box_width - 4
        box_y = y - box_height - 3
        if box_y < rect.top():
            box_y = y + 3
        box = QRectF(box_x, box_y, box_width, box_height)
        painter.setPen(QPen(QColor(15, 15, 15), 1.0))
        painter.setBrush(QColor(255, 255, 255, 242))
        painter.drawRoundedRect(box, 4, 4)
        painter.setPen(QColor(15, 15, 15))
        painter.drawText(box, Qt.AlignCenter, text)

    # ---------------------------------------------------------- 交互
    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._clamp_pan(self._scaled_pixmap())
        self._update_cursor_shape()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton:
            scaled = self._scaled_pixmap()
            if scaled.width() > self.width() or scaled.height() > self.height():
                self._dragging = True
                self._last_drag_pos = event.position()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            position = event.position()
            delta = position - self._last_drag_pos
            self._last_drag_pos = position
            self._pan += delta
            self._clamp_pan(self._scaled_pixmap())
            self.update()
            event.accept()
            return
        scaled = self._scaled_pixmap()
        rect = self._axes_rect(scaled)
        if rect is None:
            return
        position = event.position()
        if not (
            rect.left() <= position.x() <= rect.right()
            and rect.top() <= position.y() <= rect.bottom()
        ):
            return  # 光标离开坐标区时保持上次读数
        value = self._hz_at(position.y(), rect)
        if value is None:
            return
        self._cursor_hz = float(value)
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            self._update_cursor_shape()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y() or event.pixelDelta().y()
            if delta:
                self._zoom_by(1.15 if delta > 0 else 1.0 / 1.15)
                event.accept()
                return
        super().wheelEvent(event)


class LinearSpectrogramDialog(QDialog):
    """① 的查看窗口：线性声谱图 + 可移动频率光标。"""

    def __init__(
        self,
        png_path: Path,
        meta: dict,
        title: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._png_path = Path(png_path)
        self.setWindowTitle(title or "片段线性声谱图")
        self.resize(1140, 700)

        layout = QVBoxLayout(self)
        self._view = LinearSpectrogramView(
            QPixmap(str(self._png_path)), meta, self
        )
        layout.addWidget(self._view, stretch=1)

        hint = QLabel(
            "移动鼠标：横线跟随光标，读数即该高度的频率（Hz）；"
            "坐标与生成的 PNG 完全一致。按住 Ctrl 滚动鼠标滚轮可放大或缩小。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#666666;")
        layout.addWidget(hint)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        png_button = QPushButton("打开 PNG 原图")
        png_button.clicked.connect(self._open_png)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.accept)
        buttons.addWidget(png_button)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

    def _open_png(self) -> None:
        if not self._png_path.exists():
            return
        from .image_viewer import ImagePreviewDialog

        dialog = ImagePreviewDialog(self._png_path, self)
        dialog.exec()
