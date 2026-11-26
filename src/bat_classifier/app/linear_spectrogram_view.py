"""① 片段线性声谱图查看器：PNG + 可移动的频率光标（Avisoft 风格）。

原理：出图时（visualization._plot_segment_call_overview）会把坐标区的
figure 分数位置、频率范围写到同名 `.json` sidecar；这里读回来后把
PNG 画到控件上，再按鼠标 y 坐标换算出频率，画一条横线并在右端给出
Hz 读数——与 Avisoft SASLab 的频率游标等价。

概览显示 PNG；局部声谱在后台计算并直接绘制数据，复用有界缓存。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPaintEvent, QPainter, QPainterPath, QPen, QPixmap, QTransform
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
    try:
        left, bottom, width, height = (float(value) for value in bbox)
        low, high = (float(value) for value in limits)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not all(math.isfinite(value) for value in (left, bottom, width, height, low, high))
        or min(left, bottom) < 0
        or min(width, height) <= 0
        or left + width > 1.000001
        or bottom + height > 1.000001
        or low < 0
        or high <= low
    ):
        return None
    return payload


def _time_range(meta: dict) -> tuple[float, float] | None:
    try:
        low, high = (float(v) for v in meta["time_xlim"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return (low, high) if math.isfinite(low) and math.isfinite(high) and high > low else None


def resolve_source_audio(png_path: Path, meta: dict, override: dict | None = None) -> dict | None:
    """只为有有效时间坐标、存在对应 WAV 的图启用局部重算。"""
    source = override if override is not None else meta.get("source_audio")
    if not isinstance(source, dict) or _time_range(meta) is None:
        return None
    try:
        path = Path(source["path"])
        if not path.is_absolute():
            path = Path(png_path).parent / path
        offset_s = float(source.get("offset_s", 0.0))
        if not math.isfinite(offset_s) or not path.is_file() or path.suffix.lower() != ".wav":
            return None
    except (KeyError, TypeError, ValueError, OSError):
        return None
    return {"path": str(path.resolve()), "offset_s": offset_s,
            "preprocessed": source.get("preprocessed") is True}


class LinearSpectrogramView(QWidget):
    """显示 PNG，并叠加随鼠标移动的频率横线与 Hz 读数。"""

    rangeSelected = Signal(float, float)
    timeZoomRequested = Signal(float, float)
    timePanRequested = Signal(float, float)

    def __init__(self, pixmap: QPixmap, meta: dict, parent: QWidget | None = None):
        super().__init__(parent)
        self._pixmap = pixmap
        self._detail_data = None
        self._pending_time_range = None
        self._full_time_range = _time_range(meta)
        self._time_dragging = False
        self._pan_preview_range = None
        self._time_drag_moved = False
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
        self._time_navigation = False
        self._selecting = False
        self._selection: tuple[float, float] | None = None
        self.setMouseTracking(True)
        self.setMinimumSize(520, 340)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def sizeHint(self) -> QSize:
        return QSize(1140, 700)

    def set_time_navigation(self, enabled: bool) -> None:
        self._time_navigation = bool(enabled and _time_range(self._meta) is not None)
        self._update_cursor_shape()

    def set_image(self, pixmap: QPixmap, meta: dict) -> None:
        self._pixmap = pixmap
        self._detail_data = None
        self._pending_time_range = None
        self._pan_preview_range = None
        self._spectrum_image = None
        self._wave_path = None
        self._meta = meta
        self._zoom = 1.0
        self._pan = QPointF()
        self._selection = None
        self._selecting = False
        low, high = meta["freq_ylim"]
        self._cursor_hz = min(high, max(low, self._cursor_hz))
        self._update_cursor_shape()
        self.update()

    def set_detail(self, data) -> None:
        # 小型画布仅用于沿用概览的几何换算；局部声谱、波形直接用数据绘制。
        canvas = QPixmap(1400, 800)
        canvas.fill(Qt.GlobalColor.white)
        self.set_image(canvas, dict(data.meta))
        self._detail_data = data
        height, width, _ = data.rgb.shape
        self._spectrum_image = QImage(data.rgb.data, width, height, width * 3,
                                      QImage.Format.Format_RGB888).copy()
        self._wave_path_width = None
        self._update_cursor_shape()
        self.update()

    def _detail_wave_path(self, width: int):
        if self._wave_path is not None and self._wave_path_width == width:
            return self._wave_path
        import numpy as np
        data = self._detail_data
        low, high = data.meta["time_xlim"]
        x = (data.wave_times - low) / (high - low)
        path = QPainterPath()
        if len(x) > width * 2:
            # 每个屏幕列保留真实的正负峰，用独立竖线避免复杂填充路径。
            columns = np.clip((x * width).astype(int), 0, width - 1)
            mins, maxs = np.zeros(width), np.zeros(width)
            np.minimum.at(mins, columns, data.wave_values)
            np.maximum.at(maxs, columns, data.wave_values)
            for i in range(width):
                path.moveTo((i + .5) / width, float(mins[i]))
                path.lineTo((i + .5) / width, float(maxs[i]))
            self._wave_path_columns = True
        else:
            path.moveTo(float(x[0]), 0.)
            for t, v in zip(x, data.wave_values):
                path.lineTo(float(t), float(v))
            path.lineTo(float(x[-1]), 0.)
            path.closeSubpath()
            self._wave_path_columns = False
        self._wave_path, self._wave_path_width = path, width
        return path

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
        if self._time_dragging:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif (self._detail_data is not None and self._time_navigation
              or scaled.width() > self.width() or scaled.height() > self.height()):
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

    def _time_at(self, x: float, rect: QRectF) -> float:
        low, high = self._pan_preview_range or _time_range(self._meta)
        ratio = min(1.0, max(0.0, (x - rect.left()) / max(rect.width(), 1.0)))
        return low + ratio * (high - low)

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
        if self._detail_data is None:
            painter.drawPixmap(origin.toPoint(), scaled)
        else:
            self._paint_detail(painter, scaled)

        rect = self._axes_rect(scaled)
        limits = self._meta.get("freq_ylim") or []
        if rect is None or len(limits) != 2:
            return
        low, high = float(limits[0]), float(limits[1])
        if high <= low or rect.height() <= 0:
            return

        if self._selection is not None:
            t_low, t_high = _time_range(self._meta)
            left, right = sorted(self._selection)
            x1 = rect.left() + (left - t_low) / (t_high - t_low) * rect.width()
            x2 = rect.left() + (right - t_low) / (t_high - t_low) * rect.width()
            painter.fillRect(QRectF(x1, rect.top(), x2 - x1, rect.height()), QColor(180, 225, 255, 65))
            painter.setPen(QPen(QColor(220, 240, 255), 1.0, Qt.PenStyle.DashLine))
            painter.drawRect(QRectF(x1, rect.top(), x2 - x1, rect.height()))

        # 彩色深底使用亮色游标；旧灰度 PNG 仍保留黑色游标。
        cursor = min(max(self._cursor_hz, low), high)
        y = rect.bottom() - (cursor - low) / (high - low) * rect.height()
        cursor_color = QColor(235, 235, 245) if self._meta.get("spectrogram_colormap") == "avisoft_blue" else QColor(15, 15, 15)
        painter.setPen(QPen(cursor_color, 1.1))
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

    def _paint_detail(self, painter: QPainter, scaled: QPixmap) -> None:
        """直接绘制彩色频谱矩阵、波形与坐标轴，不经过 PNG 编解码。"""
        data = self._detail_data
        origin = self._image_origin(scaled)
        rect = self._axes_rect(scaled)
        left, bottom, width, height = data.meta["waveform_bbox"]
        wave = QRectF(origin.x() + left * scaled.width(),
                      origin.y() + (1 - bottom - height) * scaled.height(),
                      width * scaled.width(), height * scaled.height())
        painter.fillRect(wave, QColor("#080b19"))
        painter.fillRect(rect, QColor("#030715"))
        t0, t1 = self._pan_preview_range or data.meta["time_xlim"]
        f0, f1 = data.meta["freq_ylim"]
        x0, x1, y0, y1 = data.image_extent
        image_rect = QRectF(rect.left() + (x0 - t0) / (t1 - t0) * rect.width(),
                            rect.bottom() - (y1 - f0) / (f1 - f0) * rect.height(),
                            (x1 - x0) / (t1 - t0) * rect.width(),
                            (y1 - y0) / (f1 - f0) * rect.height())
        painter.save()
        painter.setClipRect(rect)
        # 同原 pcolormesh：每格颜色直接映射，避免插值产生额外频率纹理。
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawImage(image_rect, self._spectrum_image)
        painter.restore()
        painter.save()
        painter.setClipRect(wave)
        painter.setPen(QPen(QColor("#808080"), .5))
        painter.drawLine(QPointF(wave.left(), wave.center().y()), QPointF(wave.right(), wave.center().y()))
        original_low, original_high = data.meta["time_xlim"]
        wave_x = wave.left() + (original_low - t0) / (t1 - t0) * wave.width()
        painter.setTransform(QTransform().translate(wave_x, wave.center().y()).scale(
            wave.width() * (original_high - original_low) / (t1 - t0), -wave.height() / 2.16))
        path = self._detail_wave_path(max(1, round(wave.width())))
        if not self._wave_path_columns:
            painter.fillPath(path, QColor("#ffad00"))
        else:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        pen = QPen(QColor("#ffad00"), 1. if self._wave_path_columns else .6)
        pen.setCosmetic(True)
        painter.strokePath(path, pen)
        painter.restore()
        painter.setPen(QPen(QColor("#555566"), 1.))
        painter.drawRect(rect)
        painter.drawRect(wave)
        painter.setPen(QColor("#222222"))
        unit = data.meta["time_axis_unit"]
        for i in range(6):
            ratio = i / 5
            x = rect.left() + ratio * rect.width()
            y = rect.bottom() - ratio * rect.height()
            time = t0 + ratio * (t1 - t0)
            label = f"{time * (1000 if unit == 'ms' else 1):.6g}"
            painter.drawText(QRectF(x - 45, rect.bottom() + 5, 90, 20), Qt.AlignCenter, label)
            painter.drawText(QRectF(rect.left() - 65, y - 10, 57, 20),
                             Qt.AlignRight | Qt.AlignVCenter, f"{(f0 + ratio * (f1 - f0)) / 1000:g}")
        for v in (-1, 0, 1):
            y = wave.center().y() - v / 2.16 * wave.height()
            painter.drawText(QRectF(wave.left() - 45, y - 10, 37, 20),
                             Qt.AlignRight | Qt.AlignVCenter, str(v))
        painter.drawText(QRectF(rect.left(), rect.bottom() + 27, rect.width(), 22),
                         Qt.AlignCenter, f"Time ({unit})")
        painter.save()
        painter.translate(rect.left() - 70, rect.center().y())
        painter.rotate(-90)
        painter.drawText(QRectF(-100, -12, 200, 24), Qt.AlignCenter, "Frequency (kHz)")
        painter.restore()
        painter.drawText(QRectF(wave.left(), wave.top() - 28, wave.width(), 24),
                         Qt.AlignCenter, data.title)
        painter.setPen(QColor("#dddddd"))
        painter.drawText(wave.adjusted(8, 5, -8, -5), Qt.AlignTop | Qt.AlignRight, "波形（相对振幅）")

    # ---------------------------------------------------------- 交互
    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._clamp_pan(self._scaled_pixmap())
        self._update_cursor_shape()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton:
            scaled = self._scaled_pixmap()
            rect = self._axes_rect(scaled)
            if (self._time_navigation and event.modifiers() & Qt.KeyboardModifier.ShiftModifier
                    and rect is not None and rect.contains(event.position())):
                value = self._time_at(event.position().x(), rect)
                self._selection = (value, value)
                self._selecting = True
                self.setCursor(Qt.CursorShape.CrossCursor)
                self.update()
                event.accept()
                return
            if (self._time_navigation and self._detail_data is not None and rect is not None
                    and rect.left() <= event.position().x() <= rect.right()
                    and self._image_origin(scaled).y() <= event.position().y() <= rect.bottom()):
                self._time_dragging = True
                self._time_drag_moved = False
                self._time_drag_anchor = self._pan_preview_range or self._pending_time_range or _time_range(self._meta)
                self._time_drag_start_x = event.position().x()
                self._time_drag_width = max(1., rect.width())
                self._selection = None
                self._update_cursor_shape()
                event.accept()
                return
            if scaled.width() > self.width() or scaled.height() > self.height():
                self._dragging = True
                self._last_drag_pos = event.position()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._time_dragging:
            self._move_time_drag(event.position().x())
            event.accept()
            return
        if self._selecting:
            rect = self._axes_rect(self._scaled_pixmap())
            if rect is not None:
                self._selection = (self._selection[0], self._time_at(event.position().x(), rect))
                self.update()
            event.accept()
            return
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

    def _move_time_drag(self, x: float) -> None:
        delta = x - self._time_drag_start_x
        if abs(delta) < 3 and not self._time_drag_moved:
            return
        self._time_drag_moved = True
        low, high = self._time_drag_anchor
        span = high - low
        full_low, full_high = self._full_time_range
        left = min(full_high - span, max(full_low, low - delta / self._time_drag_width * span))
        self._pan_preview_range = (left, left + span)
        self.timePanRequested.emit(left, left + span)
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton and self._time_dragging:
            self._move_time_drag(event.position().x())
            self._time_dragging = False
            if self._time_drag_moved:
                self.timePanRequested.emit(*self._pan_preview_range)
            self._update_cursor_shape()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._selecting:
            self._selecting = False
            rect = self._axes_rect(self._scaled_pixmap())
            if rect is not None:
                low, high = sorted((self._selection[0], self._time_at(event.position().x(), rect)))
                self._selection = (low, high) if high > low else None
                if high > low:
                    self.rangeSelected.emit(low, high)
            self._update_cursor_shape()
            self.update()
            event.accept()
            return
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
                if self._time_navigation:
                    rect = self._axes_rect(self._scaled_pixmap())
                    if rect is not None:
                        low, high = self._pending_time_range or _time_range(self._meta)
                        ratio = min(1., max(0., (event.position().x() - rect.left()) / rect.width()))
                        anchor = low + ratio * (high - low)
                        span = (high - low) * (0.5 if delta > 0 else 2.0)
                        if delta > 0:
                            span = min(2.0, max(0.0003, span))
                        # 缩放前后鼠标对应同一个时刻，不把鼠标时刻强行移到图中央。
                        center = anchor + (0.5 - ratio) * span
                        self._time_dragging = False
                        self._dragging = False
                        self._pan_preview_range = None
                        self._update_cursor_shape()
                        self.timeZoomRequested.emit(center, span)
                        event.accept()
                        return
                self._zoom_by(1.15 if delta > 0 else 1.0 / 1.15)
                event.accept()
                return
        super().wheelEvent(event)


class _LocalSpectrogramWorker(QThread):
    ready = Signal(object)
    failed = Signal(str)

    def __init__(self, source: dict, low: float, high: float, frequency_limits: tuple,
                 cache, parent=None):
        super().__init__(parent)
        self._source, self._low, self._high = dict(source), low, high
        self._frequency_limits, self._cache = frequency_limits, cache

    def run(self) -> None:
        try:
            self.ready.emit(self._cache.compute(
                self._source, self._low, self._high, self._frequency_limits,
            ))
        except Exception as error:
            self.failed.emit(f"局部重算失败：{error}")


class LinearSpectrogramDialog(QDialog):
    """① 的查看窗口：线性声谱图 + 可移动频率光标。"""

    def __init__(
        self,
        png_path: Path,
        meta: dict,
        title: str = "",
        parent: QWidget | None = None,
        *,
        source_audio: dict | None = None,
    ) -> None:
        super().__init__(parent)
        self._png_path = Path(png_path)
        self._overview_path = Path(png_path)
        self._overview_meta = dict(meta)
        self._full_range = _time_range(meta)
        self._source_audio = resolve_source_audio(png_path, meta, source_audio)
        self._selected_range: tuple[float, float] | None = None
        self._worker: _LocalSpectrogramWorker | None = None
        from ..local_spectrogram import LocalSpectrogramCache
        self._cache = LocalSpectrogramCache()
        self._showing_detail = False
        self._queued_request = None
        self._pending_zoom = None
        self._restore_requested = False
        self._zoom_timer = QTimer(self)
        self._zoom_timer.setSingleShot(True)
        self._zoom_timer.setInterval(180)
        self._zoom_timer.timeout.connect(self._flush_zoom)
        self.setWindowTitle(title or "片段线性声谱图")
        self.resize(1140, 700)

        layout = QVBoxLayout(self)
        self._view = LinearSpectrogramView(
            QPixmap(str(self._png_path)), meta, self
        )
        self._view.set_time_navigation(self._source_audio is not None)
        self._view.rangeSelected.connect(self._select_range)
        self._view.timeZoomRequested.connect(self._zoom_time)
        self._view.timePanRequested.connect(self._pan_time)
        layout.addWidget(self._view, stretch=1)

        hint = QLabel(
            "移动鼠标读取频率。Shift+左键在声谱图内拖选，再点“重算选区”；"
            "Ctrl+滚轮缩放，放大后按住左键左右拖动浏览，Shift+左键仍用于选区；"
            "点“返回概览”恢复整段。局部选区最长 2 秒。"
            if self._source_audio else
            "移动鼠标读取频率；Ctrl+滚轮缩放图片。此图没有可用的关联 WAV，无法重算局部。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#666666;")
        layout.addWidget(hint)
        self._status = QLabel()
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        buttons = QHBoxLayout()
        self._detail_button = QPushButton("重算选区")
        self._detail_button.setEnabled(False)
        self._detail_button.clicked.connect(self._recompute_selection)
        self._overview_button = QPushButton("返回概览")
        self._overview_button.setEnabled(False)
        self._overview_button.clicked.connect(self._restore_overview)
        buttons.addWidget(self._detail_button)
        buttons.addWidget(self._overview_button)
        buttons.addStretch(1)
        png_button = QPushButton("打开概览 PNG 原图")
        png_button.clicked.connect(self._open_png)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.accept)
        buttons.addWidget(png_button)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

    def _select_range(self, low: float, high: float) -> None:
        self._selected_range = (low, high)
        self._detail_button.setEnabled(self._source_audio is not None and self._worker is None)
        self._status.setText(f"选中 {low:.6f}–{high:.6f} s（{(high - low) * 1000:.2f} ms）")

    def _recompute_selection(self) -> None:
        if self._selected_range is not None:
            self._zoom_timer.stop()
            self._pending_zoom = None
            self.request_detail(*self._selected_range)

    def _zoom_time(self, center: float, span: float) -> None:
        if self._full_range is None:
            return
        low, high = self._full_range
        if span >= high - low - 1e-9 or span > 2.0 + 1e-9:
            # 超过局部精算上限时返回完整概览，不能把缩小请求截成同宽平移。
            self._restore_overview()
            return
        span = min(2.0, max(0.0003, span))
        left = max(low, min(center - span / 2, high - span))
        self._pending_zoom = (left, min(high, left + span))
        self._restore_requested = False
        self._view._pending_time_range = self._pending_zoom
        self._queued_request = None
        self._zoom_timer.start(180)
        self._status.setText("松开滚轮后更新局部声谱…")

    def _pan_time(self, low: float, high: float) -> None:
        self._pending_zoom = (low, high)
        self._restore_requested = False
        self._view._pending_time_range = self._pending_zoom
        self._queued_request = None
        # 拖动时先平移当前数据，停顿后或松开时读取新范围。
        self._zoom_timer.start(180 if self._view._time_dragging else 0)
        self._status.setText(f"浏览 {low:.6f}–{high:.6f} s，松开鼠标更新细节…")

    def _flush_zoom(self) -> None:
        pending, self._pending_zoom = self._pending_zoom, None
        if pending is not None:
            self.request_detail(*pending)

    def request_detail(self, low: float, high: float) -> None:
        if self._source_audio is None or self._full_range is None:
            return
        if not math.isfinite(low) or not math.isfinite(high):
            return
        low, high = max(self._full_range[0], low), min(self._full_range[1], high)
        if high <= low:
            return
        if high - low > 2.0 + 1e-9:
            self._status.setText("局部精细查看请选择 2 秒以内的范围。")
            return
        self._restore_requested = False
        self._view._pending_time_range = (low, high)
        if self._worker is not None:
            # 保留最终目标；已经在计算的中间结果不会覆盖新请求。
            self._queued_request = (low, high)
            return
        worker = _LocalSpectrogramWorker(
            self._source_audio, low, high, tuple(self._overview_meta["freq_ylim"]), self._cache, self,
        )
        self._worker = worker
        self._detail_button.setEnabled(False)
        self._overview_button.setEnabled(True)
        self._status.setText(f"正在从 WAV 重算 {low:.6f}–{high:.6f} s…")
        worker.ready.connect(self._detail_ready)
        worker.failed.connect(self._detail_failed)
        worker.finished.connect(self._detail_finished)
        worker.start()

    def _detail_ready(self, data) -> None:
        if self._queued_request is not None or self._zoom_timer.isActive() or self._restore_requested:
            return
        self._view.set_detail(data)
        self._showing_detail = True
        self._selected_range = None
        low, high = data.meta["time_xlim"]
        self._status.setText(f"局部声谱已重算：{low:.6f}–{high:.6f} s")

    def _detail_failed(self, message: str) -> None:
        if self._queued_request is None and not self._zoom_timer.isActive():
            self._view._pan_preview_range = None
            self._view.update()
            self._status.setText(message)

    def _detail_finished(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.deleteLater()
        if self._restore_requested:
            self._restore_overview()
            return
        queued, self._queued_request = self._queued_request, None
        if queued is not None and not self._zoom_timer.isActive():
            self.request_detail(*queued)
            return
        if not self._zoom_timer.isActive():
            self._view._pending_time_range = None
        self._overview_button.setEnabled(self._showing_detail or self._zoom_timer.isActive())
        self._detail_button.setEnabled(self._selected_range is not None and self._source_audio is not None)

    def _restore_overview(self) -> None:
        self._zoom_timer.stop()
        self._pending_zoom = None
        self._queued_request = None
        self._view._pending_time_range = None
        self._view._time_dragging = False
        self._view._pan_preview_range = None
        if self._worker is not None:
            self._restore_requested = True
            self._status.setText("当前计算完成后返回概览…")
            return
        self._restore_requested = False
        self._showing_detail = False
        self._png_path = self._overview_path
        self._view.set_image(QPixmap(str(self._overview_path)), self._overview_meta)
        self._selected_range = None
        self._detail_button.setEnabled(False)
        self._overview_button.setEnabled(False)
        self._status.clear()

    def closeEvent(self, event) -> None:
        if self._worker is not None:
            self._status.setText("正在生成局部图，完成后即可关闭。")
            event.ignore()
            return
        super().closeEvent(event)

    def done(self, result: int) -> None:
        # 同时覆盖关闭按钮、Escape、窗口关闭，避免销毁仍在运行的 QThread。
        if self._worker is not None:
            self._status.setText("正在生成局部图，完成后即可关闭。")
            return
        self._zoom_timer.stop()
        self._pending_zoom = None
        self._cache.clear()
        super().done(result)

    def _open_png(self) -> None:
        if not self._png_path.exists():
            return
        from .image_viewer import ImagePreviewDialog

        dialog = ImagePreviewDialog(self._png_path, self)
        dialog.exec()
