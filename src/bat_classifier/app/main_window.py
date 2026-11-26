"""片段识别与叫声统计主窗口。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QRect, Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .segment_analysis_tab import SegmentAnalysisTab
from .settings import AppSettings


class MainWindow(QMainWindow):
    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self._settings = settings
        self._workers: list[object] = []

        self.setWindowTitle("蝙蝠声学分类系统")
        self._apply_window_geometry()

        # 日志面板先于页签构建：页签初始化时会写入首条日志。
        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumHeight(150)
        log_bar = QHBoxLayout()
        log_bar.addWidget(QLabel("运行日志"))
        log_bar.addStretch(1)
        self._log_clear_button = QPushButton("清空")
        self._log_clear_button.clicked.connect(self._log_view.clear)
        self._log_toggle_button = QPushButton("收起")
        self._log_toggle_button.clicked.connect(self._toggle_log)
        log_bar.addWidget(self._log_clear_button)
        log_bar.addWidget(self._log_toggle_button)
        self._set_log_collapsed(bool(self._settings.log_collapsed))

        tabs = QTabWidget()
        self._segment_tab = self._build_segment_tab()
        tabs.addTab(self._segment_tab, "① 片段识别与叫声统计")


        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.addWidget(tabs, stretch=1)
        root_layout.addLayout(log_bar)
        root_layout.addWidget(self._log_view)
        self.setCentralWidget(central)

        self._append_log(f"工作区：{self._settings.workspace}")
        self.setAcceptDrops(True)
        # 在子控件之前接收文件拖入，避免只读文本框和表格吞掉拖放事件。
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:
        if (
            event.type() in {QEvent.Type.DragEnter, QEvent.Type.DragMove, QEvent.Type.Drop}
            and isinstance(watched, QWidget)
            and watched.window() is self
            and event.mimeData().hasUrls()
        ):
            paths = list(dict.fromkeys(
                Path(url.toLocalFile()) for url in event.mimeData().urls()
                if url.isLocalFile()
            ))
            supported = [path for path in paths if path.is_dir() or (
                path.is_file() and path.suffix.lower() in {".wav", ".son", ".png"}
            )]
            if supported:
                event.setDropAction(Qt.DropAction.CopyAction)
                event.accept()
                if event.type() == QEvent.Type.Drop:
                    # 先完成系统拖放，再打开模态查看窗口。
                    QTimer.singleShot(0, lambda: self._open_dropped_paths(supported))
                return True
        return super().eventFilter(watched, event)

    def _open_dropped_paths(self, paths: list[Path]) -> None:
        for path in paths:
            if path.is_dir() or path.suffix.lower() == ".png":
                self._segment_tab.open_png_path(path)
            else:
                self._segment_tab.open_direct_path(path)


    def _apply_window_geometry(self) -> None:
        """优先恢复上次保存的窗口大小/位置；无效则按屏幕可用区域取 90%。"""
        width = int(self._settings.window_width)
        height = int(self._settings.window_height)
        if width > 0 and height > 0:
            x = int(self._settings.window_x)
            y = int(self._settings.window_y)
            rect = QRect(x, y, width, height)
            screens = QApplication.screens()
            if x >= 0 and y >= 0 and any(
                screen.availableGeometry().intersects(rect)
                for screen in screens
            ):
                self.setGeometry(rect)
                return
            self.resize(width, height)
            return

        screen = QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else None
        if available is not None:
            self.resize(
                max(960, int(available.width() * 0.9)),
                max(600, int(available.height() * 0.9)),
            )
        else:
            self.resize(1040, 720)

    def _save_window_geometry(self) -> None:
        self._settings.window_width = self.width()
        self._settings.window_height = self.height()
        self._settings.window_x = self.x()
        self._settings.window_y = self.y()

    def _toggle_log(self) -> None:
        collapsed = self._log_view.isVisible()  # 当前可见 → 收起
        self._set_log_collapsed(collapsed)
        self._settings.log_collapsed = collapsed
        self._settings.save()

    def _set_log_collapsed(self, collapsed: bool) -> None:
        self._log_view.setVisible(not collapsed)
        self._log_clear_button.setVisible(not collapsed)
        self._log_toggle_button.setText("展开" if collapsed else "收起")

    def _append_log(self, text: str) -> None:
        self._log_view.appendPlainText(text)

    def _track_worker(self, worker) -> None:
        worker.log.connect(self._append_log)
        worker.finished.connect(
            lambda: self._workers.remove(worker)
            if worker in self._workers
            else None
        )
        self._workers.append(worker)
        worker.start()

    def confirm_if_busy(self) -> bool:
        """有关闭确认需求时由 closeEvent 调用；返回是否仍有任务在跑。"""
        return bool(self._workers)


    def closeEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        if self.confirm_if_busy():
            answer = QMessageBox.question(
                self,
                "任务运行中",
                "后台任务尚未结束，确定退出吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            for worker in list(self._workers):
                worker.wait(100)
        if QGuiApplication.platformName() != "offscreen":
            self._save_window_geometry()
            self._settings.save()
        event.accept()


    def _build_segment_tab(self) -> QWidget:
        # 页面逻辑都在 segment_analysis_tab.py，算法在 segment_analysis.py，
        # 主窗口只负责挂载。
        return SegmentAnalysisTab(
            log=self._append_log,
            track_worker=self._track_worker,
        )
