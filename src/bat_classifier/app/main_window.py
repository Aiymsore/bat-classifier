"""桌面 App 主窗口：数据来源 / 模型训练 / 预测 / 模型管理 / 结果图像 /
批量整理与专家标注 / 片段识别与叫声统计 七页签。

底部为共享日志面板，显示后台线程重定向的库内 print 输出。
耗时操作全部走 workers.py 的 QThread，UI 线程不执行算法。
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QRect, Qt, QUrl
from PySide6.QtGui import (
    QDesktopServices,
    QGuiApplication,
    QStandardItem,
    QStandardItemModel,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..config import (
    CLUSTER_METADATA_JSON,
    COMBINED_DEMO_MODEL_PATH,
    DEMO_PREDICTION_DIR,
    FIGURE_DIR,
    MODEL_DIR,
    NOISE_GATE_MODEL_PATH,
    PULSE_ZOOM_DIR,
    RAW_DATA_DIR,
    SPECTROGRAM_DIR,
    validate_labels,
)
from ..model_store import IMPORTED_MODEL_DIR, list_models
from .curation_tab import CurationTab
from .image_viewer import ImageGalleryPane
from .model_picker import ModelPicker
from .segment_analysis_tab import SegmentAnalysisTab
from .settings import AppSettings
from .workers import (
    CalibrateModelWorker,
    ImportDataWorker,
    PredictWorker,
    TrainingWorker,
)

SUMMARY_CSV_NAME = "noise_ABCD_prediction_summary.csv"


class MainWindow(QMainWindow):
    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self._settings = settings
        self._source_path: Path | None = None
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
        tabs.addTab(self._build_source_tab(), "① 数据来源")
        tabs.addTab(self._build_train_tab(), "② 模型训练")
        tabs.addTab(self._build_predict_tab(), "③ 预测")
        tabs.addTab(self._build_model_tab(), "④ 模型管理")
        tabs.addTab(self._build_images_tab(), "⑤ 结果图像")
        tabs.addTab(self._build_curation_tab(), "⑥ 批量整理与专家标注")
        self._segment_tab = self._build_segment_tab()
        tabs.addTab(self._segment_tab, "⑦ 片段识别与叫声统计")

        self._wire_model_pickers()

        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.addWidget(tabs, stretch=1)
        root_layout.addLayout(log_bar)
        root_layout.addWidget(self._log_view)
        self.setCentralWidget(central)

        self._append_log(f"工作区：{self._settings.workspace}")
        self._refresh_source_summary()

    # ---------------- 共享 ----------------

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

    # ---------------- 模型选择（③ / ④ / ⑦ 共用） ----------------

    def _model_pickers(self) -> list[ModelPicker]:
        pickers = [
            getattr(self, "_predict_picker", None),
            getattr(self, "_manage_picker", None),
        ]
        segment_tab = getattr(self, "_segment_tab", None)
        if segment_tab is not None:
            pickers.append(segment_tab.model_picker)
        return [picker for picker in pickers if picker is not None]

    def _wire_model_pickers(self) -> None:
        """三处选择器指向同一个“当前使用模型”，任一处改动即同步。"""
        for picker in self._model_pickers():
            picker.model_changed.connect(self._on_model_selected)

    def _on_model_selected(self, path: str) -> None:
        sender = self.sender()
        self._settings.active_model_path = str(path)
        self._settings.save()
        for picker in self._model_pickers():
            if picker is sender:
                continue
            picker.set_path(path)   # 值相同则不发信号，天然不会死循环
        self._append_log(f"当前使用模型：{path}")

    def refresh_model_pickers(self) -> None:
        """② 训练完成 / ④ 导入后调用：重扫模型目录并刷新各页状态。"""
        for picker in self._model_pickers():
            picker.refresh(picker.current_path_text())
        segment_tab = getattr(self, "_segment_tab", None)
        if segment_tab is not None:
            segment_tab.refresh_models()
        if getattr(self, "_model_status_label", None) is not None:
            self._refresh_model_status()
        self._refresh_model_tab()

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

    # ---------------- ① 数据来源 ----------------

    def _build_source_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        info_group = QGroupBox("当前工作区")
        info_layout = QGridLayout()
        self._workspace_label = QLabel(str(self._settings.workspace))
        self._workspace_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._raw_summary_label = QLabel("原始数据：统计中…")
        info_layout.addWidget(QLabel("工作目录"), 0, 0)
        info_layout.addWidget(self._workspace_label, 0, 1)
        info_layout.addWidget(self._raw_summary_label, 1, 0, 1, 2)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        select_group = QGroupBox("更换训练数据来源")
        select_layout = QGridLayout()
        self._source_edit = QLineEdit()
        self._source_edit.setReadOnly(True)
        self._source_edit.setPlaceholderText(
            "尚未选择。可选一个包含WAV的文件夹，或一个zip压缩包。"
        )
        self._source_preview_label = QLabel("")
        folder_button = QPushButton("选择WAV文件夹")
        zip_button = QPushButton("选择zip压缩包")
        self._apply_source_button = QPushButton("应用此数据源")
        folder_button.clicked.connect(self._pick_source_folder)
        zip_button.clicked.connect(self._pick_source_zip)
        self._apply_source_button.clicked.connect(self._apply_source)
        select_layout.addWidget(self._source_edit, 0, 0, 1, 3)
        select_layout.addWidget(folder_button, 1, 0)
        select_layout.addWidget(zip_button, 1, 1)
        select_layout.addWidget(self._apply_source_button, 1, 2)
        select_layout.addWidget(self._source_preview_label, 2, 0, 1, 3)
        select_group.setLayout(select_layout)
        layout.addWidget(select_group)

        hint = QLabel(
            "说明：应用数据源会把选中的WAV复制（或zip解压）到工作区 "
            "data/raw，并清空原有数据；之后在“模型训练”页重新训练即可生效。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)

        if self._settings.data_source:
            self._source_edit.setText(self._settings.data_source)
            self._source_path = Path(self._settings.data_source)
        return page

    def _pick_source_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "选择WAV文件夹")
        if chosen:
            self._set_source(Path(chosen))

    def _pick_source_zip(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self, "选择zip压缩包", str(RAW_DATA_DIR.parent), "压缩包 (*.zip)"
        )
        if chosen:
            self._set_source(Path(chosen))

    def _set_source(self, path: Path) -> None:
        self._source_path = path
        self._source_edit.setText(str(path))
        self._source_preview_label.setText("统计中…")
        try:
            from ..audio_io import find_wav_files

            if path.is_dir():
                wav_files = find_wav_files(path)
                self._source_preview_label.setText(
                    f"找到 {len(wav_files)} 个WAV。"
                )
            else:
                size_mb = path.stat().st_size / 1024 / 1024
                self._source_preview_label.setText(
                    f"zip 文件，约 {size_mb:.1f} MB。"
                )
        except Exception as error:  # noqa: BLE001
            self._source_preview_label.setText(f"读取失败：{error}")

    def _apply_source(self) -> None:
        if self._source_path is None:
            QMessageBox.warning(self, "提示", "请先选择数据来源。")
            return
        answer = QMessageBox.question(
            self,
            "确认更换数据",
            "将清空 data/raw 并导入新数据，确定继续吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        self._apply_source_button.setEnabled(False)
        worker = ImportDataWorker(self._source_path)
        worker.done.connect(self._on_source_imported)
        worker.failed.connect(self._on_task_failed)
        self._track_worker(worker)

    def _on_source_imported(self, wav_count: object) -> None:
        self._apply_source_button.setEnabled(True)
        self._append_log(f"数据源导入完成：{wav_count} 个WAV。")
        self._settings.data_source = str(self._source_path or "")
        self._settings.save()
        self._refresh_source_summary()

    def _refresh_source_summary(self) -> None:
        try:
            from ..audio_io import find_wav_files

            wav_files = find_wav_files(RAW_DATA_DIR)
            self._raw_summary_label.setText(
                f"原始数据（data/raw）：{len(wav_files)} 个WAV"
            )
        except Exception as error:  # noqa: BLE001
            self._raw_summary_label.setText(f"原始数据：读取失败（{error}）")

    # ---------------- ② 模型训练 ----------------

    def _build_train_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        label_group = QGroupBox("声型标签设置")
        label_layout = QGridLayout()
        label_layout.addWidget(QLabel("声型数量"), 0, 0)
        self._label_count_spin = QSpinBox()
        self._label_count_spin.setRange(2, 6)
        self._label_count_spin.setValue(len(self._settings.labels))
        self._label_count_spin.valueChanged.connect(self._rebuild_label_rows)
        label_layout.addWidget(self._label_count_spin, 0, 1)
        note = QLabel(
            "默认 A-D 四类时沿用 FM/CF/低频/复杂 的语义归类；"
            "其他数量或名称按聚类中心峰值频率从低到高依次分配。"
        )
        note.setWordWrap(True)
        label_layout.addWidget(note, 1, 0, 1, 3)
        self._label_rows_grid = QGridLayout()
        label_layout.addLayout(self._label_rows_grid, 2, 0, 1, 3)
        label_group.setLayout(label_layout)
        layout.addWidget(label_group)

        self._start_train_button = QPushButton("开始训练")
        self._start_train_button.clicked.connect(self._start_training)
        layout.addWidget(self._start_train_button)

        self._diagnostics_check = QCheckBox(
            "训练时为每条录音生成声谱图/脉冲放大图（耗时增加，可在⑤结果图像查看）"
        )
        layout.addWidget(self._diagnostics_check)

        self._train_summary_label = QLabel("尚未训练。")
        self._train_summary_label.setWordWrap(True)
        layout.addWidget(self._train_summary_label)
        layout.addStretch(1)

        self._label_name_edits: list[QLineEdit] = []
        self._label_desc_edits: list[QLineEdit] = []
        self._rebuild_label_rows()
        return page

    def _rebuild_label_rows(self) -> None:
        for index in range(self._label_rows_grid.count()):
            item = self._label_rows_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._label_name_edits.clear()
        self._label_desc_edits.clear()

        count = self._label_count_spin.value()
        for row in range(count):
            if row < len(self._settings.labels):
                name = self._settings.labels[row]
            elif row < 4:
                name = "ABCD"[row]
            else:
                name = f"型{row + 1}"
            name_edit = QLineEdit(name)
            desc_edit = QLineEdit(
                self._settings.type_descriptions.get(name, "")
            )
            self._label_rows_grid.addWidget(QLabel(f"标签 {row + 1}"), row, 0)
            self._label_rows_grid.addWidget(name_edit, row, 1)
            self._label_rows_grid.addWidget(desc_edit, row, 2)
            self._label_name_edits.append(name_edit)
            self._label_desc_edits.append(desc_edit)

    def _collect_labels(self) -> tuple[list[str], dict[str, str]]:
        labels = [edit.text().strip() for edit in self._label_name_edits]
        labels = validate_labels(labels)
        descriptions = {
            label: edit.text().strip()
            for label, edit in zip(labels, self._label_desc_edits)
        }
        return labels, descriptions

    def _start_training(self) -> None:
        try:
            labels, descriptions = self._collect_labels()
        except ValueError as error:
            QMessageBox.warning(self, "标签无效", str(error))
            return

        self._settings.labels = labels
        self._settings.type_descriptions = descriptions
        self._settings.save()

        self._start_train_button.setEnabled(False)
        self._train_summary_label.setText("训练中，请查看底部日志…")
        self._append_log(
            f"开始训练：标签 {labels}，数据 {RAW_DATA_DIR}"
        )
        worker = TrainingWorker(
            labels,
            descriptions,
            None,
            save_diagnostics=self._diagnostics_check.isChecked(),
        )
        worker.done.connect(self._on_training_done)
        worker.failed.connect(self._on_task_failed)
        self._track_worker(worker)

    def _on_training_done(self, payload: object) -> None:
        self._start_train_button.setEnabled(True)
        acoustic = payload.get("acoustic_types", {}) if isinstance(payload, dict) else {}
        silhouette = acoustic.get("silhouette_score")
        consistency = acoustic.get("temporary_classifier_consistency")
        parts = [f"训练完成。标签：{acoustic.get('labels', '?')}"]
        if silhouette is not None:
            parts.append(f"轮廓系数 {float(silhouette):.3f}")
        if consistency is not None:
            parts.append(f"一致率 {float(consistency):.2%}")
        parts.append("可到“预测”页对新录音分类，“结果图像”页查看PCA图。")
        self._train_summary_label.setText("；".join(parts))
        self._refresh_images_tab()
        # ② 训练出的模型固定写入 models/noise_bat_ABCD_model.joblib，
        # 通知 ③/④/⑦ 重扫模型目录（已导入的模型仍在，可继续切换使用）。
        self.refresh_model_pickers()
        self._append_log(
            f"训练产物已就绪：{COMBINED_DEMO_MODEL_PATH.name}"
            "（③/④/⑦ 中作为默认模型，可与已导入模型互相切换）"
        )

    def _on_task_failed(self, message: str) -> None:
        self._apply_source_button.setEnabled(True)
        self._start_train_button.setEnabled(True)
        self._append_log(f"任务失败：{message}")
        QMessageBox.critical(self, "任务失败", message)

    # ---------------- ③ 预测 ----------------

    def _build_predict_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(
            QLabel("使用模型（② 训练的模型 / 已导入的模型，可随时切换）")
        )
        self._predict_picker = ModelPicker(
            self._settings.active_model_path, log=self._append_log
        )
        layout.addWidget(self._predict_picker)

        self._model_status_label = QLabel("模型状态：检查中…")
        self._model_status_label.setWordWrap(True)
        layout.addWidget(self._model_status_label)
        self._predict_picker.model_changed.connect(
            lambda _path: self._refresh_model_status()
        )
        self._refresh_model_status()

        path_group = QGroupBox("预测输入")
        path_layout = QGridLayout()
        self._predict_edit = QLineEdit()
        self._predict_edit.setReadOnly(True)
        self._predict_edit.setPlaceholderText("选择一个WAV文件或一个文件夹。")
        wav_button = QPushButton("选择WAV")
        dir_button = QPushButton("选择文件夹")
        self._predict_button = QPushButton("开始预测")
        wav_button.clicked.connect(self._pick_predict_wav)
        dir_button.clicked.connect(self._pick_predict_folder)
        self._predict_button.clicked.connect(self._start_predict)
        path_layout.addWidget(self._predict_edit, 0, 0, 1, 3)
        path_layout.addWidget(wav_button, 1, 0)
        path_layout.addWidget(dir_button, 1, 1)
        path_layout.addWidget(self._predict_button, 1, 2)
        path_group.setLayout(path_layout)
        layout.addWidget(path_group)

        self._result_model = QStandardItemModel()
        self._result_table = QTableView()
        self._result_table.setModel(self._result_model)
        self._result_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        layout.addWidget(self._result_table, stretch=1)

        bottom_row = QHBoxLayout()
        self._open_results_button = QPushButton("打开预测结果文件夹")
        self._open_results_button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(DEMO_PREDICTION_DIR)))
        )
        bottom_row.addWidget(self._open_results_button)
        bottom_row.addStretch(1)
        layout.addLayout(bottom_row)
        return page

    def _refresh_model_status(self) -> None:
        path = self._predict_picker.current_path()
        if path and path.exists():
            from ..model_store import inspect_model

            entry = inspect_model(path)
            if entry.valid:
                self._model_status_label.setText(
                    f"模型状态：{entry.display_name}"
                    f"（{entry.size_mb:.1f} MB，标签："
                    f"{'、'.join(entry.labels) or '?'}）"
                )
            else:
                self._model_status_label.setText(
                    f"模型状态：所选模型不可用——{entry.error}"
                )
            return
        self._model_status_label.setText(
            "模型状态：未找到完整训练模型。"
            "请先在“② 模型训练”页训练，或用“导入模型文件…”导入现成模型。"
        )

    def _pick_predict_wav(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(self, "选择WAV", "", "音频 (*.wav)")
        if chosen:
            self._predict_edit.setText(chosen)

    def _pick_predict_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "选择WAV文件夹")
        if chosen:
            self._predict_edit.setText(chosen)

    def _load_combined_payload(self) -> dict:
        import joblib

        from ..model_store import payload_error

        path = self._predict_picker.current_path()
        if path and path.exists():
            payload = joblib.load(path)
            error = payload_error(payload)
            if error:
                raise RuntimeError(f"所选模型不可用（{path}）：{error}")
            return payload

        answer = QMessageBox.question(
            self,
            "缺少模型",
            "未找到完整训练模型。\n"
            "是否用仓库自带结果表构建一个演示模型？（结果仅供流程演示）",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            raise RuntimeError("没有可用模型。")

        from ..demo import build_demo_model

        self._append_log("正在从结果表构建演示模型…")
        return build_demo_model(force=True)

    def _start_predict(self) -> None:
        text = self._predict_edit.text().strip()
        if not text:
            QMessageBox.warning(self, "提示", "请先选择WAV或文件夹。")
            return
        input_path = Path(text)
        if not input_path.exists():
            QMessageBox.warning(self, "提示", f"路径不存在：{input_path}")
            return

        try:
            payload = self._load_combined_payload()
        except Exception as error:  # noqa: BLE001
            QMessageBox.critical(self, "模型不可用", str(error))
            return

        self._predict_button.setEnabled(False)
        worker = PredictWorker(input_path, payload)
        worker.done.connect(self._on_predict_done)
        worker.failed.connect(self._on_task_failed)
        self._track_worker(worker)

    def _on_predict_done(self, _result: object) -> None:
        self._predict_button.setEnabled(True)
        self._append_log("预测完成。")
        self._show_summary_csv()
        self._refresh_images_tab()

    def _show_summary_csv(self) -> None:
        import pandas as pd

        summary_path = DEMO_PREDICTION_DIR / SUMMARY_CSV_NAME
        if not summary_path.exists():
            self._append_log("未找到预测汇总CSV。")
            return
        frame = pd.read_csv(summary_path, encoding="utf-8-sig").head(500)
        self._result_model.clear()
        self._result_model.setHorizontalHeaderLabels([str(c) for c in frame.columns])
        for _, row in frame.iterrows():
            items = [
                QStandardItem("" if pd.isna(value) else str(value))
                for value in row
            ]
            self._result_model.appendRow(items)
        self._append_log(f"已加载汇总：{summary_path.name}（最多显示500行）")

    # ---------------- ④ 模型管理 ----------------

    def _build_model_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        files_group = QGroupBox("模型文件")
        files_layout = QVBoxLayout()

        files_layout.addWidget(
            QLabel("可用模型（② 训练的模型 / 已导入的模型，可随时切换）")
        )
        self._manage_picker = ModelPicker(
            self._settings.active_model_path, log=self._append_log
        )
        self._manage_picker.model_changed.connect(
            lambda _path: self._refresh_model_tab()
        )
        files_layout.addWidget(self._manage_picker)

        self._model_files_label = QLabel("")
        self._model_files_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._model_files_label.setWordWrap(True)
        files_layout.addWidget(self._model_files_label)
        files_row = QHBoxLayout()
        import_button = QPushButton("导入模型文件…")
        import_button.setToolTip(
            "把现成的 .joblib 模型复制到 models/imported/，"
            "之后即可在 ③ 预测 / ⑦ 片段识别中选用；不会覆盖 ② 的训练产物。"
        )
        calibrate_button = QPushButton("补校准当前模型…")
        calibrate_button.setToolTip(
            "用一份带真值标签的脉冲 CSV，给当前模型补一个 "
            "CalibratedClassifierCV 校准器（不重训、不覆盖原模型，"
            "结果另存为 *_calibrated.joblib）。⑦ 的置信度将变为校准概率。"
        )
        refresh_button = QPushButton("刷新")
        open_button = QPushButton("打开模型目录")
        import_button.clicked.connect(self._import_model_file)
        calibrate_button.clicked.connect(self._calibrate_current_model)
        refresh_button.clicked.connect(self._refresh_model_tab)
        open_button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(MODEL_DIR)))
        )
        files_row.addWidget(import_button)
        files_row.addWidget(calibrate_button)
        files_row.addWidget(refresh_button)
        files_row.addWidget(open_button)
        files_row.addStretch(1)
        files_layout.addLayout(files_row)
        files_group.setLayout(files_layout)
        layout.addWidget(files_group)

        summary_group = QGroupBox("训练摘要（中文）")
        summary_layout = QVBoxLayout()
        self._summary_view = QPlainTextEdit()
        self._summary_view.setReadOnly(True)
        self._summary_view.setMaximumHeight(190)
        summary_layout.addWidget(self._summary_view)
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)

        metadata_group = QGroupBox("训练元数据（原始 JSON）")
        metadata_layout = QVBoxLayout()
        self._metadata_view = QPlainTextEdit()
        self._metadata_view.setReadOnly(True)
        metadata_layout.addWidget(self._metadata_view)
        metadata_group.setLayout(metadata_layout)
        layout.addWidget(metadata_group, stretch=1)

        self._refresh_model_tab()
        return page

    def _import_model_file(self) -> None:
        """④ 页面的导入按钮：复制现成模型到 models/imported/ 并刷新列表。"""
        self._manage_picker.import_interactively()
        self._refresh_model_tab()

    def _calibrate_current_model(self) -> None:
        """④ 页面：给当前选中的模型补 CalibratedClassifierCV 校准器。

        只 fit 校准器，不重训、不覆盖原模型；结果另存为
        models/imported/<名称>_calibrated.joblib，⑦ 的置信度随之变为校准概率。
        """
        model_path = self._manage_picker.current_path()
        if model_path is None or not model_path.exists():
            QMessageBox.information(
                self, "提示", "请先在上方选择要补校准的模型。"
            )
            return

        csv_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择带真值标签的脉冲 CSV",
            "",
            "CSV 文件 (*.csv);;所有文件 (*)",
        )
        if not csv_path:
            return

        output_path = (
            IMPORTED_MODEL_DIR / f"{model_path.stem}_calibrated.joblib"
        )
        answer = QMessageBox.question(
            self,
            "补校准",
            f"模型：{model_path.name}\n标注：{Path(csv_path).name}\n"
            f"输出：{output_path}\n\n"
            "只补一个 CalibratedClassifierCV 校准器（按 source_recording "
            "分组隔离），不重训模型、不覆盖原模型。\n"
            "数据不足以安全校准时会直接失败，绝不伪造校准概率。\n\n继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            return

        worker = CalibrateModelWorker(model_path, Path(csv_path), output_path)
        worker.done.connect(self._on_calibrate_done)
        worker.failed.connect(self._on_task_failed)
        self._track_worker(worker)
        self._append_log(
            f"开始补校准：{model_path.name} ← {Path(csv_path).name}"
        )

    def _on_calibrate_done(self, result: object) -> None:
        info = dict(result or {})
        output_path = str(info.get("output_path") or "")
        calibration = dict(info.get("calibration") or {})
        self._append_log(
            f"补校准完成：{output_path}"
            f"（method={calibration.get('calibration_method') or '?'}, "
            f"{calibration.get('fold_count') or 0} 折，"
            f"标签列={info.get('label_column') or '?'}, "
            f"分组列={info.get('source_column') or '无'}）"
        )
        self.refresh_model_pickers()
        if output_path:
            for picker in self._model_pickers():
                picker.set_path(output_path)

    def _refresh_model_tab(self) -> None:
        lines = ["【可用模型】③ 预测 / ⑦ 片段识别都能选用："]
        entries = list_models()
        if not entries:
            lines.append(
                "（models/ 下暂无可用模型：请先在 ② 模型训练页训练，"
                "或点“导入模型文件…”导入现成模型）"
            )
        for entry in entries:
            mark = "✔" if entry.valid else "✘"
            lines.append(
                f"{mark} [{entry.origin_text}] {entry.path.name}"
                f"（{entry.size_mb:.1f} MB）"
            )
            lines.append(f"    {entry.path}")
            if not entry.valid:
                lines.append(f"    ⚠ 不可用：{entry.error}")
        lines.append("")

        lines.append("【② 训练伴随文件】")
        for model_path in [
            COMBINED_DEMO_MODEL_PATH,
            CLUSTER_METADATA_JSON,
            NOISE_GATE_MODEL_PATH,
        ]:
            if model_path.exists():
                size_mb = model_path.stat().st_size / 1024 / 1024
                lines.append(f"[存在] {model_path.name}（{size_mb:.1f} MB）")
            else:
                lines.append(f"[缺失] {model_path.name}")
        self._model_files_label.setText("\n".join(lines))

        if CLUSTER_METADATA_JSON.exists():
            try:
                payload = json.loads(
                    CLUSTER_METADATA_JSON.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as error:
                self._summary_view.setPlainText(f"读取失败：{error}")
                self._metadata_view.setPlainText(f"读取失败：{error}")
                return

            self._metadata_view.setPlainText(
                json.dumps(payload, ensure_ascii=False, indent=2)
            )
            self._summary_view.setPlainText(
                self._render_zh_summary(payload)
            )
        else:
            self._summary_view.setPlainText("尚未训练，暂无摘要。")
            self._metadata_view.setPlainText("尚未训练，暂无元数据。")

    def _render_zh_summary(self, payload: dict) -> str:
        """优先显示元数据自带的 summary_zh；旧元数据现场拼一份兜底。"""
        builtin = payload.get("summary_zh")
        if isinstance(builtin, str) and builtin.strip():
            return builtin

        # 延迟导入：clustering 牵连 sklearn 等重库，避免拖慢界面启动。
        from ..clustering import build_zh_summary

        labels = payload.get("labels") or list(
            (payload.get("acoustic_types") or {}).keys()
        ) or ["A", "B", "C", "D"]
        return build_zh_summary(
            labels=[str(label) for label in labels],
            descriptions=dict(payload.get("acoustic_types") or {}),
            pulse_count=int(payload.get("pulse_count") or 0),
            recording_count=int(payload.get("recording_count") or 0),
            silhouette=payload.get("silhouette_score"),
            consistency=payload.get("temporary_classifier_consistency"),
            pca_ratios=payload.get("pca_explained_variance_ratio"),
            separation=payload.get("simultaneous_separation"),
            trained_at=payload.get("trained_at"),
        )

    # ---------------- ⑤ 结果图像 ----------------

    def _build_images_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("图库目录"))
        self._gallery_combo = QComboBox()
        self._gallery_combo.addItem("概览声谱图（spectrograms）", SPECTROGRAM_DIR)
        self._gallery_combo.addItem("脉冲放大图（pulse_zooms）", PULSE_ZOOM_DIR)
        self._gallery_combo.addItem("全部结果图（figures）", FIGURE_DIR)
        self._gallery_combo.currentIndexChanged.connect(
            lambda _: self._refresh_images_tab()
        )
        refresh_button = QPushButton("刷新")
        refresh_button.clicked.connect(self._refresh_images_tab)
        top_row.addWidget(self._gallery_combo, stretch=1)
        top_row.addWidget(refresh_button)
        layout.addLayout(top_row)

        hint = QLabel(
            "PCA聚类图在训练后生成；声谱图在预测后生成"
            "（或训练时勾选诊断图）。选中文件后可查看大图。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._images_gallery = ImageGalleryPane()
        layout.addWidget(self._images_gallery, stretch=1)

        self._refresh_images_tab()
        return page

    def _refresh_images_tab(self) -> None:
        if not hasattr(self, "_images_gallery"):
            return
        directory = self._gallery_combo.currentData()
        count = self._images_gallery.load_directory(directory)
        self._append_log(f"结果图像：{directory.name} 下找到 {count} 个PNG。")

    # ---------------- ⑥ 批量整理与专家标注 ----------------

    def _build_curation_tab(self) -> QWidget:
        # 页面逻辑都在 curation_tab.py / curation_review.py，
        # 主窗口只负责挂载，避免继续膨胀。
        return CurationTab(
            log=self._append_log,
            track_worker=self._track_worker,
            settings=self._settings,
        )

    # ---------------- ⑦ 片段识别与叫声统计 ----------------

    def _build_segment_tab(self) -> QWidget:
        # 页面逻辑都在 segment_analysis_tab.py，算法在 segment_analysis.py，
        # 主窗口只负责挂载。
        return SegmentAnalysisTab(
            log=self._append_log,
            track_worker=self._track_worker,
            initial_model_path=self._settings.active_model_path,
        )
