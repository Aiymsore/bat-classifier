"""② 低相似度 pulse 扫描页面与人工候选区。"""
from __future__ import annotations

import json
import os
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDoubleSpinBox, QFileDialog,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QProgressBar, QPushButton, QSplitter, QTableView, QTabWidget, QVBoxLayout, QWidget,
)

from ..noise_template_scan import THRESHOLD_NAME, load_scan_pulses, move_to_candidates
from ..config import RESULTS_DIR
from .image_viewer import PulseImagePane
from .ui_utils import make_scrollable
from .workers import LowSimilarityScanWorker


class NoiseTemplateScanTab(QWidget):
    def __init__(self, log, track_worker, parent=None):
        super().__init__(parent)
        self._log = log
        self._track_worker = track_worker
        self._worker = None
        self._run_dir: Path | None = None
        self._rows: list[dict] = []
        self._rows_by_key: dict[str, dict] = {}

        form = QWidget()
        form_layout = QVBoxLayout(form)
        hint = QLabel(
            "复用第①模块扫描 WAV，与蝙蝠模板比较；仅保留严格低于设定相似度的 pulse。"
            "查看后可将选中的 pulse 移入候选区；候选样本导入第①模块后归为可疑声波。")
        hint.setWordWrap(True)
        form_layout.addWidget(hint)
        inputs = QGroupBox("输入与蝙蝠声波模板")
        grid = QGridLayout(inputs)
        self._input_edit = QLineEdit()
        self._input_edit.setPlaceholderText("选择 WAV 文件或文件夹，也可以粘贴路径")
        self._template_edit = QLineEdit()
        self._template_edit.setPlaceholderText("包含 CF、CFFM 等可信蝙蝠模板的文件夹")
        wav_button = QPushButton("选择 WAV")
        folder_button = QPushButton("选择文件夹")
        template_button = QPushButton("选择模板")
        wav_button.clicked.connect(self._choose_wav)
        folder_button.clicked.connect(self._choose_folder)
        template_button.clicked.connect(self._choose_templates)
        grid.addWidget(QLabel("扫描输入"), 0, 0)
        grid.addWidget(self._input_edit, 0, 1)
        grid.addWidget(wav_button, 0, 2)
        grid.addWidget(folder_button, 0, 3)
        grid.addWidget(QLabel("蝙蝠模板"), 1, 0)
        grid.addWidget(self._template_edit, 1, 1, 1, 2)
        grid.addWidget(template_button, 1, 3)
        form_layout.addWidget(inputs)

        params = QGroupBox("扫描参数")
        grid = QGridLayout(params)
        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0, 1)
        self._threshold_spin.setDecimals(3)
        self._threshold_spin.setSingleStep(0.05)
        self._threshold_spin.setValue(0.2)
        self._threshold_spin.setToolTip("只输出相似度严格低于此值的 pulse；等于此值不输出。")
        grid.addWidget(QLabel(THRESHOLD_NAME), 0, 0)
        grid.addWidget(self._threshold_spin, 0, 1)
        grid.addWidget(QLabel("输出：相似度 < 设定值"), 0, 2)
        self._window_combo = QComboBox()
        self._window_combo.addItem("自适应延长（512/1024 点）", "adaptive")
        self._window_combo.addItem("固定 256 点", "fixed")
        self._harmonic_combo = QComboBox()
        self._harmonic_combo.addItem("增强（倍率+轨迹+包络）", "enhanced")
        self._harmonic_combo.addItem("基础（2/3 倍率）", "basic")
        grid.addWidget(QLabel("同时分量分离窗"), 1, 0)
        grid.addWidget(self._window_combo, 1, 1, 1, 2)
        grid.addWidget(QLabel("谐波处理"), 2, 0)
        grid.addWidget(self._harmonic_combo, 2, 1, 1, 2)
        form_layout.addWidget(params)
        actions = QHBoxLayout()
        self._start_button = QPushButton("开始第一次扫描")
        self._stop_button = QPushButton("停止")
        self._stop_button.setEnabled(False)
        self._load_button = QPushButton("打开已有扫描")
        self._directory_button = QPushButton("打开输出目录")
        self._directory_button.setToolTip("打开当前扫描结果；尚未加载结果时打开第②模块输出目录。")
        self._start_button.clicked.connect(self._start)
        self._stop_button.clicked.connect(self._stop)
        self._load_button.clicked.connect(self._open_existing)
        self._directory_button.clicked.connect(self._open_directory)
        for button in (self._start_button, self._stop_button, self._load_button, self._directory_button):
            actions.addWidget(button)
        form_layout.addLayout(actions)
        self._progress_bar = QProgressBar()
        self._progress_label = QLabel("尚未扫描")
        self._progress_label.setWordWrap(True)
        self._output_label = QLabel("")
        self._output_label.setWordWrap(True)
        self._output_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form_layout.addWidget(self._progress_bar)
        form_layout.addWidget(self._progress_label)

        review = QWidget()
        review_layout = QVBoxLayout(review)
        review_actions = QHBoxLayout()
        self._counts_label = QLabel("低相似度 pulse：0 ｜ 候选：0")
        self._move_button = QPushButton("将选中的 pulse 移入候选 →")
        self._move_button.setEnabled(False)
        self._move_button.clicked.connect(self._move_selected)
        review_actions.addWidget(self._counts_label, 1)
        review_actions.addWidget(self._move_button)
        review_layout.addLayout(review_actions)
        self._review_tabs = QTabWidget()
        self._pending_table, self._pending_model = self._make_table()
        self._candidate_table, self._candidate_model = self._make_table()
        self._review_tabs.addTab(self._pending_table, "低相似度 pulse")
        self._review_tabs.addTab(self._candidate_table, "候选")
        self._review_tabs.currentChanged.connect(self._preview_selection)
        self._pending_table.selectionModel().selectionChanged.connect(self._preview_selection)
        self._candidate_table.selectionModel().selectionChanged.connect(self._preview_selection)
        self._preview = PulseImagePane()
        self._pulse_info = QLabel("")
        self._pulse_info.setWordWrap(True)
        pane = QWidget()
        pane_layout = QVBoxLayout(pane)
        pane_layout.addWidget(self._pulse_info)
        pane_layout.addWidget(self._preview, 1)
        results_split = QSplitter(Qt.Vertical)
        results_split.addWidget(self._review_tabs)
        results_split.addWidget(pane)
        results_split.setStretchFactor(0, 0)
        results_split.setStretchFactor(1, 1)
        results_split.setSizes([150, 650])
        review_layout.addWidget(results_split, 1)
        split = QSplitter(Qt.Vertical)
        self._form_scroll = make_scrollable(form)
        split.addWidget(self._form_scroll)
        split.addWidget(review)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([260, 650])
        self._form_collapsed = False
        self._form_toggle = QPushButton("收起扫描参数")
        self._form_toggle.clicked.connect(lambda: self._set_form_collapsed(not self._form_collapsed))
        layout = QVBoxLayout(self)
        layout.addWidget(self._form_toggle)
        layout.addWidget(self._output_label)
        layout.addWidget(split)

    def _set_form_collapsed(self, collapsed):
        self._form_collapsed = bool(collapsed)
        self._form_scroll.setVisible(not collapsed)
        self._form_toggle.setText("展开扫描参数" if collapsed else "收起扫描参数")

    @staticmethod
    def _make_table():
        model = QStandardItemModel()
        table = QTableView()
        table.setModel(model)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSortingEnabled(True)
        return table, model

    def _choose_wav(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择扫描 WAV", "", "WAV (*.wav *.WAV)")
        if path:
            self._input_edit.setText(path)

    def _choose_folder(self):
        path = QFileDialog.getExistingDirectory(self, "选择扫描文件夹")
        if path:
            self._input_edit.setText(path)

    def _choose_templates(self):
        path = QFileDialog.getExistingDirectory(self, "选择蝙蝠声波模板文件夹")
        if path:
            self._template_edit.setText(path)

    def _start(self):
        source_text = self._input_edit.text().strip().strip('"')
        template_text = self._template_edit.text().strip().strip('"')
        if not source_text or not Path(source_text).exists():
            QMessageBox.warning(self, "输入路径", "请选择存在的 WAV 或文件夹。")
            return
        if not template_text or not Path(template_text).is_dir():
            QMessageBox.warning(self, "模板路径", "请选择蝙蝠声波模板文件夹。")
            return
        worker = LowSimilarityScanWorker(Path(source_text), Path(template_text),
            self._threshold_spin.value(), {
                "separation_window_mode": self._window_combo.currentData(),
                "harmonic_grouping_mode": self._harmonic_combo.currentData(),
            })
        self._worker = worker
        worker.progress.connect(self._on_progress)
        worker.chunk_progress.connect(self._on_chunk_progress)
        worker.done.connect(self._on_done)
        worker.failed.connect(self._on_failed)
        self._set_busy(True)
        self._progress_bar.setValue(0)
        self._progress_label.setText("正在加载模板并扫描…")
        self._log(f"② 第一次扫描：{source_text}，{THRESHOLD_NAME} < {self._threshold_spin.value():.3f}")
        self._track_worker(worker)

    def _stop(self):
        if self._worker:
            self._worker.stop()
            self._progress_label.setText("正在等待当前块结束；已输出的低分 pulse 将保留。")

    def _set_busy(self, busy):
        for widget in (self._start_button, self._load_button, self._input_edit,
                       self._template_edit, self._threshold_spin,
                       self._window_combo, self._harmonic_combo):
            widget.setEnabled(not busy)
        self._stop_button.setEnabled(busy)
        self._move_button.setEnabled(not busy and bool(self._pending_table.selectionModel().selectedRows()))

    def _on_progress(self, current, total, name):
        self._progress_label.setText(f"文件 {current}/{total}：{name}")

    def _on_chunk_progress(self, payload):
        self._progress_bar.setValue(round(float(payload.get("progress_ratio", 0)) * 100))
        self._progress_label.setText(
            f"{payload.get('segment_file', '')} ｜ "
            f"已处理 {float(payload.get('processed_until_s', 0)):.1f}/"
            f"{float(payload.get('recording_duration_s', 0)):.1f} 秒 ｜ "
            f"已输出低分 pulse：{payload.get('selected_pulse_count', 0)}")

    def _on_done(self, result):
        self._worker = None
        self._set_busy(False)
        self.load_run(Path(result["run_dir"]))
        status = result.get("status")
        self._progress_label.setText(
            f"{'已停止' if status == 'cancelled' else '扫描结束'} ｜ "
            f"处理 pulse：{result.get('processed_pulse_count', 0)} ｜ "
            f"低分输出：{result.get('selected_pulse_count', 0)} ｜ 失败文件：{result.get('failed_count', 0)}")
        if status == "success":
            self._progress_bar.setValue(100)
        self._log(f"② 输出目录：{result['run_dir']}")

    def _on_failed(self, message):
        self._worker = None
        self._set_busy(False)
        self._progress_label.setText(f"扫描失败：{message}")
        self._log(f"② 扫描失败：{message}")

    def _open_existing(self):
        path = QFileDialog.getExistingDirectory(self, "打开第②模块扫描结果")
        if path:
            try:
                self.load_run(Path(path))
            except Exception as error:
                QMessageBox.warning(self, "打开失败", str(error))

    def load_run(self, run_dir: Path, *, materialize_candidates=True):
        state = json.loads((run_dir / "review_state.json").read_text(encoding="utf-8"))
        # 旧版本已选择的候选也落实为文件移动及 pulse 命名。
        if materialize_candidates and state.get("candidate_keys"):
            move_to_candidates(run_dir, state["candidate_keys"])
        rows = load_scan_pulses(run_dir)
        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        self._run_dir = Path(run_dir)
        self._rows = rows
        self._rows_by_key = {row["pulse_key"]: row for row in rows}
        self._output_label.setText(f"输出目录：{run_dir} ｜ 本次{THRESHOLD_NAME} < {metadata['similarity_threshold']}")
        self._directory_button.setEnabled(True)
        self._refresh_tables()
        self._set_form_collapsed(True)

    def _refresh_tables(self):
        self._preview.show_image(None)
        self._pulse_info.clear()
        for table, model, status in ((self._pending_table, self._pending_model, "unselected"),
                                      (self._candidate_table, self._candidate_model, "candidate")):
            table.setSortingEnabled(False)
            model.clear()
            model.setHorizontalHeaderLabels(["录音", "WAV 编号", THRESHOLD_NAME, "峰值 kHz", "SNR dB"])
            for row in self._rows:
                if row["review_status"] != status:
                    continue
                values = [row["segment_file"], row["template_filename"], round(float(row["confidence"]), 4),
                          round(float(row["peak_freq_khz"]), 1), round(float(row["snr_db"]), 2)]
                items = []
                for value in values:
                    item = QStandardItem()
                    item.setData(value, Qt.DisplayRole)
                    item.setData(row["pulse_key"], Qt.UserRole)
                    items.append(item)
                model.appendRow(items)
            table.setSortingEnabled(True)
            table.resizeColumnsToContents()
        pending = self._pending_model.rowCount()
        candidates = self._candidate_model.rowCount()
        self._counts_label.setText(f"低相似度 pulse：{pending} ｜ 候选：{candidates}")
        self._review_tabs.setTabText(0, f"低相似度 pulse（{pending}）")
        self._review_tabs.setTabText(1, f"候选（{candidates}）")
        self._move_button.setEnabled(False)

    def _current_row(self):
        table = self._pending_table if self._review_tabs.currentIndex() == 0 else self._candidate_table
        index = table.currentIndex()
        return self._rows_by_key.get(index.data(Qt.UserRole)) if index.isValid() else None

    def _preview_selection(self, *_):
        row = self._current_row()
        if not row or self._run_dir is None:
            self._move_button.setEnabled(False)
            self._preview.show_image(None)
            self._pulse_info.clear()
            return
        image = row.get("pulse_spectrogram_file")
        self._preview.show_image(self._run_dir / image if image else None)
        self._pulse_info.setText(
            f"{row['template_filename']}（{row['pulse_id']}） ｜ {THRESHOLD_NAME}：{float(row['confidence']):.4f} ｜ "
            f"t={float(row['start_s']):.4f}–{float(row['end_s']):.4f} 秒\n"
            f"最相似蝙蝠模板：{row.get('template_label', '')} ｜ "
            f"{'候选（用户选择）' if row['review_status'] == 'candidate' else '尚未选择'}"
            + (f"\nWAV 导出失败：{row['audio_export_error']}" if row.get("audio_export_error") else ""))
        self._move_button.setEnabled(self._worker is None and self._review_tabs.currentIndex() == 0
                                     and bool(self._pending_table.selectionModel().selectedRows()))

    def _move_selected(self):
        if self._run_dir is None:
            return
        keys = [index.data(Qt.UserRole) for index in self._pending_table.selectionModel().selectedRows()]
        if not keys:
            return
        try:
            count = move_to_candidates(self._run_dir, keys)
            self.load_run(self._run_dir)
            self._log(f"② 用户选择：{count} 个 pulse 已移入候选。")
        except Exception as error:
            self.load_run(self._run_dir, materialize_candidates=False)
            QMessageBox.warning(self, "候选保存失败", str(error))

    def _open_directory(self):
        target = (self._run_dir or RESULTS_DIR / "noise_template_scan").resolve()
        try:
            if self._run_dir is None:
                target.mkdir(parents=True, exist_ok=True)
            if not target.is_dir():
                raise FileNotFoundError(f"输出目录不存在：{target}")
            if os.name == "nt":
                os.startfile(str(target), "open")
            elif not QDesktopServices.openUrl(QUrl.fromLocalFile(str(target))):
                raise OSError("系统未能打开文件管理器。")
            self._log(f"② 打开输出目录：{target}")
        except OSError as error:
            self._log(f"② 打开输出目录失败：{target} ｜ {error}")
            QMessageBox.warning(self, "打开输出目录失败", f"{target}\n\n{error}")
