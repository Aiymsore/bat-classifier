"""⑦ 片段识别与叫声统计 页签。

职责划分：
- 本文件：输入选择（外部 WAV/文件夹）、模型状态、参数、按钮、进度、
  片段级结果表格 + 选中片段详情 + 总体统计，以及把 SegmentAnalysisWorker 接到 Qt；
- segment_analysis.py：全部科研算法（检测叫声 + 逐叫声分类 + 两级统计，脱离 Qt）；
- workers.py：QThread 线程、进度/日志/异常转发、停止。

业务语义（页面显式提示）：
- 统计单位是“切割后的 WAV 片段”，一段内可能含 N 次叫声；
- 统计的是声学 call 数量，不代表蝙蝠个体数量；
- 当前部署的是临时声型（A-D）模型时，明确标注“不是专家确认物种模型”。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, Signal, QUrl
from PySide6.QtGui import (
    QDesktopServices,
    QStandardItem,
    QStandardItemModel,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ..config import (
    COMBINED_DEMO_MODEL_PATH,
    SEGMENT_LOW_CONFIDENCE_ACTION,
    SEGMENT_MIN_CONFIDENCE,
    SEGMENT_MIXED_PURITY_THRESHOLD,
    SEGMENT_UNCERTAIN_LABEL,
)
from .model_picker import ModelPicker
from .ui_utils import make_scrollable


class SegmentAnalysisTab(QWidget):
    """⑦ 页面：输入 + 模型状态 + 参数 + 进度 + 片段/总体统计。"""

    # 当前使用的模型发生变化（供 ③/④ 同步选择）
    model_changed = Signal(str)

    def __init__(
        self,
        log: Callable[[str], None],
        track_worker: Callable[[object], None],
        parent: QWidget | None = None,
        initial_model_path: str = "",
    ) -> None:
        super().__init__(parent)
        self._log = log
        self._track_worker = track_worker
        self._input_path: Path | None = None
        self._worker = None
        self._combined_payload: dict | None = None
        self._model_loaded = False
        self._loaded_model_path: str = ""
        self._initial_model_path = str(initial_model_path)
        self._run_dir: Path | None = None
        self._labels: list[str] = []
        self._uncertain_label: str = SEGMENT_UNCERTAIN_LABEL
        self._segment_rows: list[dict] = []

        self._build_ui()
        self._refresh_model_status()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # 上半部分表单区：内容高过可视高度时自动出现滚动条（小屏幕可滚）。
        form = QWidget()
        form_layout = QVBoxLayout(form)
        form_layout.setContentsMargins(0, 0, 0, 0)

        # ---- 模型状态 ----
        model_group = QGroupBox("模型状态")
        model_layout = QGridLayout()
        self._model_status_label = QLabel("模型状态：检查中…")
        self._model_status_label.setWordWrap(True)
        self._model_path_label = QLabel("")
        self._model_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._model_warning_label = QLabel("")
        self._model_warning_label.setWordWrap(True)
        self._model_warning_label.setStyleSheet("color: #b06000;")
        self._calibration_label = QLabel("")
        self._calibration_label.setWordWrap(True)
        model_layout.addWidget(self._model_status_label, 0, 0, 1, 3)
        model_layout.addWidget(self._model_path_label, 1, 0, 1, 3)
        model_layout.addWidget(self._model_warning_label, 2, 0, 1, 3)
        model_layout.addWidget(self._calibration_label, 3, 0, 1, 3)

        model_layout.addWidget(
            QLabel("使用模型（② 训练的模型 / 已导入的模型，可随时切换）"),
            4,
            0,
            1,
            3,
        )
        self._model_picker = ModelPicker(
            self._initial_model_path, log=self._log
        )
        model_layout.addWidget(self._model_picker, 5, 0, 1, 3)
        self._model_picker.model_changed.connect(self._on_model_changed)
        model_group.setLayout(model_layout)
        form_layout.addWidget(model_group)

        # ---- 输入数据 ----
        input_group = QGroupBox("输入数据（已切割好的 WAV 片段）")
        input_layout = QGridLayout()
        self._input_edit = QLineEdit()
        self._input_edit.setReadOnly(True)
        self._input_edit.setPlaceholderText(
            "选择一个 WAV 片段文件，或一个包含大量片段 WAV 的文件夹。"
        )
        wav_button = QPushButton("选择单个 WAV")
        folder_button = QPushButton("选择 WAV 文件夹")
        wav_button.clicked.connect(self._pick_wav)
        folder_button.clicked.connect(self._pick_folder)
        self._found_label = QLabel("尚未选择输入。")
        input_layout.addWidget(self._input_edit, 0, 0, 1, 3)
        input_layout.addWidget(wav_button, 1, 0)
        input_layout.addWidget(folder_button, 1, 1)
        input_layout.addWidget(self._found_label, 2, 0, 1, 3)
        input_group.setLayout(input_layout)
        form_layout.addWidget(input_group)

        # ---- 参数 ----
        params_group = QGroupBox("叫声检测与分类参数")
        params_layout = QGridLayout()
        self._quality_check = QCheckBox(
            "使用质量筛选后的有效 pulse 计数"
        )
        self._quality_check.setChecked(True)
        params_layout.addWidget(self._quality_check, 0, 0, 1, 3)

        params_layout.addWidget(QLabel("最低置信度"), 1, 0)
        self._confidence_spin = QDoubleSpinBox()
        self._confidence_spin.setRange(0.0, 1.0)
        self._confidence_spin.setSingleStep(0.05)
        self._confidence_spin.setDecimals(2)
        self._confidence_spin.setValue(SEGMENT_MIN_CONFIDENCE)
        params_layout.addWidget(self._confidence_spin, 1, 1)

        params_layout.addWidget(QLabel("低于该置信度"), 2, 0)
        self._low_conf_combo = QComboBox()
        self._low_conf_combo.addItem("标记为 uncertain", "uncertain")
        self._low_conf_combo.addItem("保留模型原标签", "keep")
        if SEGMENT_LOW_CONFIDENCE_ACTION == "keep":
            self._low_conf_combo.setCurrentIndex(1)
        params_layout.addWidget(self._low_conf_combo, 2, 1)

        self._save_audio_check = QCheckBox("导出每段独立叫声 WAV（segments/<段>/）")
        self._save_audio_check.setChecked(True)
        self._save_figure_check = QCheckBox("生成片段声谱图 + 叫声标注（figures/）")
        self._save_figure_check.setChecked(True)
        params_layout.addWidget(self._save_audio_check, 3, 0, 1, 2)
        params_layout.addWidget(self._save_figure_check, 3, 2, 1, 2)
        params_group.setLayout(params_layout)
        form_layout.addWidget(params_group)

        # ---- 按钮 ----
        button_row = QHBoxLayout()
        self._start_button = QPushButton("开始分析")
        self._stop_button = QPushButton("停止")
        self._stop_button.setEnabled(False)
        self._export_button = QPushButton("导出结果")
        self._open_dir_button = QPushButton("打开结果目录")
        self._start_button.clicked.connect(self._start)
        self._stop_button.clicked.connect(self._stop)
        self._export_button.clicked.connect(self._export_results)
        self._open_dir_button.clicked.connect(self._open_run_dir)
        for button in (
            self._start_button,
            self._stop_button,
            self._export_button,
            self._open_dir_button,
        ):
            button_row.addWidget(button)
        button_row.addStretch(1)
        form_layout.addLayout(button_row)

        # ---- 进度 ----
        progress_group = QGroupBox("进度")
        progress_layout = QGridLayout()
        self._progress_label = QLabel("尚未开始。")
        self._progress_label.setWordWrap(True)
        self._stats_label = QLabel("")
        self._stats_label.setWordWrap(True)
        progress_layout.addWidget(self._progress_label, 0, 0, 1, 4)
        progress_layout.addWidget(self._stats_label, 1, 0, 1, 4)
        progress_group.setLayout(progress_layout)
        form_layout.addWidget(progress_group)

        # ---- 结果：左表 + 右详情 ----
        result_splitter = QSplitter(Qt.Horizontal)

        self._result_model = QStandardItemModel()
        self._result_table = QTableView()
        self._result_table.setModel(self._result_model)
        self._result_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self._result_table.setSelectionBehavior(QTableView.SelectRows)
        self._result_table.setSelectionMode(QTableView.SingleSelection)
        self._result_table.clicked.connect(self._on_row_selected)
        result_splitter.addWidget(self._result_table)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        detail_group = QGroupBox("片段详情")
        detail_layout = QVBoxLayout()
        self._detail_view = QPlainTextEdit()
        self._detail_view.setReadOnly(True)
        detail_layout.addWidget(self._detail_view)
        self._figure_button = QPushButton("查看片段声谱图")
        self._figure_button.setEnabled(False)
        self._figure_button.clicked.connect(self._open_figure)
        detail_layout.addWidget(self._figure_button)
        detail_group.setLayout(detail_layout)
        right_layout.addWidget(detail_group, stretch=2)

        overall_group = QGroupBox("总体统计（按类别叫声数累加）")
        overall_layout = QVBoxLayout()
        self._overall_view = QPlainTextEdit()
        self._overall_view.setReadOnly(True)
        overall_layout.addWidget(self._overall_view)
        overall_group.setLayout(overall_layout)
        right_layout.addWidget(overall_group, stretch=1)

        result_splitter.addWidget(right_panel)
        result_splitter.setStretchFactor(0, 3)
        result_splitter.setStretchFactor(1, 2)
        result_splitter.setMinimumHeight(220)

        vertical = QSplitter(Qt.Vertical)
        vertical.setChildrenCollapsible(False)
        form_scroll = make_scrollable(form)
        form_scroll.setMinimumHeight(200)
        vertical.addWidget(form_scroll)
        vertical.addWidget(result_splitter)
        vertical.setStretchFactor(0, 0)
        vertical.setStretchFactor(1, 1)
        vertical.setSizes([460, 320])
        root.addWidget(vertical, stretch=1)

        hint = QLabel(
            "语义提醒：统计的是“声学检测到的 call 数量”，不是蝙蝠个体数量；"
            "一段 WAV 内部可能有多次叫声，逐次分类后再按该 WAV 汇总。"
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

    # ------------------------------------------------------------- 模型状态
    def _ensure_model(self) -> dict | None:
        path = self._model_picker.current_path_text()
        if self._model_loaded and path == self._loaded_model_path:
            return self._combined_payload
        try:
            from ..segment_analysis import load_classification_model

            self._combined_payload = load_classification_model(
                Path(path) if path else None
            )
        except Exception as error:  # noqa: BLE001
            self._combined_payload = None
            self._log(f"加载分类模型失败：{error}")
        self._model_loaded = True
        self._loaded_model_path = path
        return self._combined_payload

    def _invalidate_model(self) -> None:
        """强制下次 _ensure_model 重新读盘（换模型 / 重新训练后调用）。"""
        self._model_loaded = False
        self._combined_payload = None
        self._loaded_model_path = ""

    def _on_model_changed(self, path: str) -> None:
        self._invalidate_model()
        self._refresh_model_status()
        self.model_changed.emit(str(path))
        self._log(f"⑦ 当前使用模型：{path}")

    def refresh_models(self) -> None:
        """外部（如 ② 训练完成、④ 导入模型）触发：重扫模型并刷新状态。"""
        current = self._model_picker.current_path_text()
        self._model_picker.refresh(current)
        self._invalidate_model()
        self._refresh_model_status()

    def current_model_path(self) -> str:
        return self._model_picker.current_path_text()

    @property
    def model_picker(self) -> ModelPicker:
        """供主窗口与其他页签同步“当前使用模型”的选择。"""
        return self._model_picker

    def _refresh_model_status(self) -> None:
        from ..segment_analysis import describe_model

        payload = self._ensure_model()
        info = describe_model(
            payload, model_path=self._model_picker.current_path_text()
        )
        self._labels = list(info.get("labels") or [])

        # 置信度性质：校准概率 vs 未校准的模型支持度（绝不含糊）
        if info["loaded"] and info.get("is_calibrated"):
            self._calibration_label.setStyleSheet("color: #1b5e20;")
            self._calibration_label.setText(
                f"置信度：校准概率（CalibratedClassifierCV, "
                f"method={info.get('calibration_method') or '未标注'}）。"
            )
        elif info["loaded"]:
            self._calibration_label.setStyleSheet("color: #b06000;")
            self._calibration_label.setText(
                "置信度：未校准——为分类器原始 predict_proba（模型支持度），"
                "不能当“预测正确的概率”读；"
                "可在 ② 训练时自动生成校准器，或用“补校准”功能添加。"
            )
        else:
            self._calibration_label.setText("")

        if info["loaded"]:
            origin = (
                "② 训练模型"
                if Path(str(info.get("model_path", ""))).name
                == COMBINED_DEMO_MODEL_PATH.name
                else "导入/其他模型"
            )
            self._model_status_label.setText(
                f"物种分类模型：已加载（来源：{origin}，可分类）"
            )
            self._model_path_label.setText(
                f"模型路径：{info.get('model_path', '')}"
            )
        else:
            self._model_status_label.setText(
                "未检测到可用分类模型。此时仅可运行“叫声检测与计数”"
                "（label = unavailable）。"
            )
            self._model_path_label.setText(
                f"期望模型文件：{info.get('model_path', '')}"
            )

        if info.get("is_temporary_acoustic_model"):
            self._model_warning_label.setText(
                "⚠ 当前模型为临时声型模型（A-D 声学相似类型），"
                "不是专家确认物种模型。"
            )
        else:
            self._model_warning_label.setText(
                info.get("warning", "") or ""
            )

    # ------------------------------------------------------------- 输入
    def _pick_wav(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self, "选择单个 WAV", "", "音频 (*.wav)"
        )
        if chosen:
            self._set_input(Path(chosen))

    def _pick_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "选择 WAV 文件夹")
        if chosen:
            self._set_input(Path(chosen))

    def _set_input(self, path: Path) -> None:
        self._input_path = path
        self._input_edit.setText(str(path))
        self._found_label.setText("统计中…")
        try:
            from ..audio_io import find_wav_files

            if path.is_dir():
                total = len(find_wav_files(path))
                self._found_label.setText(f"找到：{total:,} 个 WAV")
            else:
                self._found_label.setText("单个 WAV 文件。")
        except Exception as error:  # noqa: BLE001
            self._found_label.setText(f"读取失败：{error}")

    # ------------------------------------------------------------- 启动/停止
    def _collect_config(self) -> dict:
        return {
            "min_confidence": float(self._confidence_spin.value()),
            "low_confidence_action": str(self._low_conf_combo.currentData()),
            "use_quality_filter": bool(self._quality_check.isChecked()),
            "save_pulse_audio": bool(self._save_audio_check.isChecked()),
            "save_figures": bool(self._save_figure_check.isChecked()),
            "uncertain_label": SEGMENT_UNCERTAIN_LABEL,
            "mixed_purity_threshold": SEGMENT_MIXED_PURITY_THRESHOLD,
        }

    def _start(self) -> None:
        if self._input_path is None:
            QMessageBox.warning(self, "提示", "请先选择 WAV 或文件夹。")
            return
        if not self._input_path.exists():
            QMessageBox.warning(self, "提示", f"路径不存在：{self._input_path}")
            return

        from .workers import SegmentAnalysisWorker

        payload = self._ensure_model()

        # ---- 断点续跑（§12）：上次未完成且有 checkpoint 时明确询问 ----
        resume: bool | None = None
        output_root: Path | None = None
        previous_run_dir = self._run_dir
        if previous_run_dir is not None:
            from ..segment_stream import find_resumable_checkpoints

            pending = find_resumable_checkpoints(previous_run_dir)
            if pending:
                answer = QMessageBox.question(
                    self,
                    "发现未完成的分析",
                    f"上次运行（{previous_run_dir.name}）有 "
                    f"{len(pending)} 个片段中断未完成。\n"
                    "已处理部分与 checkpoint 均已保存。\n\n"
                    "是 = 断点续跑（从上次位置继续，绝不重复计数）\n"
                    "否 = 重新开始（新建输出目录）",
                )
                if answer == QMessageBox.StandardButton.Yes:
                    resume = True
                    output_root = previous_run_dir
                    self._log(
                        f"断点续跑：从 {previous_run_dir.name} 的 checkpoint 继续"
                    )
                else:
                    self._log("重新开始：忽略上次 checkpoint，新建输出目录")

        self._run_dir = output_root
        self._result_model.clear()
        self._detail_view.clear()
        self._overall_view.clear()
        self._figure_button.setEnabled(False)

        worker = SegmentAnalysisWorker(
            self._input_path,
            payload,
            self._collect_config(),
            output_root=output_root,
            model_path=self._model_picker.current_path_text(),
            resume=resume,
        )
        self._worker = worker
        worker.progress.connect(self._on_progress)
        worker.chunk_progress.connect(self._on_chunk_progress)
        worker.stats.connect(self._on_stats)
        worker.done.connect(self._on_done)
        worker.failed.connect(self._on_failed)
        self._set_busy(True)
        self._log(f"开始片段识别与叫声统计：{self._input_path}")
        self._track_worker(worker)

    def _stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._log(
                "已请求停止：将在当前 chunk/片段处理完后优雅收尾。"
                "已处理部分已保存（结果增量落盘），下次开始时可断点续跑。"
            )

    def _set_busy(self, busy: bool) -> None:
        self._start_button.setEnabled(not busy)
        self._stop_button.setEnabled(busy)

    # ------------------------------------------------------------- 进度回调
    def _on_progress(self, current: int, total: int, current_file: str) -> None:
        self._progress_label.setText(
            f"{current} / {total}\n当前：{current_file}"
        )

    def _on_chunk_progress(self, payload: object) -> None:
        """⑦ 长录音块级进度（§13）：已处理时长/速度/已检测 call/分块。"""
        if not isinstance(payload, dict):
            return
        duration = float(payload.get("recording_duration_s", 0.0))
        processed = float(payload.get("processed_until_s", 0.0))
        speed = float(payload.get("speed_realtime") or 0.0)
        calls = int(payload.get("detected_call_count", 0))
        skipped = int(payload.get("fast_skipped_chunk_count", 0))
        chunk_index = payload.get("chunk_index")
        chunk_count = payload.get("chunk_count")
        if duration >= 120.0:
            time_text = f"已处理 {processed / 60.0:.0f} / {duration / 60.0:.0f} 分钟"
        else:
            time_text = f"已处理 {processed:.0f} / {duration:.0f} 秒"
        lines = [
            f"{payload.get('segment_file', '')}",
            f"{time_text}，速度约 {speed:.0f}× real-time",
            f"已检测 call：{calls}",
        ]
        if chunk_index is not None and chunk_count:
            lines.append(
                f"分块 {int(chunk_index) + 1} / {int(chunk_count)}"
                + (f"（快筛跳过 {skipped} 块）" if skipped else "")
            )
        self._progress_label.setText("\n".join(lines))

    def _on_stats(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        self._stats_label.setText(
            f"预测：{payload.get('major_label') or '—'} ｜ "
            f"检测叫声：{payload.get('detected_call_count', 0)} ｜ "
            f"成功：{payload.get('success', 0)} ｜ "
            f"失败：{payload.get('failed', 0)}"
        )

    def _on_failed(self, message: str) -> None:
        self._set_busy(False)
        self._worker = None
        self._log(f"片段分析任务失败：{message}")
        QMessageBox.critical(self, "任务失败", message)

    def _on_done(self, result: object) -> None:
        self._set_busy(False)
        self._worker = None
        if not isinstance(result, dict):
            return
        self._run_dir = Path(str(result.get("run_dir", "")))
        self._segment_rows = list(result.get("segment_results", []))
        self._labels = list(result.get("labels") or [])
        self._uncertain_label = str(
            result.get("uncertain_label") or SEGMENT_UNCERTAIN_LABEL
        )

        success = result.get("success_count", 0)
        failed = result.get("failed_count", 0)
        total_calls = sum(
            int(r.get("detected_call_count", 0)) for r in self._segment_rows
        )
        self._log(
            f"片段分析完成：{self._run_dir.name} ｜ "
            f"片段 {result.get('segment_count', 0)} 个，"
            f"成功 {success}，失败 {failed}，"
            f"有效蝙蝠叫声合计 {total_calls} 次。"
        )

        self._populate_table(result.get("segment_summary"))
        self._populate_overall(result.get("overall_summary"))

    # ------------------------------------------------------------- 结果表
    def _populate_table(self, summary_df) -> None:
        self._result_model.clear()
        if summary_df is None or len(summary_df) == 0:
            self._result_model.setHorizontalHeaderLabels(["（无结果）"])
            return

        headers = ["WAV片段", "主要类别", "有效叫声"]
        headers += list(self._labels)
        headers += ["Unknown"]
        self._result_model.setHorizontalHeaderLabels(headers)

        uncertain_key = self._uncertain_label
        for _, row in summary_df.iterrows():
            items = [
                QStandardItem(str(row.get("segment_file", ""))),
                QStandardItem(str(row.get("major_label", "") or "—")),
                QStandardItem(str(int(row.get("total_calls", 0)))),
            ]
            for label in self._labels:
                items.append(QStandardItem(str(int(row.get(label, 0)))))
            items.append(QStandardItem(str(int(row.get(uncertain_key, 0)))))
            for item in items:
                item.setEditable(False)
            self._result_model.appendRow(items)

    def _populate_overall(self, overall_df) -> None:
        if overall_df is None or len(overall_df) == 0:
            self._overall_view.setPlainText("（无总体统计）")
            return
        lines = ["类别            出现片段数   叫声次数"]
        for _, row in overall_df.iterrows():
            species = str(row.get("species", ""))
            seg_count = int(row.get("segment_count", 0))
            call_count = int(row.get("call_count", 0))
            lines.append(f"{species:<14} {seg_count:>10} {call_count:>11}")
        lines.append("")
        lines.append(
            "（注：叫声次数为该类别在所有片段中的声学 call 总数，"
            "不代表蝙蝠个体数量；出现片段数=该类别至少出现 1 次的 WAV 段数）"
        )
        self._overall_view.setPlainText("\n".join(lines))

    # ------------------------------------------------------------- 详情
    def _on_row_selected(self, index) -> None:
        row = index.row()
        if row < 0 or row >= len(self._segment_rows):
            return
        result = self._segment_rows[row]
        self._show_detail(result)
        self._figure_button.setEnabled(
            self._run_dir is not None
            and (self._run_dir / "figures" / f"{result.get('segment_id', '')}.png").exists()
        )

    def _show_detail(self, result: dict) -> None:
        label_counts = result.get("label_counts", {}) or {}
        duration = float(result.get("duration_s", 0.0))
        lines = [
            f"{result.get('segment_file', '')}",
            "",
            f"时长：{duration:.2f} s",
            f"有效监测时长：{float(result.get('valid_monitoring_duration_s', duration)):.2f} s",
            f"检测候选：{result.get('candidate_call_count', 0)}",
            f"通过质量筛选：{result.get('valid_call_count', 0)}",
            f"噪声（不计入叫声）：{result.get('noise_count', 0)}",
            f"有效蝙蝠叫声：{result.get('detected_call_count', 0)}",
            f"叫声频率：{result.get('calls_per_second', 0.0):.2f} 次/秒"
            f"（分母 = 有效监测时长）",
            f"分块：{result.get('processed_chunk_count', 0)} 块处理 / "
            f"{result.get('fast_skipped_chunk_count', 0)} 块快筛跳过",
            "",
        ]
        if result.get("export_truncated"):
            lines.append(
                f"pulse WAV 导出已达上限（导出 "
                f"{result.get('exported_pulse_count', 0)} 个）——"
                "仅影响保存的 WAV 数，计数/统计不受影响。"
            )
            lines.append("")
        if result.get("resumed"):
            lines.append("本次为断点续跑（结果已与中断前去重合并）。")
            lines.append("")
        for label in self._labels:
            lines.append(f"{label}：{int(label_counts.get(label, 0))}")
        lines.append(
            f"无法确定（{self._uncertain_label}）："
            f"{int(label_counts.get(self._uncertain_label, 0))}"
        )
        lines.append("")
        major = result.get("major_label", "") or "—"
        lines.append(f"主要类别：{major}")
        lines.append(f"纯度：{float(result.get('purity', 0.0)):.1%}")
        if result.get("consistency_note"):
            lines.append(f"一致性：{result.get('consistency_note')}")
        if result.get("status") != "success":
            lines.append("")
            lines.append(f"处理状态：{result.get('status', '')}")
            lines.append(f"错误：{result.get('error_message', '')}")
        lines.append("")
        lines.append(
            "（统计为声学 call 数量，非蝙蝠个体数量）"
        )
        self._detail_view.setPlainText("\n".join(lines))

    def _open_figure(self) -> None:
        if self._run_dir is None:
            return
        index = self._result_table.currentIndex()
        row = index.row() if index.isValid() else -1
        if row < 0 or row >= len(self._segment_rows):
            return
        segment_id = self._segment_rows[row].get("segment_id", "")
        figure_path = self._run_dir / "figures" / f"{segment_id}.png"
        if not figure_path.exists():
            QMessageBox.information(self, "提示", "该片段没有生成声谱图。")
            return
        from .image_viewer import ImagePreviewDialog

        dialog = ImagePreviewDialog(figure_path, self)
        dialog.exec()

    # ------------------------------------------------------------- 导出
    def _export_results(self) -> None:
        if self._run_dir is None:
            QMessageBox.warning(self, "提示", "请先完成一次分析再导出。")
            return
        target = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not target:
            return
        target_dir = Path(target)
        try:
            import shutil

            copied = []
            for name in (
                "segment_summary.csv",
                "pulse_predictions.csv",
                "overall_species_summary.csv",
                "run_metadata.json",
            ):
                source = self._run_dir / name
                if source.exists():
                    shutil.copy2(source, target_dir / name)
                    copied.append(name)
            self._log(f"已导出结果到 {target_dir}：{', '.join(copied)}")
        except Exception as error:  # noqa: BLE001
            QMessageBox.critical(self, "导出失败", str(error))

    def _open_run_dir(self) -> None:
        target = self._run_dir
        if target is None or not target.exists():
            QMessageBox.information(self, "提示", "尚未生成结果目录。")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))
