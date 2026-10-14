"""① 模板声波匹配、片段识别与叫声统计。"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Callable

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import (
    QDesktopServices,
    QStandardItem,
    QStandardItemModel,
)
from PySide6.QtWidgets import (
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
    SEGMENT_MIN_CONFIDENCE,
    SEGMENT_MIXED_PURITY_THRESHOLD,
)
from .ui_utils import make_scrollable


class SegmentAnalysisTab(QWidget):
    """① 页面：输入 + 模型状态 + 参数 + 进度 + 片段/总体统计。"""


    def __init__(
        self,
        log: Callable[[str], None],
        track_worker: Callable[[object], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._log = log
        self._track_worker = track_worker
        self._input_path: Path | None = None
        self._worker = None
        self._run_dir: Path | None = None
        self._labels: list[str] = []
        self._uncertain_label: str = "unknown"
        self._segment_rows: list[dict] = []
        self._template_dir: Path | None = None

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # 上半部分表单区：内容高过可视高度时自动出现滚动条（小屏幕可滚）。
        form = QWidget()
        form_layout = QVBoxLayout(form)
        form_layout.setContentsMargins(0, 0, 0, 0)

        # ---- 模板注入（第七模块独立推理模式） ----
        template_group = QGroupBox("模板声波注入（按可信文件名自动合并样本）")
        template_layout = QGridLayout()
        self._template_edit = QLineEdit()
        self._template_edit.setReadOnly(True)
        self._template_edit.setPlaceholderText(
            "例如 CF1_47K1、CF1_47K2 会合并为模板类别 CF1_47K。"
        )
        template_button = QPushButton("选择模板文件夹")
        template_button.clicked.connect(self._pick_template_folder)
        self._template_status = QLabel("请选择模板文件夹；模板文件名决定类别。")
        self._template_status.setWordWrap(True)
        self._template_status.setStyleSheet("color: #555;")
        template_layout.addWidget(self._template_edit, 0, 0, 1, 3)
        template_layout.addWidget(template_button, 1, 0)
        template_layout.addWidget(self._template_status, 1, 1, 1, 2)
        template_group.setLayout(template_layout)
        form_layout.addWidget(template_group)

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
        self._confidence_label = QLabel("最低相似度")
        params_layout.addWidget(self._confidence_label, 1, 0)
        self._confidence_spin = QDoubleSpinBox()
        self._confidence_spin.setRange(0.0, 1.0)
        self._confidence_spin.setSingleStep(0.05)
        self._confidence_spin.setDecimals(2)
        self._confidence_spin.setValue(SEGMENT_MIN_CONFIDENCE)
        params_layout.addWidget(self._confidence_spin, 1, 1)

        params_layout.addWidget(QLabel("低于最低相似度的声波归入 unknown。"), 2, 0, 1, 4)

        params_layout.addWidget(QLabel("同时分量分离窗"), 3, 0)
        self._separation_window_combo = QComboBox()
        self._separation_window_combo.addItem(
            "固定 256 点（当前）", "fixed"
        )
        self._separation_window_combo.addItem(
            "自适应延长（长分量 512/1024 点）", "adaptive"
        )
        self._separation_window_combo.setToolTip(
            "固定模式保持旧结果；自适应模式对较长叫声延长频谱分析窗，"
            "同时保留至少约 3 个分析窗。"
        )
        params_layout.addWidget(self._separation_window_combo, 3, 1, 1, 3)
        params_layout.addWidget(QLabel("谐波处理"), 4, 0)
        self._harmonic_grouping_combo = QComboBox()
        self._harmonic_grouping_combo.addItem(
            "增强（倍率+轨迹+包络）", "enhanced"
        )
        self._harmonic_grouping_combo.addItem(
            "基础（仅 2/3 倍率）", "basic"
        )
        self._harmonic_grouping_combo.setToolTip(
            "增强模式把可信谐波并入同一分量，不再单独计数；基础模式保留旧判断。"
        )
        params_layout.addWidget(self._harmonic_grouping_combo, 4, 1, 1, 3)
        params_layout.addWidget(QLabel("声谱图输出方式"), 5, 0)
        self._pulse_spectrogram_mode_combo = QComboBox()
        self._pulse_spectrogram_mode_combo.addItem("合并显示（同一时间候选共图）", "merged")
        self._pulse_spectrogram_mode_combo.addItem("分开显示（每个独立分量一图）", "separate")
        self._pulse_spectrogram_mode_combo.setToolTip(
            "分开模式使用分量音频，谐波随所属叫声一起显示。"
            "未可靠分离的候选保留原图并注明；两种方式的计数和分类一致。"
        )
        params_layout.addWidget(self._pulse_spectrogram_mode_combo, 5, 1, 1, 3)
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
        self._figure_button = QPushButton("查看片段线性声谱图")
        self._figure_button.setEnabled(False)
        self._figure_button.clicked.connect(self._open_figure)
        detail_layout.addWidget(self._figure_button)
        self._pulse_figure_button = QPushButton(
            "查看逐声波声谱图（按类别）"
        )
        self._pulse_figure_button.setEnabled(False)
        self._pulse_figure_button.setToolTip(
            "查看当前 WAV 中每个类别的独立声谱图，也包含 unknown。"
        )
        self._pulse_figure_button.clicked.connect(self._open_pulse_figures)
        detail_layout.addWidget(self._pulse_figure_button)
        self._direct_button = QPushButton("直接打开 WAV/SON 查看…")
        self._direct_button.setToolTip(
            "不跑分析，直接渲染任意 WAV 或 Avisoto SASLab 的 .son 声谱图"
        )
        self._direct_button.clicked.connect(self._open_direct_file)
        detail_layout.addWidget(self._direct_button)
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

    def _pick_template_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "选择模板声波文件夹")
        if not chosen:
            return
        path = Path(chosen)
        try:
            from ..template_matching import discover_template_groups

            groups = discover_template_groups(path)
            self._template_dir = path
            self._template_edit.setText(str(path))
            details = "；".join(
                f"{label}（{len(files)}）"
                for label, files in groups.items()
            )
            self._template_status.setText(
                f"识别到 {len(groups)} 个模板类别、"
                f"{sum(len(files) for files in groups.values())} 个 WAV：{details}"
            )
            self._log(f"模板目录已选择：{path}（{len(groups)} 类）")
        except Exception as error:  # noqa: BLE001
            self._template_dir = None
            self._template_edit.clear()
            self._template_status.setText(f"模板读取失败：{error}")
            QMessageBox.warning(self, "模板读取失败", str(error))

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
            "save_pulse_audio": True,
            "save_figures": True,
            "save_pulse_spectrograms": True,
            "pulse_spectrogram_mode": str(self._pulse_spectrogram_mode_combo.currentData()),
            "separation_window_mode": str(
                self._separation_window_combo.currentData()
            ),
            "harmonic_grouping_mode": str(
                self._harmonic_grouping_combo.currentData()
            ),
            "mixed_purity_threshold": SEGMENT_MIXED_PURITY_THRESHOLD,
            "template_unknown_label": "unknown",
        }

    def _start(self) -> None:
        if self._input_path is None:
            QMessageBox.warning(self, "提示", "请先选择 WAV 或文件夹。")
            return
        if not self._input_path.exists():
            QMessageBox.warning(self, "提示", f"路径不存在：{self._input_path}")
            return

        if self._template_dir is None or not self._template_dir.is_dir():
            QMessageBox.warning(self, "提示", "请先选择模板文件夹。")
            return

        from .workers import SegmentAnalysisWorker

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
        self._pulse_figure_button.setEnabled(False)

        worker = SegmentAnalysisWorker(
            self._input_path,
            self._collect_config(),
            output_root=output_root,
            resume=resume,
            template_dir=self._template_dir,
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
        self._separation_window_combo.setEnabled(not busy)
        self._harmonic_grouping_combo.setEnabled(not busy)
        self._pulse_spectrogram_mode_combo.setEnabled(not busy)

    # ------------------------------------------------------------- 进度回调
    def _on_progress(self, current: int, total: int, current_file: str) -> None:
        self._progress_label.setText(
            f"{current} / {total}\n当前：{current_file}"
        )

    def _on_chunk_progress(self, payload: object) -> None:
        """① 长录音块级进度（§13）：已处理时长/速度/已检测 call/分块。"""
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
            result.get("uncertain_label") or "unknown"
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
        self._pulse_figure_button.setEnabled(
            self._run_dir is not None
            and (self._run_dir / "pulse_predictions.csv").exists()
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
            f"噪声（不计入叫声）：{result.get('noise_count', 0)}",
            f"有效蝙蝠叫声：{result.get('detected_call_count', 0)}",
            f"输出声谱图：{result.get('exported_pulse_spectrogram_count', 0)} 张",
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
        segment = self._segment_rows[row]
        segment_id = segment.get("segment_id", "")
        figure_path = self._run_dir / "figures" / f"{segment_id}.png"
        if not figure_path.exists():
            QMessageBox.information(self, "提示", "该片段没有生成声谱图。")
            return

        # 交互式查看：PNG 上叠加可移动频率光标（读出图时的几何 sidecar）
        from .linear_spectrogram_view import (
            LinearSpectrogramDialog,
            load_spectrogram_meta,
        )

        meta = load_spectrogram_meta(figure_path)
        if meta is not None:
            dialog = LinearSpectrogramDialog(
                figure_path,
                meta,
                title=f"{segment.get('segment_file', '')} | 片段线性声谱图",
                parent=self,
            )
            dialog.exec()
            return

        # 老图（改动前生成）没有 sidecar → 回退静态查看
        self._log(
            f"{figure_path.name} 缺少频率光标几何信息（旧版本生成），"
            "以静态图片方式打开；重新跑一次分析即可获得光标。"
        )
        from .image_viewer import ImagePreviewDialog

        dialog = ImagePreviewDialog(figure_path, self)
        dialog.exec()

    def _open_pulse_figures(self) -> None:
        """打开当前 WAV 的逐 pulse 分类声谱图浏览器。"""
        if self._run_dir is None:
            return
        index = self._result_table.currentIndex()
        row = index.row() if index.isValid() else -1
        if row < 0 or row >= len(self._segment_rows):
            return
        result = self._segment_rows[row]
        from .image_viewer import PulseSpectrogramDialog

        dialog = PulseSpectrogramDialog(
            self._run_dir,
            labels=self._labels,
            unknown_label=self._uncertain_label,
            segment_id=str(result.get("segment_id", "")),
            parent=self,
        )
        dialog.exec()

    def _open_direct_file(self) -> None:
        """① 直接查看：不跑分析，选一个 WAV 或 SON 渲染声谱图并打开。"""
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "选择要查看的音频文件",
            "",
            "音频/声谱文件 (*.wav *.son);;WAV (*.wav);;SASLab SON (*.son)",
        )
        if not chosen:
            return
        source = Path(chosen)
        suffix = source.suffix.lower()
        if suffix not in {".wav", ".son"}:
            QMessageBox.warning(self, "提示", "只支持 .wav 或 .son 文件。")
            return

        from ..visualization import save_direct_overview

        cache_dir = Path(tempfile.gettempdir()) / "bat_classifier_quick_view"
        cache_dir.mkdir(parents=True, exist_ok=True)
        figure_path = cache_dir / f"{source.stem}_direct.png"
        try:
            ok = save_direct_overview(
                source,
                figure_path,
                title=f"{source.name} | 直接查看",
            )
        except Exception as error:  # noqa: BLE001 - 文件损坏/格式不符要给出提示
            QMessageBox.critical(
                self,
                "打开失败",
                f"无法渲染 {source.name}：\n{error}",
            )
            return
        if not ok:
            QMessageBox.warning(self, "提示", "未能从该文件生成声谱图。")
            return

        from .linear_spectrogram_view import (
            LinearSpectrogramDialog,
            load_spectrogram_meta,
        )

        meta = load_spectrogram_meta(figure_path)
        if meta is not None:
            dialog = LinearSpectrogramDialog(
                figure_path,
                meta,
                title=f"{source.name} | 线性声谱图（直接查看）",
                parent=self,
            )
            dialog.exec()
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
            pulse_dir = self._run_dir / "pulse_spectrograms"
            if pulse_dir.is_dir():
                shutil.copytree(
                    pulse_dir,
                    target_dir / "pulse_spectrograms",
                    dirs_exist_ok=True,
                )
                copied.append("pulse_spectrograms/")
            self._log(f"已导出结果到 {target_dir}：{', '.join(copied)}")
        except Exception as error:  # noqa: BLE001
            QMessageBox.critical(self, "导出失败", str(error))

    def _open_run_dir(self) -> None:
        target = self._run_dir
        if target is None or not target.exists():
            QMessageBox.information(self, "提示", "尚未生成结果目录。")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))
