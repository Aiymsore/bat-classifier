"""⑥ 批量整理与专家标注 页签。

职责划分：
- 本文件：数据集选择（外部路径，不碰 data/raw）、聚类参数、进度/统计、
  已有任务的打开与续跑、导出入口，以及把 CurationWorker 接到 Qt；
- curation_review.py：专家审核工作台（Cluster/代表样本/声谱图/标注表单）；
- curation 包：全部科研算法（可脱离 Qt 测试、CLI、服务器跑批）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..config import CURATION_DIR, CURATION_K_MAX
from .curation_review import CurationReviewPane
from .settings import AppSettings
from .expert_split_panel import ExpertSplitPanel
from .species_model_panel import SpeciesModelPanel
from .ui_utils import make_scrollable


class CurationTab(QWidget):
    """⑥ 页面：扫描/处理进度 + 参数 + 已确认物种模型 + 审核工作台。"""

    def __init__(
        self,
        log: Callable[[str], None],
        track_worker: Callable[[object], None],
        settings: AppSettings | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._log = log
        self._track_worker = track_worker
        self._settings = settings or AppSettings()
        self._input_path: Path | None = None
        self._current_run_dir: Path | None = None
        self._last_progress: tuple[int, int] | None = None
        self._last_stats: dict | None = None

        self._build_ui()
        self._refresh_runs()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # 整页滚动：⑥ 页面全部内容（数据集/参数/进度/导出/划分/物种模型 +
        # 审核工作台 + 模型对照结果 + 标注表单）都进同一个滚动区——
        # 屏幕再矮也能一路滚到底，"模型对照结果"不会被挤出可视区域。
        # 列表/表格等自带滚动条的部件留在各面板内部，外层滚动条按需出现。
        form = QWidget()
        form_layout = QVBoxLayout(form)
        form_layout.setContentsMargins(0, 0, 0, 0)

        # ---- 1. 数据集 ----
        dataset_group = QGroupBox("1. 数据集（独立于①数据来源，直接读外部路径）")
        dataset_layout = QGridLayout()

        self._path_edit = QLineEdit()
        self._path_edit.setReadOnly(True)
        self._path_edit.setPlaceholderText(
            "选择一个 WAV 文件或一个文件夹（例如 D:\\BatMonitoring\\2026）；"
            "不会复制进 data/raw，也不会清空训练数据。"
        )
        folder_button = QPushButton("选择WAV文件夹")
        file_button = QPushButton("选择WAV文件")
        folder_button.clicked.connect(self._pick_input_dir)
        file_button.clicked.connect(self._pick_input_file)
        dataset_layout.addWidget(self._path_edit, 0, 0, 1, 3)
        dataset_layout.addWidget(folder_button, 0, 3)
        dataset_layout.addWidget(file_button, 0, 4)

        self._found_label = QLabel("尚未选择输入。")
        dataset_layout.addWidget(self._found_label, 1, 0, 1, 5)

        dataset_layout.addWidget(QLabel("已有任务"), 2, 0)
        self._run_combo = QComboBox()
        self._run_combo.setMinimumWidth(240)
        refresh_runs_button = QPushButton("刷新")
        open_run_button = QPushButton("打开任务审核")
        resume_button = QPushButton("继续处理")
        refresh_runs_button.clicked.connect(self._refresh_runs)
        open_run_button.clicked.connect(self._open_selected_run)
        resume_button.clicked.connect(self._resume_selected_run)
        dataset_layout.addWidget(self._run_combo, 2, 1)
        dataset_layout.addWidget(refresh_runs_button, 2, 2)
        dataset_layout.addWidget(open_run_button, 2, 3)
        dataset_layout.addWidget(resume_button, 2, 4)
        dataset_group.setLayout(dataset_layout)
        form_layout.addWidget(dataset_group)

        # ---- 2. 聚类参数 ----
        params_group = QGroupBox("2. 聚类参数（Cluster = 声学相似组 ≠ 物种）")
        params_layout = QGridLayout()

        params_layout.addWidget(QLabel("聚类数量"), 0, 0)
        self._k_mode_combo = QComboBox()
        self._k_mode_combo.addItem("自动（Auto）", "auto")
        self._k_mode_combo.addItem("手动", "manual")
        self._k_spin = QSpinBox()
        self._k_spin.setRange(2, CURATION_K_MAX)
        self._k_spin.setValue(4)
        self._k_spin.setEnabled(False)
        self._k_mode_combo.currentIndexChanged.connect(
            lambda index: self._k_spin.setEnabled(
                self._k_mode_combo.currentData() == "manual"
            )
        )
        params_layout.addWidget(self._k_mode_combo, 0, 1)
        params_layout.addWidget(self._k_spin, 0, 2)

        self._center_spin = self._count_spin(5)
        self._boundary_spin = self._count_spin(5)
        self._random_spin = self._count_spin(5)
        self._outlier_spin = self._count_spin(5)
        params_layout.addWidget(QLabel("每类中心样本"), 0, 3)
        params_layout.addWidget(self._center_spin, 0, 4)
        params_layout.addWidget(QLabel("每类边界样本"), 1, 0)
        params_layout.addWidget(self._boundary_spin, 1, 1)
        params_layout.addWidget(QLabel("每类随机样本"), 1, 2)
        params_layout.addWidget(self._random_spin, 1, 3)
        params_layout.addWidget(QLabel("每类离群样本"), 1, 4)
        params_layout.addWidget(self._outlier_spin, 1, 5)

        self._save_audio_check = QCheckBox("切割并保存每个合格脉冲 WAV")
        self._save_audio_check.setChecked(True)
        self._save_rejected_check = QCheckBox("同时保存被拒绝脉冲 WAV（耗磁盘）")
        self._retry_failed_check = QCheckBox("重试失败文件")
        params_layout.addWidget(self._save_audio_check, 2, 0, 1, 2)
        params_layout.addWidget(self._save_rejected_check, 2, 2, 1, 2)
        params_layout.addWidget(self._retry_failed_check, 2, 4, 1, 2)

        self._start_button = QPushButton("开始整理")
        self._start_button.clicked.connect(self._start)
        params_layout.addWidget(self._start_button, 3, 0, 1, 6)
        params_group.setLayout(params_layout)
        form_layout.addWidget(params_group)

        # ---- 进度与统计 ----
        progress_group = QGroupBox("进度")
        progress_layout = QGridLayout()
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._stats_label = QLabel("尚未开始。")
        self._stats_label.setWordWrap(True)
        progress_layout.addWidget(self._progress_bar, 0, 0, 1, 6)
        progress_layout.addWidget(self._stats_label, 1, 0, 1, 6)
        progress_group.setLayout(progress_layout)
        form_layout.addWidget(progress_group)

        # ---- 导出 ----
        export_layout = QHBoxLayout()
        self._export_button = QPushButton("导出专家标注数据")
        self._export_button.clicked.connect(self._export_dataset)
        self._include_noise_check = QCheckBox("导出时包含噪声样本")
        open_run_dir_button = QPushButton("打开任务目录")
        open_run_dir_button.clicked.connect(self._open_run_dir)
        export_layout.addWidget(self._export_button)
        export_layout.addWidget(self._include_noise_check)
        export_layout.addWidget(open_run_dir_button)
        export_layout.addStretch(1)
        form_layout.addLayout(export_layout)

        hint = QLabel(
            "语义提醒：Cluster ≠ 物种 ≠ A-D 声型；最佳 silhouette K 不等于物种数量；"
            "自动聚类只用于减少专家需要人工查看的声波数量，真实物种标签只来自专家人工鉴定。"
        )
        hint.setWordWrap(True)
        form_layout.addWidget(hint)

        # ---- 2.5 专家数据划分（Train Pool / Evaluation Set，冻结考试卷） ----
        self._split_panel = ExpertSplitPanel(
            log=self._log,
            get_run_dir=lambda: self._current_run_dir or self._selected_run_dir(),
            settings=self._settings,
        )
        form_layout.addWidget(self._split_panel)

        # ---- 3. 已确认物种模型（模型对照专用） ----
        self._species_panel = SpeciesModelPanel(
            log=self._log,
            track_worker=self._track_worker,
            settings=self._settings,
            get_run_dir=lambda: self._current_run_dir or self._selected_run_dir(),
        )
        form_layout.addWidget(self._species_panel)

        # ---- 4. 审核工作台 ----
        # 与表单一起进整页滚动区；面板内部的列表/表格/声谱图自带滚动条。
        self._review = CurationReviewPane(
            log=self._log,
            add_samples_handler=self._request_add_samples,
            run_compare_handler=self._run_model_compare,
        )
        self._review.setMinimumHeight(220)
        form_layout.addWidget(self._review, stretch=1)

        form_scroll = make_scrollable(form)
        root.addWidget(form_scroll, stretch=1)

    @staticmethod
    def _count_spin(value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(0, 100)
        spin.setValue(value)
        return spin

    # ------------------------------------------------------------- 数据集
    def _pick_input_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "选择WAV文件夹")
        if chosen:
            self._set_input(Path(chosen))

    def _pick_input_file(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self, "选择WAV", "", "音频 (*.wav)"
        )
        if chosen:
            self._set_input(Path(chosen))

    def _set_input(self, path: Path) -> None:
        self._input_path = path
        self._path_edit.setText(str(path))
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

    # ------------------------------------------------------------- 已有任务
    def _refresh_runs(self) -> None:
        from ..curation.pipeline import CurationRun

        current = self._run_combo.currentData()
        self._run_combo.clear()
        try:
            runs = CurationRun.list_runs()
        except Exception as error:  # noqa: BLE001
            self._log(f"读取任务列表失败：{error}")
            return
        for run_dir in runs:
            self._run_combo.addItem(run_dir.name, str(run_dir))
        if current:
            index = self._run_combo.findData(current)
            if index >= 0:
                self._run_combo.setCurrentIndex(index)
        if not runs:
            self._run_combo.addItem("（暂无任务）", "")

    def _selected_run_dir(self) -> Path | None:
        data = self._run_combo.currentData()
        if not data:
            return None
        path = Path(str(data))
        return path if path.exists() else None

    def _open_selected_run(self) -> None:
        run_dir = self._selected_run_dir()
        if run_dir is None:
            QMessageBox.warning(self, "提示", "请先在“已有任务”中选择一个任务。")
            return
        self._load_run_for_review(run_dir)

    def _load_run_for_review(self, run_dir: Path) -> None:
        from ..curation.pipeline import CurationRun

        try:
            run = CurationRun.open(run_dir)
            summary = self._review.load_run(run)
        except Exception as error:  # noqa: BLE001
            QMessageBox.critical(self, "打开任务失败", str(error))
            return
        self._current_run_dir = Path(run.run_dir)
        self._split_panel.refresh()
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(1)
        processing = summary.get("processing", {})
        self._stats_label.setText(
            f"已载入任务：{summary.get('run_id', '')} ｜ "
            f"录音 {processing.get('total_files', 0)} 个，"
            f"候选脉冲 {processing.get('candidate_pulses', 0)}，"
            f"合格 {processing.get('accepted_pulses', 0)}，"
            f"拒绝 {processing.get('rejected_pulses', 0)}"
        )

    def _resume_selected_run(self) -> None:
        run_dir = self._selected_run_dir()
        if run_dir is None:
            QMessageBox.warning(self, "提示", "请先在“已有任务”中选择一个任务。")
            return
        from .workers import CurationWorker

        params = self._collect_params()
        worker = CurationWorker(
            run_dir=run_dir,
            params=params,
            mode="process",
        )
        self._start_worker(worker, self._on_process_done)
        self._log(f"继续处理任务：{run_dir.name}")

    # ------------------------------------------------------------- 开始整理
    def _collect_params(self) -> dict:
        return {
            "k_mode": str(self._k_mode_combo.currentData()),
            "k": int(self._k_spin.value()),
            "n_center": int(self._center_spin.value()),
            "n_boundary": int(self._boundary_spin.value()),
            "n_random": int(self._random_spin.value()),
            "n_outlier": int(self._outlier_spin.value()),
            "save_pulse_audio": bool(self._save_audio_check.isChecked()),
            "save_rejected_audio": bool(self._save_rejected_check.isChecked()),
            "retry_failed": bool(self._retry_failed_check.isChecked()),
        }

    def _start(self) -> None:
        if self._input_path is None:
            QMessageBox.warning(self, "提示", "请先选择 WAV 文件或文件夹。")
            return
        if not self._input_path.exists():
            QMessageBox.warning(self, "提示", f"路径不存在：{self._input_path}")
            return
        from .workers import CurationWorker

        worker = CurationWorker(
            input_path=self._input_path,
            params=self._collect_params(),
            mode="process",
        )
        self._start_worker(worker, self._on_process_done)
        self._log(f"开始批量整理：{self._input_path}")

    def _request_add_samples(self, cluster_id: str) -> None:
        if self._current_run_dir is None:
            QMessageBox.warning(self, "提示", "请先打开一个整理任务。")
            return
        from .workers import CurationWorker

        worker = CurationWorker(
            run_dir=self._current_run_dir,
            params=self._collect_params(),
            mode="add_samples",
            cluster_id=cluster_id,
        )
        self._start_worker(worker, self._on_add_samples_done)
        self._log(f"为 Cluster {cluster_id} 追加代表样本…")

    def _run_model_compare(self) -> None:
        """对当前任务的四类 representative 运行已确认物种模型对照。"""
        run_dir = self._current_run_dir or self._selected_run_dir()
        if run_dir is None:
            QMessageBox.warning(self, "提示", "请先打开一个整理任务。")
            return
        payload = self._species_panel.model_payload()
        if payload is None:
            QMessageBox.warning(
                self,
                "提示",
                "尚未加载已确认物种模型：请先在“已确认物种模型”区域训练或导入模型。",
            )
            return
        from .workers import SpeciesCompareWorker

        worker = SpeciesCompareWorker(
            run_dir=run_dir,
            payload=payload,
            uncertain_threshold=self._species_panel.uncertain_threshold(),
            model_name=self._species_panel.model_name(),
        )
        worker.done.connect(self._on_model_compare_done)
        worker.failed.connect(self._on_failed)
        self._track_worker(worker)
        self._log(
            f"运行模型对照：{Path(run_dir).name}"
            f"（uncertain 阈值 {self._species_panel.uncertain_threshold():.2f}）"
        )

    def _on_model_compare_done(self, result: object) -> None:
        self._review.set_comparison_result(result)
        if isinstance(result, dict) and result.get("ok"):
            self._log(
                f"模型对照完成：{len(result.get('clusters', []))} 个 Cluster，"
                f"结果表 {result.get('comparison_csv', '')}"
            )
            if result.get("training_data_overlap"):
                self._log(
                    "⚠ TRAINING_DATA_OVERLAP：部分代表样本来自模型训练数据，"
                    "结果不代表泛化能力。"
                )

    # ------------------------------------------------------------- Worker
    def _start_worker(self, worker, on_done: Callable[[object], None]) -> None:
        worker.progress.connect(self._on_progress)
        worker.stats.connect(self._on_stats)
        worker.done.connect(on_done)
        worker.failed.connect(self._on_failed)
        self._last_progress = None
        self._last_stats = None
        self._stats_label.setText("准备中…")
        self._set_busy(True)
        # _track_worker 负责 log 信号、任务登记与 start()。
        self._track_worker(worker)

    def _set_busy(self, busy: bool) -> None:
        self._start_button.setEnabled(not busy)
        self._export_button.setEnabled(not busy)

    def _on_progress(self, current: int, total: int) -> None:
        self._last_progress = (current, total)
        self._progress_bar.setRange(0, max(total, 1))
        self._progress_bar.setValue(current)
        self._render_progress()

    def _on_stats(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        self._last_stats = payload
        self._render_progress()

    def _render_progress(self) -> None:
        lines: list[str] = []
        if self._last_progress is not None:
            current, total = self._last_progress
            percent = int(current * 100 / total) if total else 100
            lines.append(f"进度：{current:,} / {total:,} WAV（{percent}%）")
        payload = self._last_stats or {}
        if payload:
            current = str(payload.get("current_file", "") or "")
            lines.append(
                f"检测脉冲：{int(payload.get('candidate_pulses', 0)):,} ｜ "
                f"合格：{int(payload.get('accepted_pulses', 0)):,} ｜ "
                f"拒绝：{int(payload.get('rejected_pulses', 0)):,} ｜ "
                f"重复：{int(payload.get('duplicate_files', 0)):,} ｜ "
                f"错误：{int(payload.get('failed_files', 0)):,} ｜ "
                f"当前：{current or '—'}"
            )
        self._stats_label.setText("\n".join(lines) or "尚未开始。")

    def _on_process_done(self, result: object) -> None:
        self._set_busy(False)
        if not isinstance(result, dict):
            return
        run_dir = Path(str(result.get("run_dir", "")))
        stats = result.get("stats", {}) if isinstance(result.get("stats"), dict) else {}
        cluster = (
            result.get("cluster", {}) if isinstance(result.get("cluster"), dict) else {}
        )

        self._refresh_runs()
        index = self._run_combo.findData(str(run_dir))
        if index >= 0:
            self._run_combo.setCurrentIndex(index)

        if cluster.get("ok"):
            self._log(
                f"批量整理完成：{run_dir.name} ｜ "
                f"候选 {int(stats.get('candidate_pulses', 0))}，"
                f"合格 {int(stats.get('accepted_pulses', 0))}，"
                f"拒绝 {int(stats.get('rejected_pulses', 0))}，"
                f"重复 {int(stats.get('duplicate_files', 0))}，"
                f"错误 {int(stats.get('failed_files', 0))} ｜ "
                f"{cluster.get('cluster_count', 0)} 个 Cluster，"
                f"代表样本 {cluster.get('representative_count', 0)} 个。"
            )
        else:
            self._log(
                "处理完成，但聚类未完成："
                f"{cluster.get('error', '未知原因')}"
            )
        self._load_run_for_review(run_dir)

    def _on_add_samples_done(self, result: object) -> None:
        self._set_busy(False)
        if not isinstance(result, dict):
            return
        self._log(
            f"Cluster {result.get('cluster_id', '')} 追加代表样本 "
            f"{int(result.get('added', 0))} 个。"
        )
        self._review.reload()

    def _on_failed(self, message: str) -> None:
        self._set_busy(False)
        self._log(f"批量整理任务失败：{message}")
        QMessageBox.critical(self, "任务失败", message)

    # ------------------------------------------------------------- 导出
    def _export_dataset(self) -> None:
        run_dir = self._current_run_dir or self._selected_run_dir()
        if run_dir is None:
            QMessageBox.warning(self, "提示", "请先打开一个整理任务再导出。")
            return
        from ..curation.pipeline import CurationRun

        try:
            run = CurationRun.open(run_dir)
            path = run.stage_export(
                include_noise=bool(self._include_noise_check.isChecked())
            )
        except Exception as error:  # noqa: BLE001
            QMessageBox.critical(self, "导出失败", str(error))
            return
        self._log(f"专家标注训练数据集已导出：{path}")

    def _open_run_dir(self) -> None:
        run_dir = self._current_run_dir or self._selected_run_dir()
        target = run_dir or CURATION_DIR
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))
