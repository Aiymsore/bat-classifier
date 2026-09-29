"""⑥ 页面的专家审核工作台（左：Cluster 列表 · 中：代表样本 · 右：声谱图与信息 · 下：标注表单）。

复用 image_viewer.ImagePreviewDialog 查看大图；职责与“结果图像”画廊明显不同
（这里按 Cluster/代表样本逐条审核），因此是独立的小型 reviewer widget，
而不是把 ImageGalleryPane 硬塞进审核流程。

科研语义：标注只写专家实际审核过的 pulse；Cluster 纯度只是 review hint，
绝不自动把一个 Cluster 全部赋成同一物种。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .curation_model_view import ModelComparisonPane
from .image_viewer import ImagePreviewDialog

if TYPE_CHECKING:
    from ..curation.pipeline import CurationRun

STATUS_OPTIONS = (
    ("已确认", "reviewed"),
    ("不确定", "uncertain"),
    ("噪声", "noise"),
    ("暂不判断", "pending"),
)
CONFIDENCE_OPTIONS = (
    ("（空）", ""),
    ("High", "high"),
    ("Medium", "medium"),
    ("Low", "low"),
)
STATUS_MARKS = {
    "reviewed": "✓",
    "uncertain": "?",
    "noise": "✕",
    "pending": "·",
}

INFO_FEATURE_KEYS = (
    "duration_ms",
    "peak_freq_khz",
    "start_freq_khz",
    "end_freq_khz",
    "frequency_drop_khz",
    "f05_khz",
    "f95_khz",
    "bandwidth_90_khz",
    "slope_khz_per_ms",
    "track_coverage",
    "max_track_jump_khz",
    "snr_db",
)
INFO_FEATURE_LABELS = {
    "duration_ms": "duration (ms)",
    "peak_freq_khz": "peak freq (kHz)",
    "start_freq_khz": "start freq (kHz)",
    "end_freq_khz": "end freq (kHz)",
    "frequency_drop_khz": "freq drop (kHz)",
    "f05_khz": "f05 (kHz)",
    "f95_khz": "f95 (kHz)",
    "bandwidth_90_khz": "bandwidth90 (kHz)",
    "slope_khz_per_ms": "slope (kHz/ms)",
    "track_coverage": "track coverage",
    "max_track_jump_khz": "max jump (kHz)",
    "snr_db": "SNR (dB)",
}


def _fmt(value: object, digits: int = 2, missing: str = "—") -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return missing
    if number != number:  # NaN
        return missing
    return f"{number:.{digits}f}"


class CurationReviewPane(QWidget):
    """专家审核工作台。由 CurationTab 挂载并提供 log / 加样本回调。"""

    def __init__(
        self,
        log: Callable[[str], None] | None = None,
        add_samples_handler: Callable[[str], None] | None = None,
        run_compare_handler: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._log = log or (lambda _text: None)
        self._add_samples_handler = add_samples_handler
        self._run_compare_handler = run_compare_handler

        self._run: "CurationRun | None" = None
        self._store = None
        self._rep_rows: list[dict] = []
        self._loading_form = False

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        top_row = QHBoxLayout()
        self._summary_label = QLabel("尚未打开整理任务。")
        self._summary_label.setWordWrap(True)
        top_row.addWidget(self._summary_label, stretch=1)
        root.addLayout(top_row)

        splitter = QSplitter(Qt.Horizontal)

        cluster_group = QGroupBox("Cluster 列表")
        cluster_layout = QVBoxLayout()
        self._cluster_list = QListWidget()
        self._cluster_list.currentRowChanged.connect(self._on_cluster_changed)
        cluster_layout.addWidget(self._cluster_list)
        cluster_group.setLayout(cluster_layout)
        splitter.addWidget(cluster_group)

        rep_group = QGroupBox("Representative 样本列表")
        rep_layout = QVBoxLayout()
        self._rep_list = QListWidget()
        self._rep_list.currentRowChanged.connect(self._on_rep_changed)
        rep_layout.addWidget(self._rep_list)
        rep_group.setLayout(rep_layout)
        splitter.addWidget(rep_group)

        detail_group = QGroupBox("声谱图 · 声学参数 · 来源信息")
        detail_layout = QVBoxLayout()
        self._spectrogram_label = QLabel("尚未选择样本。")
        self._spectrogram_label.setAlignment(Qt.AlignCenter)
        self._spectrogram_label.setMinimumSize(360, 220)
        self._spectrogram_label.setStyleSheet(
            "background-color: #202020; color: #c0c0c0;"
        )
        self._spectrogram_scroll = QScrollArea()
        self._spectrogram_scroll.setWidget(self._spectrogram_label)
        self._spectrogram_scroll.setWidgetResizable(True)

        detail_buttons = QHBoxLayout()
        large_button = QPushButton("查看大图")
        folder_button = QPushButton("打开样本文件夹")
        large_button.clicked.connect(self._open_large_spectrogram)
        folder_button.clicked.connect(self._open_sample_folder)
        detail_buttons.addStretch(1)
        detail_buttons.addWidget(large_button)
        detail_buttons.addWidget(folder_button)

        self._info_view = QPlainTextEdit()
        self._info_view.setReadOnly(True)
        self._info_view.setMaximumHeight(170)

        detail_layout.addWidget(self._spectrogram_scroll, stretch=1)
        detail_layout.addLayout(detail_buttons)
        detail_layout.addWidget(self._info_view)
        detail_group.setLayout(detail_layout)
        splitter.addWidget(detail_group)

        splitter.setSizes([210, 330, 470])
        root.addWidget(splitter, stretch=1)

        # ---- 纯度提示 ----
        purity_group = QGroupBox("Cluster 标注统计（review hint，不是自动结论）")
        purity_layout = QGridLayout()
        self._purity_label = QLabel("尚未选择 Cluster。")
        self._purity_label.setWordWrap(True)
        self._add_samples_button = QPushButton("增加 10 个代表样本")
        self._add_samples_button.setEnabled(False)
        self._add_samples_button.clicked.connect(self._request_add_samples)
        purity_layout.addWidget(self._purity_label, 0, 0, 1, 2)
        purity_layout.addWidget(self._add_samples_button, 1, 1)
        purity_group.setLayout(purity_layout)
        root.addWidget(purity_group)

        # ---- 模型对照结果（模型辅助判断，绝不覆盖专家标注） ----
        self._comparison = ModelComparisonPane(log=self._log)
        self._comparison.compare_requested.connect(self._request_model_compare)
        self._comparison.sample_activated.connect(self._open_model_sample)
        root.addWidget(self._comparison)

        # ---- 标注表单 ----
        form_group = QGroupBox("专家标注")
        form_layout = QGridLayout()
        form_layout.addWidget(QLabel("物种标签"), 0, 0)
        self._label_edit = QLineEdit()
        self._label_edit.setPlaceholderText("例如 Pipistrellus abramus")
        form_layout.addWidget(self._label_edit, 0, 1)
        form_layout.addWidget(QLabel("置信度"), 0, 2)
        self._confidence_combo = QComboBox()
        for text, _value in CONFIDENCE_OPTIONS:
            self._confidence_combo.addItem(text)
        form_layout.addWidget(self._confidence_combo, 0, 3)
        form_layout.addWidget(QLabel("状态"), 0, 4)
        self._status_combo = QComboBox()
        for text, _value in STATUS_OPTIONS:
            self._status_combo.addItem(text)
        form_layout.addWidget(self._status_combo, 0, 5)

        form_layout.addWidget(QLabel("备注"), 1, 0)
        self._notes_edit = QLineEdit()
        form_layout.addWidget(self._notes_edit, 1, 1, 1, 3)

        button_row = QHBoxLayout()
        self._prev_button = QPushButton("上一条")
        self._save_button = QPushButton("保存")
        self._save_next_button = QPushButton("保存并下一条")
        self._next_pending_button = QPushButton("下一个待审核")
        self._prev_button.clicked.connect(self._go_prev)
        self._save_button.clicked.connect(lambda: self._save(advance=False))
        self._save_next_button.clicked.connect(lambda: self._save(advance=True))
        self._next_pending_button.clicked.connect(self._go_next_pending)
        for button in (
            self._prev_button,
            self._save_button,
            self._save_next_button,
            self._next_pending_button,
        ):
            button_row.addWidget(button)
        self._form_hint = QLabel("")
        self._form_hint.setWordWrap(True)
        button_row.addWidget(self._form_hint, stretch=1)
        form_layout.addLayout(button_row, 2, 0, 1, 6)

        form_group.setLayout(form_layout)
        root.addWidget(form_group)

    # ------------------------------------------------------------ 数据加载
    def is_loaded(self) -> bool:
        return self._run is not None

    def current_cluster(self) -> str | None:
        item = self._cluster_list.currentItem()
        return item.data(Qt.UserRole) if item is not None else None

    def load_run(self, run: "CurationRun") -> dict:
        """读取 manifest/representatives/expert_annotations 并恢复审核位置。"""
        from ..curation.annotations import AnnotationStore
        from ..curation.pipeline import read_table

        representatives = read_table(
            run.tables_dir / "representatives.csv",
            key_column="pulse_id",
        )
        self._run = run
        self._comparison.clear()
        self._rep_rows = (
            representatives.to_dict("records") if not representatives.empty else []
        )
        self._store = AnnotationStore.load(run.run_dir)
        if self._rep_rows:
            import pandas as pd

            self._store.initialize_from_representatives(
                pd.DataFrame(self._rep_rows),
                save=True,
            )
        else:
            self._store.initialize_from_representatives(None, save=False)

        summary = run.review_summary()
        self._refresh_summary(summary)
        self._refresh_cluster_list(select_first=True)
        self._add_samples_button.setEnabled(bool(self._rep_rows))
        if not self._rep_rows:
            self._rep_list.clear()
            self._spectrogram_label.setText("该任务还没有代表样本（请先完成聚类）。")
            self._info_view.clear()
            self._purity_label.setText("尚未选择 Cluster。")
            self._log(
                f"已打开任务 {summary['run_id']}：尚无代表样本。"
            )
            return summary

        self._log(
            f"已打开任务 {summary['run_id']}："
            f"总代表样本 {summary['representative_count']}，"
            f"已审核 {summary['reviewed_count']}，"
            f"待审核 {summary['pending_count']}。"
        )
        return summary

    def reload(self) -> None:
        """追加样本或外部修改后刷新（保持当前 Cluster 选择）。"""
        if self._run is None:
            return
        from ..curation.annotations import AnnotationStore
        from ..curation.pipeline import CurationRun, read_table

        run = CurationRun.open(self._run.run_dir)
        representatives = read_table(
            run.tables_dir / "representatives.csv",
            key_column="pulse_id",
        )
        self._run = run
        self._comparison.clear()
        self._rep_rows = (
            representatives.to_dict("records") if not representatives.empty else []
        )
        self._store = AnnotationStore.load(run.run_dir)
        if self._rep_rows:
            import pandas as pd

            self._store.initialize_from_representatives(
                pd.DataFrame(self._rep_rows),
                save=True,
            )
        keep_cluster = self.current_cluster()
        self._refresh_summary(run.review_summary())
        self._refresh_cluster_list(preferred=keep_cluster)
        self._log("代表样本已刷新。")

    def _refresh_summary(self, summary: dict) -> None:
        processing = summary.get("processing", {})
        self._summary_label.setText(
            f"任务 {summary.get('run_id', '')} ｜ {summary.get('run_dir', '')}\n"
            f"总代表样本：{summary.get('representative_count', 0)} ｜ "
            f"已审核：{summary.get('reviewed_count', 0)} ｜ "
            f"待审核：{summary.get('pending_count', 0)} ｜ "
            f"不确定：{summary.get('uncertain_count', 0)} ｜ "
            f"噪声：{summary.get('noise_count', 0)} ｜ "
            f"录音 {processing.get('total_files', 0)} 个"
            f"（完成 {processing.get('completed_files', 0)}，"
            f"重复 {processing.get('duplicate_files', 0)}，"
            f"失败 {processing.get('failed_files', 0)}）"
        )

    def _cluster_ids(self) -> list[str]:
        return sorted(
            {str(row.get("cluster_id", "")) for row in self._rep_rows}
        )

    def _refresh_cluster_list(
        self,
        select_first: bool = False,
        preferred: str | None = None,
    ) -> None:
        current = preferred or self.current_cluster()
        self._cluster_list.blockSignals(True)
        self._cluster_list.clear()
        for cluster_id in self._cluster_ids():
            stats = (
                self._store.cluster_stats(cluster_id)
                if self._store is not None
                else {}
            )
            total = sum(
                1
                for row in self._rep_rows
                if str(row.get("cluster_id", "")) == cluster_id
            )
            pending = int(stats.get("pending_count", 0)) if stats else total
            item = QListWidgetItem(
                f"Cluster {cluster_id}（待审核 {pending} / {total}）"
            )
            item.setData(Qt.UserRole, cluster_id)
            self._cluster_list.addItem(item)
        self._cluster_list.blockSignals(False)

        target_row = 0
        if current is not None:
            for row in range(self._cluster_list.count()):
                if self._cluster_list.item(row).data(Qt.UserRole) == current:
                    target_row = row
                    break
        if self._cluster_list.count() and (select_first or current is not None):
            self._cluster_list.setCurrentRow(target_row)
        elif not self._cluster_list.count():
            self._rep_list.clear()

    def _rows_for_cluster(self, cluster_id: str | None) -> list[dict]:
        if cluster_id is None:
            return []
        return [
            row
            for row in self._rep_rows
            if str(row.get("cluster_id", "")) == str(cluster_id)
        ]

    def _on_cluster_changed(self, _row: int) -> None:
        cluster_id = self.current_cluster()
        rows = self._rows_for_cluster(cluster_id)
        self._rep_list.blockSignals(True)
        self._rep_list.clear()
        for row in rows:
            pulse_id = str(row.get("pulse_id", ""))
            status = self._status_of(pulse_id)
            mark = STATUS_MARKS.get(status, "·")
            reason = str(row.get("selection_reason", ""))
            item = QListWidgetItem(f"{mark} {pulse_id}  [{reason}]")
            item.setData(Qt.UserRole, pulse_id)
            self._rep_list.addItem(item)
        self._rep_list.blockSignals(False)

        target = 0
        for index, row in enumerate(rows):
            if self._status_of(str(row.get("pulse_id", ""))) == "pending":
                target = index
                break
        if rows:
            self._rep_list.setCurrentRow(target)
        else:
            self._spectrogram_label.setText("该 Cluster 没有代表样本。")
            self._info_view.clear()
        self._refresh_purity(cluster_id)
        self._comparison.show_cluster(cluster_id)

    def _status_of(self, pulse_id: str) -> str:
        if self._store is None:
            return "pending"
        row = self._store_row(pulse_id)
        return str(row.get("review_status", "pending")) if row else "pending"

    def _store_row(self, pulse_id: str) -> dict | None:
        if self._store is None:
            return None
        frame = self._store.to_frame()
        if frame.empty:
            return None
        matched = frame[frame["pulse_id"].astype(str) == str(pulse_id)]
        if matched.empty:
            return None
        return matched.iloc[0].to_dict()

    # ------------------------------------------------------------ 样本详情
    def _current_rep(self) -> dict | None:
        row = self._rep_list.currentRow()
        rows = self._rows_for_cluster(self.current_cluster())
        if 0 <= row < len(rows):
            return rows[row]
        return None

    def _on_rep_changed(self, _row: int) -> None:
        rep = self._current_rep()
        if rep is None or self._run is None:
            return

        self._show_spectrogram(rep)
        self._info_view.setPlainText(self._build_info_text(rep))
        self._fill_form(rep)

    def _show_spectrogram(self, rep: dict) -> None:
        assert self._run is not None
        from ..curation.pipeline import resolve_run_path

        text = str(rep.get("spectrogram_file", "") or "")
        path = resolve_run_path(text, self._run.run_dir) if text else None
        if path is None:
            self._spectrogram_label.setText(
                f"没有声谱图：{rep.get('pulse_id', '')}"
            )
            self._spectrogram_label.setPixmap(QPixmap())
            return
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._spectrogram_label.setText(f"无法加载图像：{path.name}")
            return
        scaled = pixmap.scaled(
            self._spectrogram_scroll.viewport().size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._spectrogram_label.setPixmap(scaled)
        self._spectrogram_label.setText("")

    def _build_info_text(self, rep: dict) -> str:
        lines = [
            f"Cluster：{rep.get('cluster_label', '')}",
            f"Pulse ID：{rep.get('pulse_id', '')}",
            f"selection_reason：{rep.get('selection_reason', '')}",
            (
                f"距质心 { _fmt(rep.get('distance_to_center'), 3) } ｜ "
                f"第二近质心 { _fmt(rep.get('second_center_distance'), 3) } ｜ "
                f"boundary margin { _fmt(rep.get('boundary_margin'), 3) }"
            ),
            "",
            "声学参数：",
        ]
        for key in INFO_FEATURE_KEYS:
            if key in rep:
                digits = 4 if key == "track_coverage" else 2
                lines.append(
                    f"  {INFO_FEATURE_LABELS.get(key, key)}："
                    f"{_fmt(rep.get(key), digits)}"
                )
        lines.extend(
            [
                "",
                "来源信息：",
                f"  source_file：{rep.get('source_file', '')}",
                f"  source_path：{rep.get('source_path', '')}",
                (
                    f"  位置：{ _fmt(rep.get('start_s'), 3) } s → "
                    f"{ _fmt(rep.get('end_s'), 3) } s"
                ),
                f"  audio_hash：{str(rep.get('audio_hash', ''))[:16]}…",
                f"  pulse_file：{rep.get('pulse_file', '') or '（未切盘，按需重切）'}",
            ]
        )
        return "\n".join(lines)

    # ------------------------------------------------------------ 表单
    def _fill_form(self, rep: dict) -> None:
        self._loading_form = True
        try:
            stored = self._store_row(str(rep.get("pulse_id", ""))) or {}
            self._label_edit.setText(str(stored.get("expert_label", "") or ""))
            self._notes_edit.setText(str(stored.get("notes", "") or ""))

            confidence = str(stored.get("expert_confidence", "") or "")
            confidence_index = next(
                (
                    index
                    for index, (_text, value) in enumerate(CONFIDENCE_OPTIONS)
                    if value == confidence
                ),
                0,
            )
            self._confidence_combo.setCurrentIndex(confidence_index)

            status = str(stored.get("review_status", "pending") or "pending")
            status_index = next(
                (
                    index
                    for index, (_text, value) in enumerate(STATUS_OPTIONS)
                    if value == status
                ),
                len(STATUS_OPTIONS) - 1,
            )
            self._status_combo.setCurrentIndex(status_index)
            reviewed_at = str(stored.get("reviewed_at", "") or "")
            self._form_hint.setText(
                f"上次保存：{reviewed_at}" if reviewed_at else ""
            )
        finally:
            self._loading_form = False

    def _collect_form(self) -> tuple[str, str, str, str]:
        label = self._label_edit.text().strip()
        confidence = CONFIDENCE_OPTIONS[self._confidence_combo.currentIndex()][1]
        status = STATUS_OPTIONS[self._status_combo.currentIndex()][1]
        notes = self._notes_edit.text().strip()
        if status == "reviewed" and not label:
            raise ValueError("状态为“已确认”时必须填写物种标签。")
        return label, confidence, status, notes

    def _save(self, advance: bool) -> bool:
        rep = self._current_rep()
        if rep is None or self._store is None:
            QMessageBox.warning(self, "提示", "请先选择要标注的代表样本。")
            return False
        try:
            label, confidence, status, notes = self._collect_form()
        except ValueError as error:
            QMessageBox.warning(self, "标注不完整", str(error))
            return False

        pulse_id = str(rep.get("pulse_id", ""))
        self._store.update(pulse_id, label, confidence, status, notes)
        self._log(
            f"已保存标注：Cluster {rep.get('cluster_id', '')} "
            f"{pulse_id} → 状态 {status}"
            + (f"，标签 “{label}”" if label else "")
        )
        current_cluster = self.current_cluster()
        self._refresh_cluster_list(preferred=current_cluster)
        self._refresh_purity(current_cluster)
        if self._run is not None:
            self._refresh_summary(self._run.review_summary())

        # 更新中间列表的状态标记（保持当前行）
        current_row = self._rep_list.currentRow()
        rows = self._rows_for_cluster(current_cluster)
        self._rep_list.blockSignals(True)
        for index, row in enumerate(rows):
            mark = STATUS_MARKS.get(
                self._status_of(str(row.get("pulse_id", ""))), "·"
            )
            reason = str(row.get("selection_reason", ""))
            self._rep_list.item(index).setText(
                f"{mark} {row.get('pulse_id', '')}  [{reason}]"
            )
        self._rep_list.blockSignals(False)

        if advance:
            if not self._go_next_pending():
                current = current_row
                if 0 <= current + 1 < len(rows):
                    self._rep_list.setCurrentRow(current + 1)
                else:
                    self._form_hint.setText("本簇代表样本已全部过了一遍。")
                    self._log(
                        f"Cluster {current_cluster} 的代表样本已全部过了一遍。"
                    )
        elif 0 <= current_row < self._rep_list.count():
            self._rep_list.setCurrentRow(current_row)
        return True

    # ------------------------------------------------------------ 导航
    def _go_prev(self) -> None:
        row = self._rep_list.currentRow()
        if row > 0:
            self._rep_list.setCurrentRow(row - 1)

    def _go_next_pending(self) -> bool:
        cluster_id = self.current_cluster()
        rows = self._rows_for_cluster(cluster_id)
        current = self._rep_list.currentRow()
        for index in range(current + 1, len(rows)):
            if self._status_of(str(rows[index].get("pulse_id", ""))) == "pending":
                self._rep_list.setCurrentRow(index)
                return True
        for index in range(0, min(current + 1, len(rows))):
            if self._status_of(str(rows[index].get("pulse_id", ""))) == "pending":
                self._rep_list.setCurrentRow(index)
                self._form_hint.setText("已跳到本簇第一个仍待审核的样本。")
                return True
        self._form_hint.setText("本簇没有待审核样本了。")
        return False

    # ------------------------------------------------------------ 纯度提示
    def _refresh_purity(self, cluster_id: str | None) -> None:
        if cluster_id is None or self._store is None:
            self._purity_label.setText("尚未选择 Cluster。")
            return
        stats = self._store.cluster_stats(cluster_id)
        counts_text = " ｜ ".join(
            f"{label} {count}"
            for label, count in stats.get("label_counts", {}).items()
        ) or "（尚无标注）"
        dominant = stats.get("dominant_label", "")
        fraction = float(stats.get("dominant_fraction", 0.0))
        lines = [
            f"Cluster {cluster_id}：已审核 {stats.get('reviewed_count', 0)}"
            f" ｜ 待审核 {stats.get('pending_count', 0)}"
            f" ｜ 不确定 {stats.get('uncertain_count', 0)}"
            f" ｜ 噪声 {stats.get('noise_count', 0)}",
            f"标注分布：{counts_text}",
        ]
        if dominant:
            lines.append(
                f"主导标签：{dominant}（{fraction:.0%} of 已审核）"
            )
        if stats.get("mixed"):
            lines.append(
                "⚠ 可能为混合声学簇（review hint，不是结论）："
                "建议增加本簇抽样数量复核。"
            )
        self._purity_label.setText("\n".join(lines))

    def _request_add_samples(self) -> None:
        cluster_id = self.current_cluster()
        if not cluster_id:
            QMessageBox.warning(self, "提示", "请先选择 Cluster。")
            return
        if self._add_samples_handler is not None:
            self._add_samples_handler(cluster_id)

    # ------------------------------------------------------------ 模型对照
    def set_comparison_result(self, result: object) -> None:
        """接收 SpeciesCompareWorker 的对照结果并刷新当前 Cluster 显示。"""
        self._comparison.set_result(result if isinstance(result, dict) else None)
        self._comparison.show_cluster(self.current_cluster())

    def _request_model_compare(self) -> None:
        if self._run_compare_handler is not None:
            self._run_compare_handler()
        else:  # pragma: no cover - 未挂载处理器
            self._log("当前页面未挂载模型对照处理器。")

    def _open_model_sample(self, row: object) -> None:
        """双击模型对照样本 → 复用声谱图查看器查看真实 WAV 声谱图。"""
        if not isinstance(row, dict) or self._run is None:
            return
        from ..curation.pipeline import resolve_run_path

        text = str(row.get("spectrogram_file", "") or "")
        path = resolve_run_path(text, self._run.run_dir) if text else None
        if path is None:
            QMessageBox.information(
                self,
                "提示",
                f"样本 {row.get('pulse_id', '')} 没有声谱图（可先在样本列表中选中生成）。",
            )
            return
        ImagePreviewDialog(path, self).exec()

    # ------------------------------------------------------------ 打开图片
    def _open_large_spectrogram(self) -> None:
        rep = self._current_rep()
        if rep is None or self._run is None:
            return
        from ..curation.pipeline import resolve_run_path

        text = str(rep.get("spectrogram_file", "") or "")
        path = resolve_run_path(text, self._run.run_dir) if text else None
        if path is None:
            QMessageBox.information(self, "提示", "当前样本没有声谱图。")
            return
        ImagePreviewDialog(path, self).exec()

    def _open_sample_folder(self) -> None:
        rep = self._current_rep()
        if rep is None or self._run is None:
            return
        from ..curation.pipeline import resolve_run_path

        text = str(rep.get("pulse_file", "") or "")
        path = resolve_run_path(text, self._run.run_dir) if text else None
        target = path.parent if path is not None else self._run.run_dir
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))
