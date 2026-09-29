"""⑥ 页面的「已确认物种模型」管理区域。

职责：训练 / 导入 / 切换 / 查看 / 卸载 专家确认物种模型，
并维护 uncertain 阈值（配置项）与当前模型状态显示。

科研语义：
- 该模型只服务于 ⑥ 的 Cluster representative → 物种判断，
  与 A-D 临时声型（KMeans 伪标签）模型完全分开；
- 未校准模型必须显式显示“未校准模型”，概率只称“模型支持度”；
- 模型结果永不写入 expert_label。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..config import SPECIES_MODEL_DIR
from ..species_model import describe_species_model, model_status_text
from .settings import AppSettings


class TrainingSourceDialog(QDialog):
    """训练新模型：选择带专家确认标签的数据（CSV / ⑥ 任务目录，可多来源）。"""

    def __init__(
        self,
        current_run_dir: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("训练新模型 · 选择专家确认标签数据")
        self.resize(640, 420)
        self._current_run_dir = current_run_dir

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "只允许 review_status == reviewed 且 expert_label 非空的数据参与训练；"
                "cluster_id / A-D 聚类标签绝不能作为训练标签。\n"
                "数据需包含：pulse_id、expert_label、source_file/source_recording "
                "以及模型所需的声学特征列。"
            )
        )

        self._sources = QListWidget()
        layout.addWidget(self._sources, stretch=1)

        buttons = QHBoxLayout()
        add_csv = QPushButton("添加 CSV")
        add_run = QPushButton("添加 ⑥ 任务目录")
        add_current = QPushButton("添加当前任务")
        remove_button = QPushButton("移除选中")
        add_csv.clicked.connect(self._add_csv)
        add_run.clicked.connect(self._add_run)
        add_current.clicked.connect(self._add_current)
        remove_button.clicked.connect(self._remove_selected)
        for button in (add_csv, add_run, add_current, remove_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        form = QGridLayout()
        form.addWidget(QLabel("模型名称"), 0, 0)
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("例如 species_2026_01")
        form.addWidget(self._name_edit, 0, 1)
        form.addWidget(QLabel("概率校准"), 1, 0)
        self._calibration_combo = QComboBox()
        self._calibration_combo.addItem("自动（小样本强制 sigmoid）", "auto")
        self._calibration_combo.addItem("sigmoid（推荐）", "sigmoid")
        self._calibration_combo.addItem("isotonic（仅大样本）", "isotonic")
        form.addWidget(self._calibration_combo, 1, 1)
        layout.addLayout(form)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self._buttons.button(QDialogButtonBox.Ok).setText("开始训练")
        self._buttons.accepted.connect(self._accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        if current_run_dir is not None:
            self._add_current()

    # ---------------- 数据源 ----------------
    def _add_source(self, path: Path) -> None:
        text = str(path)
        for index in range(self._sources.count()):
            if self._sources.item(index).text() == text:
                return
        self._sources.addItem(text)

    def _add_csv(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择专家标注 CSV", "", "表格 (*.csv)"
        )
        for path in paths:
            self._add_source(Path(path))

    def _add_run(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "选择 ⑥ 批量整理任务目录")
        if chosen:
            self._add_source(Path(chosen))

    def _add_current(self) -> None:
        if self._current_run_dir is not None:
            self._add_source(Path(self._current_run_dir))

    def _remove_selected(self) -> None:
        for item in self._sources.selectedItems():
            self._sources.takeItem(self._sources.row(item))

    def _accept(self) -> None:
        if self._sources.count() == 0:
            QMessageBox.warning(self, "提示", "请至少添加一个训练数据来源。")
            return
        self.accept()

    # ---------------- 结果 ----------------
    def sources(self) -> list[str]:
        return [
            self._sources.item(index).text()
            for index in range(self._sources.count())
        ]

    def model_name(self) -> str:
        return self._name_edit.text().strip()

    def calibration_method(self) -> str:
        return str(self._calibration_combo.currentData())


class SpeciesModelPanel(QGroupBox):
    """已确认物种模型区域（训练 / 导入 / 校准 / 切换 / 查看 / 卸载）。"""

    model_changed = Signal()

    def __init__(
        self,
        log: Callable[[str], None],
        track_worker: Callable[[object], None],
        settings: AppSettings,
        get_run_dir: Callable[[], Path | None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("已确认物种模型（专家标签监督模型，服务于模型对照）", parent)
        self._log = log
        self._track_worker = track_worker
        self._settings = settings
        self._get_run_dir = get_run_dir
        self._payload: dict | None = None
        self._model_dir: str = ""

        self._build_ui()
        self._restore_from_settings()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._status_label = QLabel("当前模型：未加载")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        buttons = QHBoxLayout()
        self._train_button = QPushButton("训练新模型")
        self._import_button = QPushButton("导入现成模型")
        self._calibrate_button = QPushButton("使用标注数据校准此模型")
        self._switch_button = QPushButton("切换模型")
        self._info_button = QPushButton("查看模型信息")
        self._unload_button = QPushButton("卸载模型")
        self._train_button.clicked.connect(self._train)
        self._import_button.clicked.connect(self._import)
        self._calibrate_button.clicked.connect(self._calibrate)
        self._switch_button.clicked.connect(self._switch)
        self._info_button.clicked.connect(self._show_info)
        self._unload_button.clicked.connect(self._unload)
        for button in (
            self._train_button,
            self._import_button,
            self._calibrate_button,
            self._switch_button,
            self._info_button,
            self._unload_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("uncertain 阈值"))
        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0.05, 0.99)
        self._threshold_spin.setSingleStep(0.01)
        self._threshold_spin.setDecimals(2)
        self._threshold_spin.setValue(float(self._settings.species_uncertain_threshold))
        self._threshold_spin.valueChanged.connect(self._on_threshold_changed)
        threshold_row.addWidget(self._threshold_spin)
        threshold_row.addWidget(
            QLabel("最大校准概率低于该值 → predicted_label = uncertain（不硬分）")
        )
        threshold_row.addStretch(1)
        layout.addLayout(threshold_row)

    # ------------------------------------------------------------ 状态显示
    def _refresh_status(self) -> None:
        if self._payload is None:
            if self._model_dir:
                self._status_label.setText(
                    f"当前模型：未加载\n模型不可用：{self._model_dir}"
                    "（路径不存在或无法读取，可能已被移动或删除）"
                )
            else:
                self._status_label.setText("当前模型：未加载")
        else:
            description = describe_species_model(self._payload, path=self._model_dir)
            self._status_label.setText(model_status_text(description))
        loaded = self._payload is not None
        self._calibrate_button.setEnabled(
            loaded and not bool(self._payload.get("is_calibrated"))
        )
        self._info_button.setEnabled(loaded)
        self._unload_button.setEnabled(loaded)

    def _restore_from_settings(self) -> None:
        saved = str(self._settings.species_model_path or "")
        if not saved:
            self._refresh_status()
            return
        if not Path(saved).exists():
            self._payload = None
            self._model_dir = saved
            self._status_label.setText(
                f"当前模型：未加载\n模型不可用：{saved}（路径不存在，可能已被移动或删除）"
            )
            self._log(f"已确认物种模型不可用：{saved}")
            self._refresh_status()
            return
        self._start_worker("load", model_path=saved)

    # ------------------------------------------------------------ 数据
    def model_payload(self) -> dict | None:
        return self._payload

    def uncertain_threshold(self) -> float:
        return float(self._threshold_spin.value())

    def model_name(self) -> str:
        description = describe_species_model(self._payload, path=self._model_dir)
        return str(description.get("name") or "") or Path(self._model_dir or "").name

    def _current_run_dir(self) -> Path | None:
        try:
            return self._get_run_dir() if self._get_run_dir else None
        except Exception:  # noqa: BLE001 - UI 回调兜底
            return None

    # ------------------------------------------------------------ 操作
    def _train(self) -> None:
        dialog = TrainingSourceDialog(self._current_run_dir(), self)
        if dialog.exec() != QDialog.Accepted:
            return
        name = dialog.model_name()
        if not name:
            QMessageBox.warning(self, "提示", "请填写模型名称。")
            return
        # 已冻结的专家数据划分存在时自动走“冻结 Evaluation Set”训练路径。
        expert_split = ""
        pseudo_source = ""
        for source in dialog.sources():
            split_path = Path(source) / "tables" / "expert_split.json"
            if split_path.exists():
                expert_split = str(split_path)
                pseudo_path = Path(source) / "tables" / "pseudo_training_dataset.csv"
                if pseudo_path.exists():
                    pseudo_source = str(pseudo_path)
                break
        self._start_worker(
            "train",
            sources=dialog.sources(),
            model_name=name,
            calibration_method=dialog.calibration_method(),
            expert_split=expert_split,
            pseudo_source=pseudo_source,
        )

    def _import(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self, "导入现成监督模型", "", "模型文件 (*.joblib)"
        )
        if not chosen:
            return
        self._start_worker("import", model_path=chosen)

    def _calibrate(self) -> None:
        if self._payload is None:
            return
        dialog = TrainingSourceDialog(self._current_run_dir(), self)
        dialog.setWindowTitle("使用标注数据校准此模型（需真实专家标签）")
        if dialog.exec() != QDialog.Accepted:
            return
        self._start_worker(
            "calibrate",
            sources=dialog.sources(),
            model_path=self._model_dir,
            calibration_method=dialog.calibration_method(),
        )

    def _switch(self) -> None:
        candidates: list[str] = []
        if SPECIES_MODEL_DIR.exists():
            candidates = sorted(
                str(item)
                for item in SPECIES_MODEL_DIR.iterdir()
                if item.is_dir()
            )
        chosen: str | None = None
        if candidates:
            names = [Path(item).name for item in candidates]
            name, ok = QInputDialog.getItem(
                self, "切换模型", "选择已保存的物种模型", names, 0, False
            )
            if ok and name:
                chosen = str(SPECIES_MODEL_DIR / name)
        if chosen is None:
            picked, _ = QFileDialog.getOpenFileName(
                self, "选择模型文件", str(SPECIES_MODEL_DIR), "模型文件 (*.joblib)"
            )
            chosen = picked or None
        if not chosen:
            return
        self._start_worker("load", model_path=chosen)

    def _show_info(self) -> None:
        if self._payload is None:
            return
        import json

        metadata = self._payload.get("training_metadata") or {}
        evaluation = self._payload.get("evaluation_metrics") or {}
        text = (
            f"{model_status_text(describe_species_model(self._payload, self._model_dir))}\n\n"
            f"【training_metadata JSON】\n"
            f"{json.dumps(metadata, ensure_ascii=False, indent=2, default=str)}\n\n"
            f"【evaluation JSON】\n"
            f"{json.dumps(evaluation, ensure_ascii=False, indent=2, default=str)}"
        )
        dialog = QDialog(self)
        dialog.setWindowTitle("已确认物种模型 · 模型信息")
        dialog.resize(720, 560)
        layout = QVBoxLayout(dialog)
        from PySide6.QtWidgets import QPlainTextEdit

        view = QPlainTextEdit()
        view.setReadOnly(True)
        view.setPlainText(text)
        layout.addWidget(view)
        close_box = QDialogButtonBox(QDialogButtonBox.Close)
        close_box.rejected.connect(dialog.reject)
        layout.addWidget(close_box)
        dialog.exec()

    def _unload(self) -> None:
        self._payload = None
        self._model_dir = ""
        self._settings.species_model_path = ""
        self._settings.save()
        self._refresh_status()
        self._log("已卸载已确认物种模型。")
        self.model_changed.emit()

    def _on_threshold_changed(self, value: float) -> None:
        self._settings.species_uncertain_threshold = float(value)
        self._settings.save()

    # ------------------------------------------------------------ Worker
    def _start_worker(
        self,
        action: str,
        sources: list[str] | None = None,
        model_name: str = "",
        model_path: str = "",
        calibration_method: str = "sigmoid",
        expert_split: str = "",
        pseudo_source: str = "",
    ) -> None:
        from .workers import SpeciesModelWorker

        worker = SpeciesModelWorker(
            action,
            sources=sources,
            model_name=model_name,
            model_path=model_path,
            calibration_method=calibration_method,
            expert_split=expert_split,
            pseudo_source=pseudo_source,
        )
        worker.done.connect(self._on_done)
        worker.failed.connect(self._on_failed)
        self._track_worker(worker)
        self._set_busy(True)

    def _set_busy(self, busy: bool) -> None:
        for button in (
            self._train_button,
            self._import_button,
            self._calibrate_button,
            self._switch_button,
            self._unload_button,
        ):
            button.setEnabled(not busy)
        if not busy:
            self._refresh_status()

    def _on_done(self, result: object) -> None:
        self._set_busy(False)
        if not isinstance(result, dict):
            return
        payload = result.get("payload")
        model_dir = str(result.get("model_dir", ""))
        if not isinstance(payload, dict):
            self._log("物种模型操作未返回模型对象。")
            return
        self._payload = payload
        self._model_dir = model_dir
        self._settings.species_model_path = model_dir
        self._settings.save()
        self._refresh_status()

        action = str(result.get("action", ""))
        messages = {
            "train": f"已确认物种模型训练完成：{model_dir}",
            "import": f"已导入已确认物种模型：{model_dir}",
            "load": f"已切换已确认物种模型：{model_dir}",
            "calibrate": f"已用标注数据完成概率校准：{model_dir}",
        }
        self._log(messages.get(action, f"物种模型已就绪：{model_dir}"))
        for note in result.get("notes", []) or []:
            self._log(f"  · {note}")
        evaluation = result.get("evaluation")
        if isinstance(evaluation, dict):
            labels = [str(item) for item in (payload.get("class_labels") or [])]
            self._log(
                f"  · 训练样本：{int((payload.get('training_metadata') or {}).get('training_sample_count') or 0)}"
                f" ｜ 物种数：{len(labels)}（{'、'.join(labels)}）"
            )
            self._log(
                f"  · 验证集 {int(evaluation.get('sample_count') or 0)} 条 ｜ "
                f"Accuracy {float(evaluation.get('accuracy') or 0):.3f} ｜ "
                f"Macro Precision {float(evaluation.get('macro_precision') or 0):.3f} ｜ "
                f"Macro Recall {float(evaluation.get('macro_recall') or 0):.3f} ｜ "
                f"Macro F1 {float(evaluation.get('macro_f1') or 0):.3f}"
            )
            self._log(
                f"  · Log loss {float(evaluation.get('log_loss') or 0):.3f} ｜ "
                f"Brier score {float(evaluation.get('brier_score') or 0):.3f} ｜ "
                f"ECE {float((evaluation.get('calibration') or {}).get('ece') or 0):.3f}"
            )
            matrix = (evaluation.get("confusion_matrix") or {})
            if matrix.get("matrix"):
                self._log(
                    f"  · 混淆矩阵（行=真实，列=预测，类别 {matrix.get('labels')}）："
                    f"{matrix.get('matrix')}"
                )
        self.model_changed.emit()

    def _on_failed(self, message: str) -> None:
        self._set_busy(False)
        self._log(f"已确认物种模型操作失败：{message}")
        QMessageBox.critical(self, "已确认物种模型", message)
