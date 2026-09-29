"""⑥「专家数据划分」子面板：Expert Train Pool / Evaluation Set（冻结考试卷）。

- 生成划分：**先预览再落盘**（总量 / 两侧 / 逐类 / 录音数 / 覆盖警告），
  确认后冻结为 ``tables/expert_split.json``；
- 重新生成划分：强警告"新旧模型指标不再完全可比"，并递增 split_version；
- 查看评估表：Support / Precision / Recall / F1 / Macro-F1，指标带 tooltip，
  评估真值为 0 的物种显示 N/A（绝不输出误导性的 F1 = 0）；
- 新旧模型对比：必须在同一冻结 Evaluation Set 上，split_version 不一致时
  明确警告不可比。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..expert_split import (
    ExpertSplitStore,
    gold_record_from_split,
    make_expert_split,
    split_summary,
)

METRIC_TOOLTIPS = {
    "Support": "Expert Evaluation Set 中专家确认真正属于这个物种的 pulse 数。",
    "Precision": "模型说是这个物种的样本中，有多少是真的。",
    "Recall": "专家确认真正属于这个物种的样本中，模型找出了多少。",
    "F1": "Precision 和 Recall 的综合指标（调和平均）。",
    "Macro-F1": "每个物种的 F1 等权平均（只对 Evaluation 真值 ≥ 1 的物种计算）。",
    "N/A": "无评估真值（Support = 0）：该物种无法计算 Recall / F1，不是模型表现为 0。",
}

_SPECIES_METRIC_FORMAT = "{:.3f}"


def _format_metric(value: object) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return "N/A"
    return _SPECIES_METRIC_FORMAT.format(number)


class _PreviewDialog(QDialog):
    """划分预览：确认后才落盘冻结。"""

    def __init__(self, summary: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("专家数据划分预览（确认后冻结）")
        layout = QVBoxLayout(self)
        info = QLabel(
            f"专家总样本：{summary.get('total', 0)}\n"
            f"Expert Train Pool：{summary.get('train_pulse_count', 0)}\n"
            f"Expert Evaluation Set：{summary.get('evaluation_pulse_count', 0)}\n"
            f"Train recordings：{summary.get('train_recording_count', 0)}\n"
            f"Evaluation recordings：{summary.get('evaluation_recording_count', 0)}\n"
            f"状态：{summary.get('status', '')}"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        table = QTableWidget(0, 3)
        table.setHorizontalHeaderLabels(["Species", "Train", "Evaluation"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        rows = summary.get("per_class") or []
        table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            table.setItem(row_index, 0, QTableWidgetItem(str(row.get("species", ""))))
            table.setItem(
                row_index, 1, QTableWidgetItem(str(row.get("train_count", 0)))
            )
            eval_item = QTableWidgetItem(str(row.get("evaluation_count", 0)))
            if int(row.get("evaluation_count", 0)) == 0:
                eval_item.setToolTip(METRIC_TOOLTIPS["N/A"])
            table.setItem(row_index, 2, eval_item)
        layout.addWidget(table)

        warnings = []
        for label in summary.get("insufficient_classes") or []:
            warnings.append(
                f"INSUFFICIENT_EVAL_COVERAGE：{label} 的专家 recording 数量不足，"
                "无法建立独立 Evaluation Set（该物种无法计算 Recall / F1）。"
            )
        for label in summary.get("uncovered_classes") or []:
            warnings.append(f"该物种当前无法计算 Recall / F1：{label}（Evaluation 真值 = 0）")
        if warnings:
            warning_label = QLabel("\n".join(warnings))
            warning_label.setWordWrap(True)
            warning_label.setStyleSheet("color: #b00;")
            layout.addWidget(warning_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self
        )
        buttons.button(QDialogButtonBox.Ok).setText("保存并冻结")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class _EvaluationTableDialog(QDialog):
    """评估表：Support / Precision / Recall / F1（带 tooltip）。"""

    def __init__(
        self,
        rows: list[dict],
        macro_f1: object,
        macro_classes: list[str],
        macro_excluded: list[str],
        title: str = "评估表（Expert Evaluation Set，冻结）",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        summary = QLabel(
            f"Macro-F1：{_format_metric(macro_f1)}"
            f"（对 {'、'.join(macro_classes) or '—'} 等权平均）"
        )
        if macro_excluded:
            summary.setText(
                summary.text()
                + f"\n已排除（Evaluation 真值 = 0）：{'、'.join(macro_excluded)}"
                "——这些物种无法计算 Recall / F1，不是模型表现为 0。"
            )
        summary.setWordWrap(True)
        summary.setToolTip(METRIC_TOOLTIPS["Macro-F1"])
        layout.addWidget(summary)

        table = QTableWidget(0, 6)
        headers = ["Species", "Support", "Precision", "Recall", "F1", "状态"]
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        for column, header in enumerate(headers):
            item = table.horizontalHeaderItem(column)
            if item is not None and header in METRIC_TOOLTIPS:
                item.setToolTip(METRIC_TOOLTIPS[header])
        table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            table.setItem(row_index, 0, QTableWidgetItem(str(row.get("species", ""))))
            support_item = QTableWidgetItem(str(row.get("support", 0)))
            support_item.setToolTip(METRIC_TOOLTIPS["Support"])
            table.setItem(row_index, 1, support_item)
            for column, key in ((2, "precision"), (3, "recall"), (4, "f1")):
                value = row.get(key)
                item = QTableWidgetItem(_format_metric(value))
                if value is None:
                    item.setToolTip(METRIC_TOOLTIPS["N/A"])
                else:
                    item.setToolTip(METRIC_TOOLTIPS[headers[column]])
                table.setItem(row_index, column, item)
            table.setItem(
                row_index, 5, QTableWidgetItem(str(row.get("status", "") or "—"))
            )
        layout.addWidget(table)
        buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class ExpertSplitPanel(QGroupBox):
    """专家数据划分（Train Pool / Evaluation Set）+ 评估表 / 新旧模型对比。"""

    def __init__(
        self,
        log: Callable[[str], None],
        get_run_dir: Callable[[], Path | None],
        settings=None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            "专家数据划分（Expert Train Pool / Evaluation Set，按 source_recording 整体拆分）",
            parent,
        )
        self._log = log
        self._get_run_dir = get_run_dir
        self._settings = settings
        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._status_label = QLabel("尚未生成专家数据划分。")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Species", "Train", "Evaluation"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setMinimumHeight(110)
        layout.addWidget(self._table)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Evaluation 目标比例"))
        self._fraction_spin = QDoubleSpinBox()
        self._fraction_spin.setRange(0.05, 0.50)
        self._fraction_spin.setSingleStep(0.05)
        self._fraction_spin.setDecimals(2)
        self._fraction_spin.setValue(0.20)
        controls.addWidget(self._fraction_spin)
        controls.addWidget(
            QLabel("（比例只是目标：真正拆分单位是 source_recording，绝不拆同一录音的 pulse）")
        )
        controls.addStretch(1)
        layout.addLayout(controls)

        buttons = QHBoxLayout()
        self._generate_button = QPushButton("生成专家数据划分")
        self._regenerate_button = QPushButton("重新生成划分")
        self._refresh_button = QPushButton("刷新")
        self._evaluation_button = QPushButton("查看评估表")
        self._compare_button = QPushButton("新旧模型对比")
        self._generate_button.clicked.connect(self._generate)
        self._regenerate_button.clicked.connect(self._regenerate)
        self._refresh_button.clicked.connect(self.refresh)
        self._evaluation_button.clicked.connect(self._show_evaluation)
        self._compare_button.clicked.connect(self._compare_models)
        for button in (
            self._generate_button,
            self._regenerate_button,
            self._refresh_button,
            self._evaluation_button,
            self._compare_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

    # ---------------------------------------------------------------- 数据
    def _run_dir(self) -> Path | None:
        try:
            return self._get_run_dir() if self._get_run_dir else None
        except Exception:  # noqa: BLE001 - UI 回调兜底
            return None

    def _expert_frame(self) -> pd.DataFrame:
        run_dir = self._run_dir()
        if run_dir is None:
            raise ValueError("请先选择/打开一个 ⑥ 任务。")
        annotations_path = Path(run_dir) / "tables" / "expert_annotations.csv"
        if not annotations_path.exists():
            raise ValueError("该任务还没有专家标注（expert_annotations.csv 不存在）。")
        try:
            frame = pd.read_csv(annotations_path, encoding="utf-8-sig").fillna("")
        except UnicodeDecodeError:
            frame = pd.read_csv(annotations_path).fillna("")
        label = frame.get("expert_label", pd.Series("", index=frame.index)).astype(str)
        status = frame.get("review_status", pd.Series("", index=frame.index)).astype(str)
        data = frame[(status == "reviewed") & (label.str.strip() != "")].copy()
        if data.empty:
            raise ValueError("还没有 reviewed 且带标签的专家标注，无法划分。")
        data["expert_label"] = data["expert_label"].astype(str)
        data["pulse_id"] = data["pulse_id"].astype(str)
        return data

    def refresh(self) -> None:
        run_dir = self._run_dir()
        self._table.setRowCount(0)
        if run_dir is None:
            self._status_label.setText("尚未选择 ⑥ 任务。")
            self._generate_button.setEnabled(False)
            self._regenerate_button.setEnabled(False)
            return
        store = ExpertSplitStore.load(Path(run_dir))
        summary = split_summary(store.record)
        try:
            expert_total = int(len(self._expert_frame()))
        except Exception:  # noqa: BLE001 - 没有标注时只显示划分状态
            expert_total = 0

        if not store.frozen:
            self._status_label.setText(
                f"当前专家标注总数：{expert_total}\n"
                "Evaluation Set：尚未冻结。生成划分后，v1/v2/v3 模型比较"
                "默认使用同一份 Evaluation Set。"
            )
            self._generate_button.setEnabled(expert_total > 0)
            self._regenerate_button.setEnabled(False)
            return

        rows = summary.get("per_class") or []
        self._table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            self._table.setItem(
                row_index, 0, QTableWidgetItem(str(row.get("species", "")))
            )
            self._table.setItem(
                row_index, 1, QTableWidgetItem(str(row.get("train_count", 0)))
            )
            eval_item = QTableWidgetItem(str(row.get("evaluation_count", 0)))
            if int(row.get("evaluation_count", 0)) == 0:
                eval_item.setToolTip(METRIC_TOOLTIPS["N/A"])
            self._table.setItem(row_index, 2, eval_item)

        lines = [
            f"当前专家标注总数：{summary.get('total', expert_total)}",
            f"Expert Train Pool：{summary.get('train_pulse_count', 0)}"
            f"（Train recordings {summary.get('train_recording_count', 0)}）",
            f"Expert Evaluation Set：{summary.get('evaluation_pulse_count', 0)}"
            f"（Evaluation recordings {summary.get('evaluation_recording_count', 0)}）",
            f"当前 Evaluation Set：已冻结（split_version="
            f"{summary.get('split_version', 1)}，{summary.get('created_at', '')}）",
            f"划分状态：{summary.get('status', '')}",
        ]
        for label in summary.get("insufficient_classes") or []:
            lines.append(
                f"INSUFFICIENT_EVAL_COVERAGE：{label} 的专家 recording 数量不足，"
                "无法建立独立 Evaluation Set（绝不拆同一录音来凑 F1）。"
            )
        for row in rows:
            if int(row.get("evaluation_count", 0)) == 0:
                lines.append(f"该物种当前无法计算 Recall / F1：{row.get('species')}")
        self._status_label.setText("\n".join(lines))
        self._generate_button.setEnabled(False)
        self._regenerate_button.setEnabled(True)

    # ------------------------------------------------------------ 生成 / 冻结
    def _generate(self) -> None:
        self._generate_split(replace=False)

    def _regenerate(self) -> None:
        answer = QMessageBox.warning(
            self,
            "重新生成专家数据划分",
            "重新划分后，新旧模型指标将不再与之前 evaluation version 完全可比。\n"
            "（会生成新的 split_version；此前的 v1/v2 对比请以旧版本为准。）\n\n"
            "确定要重新生成划分吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self._generate_split(replace=True)

    def _generate_split(self, replace: bool) -> None:
        try:
            data = self._expert_frame()
            record = make_expert_split(
                data,
                eval_fraction=float(self._fraction_spin.value()),
            )
        except Exception as error:  # noqa: BLE001 - UI 显示全部失败原因
            QMessageBox.critical(self, "专家数据划分失败", str(error))
            self._log(f"专家数据划分失败：{error}")
            return

        summary = split_summary(record)
        dialog = _PreviewDialog(summary, self)
        if dialog.exec() != QDialog.Accepted:
            self._log("已取消专家数据划分（未写入任何文件）。")
            return

        run_dir = self._run_dir()
        store = ExpertSplitStore.load(Path(run_dir))
        try:
            saved = store.save(record, replace=replace)
        except Exception as error:  # noqa: BLE001
            QMessageBox.critical(self, "保存失败", str(error))
            return
        self._log(
            f"专家数据划分已冻结：split_version={saved.get('split_version')}，"
            f"Train {saved.get('train_pulse_count')} 条"
            f"（{saved.get('train_recording_count')} 录音）/ "
            f"Evaluation {saved.get('evaluation_pulse_count')} 条"
            f"（{saved.get('evaluation_recording_count')} 录音），"
            f"状态 {saved.get('status')}。"
        )
        self.refresh()

    # ------------------------------------------------------------ 评估表
    def _evaluation_rows(self, model_dir: Path) -> tuple[list[dict], dict]:
        table_path = Path(model_dir) / "expert_evaluation.csv"
        metadata_path = Path(model_dir) / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        rows: list[dict] = []
        if table_path.exists():
            frame = pd.read_csv(table_path, encoding="utf-8-sig").fillna("")
            for _, row in frame.iterrows():
                record = {
                    "species": str(row.get("species", "")),
                    "support": int(row.get("support") or 0),
                    "precision": row.get("precision") or None,
                    "recall": row.get("recall") or None,
                    "f1": row.get("f1") or None,
                    "status": str(row.get("status", "") or ""),
                }
                rows.append(record)
        else:
            evaluation = (metadata.get("evaluation") or {})
            rows = []
        return rows, metadata

    def _show_evaluation(self) -> None:
        model_dir = self._pick_model_dir("选择要查看评估表的模型")
        if model_dir is None:
            return
        rows, metadata = self._evaluation_rows(model_dir)
        if not rows:
            QMessageBox.information(
                self,
                "没有评估表",
                "该模型没有冻结评估结果（expert_evaluation.csv）。\n"
                "请在生成专家数据划分后用「训练新模型」训练，评估表会随模型保存。",
            )
            return
        evaluation = metadata.get("evaluation") or {}
        dialog = _EvaluationTableDialog(
            rows,
            macro_f1=evaluation.get("macro_f1_supported"),
            macro_classes=list(evaluation.get("macro_f1_classes") or [])
            if isinstance(evaluation.get("macro_f1_classes"), list)
            else [],
            macro_excluded=list(evaluation.get("macro_f1_excluded") or [])
            if isinstance(evaluation.get("macro_f1_excluded"), list)
            else [],
            parent=self,
        )
        dialog.exec()

    def _compare_models(self) -> None:
        """新旧模型对比：§十一——两者**同时重新**在同一个 Gold Evaluation 上评估。

        绝不拿旧模型以前缓存的结果与新模型当前结果直接比较。
        """
        run_dir = self._run_dir()
        if run_dir is None:
            QMessageBox.information(self, "无法对比", "请先选择/打开一个 ⑥ 任务。")
            return
        store = ExpertSplitStore.load(Path(run_dir))
        if not store.frozen:
            QMessageBox.information(
                self,
                "无法对比",
                "尚未生成专家数据划分（没有固定的 Gold Evaluation）。\n"
                "新旧模型必须在同一个固定 Evaluation 上重新评估才能比较。",
            )
            return
        gold = gold_record_from_split(store.record)
        if not gold["truth"]:
            QMessageBox.information(
                self,
                "无法对比",
                "该划分缺少冻结真值快照（evaluation_truth）。\n"
                "请重新生成专家数据划分以固定考试卷。",
            )
            return

        old_dir = self._pick_model_dir("选择旧模型（baseline）")
        if old_dir is None:
            return
        new_dir = self._pick_model_dir("选择新模型（candidate）")
        if new_dir is None:
            return

        try:
            from ..species_model import (
                compare_models_on_gold_evaluation,
                load_species_model,
            )

            old_payload = load_species_model(old_dir)
            new_payload = load_species_model(new_dir)
            frame = self._feature_frame(run_dir)
            result = compare_models_on_gold_evaluation(
                old_payload, new_payload, frame, gold
            )
        except Exception as error:  # noqa: BLE001 - UI 显示全部失败原因
            QMessageBox.critical(self, "对比失败", str(error))
            self._log(f"新旧模型对比失败：{error}")
            return

        old_metrics = result["old"]
        new_metrics = result["new"]
        comparison = result["comparison"]
        drops = comparison.get("recall_drop_classes") or []
        lines = [
            f"同一 Gold Evaluation v{result['gold_eval_version']}（冻结专家真值，"
            f"{len(result['evaluation_pulse_ids'])} 题）——两个模型均重新预测，未使用缓存结果。",
            f"旧模型 Macro-F1：{_format_metric(old_metrics.get('macro_f1_supported'))}"
            f" ｜ log loss：{_format_metric(old_metrics.get('log_loss'))}",
            f"新模型 Macro-F1：{_format_metric(new_metrics.get('macro_f1_supported'))}"
            f" ｜ log loss：{_format_metric(new_metrics.get('log_loss'))}",
            f"Δ Macro-F1：{_format_metric(comparison.get('macro_f1_delta'))}",
        ]
        if drops:
            lines.append(f"⚠ 以下物种 Recall 下降超过容忍值：{'、'.join(drops)}")
        lines.append(
            "建议采纳："
            + ("是" if result["adoptable"] else "否（不退步才换模型）")
        )
        QMessageBox.information(
            self,
            "新旧模型对比（同一 Fixed Gold Evaluation）",
            "\n".join(lines),
        )
        self._log("新旧模型对比：" + " ｜ ".join(lines))

    def _feature_frame(self, run_dir: Path) -> pd.DataFrame:
        """Gold Evaluation 的特征来源（clustered_pulses 优先）。"""
        tables = Path(run_dir) / "tables"
        for name in ("clustered_pulses.csv", "expert_training_dataset.csv"):
            path = tables / name
            if path.exists():
                try:
                    return pd.read_csv(path, encoding="utf-8-sig").fillna("")
                except UnicodeDecodeError:
                    return pd.read_csv(path).fillna("")
        raise ValueError(
            "任务目录缺少特征表（clustered_pulses.csv / expert_training_dataset.csv）。"
        )

    def _pick_model_dir(self, title: str) -> Path | None:
        from PySide6.QtWidgets import QFileDialog

        default = ""
        if self._settings is not None:
            default = str(getattr(self._settings, "species_model_path", "") or "")
        chosen = QFileDialog.getExistingDirectory(self, title, default)
        return Path(chosen) if chosen else None
