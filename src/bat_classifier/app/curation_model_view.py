"""⑥ 审核工作台的「模型对照结果」区域（Cluster 四类 representative 分别展示）。

展示结构（对应科研要求）：
- 四行小表：Center / Boundary / Random / Outlier 各自的
  主导物种 · 一致率 · 平均校准概率（未校准模型显示“平均模型支持度”）；
- 综合判断：unique representative 去重后的总体一致率与总体平均校准概率；
- 点击某一类可查看该类全部样本的逐条模型判断（双击可在声谱图查看器中打开）。

agreement 与 confidence 是两个不同指标，绝不在这里混用。
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import (
    MODEL_STATUS_HIGH,
    MODEL_STATUS_INSUFFICIENT,
    MODEL_STATUS_LOW,
    MODEL_STATUS_MIXED,
    MODEL_STATUS_MODERATE,
    MODEL_STATUS_NO_MODEL,
    SPECIES_UNCERTAIN_LABEL,
)
from ..curation.model_comparison import REASON_LABELS, REASON_ORDER, split_reasons

STATUS_TEXT = {
    MODEL_STATUS_HIGH: "高一致性（暂不需要人工复核）",
    MODEL_STATUS_MODERATE: "中等一致性",
    MODEL_STATUS_LOW: "低一致性（需要人工审核）",
    MODEL_STATUS_MIXED: "混杂（Center 与 Boundary 主导物种不同，需要人工审核）",
    MODEL_STATUS_INSUFFICIENT: "样本不足（需要人工审核）",
    MODEL_STATUS_NO_MODEL: "未加载模型",
}


def _percent(value: object, digits: int = 1) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "—"
    if number != number:  # NaN
        return "—"
    return f"{number * 100:.{digits}f}%"


class ModelComparisonPane(QGroupBox):
    """模型对照结果：四类 representative 独立统计 + Cluster 综合判断。"""

    compare_requested = Signal()
    sample_activated = Signal(object)  # 逐条样本（dict），双击查看声谱图

    def __init__(
        self,
        log: Callable[[str], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("模型对照结果（模型辅助判断，不覆盖专家标注）", parent)
        self._log = log or (lambda _text: None)
        self._result: dict | None = None
        self._predictions: pd.DataFrame | None = None
        self._build_ui()
        self._show_empty()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        self._model_label = QLabel("尚未运行模型对照。")
        self._model_label.setWordWrap(True)
        self._run_button = QPushButton("运行模型对照")
        self._run_button.clicked.connect(self.compare_requested.emit)
        header.addWidget(self._model_label, stretch=1)
        header.addWidget(self._run_button)
        layout.addLayout(header)

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ["类型", "主导物种", "一致率", "平均校准概率", "样本数", "不确定"]
        )
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.NoSelection)
        self._table.verticalHeader().setVisible(False)
        self._table.setMinimumHeight(150)  # 4 行 + 表头完整可见（整页滚动时不被压扁）
        self._table.setMaximumHeight(190)
        layout.addWidget(self._table)

        self._overall_label = QLabel("综合判断：尚未运行模型对照。")
        self._overall_label.setWordWrap(True)
        layout.addWidget(self._overall_label)

        detail_row = QHBoxLayout()
        detail_row.addWidget(QLabel("逐样本查看"))
        self._detail_combo = QComboBox()
        for reason in REASON_ORDER:
            self._detail_combo.addItem(REASON_LABELS[reason], reason)
        self._detail_combo.addItem("综合（去重）", "overall")
        self._detail_combo.currentIndexChanged.connect(self._refresh_detail)
        detail_row.addWidget(self._detail_combo)
        detail_row.addWidget(QLabel("双击样本可在声谱图查看器中打开"))
        detail_row.addStretch(1)
        layout.addLayout(detail_row)

        self._detail_list = QListWidget()
        self._detail_list.setMinimumHeight(120)  # 逐样本列表至少可读（可滚）
        self._detail_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self._detail_list, stretch=1)

    # ------------------------------------------------------------ 数据
    def set_result(self, result: dict | None) -> None:
        self._result = result if isinstance(result, dict) else None
        if self._result and self._result.get("ok"):
            predictions = self._result.get("predictions")
            self._predictions = (
                predictions if isinstance(predictions, pd.DataFrame) else None
            )
            kind = (
                "calibrated probability（校准概率）"
                if self._result.get("is_calibrated")
                else "未校准模型：以下概率为模型支持度（model support），不是校准概率"
            )
            self._model_label.setText(
                f"模型：{self._result.get('model_name', '')} ｜ "
                f"uncertain 阈值 {float(self._result.get('uncertain_threshold', 0.0)):.2f} ｜ "
                f"概率类型：{kind}"
                + (
                    " ｜ ⚠ TRAINING_DATA_OVERLAP：代表样本与模型训练数据重合，不代表泛化能力"
                    if self._result.get("training_data_overlap")
                    else ""
                )
            )
        else:
            self._predictions = None
            message = (
                self._result.get("error", "尚未运行模型对照。")
                if self._result
                else "尚未运行模型对照。"
            )
            self._model_label.setText(str(message))
            self._show_empty()
            return
        self._refresh_table()

    def clear(self) -> None:
        self.set_result(None)

    def show_cluster(self, cluster_id: str | None) -> None:
        self._current_cluster = str(cluster_id) if cluster_id is not None else None
        self._refresh_table()

    # ------------------------------------------------------------ 渲染
    def _show_empty(self) -> None:
        self._table.setRowCount(0)
        self._overall_label.setText("综合判断：尚未运行模型对照。")
        self._detail_list.clear()

    def _cluster_result(self) -> dict | None:
        if not self._result or not self._result.get("ok"):
            return None
        cluster_id = getattr(self, "_current_cluster", None)
        for cluster in self._result.get("clusters", []):
            if str(cluster.get("cluster_id")) == str(cluster_id):
                return cluster
        return None

    def _probability_column(self) -> str:
        if self._result and self._result.get("is_calibrated"):
            return "average_calibrated_probability"
        return "average_model_support"

    def _probability_header(self) -> str:
        if self._result and self._result.get("is_calibrated"):
            return "平均校准概率"
        return "平均模型支持度（未校准）"

    def _refresh_table(self) -> None:
        cluster = self._cluster_result()
        self._table.setHorizontalHeaderLabels(
            ["类型", "主导物种", "一致率", self._probability_header(), "样本数", "不确定"]
        )
        if cluster is None:
            self._table.setRowCount(0)
            self._overall_label.setText(
                "综合判断：该 Cluster 尚无模型对照结果。"
            )
            self._detail_list.clear()
            return

        groups = cluster.get("groups", {})
        probability_key = self._probability_column()
        self._table.setRowCount(len(REASON_ORDER))
        for row_index, reason in enumerate(REASON_ORDER):
            stats = groups.get(reason, {})
            values = [
                REASON_LABELS[reason],
                str(stats.get("dominant_label", "") or "—"),
                _percent(stats.get("agreement_rate")),
                _percent(stats.get(probability_key)),
                str(int(stats.get("sample_count", 0))),
                str(int(stats.get("uncertain_count", 0))),
            ]
            for column_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(Qt.ItemIsEnabled)
                self._table.setItem(row_index, column_index, item)

        overall = cluster.get("overall", {})
        status = str(cluster.get("model_status", ""))
        flags = cluster.get("flags") or []
        probability_text = (
            f"总体平均校准概率：{_percent(overall.get(probability_key))}"
            if self._result.get("is_calibrated")
            else f"总体平均模型支持度（未校准）：{_percent(overall.get(probability_key))}"
        )
        lines = [
            "综合判断（unique representative 去重后统计）",
            f"主导物种：{overall.get('dominant_label', '—')}",
            (
                f"总体一致率：{_percent(overall.get('agreement_rate'))}"
                f"（{int(overall.get('dominant_count', 0))} / {int(overall.get('sample_count', 0))}）"
            ),
            probability_text,
            f"状态：{STATUS_TEXT.get(status, status)}",
        ]
        if flags:
            lines.append(f"标记：{'、'.join(str(flag) for flag in flags)}")
        lines.append(
            "提示：模型结果是 model_inferred_label，绝不覆盖 expert_label；"
            "低一致性 Cluster 请继续人工审核。"
        )
        self._overall_label.setText("\n".join(lines))
        self._refresh_detail()

    def _detail_rows(self) -> list[dict]:
        cluster_id = getattr(self, "_current_cluster", None)
        if self._predictions is None or cluster_id is None:
            return []
        frame = self._predictions
        if "cluster_id" not in frame.columns:
            return []
        rows = frame[frame["cluster_id"].astype(str) == str(cluster_id)]
        reason = str(self._detail_combo.currentData() or "overall")
        if reason != "overall":
            rows = rows[
                rows["selection_reason"].map(
                    lambda value, r=reason: r in split_reasons(value)
                )
            ]
        else:
            rows = rows.drop_duplicates(subset=["pulse_id"], keep="first")
        return rows.to_dict("records")

    def _refresh_detail(self) -> None:
        self._detail_list.clear()
        for row in self._detail_rows():
            label = str(row.get("model_inferred_label", "") or "")
            if not label:
                continue
            probability = row.get("calibrated_probability")
            kind = str(row.get("probability_kind", "") or "")
            try:
                probability_text = _percent(probability)
            except Exception:  # noqa: BLE001
                probability_text = "—"
            uncertain_mark = "（uncertain）" if bool(row.get("is_uncertain")) else ""
            suffix = "" if kind == "calibrated_probability" else "（支持度）"
            item = QListWidgetItem(
                f"{row.get('pulse_id', '')} → {label}{uncertain_mark} → "
                f"{probability_text}{suffix}"
            )
            item.setData(Qt.UserRole, row)
            self._detail_list.addItem(item)

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        row = item.data(Qt.UserRole)
        if isinstance(row, dict):
            self.sample_activated.emit(row)
