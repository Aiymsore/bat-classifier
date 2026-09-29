"""模型选择控件：列出可用模型 + 「导入模型文件…」按钮（③ / ④ / ⑦ 共用）。

- 下拉框列出 model_store.list_models() 的结果，默认选中 ② 训练出的模型；
- 「导入模型文件…」把用户选中的 .joblib 复制到 models/imported/（不会覆盖
  ② 的训练产物），导入成功后自动选中新模型；
- 选择变化通过 model_changed(str) 信号向外广播。

本控件只做 UI 与文件搬运，不训练模型。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from ..model_store import (
    TRAINED_MODEL_PATH,
    import_model_file,
    list_models,
)


class ModelPicker(QWidget):
    """模型下拉选择 + 导入按钮。当前模型路径用 current_path() 取。"""

    model_changed = Signal(str)   # 当前选中模型的绝对路径

    def __init__(
        self,
        initial_path: str = "",
        log: Callable[[str], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._log = log or (lambda text: None)
        self._initial = str(initial_path or TRAINED_MODEL_PATH)
        self._syncing = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._combo = QComboBox()
        self._combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._combo.setToolTip("选择本次使用的分类模型（② 训练的或已导入的）")

        self._import_button = QPushButton("导入模型文件…")
        self._import_button.setToolTip(
            "选择一个 .joblib 模型文件，复制到 models/imported/ 后即可使用；"
            "不会覆盖 ② 训练出的模型。"
        )
        self._refresh_button = QPushButton("刷新")

        layout.addWidget(self._combo, stretch=1)
        layout.addWidget(self._import_button)
        layout.addWidget(self._refresh_button)

        self._combo.currentIndexChanged.connect(self._on_index_changed)
        self._import_button.clicked.connect(self.import_interactively)
        self._refresh_button.clicked.connect(self.refresh)

        self.refresh(self._initial)

    # --------------------------------------------------------------- API
    def current_path(self) -> Path | None:
        path = self._combo.currentData()
        return Path(str(path)) if path else None

    def current_path_text(self) -> str:
        path = self.current_path()
        return str(path) if path else ""

    def set_path(self, path: Path | str) -> None:
        """把选择同步到指定模型（找不到该模型时静默忽略）。"""
        if not path:
            return
        index = self._combo.findData(str(Path(path)))
        if index < 0 or index == self._combo.currentIndex():
            return
        self._combo.setCurrentIndex(index)   # 触发 _on_index_changed → 发信号

    def refresh(self, select_path: Path | str | None = None) -> None:
        """重扫模型目录并重建下拉框；select_path 为空则保留当前选择。"""
        previous = (
            str(Path(select_path))
            if select_path
            else (self.current_path_text() or self._initial)
        )
        entries = list_models()

        self._syncing = True
        self._combo.blockSignals(True)
        self._combo.clear()
        for entry in entries:
            tooltip = f"{entry.path}\n来源：{entry.origin_text}"
            if entry.labels:
                tooltip += f"\n标签：{'、'.join(entry.labels)}"
            tooltip += "\n大小：%.1f MB" % entry.size_mb
            if not entry.valid:
                tooltip += f"\n⚠ 不可用：{entry.error}"
            self._combo.addItem(entry.display_name, str(entry.path))
            self._combo.setItemData(
                self._combo.count() - 1,
                tooltip,
                Qt.ToolTipRole,
            )
        self._combo.blockSignals(False)
        self._syncing = False

        if self._combo.count() == 0:
            self._combo.addItem("（models/ 下暂无可用模型）", "")
            return

        index = self._combo.findData(previous)
        if index < 0:
            index = 0   # 找不到原选择 → 回退到第一个（② 训练模型优先）
        self._combo.setCurrentIndex(index)

    # ------------------------------------------------------- 信号 / 导入
    def _on_index_changed(self, index: int) -> None:
        if self._syncing:
            return
        path = self.current_path_text()
        if path:
            self.model_changed.emit(path)

    def import_interactively(self) -> None:
        """弹出文件选择框，把选中的模型复制进 models/imported/ 并选中它。"""
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "导入模型文件",
            "",
            "模型文件 (*.joblib);;所有文件 (*)",
        )
        if not chosen:
            return

        try:
            target = import_model_file(Path(chosen))
        except Exception as error:  # noqa: BLE001 - 面向用户展示即可
            self._log(f"导入模型失败：{error}")
            QMessageBox.warning(self, "导入失败", str(error))
            return

        self._log(f"已导入模型：{target}")
        # refresh() 结束时会把选择切到新模型并发出 model_changed
        self.refresh(target)
        QMessageBox.information(
            self,
            "导入成功",
            f"模型已导入：\n{target}\n\n"
            "已切换为该模型；② 训练出的模型仍保留在 models/ 中可随时切回。",
        )
