"""后台工作线程：把库函数的耗时操作移出 UI 线程，并回传日志。

所有 Worker 都是 QThread 子类：
- log    信号：库内 print 的输出行（经 stdout 重定向）；
- done   信号：任务成功的结果对象（dict / int 等）；
- failed 信号：异常文本。
库模块在本文件顶部导入是安全的——
run_app() 已在实例化任何 Worker 之前应用工作区环境变量。
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

from PySide6.QtCore import QThread, Signal


class LogStream(io.TextIOBase):
    """把 print 输出逐行转发为 Qt 信号的 stdout 替身。"""

    def __init__(self, emit) -> None:
        super().__init__()
        self._emit = emit
        self._buffer = ""

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit(line)
        if self._buffer:
            self._emit(self._buffer)
            self._buffer = ""
        return len(text)

    def flush(self) -> None:
        return None


class _LoggedWorker(QThread):
    """提供 stdout 重定向和统一异常处理的基类。"""

    log = Signal(str)
    done = Signal(object)
    failed = Signal(str)

    def _emit_line(self, line: str) -> None:
        self.log.emit(line)

    def _run_logged(self, function) -> None:
        stream = LogStream(self._emit_line)
        try:
            with contextlib.redirect_stdout(stream):
                result = function()
        except Exception as error:  # noqa: BLE001 - 工作线程兜底
            self.failed.emit(f"{type(error).__name__}: {error}")
            return
        self.done.emit(result)


class ImportDataWorker(_LoggedWorker):
    """把用户选择的数据源（文件夹/zip）导入工作区 data/raw。"""

    def __init__(self, source: Path, parent=None) -> None:
        super().__init__(parent)
        self._source = Path(source)

    def run(self) -> None:
        from ..pulse_table import prepare_demo_dataset

        self._run_logged(
            lambda: prepare_demo_dataset(self._source)
        )


class TrainingWorker(_LoggedWorker):
    """执行完整训练：建表 → 门控 → 聚类 → 保存两级模型。"""

    def __init__(
        self,
        labels: list[str],
        type_descriptions: dict[str, str],
        data_source: Path | None,
        save_diagnostics: bool | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._labels = list(labels)
        self._descriptions = dict(type_descriptions)
        self._source = data_source
        self._save_diagnostics = save_diagnostics

    def run(self) -> None:
        from ..pipeline import run_full_training

        self._run_logged(
            lambda: run_full_training(
                self._labels,
                self._descriptions,
                self._source,
                self._save_diagnostics,
            )
        )


class PredictWorker(_LoggedWorker):
    """对新 WAV/文件夹执行两级预测。"""

    def __init__(self, input_path: Path, payload: dict, parent=None) -> None:
        super().__init__(parent)
        self._input_path = Path(input_path)
        self._payload = payload

    def run(self) -> None:
        from ..prediction import predict_demo_path_two_stage

        self._run_logged(
            lambda: predict_demo_path_two_stage(
                self._input_path,
                self._payload,
            )
        )
