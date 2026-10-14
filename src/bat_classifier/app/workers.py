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
import threading
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

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """请求任务在下一处处理边界停止，保留已输出的结果。"""
        self._stop_event.set()

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


class SegmentAnalysisWorker(_LoggedWorker):
    """① 片段识别与叫声统计：检测叫声 + 逐叫声分类 + 片段/总体统计。

    只负责线程、进度 signal、停止、日志与异常转发；
    核心算法在 segment_analysis.py（完全脱离 Qt，可单独测试）。
    """

    progress = Signal(int, int, str)   # current, total, current_file
    chunk_progress = Signal(object)    # ① 长录音块级进度 payload（§13）
    stats = Signal(object)             # 运行统计 payload dict

    def __init__(
        self,
        input_path: Path,
        config: dict,
        output_root: Path | None = None,
        resume: bool | None = None,
        template_dir: Path | str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._input_path = Path(input_path)
        self._config = dict(config)
        self._output_root = Path(output_root) if output_root else None
        self._resume = resume
        self._template_dir = Path(template_dir) if template_dir else None
        self._success_count = 0
        self._failed_count = 0


    def _emit_progress(self, current: int, total: int, current_file: str) -> None:
        self.progress.emit(int(current), int(total), str(current_file))

    def _emit_chunk_progress(self, payload: dict) -> None:
        self.chunk_progress.emit(dict(payload))

    def _emit_segment_stats(self, result: dict) -> None:
        if result.get("status") == "success":
            self._success_count += 1
        else:
            self._failed_count += 1
        self.stats.emit(
            {
                "success": self._success_count,
                "failed": self._failed_count,
                "current_file": result.get("segment_file", ""),
                "major_label": result.get("major_label", ""),
                "detected_call_count": result.get("detected_call_count", 0),
            }
        )

    def _should_stop(self) -> bool:
        return self._stop_event.is_set()

    def run(self) -> None:
        from dataclasses import fields

        from ..segment_analysis import (
            SegmentAnalysisConfig,
            analyze_segment_folder,
        )

        def job() -> dict:
            known = {field.name for field in fields(SegmentAnalysisConfig)}
            config = SegmentAnalysisConfig(
                **{
                    key: value
                    for key, value in self._config.items()
                    if key in known
                }
            ).validate()
            if self._template_dir is None:
                raise ValueError("请先选择模板文件夹。")
            if self._template_dir is not None:
                from ..template_matching import build_template_model

                print(f"正在加载模板目录：{self._template_dir}")
                template_payload = build_template_model(
                    self._template_dir,
                    min_similarity=float(
                        self._config.get(
                            "min_confidence", 0.60
                        )
                    ),
                )
                print(
                    "模板类别："
                    f"{len(template_payload.get('labels') or [])} 个，"
                    f"样本：{template_payload.get('template_count', 0)} 个"
                )
                errors = template_payload.get("errors") or []
                if errors:
                    print(f"跳过无法读取的模板：{len(errors)} 个")

            return analyze_segment_folder(
                self._input_path,
                config,
                output_root=self._output_root,
                progress=self._emit_progress,
                chunk_progress=self._emit_chunk_progress,
                on_segment_done=self._emit_segment_stats,
                should_stop=self._should_stop,
                resume=self._resume,
                template_payload=template_payload,
            )

        self._run_logged(job)


class LowSimilarityScanWorker(_LoggedWorker):
    """② 第一次模板扫描；只导出低分 pulse，候选由页面人工选择。"""

    progress = Signal(int, int, str)
    chunk_progress = Signal(object)

    def __init__(self, input_path: Path, template_dir: Path, threshold: float, config: dict, parent=None):
        super().__init__(parent)
        self.input_path = Path(input_path)
        self.template_dir = Path(template_dir)
        self.threshold = float(threshold)
        self.config = dict(config)


    def run(self) -> None:
        from ..noise_template_scan import scan_low_similarity_pulses
        from ..segment_analysis import SegmentAnalysisConfig

        self._run_logged(lambda: scan_low_similarity_pulses(
            self.input_path, self.template_dir, similarity_threshold=self.threshold,
            analysis_config=SegmentAnalysisConfig(**self.config),
            progress=lambda current, total, name: self.progress.emit(current, total, name),
            chunk_progress=lambda payload: self.chunk_progress.emit(payload),
            should_stop=self._stop_event.is_set,
        ))
