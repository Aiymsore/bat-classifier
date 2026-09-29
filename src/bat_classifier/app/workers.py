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


class CurationWorker(_LoggedWorker):
    """⑥ 批量整理与专家标注：扫描 → 逐录音处理 → 聚类与代表样本。

    mode：
    - "process"（默认）：新建或续跑一个 run，完成处理后执行聚类；
    - "add_samples"：为已有 run 的指定 Cluster 追加代表样本。

    progress/stats 信号供 UI 显示批处理进度与统计；
    算法本体在 curation 包里，完全不依赖 Qt。
    """

    progress = Signal(int, int)   # current, total
    stats = Signal(object)        # 统计 payload dict

    def __init__(
        self,
        input_path: Path | None = None,
        run_dir: Path | None = None,
        params: dict | None = None,
        mode: str = "process",
        cluster_id: str | None = None,
        add_count: int | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._input_path = Path(input_path) if input_path else None
        self._run_dir = Path(run_dir) if run_dir else None
        self._params = dict(params or {})
        self._mode = mode
        self._cluster_id = cluster_id
        self._add_count = add_count

    def _emit_progress(self, current: int, total: int) -> None:
        self.progress.emit(int(current), int(total))

    def _emit_stats(self, payload: dict) -> None:
        self.stats.emit(payload)

    def run(self) -> None:
        from ..curation import CurationParams, CurationRun

        def build_params() -> CurationParams:
            from dataclasses import fields

            known = {field.name for field in fields(CurationParams)}
            return CurationParams(
                **{
                    key: value
                    for key, value in self._params.items()
                    if key in known
                }
            ).validate()

        def job_process() -> dict:
            params = build_params()
            if self._run_dir is not None:
                run = CurationRun.open(self._run_dir)
                print(f"继续已有任务：{run.run_dir}")
            else:
                if self._input_path is None:
                    raise ValueError("缺少输入路径。")
                run = CurationRun.create(self._input_path)
                print(f"新建批量整理任务：{run.run_dir}")

            run.stage_scan()
            process_stats = run.stage_process(
                params,
                progress=self._emit_progress,
                stats=self._emit_stats,
            )
            cluster_result = run.stage_cluster(
                params,
                stats=self._emit_stats,
            )
            if not cluster_result.get("ok"):
                print(f"聚类未完成：{cluster_result.get('error')}")
            return {
                "action": "process",
                "run_dir": str(run.run_dir),
                "stats": process_stats,
                "cluster": cluster_result,
            }

        def job_add_samples() -> dict:
            params = build_params()
            from ..curation.representatives import add_cluster_representatives

            if self._run_dir is None:
                raise ValueError("缺少任务目录。")
            run = CurationRun.open(self._run_dir)
            added = add_cluster_representatives(
                run,
                str(self._cluster_id or ""),
                params,
                count=self._add_count,
            )
            return {
                "action": "add_samples",
                "run_dir": str(run.run_dir),
                "cluster_id": str(self._cluster_id or ""),
                "added": int(len(added)),
            }

        if self._mode == "add_samples":
            self._run_logged(job_add_samples)
        else:
            self._run_logged(job_process)


class SegmentAnalysisWorker(_LoggedWorker):
    """⑦ 片段识别与叫声统计：检测叫声 + 逐叫声分类 + 片段/总体统计。

    只负责线程、进度 signal、停止、日志与异常转发；
    核心算法在 segment_analysis.py（完全脱离 Qt，可单独测试）。
    """

    progress = Signal(int, int, str)   # current, total, current_file
    chunk_progress = Signal(object)    # ⑦ 长录音块级进度 payload（§13）
    stats = Signal(object)             # 运行统计 payload dict

    def __init__(
        self,
        input_path: Path,
        combined_payload: dict | None,
        config: dict,
        output_root: Path | None = None,
        model_path: Path | str | None = None,
        resume: bool | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._input_path = Path(input_path)
        self._combined_payload = combined_payload
        self._config = dict(config)
        self._output_root = Path(output_root) if output_root else None
        self._model_path = str(model_path or "")
        self._resume = resume
        self._stop_event = threading.Event()
        self._success_count = 0
        self._failed_count = 0

    def stop(self) -> None:
        """请求在下一个 chunk/片段边界优雅停止（不杀线程，安全收尾）。"""
        self._stop_event.set()

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
            return analyze_segment_folder(
                self._input_path,
                self._combined_payload,
                config,
                output_root=self._output_root,
                progress=self._emit_progress,
                chunk_progress=self._emit_chunk_progress,
                on_segment_done=self._emit_segment_stats,
                should_stop=self._should_stop,
                model_path=self._model_path or None,
                resume=self._resume,
            )

        self._run_logged(job)


class CalibrateModelWorker(_LoggedWorker):
    """补校准：给**已训练好的两级模型**挂 CalibratedClassifierCV 校准器。

    只 fit 校准器（按 source_recording 分组隔离），不重训模型、不覆盖原模型；
    结果保存为新文件。数据不足以安全校准时明确失败，绝不伪造校准概率。
    """

    def __init__(
        self,
        model_path: Path,
        csv_path: Path,
        output_path: Path,
        method: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._model_path = Path(model_path)
        self._csv_path = Path(csv_path)
        self._output_path = Path(output_path)
        self._method = method or ""

    def run(self) -> None:
        import joblib
        import pandas as pd

        from ..calibration import (
            attach_calibration,
            detect_label_column,
            detect_source_column,
        )

        def job() -> dict:
            payload = joblib.load(self._model_path)
            frame = pd.read_csv(self._csv_path, encoding="utf-8-sig")

            acoustic = (
                payload.get("acoustic_types")
                if isinstance(payload, dict)
                else None
            )
            if not isinstance(acoustic, dict):
                raise ValueError(
                    "模型文件不是两级模型（缺 acoustic_types 段）。"
                )

            label_column = detect_label_column(frame)
            if not label_column:
                raise ValueError(
                    "标注 CSV 里找不到真值标签列（需含 "
                    "acoustic_type / label / expert_label / true_label / "
                    "species 之一）；模型自己的 predicted_label 不能当真值。"
                )
            source_column = detect_source_column(frame)
            labels = frame[label_column]
            groups = frame[source_column] if source_column else None

            calibrated_payload = attach_calibration(
                payload,
                frame,
                labels,
                groups=groups,
                method=self._method or None,
            )
            info = dict(
                (calibrated_payload.get("acoustic_types") or {}).get(
                    "calibration"
                )
                or {}
            )
            if not info.get("is_calibrated"):
                raise ValueError(
                    "数据不足以安全校准，模型保持未校准、未生成新文件："
                    + "；".join(info.get("warnings") or ["未知原因"])
                )

            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(calibrated_payload, self._output_path)
            return {
                "output_path": str(self._output_path),
                "label_column": label_column,
                "source_column": source_column,
                "calibration": info,
            }

        self._run_logged(job)


class SpeciesModelWorker(_LoggedWorker):
    """⑥ 已确认物种模型：训练 / 导入 / 切换(加载) / 用标注数据校准。

    算法本体在 species_model.py（完全脱离 Qt）；本类只负责线程、
    日志转发与异常上报。action：
    - "train"     用专家确认标签训练新模型（RF + 概率校准）；
    - "import"    导入现成监督模型并校验特征/类别/校准状态；
    - "load"      切换到另一个已保存模型；
    - "calibrate" 用真实标注数据为未校准模型补概率校准。
    """

    def __init__(
        self,
        action: str,
        sources: list | None = None,
        model_name: str = "",
        model_path: str = "",
        calibration_method: str = "sigmoid",
        expert_split: str = "",
        pseudo_source: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._action = str(action)
        self._sources = [str(item) for item in (sources or [])]
        self._model_name = str(model_name or "")
        self._model_path = str(model_path or "")
        self._calibration_method = str(calibration_method or "sigmoid")
        self._expert_split = str(expert_split or "")
        self._pseudo_source = str(pseudo_source or "")

    def run(self) -> None:
        from .. import species_model

        def job() -> dict:
            if self._action == "train":
                result = species_model.train_species_model(
                    self._sources,
                    self._model_name,
                    calibration_method=self._calibration_method,
                    expert_split=self._expert_split or None,
                    pseudo_source=self._pseudo_source or None,
                )
                return {
                    "action": "train",
                    "model_dir": result["model_dir"],
                    "payload": result["payload"],
                    "evaluation": result.get("evaluation"),
                    "notes": result.get("notes", []),
                    "is_calibrated": result.get("is_calibrated", False),
                }
            if self._action == "import":
                result = species_model.import_species_model(
                    self._model_path,
                    self._model_name or None,
                )
                return {
                    "action": "import",
                    "model_dir": result["model_dir"],
                    "payload": result["payload"],
                }
            if self._action == "load":
                payload = species_model.load_species_model(self._model_path)
                return {
                    "action": "load",
                    "model_dir": self._model_path,
                    "payload": payload,
                }
            if self._action == "calibrate":
                payload = species_model.load_species_model(self._model_path)
                result = species_model.calibrate_species_model_with_labels(
                    payload,
                    self._sources,
                    calibration_method=self._calibration_method,
                )
                return {
                    "action": "calibrate",
                    "model_dir": result["model_dir"],
                    "payload": result["payload"],
                }
            raise ValueError(f"未知操作：{self._action}")

        self._run_logged(job)


class SpeciesCompareWorker(_LoggedWorker):
    """⑥ 模型对照：Cluster 四类 representative → 已确认物种模型判断与统计。"""

    def __init__(
        self,
        run_dir: Path,
        payload: dict,
        uncertain_threshold: float,
        model_name: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._run_dir = Path(run_dir)
        self._payload = payload
        self._threshold = float(uncertain_threshold)
        self._model_name = str(model_name or "")

    def run(self) -> None:
        from ..curation.model_comparison import run_model_comparison
        from ..curation.pipeline import CurationRun

        def job() -> dict:
            run = CurationRun.open(self._run_dir)
            return run_model_comparison(
                run,
                self._payload,
                uncertain_threshold=self._threshold,
                model_name=self._model_name,
            )

        self._run_logged(job)
