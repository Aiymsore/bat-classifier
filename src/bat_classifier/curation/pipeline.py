"""大规模未标注 WAV 的批量整理（⑥ 批量整理与专家标注 的算法层）。

三段式，全部可脱离 Qt 调用（CLI / 测试 / 服务器跑批）：

- Stage 1 scan     扫描输入（单 WAV 或文件夹递归），建立 run 与 manifest；
- Stage 2 process  逐录音 检测 → 同时声波分离 → 质量筛选 → 切割写盘，
                   一条一条处理、及时写结果、释放音频，支持断点续跑；
- Stage 3 cluster  载入 accepted 特征表 → StandardScaler → KMeans
                   （自动/手动 K）→ 代表样本 → 图表与审核数据（见 clustering.py）。

隔离约定：
- 输入直接读用户选择的外部路径，**绝不**复制进 data/raw，也绝不调用
  prepare_demo_dataset 的清空逻辑；
- 输出只写 results/curation/<run_id>/，不覆盖 predictions/tables/figures/models。

Cluster ≠ 物种：聚类只负责找声学相似组以减少专家审核量，
真实物种标签只能来自专家人工鉴定。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import math

import numpy as np
import pandas as pd
import soundfile as sf

from ..audio_io import calculate_audio_content_hash, find_wav_files, load_audio
from ..config import (
    BASE_DIR,
    CURATION_ANALYSIS_SPAN_S,
    CURATION_DEFAULT_N_BOUNDARY,
    CURATION_DEFAULT_N_CENTER,
    CURATION_DEFAULT_N_OUTLIER,
    CURATION_DEFAULT_N_RANDOM,
    CURATION_DIR,
    CURATION_K_MAX,
    CURATION_K_MIN,
    DETECTION_CHUNK_SECONDS,
    RANDOM_STATE,
)
from ..detection import detect_and_refine_pulses_chunked
from ..pulse_table import create_demo_pulse_rows

# ---------------------------------------------------------------------------
# manifest 状态
# ---------------------------------------------------------------------------
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_DUPLICATE = "duplicate"

RUNNABLE_STATUSES = (STATUS_PENDING, STATUS_PROCESSING)

MANIFEST_NAME = "manifest.json"
METADATA_NAME = "run_metadata.json"
RECORDINGS_CSV = "recordings.csv"
ALL_PULSES_CSV = "all_pulses.csv"
ACCEPTED_PULSES_CSV = "accepted_pulses.csv"
REJECTED_PULSES_CSV = "rejected_pulses.csv"

# manifest 落盘频率（每处理 N 条录音强制写一次，兼顾崩溃恢复与 IO 成本）
MANIFEST_FLUSH_EVERY = 5


def analysis_max_windows() -> int:
    """⑥ 每条录音的检测窗口数：把分析跨度延长到 CURATION_ANALYSIS_SPAN_S 秒。

    ≤60 秒的录音整条全覆盖；更长的录音在整个时长上均匀抽这么多窗口。
    """
    return max(
        1,
        math.ceil(CURATION_ANALYSIS_SPAN_S / max(DETECTION_CHUNK_SECONDS, 1e-9)),
    )

ProgressCallback = Callable[[int, int], None]
StatsCallback = Callable[[dict], None]


# ---------------------------------------------------------------------------
# 通用小工具：原子写与表格读取
# ---------------------------------------------------------------------------
def atomic_write_text(path: Path, text: str) -> None:
    """先写临时文件再 replace，避免写到一半损坏。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2),
    )


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def read_table(
    path: Path,
    key_column: str | None = None,
    str_columns: tuple[str, ...] = ("cluster_id", "pulse_id", "audio_hash"),
) -> pd.DataFrame:
    """读取 CSV；文件不存在时返回空表。截断行（缺 pulse_id 等）会被丢弃。

    str_columns 中的列强制按字符串读取（cluster_id 是 "001"，
    直接读会被 pandas 推断成整数 1 从而丢掉前导零）。
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        header = pd.read_csv(path, nrows=0).columns
        dtype = {name: "string" for name in str_columns if name in header}
        frame = pd.read_csv(path, encoding="utf-8-sig", dtype=dtype or None)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if key_column and not frame.empty and key_column in frame.columns:
        frame = frame[frame[key_column].notna()].copy()
    return frame


def append_rows(frame: pd.DataFrame, path: Path) -> None:
    """按已有文件头追加行（首写用 utf-8-sig，追加用 utf-8 避免行内 BOM）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        header = pd.read_csv(path, nrows=0).columns.tolist()
        frame = frame.reindex(columns=header)
        frame.to_csv(path, mode="a", header=False, index=False, encoding="utf-8")
    else:
        frame.to_csv(path, index=False, encoding="utf-8-sig")


def run_relative(path_text: str, run_dir: Path) -> str:
    """把路径尽量转成相对 run 目录的文本，便于 run 目录整体移动/归档。"""
    if not path_text:
        return ""
    path = Path(path_text)
    if not path.is_absolute():
        path = BASE_DIR / path
    for base in (Path(run_dir), BASE_DIR):
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)


def resolve_run_path(path_text: str, run_dir: Path) -> Path | None:
    """把 CSV 中保存的路径文本还原为绝对路径（run 相对 → workspace 相对 → 绝对）。"""
    if not path_text:
        return None
    path = Path(str(path_text))
    if path.is_absolute():
        return path if path.exists() else None
    for base in (Path(run_dir), BASE_DIR):
        candidate = base / path
        if candidate.exists():
            return candidate
    return None


def _software_version() -> str:
    """尽量记录 git commit，方便科研复现；失败时返回 unknown。"""
    try:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=creationflags,
        )
        commit = result.stdout.strip()
        if result.returncode == 0 and commit:
            return f"git:{commit}"
    except Exception:  # noqa: BLE001 - 打包环境可能没有 git
        pass
    return "unknown"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
@dataclass
class CurationParams:
    """一次批量整理任务的全部可调参数（处理 + 聚类）。"""

    k_mode: str = "auto"           # "auto" | "manual"
    k: int = 4                     # manual 模式的聚类数量
    n_center: int = CURATION_DEFAULT_N_CENTER
    n_boundary: int = CURATION_DEFAULT_N_BOUNDARY
    n_random: int = CURATION_DEFAULT_N_RANDOM
    n_outlier: int = CURATION_DEFAULT_N_OUTLIER
    save_pulse_audio: bool = True      # 切割并保存每个合格脉冲 WAV
    save_rejected_audio: bool = False  # 同时保存被拒绝脉冲 WAV（耗磁盘）
    retry_failed: bool = False         # 续跑时是否重试 failed 文件
    random_state: int = RANDOM_STATE

    def validate(self) -> "CurationParams":
        if self.k_mode not in ("auto", "manual"):
            self.k_mode = "auto"
        self.k = int(min(max(self.k, CURATION_K_MIN), CURATION_K_MAX))
        for name in ("n_center", "n_boundary", "n_random", "n_outlier"):
            setattr(self, name, int(max(0, getattr(self, name))))
        return self

    def to_dict(self) -> dict:
        return asdict(self.validate())


# ---------------------------------------------------------------------------
# CurationRun：一次整理任务的运行目录与三段式编排
# ---------------------------------------------------------------------------
class CurationRun:
    """results/curation/<run_id>/ 的对象化封装。

    manifest.json 记录每个录音的状态（pending/processing/completed/
    failed/duplicate），保证关机/崩溃后可以继续，completed 直接跳过。
    """

    def __init__(
        self,
        run_dir: Path,
        metadata: dict,
        manifest: dict,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.metadata = metadata
        self.manifest = manifest

    # ---------------- 目录与文件 ----------------
    @property
    def tables_dir(self) -> Path:
        return self.run_dir / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.run_dir / "figures"

    @property
    def clusters_dir(self) -> Path:
        return self.run_dir / "clusters"

    @property
    def audio_dir(self) -> Path:
        return self.run_dir / "audio"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / MANIFEST_NAME

    @property
    def metadata_path(self) -> Path:
        return self.run_dir / METADATA_NAME

    @property
    def files(self) -> list[dict]:
        return self.manifest.setdefault("files", [])

    def save_manifest(self) -> None:
        atomic_write_json(self.manifest_path, self.manifest)

    def save_metadata(self) -> None:
        atomic_write_json(self.metadata_path, self.metadata)

    # ---------------- 创建 / 打开 ----------------
    @classmethod
    def create(
        cls,
        input_path: Path,
        base_dir: Path | None = None,
        run_id: str | None = None,
    ) -> "CurationRun":
        input_path = Path(input_path).expanduser()
        if not input_path.exists():
            raise FileNotFoundError(f"输入路径不存在：{input_path}")

        base = Path(base_dir) if base_dir else CURATION_DIR
        base.mkdir(parents=True, exist_ok=True)
        if not run_id:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base / run_id
        suffix = 1
        while run_dir.exists():
            suffix += 1
            run_dir = base / f"{run_id}_{suffix}"
        run_id = run_dir.name

        for directory in (
            run_dir,
            run_dir / "tables",
            run_dir / "figures",
            run_dir / "clusters",
            run_dir / "audio" / "accepted",
            run_dir / "audio" / "rejected",
            run_dir / "audio" / "separated",
        ):
            directory.mkdir(parents=True, exist_ok=True)

        now = _now()
        metadata = {
            "run_id": run_id,
            "created_at": now,
            "input_path": str(input_path),
            "file_count": 0,
            "unique_recording_count": 0,
            "duplicate_count": 0,
            "failed_file_count": 0,
            "candidate_pulses": 0,
            "accepted_pulses": 0,
            "rejected_pulses": 0,
            "cluster_count": None,
            "selected_k": None,
            "silhouette_score": None,
            "random_state": RANDOM_STATE,
            "representatives_per_strategy": None,
            "params": None,
            "software_version": _software_version(),
            "warning": (
                "Cluster 只是声学相似组，不等于物种，也不等于 A-D 声型；"
                "同一物种可能落入多个 cluster，同一 cluster 也可能包含多个物种；"
                "真实物种标签只能来自专家人工鉴定。"
            ),
        }
        manifest = {
            "run_id": run_id,
            "created_at": now,
            "input_path": str(input_path),
            "files": [],
        }
        run = cls(run_dir, metadata, manifest)
        run.save_manifest()
        run.save_metadata()
        return run

    @classmethod
    def open(cls, run_dir: Path) -> "CurationRun":
        run_dir = Path(run_dir)
        manifest_path = run_dir / MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"不是有效的整理任务目录（缺少 manifest）：{run_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata_path = run_dir / METADATA_NAME
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {"run_id": run_dir.name}
        )
        return cls(run_dir, metadata, manifest)

    @staticmethod
    def list_runs(base_dir: Path | None = None) -> list[Path]:
        base = Path(base_dir) if base_dir else CURATION_DIR
        if not base.exists():
            return []
        return sorted(
            (
                directory
                for directory in base.iterdir()
                if directory.is_dir()
                and (directory / MANIFEST_NAME).exists()
            ),
            key=lambda directory: directory.name,
            reverse=True,
        )

    # ---------------- Stage 1：扫描 ----------------
    def stage_scan(self) -> int:
        """扫描输入并把新发现的 WAV 追加为 pending；返回扫描到的总数。"""
        input_path = Path(self.metadata["input_path"])
        if input_path.is_file():
            wav_files = [input_path] if input_path.suffix.lower() == ".wav" else []
            root = input_path.parent
        elif input_path.is_dir():
            wav_files = find_wav_files(input_path)
            root = input_path
        else:
            raise FileNotFoundError(f"输入路径不存在：{input_path}")

        known = {
            str(entry.get("source_path", "")): entry for entry in self.files
        }
        for wav_path in wav_files:
            key = str(wav_path)
            if key in known:
                continue
            try:
                relative = str(wav_path.relative_to(root))
            except ValueError:
                relative = wav_path.name
            self.files.append(
                {
                    "source_file": wav_path.name,
                    "relative_path": relative,
                    "source_path": key,
                    "status": STATUS_PENDING,
                    "audio_hash": "",
                    "duration_s": 0.0,
                    "sample_rate": 0,
                    "duplicate_of": "",
                    "error": "",
                    "raw_candidate_count": 0,
                    "refined_candidate_count": 0,
                    "pulse_row_count": 0,
                    "accepted_pulses": 0,
                    "rejected_pulses": 0,
                    "updated_at": _now(),
                }
            )
        self.metadata["file_count"] = len(self.files)
        self.save_manifest()
        self.save_metadata()
        return len(wav_files)

    # ---------------- Stage 2：检测 + 质量筛选 + 切割 ----------------
    def stage_process(
        self,
        params: CurationParams | None = None,
        progress: ProgressCallback | None = None,
        stats: StatsCallback | None = None,
    ) -> dict:
        """逐录音处理；一条失败只记录该条，不影响整批。可重复调用续跑。"""
        params = (params or CurationParams()).validate()
        self.stage_scan()

        runnable = [
            entry
            for entry in self.files
            if entry.get("status") in RUNNABLE_STATUSES
            or (params.retry_failed and entry.get("status") == STATUS_FAILED)
        ]
        total = len(runnable)

        # 续跑一致性：待处理文件的旧结果行先删除，避免重复/过期行。
        all_pulses_path = self.tables_dir / ALL_PULSES_CSV
        existing = read_table(all_pulses_path, key_column="pulse_id")
        if not existing.empty and runnable:
            pending_sources = {
                str(entry.get("source_path", "")) for entry in runnable
            }
            if "source_path" in existing.columns:
                pruned = existing[
                    ~existing["source_path"].astype(str).isin(pending_sources)
                ]
                if len(pruned) != len(existing):
                    atomic_write_csv(pruned, all_pulses_path)

        # 已完成录音的哈希（跨进程恢复去重依据）
        seen_hashes: dict[str, str] = {}
        for entry in self.files:
            if entry.get("status") == STATUS_COMPLETED:
                audio_hash = str(entry.get("audio_hash") or "")
                if audio_hash and audio_hash not in seen_hashes:
                    seen_hashes[audio_hash] = str(
                        entry.get("relative_path") or entry.get("source_path", "")
                    )

        for index, entry in enumerate(runnable, start=1):
            entry["status"] = STATUS_PROCESSING
            self._process_one(entry, params, seen_hashes)

            if progress is not None:
                progress(index, total)
            if stats is not None:
                stats(self.stats_payload(current=entry.get("relative_path", "")))
            if index % MANIFEST_FLUSH_EVERY == 0:
                self.save_manifest()

        self.save_manifest()
        self._write_recordings_csv()
        self._write_split_tables()
        self._update_metadata_counts()
        self.save_metadata()
        if stats is not None:
            stats(self.stats_payload(current=""))
        return self.stats_payload(current="")

    def _process_one(
        self,
        entry: dict,
        params: CurationParams,
        seen_hashes: dict[str, str],
    ) -> None:
        """处理单个录音（成功后直接把结果行追加到 all_pulses.csv）。

        任何异常都只影响本条（failed），不中断整批。"""
        source_path = Path(str(entry["source_path"]))
        y: np.ndarray | None = None
        try:
            if not source_path.exists():
                raise FileNotFoundError("文件不存在（可能已被移动或删除）")
            y, sr = load_audio(source_path)
            audio_hash = calculate_audio_content_hash(y, sr)
            entry["audio_hash"] = audio_hash
            entry["duration_s"] = round(len(y) / sr, 3)
            entry["sample_rate"] = int(sr)

            if audio_hash in seen_hashes:
                # 声音内容去重：文件名/元数据不同但 PCM 相同也算重复。
                entry["status"] = STATUS_DUPLICATE
                entry["duplicate_of"] = seen_hashes[audio_hash]
                entry["error"] = ""
                entry["updated_at"] = _now()
                return

            (
                raw_candidate_count,
                refined_pulses,
                _,
            ) = detect_and_refine_pulses_chunked(
                y,
                sr,
                max_windows=analysis_max_windows(),
            )
            rows = create_demo_pulse_rows(
                source_path,
                y,
                sr,
                refined_pulses,
                separated_dir=self.audio_dir / "separated",
            )

            prefix = audio_hash[:16]
            kept: list[dict] = []
            accepted_count = 0
            for pulse_index, row in enumerate(rows, start=1):
                pulse_id = f"{prefix}_p{pulse_index:06d}"
                accepted = bool(row["accepted"])
                row["pulse_id"] = pulse_id
                row["audio_hash"] = audio_hash
                row["source_file"] = entry["relative_path"]
                row["source_path"] = str(source_path)
                row["group_id"] = entry["relative_path"]
                row["temporary_label"] = ""
                row["pulse_file"] = self._store_pulse_audio(
                    row,
                    y,
                    sr,
                    accepted=accepted,
                    params=params,
                )
                kept.append(row)
                accepted_count += int(accepted)

            if kept:
                append_rows(pd.DataFrame(kept), self.tables_dir / ALL_PULSES_CSV)

            entry["status"] = STATUS_COMPLETED
            entry["error"] = ""
            entry["raw_candidate_count"] = int(raw_candidate_count)
            entry["refined_candidate_count"] = int(len(refined_pulses))
            entry["pulse_row_count"] = int(len(kept))
            entry["accepted_pulses"] = int(accepted_count)
            entry["rejected_pulses"] = int(len(kept) - accepted_count)
            entry["updated_at"] = _now()
            seen_hashes.setdefault(audio_hash, entry["relative_path"])
        except Exception as error:  # noqa: BLE001 - 单文件失败不能毁掉整批
            entry["status"] = STATUS_FAILED
            entry["error"] = f"{type(error).__name__}: {error}"
            entry["updated_at"] = _now()
        finally:
            y = None  # 及时释放音频，避免整批 WAV 常驻内存

    def _store_pulse_audio(
        self,
        row: dict,
        y: np.ndarray,
        sr: int,
        accepted: bool,
        params: CurationParams,
    ) -> str:
        """把一个脉冲写到 run 的 audio/ 目录，返回相对 run 的路径。

        分离分量必须沿用已经分离出来的 WAV（绝不从混合录音重切），
        未分离的普通脉冲才按 start_s/end_s 从源波形切割。
        """
        existing = str(row.get("pulse_file", "") or "")
        existing_path = resolve_run_path(existing, self.run_dir) if existing else None

        want_audio = params.save_pulse_audio if accepted else params.save_rejected_audio
        folder = "accepted" if accepted else "rejected"

        if existing_path is not None:
            # 分离得到的真实分量：沿用已分离 WAV，绝不从混合录音重切。
            if not want_audio:
                return run_relative(existing, self.run_dir)
            destination = self.audio_dir / folder / f"{row['pulse_id']}.wav"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.resolve() != existing_path.resolve():
                shutil.copy2(existing_path, destination)
            return run_relative(str(destination), self.run_dir)

        if not want_audio:
            # 未写盘：留空，代表样本导出时按需从源录音重切（未分离脉冲安全）。
            return ""

        destination = self.audio_dir / folder / f"{row['pulse_id']}.wav"
        destination.parent.mkdir(parents=True, exist_ok=True)
        start_sample = max(0, int(round(float(row["start_s"]) * sr)))
        end_sample = min(len(y), int(round(float(row["end_s"]) * sr)))
        segment = y[start_sample:end_sample]
        if len(segment) < 32:
            return ""
        sf.write(destination, segment.astype(np.float32), sr, subtype="PCM_16")
        return run_relative(str(destination), self.run_dir)

    # ---------------- 统计与落盘 ----------------
    def stats_payload(self, current: str = "") -> dict:
        files = self.files
        status_counts: dict[str, int] = {}
        for entry in files:
            status = str(entry.get("status", STATUS_PENDING))
            status_counts[status] = status_counts.get(status, 0) + 1

        def _sum(key: str) -> int:
            return int(sum(int(entry.get(key, 0) or 0) for entry in files))

        return {
            "current_file": current,
            "total_files": len(files),
            "completed_files": status_counts.get(STATUS_COMPLETED, 0),
            "pending_files": (
                status_counts.get(STATUS_PENDING, 0)
                + status_counts.get(STATUS_PROCESSING, 0)
            ),
            "duplicate_files": status_counts.get(STATUS_DUPLICATE, 0),
            "failed_files": status_counts.get(STATUS_FAILED, 0),
            "candidate_pulses": _sum("pulse_row_count"),
            "accepted_pulses": _sum("accepted_pulses"),
            "rejected_pulses": _sum("rejected_pulses"),
        }

    def _write_recordings_csv(self) -> None:
        if not self.files:
            return
        frame = pd.DataFrame(self.files)
        atomic_write_csv(frame, self.tables_dir / RECORDINGS_CSV)

    def _write_split_tables(self) -> None:
        """把 accepted / rejected 分别落盘（rejected 永远保留，不静默删除）。"""
        all_pulses_path = self.tables_dir / ALL_PULSES_CSV
        if not all_pulses_path.exists():
            return
        from ..features import accepted_mask

        pulses = read_table(all_pulses_path, key_column="pulse_id")
        if "accepted" not in pulses.columns:
            return
        mask = accepted_mask(pulses["accepted"])
        atomic_write_csv(pulses.loc[mask], self.tables_dir / ACCEPTED_PULSES_CSV)
        atomic_write_csv(pulses.loc[~mask], self.tables_dir / REJECTED_PULSES_CSV)

    def _update_metadata_counts(self) -> None:
        payload = self.stats_payload()
        self.metadata.update(
            {
                "file_count": payload["total_files"],
                "unique_recording_count": payload["completed_files"],
                "duplicate_count": payload["duplicate_files"],
                "failed_file_count": payload["failed_files"],
                "candidate_pulses": payload["candidate_pulses"],
                "accepted_pulses": payload["accepted_pulses"],
                "rejected_pulses": payload["rejected_pulses"],
                # 科研复现：本任务实际使用的分析跨度
                "analysis_span_s": float(CURATION_ANALYSIS_SPAN_S),
                "analysis_windows_per_recording": analysis_max_windows(),
            }
        )

    # ---------------- Stage 3：聚类与代表样本 ----------------
    def stage_cluster(
        self,
        params: CurationParams | None = None,
        stats: StatsCallback | None = None,
    ) -> dict:
        """聚类 + 代表样本 + 图表 + 审核数据初始化。失败返回 ok=False。"""
        from .clustering import cluster_run

        params = (params or CurationParams()).validate()
        result = cluster_run(self, params, stats=stats)
        if result.get("ok"):
            self._update_metadata_counts()
            self.metadata["params"] = params.to_dict()
            self.metadata["random_state"] = params.random_state
            self.save_metadata()
        return result

    def stage_export(self, include_noise: bool = False) -> Path:
        """导出专家标注监督训练数据集 CSV。"""
        from .annotations import export_training_dataset

        return export_training_dataset(
            self.run_dir,
            include_noise=include_noise,
        )

    # ---------------- 审核摘要 ----------------
    def review_summary(self) -> dict:
        """打开已有 run 时给 UI 的概览：代表样本/已审核/待审核。"""
        from .annotations import AnnotationStore

        representatives = read_table(
            self.tables_dir / "representatives.csv",
            key_column="pulse_id",
        )
        store = AnnotationStore.load(self.run_dir)
        store.initialize_from_representatives(representatives, save=False)

        total = len(representatives)
        status_counts = (
            store.to_frame()["review_status"].value_counts().to_dict()
            if not store.frame.empty
            else {}
        )
        reviewed = int(status_counts.get("reviewed", 0))
        pending = int(status_counts.get("pending", 0))
        return {
            "run_dir": str(self.run_dir),
            "run_id": self.manifest.get("run_id", self.run_dir.name),
            "representative_count": total,
            "reviewed_count": reviewed,
            "pending_count": pending,
            "uncertain_count": int(status_counts.get("uncertain", 0)),
            "noise_count": int(status_counts.get("noise", 0)),
            "processing": self.stats_payload(),
            "clustered": (self.tables_dir / "clustered_pulses.csv").exists(),
        }


def create_run(input_path: Path, base_dir: Path | None = None) -> CurationRun:
    """便捷函数：创建并扫描一个新任务。"""
    run = CurationRun.create(input_path, base_dir=base_dir)
    run.stage_scan()
    return run


def open_run(run_dir: Path) -> CurationRun:
    return CurationRun.open(run_dir)
