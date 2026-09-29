"""批量整理与专家标注（⑥ curation）算法层单元/集成测试。

重点覆盖：扫描、内容去重、质量筛选保留、K 自动选择边界、
四类代表样本抽样与去重、标注持久化、断点续跑、导出过滤。
不测 Qt 像素布局。
"""

from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bat_classifier.curation.annotations import (  # noqa: E402
    AnnotationStore,
    export_training_dataset,
)
from bat_classifier.curation.clustering import (  # noqa: E402
    candidate_k_values,
    evaluate_k,
    prepare_cluster_frame,
    select_best_k,
)
from bat_classifier.curation.pipeline import (  # noqa: E402
    STATUS_COMPLETED,
    STATUS_DUPLICATE,
    STATUS_FAILED,
    CurationParams,
    CurationRun,
    analysis_max_windows,
    atomic_write_csv,
)
from bat_classifier.detection import detect_and_refine_pulses_chunked  # noqa: E402
from bat_classifier.curation.representatives import (  # noqa: E402
    select_representatives,
)
from bat_classifier.config import (  # noqa: E402
    CLUSTER_FEATURE_COLUMNS,
    CURATION_ANALYSIS_SPAN_S,
    CURATION_AUTO_K_MAX,
    CURATION_K_MAX,
    DETECTION_CHUNK_SECONDS,
)
from bat_classifier.features import evaluate_pulse_quality  # noqa: E402

SR = 250_000


# ---------------------------------------------------------------------------
# 测试工具
# ---------------------------------------------------------------------------
def _sweep(start_khz: float, end_khz: float, duration_s: float = 0.005,
           amplitude: float = 0.8) -> np.ndarray:
    samples = int(SR * duration_s)
    frequencies = np.linspace(start_khz * 1000, end_khz * 1000, samples)
    phase = 2 * np.pi * np.cumsum(frequencies) / SR
    return (amplitude * np.sin(phase)).astype(np.float32)


def _recording(*sweeps: np.ndarray, length_s: float = 0.4,
               seed: int = 0) -> np.ndarray:
    """把若干脉冲摆进一段静音（可选微弱底噪）里。"""
    total = np.zeros(int(SR * length_s), dtype=np.float32)
    offsets = np.linspace(0.03, length_s - 0.05, max(len(sweeps), 1))
    for offset, sweep in zip(offsets, sweeps):
        start = int(offset * SR)
        end = min(len(total), start + len(sweep))
        total[start:end] = sweep[: end - start]
    if seed:
        rng = np.random.default_rng(seed)
        total += (rng.uniform(-0.01, 0.01, size=len(total))).astype(np.float32)
    return total


def _write(path: Path, audio: np.ndarray, sr: int = SR) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr, subtype="PCM_16")
    return path


def _write_with_extra_metadata(source: Path, destination: Path) -> Path:
    """同样 PCM，但在 RIFF 头里插入一段 LIST/INFO 元数据（元数据不同）。"""
    data = source.read_bytes()
    payload = b"INFO" + b"ISFT" + struct.pack("<I", 4) + b"meta"
    chunk = b"LIST" + struct.pack("<I", len(payload)) + payload
    patched = (
        data[:12]
        + chunk
        + data[12:]
    )
    size_field = struct.pack("<I", len(patched) - 8)
    patched = patched[:4] + size_field + patched[8:]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(patched)
    return destination


def _make_input_dir(root: Path) -> Path:
    """3 个互不相同的录音，含递归子目录与大小写混用的扩展名。"""
    input_dir = root / "input"
    _write(input_dir / "a.wav", _recording(_sweep(80, 40), _sweep(60, 35)))
    _write(input_dir / "B.WAV", _recording(_sweep(95, 55), _sweep(70, 45)))
    _write(
        input_dir / "sub" / "c.wav",
        _recording(_sweep(110, 60), _sweep(50, 30), seed=7),
    )
    (input_dir / "not_audio.txt").write_text("ignore me", encoding="utf-8")
    return input_dir


def _representative_frame() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """构造 2 个 Cluster、5 个 pulse 的合成代表样本抽样数据。"""
    rows = [
        # cluster 001（raw 0，质心 (0,0)）
        ("p_center", 0, "001", (0.0, 0.0)),
        ("p_near", 0, "001", (1.0, 0.0)),
        ("p_boundary", 0, "001", (7.0, 0.0)),
        ("p_outlier", 0, "001", (0.0, 9.0)),
        # cluster 002（raw 1，质心 (10,0)）：只有 1 个点 → 四种策略全命中
        ("p_alone", 1, "002", (10.0, 0.0)),
    ]
    frame = pd.DataFrame(
        {
            "pulse_id": [row[0] for row in rows],
            "raw_cluster": [row[1] for row in rows],
            "cluster_id": [row[2] for row in rows],
            "source_file": ["rec.wav"] * len(rows),
            "source_path": [""] * len(rows),
            "audio_hash": ["abc"] * len(rows),
            "start_s": [0.0] * len(rows),
            "end_s": [0.005] * len(rows),
            "duration_ms": [5.0] * len(rows),
            "snr_db": [20.0] * len(rows),
        }
    )
    for column in CLUSTER_FEATURE_COLUMNS:
        frame[column] = 0.0

    scaled = np.array([row[3] for row in rows], dtype=float)
    centroids = np.array([[0.0, 0.0], [10.0, 0.0]])
    id_mapping = {0: "001", 1: "002"}
    return frame, scaled, centroids, id_mapping


# ---------------------------------------------------------------------------
# Stage 1：扫描
# ---------------------------------------------------------------------------
class ScanTest(unittest.TestCase):
    def test_scan_single_wav(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav = _write(root / "single.wav", _recording(_sweep(80, 40)))
            run = CurationRun.create(wav, base_dir=root / "runs")
            total = run.stage_scan()
            self.assertEqual(total, 1)
            self.assertEqual(len(run.files), 1)
            self.assertEqual(run.files[0]["source_file"], "single.wav")
            self.assertEqual(run.files[0]["status"], "pending")

    def test_scan_directory_recursive_and_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = _make_input_dir(root)
            run = CurationRun.create(input_dir, base_dir=root / "runs")
            total = run.stage_scan()
            self.assertEqual(total, 3)  # .wav/.WAV/子目录，忽略 txt
            names = {entry["source_file"] for entry in run.files}
            self.assertEqual(names, {"a.wav", "B.WAV", "c.wav"})
            relative = {
                entry["relative_path"] for entry in run.files
            }
            self.assertIn(str(Path("sub") / "c.wav"), relative)
            # 重复扫描不产生重复条目
            run.stage_scan()
            self.assertEqual(len(run.files), 3)

    def test_corrupt_wav_marks_failed_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            _write(input_dir / "good.wav", _recording(_sweep(80, 40)))
            (input_dir / "broken.wav").write_bytes(b"RIFF0000NOTAWAVE")

            run = CurationRun.create(input_dir, base_dir=root / "runs")
            stats = run.stage_process(CurationParams())

            statuses = {
                entry["source_file"]: entry["status"]
                for entry in run.files
            }
            self.assertEqual(statuses["good.wav"], STATUS_COMPLETED)
            self.assertEqual(statuses["broken.wav"], STATUS_FAILED)
            broken = next(
                entry for entry in run.files
                if entry["source_file"] == "broken.wav"
            )
            self.assertTrue(broken["error"])
            # 单条失败不影响整批：好文件照样出结果
            self.assertGreaterEqual(stats["completed_files"], 1)
            self.assertEqual(stats["failed_files"], 1)


# ---------------------------------------------------------------------------
# 内容去重（相同 PCM、不同元数据）
# ---------------------------------------------------------------------------
class DuplicateTest(unittest.TestCase):
    def test_same_pcm_different_metadata_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            original = _write(
                input_dir / "first.wav",
                _recording(_sweep(80, 40), _sweep(55, 30)),
            )
            _write_with_extra_metadata(original, input_dir / "second.wav")

            run = CurationRun.create(input_dir, base_dir=root / "runs")
            stats = run.stage_process(CurationParams())

            statuses = {
                entry["source_file"]: entry["status"]
                for entry in run.files
            }
            self.assertEqual(statuses["first.wav"], STATUS_COMPLETED)
            self.assertEqual(statuses["second.wav"], STATUS_DUPLICATE)
            duplicate = next(
                entry for entry in run.files
                if entry["source_file"] == "second.wav"
            )
            self.assertEqual(duplicate["duplicate_of"], "first.wav")
            self.assertEqual(stats["duplicate_files"], 1)
            # 重复文件不再产生脉冲结果
            pulses = pd.read_csv(
                run.tables_dir / "all_pulses.csv", encoding="utf-8-sig"
            )
            self.assertFalse(
                (pulses["source_file"] == "second.wav").any()
            )


# ---------------------------------------------------------------------------
# Stage 2：质量筛选 accepted/rejected 都保留
# ---------------------------------------------------------------------------
class QualityRetentionTest(unittest.TestCase):
    def _process(self, root: Path):
        input_dir = root / "input"
        sweep_audio = _recording(
            _sweep(80, 40), _sweep(62, 34), _sweep(95, 60)
        )
        rng = np.random.default_rng(3)
        noise_burst = (
            rng.uniform(-1.0, 1.0, size=int(SR * 0.008)) * 0.9
        ).astype(np.float32)
        mixed = sweep_audio.copy()
        position = int(0.32 * SR)
        mixed[position : position + len(noise_burst)] = noise_burst
        _write(input_dir / "mix.wav", mixed)

        run = CurationRun.create(input_dir, base_dir=root / "runs")
        # 硬性隔离：绝不允许触发 data/raw 清空逻辑。
        # 同时关闭实验性同时声波分离，让宽带噪声作为单一行进入质量筛选
        # （分离开启时它会被拆成若干条窄带分量，本测试关注的是
        #  accepted/rejected 的保留与 rejection_reason，见 EndToEndTest 覆盖分离链路）。
        with mock.patch(
            "bat_classifier.pulse_table._clear_raw_data_dir",
            side_effect=AssertionError("curation 不允许清空 data/raw"),
        ), mock.patch(
            "bat_classifier.separation.ENABLE_SIMULTANEOUS_SEPARATION", False
        ):
            stats = run.stage_process(CurationParams())
        return run, stats

    def test_accepted_and_rejected_both_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run, stats = self._process(Path(tmp))
            pulses = pd.read_csv(
                run.tables_dir / "all_pulses.csv", encoding="utf-8-sig"
            )
            self.assertFalse(pulses.empty)
            self.assertTrue(pulses["pulse_id"].is_unique)

            accepted = pulses["accepted"].astype(str).str.lower().isin(
                {"true", "1", "yes"}
            )
            # 至少有合格脉冲（干净扫频），也必须保留被拒绝的候选
            self.assertGreater(int(accepted.sum()), 0)
            self.assertGreater(int((~accepted).sum()), 0)

            rejected_rows = pulses.loc[~accepted]
            self.assertTrue(
                rejected_rows["rejection_reason"].fillna("").astype(str)
                .str.len().gt(0).all(),
                "被拒绝的脉冲必须带 rejection_reason",
            )
            # 汇总计数与逐行数据一致
            self.assertEqual(
                stats["accepted_pulses"], int(accepted.sum())
            )
            self.assertEqual(
                stats["rejected_pulses"], int((~accepted).sum())
            )
            # manifest 与 CSV 行数一致（rejected 未被静默删除）
            self.assertEqual(
                stats["candidate_pulses"], int(len(pulses))
            )

    def test_quality_decision_is_reproducible_from_row(self) -> None:
        """每一行的 accepted 都能用现有 evaluate_pulse_quality 复算出来。"""
        with tempfile.TemporaryDirectory() as tmp:
            run, _ = self._process(Path(tmp))
            pulses = pd.read_csv(
                run.tables_dir / "all_pulses.csv", encoding="utf-8-sig"
            )
            for _, row in pulses.iterrows():
                payload = row.to_dict()
                accepted, reason = evaluate_pulse_quality(payload, payload)
                stored = str(payload["accepted"]).lower() in {"true", "1", "yes"}
                self.assertEqual(accepted, stored)
                stored_reason = payload.get("rejection_reason", "")
                stored_reason = (
                    "" if pd.isna(stored_reason) else str(stored_reason)
                )
                self.assertEqual(reason, stored_reason)


# ---------------------------------------------------------------------------
# Stage 4：K 自动选择
# ---------------------------------------------------------------------------
class KAutoTest(unittest.TestCase):
    def _scaled(self, n: int, dim: int = 4) -> np.ndarray:
        return np.random.default_rng(0).normal(size=(n, dim))

    def test_auto_k_never_exceeds_sample_count(self) -> None:
        scaled = self._scaled(5)
        params = CurationParams(k_mode="auto")
        candidates = candidate_k_values(scaled, params)
        self.assertTrue(candidates)
        self.assertLessEqual(max(candidates), len(scaled) - 1)
        self.assertLessEqual(max(candidates), CURATION_AUTO_K_MAX)
        self.assertGreaterEqual(min(candidates), 2)

        records = [evaluate_k(scaled, k) for k in candidates]
        best = select_best_k(records)
        self.assertIn(best, candidates)

    def test_manual_k_is_clamped_to_sample_count(self) -> None:
        scaled = self._scaled(6)
        # 手动 K 拉到上限（CURATION_K_MAX）也不会超过样本数-1
        params = CurationParams(k_mode="manual", k=CURATION_K_MAX)
        candidates = candidate_k_values(scaled, params)
        self.assertEqual(candidates, [5])  # 不会 K >= n
        # 样本足够时，上限以内的 K 原样生效
        params = CurationParams(k_mode="manual", k=20)
        self.assertEqual(candidate_k_values(self._scaled(26), params), [20])

    def test_single_k_failure_does_not_crash_batch(self) -> None:
        from bat_classifier.curation import clustering as cur_clustering
        from sklearn.cluster import KMeans as RealKMeans

        scaled = self._scaled(40)

        def flaky_kmeans(*args, **kwargs):
            if int(kwargs.get("n_clusters", 0)) == 3:
                raise RuntimeError("模拟 K=3 拟合失败")
            return RealKMeans(*args, **kwargs)

        params = CurationParams(k_mode="auto")
        with mock.patch.object(
            cur_clustering, "KMeans", side_effect=flaky_kmeans
        ):
            records = [
                evaluate_k(scaled, k) for k in candidate_k_values(scaled, params)
            ]
        failed = [record for record in records if record["error"]]
        self.assertTrue(failed and all(record["k"] == 3 for record in failed))
        best = select_best_k(records)
        self.assertIsNotNone(best)
        self.assertNotEqual(best, 3)

    def test_too_few_samples_returns_empty_candidates(self) -> None:
        params = CurationParams(k_mode="auto")
        self.assertEqual(candidate_k_values(self._scaled(1), params), [])


# ---------------------------------------------------------------------------
# 代表样本抽样：center / boundary / random / outlier 与去重
# ---------------------------------------------------------------------------
class RepresentativeSamplingTest(unittest.TestCase):
    def _select(self, **overrides):
        frame, scaled, centroids, id_mapping = _representative_frame()
        defaults = dict(
            n_center=1, n_boundary=1, n_random=1, n_outlier=1,
            k_mode="auto", k=4,
        )
        defaults.update(overrides)
        params = CurationParams(**defaults)
        representatives = select_representatives(
            frame, scaled, centroids, id_mapping, params
        )
        return frame, scaled, representatives

    def test_four_strategies_pick_expected_pulses(self) -> None:
        _, _, representatives = self._select()
        reasons: dict[str, set[str]] = {}
        for _, row in representatives.iterrows():
            for reason in str(row["selection_reason"]).split("|"):
                reasons.setdefault(row["pulse_id"], set()).add(reason)

        # cluster 001：中心 / 边界 / 离群各就各位
        self.assertIn("center", reasons["p_center"])
        self.assertIn("boundary", reasons["p_boundary"])
        self.assertIn("outlier", reasons["p_outlier"])
        # 随机抽样只会落在本 Cluster 的真实样本里（每簇抽 n_random 个）
        cluster_one = set(
            representatives[representatives["cluster_id"] == "001"]["pulse_id"]
        )
        cluster_one_reasons: set[str] = set()
        for pulse_id in cluster_one:
            cluster_one_reasons |= reasons[pulse_id]
        self.assertIn("random", cluster_one_reasons)

        # 距离字段正确：center 距离最小，outlier 距离最大
        cluster_rows = representatives[
            representatives["cluster_id"] == "001"
        ].set_index("pulse_id")
        self.assertEqual(
            cluster_rows["distance_to_center"].idxmin(), "p_center"
        )
        self.assertEqual(
            cluster_rows["distance_to_center"].idxmax(), "p_outlier"
        )
        # boundary 的 margin 最小
        self.assertEqual(
            cluster_rows["boundary_margin"].idxmin(), "p_boundary"
        )

    def test_single_pulse_cluster_keeps_one_row_with_all_reasons(self) -> None:
        _, _, representatives = self._select()
        alone = representatives[representatives["pulse_id"] == "p_alone"]
        self.assertEqual(len(alone), 1, "同一 pulse 不允许复制多行")
        self.assertEqual(
            alone.iloc[0]["selection_reason"],
            "center|boundary|random|outlier",
        )

    def test_no_duplicate_pulse_rows(self) -> None:
        _, _, representatives = self._select()
        self.assertTrue(representatives["pulse_id"].is_unique)

    def test_random_sampling_is_reproducible(self) -> None:
        _, _, first = self._select(n_center=0, n_boundary=0, n_outlier=0,
                                   n_random=2, random_state=42)
        _, _, second = self._select(n_center=0, n_boundary=0, n_outlier=0,
                                    n_random=2, random_state=42)
        self.assertEqual(
            sorted(first["pulse_id"]), sorted(second["pulse_id"])
        )
        _, _, third = self._select(n_center=0, n_boundary=0, n_outlier=0,
                                   n_random=2, random_state=7)
        # 形状正确：cluster 001 抽 2 个、单点 cluster 002 抽 1 个
        self.assertEqual(
            len(third[third["cluster_id"] == "001"]), 2
        )
        self.assertEqual(
            len(third[third["cluster_id"] == "002"]), 1
        )

    def test_exclude_supports_adding_samples(self) -> None:
        frame, scaled, centroids, id_mapping = _representative_frame()
        params = CurationParams(
            n_center=1, n_boundary=1, n_random=0, n_outlier=1
        )
        first = select_representatives(
            frame, scaled, centroids, id_mapping, params
        )
        excluded = set(first["pulse_id"])
        # cluster 001 只剩 1 个未选点；追加抽样必须跳过已选的
        second = select_representatives(
            frame, scaled, centroids, id_mapping, params,
            exclude_pulse_ids=excluded,
        )
        self.assertTrue(set(second["pulse_id"]).isdisjoint(excluded))
        cluster_one = second[second["cluster_id"] == "001"]
        self.assertEqual(list(cluster_one["pulse_id"]), ["p_near"])


# ---------------------------------------------------------------------------
# 标注持久化
# ---------------------------------------------------------------------------
class AnnotationPersistenceTest(unittest.TestCase):
    def _store_with_reps(self, run_dir: Path) -> AnnotationStore:
        (run_dir / "tables").mkdir(parents=True, exist_ok=True)
        reps = pd.DataFrame(
            [
                {
                    "pulse_id": "h1_p000001",
                    "cluster_id": "001",
                    "selection_reason": "center",
                    "source_file": "a.wav",
                    "source_path": "C:/x/a.wav",
                },
                {
                    "pulse_id": "h1_p000002",
                    "cluster_id": "001",
                    "selection_reason": "boundary",
                    "source_file": "a.wav",
                    "source_path": "C:/x/a.wav",
                },
            ]
        )
        atomic_write_csv(reps, run_dir / "tables" / "representatives.csv")
        store = AnnotationStore.load(run_dir)
        store.initialize_from_representatives(reps)
        return store

    def test_save_and_reload_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            store = self._store_with_reps(run_dir)
            self.assertEqual(store.counts()["pending"], 2)

            store.update(
                "h1_p000001",
                expert_label="Pipistrellus abramus",
                expert_confidence="high",
                review_status="reviewed",
                notes="clear FM sweep",
            )

            reloaded = AnnotationStore.load(run_dir)
            row = reloaded.to_frame().set_index("pulse_id").loc["h1_p000001"]
            self.assertEqual(row["expert_label"], "Pipistrellus abramus")
            self.assertEqual(row["expert_confidence"], "high")
            self.assertEqual(row["review_status"], "reviewed")
            self.assertEqual(row["notes"], "clear FM sweep")
            self.assertTrue(row["reviewed_at"])

            untouched = reloaded.to_frame().set_index("pulse_id").loc[
                "h1_p000002"
            ]
            self.assertEqual(untouched["review_status"], "pending")
            self.assertEqual(untouched["expert_label"], "")

    def test_reinit_keeps_expert_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            store = self._store_with_reps(run_dir)
            store.update("h1_p000002", "Myotis sp.", "low", "uncertain", "?")

            reps = pd.read_csv(
                run_dir / "tables" / "representatives.csv",
                encoding="utf-8-sig",
                dtype={"cluster_id": str},
            )
            store.initialize_from_representatives(reps)
            row = store.to_frame().set_index("pulse_id").loc["h1_p000002"]
            self.assertEqual(row["expert_label"], "Myotis sp.")
            self.assertEqual(row["review_status"], "uncertain")


# ---------------------------------------------------------------------------
# 断点续跑
# ---------------------------------------------------------------------------
class ResumeTest(unittest.TestCase):
    def test_resume_processes_only_remaining_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            for index in range(10):
                # 每条录音内容不同，避免被去重
                _write(
                    input_dir / f"rec_{index:02d}.wav",
                    _recording(_sweep(70 + index, 40 + index)),
                )

            run = CurationRun.create(input_dir, base_dir=root / "runs")
            run.stage_scan()
            self.assertEqual(len(run.files), 10)

            # 模拟“前 5 个已完成”
            for entry in run.files[:5]:
                entry["status"] = STATUS_COMPLETED
                entry["audio_hash"] = f"fake_{entry['source_file']}"
            run.save_manifest()

            calls: list[tuple[int, int]] = []
            run.stage_process(
                CurationParams(),
                progress=lambda current, total: calls.append((current, total)),
            )

            self.assertEqual(len(calls), 5)
            self.assertTrue(all(total == 5 for _current, total in calls))

            # 已完成的 5 个没有被重新处理（标记保持原样）
            for entry in run.files[:5]:
                self.assertEqual(entry["duration_s"], 0.0)
                self.assertEqual(entry["pulse_row_count"], 0)
            for entry in run.files[5:]:
                self.assertEqual(entry["status"], STATUS_COMPLETED)
                self.assertGreater(entry["duration_s"], 0.0)

    def test_failed_files_only_retry_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            _write(input_dir / "ok.wav", _recording(_sweep(80, 40)))
            (input_dir / "bad.wav").write_bytes(b"RIFF0000NOTAWAVE")

            run = CurationRun.create(input_dir, base_dir=root / "runs")
            run.stage_process(CurationParams())
            bad = next(e for e in run.files if e["source_file"] == "bad.wav")
            self.assertEqual(bad["status"], STATUS_FAILED)

            # 默认不重试 failed
            calls: list[int] = []
            run.stage_process(
                CurationParams(),
                progress=lambda current, _total: calls.append(current),
            )
            self.assertEqual(calls, [])

            # 显式要求重试
            run.stage_process(
                CurationParams(retry_failed=True),
                progress=lambda current, _total: calls.append(current),
            )
            self.assertEqual(calls, [1])
            bad = next(e for e in run.files if e["source_file"] == "bad.wav")
            self.assertEqual(bad["status"], STATUS_FAILED)


# ---------------------------------------------------------------------------
# 导出监督训练数据集
# ---------------------------------------------------------------------------
class ExportTest(unittest.TestCase):
    def _build_run(self, run_dir: Path) -> None:
        (run_dir / "tables").mkdir(parents=True, exist_ok=True)
        rows = []
        for index in range(1, 5):
            row = {
                "pulse_id": f"p{index}",
                "cluster_id": "001",
                "selection_reason": "center",
                "source_file": "a.wav",
                "source_path": "C:/data/a.wav",
                "audio_hash": "hash42",
                "start_s": 0.1,
                "end_s": 0.105,
                "duration_ms": 5.0,
                "snr_db": 15.0,
            }
            for column in CLUSTER_FEATURE_COLUMNS:
                row[column] = 1.0
            rows.append(row)
        reps = pd.DataFrame(rows)
        atomic_write_csv(reps, run_dir / "tables" / "representatives.csv")

        store = AnnotationStore.load(run_dir)
        store.initialize_from_representatives(reps)
        # p1：有效已审核 → 导出
        store.update("p1", "Pipistrellus abramus", "high", "reviewed", "ok")
        # p2：仍未审核 → 不导出
        # p3：已审核但标签为空 → 不导出
        store.update("p3", "", "high", "reviewed", "forgot label")
        # p4：噪声 → 默认不导出，开关后导出
        store.update("p4", "", "", "noise", "sniff")

    def test_only_reviewed_with_label_is_exported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            self._build_run(run_dir)
            path = export_training_dataset(run_dir)
            exported = pd.read_csv(path, encoding="utf-8-sig")
            self.assertEqual(list(exported["pulse_id"]), ["p1"])
            # 必须保留按录音分组所需的字段
            for column in (
                "source_file",
                "source_path",
                "audio_hash",
                "start_s",
                "end_s",
                "expert_label",
            ):
                self.assertIn(column, exported.columns)

    def test_noise_can_be_included_on_demand(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            self._build_run(run_dir)
            path = export_training_dataset(run_dir, include_noise=True)
            exported = pd.read_csv(path, encoding="utf-8-sig")
            self.assertEqual(
                sorted(exported["pulse_id"]), ["p1", "p4"]
            )
            noise_row = exported[exported["pulse_id"] == "p4"].iloc[0]
            self.assertEqual(noise_row["expert_label"], "noise")


# ---------------------------------------------------------------------------
# 集成：扫描 → 处理 → 聚类 → 代表样本 → 审核数据
# ---------------------------------------------------------------------------
class EndToEndTest(unittest.TestCase):
    def test_full_run_produces_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = _make_input_dir(root)
            run = CurationRun.create(
                input_dir, base_dir=root / "runs"
            )
            with mock.patch(
                "bat_classifier.pulse_table._clear_raw_data_dir",
                side_effect=AssertionError("curation 不允许清空 data/raw"),
            ):
                stats = run.stage_process(
                    CurationParams(save_pulse_audio=True)
                )
                self.assertGreater(stats["accepted_pulses"], 0)
                result = run.stage_cluster(
                    CurationParams(
                        k_mode="auto",
                        n_center=2,
                        n_boundary=2,
                        n_random=1,
                        n_outlier=1,
                    )
                )

            self.assertTrue(result.get("ok"), result.get("error"))
            self.assertGreaterEqual(result["cluster_count"], 1)
            self.assertGreater(result["representative_count"], 0)

            # 目录结构（run 隔离，不写 data/raw）
            expected_files = [
                "run_metadata.json",
                "manifest.json",
                "tables/recordings.csv",
                "tables/all_pulses.csv",
                "tables/accepted_pulses.csv",
                "tables/rejected_pulses.csv",
                "tables/clustered_pulses.csv",
                "tables/cluster_summary.csv",
                "tables/representatives.csv",
                "tables/expert_annotations.csv",
                "figures/cluster_pca.png",
                "figures/cluster_quality.png",
            ]
            for relative in expected_files:
                self.assertTrue(
                    (run.run_dir / relative).exists(),
                    f"缺少输出：{relative}",
                )
            cluster_dirs = list(run.clusters_dir.glob("cluster_*"))
            self.assertTrue(cluster_dirs, "缺少 cluster 目录")
            self.assertTrue(
                any((cluster_dirs[0] / "audio").glob("*.wav")),
                "代表样本必须导出真实 WAV",
            )
            self.assertTrue(
                any((cluster_dirs[0] / "spectrograms").glob("*.png")),
                "代表样本必须导出声谱图",
            )
            self.assertTrue(
                any((run.audio_dir / "accepted").glob("*.wav")),
                "合格脉冲应切割保存",
            )

            # 代表样本表字段满足审核需求
            representatives = pd.read_csv(
                run.tables_dir / "representatives.csv",
                encoding="utf-8-sig",
                dtype={"cluster_id": str},
            )
            for column in (
                "pulse_id",
                "cluster_id",
                "selection_reason",
                "distance_to_center",
                "second_center_distance",
                "boundary_margin",
                "source_file",
                "source_path",
                "pulse_file",
                "spectrogram_file",
            ):
                self.assertIn(column, representatives.columns)
            self.assertTrue(representatives["pulse_id"].is_unique)

            # 标注表按代表样本初始化为 pending
            annotations = pd.read_csv(
                run.tables_dir / "expert_annotations.csv", encoding="utf-8-sig"
            )
            self.assertEqual(len(annotations), len(representatives))
            self.assertTrue(
                (annotations["review_status"] == "pending").all()
            )

            # Cluster 命名规范：001/002/…，不是 A-D
            cluster_ids = set(representatives["cluster_id"])
            for cluster_id in cluster_ids:
                self.assertTrue(cluster_id.isdigit())
                self.assertEqual(len(cluster_id), 3)

            # run metadata 记录科研复现所需字段
            import json

            metadata = json.loads(
                (run.run_dir / "run_metadata.json").read_text(encoding="utf-8")
            )
            for key in (
                "run_id",
                "created_at",
                "input_path",
                "file_count",
                "unique_recording_count",
                "duplicate_count",
                "candidate_pulses",
                "accepted_pulses",
                "rejected_pulses",
                "cluster_count",
                "silhouette_score",
                "random_state",
                "representatives_per_strategy",
            ):
                self.assertIn(key, metadata)


# ---------------------------------------------------------------------------
# 长录音分析跨度（⑥ 默认 60 秒，①②③ 不变）
# ---------------------------------------------------------------------------
class AnalysisSpanTest(unittest.TestCase):
    def test_span_helper_matches_config(self) -> None:
        import math

        expected = math.ceil(
            CURATION_ANALYSIS_SPAN_S / DETECTION_CHUNK_SECONDS
        )
        self.assertEqual(analysis_max_windows(), expected)
        self.assertGreaterEqual(expected, 15)  # 60s / 4s
        # ①②③ 演示/预测仍然只抽 2 个窗口（约 2×4 秒），互不影响
        from bat_classifier.config import MAX_ANALYSIS_WINDOWS_PER_RECORDING

        self.assertEqual(MAX_ANALYSIS_WINDOWS_PER_RECORDING, 2)

    def test_long_recording_windows_reach_the_end(self) -> None:
        """70 秒录音：首、中、尾窗口内的脉冲都必须被检测到。"""
        duration_s = 70.0
        rng = np.random.default_rng(11)
        noise = rng.uniform(-1e-4, 1e-4, size=int(SR * duration_s))
        y = noise.astype(np.float32)

        windows = analysis_max_windows()
        window_starts = np.linspace(
            0, duration_s - DETECTION_CHUNK_SECONDS, windows
        )
        planted = [
            float(window_starts[0]) + 1.0,
            float(window_starts[windows // 2]) + 1.0,
            float(window_starts[-1]) + 1.0,
        ]
        sweep = _sweep(80, 40)
        for position in planted:
            start = int(position * SR)
            y[start : start + len(sweep)] += sweep

        raw, refined, _threshold = detect_and_refine_pulses_chunked(
            y, SR, max_windows=windows
        )
        self.assertGreaterEqual(raw, len(planted))
        for position in planted:
            self.assertTrue(
                any(
                    abs(pulse["start_s"] - position) <= 0.05
                    for pulse in refined
                ),
                f"{position}s 处的脉冲未被分析到：{refined}",
            )

    def test_curation_passes_span_to_detection(self) -> None:
        """⑥ 流水线必须把扩展窗口数传给检测（小文件也断言参数）。"""
        from bat_classifier.curation import pipeline as pipeline_mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            _write(input_dir / "a.wav", _recording(_sweep(80, 40)))
            run = CurationRun.create(input_dir, base_dir=root / "runs")

            original = pipeline_mod.detect_and_refine_pulses_chunked
            captured: dict = {}

            def spy(signal, sample_rate, **kwargs):
                captured.update(kwargs)
                return original(signal, sample_rate, **kwargs)

            with mock.patch.object(
                pipeline_mod,
                "detect_and_refine_pulses_chunked",
                side_effect=spy,
            ):
                run.stage_process(CurationParams())

            self.assertEqual(
                captured.get("max_windows"), analysis_max_windows()
            )


# ---------------------------------------------------------------------------
# 聚类特征表准备
# ---------------------------------------------------------------------------
class PrepareFrameTest(unittest.TestCase):
    def test_only_accepted_rows_enter_clustering(self) -> None:
        rows = []
        for index, accepted in enumerate([True, False, True], start=1):
            row = {
                "accepted": accepted,
                "pulse_id": f"p{index}",
                "source_file": "a.wav",
            }
            for column in CLUSTER_FEATURE_COLUMNS:
                row[column] = float(index)
            rows.append(row)
        frame = prepare_cluster_frame(pd.DataFrame(rows))
        self.assertEqual(list(frame["pulse_id"]), ["p1", "p3"])

    def test_missing_columns_raise_clear_error(self) -> None:
        with self.assertRaises(ValueError):
            prepare_cluster_frame(pd.DataFrame([{"accepted": True}]))


if __name__ == "__main__":
    unittest.main()
