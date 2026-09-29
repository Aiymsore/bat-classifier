"""⑦ 长录音分析基准测试（before/after 流式优化通用）。

用法：
    python scripts/benchmark_segment.py --length 60 --calls-per-s 0.5
    python scripts/benchmark_segment.py --length 600 --fastskip on --workers 2

输出 JSON：wall_s / peak_rss_mb / 各类计数 / chunk 统计。
进程独立运行，PeakWorkingSetSize 即该次运行的真实峰值内存。
"""
from __future__ import annotations

import argparse
import ctypes
import json
import shutil
import sys
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SR = 250_000


class _PMC(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def peak_rss_mb() -> float:
    pmc = _PMC()
    pmc.cb = ctypes.sizeof(_PMC)
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    handle = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(
        ctypes.c_void_p(handle), ctypes.byref(pmc), pmc.cb
    )
    if not ok:
        return -1.0
    return round(pmc.PeakWorkingSetSize / 1e6, 1)


def generate_wav(path: Path, length_s: float, calls_per_s: float) -> int:
    """分块写出合成超声录音（FM 扫频 + 弱噪声），返回写入的 call 数。"""
    import soundfile as sf

    rng = np.random.default_rng(7)
    call_period = 1.0 / calls_per_s if calls_per_s > 0 else None
    block_s = 10.0
    written_calls = 0
    next_call_s = 0.2
    # 超过 4GB 的 RIFF WAV 不可用；大文件直接用 RF64（libsndfile 支持）
    projected_bytes = int(length_s * SR * 2)
    container = "RF64" if projected_bytes > 3_500_000_000 else "WAV"
    with sf.SoundFile(
        path, "w", samplerate=SR, channels=1, subtype="PCM_16", format=container
    ) as handle:
        done = 0.0
        while done < length_s:
            n = int(min(block_s, length_s - done) * SR)
            block = rng.uniform(-0.001, 0.001, size=n).astype(np.float32)
            if call_period:
                t = next_call_s
                while t < done + n / SR and t < length_s:
                    start = int((t - done) * SR)
                    sweep_n = int(0.006 * SR)
                    freqs = np.linspace(90_000, 50_000, sweep_n)
                    phase = 2 * np.pi * np.cumsum(freqs) / SR
                    sweep = (0.8 * np.sin(phase)).astype(np.float32)
                    end = min(n, start + sweep_n)
                    if 0 <= start < n:
                        block[start:end] += sweep[: end - start]
                        written_calls += 1
                    t += call_period
                next_call_s = t
            handle.write(block)
            done += n / SR
    return written_calls


class DeterministicClassifier:
    classes_ = np.array(["A", "B", "C", "D"])

    def _labels(self, X):
        dur = np.asarray(X["duration_ms"], dtype=float)
        return np.array(["A", "B", "C", "D"])[np.round(dur).astype(int) % 4]

    def predict(self, X):
        return self._labels(X)

    def predict_proba(self, X):
        labels = self._labels(X)
        out = np.full((len(X), 4), 0.1)
        for row, label in enumerate(labels):
            out[row, list(self.classes_).index(label)] = 0.7
        return out


class AllBatNoiseModel:
    classes_ = np.array(["bat", "noise"])

    def predict_proba(self, X):
        return np.tile([0.99, 0.01], (len(X), 1))


def build_payload():
    from bat_classifier.config import (
        CLUSTER_FEATURE_COLUMNS,
        NOISE_GATE_FEATURE_COLUMNS,
    )

    return {
        "noise_gate": {
            "model": AllBatNoiseModel(),
            "feature_columns": list(NOISE_GATE_FEATURE_COLUMNS),
            "classes": ["bat", "noise"],
            "bat_probability_threshold": 0.5,
        },
        "acoustic_types": {
            "scaler": None,
            "kmeans": None,
            "temporary_classifier": DeterministicClassifier(),
            "raw_cluster_to_type": {},
            "feature_columns": list(CLUSTER_FEATURE_COLUMNS),
            "labels": ["A", "B", "C", "D"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=float, default=60.0)
    parser.add_argument("--calls-per-s", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--fastskip", choices=["on", "off"], default=None)
    parser.add_argument("--chunk", type=float, default=None)
    parser.add_argument("--reuse-wav", type=str, default="")
    args = parser.parse_args()

    from bat_classifier import config as cfg
    from bat_classifier import segment_stream as stream_cfg
    from bat_classifier.segment_analysis import (
        SegmentAnalysisConfig,
        analyze_segment,
    )

    # 引擎运行时读取的是 segment_stream 模块内的常量副本——必须改那里
    if args.chunk is not None:
        stream_cfg.SEGMENT_STREAM_CHUNK_SECONDS = float(args.chunk)
    if args.workers is not None:
        stream_cfg.SEGMENT_STREAM_MAX_WORKERS = int(args.workers)
    if args.fastskip is not None:
        stream_cfg.SEGMENT_FASTSKIP_ENABLED = args.fastskip == "on"

    case = Path(cfg.RESULTS_DIR).parent / f".bench_tmp_{uuid.uuid4().hex}"
    case.mkdir(parents=True, exist_ok=True)
    if args.reuse_wav:
        wav = Path(args.reuse_wav)
    else:
        wav = case / f"bench_{uuid.uuid4().hex}.wav"
        generated_calls = generate_wav(wav, args.length, args.calls_per_s)
        print(f"generated {wav.name}: {generated_calls} calls, {args.length}s")

    config = SegmentAnalysisConfig(
        min_confidence=0.5,
        low_confidence_action="uncertain",
        save_pulse_audio=False,
        save_figures=False,
    ).validate()

    start = time.perf_counter()
    result = analyze_segment(wav, build_payload(), config)
    wall_s = time.perf_counter() - start

    payload = {
        "length_s": args.length,
        "calls_per_s": args.calls_per_s,
        "wall_s": round(wall_s, 2),
        "realtime_factor": round(args.length / wall_s, 2) if wall_s else None,
        "peak_rss_mb": peak_rss_mb(),
        "raw_candidate_count": result.get("raw_candidate_count"),
        "detected_call_count": result.get("detected_call_count"),
        "candidate_call_count": result.get("candidate_call_count"),
        "valid_call_count": result.get("valid_call_count"),
        "calls_per_second": result.get("calls_per_second"),
        "label_counts": result.get("label_counts"),
        "processed_chunk_count": result.get("processed_chunk_count"),
        "fast_skipped_chunk_count": result.get("fast_skipped_chunk_count"),
        "chunk_seconds": getattr(stream_cfg, "SEGMENT_STREAM_CHUNK_SECONDS", None),
        "max_workers": getattr(stream_cfg, "SEGMENT_STREAM_MAX_WORKERS", None),
        "fastskip": getattr(stream_cfg, "SEGMENT_FASTSKIP_ENABLED", None),
    }
    print("BENCH_RESULT " + json.dumps(payload, ensure_ascii=False))
    if not args.reuse_wav:
        shutil.rmtree(case, ignore_errors=True)


if __name__ == "__main__":
    main()
