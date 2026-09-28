"""声型标签参数化与数据来源切换的单元测试。"""

from __future__ import annotations

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

from bat_classifier.clustering import build_zh_summary, create_acoustic_type_mapping
from bat_classifier.config import validate_labels
from bat_classifier.demo import _unique_cluster_mapping


def _centers(peak_freqs: list[float]) -> pd.DataFrame:
    """构造只区分峰值频率的聚类中心表（列结构与真实一致）。"""
    columns = [
        "duration_ms",
        "peak_freq_khz",
        "start_freq_khz",
        "end_freq_khz",
        "frequency_drop_khz",
        "f05_khz",
        "f95_khz",
        "bandwidth_90_khz",
        "slope_khz_per_ms",
        "track_coverage",
        "max_track_jump_khz",
    ]
    frame = pd.DataFrame(0.0, index=range(len(peak_freqs)), columns=columns)
    frame.index.name = "raw_cluster"
    frame["peak_freq_khz"] = peak_freqs
    return frame


class ValidateLabelsTest(unittest.TestCase):
    def test_default_labels_pass(self) -> None:
        self.assertEqual(validate_labels(["A", "B", "C", "D"]), ["A", "B", "C", "D"])

    def test_rejects_duplicates(self) -> None:
        with self.assertRaises(ValueError):
            validate_labels(["A", "A"])

    def test_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            validate_labels(["A", "  "])

    def test_rejects_path_unsafe(self) -> None:
        with self.assertRaises(ValueError):
            validate_labels(["A/B"])


class MappingTest(unittest.TestCase):
    def test_default_labels_use_semantic_path(self) -> None:
        mapping = create_acoustic_type_mapping(
            _centers([20.0, 40.0, 60.0, 80.0])
        )
        self.assertEqual(sorted(mapping.values()), ["A", "B", "C", "D"])
        self.assertEqual(len(set(mapping.keys())), 4)

    def test_generic_mapping_orders_by_peak_frequency(self) -> None:
        labels = ["低频型", "中频型", "高频型"]
        mapping = create_acoustic_type_mapping(
            _centers([70.0, 30.0, 50.0]), labels
        )
        # 峰值频率 30/50/70 kHz 依次对应标签列表顺序。
        self.assertEqual(mapping[1], "低频型")
        self.assertEqual(mapping[2], "中频型")
        self.assertEqual(mapping[0], "高频型")


class DemoMappingTest(unittest.TestCase):
    def test_three_label_assignment(self) -> None:
        labels = pd.Series(["L1", "L2", "L3"] * 3)
        # 簇0全是L1，簇1全是L2，簇2全是L3。
        raw_clusters = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        mapping = _unique_cluster_mapping(raw_clusters, labels, ["L1", "L2", "L3"])
        self.assertEqual(mapping, {0: "L1", 1: "L2", 2: "L3"})


class ZhSummaryTest(unittest.TestCase):
    SEPARATION = {
        "enabled": True,
        "minimum_frequency_gap_khz": 1.5,
        "maximum_components": 3,
    }

    def test_full_summary_renders_all_sections(self) -> None:
        text = build_zh_summary(
            labels=["A", "B", "C", "D"],
            descriptions={"A": "FM型"},
            pulse_count=2616,
            recording_count=3,
            silhouette=0.272,
            consistency=0.9817,
            pca_ratios=[0.432, 0.208],
            separation=self.SEPARATION,
            trained_at="2026-09-27 14:30:00",
        )
        self.assertIn("【训练摘要】 2026-09-27", text)
        self.assertIn("2616 个合格脉冲，来自 3 条录音", text)
        self.assertIn("轮廓系数 0.272", text)
        self.assertIn("有一定结构，但组间重叠较多", text)
        self.assertIn("98.17%", text)
        self.assertIn("共解释 64.0%", text)
        self.assertIn("声波分离：已启用", text)
        self.assertIn("不是物种标签", text)

    def test_none_fields_render_gracefully(self) -> None:
        text = build_zh_summary(
            labels=["甲", "乙"],
            descriptions={},
            pulse_count=50,
            recording_count=2,
            silhouette=None,
            consistency=None,
            pca_ratios=[],
            separation={"enabled": False},
        )
        self.assertIn("未计算轮廓系数", text)
        self.assertIn("本次未评估", text)
        self.assertIn("声波分离：未启用", text)
        self.assertNotIn("PCA：", text)
        self.assertIn("甲：", text)  # 缺失描述时显示占位而不是报错
        self.assertIn("（无描述）", text)


class PrepareDatasetSourceTest(unittest.TestCase):
    def test_folder_source_replaces_raw_dir(self) -> None:
        from bat_classifier import pulse_table

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_dir = tmp_path / "source"
            source_dir.mkdir()
            sr = 8000
            for index in range(2):
                segment = np.sin(
                    2 * np.pi * 500 * np.arange(sr) / sr
                ).astype(np.float32) * 0.3
                sf.write(source_dir / f"rec_{index}.wav", segment, sr)

            patched_dirs = {
                name: tmp_path / name
                for name in [
                    "RAW_DATA_DIR",
                    "RESULTS_DIR",
                    "TABLE_DIR",
                    "FIGURE_DIR",
                    "AUDIO_OUTPUT_DIR",
                    "MODEL_DIR",
                    "SPECTROGRAM_DIR",
                    "PULSE_ZOOM_DIR",
                    "DEMO_PREDICTION_DIR",
                    "CLUSTER_EXAMPLE_DIR",
                    "SEPARATED_PULSE_DIR",
                ]
            }
            with mock.patch.multiple(
                pulse_table, create=True, **patched_dirs
            ):
                stale = pulse_table.RAW_DATA_DIR / "stale.txt"
                pulse_table.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
                stale.write_text("old", encoding="utf-8")

                count = pulse_table.prepare_demo_dataset(source_dir)

                self.assertEqual(count, 2)
                imported = list(pulse_table.RAW_DATA_DIR.glob("*.wav"))
                self.assertEqual(len(imported), 2)
                self.assertFalse(stale.exists(), "旧数据应被清空")

    def test_rejects_invalid_source(self) -> None:
        from bat_classifier import pulse_table

        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "not_exists"
            with self.assertRaises(FileNotFoundError):
                pulse_table.prepare_demo_dataset(missing)


if __name__ == "__main__":
    unittest.main()
