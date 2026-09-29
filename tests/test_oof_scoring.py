"""OOF Scoring 测试（⑥ 伪标签反哺 Step 2：I9/I10/I11）。

覆盖需求 Test：
8. Cluster OOF consistency 只用“没见过该 recording 的模型”打分
   （recording_A 的打分模型训练集不含 recording_A）；
以及：OOF 不可用时零传播（绝不回退 Production 模型）、fold 可复现。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bat_classifier.config import (  # noqa: E402
    SPECIES_FEATURE_COLUMNS,
    SPECIES_MIN_CLASS_SAMPLES,
)
from bat_classifier.oof_scoring import (  # noqa: E402
    FULL_POOL_FOLD_ID,
    OOF_UNAVAILABLE_INSUFFICIENT,
    OOFScoring,
)

SPECIES_PEAKS = {"Species A": 25.0, "Species B": 45.0, "Species C": 65.0}


def _make_pool(
    species_recordings: dict[str, int],
    pulses_per_recording: int = 4,
    prefix: str = "rec",
) -> pd.DataFrame:
    rows: list[dict] = []
    pulse_index = 0
    for species, recording_count in species_recordings.items():
        peak = SPECIES_PEAKS[species]
        for recording in range(recording_count):
            source_file = f"{species.replace(' ', '')}_{prefix}{recording}.wav"
            for pulse in range(pulses_per_recording):
                features = {column: 30.0 for column in SPECIES_FEATURE_COLUMNS}
                features["peak_freq_khz"] = peak + 0.1 * pulse
                features["duration_ms"] = 20.0 + pulse
                rows.append(
                    {
                        "pulse_id": f"P{pulse_index:05d}",
                        "expert_label": species,
                        "source_file": source_file,
                        "source_path": source_file,
                        "audio_hash": f"hash-{source_file}",
                        **features,
                    }
                )
                pulse_index += 1
    return pd.DataFrame(rows)


class TestOOFScoringModels(unittest.TestCase):
    """Test 8：评分模型绝不含候选所属 recording。"""

    def test_holdout_fold_never_trained_on_scored_recording(self) -> None:
        pool = _make_pool(
            {"Species A": 4, "Species B": 4, "Species C": 4},
            pulses_per_recording=max(2, SPECIES_MIN_CLASS_SAMPLES),
        )
        oof = OOFScoring.build(pool, random_state=42)
        self.assertTrue(oof.available)

        recording_a = "SpeciesA_rec0.wav"
        entry = oof.scoring_entry_for(recording_a)
        self.assertIsNotNone(entry)
        self.assertNotIn(recording_a, entry["train_recordings"])
        self.assertTrue(entry["fold_id"].startswith("oof_fold_"))

        # 预测行携带完整可追溯字段
        rows = pool[pool["source_file"] == recording_a]
        scored = oof.predict(rows)
        self.assertTrue(
            (scored["oof_fold_id"] == entry["fold_id"]).all()
        )
        self.assertTrue(scored["oof_available"].all())
        for _, row in scored.iterrows():
            self.assertNotIn(recording_a, entry["train_recordings"])
            self.assertTrue(str(row["oof_train_recordings_hash"]))
            self.assertTrue(str(row["oof_scoring_model_version"]))

    def test_unseen_recording_scored_by_full_pool_model(self) -> None:
        pool = _make_pool({"Species A": 4, "Species B": 4})
        oof = OOFScoring.build(pool, random_state=42)
        entry = oof.scoring_entry_for("ghost_recording.wav")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["fold_id"], FULL_POOL_FOLD_ID)

    def test_fold_metadata_lists_train_and_score_recordings(self) -> None:
        pool = _make_pool(
            {"Species A": 5, "Species B": 5},
            pulses_per_recording=max(2, SPECIES_MIN_CLASS_SAMPLES),
        )
        oof = OOFScoring.build(pool, random_state=42)
        metadata = oof.metadata()
        self.assertTrue(metadata["available"])
        for fold in metadata["folds"]:
            self.assertTrue(
                set(fold["train_recordings"]).isdisjoint(
                    set(fold["score_recordings"])
                )
            )
            self.assertTrue(fold["train_recordings_hash"])

    def test_reproducible_fold_assignment(self) -> None:
        pool = _make_pool(
            {"Species A": 5, "Species B": 5},
            pulses_per_recording=max(2, SPECIES_MIN_CLASS_SAMPLES),
        )
        first = OOFScoring.build(pool, random_state=42)
        second = OOFScoring.build(pool, random_state=42)
        for recording in pool["source_file"].astype(str).unique():
            entry_a = first.scoring_entry_for(recording)
            entry_b = second.scoring_entry_for(recording)
            self.assertEqual(entry_a["fold_id"], entry_b["fold_id"])
            self.assertEqual(
                entry_a["train_recordings_hash"], entry_b["train_recordings_hash"]
            )


class TestOOFUnavailable(unittest.TestCase):
    """Test 9 前半：OOF 不可用 → 零传播，绝不回退 Production 模型。"""

    def test_insufficient_recordings_marks_unavailable(self) -> None:
        pool = _make_pool({"Species A": 1, "Species B": 2})
        self.assertLess(pool["source_file"].nunique(), 4)
        oof = OOFScoring.build(pool, random_state=42)
        self.assertFalse(oof.available)
        self.assertEqual(
            oof.unavailable_reason, OOF_UNAVAILABLE_INSUFFICIENT
        )

        scored = oof.predict(pool)
        self.assertFalse(scored["oof_available"].any())
        self.assertTrue(
            (
                scored["oof_unavailable_reason"] == OOF_UNAVAILABLE_INSUFFICIENT
            ).all()
        )
        # 不存在任何可用于准入的概率列值
        self.assertTrue(
            scored["oof_top1_probability"].isna().all()
            or (scored["oof_top1_probability"] == "").all()
        )
        metadata = oof.metadata()
        self.assertFalse(metadata["available"])


if __name__ == "__main__":
    unittest.main()
