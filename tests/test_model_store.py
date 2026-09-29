"""model_store 测试：模型列表 / 结构校验 / 导入（含不覆盖 ② 训练产物）。

只测算法与文件行为，不测 Qt 控件。
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bat_classifier.model_store import (  # noqa: E402
    AUX_MODEL_NAMES,
    TRAINED_MODEL_PATH,
    ModelEntry,
    import_model_file,
    inspect_model,
    list_models,
    payload_error,
)


# 可被 ③ 预测 / ⑦ 逐叫声分类使用的最小合法 payload
GOOD_PAYLOAD = {
    "noise_gate": {"classifier": "gate"},
    "acoustic_types": {
        "temporary_classifier": "clf",
        "feature_columns": ["duration_ms", "peak_freq_khz"],
        "labels": ["A", "B"],
    },
}


def _make_case_dir() -> Path:
    base = Path(tempfile.gettempdir()) / "_bat_model_store_tests"
    base.mkdir(parents=True, exist_ok=True)
    case = base / f"case_{uuid.uuid4().hex}"
    case.mkdir(parents=True, exist_ok=True)
    return case


def _dump(payload, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
    return path


class PayloadErrorTest(unittest.TestCase):
    def test_valid_payload(self) -> None:
        self.assertIsNone(payload_error(GOOD_PAYLOAD))

    def test_reject_non_dict(self) -> None:
        self.assertIn("不是模型字典", payload_error([1, 2, 3]) or "")

    def test_reject_missing_acoustic_types(self) -> None:
        error = payload_error({"noise_gate": {}})
        self.assertIn("acoustic_types", error or "")

    def test_reject_missing_classifier(self) -> None:
        error = payload_error({"acoustic_types": {"feature_columns": ["a"]}})
        self.assertIn("temporary_classifier", error or "")

    def test_reject_missing_feature_columns(self) -> None:
        error = payload_error(
            {"acoustic_types": {"temporary_classifier": "clf"}}
        )
        self.assertIn("feature_columns", error or "")

    def test_reject_species_style_payload(self) -> None:
        # ⑥ 物种模型的 payload 没有两级结构，不能被 ③/⑦ 直接用
        error = payload_error({"model": "rf", "classes": ["A"]})
        self.assertIsNotNone(error)


class InspectModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.case = _make_case_dir()

    def tearDown(self) -> None:
        shutil.rmtree(self.case, ignore_errors=True)

    def test_missing_file(self) -> None:
        entry = inspect_model(self.case / "nope.joblib")
        self.assertFalse(entry.valid)
        self.assertIn("不存在", entry.error)

    def test_valid_file_keeps_labels(self) -> None:
        path = _dump(GOOD_PAYLOAD, self.case / "good.joblib")
        entry = inspect_model(path)
        self.assertTrue(entry.valid)
        self.assertEqual(entry.labels, ("A", "B"))
        self.assertGreater(entry.size_mb, 0.0)

    def test_corrupted_file_is_reported_not_raised(self) -> None:
        path = self.case / "broken.joblib"
        path.write_bytes(b"not a joblib file")
        entry = inspect_model(path)
        self.assertFalse(entry.valid)
        self.assertIn("加载失败", entry.error)

    def test_origin_labels(self) -> None:
        path = _dump(GOOD_PAYLOAD, self.case / "other.joblib")
        self.assertEqual(inspect_model(path).origin, "other")
        self.assertEqual(
            inspect_model(path, origin="trained").origin, "trained"
        )
        trained = ModelEntry(path=TRAINED_MODEL_PATH, origin="trained", valid=True)
        self.assertEqual(trained.origin_text, "② 训练")
        self.assertIn("② 训练模型（默认）", trained.display_name)


class ListModelsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.case = _make_case_dir()
        self.model_dir = self.case / "models"

    def tearDown(self) -> None:
        shutil.rmtree(self.case, ignore_errors=True)

    def test_empty_dir(self) -> None:
        self.assertEqual(list_models(self.model_dir), [])

    def test_skips_auxiliary_files(self) -> None:
        _dump(GOOD_PAYLOAD, self.model_dir / "noise_bat_ABCD_model.joblib")
        for name in AUX_MODEL_NAMES:
            _dump({"noise_gate": {}}, self.model_dir / name)
        entries = list_models(self.model_dir)
        names = {entry.path.name for entry in entries}
        self.assertEqual(names, {"noise_bat_ABCD_model.joblib"})

    def test_imported_files_listed_with_origin(self) -> None:
        _dump(
            GOOD_PAYLOAD,
            self.model_dir / "imported" / "my_model.joblib",
        )
        entries = list_models(self.model_dir)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].origin, "imported")
        self.assertTrue(entries[0].valid)

    def test_invalid_file_reported_in_listing(self) -> None:
        path = self.model_dir / "bad.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"garbage")
        entries = list_models(self.model_dir)
        self.assertEqual(len(entries), 1)
        self.assertFalse(entries[0].valid)
        self.assertTrue(entries[0].error)


class ImportModelFileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.case = _make_case_dir()
        self.model_dir = self.case / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.source = _dump(GOOD_PAYLOAD, self.case / "external.joblib")

    def tearDown(self) -> None:
        shutil.rmtree(self.case, ignore_errors=True)

    def test_copies_into_imported_dir(self) -> None:
        target = import_model_file(self.source, self.model_dir)
        self.assertTrue(target.exists())
        self.assertEqual(target.parent, self.model_dir / "imported")
        self.assertEqual(target.name, "external.joblib")
        # 源文件不被动
        self.assertTrue(self.source.exists())

    def test_never_overwrites_trained_model(self) -> None:
        trained = _dump(GOOD_PAYLOAD, self.model_dir / "noise_bat_ABCD_model.joblib")
        trained_before = trained.read_bytes()

        # 同名文件导入到 imported/ 下，不触碰 ② 的训练产物
        same_name = _dump(
            {"acoustic_types": {"temporary_classifier": "x", "feature_columns": ["a"]}},
            self.case / "noise_bat_ABCD_model.joblib",
        )
        target = import_model_file(same_name, self.model_dir)
        self.assertNotEqual(target, trained)
        self.assertEqual(trained.read_bytes(), trained_before)

    def test_collision_gets_suffix(self) -> None:
        first = import_model_file(self.source, self.model_dir)
        second = import_model_file(self.source, self.model_dir)
        self.assertNotEqual(first, second)
        self.assertTrue(second.name.startswith("external_"))
        self.assertTrue(first.exists() and second.exists())

    def test_reject_invalid_payload(self) -> None:
        bad = _dump({"model": "not-two-stage"}, self.case / "bad.joblib")
        with self.assertRaises(ValueError):
            import_model_file(bad, self.model_dir)
        self.assertFalse((self.model_dir / "imported").exists())

    def test_reject_wrong_suffix(self) -> None:
        txt = self.case / "model.txt"
        txt.write_text("hello", encoding="utf-8")
        with self.assertRaises(ValueError):
            import_model_file(txt, self.model_dir)

    def test_reject_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            import_model_file(self.case / "gone.joblib", self.model_dir)


if __name__ == "__main__":
    unittest.main()
