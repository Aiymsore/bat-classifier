from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bat_classifier.demo import DEFAULT_SAMPLE_PATH, run_demo


class DemoSmokeTest(unittest.TestCase):
    def test_bundled_demo_runs_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "demo_model.joblib"
            result = run_demo(
                input_path=DEFAULT_SAMPLE_PATH,
                model_path=model_path,
                rebuild_model=True,
            )

        self.assertIn(result["status"], {"success", "noise_only"})
        self.assertGreater(int(result.get("candidate_count", 0)), 0)
        self.assertTrue(Path(result["prediction_csv"]).exists())


if __name__ == "__main__":
    unittest.main()
