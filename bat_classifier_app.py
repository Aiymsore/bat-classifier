from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bat_classifier.app import run_app


if __name__ == "__main__":
    raise SystemExit(run_app())
