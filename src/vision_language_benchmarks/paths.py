from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("VLBENCH_DATA_DIR", PROJECT_ROOT / "data")).resolve()
CACHE_ROOT = Path(os.environ.get("VLBENCH_CACHE_DIR", PROJECT_ROOT / ".cache")).resolve()
RESULTS_ROOT = PROJECT_ROOT / "results" / "v0.1.0"
RAW_RESULTS_ROOT = RESULTS_ROOT / "raw"
RUN_MANIFEST_PATH = RESULTS_ROOT / "run-manifest.json"
