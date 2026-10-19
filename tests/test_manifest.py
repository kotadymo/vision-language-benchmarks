from __future__ import annotations

import pytest

from tests.helpers import make_result
from vision_language_benchmarks.atomic import atomic_write_json
from vision_language_benchmarks.evaluation.manifest import RunManifest
from vision_language_benchmarks.evaluation.runner import _run_id


def test_run_manifest_requires_explicit_restart_for_terminal_runs(tmp_path) -> None:
    manifest = RunManifest(tmp_path / "run-manifest.json")
    manifest.transition("run-1", "running", {"configuration_hash": "config"})
    manifest.transition("run-1", "completed", {"raw_path": "raw/run-1.json"})

    with pytest.raises(ValueError, match="restart must be explicit"):
        manifest.transition("run-1", "running", {})

    manifest.transition("run-1", "running", {}, restart=True)
    assert manifest.get("run-1")["status"] == "running"


def test_resume_reuses_only_matching_completed_results(tmp_path) -> None:
    configuration_hash = "c" * 64
    run_id = configuration_hash[:16]
    results_root = tmp_path
    raw_path = results_root / "raw" / f"{run_id}.json"
    result = make_result(run_id=run_id)
    result["benchmark_metadata"]["configuration_hash"] = configuration_hash
    atomic_write_json(raw_path, result)

    manifest = RunManifest(results_root / "run-manifest.json")
    manifest.transition(
        run_id,
        "running",
        {"configuration_hash": configuration_hash, "raw_path": "raw/" + raw_path.name},
    )
    manifest.transition(run_id, "completed", {})

    assert _run_id(configuration_hash, results_root, resume=True, force=False) == run_id
