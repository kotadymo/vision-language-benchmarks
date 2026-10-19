from __future__ import annotations

import pytest

from tests.helpers import make_result
from vision_language_benchmarks.atomic import atomic_write_json, content_hash
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


def test_retrieval_variant_changes_hash_and_resume_identity(tmp_path) -> None:
    common_configuration = {
        "model": "openclip_multilingual_b32",
        "dataset": "crossmodal3600",
        "protocol": "p2",
        "language": "en",
        "precision": "fp32",
        "device": "cuda",
        "batch_size": 16,
        "seed": 0,
        "dataset_checksum": "b" * 64,
    }
    all_captions_hash = content_hash(
        {**common_configuration, "retrieval_variant": "all_captions"}
    )
    first_caption_hash = content_hash(
        {**common_configuration, "retrieval_variant": "official_first_caption"}
    )
    assert all_captions_hash != first_caption_hash

    run_id = all_captions_hash[:16]
    results_root = tmp_path
    raw_path = results_root / "raw" / f"{run_id}.json"
    result = make_result(
        model_key="openclip_multilingual_b32",
        dataset_key="crossmodal3600",
        protocol="p2",
        run_id=run_id,
        retrieval_variant="all_captions",
    )
    result["benchmark_metadata"]["configuration_hash"] = all_captions_hash
    atomic_write_json(raw_path, result)

    manifest = RunManifest(results_root / "run-manifest.json")
    manifest.transition(
        run_id,
        "running",
        {
            "configuration_hash": all_captions_hash,
            "raw_path": "raw/" + raw_path.name,
        },
    )
    manifest.transition(run_id, "completed", {})

    assert _run_id(all_captions_hash, results_root, resume=True, force=False) == run_id
    assert (
        _run_id(first_caption_hash, results_root, resume=True, force=False)
        == first_caption_hash[:16]
    )
