from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from tests.helpers import make_result
from vision_language_benchmarks.atomic import atomic_write_json
from vision_language_benchmarks.reporting.aggregate import generate_reports
from vision_language_benchmarks.reporting.lifecycle import build_result_lifecycle

FROZEN_SOURCE_COMMIT = "a" * 40
DIMENSIONS = (768, 512, 256, 128)


def _make_mrl_results() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for dataset_key in ("cifar10", "cifar100"):
        for protocol in ("p0", "p1"):
            for dimension in DIMENSIONS:
                results.append(
                    make_result(
                        model_key="embeddinggemma2",
                        dataset_key=dataset_key,
                        protocol=protocol,
                        run_id=f"{dataset_key}-{protocol}-{dimension}",
                        embedding_dimension=dimension,
                    )
                )
    for language in ("en", "ja"):
        for dimension in DIMENSIONS:
            results.append(
                make_result(
                    model_key="embeddinggemma2",
                    dataset_key="crossmodal3600",
                    language=language,
                    protocol="p2",
                    run_id=f"xm3600-{language}-{dimension}",
                    retrieval_variant="official_first_caption",
                    embedding_dimension=dimension,
                )
            )

    for result in results:
        metadata = result["benchmark_metadata"]
        configuration = metadata["run_configuration"]
        metadata.update(
            {
                "device": "cuda",
                "requested_device": "cuda",
                "reproducibility_mode": "strict",
                "expected_source_commit_sha": FROZEN_SOURCE_COMMIT,
                "staging_directory": ".cache/benchmark-staging/mrl-tests",
                "working_tree_dirty": False,
                "batch_size_requested": 16,
                "preprocessing_config_sha256": "f" * 64,
                "prompt_name": (
                    "Classification"
                    if result["task"] == "classification"
                    else "SearchQuery"
                ),
            }
        )
        metadata["run_configuration"] = {
            **configuration,
            "device": "cuda",
            "batch_size_requested": 16,
            "reproducibility_mode": "strict",
            "expected_source_commit_sha": FROZEN_SOURCE_COMMIT,
        }
        metadata["source_commit_sha"] = FROZEN_SOURCE_COMMIT
    return results


def _make_legacy_768_results(
    current_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    previous_results: list[dict[str, Any]] = []
    publication_states: dict[str, str] = {}
    for current in current_results:
        current_metadata = current["benchmark_metadata"]
        if current_metadata.get("embedding_dimension") != 768:
            continue
        current_configuration = current_metadata["run_configuration"]
        dataset_key = current_configuration["dataset_key"]
        run_id = f"legacy-{dataset_key}-{current['language']}-{current_configuration['protocol']}"
        previous = make_result(
            model_key="embeddinggemma2",
            dataset_key=dataset_key,
            language=current["language"],
            protocol=current_configuration["protocol"],
            run_id=run_id,
            retrieval_variant=current_metadata.get("retrieval_variant"),
        )
        previous_metadata = previous["benchmark_metadata"]
        previous_configuration = previous_metadata["run_configuration"]
        previous_metadata.update(
            {
                key: current_metadata.get(key)
                for key in (
                    "dataset_checksum",
                    "dataset_revision",
                    "dataset_split",
                    "checkpoint_revision",
                    "checkpoint_sha256",
                    "preprocessing_config_sha256",
                    "precision",
                    "seed",
                    "batch_size_requested",
                    "prompt_name",
                    "template_hash",
                    "classnames_hash",
                    "image_id_order_hash",
                    "text_id_order_hash",
                    "candidate_counts",
                    "text_content_sha256",
                    "image_ids",
                    "text_ids",
                )
                if key in current_metadata
            }
        )
        previous_metadata.update(
            {
                "model_implementation_version": "0.3.0",
                "source_commit_sha": (
                    "60a61259c95e7c387e896b23084e3c3d0fc375f5"
                    if dataset_key.startswith("cifar")
                    else "9d170b6071eecd75d7100ca70092d44d86650f01"
                ),
                "working_tree_dirty": False,
                "batch_size_requested": 16,
            }
        )
        previous_configuration.update(
            {
                "batch_size_requested": 16,
                "dataset_checksum": current_configuration.get("dataset_checksum"),
            }
        )
        for metric, value in current["metrics"].items():
            previous["metrics"][metric] = value - 0.01
        previous_results.append(previous)
        publication_states[run_id] = (
            "historical" if current["task"] == "classification" else "active"
        )

    superseded = make_result(
        model_key="embeddinggemma2",
        dataset_key="crossmodal3600",
        language="en",
        protocol="p2",
        run_id="legacy-superseded-xm3600-en",
        retrieval_variant="official_first_caption",
    )
    previous_results.append(superseded)
    publication_states[superseded["benchmark_metadata"]["run_id"]] = "superseded"
    return previous_results, publication_states


def test_publishable_mrl_report_validates_all_runs_and_writes_dimension_artifacts(
    tmp_path: Path,
) -> None:
    results = _make_mrl_results()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for result in results:
        run_id = result["benchmark_metadata"]["run_id"]
        atomic_write_json(raw_dir / f"{run_id}.json", result)

    lifecycle = build_result_lifecycle(results, FROZEN_SOURCE_COMMIT)
    output_dir = tmp_path / "reports"
    lifecycle_path = output_dir / "validation" / "result-lifecycle.json"
    atomic_write_json(lifecycle_path, lifecycle)

    files = generate_reports(
        raw_dir,
        output_dir,
        publishable=True,
        lifecycle_path=lifecycle_path,
        expected_source_commit_sha=FROZEN_SOURCE_COMMIT,
    )

    audit = json.loads(
        (output_dir / "validation" / "embedding-dimension-audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["run_count"] == 24
    assert audit["clean_active_run_count"] == 24
    assert audit["xm3600_candidate_order_verified_for_all_dimensions"] is True

    with files["summary_csv"].open(encoding="utf-8", newline="") as stream:
        summary = list(csv.DictReader(stream))
    with files["embedding_dimension_summary_csv"].open(
        encoding="utf-8",
        newline="",
    ) as stream:
        dimension_summary = list(csv.DictReader(stream))
    report = files["report_markdown"].read_text(encoding="utf-8")
    analysis = files["embedding_dimension_analysis_markdown"].read_text(
        encoding="utf-8"
    )

    assert len(summary) == 24
    assert {row["embedding_dimension"] for row in summary} == {
        "768",
        "512",
        "256",
        "128",
    }
    assert len(dimension_summary) == 24
    assert next(
        row for row in dimension_summary if row["embedding_dimension"] == "128"
    )["fp32_vector_bytes"] == "512"
    assert "native `truncate_dim`" in report
    assert "storage_reduction_percent" in files["embedding_dimension_summary_csv"].read_text(
        encoding="utf-8"
    )
    assert "independent full-dataset run" in analysis
    assert "embedding_dimension_storage_png" in files
    assert files["embedding_dimension_storage_png"].is_file()


def test_publishable_mrl_report_compares_only_eligible_legacy_768d_results(
    tmp_path: Path,
) -> None:
    current_results = _make_mrl_results()
    current_raw_dir = tmp_path / "current-raw"
    current_raw_dir.mkdir()
    for result in current_results:
        run_id = result["benchmark_metadata"]["run_id"]
        atomic_write_json(current_raw_dir / f"{run_id}.json", result)

    current_output_dir = tmp_path / "current-reports"
    current_lifecycle_path = current_output_dir / "validation" / "result-lifecycle.json"
    atomic_write_json(
        current_lifecycle_path,
        build_result_lifecycle(current_results, FROZEN_SOURCE_COMMIT),
    )

    previous_results, previous_states = _make_legacy_768_results(current_results)
    cifar_baseline = next(
        result
        for result in previous_results
        if result["benchmark_metadata"]["run_configuration"]["dataset_key"] == "cifar10"
        and result["benchmark_metadata"]["run_configuration"]["protocol"] == "p0"
    )
    cifar_baseline["benchmark_metadata"].pop("preprocessing_config_sha256")
    previous_raw_dir = tmp_path / "previous-raw"
    previous_raw_dir.mkdir()
    for result in previous_results:
        run_id = result["benchmark_metadata"]["run_id"]
        atomic_write_json(previous_raw_dir / f"{run_id}.json", result)
    previous_lifecycle_path = tmp_path / "previous-lifecycle.json"
    atomic_write_json(
        previous_lifecycle_path,
        {
            "schema_version": "1.0.0",
            "frozen_source_commit_sha": "9" * 40,
            "entries": [
                {
                    "run_id": run_id,
                    "state": state,
                    "leaderboard_eligible": state in {"active", "historical"},
                }
                for run_id, state in previous_states.items()
            ],
            "attempts": [],
        },
    )

    files = generate_reports(
        current_raw_dir,
        current_output_dir,
        publishable=True,
        lifecycle_path=current_lifecycle_path,
        expected_source_commit_sha=FROZEN_SOURCE_COMMIT,
        baseline_input_path=previous_raw_dir,
        baseline_lifecycle_path=previous_lifecycle_path,
    )

    audit = json.loads(
        (
            current_output_dir
            / "validation"
            / "embedding-dimension-baseline-comparison.json"
        ).read_text(encoding="utf-8")
    )
    assert audit["status"] == "conditions_not_fully_verified"
    assert audit["comparison_count"] == 6
    assert {
        comparison["previous"]["run_id"] for comparison in audit["comparisons"]
    } == {
        result["benchmark_metadata"]["run_id"]
        for result in previous_results
        if previous_states[result["benchmark_metadata"]["run_id"]]
        in {"active", "historical"}
    }
    retrieval_comparison = next(
        comparison
        for comparison in audit["comparisons"]
        if comparison["condition"]["dataset"] == "crossmodal3600"
        and comparison["condition"]["language"] == "en"
    )
    assert retrieval_comparison["previous"]["source_commit_sha"] == (
        "9d170b6071eecd75d7100ca70092d44d86650f01"
    )
    assert retrieval_comparison["previous"]["working_tree_dirty"] is False
    assert retrieval_comparison["checks"]["preprocessing_config"] is True
    assert abs(
        retrieval_comparison["metrics"]["image_retrieval_recall@1"][
            "difference_percentage_points"
        ]
        - 1.0
    ) < 1e-12
    analysis = files["embedding_dimension_analysis_markdown"].read_text(
        encoding="utf-8"
    )
    assert "Fresh 768D results versus the v0.3.0 baseline" in analysis
    assert "Unverified historical fields: preprocessing_config." in analysis
