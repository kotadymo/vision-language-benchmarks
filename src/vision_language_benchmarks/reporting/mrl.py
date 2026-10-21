from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from vision_language_benchmarks.embedding_dimensions import (
    EMBEDDING_DIMENSION_STRATEGY,
    EMBEDDINGGEMMA_MODEL_KEY,
    NATIVE_EMBEDDING_DIMENSION,
    SUPPORTED_EMBEDDING_DIMENSIONS,
)
from vision_language_benchmarks.errors import DataIntegrityError
from vision_language_benchmarks.registry import get_dataset, get_model
from vision_language_benchmarks.reporting.retrieval_variant import retrieval_variant_info

Result = dict[str, Any]
CLASSIFICATION_METRICS = ("acc1", "acc5")
RETRIEVAL_METRICS = (
    "image_retrieval_recall@1",
    "image_retrieval_recall@5",
    "image_retrieval_recall@10",
    "text_retrieval_recall@1",
    "text_retrieval_recall@5",
    "text_retrieval_recall@10",
)
SUMMARY_FIELDS = (
    "run_id",
    "dataset_key",
    "task",
    "protocol",
    "language",
    "retrieval_variant",
    "precision",
    "embedding_dimension",
    "fp32_vector_bytes",
    "storage_fraction_of_768",
    "storage_reduction_percent",
    "acc1_fraction",
    "acc1_delta_percentage_points_vs_768",
    "acc5_fraction",
    "acc5_delta_percentage_points_vs_768",
    "image_retrieval_recall@1_fraction",
    "image_retrieval_recall@1_delta_percentage_points_vs_768",
    "image_retrieval_recall@5_fraction",
    "image_retrieval_recall@5_delta_percentage_points_vs_768",
    "image_retrieval_recall@10_fraction",
    "image_retrieval_recall@10_delta_percentage_points_vs_768",
    "text_retrieval_recall@1_fraction",
    "text_retrieval_recall@1_delta_percentage_points_vs_768",
    "text_retrieval_recall@5_fraction",
    "text_retrieval_recall@5_delta_percentage_points_vs_768",
    "text_retrieval_recall@10_fraction",
    "text_retrieval_recall@10_delta_percentage_points_vs_768",
    "throughput_images_per_sec",
    "throughput_texts_per_sec",
    "inference_ms_median",
    "gpu_peak_allocated_bytes",
    "gpu_peak_reserved_bytes",
    "oom_retry_count",
    "source_commit_sha",
    "checkpoint_revision",
    "checkpoint_sha256",
    "dataset_checksum",
)


def _metadata(result: Result) -> dict[str, Any]:
    return result["benchmark_metadata"]


def _configuration(result: Result) -> dict[str, Any]:
    value = _metadata(result).get("run_configuration", {})
    return value if isinstance(value, dict) else {}


def _embedding_results(results: list[Result]) -> list[Result]:
    return [
        result
        for result in results
        if _metadata(result).get("embedding_dimension") is not None
    ]


def _condition_key(result: Result) -> tuple[str, str, str, str, str, int, str]:
    configuration = _configuration(result)
    metadata = _metadata(result)
    variant, _ = retrieval_variant_info(result)
    return (
        str(result["task"]),
        str(result["dataset"]),
        str(configuration.get("protocol", "")),
        str(result["language"]),
        variant or "",
        int(metadata["embedding_dimension"]),
        str(metadata.get("precision", "")),
    )


def _expected_conditions() -> set[tuple[str, str, str, str, str, int, str]]:
    conditions: set[tuple[str, str, str, str, str, int, str]] = set()
    for dataset_key in ("cifar10", "cifar100"):
        for protocol in ("p0", "p1"):
            for dimension in SUPPORTED_EMBEDDING_DIMENSIONS:
                conditions.add(
                    (
                        "classification",
                        dataset_key,
                        protocol,
                        "en",
                        "",
                        dimension,
                        "fp32",
                    )
                )
    for language in ("en", "ja"):
        for dimension in SUPPORTED_EMBEDDING_DIMENSIONS:
            conditions.add(
                (
                    "retrieval",
                    "crossmodal3600",
                    "p2",
                    language,
                    "official_first_caption",
                    dimension,
                    "fp32",
                )
            )
    return conditions


def build_embedding_dimension_audit(
    results: list[Result],
    publication_states: dict[str, str],
    frozen_source_commit_sha: str,
) -> dict[str, Any]:
    dimension_results = _embedding_results(results)
    if len(dimension_results) != 24:
        raise DataIntegrityError(
            "The EmbeddingGemma MRL audit requires all 24 primary dimension runs."
        )

    actual_conditions = [_condition_key(result) for result in dimension_results]
    if len(actual_conditions) != len(set(actual_conditions)):
        raise DataIntegrityError("The MRL suite contains duplicate run conditions.")
    missing = _expected_conditions().difference(actual_conditions)
    unexpected = set(actual_conditions).difference(_expected_conditions())
    if missing or unexpected:
        raise DataIntegrityError(
            "The MRL suite does not match the required 24-run matrix "
            f"(missing={sorted(missing)!r}, unexpected={sorted(unexpected)!r})."
        )

    run_ids: set[str] = set()
    source_commits: set[str] = set()
    checkpoint_revisions: set[str] = set()
    checkpoint_hashes: set[str] = set()
    lock_hashes: set[str] = set()
    dataset_checksums: dict[str, set[str]] = defaultdict(set)
    candidate_baselines: dict[str, dict[str, Any]] = {}
    classification_baselines: dict[tuple[str, str], dict[str, Any]] = {}
    run_records: list[dict[str, Any]] = []

    for result in dimension_results:
        metadata = _metadata(result)
        configuration = _configuration(result)
        run_id = metadata["run_id"]
        if run_id in run_ids:
            raise DataIntegrityError(f"The MRL suite repeats run ID {run_id!r}.")
        run_ids.add(run_id)
        if result["status"] != "completed" or publication_states.get(run_id) != "active":
            raise DataIntegrityError(f"MRL run {run_id} is not a completed active result.")
        if (
            metadata.get("reproducibility_mode") != "strict"
            or metadata.get("working_tree_dirty") is not False
            or metadata.get("source_commit_sha") != frozen_source_commit_sha
            or metadata.get("expected_source_commit_sha") != frozen_source_commit_sha
            or configuration.get("reproducibility_mode") != "strict"
            or configuration.get("expected_source_commit_sha") != frozen_source_commit_sha
        ):
            raise DataIntegrityError(f"MRL run {run_id} lacks clean frozen-source provenance.")
        if (
            configuration.get("model_key") != EMBEDDINGGEMMA_MODEL_KEY
            or configuration.get("embedding_dimension") != metadata["embedding_dimension"]
            or metadata.get("embedding_dimension_strategy") != EMBEDDING_DIMENSION_STRATEGY
            or metadata.get("native_embedding_dimension") != NATIVE_EMBEDDING_DIMENSION
            or metadata.get("embedding_generation_mode") != "native_truncate_dim"
            or metadata.get("model_implementation_version") != "0.4.0"
        ):
            raise DataIntegrityError(f"MRL run {run_id} did not use native v0.4.0 inference.")
        if (
            metadata.get("precision") != "fp32"
            or metadata.get("device") != "cuda"
            or metadata.get("seed") != 0
            or configuration.get("batch_size_requested") != 16
        ):
            raise DataIntegrityError(f"MRL run {run_id} does not match the benchmark settings.")

        source_commits.add(str(metadata.get("source_commit_sha")))
        checkpoint_revisions.add(str(metadata.get("checkpoint_revision")))
        checkpoint_sha256 = metadata.get("checkpoint_sha256")
        lock_sha256 = metadata.get("uv_lock_sha256")
        dataset_checksum = metadata.get("dataset_checksum")
        for name, value in (
            ("checkpoint", checkpoint_sha256),
            ("lockfile", lock_sha256),
            ("dataset", dataset_checksum),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise DataIntegrityError(
                    f"MRL run {run_id} has an invalid {name} SHA-256."
                )
        checkpoint_hashes.add(checkpoint_sha256)
        lock_hashes.add(lock_sha256)
        dataset_checksums[str(result["dataset"])].add(dataset_checksum)
        if (
            metadata.get("checkpoint_revision")
            != get_model(EMBEDDINGGEMMA_MODEL_KEY).checkpoint_revision
        ):
            raise DataIntegrityError(f"MRL run {run_id} has an unexpected checkpoint identity.")

        dimension = int(metadata["embedding_dimension"])
        if result["task"] == "classification":
            dataset = get_dataset(str(result["dataset"]))
            if (
                metadata.get("image_count") != dataset.expected_images
                or metadata.get("class_count") != dataset.expected_classes
            ):
                raise DataIntegrityError(f"MRL run {run_id} has unexpected CIFAR candidates.")
            identity = (str(result["dataset"]), str(configuration["protocol"]))
            candidate = {
                "image_id_order_hash": metadata.get("image_id_order_hash"),
                "text_id_order_hash": metadata.get("text_id_order_hash"),
                "classnames_hash": metadata.get("classnames_hash"),
                "template_hash": metadata.get("template_hash"),
            }
            previous = classification_baselines.setdefault(identity, candidate)
            if previous != candidate:
                raise DataIntegrityError(
                    f"CIFAR candidates or prompts changed across MRL dimensions for {identity}."
                )
        else:
            if (
                result["dataset"] != "crossmodal3600"
                or metadata.get("retrieval_variant") != "official_first_caption"
                or metadata.get("image_count") != 3_600
                or metadata.get("text_count") != 3_600
                or metadata.get("one_caption_per_image") is not True
                or metadata.get("candidate_counts") != {"images": 3_600, "texts": 3_600}
            ):
                raise DataIntegrityError(
                    f"MRL run {run_id} has unexpected XM3600 first-caption candidates."
                )
            image_ids = metadata.get("image_ids")
            text_ids = metadata.get("text_ids")
            if (
                not isinstance(image_ids, list)
                or len(image_ids) != 3_600
                or not isinstance(text_ids, list)
                or text_ids != [f"{image_id}#0" for image_id in image_ids]
            ):
                raise DataIntegrityError(f"MRL run {run_id} has invalid ordered candidates.")
            language_identity = str(result["language"])
            candidate = {
                "dataset_checksum": metadata.get("dataset_checksum"),
                "image_ids": image_ids,
                "text_ids": text_ids,
                "image_id_order_hash": metadata.get("image_id_order_hash"),
                "text_id_order_hash": metadata.get("text_id_order_hash"),
                "text_content_sha256": metadata.get("text_content_sha256"),
            }
            previous = candidate_baselines.setdefault(language_identity, candidate)
            if previous != candidate:
                raise DataIntegrityError(
                    f"XM3600 candidate ordering or caption content changed for "
                    f"{language_identity}/{dimension}D."
                )

        run_records.append(
            {
                "run_id": run_id,
                "condition": {
                    "task": result["task"],
                    "dataset": result["dataset"],
                    "protocol": configuration["protocol"],
                    "language": result["language"],
                    "retrieval_variant": metadata.get("retrieval_variant"),
                    "embedding_dimension": dimension,
                },
                "source_commit_sha": metadata["source_commit_sha"],
                "working_tree_dirty": metadata["working_tree_dirty"],
                "dataset_checksum": metadata.get("dataset_checksum"),
                "checkpoint_revision": metadata.get("checkpoint_revision"),
                "checkpoint_sha256": metadata.get("checkpoint_sha256"),
                "image_count": metadata.get("image_count"),
                "text_count": metadata.get("text_count"),
                "batch_size_requested": configuration.get("batch_size_requested"),
                "image_batch_size_actual": metadata.get("image_batch_size_actual"),
                "text_batch_size_actual": metadata.get("text_batch_size_actual"),
                "oom_history": metadata.get("oom_history", []),
            }
        )

    if source_commits != {frozen_source_commit_sha}:
        raise DataIntegrityError("MRL runs do not share the frozen source revision.")
    if len(checkpoint_revisions) != 1 or len(checkpoint_hashes) != 1:
        raise DataIntegrityError("MRL runs use inconsistent model checkpoints.")
    if len(lock_hashes) != 1:
        raise DataIntegrityError("MRL runs use inconsistent lockfiles.")
    if any(len(checksums) != 1 for checksums in dataset_checksums.values()):
        raise DataIntegrityError("MRL runs use inconsistent dataset checksums.")

    for dimension in SUPPORTED_EMBEDDING_DIMENSIONS:
        english = candidate_baselines.get("en")
        japanese = candidate_baselines.get("ja")
        if english is None or japanese is None:
            raise DataIntegrityError(f"XM3600 language pair is missing at {dimension}D.")
        for field in (
            "dataset_checksum",
            "image_ids",
            "text_ids",
            "image_id_order_hash",
            "text_id_order_hash",
        ):
            if english[field] != japanese[field]:
                raise DataIntegrityError(
                    f"XM3600 English/Japanese candidates differ at {dimension}D: {field}."
                )

    return {
        "schema_version": "1.0.0",
        "status": "completed",
        "suite": "embeddinggemma2-native-mrl",
        "frozen_source_commit_sha": frozen_source_commit_sha,
        "model_key": EMBEDDINGGEMMA_MODEL_KEY,
        "openclip_gemma_version": "0.4.0",
        "embedding_dimensions": list(SUPPORTED_EMBEDDING_DIMENSIONS),
        "dimension_strategy": EMBEDDING_DIMENSION_STRATEGY,
        "native_inference": "independent encode_image/encode_text truncate_dim per run",
        "run_count": len(run_records),
        "clean_active_run_count": sum(
            record["working_tree_dirty"] is False for record in run_records
        ),
        "checkpoint_revision": next(iter(checkpoint_revisions)),
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "uv_lock_sha256": next(iter(lock_hashes)),
        "dataset_checksums": {
            dataset: next(iter(checksums))
            for dataset, checksums in sorted(dataset_checksums.items())
        },
        "xm3600_candidate_order_verified_for_all_dimensions": True,
        "runs": sorted(
            run_records,
            key=lambda record: (
                record["condition"]["dataset"],
                record["condition"]["protocol"],
                record["condition"]["language"],
                record["condition"]["embedding_dimension"],
            ),
        ),
    }


def _baseline_condition_key(result: Result) -> tuple[str, str, str, str, str, str]:
    configuration = _configuration(result)
    metadata = _metadata(result)
    variant, _ = retrieval_variant_info(result)
    return (
        str(result["task"]),
        str(result["dataset"]),
        str(configuration.get("protocol", "")),
        str(result["language"]),
        variant or "",
        str(metadata.get("precision", "")),
    )


def _same_recorded_value(previous: Any, current: Any) -> bool:
    if previous is None or current is None:
        return False
    if isinstance(previous, str) and not previous:
        return False
    if isinstance(current, str) and not current:
        return False
    return previous == current


def build_embedding_dimension_baseline_comparison(
    dimension_results: list[Result],
    previous_results: list[Result],
    previous_publication_states: dict[str, str],
    frozen_source_commit_sha: str,
    previous_frozen_source_commit_sha: str,
) -> dict[str, Any]:
    expected_keys = {
        ("classification", "cifar10", protocol, "en", "", "fp32")
        for protocol in ("p0", "p1")
    } | {
        ("classification", "cifar100", protocol, "en", "", "fp32")
        for protocol in ("p0", "p1")
    } | {
        ("retrieval", "crossmodal3600", "p2", language, "official_first_caption", "fp32")
        for language in ("en", "ja")
    }
    current_768: dict[tuple[str, str, str, str, str, str], Result] = {}
    for result in _embedding_results(dimension_results):
        if _metadata(result).get("embedding_dimension") != NATIVE_EMBEDDING_DIMENSION:
            continue
        key = _baseline_condition_key(result)
        if key in current_768:
            raise DataIntegrityError(f"Duplicate current 768D baseline condition: {key}.")
        current_768[key] = result

    previous_768: dict[tuple[str, str, str, str, str, str], Result] = {}
    for result in previous_results:
        configuration = _configuration(result)
        metadata = _metadata(result)
        run_id = metadata["run_id"]
        if (
            configuration.get("model_key") != EMBEDDINGGEMMA_MODEL_KEY
            or metadata.get("embedding_dimension") not in {None, NATIVE_EMBEDDING_DIMENSION}
            or metadata.get("model_implementation_version") != "0.3.0"
            or previous_publication_states.get(run_id) not in {"active", "historical"}
            or result["status"] != "completed"
        ):
            continue
        key = _baseline_condition_key(result)
        if key not in expected_keys:
            continue
        if key in previous_768:
            raise DataIntegrityError(f"Duplicate historical 768D baseline condition: {key}.")
        previous_768[key] = result

    if set(current_768) != expected_keys or set(previous_768) != expected_keys:
        raise DataIntegrityError(
            "The v0.3.0 compatibility comparison requires four CIFAR and two "
            "official-first-caption 768D baseline pairs."
        )

    comparisons: list[dict[str, Any]] = []
    maximum_absolute_difference = 0.0
    for key in sorted(expected_keys):
        previous = previous_768[key]
        current = current_768[key]
        previous_metadata = _metadata(previous)
        current_metadata = _metadata(current)
        previous_configuration = _configuration(previous)
        current_configuration = _configuration(current)
        retrieval_variant, _ = retrieval_variant_info(current)
        previous_prompt = previous_metadata.get("prompt_name")
        current_prompt = current_metadata.get("prompt_name")
        prompt_configuration_matches = (
            _same_recorded_value(previous_prompt, current_prompt)
            and previous_metadata.get("template_hash") == current_metadata.get("template_hash")
            and previous_metadata.get("classnames_hash")
            == current_metadata.get("classnames_hash")
        )
        previous_preprocessing_hash = previous_metadata.get(
            "preprocessing_config_sha256"
        )
        current_preprocessing_hash = current_metadata.get(
            "preprocessing_config_sha256"
        )
        checks = {
            "dataset_checksum": (
                _same_recorded_value(
                    previous_metadata.get("dataset_checksum"),
                    current_metadata.get("dataset_checksum"),
                )
            ),
            "dataset_revision": (
                _same_recorded_value(
                    previous_metadata.get("dataset_revision"),
                    current_metadata.get("dataset_revision"),
                )
            ),
            "dataset_split": (
                _same_recorded_value(
                    previous_metadata.get("dataset_split"),
                    current_metadata.get("dataset_split"),
                )
            ),
            "checkpoint_revision": (
                _same_recorded_value(
                    previous_metadata.get("checkpoint_revision"),
                    current_metadata.get("checkpoint_revision"),
                )
            ),
            "checkpoint_sha256": (
                _same_recorded_value(
                    previous_metadata.get("checkpoint_sha256"),
                    current_metadata.get("checkpoint_sha256"),
                )
            ),
            "preprocessing_config": (
                previous_preprocessing_hash == current_preprocessing_hash
                if isinstance(previous_preprocessing_hash, str)
                and previous_preprocessing_hash
                and isinstance(current_preprocessing_hash, str)
                and current_preprocessing_hash
                else None
            ),
            "precision": (
                _same_recorded_value(
                    previous_metadata.get("precision"),
                    current_metadata.get("precision"),
                )
            ),
            "seed": _same_recorded_value(
                previous_metadata.get("seed"),
                current_metadata.get("seed"),
            ),
            "requested_batch_size": (
                _same_recorded_value(
                    previous_configuration.get("batch_size_requested"),
                    current_configuration.get("batch_size_requested"),
                )
            ),
            "prompt_configuration": prompt_configuration_matches,
        }
        if retrieval_variant is not None:
            checks.update(
                {
                    "retrieval_variant": (
                        _same_recorded_value(
                            retrieval_variant_info(previous)[0],
                            retrieval_variant,
                        )
                    ),
                    "candidate_counts": (
                        _same_recorded_value(
                            previous_metadata.get("candidate_counts"),
                            current_metadata.get("candidate_counts"),
                        )
                    ),
                    "image_id_order": (
                        _same_recorded_value(
                            previous_metadata.get("image_id_order_hash"),
                            current_metadata.get("image_id_order_hash"),
                        )
                        and isinstance(previous_metadata.get("image_ids"), list)
                        and bool(previous_metadata.get("image_ids"))
                        and previous_metadata.get("image_ids")
                        == current_metadata.get("image_ids")
                    ),
                    "text_id_order": (
                        _same_recorded_value(
                            previous_metadata.get("text_id_order_hash"),
                            current_metadata.get("text_id_order_hash"),
                        )
                        and isinstance(previous_metadata.get("text_ids"), list)
                        and bool(previous_metadata.get("text_ids"))
                        and previous_metadata.get("text_ids")
                        == current_metadata.get("text_ids")
                    ),
                    "selected_caption_content": (
                        _same_recorded_value(
                            previous_metadata.get("text_content_sha256"),
                            current_metadata.get("text_content_sha256"),
                        )
                    ),
                }
            )
        else:
            checks.update(
                {
                    "image_id_order": (
                        _same_recorded_value(
                            previous_metadata.get("image_id_order_hash"),
                            current_metadata.get("image_id_order_hash"),
                        )
                    ),
                    "text_id_order": (
                        _same_recorded_value(
                            previous_metadata.get("text_id_order_hash"),
                            current_metadata.get("text_id_order_hash"),
                        )
                    ),
                }
            )

        metric_differences: dict[str, dict[str, float | None]] = {}
        for metric in sorted(set(previous.get("metrics", {})) | set(current.get("metrics", {}))):
            old_value = previous.get("metrics", {}).get(metric)
            new_value = current.get("metrics", {}).get(metric)
            if not isinstance(old_value, (int, float)) or not isinstance(
                new_value,
                (int, float),
            ):
                metric_differences[metric] = {
                    "previous_value": None,
                    "current_value": None,
                    "difference_fraction": None,
                    "difference_percentage_points": None,
                }
                continue
            difference = float(new_value) - float(old_value)
            maximum_absolute_difference = max(
                maximum_absolute_difference,
                abs(difference),
            )
            metric_differences[metric] = {
                "previous_value": float(old_value),
                "current_value": float(new_value),
                "difference_fraction": difference,
                "difference_percentage_points": difference * 100,
            }

        failed_checks = [
            name for name, passed in checks.items() if passed is False
        ]
        unverified_checks = [
            name for name, passed in checks.items() if passed is None
        ]
        comparisons.append(
            {
                "condition": {
                    "task": key[0],
                    "dataset": key[1],
                    "protocol": key[2],
                    "language": key[3],
                    "retrieval_variant": retrieval_variant,
                    "embedding_dimension": NATIVE_EMBEDDING_DIMENSION,
                },
                "previous": {
                    "run_id": previous_metadata["run_id"],
                    "model_implementation_version": previous_metadata.get(
                        "model_implementation_version"
                    ),
                    "source_commit_sha": previous_metadata.get("source_commit_sha"),
                    "working_tree_dirty": previous_metadata.get("working_tree_dirty"),
                    "checkpoint_revision": previous_metadata.get("checkpoint_revision"),
                    "checkpoint_sha256": previous_metadata.get("checkpoint_sha256"),
                },
                "current": {
                    "run_id": current_metadata["run_id"],
                    "model_implementation_version": current_metadata.get(
                        "model_implementation_version"
                    ),
                    "source_commit_sha": current_metadata.get("source_commit_sha"),
                    "working_tree_dirty": current_metadata.get("working_tree_dirty"),
                    "checkpoint_revision": current_metadata.get("checkpoint_revision"),
                    "checkpoint_sha256": current_metadata.get("checkpoint_sha256"),
                },
                "dataset_checksum": current_metadata.get("dataset_checksum"),
                "previous_dataset_checksum": previous_metadata.get("dataset_checksum"),
                "candidate_counts": current_metadata.get("candidate_counts"),
                "candidate_ordering": {
                    "previous": {
                        "image_id_order_hash": previous_metadata.get(
                            "image_id_order_hash"
                        ),
                        "text_id_order_hash": previous_metadata.get("text_id_order_hash"),
                        "text_content_sha256": previous_metadata.get(
                            "text_content_sha256"
                        ),
                    },
                    "current": {
                        "image_id_order_hash": current_metadata.get(
                            "image_id_order_hash"
                        ),
                        "text_id_order_hash": current_metadata.get("text_id_order_hash"),
                        "text_content_sha256": current_metadata.get(
                            "text_content_sha256"
                        ),
                    },
                },
                "conditions_match": all(passed is True for passed in checks.values()),
                "failed_checks": failed_checks,
                "unverified_checks": unverified_checks,
                "checks": checks,
                "metrics": metric_differences,
            }
        )

    has_failed_checks = any(comparison["failed_checks"] for comparison in comparisons)
    has_unverified_checks = any(
        comparison["unverified_checks"] for comparison in comparisons
    )
    status = (
        "conditions_differ"
        if has_failed_checks
        else "conditions_not_fully_verified"
        if has_unverified_checks
        else "matched_conditions"
    )
    return {
        "schema_version": "1.0.0",
        "status": status,
        "baseline_model_version": "0.3.0",
        "current_model_version": "0.4.0",
        "current_source_commit_sha": frozen_source_commit_sha,
        "previous_publication_frozen_source_commit_sha": previous_frozen_source_commit_sha,
        "comparison_count": len(comparisons),
        "historical_selection_policy": (
            "Completed v0.3.0 EmbeddingGemma records marked active or historical by the "
            "baseline lifecycle; superseded records are excluded."
        ),
        "maximum_absolute_metric_difference_fraction": maximum_absolute_difference,
        "maximum_absolute_metric_difference_percentage_points": (
            maximum_absolute_difference * 100
        ),
        "comparisons": comparisons,
        "interpretation": (
            "Metric differences are reported as measured. The compatibility comparison does "
            "not rewrite baseline records or establish equivalence when conditions differ."
        ),
    }


def _summary_rows(results: list[Result]) -> list[dict[str, Any]]:
    dimension_results = _embedding_results(results)
    baselines: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for result in dimension_results:
        metadata = _metadata(result)
        if metadata.get("embedding_dimension") == NATIVE_EMBEDDING_DIMENSION:
            task_key = (
                str(result["task"]),
                str(result["dataset"]),
                str(result["language"]),
                str(_configuration(result).get("protocol", "")),
                str(retrieval_variant_info(result)[0] or ""),
                str(metadata.get("precision", "")),
            )
            baselines[task_key] = result["metrics"]

    rows: list[dict[str, Any]] = []
    for result in dimension_results:
        metadata = _metadata(result)
        configuration = _configuration(result)
        metrics = result.get("metrics", {})
        task_key = (
            str(result["task"]),
            str(result["dataset"]),
            str(result["language"]),
            str(configuration.get("protocol", "")),
            str(retrieval_variant_info(result)[0] or ""),
            str(metadata.get("precision", "")),
        )
        baseline = baselines.get(task_key, {})
        dimension = int(metadata["embedding_dimension"])
        row: dict[str, Any] = {
            "run_id": metadata["run_id"],
            "dataset_key": result["dataset"],
            "task": result["task"],
            "protocol": configuration.get("protocol", ""),
            "language": result["language"],
            "retrieval_variant": retrieval_variant_info(result)[0] or "",
            "precision": metadata.get("precision", ""),
            "embedding_dimension": dimension,
            "fp32_vector_bytes": dimension * 4,
            "storage_fraction_of_768": dimension / NATIVE_EMBEDDING_DIMENSION,
            "storage_reduction_percent": (
                1 - dimension / NATIVE_EMBEDDING_DIMENSION
            )
            * 100,
            "throughput_images_per_sec": metadata.get("throughput_images_per_sec"),
            "throughput_texts_per_sec": metadata.get("throughput_texts_per_sec"),
            "inference_ms_median": metadata.get("inference_ms_median"),
            "gpu_peak_allocated_bytes": metadata.get("gpu_peak_allocated_bytes"),
            "gpu_peak_reserved_bytes": metadata.get("gpu_peak_reserved_bytes"),
            "oom_retry_count": len(metadata.get("oom_history") or []),
            "source_commit_sha": metadata.get("source_commit_sha"),
            "checkpoint_revision": metadata.get("checkpoint_revision"),
            "checkpoint_sha256": metadata.get("checkpoint_sha256"),
            "dataset_checksum": metadata.get("dataset_checksum"),
        }
        for metric in (*CLASSIFICATION_METRICS, *RETRIEVAL_METRICS):
            metric_value = metrics.get(metric)
            baseline_value = baseline.get(metric)
            row[f"{metric}_fraction"] = metric_value
            row[f"{metric}_delta_percentage_points_vs_768"] = (
                (float(metric_value) - float(baseline_value)) * 100
                if isinstance(metric_value, (int, float))
                and isinstance(baseline_value, (int, float))
                else None
            )
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            str(row["dataset_key"]),
            str(row["protocol"]),
            str(row["language"]),
            -int(row["embedding_dimension"]),
        ),
    )


def _percentage(value: Any) -> str:
    return "—" if not isinstance(value, (int, float)) else f"{float(value) * 100:.2f}%"


def _delta(value: Any) -> str:
    return "—" if not isinstance(value, (int, float)) else f"{float(value):+.2f} pp"


def _rate(value: Any) -> str:
    return "—" if not isinstance(value, (int, float)) else f"{float(value):.1f}"


def _native_mrl_validation_summary(report: dict[str, Any]) -> list[str]:
    measurements = report.get("measurements")
    if not isinstance(measurements, list):
        raise DataIntegrityError("Native MRL validation report has no measurement list.")
    if not measurements:
        return [
            f"- Validation status: `{report.get('status', 'unknown')}`; no numerical "
            "measurements were recorded."
        ]

    native_errors = [
        measurement["native_vs_reference"].get("max_absolute_difference")
        for measurement in measurements
        if isinstance(
            measurement["native_vs_reference"].get("max_absolute_difference"),
            (int, float),
        )
    ]
    adapter_errors = [
        measurement["adapter_output"]["vs_direct_native"].get(
            "max_absolute_difference"
        )
        for measurement in measurements
        if isinstance(
            measurement["adapter_output"]["vs_direct_native"].get(
                "max_absolute_difference"
            ),
            (int, float),
        )
    ]
    relative_errors = [
        measurement["native_vs_reference"].get("max_relative_difference")
        for measurement in measurements
        if isinstance(
            measurement["native_vs_reference"].get("max_relative_difference"),
            (int, float),
        )
    ]
    cosine_similarities = [
        measurement["native_vs_reference"].get("minimum_cosine_similarity")
        for measurement in measurements
        if isinstance(
            measurement["native_vs_reference"].get("minimum_cosine_similarity"),
            (int, float),
        )
    ]
    norm_deviations = [
        measurement["native_vs_reference"].get(
            "maximum_l2_norm_deviation_from_one"
        )
        for measurement in measurements
        if isinstance(
            measurement["native_vs_reference"].get(
                "maximum_l2_norm_deviation_from_one"
            ),
            (int, float),
        )
    ]
    native_max = max(native_errors, default=None)
    adapter_max = max(adapter_errors, default=None)
    relative_max = max(relative_errors, default=None)
    cosine_min = min(cosine_similarities, default=None)
    norm_deviation_max = max(norm_deviations, default=None)
    compared_embedding_count = sum(
        int(measurement["native_vs_reference"].get("compared_embedding_count", 0))
        for measurement in measurements
    )
    invalid_embedding_count = sum(
        int(measurement["native_vs_reference"].get("invalid_embedding_count", 0))
        + int(
            measurement["adapter_output"]["vs_direct_native"].get(
                "invalid_embedding_count",
                0,
            )
        )
        for measurement in measurements
    )
    sample_counts = report.get("sample_counts", {})
    classification_count = sum(
        int(value)
        for value in sample_counts.get("classification_text", {}).values()
    )
    retrieval_count = sum(
        int(value) for value in sample_counts.get("retrieval_text", {}).values()
    )
    validation_status = (
        "PASSED"
        if report.get("overall_passed") is True
        else str(report.get("status", "unknown")).upper()
    )
    tolerance = report.get("tolerance", {})
    check_label = "check" if len(measurements) == 1 else "checks"

    def scientific(value: Any) -> str:
        return "n/a" if not isinstance(value, (int, float)) else f"{float(value):.3e}"

    def decimal(value: Any) -> str:
        return "n/a" if not isinstance(value, (int, float)) else f"{float(value):.17g}"

    return [
        f"- Validation status: **{validation_status}** across {len(measurements)} "
        f"dimension/category {check_label} and {compared_embedding_count} "
        "embedding comparisons.",
        f"- Samples: {sample_counts.get('images', 0)} CIFAR-10 images, "
        f"{classification_count} classification prompts, and {retrieval_count} XM3600 "
        "retrieval texts.",
        f"- Maximum native-versus-reference absolute difference: "
        f"`{scientific(native_max)}`; maximum adapter NumPy FP32 conversion difference: "
        f"`{scientific(adapter_max)}`; invalid embeddings: `{invalid_embedding_count}`.",
        f"- Maximum meaningful relative difference: `{scientific(relative_max)}`; "
        f"minimum cosine similarity: `{decimal(cosine_min)}`; maximum unit-norm "
        f"deviation: `{scientific(norm_deviation_max)}`.",
        f"- Tolerance: `rtol={float(tolerance.get('rtol', 0.0)):.0e}`, "
        f"`atol={float(tolerance.get('atol', 0.0)):.0e}`. Validation source commit: "
        f"`{report.get('source_commit_sha', 'not recorded')}`.",
    ]


def write_embedding_dimension_analysis(
    results: list[Result],
    output_dir: Path,
    baseline_comparison: dict[str, Any] | None = None,
    native_validation_report: dict[str, Any] | None = None,
) -> dict[str, Path]:
    rows = _summary_rows(results)
    if not rows:
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "embedding-dimension-summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in SUMMARY_FIELDS} for row in rows)

    classification_rows = [row for row in rows if row["task"] == "classification"]
    retrieval_rows = [row for row in rows if row["task"] == "retrieval"]
    lines = [
        "# EmbeddingGemma 2 Matryoshka dimension comparison",
        "",
        "Each row comes from an independent full-dataset run. Both image and text "
        "embeddings were generated by `openclip-gemma==0.4.0` with the selected "
        "`truncate_dim`; no 768D embedding was reused for a smaller dimension.",
        "",
        f"The native strategy is `{EMBEDDING_DIMENSION_STRATEGY}`. The vector-size "
        "columns below are calculated for FP32 output (`dimension × 4` bytes) and exclude "
        "model weights, indexes, metadata, and storage-system overhead.",
        "",
    ]
    if native_validation_report is not None:
        lines.extend(
            [
                "## Native MRL API equivalence validation",
                "",
                "This separate small-sample check validates native embedding numerics; it "
                "does not reproduce the full-dataset accuracy measurements.",
                "",
                *_native_mrl_validation_summary(native_validation_report),
                "",
                "[Detailed numerical results](validation/native-mrl-api-equivalence.md) · "
                "[machine-readable JSON](validation/native-mrl-api-equivalence.json).",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Native API equivalence is a separate small-sample validation and is not "
                "inferred from the full-dataset accuracy results.",
                "",
            ]
        )
    lines.extend(
        [
            "[Download the machine-readable comparison](embedding-dimension-summary.csv).",
            "",
            "## CIFAR zero-shot classification",
            "",
            "| Dataset | Protocol | Dimension | Top-1 | Δ vs 768D | Top-5 | Δ vs 768D | "
            "Images/s | FP32 bytes/vector |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in classification_rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["dataset_key"]),
                    str(row["protocol"]).upper(),
                    f'{row["embedding_dimension"]}D',
                    _percentage(row["acc1_fraction"]),
                    _delta(row["acc1_delta_percentage_points_vs_768"]),
                    _percentage(row["acc5_fraction"]),
                    _delta(row["acc5_delta_percentage_points_vs_768"]),
                    _rate(row["throughput_images_per_sec"]),
                    str(row["fp32_vector_bytes"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## XM3600 official first-caption retrieval",
            "",
            "| Language | Dimension | T→I R@1 / R@5 / R@10 | "
            "I→T R@1 / R@5 / R@10 | Images/s | Texts/s | FP32 bytes/vector |",
            "|---|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in retrieval_rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["language"]).upper(),
                    f'{row["embedding_dimension"]}D',
                    " / ".join(
                        _percentage(row[f"image_retrieval_recall@{cutoff}_fraction"])
                        for cutoff in (1, 5, 10)
                    ),
                    " / ".join(
                        _percentage(row[f"text_retrieval_recall@{cutoff}_fraction"])
                        for cutoff in (1, 5, 10)
                    ),
                    _rate(row["throughput_images_per_sec"]),
                    _rate(row["throughput_texts_per_sec"]),
                    str(row["fp32_vector_bytes"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Recall values are reported as measured. Deltas in the CSV are percentage-point "
            "differences from the matching 768D run (same dataset, protocol, language, and "
            "precision). Throughput and latency are the measurements from each independent "
            "run; they are not extrapolated from the 768D result.",
        ]
    )
    if baseline_comparison is not None:
        unverified_checks = sorted(
            {
                name
                for comparison in baseline_comparison["comparisons"]
                for name in comparison["unverified_checks"]
            }
        )
        failed_checks = sorted(
            {
                name
                for comparison in baseline_comparison["comparisons"]
                for name in comparison["failed_checks"]
            }
        )
        lines.extend(
            [
                "",
                "## Fresh 768D results versus the v0.3.0 baseline",
                "",
                f"Condition check status: `{baseline_comparison['status']}`. Differences are "
                "reported as measured; baseline JSON files remain unchanged.",
                "",
                "Equal numerical metrics do not establish identical historical execution "
                "conditions.",
            ]
        )
        if unverified_checks:
            lines.extend(
                [
                    "",
                    "Unverified historical fields: "
                    + ", ".join(unverified_checks)
                    + ".",
                ]
            )
        if failed_checks:
            lines.extend(
                [
                    "",
                    "Condition differences detected in: "
                    + ", ".join(failed_checks)
                    + ".",
                ]
            )
        lines.extend(
            [
                "",
                (
                    "| Dataset | Protocol | Language | Metric | v0.3.0 | "
                    "v0.4.0 768D | Difference (pp) |"
                ),
                "|---|---|---|---|---:|---:|---:|",
            ]
        )
        for comparison in baseline_comparison["comparisons"]:
            condition = comparison["condition"]
            for metric, values in comparison["metrics"].items():
                if values["previous_value"] is None:
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            str(condition["dataset"]),
                            str(condition["protocol"]).upper(),
                            str(condition["language"]).upper(),
                            metric,
                            _percentage(values["previous_value"]),
                            _percentage(values["current_value"]),
                            _delta(values["difference_percentage_points"]),
                        )
                    )
                    + " |"
                )
    lines.append("")
    markdown_path = output_dir / "embedding-dimension-analysis.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return {
        "embedding_dimension_summary_csv": csv_path,
        "embedding_dimension_analysis_markdown": markdown_path,
    }


def render_embedding_dimension_section(
    results: list[Result],
    native_validation_report: dict[str, Any] | None = None,
) -> list[str]:
    rows = _summary_rows(results)
    if not rows:
        return []
    lines = [
        "## Matryoshka embedding dimensions",
        "",
        "EmbeddingGemma results use independent native `truncate_dim` inference at each "
        "dimension. The [dimension-wise analysis](embedding-dimension-analysis.md) reports "
        "classification and retrieval metrics, measured throughput, and FP32 vector-size "
        "trade-offs relative to 768D.",
    ]
    if native_validation_report is not None:
        lines.extend(
            [
                "",
                "## Native MRL API equivalence validation",
                "",
                "The native `truncate_dim` output was compared with L2-normalized leading "
                "coordinates of an independently computed native 768D output. This small-sample "
                "check is distinct from the 24 full-dataset benchmark measurements.",
                "",
                *_native_mrl_validation_summary(native_validation_report),
                "",
                "See the [detailed validation report](validation/native-mrl-api-equivalence.md) "
                "and [machine-readable JSON](validation/native-mrl-api-equivalence.json).",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Native MRL API equivalence is a separate opt-in small-sample check; it does "
                "not reproduce full-dataset accuracy results.",
            ]
        )
    lines.extend(
        [
            "",
            "Ranking-agreement metrics are not reported: the published raw records contain "
            "aggregate Recall@K and candidate-order provenance, not per-query rankings or "
            "embeddings. Such an analysis requires separate full-dataset native inference.",
            "",
        ]
    )
    return lines
