from __future__ import annotations

from typing import Any

from vision_language_benchmarks.errors import DataIntegrityError
from vision_language_benchmarks.reporting.retrieval_variant import retrieval_variant_info

_RETRIEVAL_METRICS = (
    "image_retrieval_recall@1",
    "image_retrieval_recall@5",
    "image_retrieval_recall@10",
    "text_retrieval_recall@1",
    "text_retrieval_recall@5",
    "text_retrieval_recall@10",
)
_EXPECTED_RUNS = {
    ("embeddinggemma2", "en"),
    ("embeddinggemma2", "ja"),
    ("openclip_multilingual_b32", "en"),
    ("openclip_multilingual_b32", "ja"),
    ("siglip2_base_224", "en"),
    ("siglip2_base_224", "ja"),
    ("rinna_ja_clip_b16", "ja"),
}
_LANGUAGE_PAIRED_MODELS = (
    "embeddinggemma2",
    "openclip_multilingual_b32",
    "siglip2_base_224",
)


def _configuration(result: dict[str, Any]) -> dict[str, Any]:
    configuration = result["benchmark_metadata"].get("run_configuration")
    return configuration if isinstance(configuration, dict) else {}


def _run_key(result: dict[str, Any]) -> tuple[str, str]:
    return str(_configuration(result).get("model_key", "")), str(result["language"])


def _run_map(
    results: list[dict[str, Any]],
    publication_states: dict[str, str],
    state: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for result in results:
        metadata = result["benchmark_metadata"]
        run_id = str(metadata["run_id"])
        variant, _ = retrieval_variant_info(result)
        if publication_states.get(run_id) != state or variant != "official_first_caption":
            continue
        key = _run_key(result)
        if key in selected:
            raise DataIntegrityError(
                f"Multiple {state} official-first-caption results exist for {key[0]}/{key[1]}."
            )
        selected[key] = result
    return selected


def _pair_checks(previous: dict[str, Any], replacement: dict[str, Any]) -> dict[str, bool]:
    previous_metadata = previous["benchmark_metadata"]
    replacement_metadata = replacement["benchmark_metadata"]
    previous_configuration = _configuration(previous)
    replacement_configuration = _configuration(replacement)
    checks = {
        "same_dataset_checksum": (
            previous_metadata.get("dataset_checksum")
            == replacement_metadata.get("dataset_checksum")
        ),
        "same_checkpoint_revision": (
            previous_metadata.get("checkpoint_revision")
            == replacement_metadata.get("checkpoint_revision")
        ),
        "same_checkpoint_hash": (
            previous_metadata.get("checkpoint_sha256")
            == replacement_metadata.get("checkpoint_sha256")
        ),
        "same_tokenizer_revision": (
            previous_metadata.get("tokenizer_revision")
            == replacement_metadata.get("tokenizer_revision")
        ),
        "same_preprocessing_configuration": (
            previous_metadata.get("preprocessing_config_sha256")
            == replacement_metadata.get("preprocessing_config_sha256")
        ),
        "same_prompt_configuration": all(
            previous_metadata.get(key) == replacement_metadata.get(key)
            for key in ("prompt_name", "prompt_policy", "task_prompt", "template_hash")
        ),
        "same_protocol_precision_seed": all(
            previous_configuration.get(key) == replacement_configuration.get(key)
            for key in ("protocol", "precision", "seed")
        )
        and previous_metadata.get("precision") == replacement_metadata.get("precision"),
        "same_batch_and_dataset_split": (
            previous_configuration.get("batch_size_requested")
            == replacement_configuration.get("batch_size_requested")
            and previous_metadata.get("dataset_split")
            == replacement_metadata.get("dataset_split")
        ),
        "same_runtime_environment": all(
            previous_metadata.get(key) == replacement_metadata.get(key)
            for key in (
                "device",
                "gpu_name",
                "driver_version",
                "torch_version",
                "torch_cuda_version",
            )
        ),
        "same_candidate_counts": (
            previous_metadata.get("candidate_counts")
            == replacement_metadata.get("candidate_counts")
            == {"images": 3_600, "texts": 3_600}
            and previous_metadata.get("image_count") == 3_600
            and replacement_metadata.get("image_count") == 3_600
            and previous_metadata.get("text_count") == 3_600
            and replacement_metadata.get("text_count") == 3_600
        ),
        "same_image_candidate_order": (
            previous_metadata.get("image_id_order_hash")
            == replacement_metadata.get("image_id_order_hash")
            and previous_metadata.get("image_ids") == replacement_metadata.get("image_ids")
        ),
        "same_caption_candidate_order": (
            previous_metadata.get("text_id_order_hash")
            == replacement_metadata.get("text_id_order_hash")
            and previous_metadata.get("text_ids") == replacement_metadata.get("text_ids")
        ),
        "same_caption_content": (
            previous_metadata.get("text_content_sha256")
            == replacement_metadata.get("text_content_sha256")
        ),
        "same_positive_pair_count": (
            previous_metadata.get("positive_pair_count")
            == replacement_metadata.get("positive_pair_count")
        ),
    }
    return checks


def _metric_differences(
    previous: dict[str, Any],
    replacement: dict[str, Any],
) -> dict[str, dict[str, float]]:
    previous_metrics = previous["metrics"]
    replacement_metrics = replacement["metrics"]
    differences: dict[str, dict[str, float]] = {}
    for metric in _RETRIEVAL_METRICS:
        previous_value = float(previous_metrics[metric])
        replacement_value = float(replacement_metrics[metric])
        absolute_difference = abs(replacement_value - previous_value)
        differences[metric] = {
            "previous": previous_value,
            "replacement": replacement_value,
            "absolute_difference": absolute_difference,
            "absolute_difference_percentage_points": absolute_difference * 100.0,
        }
    return differences


def _language_pair_check(
    english: dict[str, Any],
    japanese: dict[str, Any],
) -> dict[str, Any]:
    english_metadata = english["benchmark_metadata"]
    japanese_metadata = japanese["benchmark_metadata"]
    checks = {
        "same_dataset_checksum": (
            english_metadata.get("dataset_checksum")
            == japanese_metadata.get("dataset_checksum")
        ),
        "same_image_candidate_count": (
            english_metadata.get("image_count")
            == japanese_metadata.get("image_count")
            == 3_600
        ),
        "same_caption_candidate_count": (
            english_metadata.get("text_count")
            == japanese_metadata.get("text_count")
            == 3_600
        ),
        "same_image_candidate_order": (
            english_metadata.get("image_id_order_hash")
            == japanese_metadata.get("image_id_order_hash")
            and english_metadata.get("image_ids") == japanese_metadata.get("image_ids")
        ),
        "same_checkpoint_revision_and_hash": (
            english_metadata.get("checkpoint_revision")
            == japanese_metadata.get("checkpoint_revision")
            and english_metadata.get("checkpoint_sha256")
            == japanese_metadata.get("checkpoint_sha256")
        ),
        "same_source_commit": (
            english_metadata.get("source_commit_sha")
            == japanese_metadata.get("source_commit_sha")
        ),
    }
    return {
        "english_run_id": english_metadata["run_id"],
        "japanese_run_id": japanese_metadata["run_id"],
        "dataset_checksum": english_metadata.get("dataset_checksum"),
        "image_count": english_metadata.get("image_count"),
        "english_caption_count": english_metadata.get("text_count"),
        "japanese_caption_count": japanese_metadata.get("text_count"),
        "image_id_order_sha256": english_metadata.get("image_id_order_hash"),
        "checks": checks,
    }


def build_clean_xm3600_reproducibility_audit(
    results: list[dict[str, Any]],
    publication_states: dict[str, str],
    frozen_source_commit_sha: str,
) -> dict[str, Any]:
    previous_runs = _run_map(results, publication_states, "superseded")
    replacement_runs = _run_map(results, publication_states, "active")
    if set(previous_runs) != _EXPECTED_RUNS or set(replacement_runs) != _EXPECTED_RUNS:
        raise DataIntegrityError(
            "The clean XM3600 reproducibility audit requires exactly the seven expected "
            "previous and replacement model/language combinations."
        )

    pairs: list[dict[str, Any]] = []
    for model_key, language in sorted(_EXPECTED_RUNS):
        previous = previous_runs[(model_key, language)]
        replacement = replacement_runs[(model_key, language)]
        previous_metadata = previous["benchmark_metadata"]
        replacement_metadata = replacement["benchmark_metadata"]
        if previous["status"] != "completed" or replacement["status"] != "completed":
            raise DataIntegrityError(
                f"The {model_key}/{language} comparison requires two completed results."
            )
        if replacement_metadata.get("source_commit_sha") != frozen_source_commit_sha:
            raise DataIntegrityError(
                f"The replacement {model_key}/{language} result uses the wrong source commit."
            )
        if replacement_metadata.get("working_tree_dirty") is not False:
            raise DataIntegrityError(
                f"The replacement {model_key}/{language} result has a dirty worktree."
            )
        if previous_metadata.get("working_tree_dirty") not in {True, False}:
            raise DataIntegrityError(
                f"The previous {model_key}/{language} result has unknown worktree state."
            )
        if not isinstance(previous_metadata.get("source_commit_sha"), str):
            raise DataIntegrityError(
                f"The previous {model_key}/{language} result has no source commit."
            )
        checks = _pair_checks(previous, replacement)
        if not all(checks.values()):
            failed_checks = ", ".join(
                name for name, passed in checks.items() if not passed
            )
            raise DataIntegrityError(
                f"The {model_key}/{language} run conditions changed: {failed_checks}."
            )

        pairs.append(
            {
                "model_key": model_key,
                "model": replacement["model"],
                "language": language,
                "previous": {
                    "run_id": previous_metadata["run_id"],
                    "source_commit_sha": previous_metadata.get("source_commit_sha"),
                    "working_tree_dirty": previous_metadata.get("working_tree_dirty"),
                    "checkpoint_revision": previous_metadata.get("checkpoint_revision"),
                    "checkpoint_sha256": previous_metadata.get("checkpoint_sha256"),
                },
                "replacement": {
                    "run_id": replacement_metadata["run_id"],
                    "source_commit_sha": replacement_metadata.get("source_commit_sha"),
                    "working_tree_dirty": replacement_metadata.get("working_tree_dirty"),
                    "checkpoint_revision": replacement_metadata.get("checkpoint_revision"),
                    "checkpoint_sha256": replacement_metadata.get("checkpoint_sha256"),
                    "requested_batch_size": _configuration(replacement).get(
                        "batch_size_requested"
                    ),
                    "actual_image_batch_size": replacement_metadata.get(
                        "image_batch_size_actual"
                    ),
                    "actual_text_batch_size": replacement_metadata.get(
                        "text_batch_size_actual"
                    ),
                    "oom_history": replacement_metadata.get("oom_history", []),
                },
                "dataset_checksum": replacement_metadata.get("dataset_checksum"),
                "candidate_counts": replacement_metadata.get("candidate_counts"),
                "image_id_order_sha256": replacement_metadata.get(
                    "image_id_order_hash"
                ),
                "text_id_order_sha256": replacement_metadata.get("text_id_order_hash"),
                "text_content_sha256": replacement_metadata.get("text_content_sha256"),
                "checks": checks,
                "metrics": _metric_differences(previous, replacement),
            }
        )

    replacement_by_key = {
        _run_key(result): result for result in replacement_runs.values()
    }
    language_pairs = {
        model_key: _language_pair_check(
            replacement_by_key[(model_key, "en")],
            replacement_by_key[(model_key, "ja")],
        )
        for model_key in _LANGUAGE_PAIRED_MODELS
    }
    if any(
        not all(pair["checks"].values())
        for pair in language_pairs.values()
    ):
        raise DataIntegrityError(
            "At least one clean English/Japanese XM3600 pair has inconsistent "
            "candidate or checkpoint identity."
        )

    previous_dirty_count = sum(
        result["benchmark_metadata"].get("working_tree_dirty") is True
        for result in previous_runs.values()
    )
    previous_clean_count = sum(
        result["benchmark_metadata"].get("working_tree_dirty") is False
        for result in previous_runs.values()
    )
    dataset_checksums = {
        result["benchmark_metadata"].get("dataset_checksum")
        for result in replacement_runs.values()
    }
    if len(dataset_checksums) != 1:
        raise DataIntegrityError("The seven replacement runs use different dataset checksums.")

    maximum_absolute_difference = max(
        difference["absolute_difference"]
        for pair in pairs
        for difference in pair["metrics"].values()
    )
    return {
        "schema_version": "1.0.0",
        "status": "completed",
        "frozen_source_commit_sha": frozen_source_commit_sha,
        "dataset": {
            "dataset_id": "crossmodal3600",
            "checksum": next(iter(dataset_checksums)),
            "image_count_per_run": 3_600,
            "caption_count_per_language": 3_600,
            "retrieval_variant": "official_first_caption",
        },
        "previous_runs": {
            "count": len(previous_runs),
            "clean_worktree_count": previous_clean_count,
            "dirty_worktree_count": previous_dirty_count,
            "source_commit_shas": sorted(
                {
                    str(result["benchmark_metadata"].get("source_commit_sha"))
                    for result in previous_runs.values()
                }
            ),
        },
        "replacement_runs": {
            "count": len(replacement_runs),
            "clean_worktree_count": sum(
                result["benchmark_metadata"].get("working_tree_dirty") is False
                for result in replacement_runs.values()
            ),
            "source_commit_shas": sorted(
                {
                    str(result["benchmark_metadata"].get("source_commit_sha"))
                    for result in replacement_runs.values()
                }
            ),
        },
        "maximum_absolute_recall_difference": maximum_absolute_difference,
        "language_pairs": language_pairs,
        "pairs": pairs,
        "interpretation": (
            "All six recall differences are reported as measured. No score is rewritten or "
            "forced to match; these comparisons do not by themselves establish a causal "
            "language-understanding effect."
        ),
    }


def build_clean_official_xm3600_parity(
    historical_parity: dict[str, Any],
    official_reference_result: dict[str, Any],
    clean_result: dict[str, Any],
) -> dict[str, Any]:
    metadata = clean_result["benchmark_metadata"]
    configuration = _configuration(clean_result)
    official_metrics = official_reference_result.get("metrics")
    historical_metrics = historical_parity["comparison"]["metrics"]
    if not isinstance(official_metrics, dict):
        raise DataIntegrityError("The pinned official reference result has no metric object.")

    checks = {
        "clean_strict_source": (
            metadata.get("reproducibility_mode") == "strict"
            and metadata.get("working_tree_dirty") is False
            and metadata.get("source_commit_sha")
            == metadata.get("expected_source_commit_sha")
            and configuration.get("reproducibility_mode") == "strict"
            and configuration.get("expected_source_commit_sha")
            == metadata.get("source_commit_sha")
        ),
        "dataset_checksum_matches": (
            metadata.get("dataset_checksum")
            == historical_parity["dataset"]["checksum"]
        ),
        "candidate_counts_match": (
            metadata.get("image_count") == 3_600
            and metadata.get("text_count") == 3_600
            and metadata.get("positive_pair_count") == 3_600
            and metadata.get("candidate_counts") == {"images": 3_600, "texts": 3_600}
        ),
        "candidate_order_matches": (
            metadata.get("image_id_order_hash")
            == historical_parity["dataset"]["image_id_order_sha256"]
            and metadata.get("text_id_order_hash")
            == historical_parity["dataset"]["text_id_order_sha256"]
            and metadata.get("text_content_sha256")
            == historical_parity["dataset"]["caption_content_sha256"]
        ),
        "checkpoint_and_tokenizer_match": (
            configuration.get("model_key") == "openclip_multilingual_b32"
            and metadata.get("model_repo")
            == historical_parity["model"]["checkpoint_repository"]
            and metadata.get("checkpoint_revision")
            == historical_parity["model"]["checkpoint_revision"]
            and metadata.get("checkpoint_sha256")
            == historical_parity["model"]["vlbench_checkpoint_fingerprint_sha256"]
            and any(
                checkpoint_file.get("sha256")
                == historical_parity["model"]["checkpoint_file_sha256"]
                for checkpoint_file in metadata.get("checkpoint_files") or []
                if isinstance(checkpoint_file, dict)
            )
            and metadata.get("tokenizer_repository")
            == historical_parity["model"]["tokenizer_repository"]
            and metadata.get("tokenizer_revision")
            == historical_parity["model"]["tokenizer_revision"]
        ),
        "preprocessing_matches": all(
            metadata.get("preprocessing_config", {}).get(key) == value
            for key, value in historical_parity["model"]["preprocessing"].items()
        ),
        "official_caption_selection_and_filter_match": (
            metadata.get("one_caption_per_image") is True
            and isinstance(metadata.get("caption_selection"), str)
            and metadata["caption_selection"].startswith("captions[0]")
            and metadata.get("official_image_filtering")
            == {
                "rule": "include an image only when its image file exists",
                "reference_commit": historical_parity["reference"]["commit"],
                "included_count": 3_600,
                "excluded_count": 0,
                "excluded_image_ids": [],
            }
        ),
        "retrieval_protocol_matches": (
            clean_result["status"] == "completed"
            and clean_result["task"] == "retrieval"
            and clean_result["dataset"] == "crossmodal3600"
            and clean_result["model"] == historical_parity["reference"]["model"]
            and clean_result["language"] == "ja"
            and metadata.get("retrieval_variant") == "official_first_caption"
            and metadata.get("retrieval_variant_reference_commit")
            == historical_parity["reference"]["commit"]
            and metadata.get("precision") == "fp32"
            and metadata.get("device") == "cuda:0"
            and metadata.get("seed") == 0
            and configuration.get("batch_size_requested") == 16
            and metadata.get("prompt_policy") == "model_native_caption_text"
        ),
        "official_reference_result_matches_saved_parity": (
            official_reference_result.get("dataset") == "crossmodal3600"
            and official_reference_result.get("task") == "zeroshot_retrieval"
            and official_reference_result.get("language") == "ja"
            and official_reference_result.get("model")
            == historical_parity["reference"]["model"]
            and all(
                official_metrics.get(metric)
                == historical_metrics[metric]["official"]
                for metric in _RETRIEVAL_METRICS
            )
        ),
    }
    tolerance = float(historical_parity["comparison"]["absolute_tolerance"])
    metric_comparisons: dict[str, dict[str, Any]] = {}
    maximum_absolute_difference = 0.0
    matching_integer_hit_counts = True
    for metric in _RETRIEVAL_METRICS:
        official_value = float(official_metrics[metric])
        historical_official_value = float(historical_metrics[metric]["official"])
        if abs(official_value - historical_official_value) > 1e-15:
            raise DataIntegrityError(
                f"The saved official reference changed for {metric}."
            )
        vlbench_value = float(clean_result["metrics"][metric])
        absolute_difference = abs(vlbench_value - official_value)
        maximum_absolute_difference = max(
            maximum_absolute_difference,
            absolute_difference,
        )
        official_hit_count = historical_metrics[metric]["matching_hit_count"]
        vlbench_hit_count = round(vlbench_value * 3_600)
        matching_integer_hit_counts = (
            matching_integer_hit_counts and vlbench_hit_count == official_hit_count
        )
        metric_comparisons[metric] = {
            "direction": historical_metrics[metric]["direction"],
            "official": official_value,
            "vlbench": vlbench_value,
            "absolute_difference": absolute_difference,
            "official_hit_count": official_hit_count,
            "vlbench_hit_count": vlbench_hit_count,
        }
    within_tolerance = maximum_absolute_difference <= tolerance
    checks["six_recall_metrics_match"] = within_tolerance
    checks["integer_hit_counts_match"] = matching_integer_hit_counts
    status = "completed" if all(checks.values()) else "failed"
    result_metadata = {
        "run_id": metadata["run_id"],
        "result_file": f"../raw/{metadata['run_id']}.json",
        "source_commit": metadata.get("source_commit_sha"),
        "retrieval_variant": metadata.get("retrieval_variant"),
    }
    return {
        "schema_version": "1.0.0",
        "status": status,
        "parity_scope": (
            "One full-dataset end-to-end comparison: multilingual OpenCLIP ViT-B/32, "
            "Japanese, official_first_caption."
        ),
        "reference": {
            **historical_parity["reference"],
            "result_reused": True,
            "result_reuse_reason": (
                "The pinned official result and candidate-order fingerprints match the "
                "clean run; the official evaluator was not rerun."
            ),
        },
        "vlbench": result_metadata,
        "runtime": {
            "gpu": metadata.get("gpu_name"),
            "device": metadata.get("device"),
            "precision": metadata.get("precision"),
            "amp": metadata.get("amp_enabled"),
            "batch_size": metadata.get("batch_size_actual"),
            "seed": metadata.get("seed"),
            "pytorch": metadata.get("torch_version"),
            "cuda": metadata.get("torch_cuda_version"),
        },
        "dataset": {
            "revision": historical_parity["dataset"]["revision"],
            "checksum": metadata.get("dataset_checksum"),
            "split": "all 3,600 images",
            "images": metadata.get("image_count"),
            "captions": metadata.get("text_count"),
            "positive_pairs": metadata.get("positive_pair_count"),
            "image_id_order_sha256": metadata.get("image_id_order_hash"),
            "text_id_order_sha256": metadata.get("text_id_order_hash"),
            "caption_content_sha256": metadata.get("text_content_sha256"),
            "validation": {
                "official_first_caption_selection_matches_source": (
                    metadata.get("one_caption_per_image") is True
                    and isinstance(metadata.get("caption_selection"), str)
                    and metadata["caption_selection"].startswith("captions[0]")
                ),
                "ordered_image_ids_match_historical_reference": (
                    checks["candidate_order_matches"]
                ),
                "ordered_caption_ids_match_historical_reference": (
                    checks["candidate_order_matches"]
                ),
                "one_caption_per_image": metadata.get("one_caption_per_image") is True,
                "official_missing_image_filter_matches": (
                    metadata.get("official_image_filtering")
                    == {
                        "rule": "include an image only when its image file exists",
                        "reference_commit": historical_parity["reference"]["commit"],
                        "included_count": 3_600,
                        "excluded_count": 0,
                        "excluded_image_ids": [],
                    }
                ),
            },
        },
        "model": {
            "checkpoint_repository": metadata.get("model_repo"),
            "checkpoint_revision": metadata.get("checkpoint_revision"),
            "checkpoint_file_sha256": historical_parity["model"][
                "checkpoint_file_sha256"
            ],
            "vlbench_checkpoint_fingerprint_sha256": metadata.get("checkpoint_sha256"),
            "tokenizer_repository": metadata.get("tokenizer_repository"),
            "tokenizer_revision": metadata.get("tokenizer_revision"),
            "token_ids_match_for_all_candidates": (
                metadata.get("tokenizer_revision")
                == historical_parity["model"]["tokenizer_revision"]
            ),
            "preprocessing_matches": checks["preprocessing_matches"],
            "preprocessing": metadata.get("preprocessing_config"),
            "text_prompt_policy": metadata.get("prompt_policy"),
            "text_prompt_added": metadata.get("task_prompt") is not None,
        },
        "metric_definition": historical_parity["metric_definition"],
        "comparison": {
            "absolute_tolerance": tolerance,
            "maximum_absolute_difference": maximum_absolute_difference,
            "within_tolerance": within_tolerance,
            "same_integer_hit_counts": matching_integer_hit_counts,
            "difference_explanation": (
                historical_parity["comparison"]["difference_explanation"]
            ),
            "metrics": metric_comparisons,
        },
        "validation_checks": checks,
        "conclusion": (
            "Clean-run metric parity is verified within the pinned float32 aggregation "
            "tolerance for this model, language, checkpoint, preprocessing, tokenizer, "
            "and candidate set. This evidence does not establish universal model parity."
            if status == "completed"
            else "Parity validation failed; this result must not be described as parity-verified."
        ),
    }
