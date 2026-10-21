from __future__ import annotations

from collections import defaultdict
from typing import Any

from vision_language_benchmarks.reporting.retrieval_variant import (
    retrieval_variant_info,
)

Result = dict[str, Any]


def _device_identity(metadata: dict[str, Any]) -> str:
    device = str(metadata.get("device", ""))
    # The benchmark runner uses CUDA device 0 for an unindexed CUDA request.
    return "cuda:0" if device == "cuda" else device


def paired_language_results(results: list[Result]) -> list[tuple[Result, Result]]:
    grouped: dict[tuple[str, ...], dict[str, list[Result]]] = defaultdict(
        lambda: {"en": [], "ja": []}
    )
    for result in results:
        if (
            result.get("status") != "completed"
            or result.get("task") != "retrieval"
            or result.get("language") not in {"en", "ja"}
        ):
            continue

        metadata = result["benchmark_metadata"]
        configuration = metadata.get("run_configuration", {})
        if not isinstance(configuration, dict):
            continue
        image_order_hash = metadata.get("image_id_order_hash")
        dataset_checksum = metadata.get("dataset_checksum")
        candidate_count = metadata.get("text_count")
        image_count = metadata.get("image_count")
        source_commit_sha = metadata.get("source_commit_sha")
        dataset_revision = metadata.get("dataset_revision")
        checkpoint_revision = metadata.get("checkpoint_revision")
        checkpoint_sha256 = metadata.get("checkpoint_sha256")
        preprocessing_hash = metadata.get("preprocessing_config_sha256")
        tokenizer_version = metadata.get("tokenizer_implementation_version")
        text_order_hash = metadata.get("text_id_order_hash")
        if not isinstance(image_order_hash, str) or not image_order_hash:
            continue
        if not isinstance(dataset_checksum, str) or not dataset_checksum:
            continue
        if (
            not isinstance(candidate_count, int)
            or isinstance(candidate_count, bool)
            or not isinstance(image_count, int)
            or isinstance(image_count, bool)
        ):
            continue
        if not all(
            isinstance(value, str) and value
            for value in (
                source_commit_sha,
                dataset_revision,
                checkpoint_revision,
                checkpoint_sha256,
                preprocessing_hash,
                tokenizer_version,
                text_order_hash,
            )
        ):
            continue
        image_ids = metadata.get("image_ids")
        text_ids = metadata.get("text_ids")
        if (
            not isinstance(image_ids, list)
            or len(image_ids) != image_count
            or not isinstance(text_ids, list)
            or len(text_ids) != candidate_count
        ):
            continue
        variant, _ = retrieval_variant_info(result)
        if result["dataset"] == "crossmodal3600" and variant is None:
            continue
        if (
            variant == "official_first_caption"
            and (
                metadata.get("one_caption_per_image") is not True
                or metadata.get("image_count") != candidate_count
            )
        ):
            continue
        key = (
            str(result["dataset"]),
            str(configuration.get("model_key", result["model"])),
            str(configuration.get("protocol", "")),
            str(variant),
            str(metadata.get("precision", "")),
            str(
                configuration.get(
                    "embedding_dimension",
                    metadata.get("embedding_dimension", ""),
                )
            ),
            dataset_checksum,
            dataset_revision,
            str(metadata.get("dataset_split", "")),
            str(metadata.get("image_count", "")),
            image_order_hash,
            str(candidate_count),
            str(metadata.get("positive_pair_count", "")),
            text_order_hash,
            str(metadata.get("preprocessing_version", "")),
            str(metadata.get("preprocessing_config_sha256", "")),
            str(metadata.get("tokenizer_implementation_version", "")),
            str(metadata.get("tokenizer_repository", "")),
            str(metadata.get("tokenizer_revision", "")),
            str(metadata.get("prompt_policy", "")),
            str(metadata.get("prompt_name", "")),
            str(metadata.get("task_prompt", "")),
            str(metadata.get("seed", "")),
            _device_identity(metadata),
            str(metadata.get("gpu_name", "")),
            str(metadata.get("gpu_vram_bytes", "")),
            str(metadata.get("driver_version", "")),
            str(metadata.get("torch_version", "")),
            str(metadata.get("torch_cuda_version", "")),
            source_commit_sha,
            checkpoint_revision,
            checkpoint_sha256,
        )
        grouped[key][result["language"]].append(result)

    pairs: list[tuple[Result, Result]] = []
    for key, languages in grouped.items():
        english = languages["en"]
        japanese = languages["ja"]
        if len(english) == 1 and len(japanese) == 1:
            english_metadata = english[0]["benchmark_metadata"]
            japanese_metadata = japanese[0]["benchmark_metadata"]
            english_image_ids = english_metadata.get("image_ids")
            japanese_image_ids = japanese_metadata.get("image_ids")
            if (
                isinstance(english_image_ids, list)
                and isinstance(japanese_image_ids, list)
                and english_image_ids != japanese_image_ids
            ):
                continue
            if key[3] == "official_first_caption":
                english_text_ids = english_metadata.get("text_ids")
                japanese_text_ids = japanese_metadata.get("text_ids")
                if not isinstance(english_text_ids, list) or not isinstance(
                    japanese_text_ids, list
                ):
                    continue
                if english_text_ids != japanese_text_ids:
                    continue
            pairs.append((english[0], japanese[0]))

    return sorted(
        pairs,
        key=lambda pair: (
            str(pair[0]["dataset"]),
            str(pair[0]["benchmark_metadata"]["run_configuration"].get("protocol", "")),
            str(retrieval_variant_info(pair[0])[0]),
            str(pair[0]["benchmark_metadata"].get("precision", "")),
            str(
                pair[0]["benchmark_metadata"]
                .get("run_configuration", {})
                .get("embedding_dimension", "")
            ),
            str(
                pair[0]["benchmark_metadata"]["run_configuration"].get(
                    "model_key", pair[0]["model"]
                )
            ),
        ),
    )
