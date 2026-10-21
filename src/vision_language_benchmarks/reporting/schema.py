from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema

from vision_language_benchmarks.embedding_dimensions import resolve_embedding_dimension
from vision_language_benchmarks.registry import (
    RETRIEVAL_VARIANTS,
    get_dataset,
)

_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@lru_cache(maxsize=1)
def result_schema() -> dict[str, Any]:
    schema_path = Path(__file__).with_name("result.schema.json")
    return json.loads(schema_path.read_text(encoding="utf-8"))


def validate_result(result: dict[str, Any]) -> None:
    jsonschema.validate(instance=result, schema=result_schema())
    metadata = result["benchmark_metadata"]
    if result["status"] != metadata["status"]:
        raise ValueError("Result and benchmark metadata statuses must match.")
    if metadata.get("reproducibility_mode") == "strict":
        source_commit_sha = metadata.get("source_commit_sha")
        expected_source_commit_sha = metadata.get("expected_source_commit_sha")
        configuration = metadata.get("run_configuration")
        if metadata.get("working_tree_dirty") is not False:
            raise ValueError("Strict reproducibility results must have a clean worktree.")
        if (
            not isinstance(source_commit_sha, str)
            or not _COMMIT_PATTERN.fullmatch(source_commit_sha)
            or expected_source_commit_sha != source_commit_sha
        ):
            raise ValueError(
                "Strict reproducibility results must match the expected full source commit."
            )
        if (
            not isinstance(configuration, dict)
            or configuration.get("reproducibility_mode") != "strict"
            or configuration.get("expected_source_commit_sha") != source_commit_sha
        ):
            raise ValueError(
                "Strict reproducibility run configuration must match its source commit."
            )
        if not isinstance(metadata.get("staging_directory"), str):
            raise ValueError(
                "Strict reproducibility results must identify their staging directory."
            )
    configuration = metadata.get("run_configuration")
    configured_dimension = (
        configuration.get("embedding_dimension")
        if isinstance(configuration, dict)
        else None
    )
    embedding_dimension = metadata.get("embedding_dimension")
    if embedding_dimension is not None or configured_dimension is not None:
        if embedding_dimension != configured_dimension:
            raise ValueError(
                "Result metadata and run configuration embedding dimensions must match."
            )
        model_key = configuration.get("model_key") if isinstance(configuration, dict) else None
        resolved_dimension = resolve_embedding_dimension(model_key, embedding_dimension)
        if resolved_dimension != embedding_dimension:
            raise ValueError("Embedding dimension metadata is inconsistent.")
        if result["status"] == "completed":
            if metadata.get("embedding_dimension_strategy") != "leading-dimensions-l2-v1":
                raise ValueError(
                    "Completed embedding dimension results must identify the truncation strategy."
                )
            if metadata.get("native_embedding_dimension") != 768:
                raise ValueError(
                    "Completed embedding dimension results must identify the 768D native space."
                )
            if metadata.get("embedding_generation_mode") != "native_truncate_dim":
                raise ValueError(
                    "Completed embedding dimension results must use native truncate_dim inference."
                )
            if (
                metadata.get("adapter_name") != "openclip_gemma"
                or metadata.get("model_implementation_version") != "0.4.0"
            ):
                raise ValueError(
                    "Completed embedding dimension results require openclip-gemma==0.4.0."
                )
    retrieval_variant = metadata.get("retrieval_variant")
    if retrieval_variant is not None:
        if retrieval_variant not in RETRIEVAL_VARIANTS:
            raise ValueError(f"Unsupported retrieval variant: {retrieval_variant!r}.")
        if result["task"] != "retrieval" or result["dataset"] != "crossmodal3600":
            raise ValueError("Retrieval variants are valid only for XM3600 retrieval results.")
        configuration = metadata.get("run_configuration")
        if (
            not isinstance(configuration, dict)
            or configuration.get("retrieval_variant") != retrieval_variant
        ):
            raise ValueError(
                "Result metadata and run configuration retrieval variants must match."
            )
    if result["status"] == "completed":
        metrics = result.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError("Completed results must contain measured metrics.")
        required_metrics = (
            {"acc1", "acc5"}
            if result["task"] == "classification"
            else {
                "image_retrieval_recall@1",
                "image_retrieval_recall@5",
                "image_retrieval_recall@10",
                "text_retrieval_recall@1",
                "text_retrieval_recall@5",
                "text_retrieval_recall@10",
            }
        )
        if missing := required_metrics.difference(metrics):
            raise ValueError(
                f"Completed {result['task']} results are missing metrics: "
                f"{', '.join(sorted(missing))}."
            )
        if retrieval_variant is not None:
            image_count = metadata.get("image_count")
            text_count = metadata.get("text_count")
            candidates = metadata.get("candidate_counts")
            if (
                not isinstance(image_count, int)
                or not isinstance(text_count, int)
                or not isinstance(candidates, dict)
                or candidates.get("images") != image_count
                or candidates.get("texts") != text_count
            ):
                raise ValueError("Completed XM3600 retrieval candidate counts are inconsistent.")
            expected_count = get_dataset("crossmodal3600").expected_text_count(
                result["language"], retrieval_variant
            )
            if expected_count is not None and text_count != expected_count:
                raise ValueError(
                    f"Completed {retrieval_variant} run has {text_count} text candidates; "
                    f"expected {expected_count}."
                )
            if metadata.get("positive_pair_count") != text_count:
                raise ValueError(
                    "Every XM3600 text candidate must have exactly one positive image."
                )
            image_ids = metadata.get("image_ids")
            text_ids = metadata.get("text_ids")
            if not isinstance(image_ids, list) or len(image_ids) != image_count:
                raise ValueError("Completed XM3600 results must record ordered image IDs.")
            if not isinstance(text_ids, list) or len(text_ids) != text_count:
                raise ValueError("Completed XM3600 results must record ordered text IDs.")
            if not metadata.get("image_id_order_hash") or not metadata.get(
                "text_id_order_hash"
            ):
                raise ValueError("Completed XM3600 results must record ordered candidate hashes.")
            if not metadata.get("text_content_sha256"):
                raise ValueError("Completed XM3600 results must hash the selected caption text.")
            if retrieval_variant == "official_first_caption":
                if metadata.get("one_caption_per_image") is not True:
                    raise ValueError(
                        "official_first_caption results must contain one caption per image."
                    )
                if image_count != text_count:
                    raise ValueError(
                        "official_first_caption image and caption candidate counts must match."
                    )
                if text_ids != [f"{image_id}#0" for image_id in image_ids]:
                    raise ValueError(
                        "official_first_caption text IDs must follow the ordered image IDs."
                    )
    elif "metrics" in result:
        raise ValueError("Unsuccessful results must not contain measured metrics.")
