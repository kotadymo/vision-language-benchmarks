from __future__ import annotations

from typing import Any

from vision_language_benchmarks.registry import RETRIEVAL_VARIANTS, get_dataset

Result = dict[str, Any]


def retrieval_variant_info(result: Result) -> tuple[str | None, str | None]:
    if result.get("task") != "retrieval":
        return None, None

    metadata = result.get("benchmark_metadata", {})
    configuration = metadata.get("run_configuration", {})
    recorded = metadata.get("retrieval_variant")
    if recorded is None and isinstance(configuration, dict):
        recorded = configuration.get("retrieval_variant")
    if recorded in RETRIEVAL_VARIANTS:
        source = metadata.get("retrieval_variant_source")
        return recorded, str(source or "run_configuration")

    if result.get("dataset") != "crossmodal3600":
        return None, None
    language = result.get("language")
    image_count = metadata.get("image_count")
    text_count = metadata.get("text_count")
    expected_text_count = (
        get_dataset("crossmodal3600").expected_text_count(str(language))
        if isinstance(language, str)
        else None
    )
    if (
        image_count == get_dataset("crossmodal3600").expected_images
        and text_count == expected_text_count
        and metadata.get("positive_pair_count") == text_count
    ):
        return "all_captions", "legacy_candidate_counts"
    return None, "legacy_variant_unrecorded"


def retrieval_variant_label(variant: str | None) -> str:
    if variant == "official_first_caption":
        return "official_first_caption"
    if variant == "all_captions":
        return "all_captions"
    return "unspecified"
