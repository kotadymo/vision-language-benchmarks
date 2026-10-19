from __future__ import annotations

from collections import defaultdict
from typing import Any

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
        if not isinstance(image_order_hash, str) or not image_order_hash:
            continue
        if not isinstance(dataset_checksum, str) or not dataset_checksum:
            continue

        key = (
            str(result["dataset"]),
            str(configuration.get("model_key", result["model"])),
            str(configuration.get("protocol", "")),
            str(metadata.get("precision", "")),
            dataset_checksum,
            str(metadata.get("dataset_split", "")),
            str(metadata.get("image_count", "")),
            image_order_hash,
            str(metadata.get("seed", "")),
            _device_identity(metadata),
            str(metadata.get("gpu_name", "")),
            str(metadata.get("gpu_vram_bytes", "")),
            str(metadata.get("driver_version", "")),
            str(metadata.get("torch_version", "")),
            str(metadata.get("torch_cuda_version", "")),
            str(metadata.get("source_commit_sha", "")),
            str(metadata.get("checkpoint_revision", "")),
            str(metadata.get("checkpoint_sha256", "")),
        )
        grouped[key][result["language"]].append(result)

    pairs: list[tuple[Result, Result]] = []
    for languages in grouped.values():
        english = languages["en"]
        japanese = languages["ja"]
        if len(english) == 1 and len(japanese) == 1:
            pairs.append((english[0], japanese[0]))

    return sorted(
        pairs,
        key=lambda pair: (
            str(pair[0]["dataset"]),
            str(pair[0]["benchmark_metadata"]["run_configuration"].get("protocol", "")),
            str(pair[0]["benchmark_metadata"].get("precision", "")),
            str(
                pair[0]["benchmark_metadata"]["run_configuration"].get(
                    "model_key", pair[0]["model"]
                )
            ),
        ),
    )
