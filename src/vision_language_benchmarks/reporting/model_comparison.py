from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from vision_language_benchmarks.errors import DataIntegrityError
from vision_language_benchmarks.registry import get_model
from vision_language_benchmarks.reporting.retrieval_variant import retrieval_variant_info

Result = dict[str, Any]
ComparisonSource = tuple[str, list[Result], dict[str, str]]

_PACKAGE_NAMES = {
    "japanese_clip": "japanese-clip",
    "open_clip": "open-clip-torch",
    "openclip_gemma": "openclip-gemma",
    "openvlclip_dinotxt": "openvlclip",
    "openvlclip_v07": "openvlclip",
    "transformers_siglip2": "transformers",
}

_RETRIEVAL_METRICS = (
    "image_retrieval_recall@1",
    "image_retrieval_recall@5",
    "image_retrieval_recall@10",
    "text_retrieval_recall@1",
    "text_retrieval_recall@5",
    "text_retrieval_recall@10",
)

MODEL_COMPARISON_FIELDS = (
    "comparison_condition",
    "dataset_key",
    "task",
    "protocol",
    "prompt_policy",
    "prompt_name",
    "model_input_policy",
    "language",
    "retrieval_variant",
    "precision",
    "split",
    "dataset_checksum",
    "image_count",
    "text_count",
    "text_encoding_count",
    "class_count",
    "model_key",
    "model",
    "run_id",
    "campaign",
    "publication_state",
    "embedding_dimension",
    "fp32_bytes_per_vector",
    "fp32_embedding_payload_bytes",
    "acc1_fraction",
    "acc1_correct_count",
    "acc1_total_count",
    "acc5_fraction",
    "acc5_correct_count",
    "acc5_total_count",
    *(
        field
        for metric in _RETRIEVAL_METRICS
        for field in (f"{metric}_fraction", f"{metric}_hits", f"{metric}_queries")
    ),
    "throughput_images_per_sec",
    "throughput_texts_per_sec",
    "inference_ms_median",
    "gpu_name",
    "gpu_peak_allocated_bytes",
    "gpu_peak_reserved_bytes",
    "batch_size_actual",
    "image_batch_size_actual",
    "text_batch_size_actual",
    "package",
    "package_version",
    "checkpoint_revision",
    "checkpoint_sha256",
    "model_checkpoint_size_bytes",
    "source_commit_sha",
    "raw_result_path",
)


def _metadata(result: Result) -> dict[str, Any]:
    return result["benchmark_metadata"]


def _configuration(result: Result) -> dict[str, Any]:
    value = _metadata(result).get("run_configuration", {})
    return value if isinstance(value, dict) else {}


def exact_hit_count(
    result: Result,
    metric_key: str,
    denominator_key: str,
) -> int | None:
    metric = result.get("metrics", {}).get(metric_key)
    if metric is None:
        return None
    metadata = _metadata(result)
    denominator = metadata.get(denominator_key)
    run_id = metadata.get("run_id", "unknown")
    if (
        not isinstance(metric, (int, float))
        or isinstance(metric, bool)
        or not math.isfinite(float(metric))
    ):
        raise DataIntegrityError(
            f"Run {run_id} has a non-finite or non-numeric value for {metric_key}."
        )
    if (
        not isinstance(denominator, int)
        or isinstance(denominator, bool)
        or denominator <= 0
    ):
        raise DataIntegrityError(
            f"Run {run_id} is missing a positive integer {denominator_key} "
            f"for {metric_key}."
        )
    exact_value = float(metric) * denominator
    hit_count = round(exact_value)
    if abs(exact_value - hit_count) > 1e-6 or not 0 <= hit_count <= denominator:
        raise DataIntegrityError(
            f"Run {run_id} metric {metric_key} does not resolve to an exact "
            f"integer hit count over {denominator} queries."
        )
    return int(hit_count)


def _dimension(result: Result) -> int | None:
    metadata = _metadata(result)
    configuration = _configuration(result)
    for value in (
        configuration.get("embedding_dimension"),
        metadata.get("embedding_dimension"),
        metadata.get("global_embedding_dimension"),
    ):
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    model_key = configuration.get("model_key")
    if isinstance(model_key, str):
        try:
            return get_model(model_key).global_embedding_dimension
        except ValueError:
            return None
    return None


def _condition_key(result: Result) -> tuple[Any, ...]:
    metadata = _metadata(result)
    configuration = _configuration(result)
    variant, _ = retrieval_variant_info(result)
    return (
        result.get("dataset"),
        result.get("task"),
        metadata.get("dataset_split"),
        result.get("language"),
        configuration.get("protocol"),
        variant,
        metadata.get("precision"),
        metadata.get("dataset_checksum"),
        metadata.get("image_count"),
        metadata.get("text_count"),
        metadata.get("class_count"),
        metadata.get("positive_pair_count"),
    )


def _is_comparable_condition(result: Result) -> bool:
    metadata = _metadata(result)
    configuration = _configuration(result)
    if metadata.get("precision") != "fp32":
        return False
    if result.get("task") == "classification":
        return (
            result.get("dataset") in {"cifar10", "cifar100"}
            and result.get("language") == "en"
            and configuration.get("protocol") in {"p0", "p1"}
        )
    return (
        result.get("task") == "retrieval"
        and result.get("dataset") == "crossmodal3600"
        and result.get("language") in {"en", "ja"}
        and configuration.get("protocol") == "p2"
        and retrieval_variant_info(result)[0] == "official_first_caption"
    )


def _checkpoint_size(metadata: Mapping[str, Any]) -> int | None:
    direct_size = metadata.get("model_checkpoint_size_bytes")
    if isinstance(direct_size, int) and not isinstance(direct_size, bool):
        return direct_size
    components = metadata.get("checkpoint_components")
    if isinstance(components, list):
        sizes = [
            component.get("size_bytes")
            for component in components
            if isinstance(component, dict)
            and component.get("kind") == "weight"
            and isinstance(component.get("size_bytes"), int)
            and not isinstance(component.get("size_bytes"), bool)
        ]
        if sizes:
            return sum(sizes)
    return None


def _package_info(metadata: Mapping[str, Any]) -> tuple[str, str | None]:
    adapter = metadata.get("adapter_name")
    package = metadata.get("model_library")
    if not isinstance(package, str) or not package:
        package = _PACKAGE_NAMES.get(str(adapter), "not recorded")
    version = metadata.get("model_library_version")
    if not isinstance(version, str) or not version:
        version = metadata.get("model_implementation_version")
    return package, version if isinstance(version, str) and version else None


def _condition_id(result: Result) -> str:
    metadata = _metadata(result)
    configuration = _configuration(result)
    variant, _ = retrieval_variant_info(result)
    parts = [
        str(result.get("dataset", "")),
        str(configuration.get("protocol", "")),
        str(result.get("language", "")),
        str(metadata.get("precision", "")),
    ]
    if variant is not None:
        parts.insert(3, variant)
    return ":".join(parts)


def _comparison_row(
    result: Result,
    *,
    campaign: str,
    current_campaign: str,
    publication_state: str,
) -> dict[str, Any]:
    metadata = _metadata(result)
    configuration = _configuration(result)
    metrics = result.get("metrics", {})
    dimension = _dimension(result)
    precision = metadata.get("precision")
    bytes_per_vector = dimension * 4 if precision == "fp32" and dimension else None
    image_count = metadata.get("image_count")
    text_count = metadata.get("text_count")
    payload_bytes = (
        (image_count + text_count) * bytes_per_vector
        if isinstance(image_count, int)
        and not isinstance(image_count, bool)
        and isinstance(text_count, int)
        and not isinstance(text_count, bool)
        and bytes_per_vector is not None
        else None
    )
    model_key = configuration.get("model_key", result.get("model"))
    try:
        model_name = get_model(str(model_key)).name
    except ValueError:
        model_name = str(result.get("model", model_key))
    package, package_version = _package_info(metadata)
    input_policy = metadata.get("model_input_policy")
    raw_result_path = (
        f"raw/{metadata['run_id']}.json"
        if campaign == current_campaign
        else f"../{campaign}/raw/{metadata['run_id']}.json"
    )

    row: dict[str, Any] = {
        "comparison_condition": _condition_id(result),
        "dataset_key": result.get("dataset"),
        "task": result.get("task"),
        "protocol": configuration.get("protocol"),
        "prompt_policy": metadata.get("prompt_policy"),
        "prompt_name": metadata.get("prompt_name"),
        "model_input_policy": (
            json.dumps(input_policy, ensure_ascii=False, sort_keys=True)
            if isinstance(input_policy, dict)
            else None
        ),
        "language": result.get("language"),
        "retrieval_variant": retrieval_variant_info(result)[0],
        "precision": precision,
        "split": metadata.get("dataset_split"),
        "dataset_checksum": metadata.get("dataset_checksum"),
        "image_count": image_count,
        "text_count": text_count,
        "text_encoding_count": metadata.get(
            "text_encoding_count",
            text_count,
        ),
        "class_count": metadata.get("class_count"),
        "model_key": model_key,
        "model": model_name,
        "run_id": metadata.get("run_id"),
        "campaign": campaign,
        "publication_state": publication_state,
        "embedding_dimension": dimension,
        "fp32_bytes_per_vector": bytes_per_vector,
        "fp32_embedding_payload_bytes": payload_bytes,
        "acc1_fraction": metrics.get("acc1"),
        "acc1_correct_count": exact_hit_count(result, "acc1", "image_count"),
        "acc1_total_count": image_count if "acc1" in metrics else None,
        "acc5_fraction": metrics.get("acc5"),
        "acc5_correct_count": exact_hit_count(result, "acc5", "image_count"),
        "acc5_total_count": image_count if "acc5" in metrics else None,
        "throughput_images_per_sec": metadata.get("throughput_images_per_sec"),
        "throughput_texts_per_sec": metadata.get("throughput_texts_per_sec"),
        "inference_ms_median": metadata.get("inference_ms_median"),
        "gpu_name": metadata.get("gpu_name"),
        "gpu_peak_allocated_bytes": metadata.get("gpu_peak_allocated_bytes"),
        "gpu_peak_reserved_bytes": metadata.get("gpu_peak_reserved_bytes"),
        "batch_size_actual": metadata.get("batch_size_actual"),
        "image_batch_size_actual": metadata.get("image_batch_size_actual"),
        "text_batch_size_actual": metadata.get("text_batch_size_actual"),
        "package": package,
        "package_version": package_version,
        "checkpoint_revision": metadata.get("checkpoint_revision"),
        "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "model_checkpoint_size_bytes": _checkpoint_size(metadata),
        "source_commit_sha": metadata.get("source_commit_sha"),
        "raw_result_path": raw_result_path,
    }
    for metric_key in _RETRIEVAL_METRICS:
        denominator_key = (
            "text_count" if metric_key.startswith("image_retrieval_") else "image_count"
        )
        row[f"{metric_key}_fraction"] = metrics.get(metric_key)
        row[f"{metric_key}_hits"] = exact_hit_count(
            result,
            metric_key,
            denominator_key,
        )
        row[f"{metric_key}_queries"] = (
            metadata.get(denominator_key) if metric_key in metrics else None
        )
    return row


def build_model_comparison_rows(
    primary_results: list[Result],
    primary_states: Mapping[str, str],
    primary_campaign: str,
    comparison_sources: Sequence[ComparisonSource],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    primary_groups: dict[tuple[Any, ...], list[Result]] = defaultdict(list)
    for primary in primary_results:
        if primary.get("status") != "completed" or not _is_comparable_condition(primary):
            continue
        primary_metadata = _metadata(primary)
        primary_run_id = str(primary_metadata["run_id"])
        primary_key = _condition_key(primary)
        current_state = primary_states.get(primary_run_id)
        if current_state not in {"active", "historical"}:
            raise DataIntegrityError(
                f"Primary comparison run {primary_run_id} has no eligible publication state."
            )
        primary_groups[primary_key].append(primary)

    for primary_key, primary_group in primary_groups.items():
        condition_rows = [
            _comparison_row(
                primary,
                campaign=primary_campaign,
                current_campaign=primary_campaign,
                publication_state=primary_states[
                    str(_metadata(primary)["run_id"])
                ],
            )
            for primary in primary_group
        ]
        primary_model_keys = {
            _configuration(primary).get("model_key") for primary in primary_group
        }
        selected: dict[
            tuple[Any, int | None],
            tuple[tuple[int, int], Result, str, str],
        ] = {}
        for source_index, (campaign, candidates, candidate_states) in enumerate(
            comparison_sources
        ):
            for candidate in candidates:
                if (
                    candidate.get("status") != "completed"
                    or not _is_comparable_condition(candidate)
                    or _condition_key(candidate) != primary_key
                ):
                    continue
                candidate_metadata = _metadata(candidate)
                candidate_run_id = str(candidate_metadata["run_id"])
                state = candidate_states.get(candidate_run_id)
                if state not in {"active", "historical"}:
                    raise DataIntegrityError(
                        f"Comparison run {candidate_run_id} has no eligible publication state."
                    )
                configuration = _configuration(candidate)
                candidate_model_key = configuration.get("model_key")
                if not isinstance(candidate_model_key, str):
                    raise DataIntegrityError(
                        f"Comparison run {candidate_run_id} has no registered model key."
                    )
                dimension = _dimension(candidate)
                if (
                    candidate_model_key in primary_model_keys
                    or (
                        candidate_model_key == "embeddinggemma2"
                        and dimension != 768
                    )
                ):
                    continue
                selection_key = (candidate_model_key, dimension)
                rank = (int(state == "active"), source_index)
                previous = selected.get(selection_key)
                if previous is None or rank > previous[0]:
                    selected[selection_key] = (rank, candidate, campaign, state)
                elif rank == previous[0]:
                    raise DataIntegrityError(
                        "Matched historical comparison is ambiguous for "
                        f"{candidate_model_key} in {campaign}: both {candidate_run_id} "
                        f"and {previous[1]['benchmark_metadata']['run_id']} have equal priority."
                    )

        selected_candidates = sorted(
            selected.values(),
            key=lambda item: (
                str(_configuration(item[1]).get("model_key", "")),
                _dimension(item[1]) or 0,
            ),
        )
        for (_, candidate, campaign, candidate_state) in selected_candidates:
            condition_rows.append(
                _comparison_row(
                    candidate,
                    campaign=campaign,
                    current_campaign=primary_campaign,
                    publication_state=candidate_state,
                )
            )
        rows.extend(condition_rows)

    return sorted(
        rows,
        key=lambda row: (
            str(row["dataset_key"]),
            str(row["protocol"]),
            str(row["language"]),
            str(row["retrieval_variant"] or ""),
            0 if row["campaign"] == primary_campaign else 1,
            str(row["model_key"]),
        ),
    )


def render_model_comparison_section(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["dataset_key"]),
                str(row["task"]),
                str(row["protocol"]),
                str(row["language"]),
                str(row["retrieval_variant"] or ""),
            )
        ].append(row)

    lines = [
        "## Matched historical model comparisons",
        "",
        "Rows are matched to each new benchmark run by dataset checksum, split, task, "
        "protocol, language, precision, sample/candidate counts, and retrieval variant. "
        "Only lifecycle-eligible completed historical results are included. Matching "
        "protocol labels do not imply identical model-side preprocessing or instructions.",
    ]
    lines.append(
        "P0 keeps the common 18-template candidate-text protocol; P1 remains model-specific."
    )
    if any(row.get("model_key") == "siglip2_base_224" for row in rows):
        lines.append(
            "The historical direct Transformers SigLIP 2 CIFAR results predate the "
            "PAD-derived attention mask used by OpenVLCLIP v0.7.0. On the same first "
            "128 CIFAR-10 P0 images and 180 prompts, OpenVLCLIP features matched the "
            "current masked direct adapter exactly, while the legacy unmasked text "
            "features differed (maximum absolute difference 6.9498; mean prompt "
            "cosine 0.6882). The historical 93.62% Top-1 result is retained unchanged "
            "as context, not a controlled like-for-like quality comparison. The subset "
            "diagnostic is `validation/siglip2-p0-parity.json`."
        )
    if any(row.get("model_key") == "openvlclip_qwen3_vl_embedding_2b" for row in rows):
        lines.append(
            "Qwen's fixed classification instruction and separate retrieval query/document "
            "instructions are recorded in the comparison CSV and raw JSON. Qwen retrieval "
            "throughput counts both text-encoding passes."
        )
    lines.extend(
        [
            "Machine-readable rows: [model-comparison.csv](model-comparison.csv); selection "
            "details: [model-comparison-audit.json](validation/model-comparison-audit.json).",
            "",
        ]
    )
    for key, items in sorted(grouped.items()):
        dataset_key, task, protocol, language, variant = key
        title = (
            f"{dataset_key.upper()} · {protocol.upper()} · {language.upper()} · FP32"
        )
        if variant:
            title += f" · {variant}"
        if language == "ja":
            title += " · exploratory"
        lines.extend([f"### {title}", ""])
        if task == "classification":
            lines.extend(
                [
                    "| Model | Top-1 (correct/total) | Top-5 (correct/total) | "
                    "Images/s | Peak allocated/reserved (GiB) | Embedding/vector bytes | "
                    "FP32 image+text payload | Package | Source commit | Raw result |",
                    "|---|---:|---:|---:|---:|---|---:|---|---|---|",
                ]
            )
            for row in items:
                values = (
                    row["model"],
                    _metric_cell(row, "acc1", "acc1_correct_count", "acc1_total_count"),
                    _metric_cell(row, "acc5", "acc5_correct_count", "acc5_total_count"),
                    _format_rate(row.get("throughput_images_per_sec")),
                    _format_memory(row),
                    _format_vector_storage(row),
                    _format_payload(row),
                    _format_package(row),
                    _short_sha(row.get("source_commit_sha")),
                    _raw_link(row),
                )
                lines.append("| " + " | ".join(_markdown_cell(value) for value in values) + " |")
        else:
            lines.extend(
                [
                    "| Model | T→I R@1 | T→I R@5 | T→I R@10 | "
                    "I→T R@1 | I→T R@5 | I→T R@10 | Images/s | Texts/s | "
                    "Peak allocated/reserved (GiB) | Embedding/vector bytes | "
                    "FP32 image+text payload | Package | Source commit | Raw result |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|",
                ]
            )
            for row in items:
                values = (
                    row["model"],
                    *(
                        _metric_cell(
                            row,
                            metric_key,
                            f"{metric_key}_hits",
                            f"{metric_key}_queries",
                        )
                        for metric_key in _RETRIEVAL_METRICS
                    ),
                    _format_rate(row.get("throughput_images_per_sec")),
                    _format_rate(row.get("throughput_texts_per_sec")),
                    _format_memory(row),
                    _format_vector_storage(row),
                    _format_payload(row),
                    _format_package(row),
                    _short_sha(row.get("source_commit_sha")),
                    _raw_link(row),
                )
                lines.append("| " + " | ".join(_markdown_cell(value) for value in values) + " |")
        lines.append("")

    lines.extend(
        [
            "### Interpretation and comparability notes",
            "",
            "- Each recall/accuracy percentage includes the exact hit count and query denominator.",
            "- The payload column is derived as `(image count + text/prompt count) × "
            "embedding dimension × 4 bytes`; it is an FP32 vector payload calculation, "
            "not a measured on-disk file size, and excludes indexes/metadata.",
            "- Equal embedding dimensions do not make model spaces interchangeable. "
            "Storage calculations describe vector payload only, not cross-model index "
            "compatibility.",
            "- Legacy raw results that do not record an embedding dimension are left blank "
            "for storage calculations rather than inferred from the model name.",
            "- Throughput and memory are actual recorded values. The historical campaigns "
            "used the same GPU and FP32 here, but batch sizes, software revisions, and "
            "source commits differ; consult the CSV and raw records for details.",
            "- Japanese captions were evaluated as supplied, without translation. Japanese "
            "XM3600 retrieval is exploratory and is not evidence of Japanese semantic quality.",
            "",
        ]
    )
    return lines


def _metric_cell(
    row: Mapping[str, Any],
    metric_name: str,
    hits_key: str,
    queries_key: str,
) -> str:
    value = row.get(f"{metric_name}_fraction")
    hits = row.get(hits_key)
    queries = row.get(queries_key)
    if not isinstance(value, (int, float)):
        return "—"
    suffix = (
        f" ({hits:,}/{queries:,})"
        if isinstance(hits, int)
        and not isinstance(hits, bool)
        and isinstance(queries, int)
        and not isinstance(queries, bool)
        else ""
    )
    return f"{float(value) * 100:.2f}%{suffix}"


def _format_rate(value: Any) -> str:
    return "—" if not isinstance(value, (int, float)) else f"{float(value):.2f}"


def _format_memory(row: Mapping[str, Any]) -> str:
    allocated = row.get("gpu_peak_allocated_bytes")
    reserved = row.get("gpu_peak_reserved_bytes")
    if not isinstance(allocated, int) or not isinstance(reserved, int):
        return "—"
    return f"{allocated / 1024**3:.2f} / {reserved / 1024**3:.2f}"


def _format_vector_storage(row: Mapping[str, Any]) -> str:
    dimension = row.get("embedding_dimension")
    vector_bytes = row.get("fp32_bytes_per_vector")
    if not isinstance(dimension, int) or not isinstance(vector_bytes, int):
        return "not recorded"
    return f"{dimension:,}D · {vector_bytes:,} B/vector"


def _format_payload(row: Mapping[str, Any]) -> str:
    value = row.get("fp32_embedding_payload_bytes")
    if not isinstance(value, int):
        return "not recorded"
    return f"{value / 1024**2:.2f} MiB"


def _format_package(row: Mapping[str, Any]) -> str:
    package = row.get("package")
    version = row.get("package_version")
    if not isinstance(package, str) or package == "not recorded":
        return "not recorded"
    if not isinstance(version, str):
        return f"{package} (version not recorded)"
    return f"{package}=={version}"


def _short_sha(value: Any) -> str:
    return value[:12] if isinstance(value, str) and len(value) == 40 else "not recorded"


def _raw_link(row: Mapping[str, Any]) -> str:
    run_id = row.get("run_id")
    path = row.get("raw_result_path")
    if not isinstance(run_id, str) or not isinstance(path, str):
        return "not recorded"
    return f"[{run_id}]({path})"


def _markdown_cell(value: Any) -> str:
    return str(value if value is not None else "—").replace("|", "\\|").replace("\n", " ")
