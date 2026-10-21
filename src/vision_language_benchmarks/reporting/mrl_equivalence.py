from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike


def compare_native_mrl_batch(
    model: Any,
    inputs: Any,
    modality: Literal["image", "text"],
    dimensions: Sequence[int],
    torch_module: Any,
    *,
    reference_dimension: int = 768,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> dict[int, dict[str, Any]]:
    if modality not in {"image", "text"}:
        raise ValueError("modality must be 'image' or 'text'.")
    ordered_dimensions = tuple(dimensions)
    if not ordered_dimensions or len(set(ordered_dimensions)) != len(ordered_dimensions):
        raise ValueError("dimensions must be a non-empty sequence without duplicates.")
    if any(
        not isinstance(dimension, int)
        or isinstance(dimension, bool)
        or dimension < 1
        or dimension > reference_dimension
        for dimension in ordered_dimensions
    ):
        raise ValueError(
            f"dimensions must be integers between 1 and {reference_dimension}."
        )
    if rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be non-negative.")

    encoder = getattr(model, f"encode_{modality}", None)
    if not callable(encoder):
        raise TypeError(f"model must provide a callable encode_{modality} method.")

    comparisons: dict[int, dict[str, Any]] = {}
    with torch_module.inference_mode():
        full_embeddings = encoder(
            inputs,
            normalize=False,
            truncate_dim=reference_dimension,
        )
        if full_embeddings.ndim != 2 or full_embeddings.shape[1] != reference_dimension:
            raise ValueError(
                f"Expected {reference_dimension}D full embeddings, received "
                f"shape {tuple(full_embeddings.shape)}."
            )

        for dimension in ordered_dimensions:
            native_embeddings = encoder(
                inputs,
                normalize=False,
                truncate_dim=dimension,
            )
            reference_embeddings = torch_module.nn.functional.normalize(
                full_embeddings[..., :dimension],
                p=2,
                dim=-1,
            )
            if tuple(native_embeddings.shape) != tuple(reference_embeddings.shape):
                raise ValueError(
                    "Native and reference output shapes differ for "
                    f"{modality} truncate_dim={dimension}: "
                    f"{tuple(native_embeddings.shape)} and "
                    f"{tuple(reference_embeddings.shape)}."
                )
            native_array = np.asarray(
                native_embeddings.detach().float().cpu().numpy(),
                dtype=np.float32,
            )
            reference_array = np.asarray(
                reference_embeddings.detach().float().cpu().numpy(),
                dtype=np.float32,
            )
            comparisons[dimension] = {
                "native_embeddings": native_array,
                "reference_embeddings": reference_array,
                "comparison": compare_embedding_arrays(
                    native_array,
                    reference_array,
                    rtol=rtol,
                    atol=atol,
                ),
            }
    return comparisons


def compare_embedding_arrays(
    native: ArrayLike,
    reference: ArrayLike,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> dict[str, Any]:
    if rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be non-negative.")

    native_array = np.asarray(native, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if native_array.ndim != 2 or reference_array.ndim != 2:
        raise ValueError("Embedding arrays must be two-dimensional.")
    if native_array.shape != reference_array.shape:
        raise ValueError(
            "Native and reference embedding shapes must match; received "
            f"{native_array.shape} and {reference_array.shape}."
        )
    if native_array.shape[0] == 0 or native_array.shape[1] == 0:
        raise ValueError("Embedding arrays must contain at least one vector and coordinate.")

    native_finite = np.isfinite(native_array).all(axis=1)
    reference_finite = np.isfinite(reference_array).all(axis=1)
    native_norms = np.full(native_array.shape[0], np.nan, dtype=np.float64)
    reference_norms = np.full(reference_array.shape[0], np.nan, dtype=np.float64)
    native_norms[native_finite] = np.linalg.norm(native_array[native_finite], axis=1)
    reference_norms[reference_finite] = np.linalg.norm(
        reference_array[reference_finite],
        axis=1,
    )

    non_finite = (
        ~native_finite
        | ~reference_finite
        | ~np.isfinite(native_norms)
        | ~np.isfinite(reference_norms)
    )
    zero_norm = (
        (native_finite & np.isfinite(native_norms) & (native_norms == 0))
        | (reference_finite & np.isfinite(reference_norms) & (reference_norms == 0))
    )
    invalid = non_finite | zero_norm
    valid = ~invalid
    max_absolute_difference: float | None = None
    mean_absolute_difference: float | None = None
    max_relative_difference: float | None = None
    relative_error_coordinate_count = 0
    minimum_cosine_similarity: float | None = None

    if valid.any():
        valid_native = native_array[valid]
        valid_reference = reference_array[valid]
        absolute_difference = np.abs(valid_native - valid_reference)
        max_absolute_difference = float(np.max(absolute_difference))
        mean_absolute_difference = float(np.mean(absolute_difference))

        meaningful_relative = np.abs(valid_reference) > atol
        relative_error_coordinate_count = int(np.count_nonzero(meaningful_relative))
        if relative_error_coordinate_count:
            relative_difference = (
                absolute_difference[meaningful_relative]
                / np.abs(valid_reference[meaningful_relative])
            )
            max_relative_difference = float(np.max(relative_difference))

        cosine_similarity = np.sum(valid_native * valid_reference, axis=1) / (
            native_norms[valid] * reference_norms[valid]
        )
        minimum_cosine_similarity = float(np.min(cosine_similarity))

    finite_norms = np.concatenate(
        (
            native_norms[np.isfinite(native_norms)],
            reference_norms[np.isfinite(reference_norms)],
        )
    )
    maximum_norm_deviation: float | None = None
    unit_norms_close = False
    if finite_norms.size:
        maximum_norm_deviation = float(np.max(np.abs(finite_norms - 1.0)))
        unit_norms_close = bool(
            np.allclose(finite_norms, np.ones_like(finite_norms), rtol=rtol, atol=atol)
        )

    allclose = bool(
        np.allclose(native_array, reference_array, rtol=rtol, atol=atol, equal_nan=False)
    )
    invalid_embedding_count = int(np.count_nonzero(invalid))
    return {
        "compared_embedding_count": int(native_array.shape[0]),
        "valid_comparison_count": int(np.count_nonzero(valid)),
        "non_finite_embedding_count": int(np.count_nonzero(non_finite)),
        "zero_norm_embedding_count": int(np.count_nonzero(zero_norm)),
        "invalid_embedding_count": invalid_embedding_count,
        "max_absolute_difference": max_absolute_difference,
        "mean_absolute_difference": mean_absolute_difference,
        "max_relative_difference": max_relative_difference,
        "relative_error_reference_abs_threshold": float(atol),
        "relative_error_coordinate_count": relative_error_coordinate_count,
        "minimum_cosine_similarity": minimum_cosine_similarity,
        "maximum_l2_norm_deviation_from_one": maximum_norm_deviation,
        "tolerance": {"rtol": float(rtol), "atol": float(atol)},
        "within_tolerance": allclose,
        "passed": bool(
            allclose and invalid_embedding_count == 0 and unit_norms_close
        ),
    }


def _scientific(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3e}"


def render_native_mrl_api_equivalence_markdown(
    report: Mapping[str, Any],
) -> str:
    environment = report["environment"]
    checkpoint = report["checkpoint"]
    sample_selection: Mapping[str, Any] = report["sample_selection"]
    measurements: Sequence[Mapping[str, Any]] = report["measurements"]
    lines = [
        "# Native MRL API equivalence validation",
        "",
        "This is a small-sample numerical validation of native `truncate_dim` behavior. "
        "It is separate from the 24 full-dataset benchmark measurements and does not "
        "reproduce their classification or retrieval accuracy scores.",
        "",
        f"- Validation status: **{str(report['status']).upper()}**",
        f"- Validation source commit: `{report['source_commit_sha']}`",
        f"- Execution time (UTC): `{report['timestamp_utc']}`",
        f"- Package: `openclip-gemma=={report['library_version']}`",
        f"- Model: `{report['model_id']}` at revision `{report['model_revision']}`",
        f"- Checkpoint SHA-256: `{checkpoint['sha256']}`",
        f"- Device: {environment['gpu_name']} · CUDA {environment['cuda_runtime']} · "
        f"PyTorch {environment['pytorch_version']} · Python {environment['python_version']}",
        f"- Worktree clean during validation: `{report['working_tree_dirty'] is False}`",
        "",
        "For each dimension, native image/text outputs were produced by the released "
        "model API with `truncate_dim=N`. The reference independently normalizes the "
        "leading N coordinates of an output produced with `truncate_dim=768`. The "
        "manual prefix operation is used only for this equivalence check; it did not "
        "replace native inference in any published benchmark run.",
        "",
        f"Tolerance: `rtol={report['tolerance']['rtol']:.0e}`, "
        f"`atol={report['tolerance']['atol']:.0e}`. Relative error is reported only "
        "where the absolute reference coordinate exceeds the absolute tolerance.",
        "",
        "## Dataset samples and prompt paths",
        "",
    ]
    for dataset_id, selection in sample_selection.items():
        lines.append(
            f"- `{dataset_id}`: {selection['selection_description']} "
            f"(checksum `{selection['dataset_checksum']}`)."
        )
        image_sample_count = selection.get("image_sample_count")
        if image_sample_count is not None:
            lines.append(f"  - {image_sample_count} image samples.")
        for language, text_selection in selection.get("text_samples", {}).items():
            lines.append(
                f"  - `{language}`: {text_selection['sample_count']} selected texts; "
                f"text IDs hash `{text_selection['text_ids_sha256']}`, content hash "
                f"`{text_selection['text_content_sha256']}`."
            )

    lines.extend(
        [
            "",
            "Classification uses the existing CIFAR-10 P0 and P1 prompt-generation "
            "functions with the EmbeddingGemma `Classification` task prompt. Retrieval "
            "uses the existing XM3600 `official_first_caption` candidate construction "
            "and the `SearchQuery` task prompt for English and Japanese.",
            "",
            "## Native output versus mathematical reference",
            "",
            "| Dimension | Modality | Dataset / protocol / language | Embeddings | "
            "Max abs. | Mean abs. | Max relative* | Min cosine | Max norm deviation | "
            "Invalid | Result |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    ordered_measurements = sorted(
        measurements,
        key=lambda measurement: (
            int(measurement["embedding_dimension"]),
            str(measurement["modality"]),
            str(measurement.get("protocol", "")),
            str(measurement.get("language", "")),
        ),
    )
    for measurement in ordered_measurements:
        comparison = measurement["native_vs_reference"]
        context = ", ".join(str(value) for value in measurement["dataset_ids"])
        for field in ("protocol", "language", "task_prompt_name"):
            value = measurement.get(field)
            if value:
                context += f", {value}"
        lines.append(
            "| "
            + " | ".join(
                (
                    str(measurement["embedding_dimension"]),
                    str(measurement["modality"]),
                    context,
                    str(comparison["compared_embedding_count"]),
                    _scientific(comparison["max_absolute_difference"]),
                    _scientific(comparison["mean_absolute_difference"]),
                    _scientific(comparison["max_relative_difference"]),
                    _scientific(comparison["minimum_cosine_similarity"]),
                    _scientific(comparison["maximum_l2_norm_deviation_from_one"]),
                    str(comparison["invalid_embedding_count"]),
                    "PASS" if comparison["passed"] else "FAIL",
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "* Relative error omits reference coordinates whose magnitude is at most "
            "`atol`; the number of included coordinates is recorded in the JSON.",
            "",
            "## Adapter NumPy conversion",
            "",
            "The adapter's returned NumPy FP32 arrays are compared with the direct native "
            "model outputs for the same inputs and dimension.",
            "",
            "| Dimension | Modality | Dataset / protocol / language | Dtype | Max abs. | "
            "Mean abs. | Max norm deviation | Invalid | Result |",
            "|---:|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for measurement in ordered_measurements:
        adapter_output = measurement["adapter_output"]
        comparison = adapter_output["vs_direct_native"]
        context = ", ".join(str(value) for value in measurement["dataset_ids"])
        for field in ("protocol", "language", "task_prompt_name"):
            value = measurement.get(field)
            if value:
                context += f", {value}"
        lines.append(
            "| "
            + " | ".join(
                (
                    str(measurement["embedding_dimension"]),
                    str(measurement["modality"]),
                    context,
                    str(adapter_output["dtype"]),
                    _scientific(comparison["max_absolute_difference"]),
                    _scientific(comparison["mean_absolute_difference"]),
                    _scientific(comparison["maximum_l2_norm_deviation_from_one"]),
                    str(comparison["invalid_embedding_count"]),
                    "PASS" if comparison["passed"] else "FAIL",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Ranking agreement is not reported because the published raw records preserve "
            "aggregate Recall@K and candidate ordering, not per-query rankings or embeddings. "
            "That analysis requires separate full-dataset native inference.",
            "",
        ]
    )
    return "\n".join(lines)
