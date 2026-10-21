from __future__ import annotations

import gc
import json
import platform
import random
import statistics
import subprocess
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from vision_language_benchmarks.adapters import create_adapter
from vision_language_benchmarks.atomic import atomic_write_json, content_hash, sha256_file
from vision_language_benchmarks.datasets.prepare import prepare_dataset
from vision_language_benchmarks.embedding_dimensions import (
    EMBEDDING_DIMENSION_STRATEGY,
    NATIVE_EMBEDDING_DIMENSION,
    resolve_embedding_dimension,
)
from vision_language_benchmarks.errors import (
    BatchSizeExhaustedError,
    BenchmarkError,
    DataIntegrityError,
    DependencyUnavailableError,
)
from vision_language_benchmarks.evaluation.engine import (
    evaluate_classification,
    evaluate_retrieval,
)
from vision_language_benchmarks.evaluation.manifest import RunManifest
from vision_language_benchmarks.hardware import doctor_report
from vision_language_benchmarks.paths import PROJECT_ROOT, RESULTS_ROOT
from vision_language_benchmarks.prompts import (
    classification_prompt_name,
    retrieval_prompt_name,
)
from vision_language_benchmarks.registry import (
    ProtocolName,
    RetrievalVariant,
    canonical_language,
    get_dataset,
    get_model,
    language_is_supported,
    normalize_retrieval_variant,
)
from vision_language_benchmarks.reporting.schema import validate_result
from vision_language_benchmarks.reproducibility import (
    inspect_git_repository,
    require_clean_source_preflight,
)

RUN_SCHEMA_VERSION = "1.0.0"
KNOWN_RUN_ERRORS = (
    BenchmarkError,
    OSError,
    RuntimeError,
    ValueError,
    ImportError,
    TimeoutError,
    ConnectionError,
)


def _source_commit_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    if completed.returncode != 0:
        raise DataIntegrityError(
            f"Cannot determine the source commit SHA: {completed.stderr.strip()}."
        )
    return completed.stdout.strip()


def _working_tree_dirty() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    if completed.returncode != 0:
        raise DataIntegrityError(
            f"Cannot inspect the source worktree: {completed.stderr.strip()}."
        )
    return bool(completed.stdout.strip())


def _lock_sha256() -> str | None:
    lock_path = PROJECT_ROOT / "uv.lock"
    return sha256_file(lock_path) if lock_path.is_file() else None


def _device_index(device: str) -> int:
    if ":" not in device:
        return 0
    try:
        return int(device.rsplit(":", maxsplit=1)[1])
    except ValueError as error:
        raise ValueError(f"Invalid CUDA device specifier: {device!r}.") from error


def _hardware_values(report: dict[str, Any], device: str) -> dict[str, Any]:
    torch_info = report.get("torch", {})
    gpu_name = None
    gpu_memory = None
    driver_version = None
    devices = torch_info.get("devices", [])
    index = _device_index(device) if device.startswith("cuda") else 0
    if index < len(devices):
        gpu_name = devices[index].get("name")
        gpu_memory = devices[index].get("total_memory_bytes")
    smi = report.get("nvidia_smi", {})
    smi_devices = smi.get("devices", [])
    if index < len(smi_devices):
        driver_version = smi_devices[index].get("driver_version")
    return {
        "gpu_name": gpu_name,
        "gpu_vram_bytes": gpu_memory,
        "driver_version": driver_version,
        "torch_version": torch_info.get("version"),
        "torch_cuda_version": torch_info.get("cuda_runtime"),
    }


def _make_result(
    *,
    model_key: str,
    dataset_key: str,
    protocol: ProtocolName,
    language: str,
    precision: str,
    requested_device: str,
    actual_device: str,
    batch_size: int,
    seed: int,
    embedding_dimension: int | None = None,
    status: str,
    reason: str | None,
    source_commit_sha: str,
    lock_sha256: str | None,
    dirty: bool,
    hardware: dict[str, Any],
    configuration_hash: str,
    run_id: str,
    data_terms_confirmed: bool = False,
    retrieval_variant: RetrievalVariant | None = None,
    reproducibility_mode: str | None = None,
    expected_source_commit_sha: str | None = None,
    staging_directory: str | None = None,
    dataset_record: dict[str, Any] | None = None,
    model_metadata: dict[str, Any] | None = None,
    evaluation_metadata: dict[str, Any] | None = None,
    metrics: dict[str, float | None] | None = None,
    timings: dict[str, Any] | None = None,
    oom_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    model = get_model(model_key)
    dataset = get_dataset(dataset_key)
    normalized_language = canonical_language(language)
    language_code = None
    if normalized_language in dataset.languages:
        language_code = dataset.source_language_code(normalized_language)
    record = dataset_record or {}
    adapter_metadata = model_metadata or {}
    evaluation = evaluation_metadata or {}
    run_timings = timings or {}
    default_prompt_name = (
        classification_prompt_name(model.key)
        if dataset.task == "classification"
        else retrieval_prompt_name(model.key)
    )
    preprocessing_config = adapter_metadata.get("preprocessing_config")
    if not isinstance(preprocessing_config, dict):
        preprocessing_config = {
            "input_resolution": adapter_metadata.get("input_resolution"),
            "image_batching_strategy": adapter_metadata.get("image_batching_strategy"),
            "max_text_length": adapter_metadata.get("max_text_length"),
        }
    preprocessing_version = adapter_metadata.get(
        "preprocessing_version",
        adapter_metadata.get("implementation_version"),
    )
    run_configuration = {
        "model_key": model.key,
        "dataset_key": dataset.key,
        "protocol": protocol,
        "language": normalized_language,
        "precision": precision,
        "device": actual_device,
        "batch_size_requested": batch_size,
        "seed": seed,
        "dataset_checksum": record.get("dataset_checksum"),
        "data_terms_confirmed": data_terms_confirmed,
    }
    if embedding_dimension is not None:
        run_configuration["embedding_dimension"] = embedding_dimension
    if retrieval_variant is not None:
        run_configuration["retrieval_variant"] = retrieval_variant
    if reproducibility_mode == "strict":
        run_configuration["reproducibility_mode"] = reproducibility_mode
        run_configuration["expected_source_commit_sha"] = expected_source_commit_sha
    metadata = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "configuration_hash": configuration_hash,
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": status,
        "reason_if_not_success": reason,
        "source_commit_sha": source_commit_sha,
        "working_tree_dirty": dirty,
        "framework_name": "vision-language-benchmarks",
        "framework_ref": "v0.1.0",
        "python_version": platform.python_version(),
        "uv_lock_sha256": lock_sha256,
        "model_repo": model.hf_repository or model.model_id,
        "checkpoint_revision": model.checkpoint_revision,
        "model_checkpoint_filename": model.checkpoint_filename,
        "model_checkpoint_size_bytes": model.checkpoint_size_bytes,
        "model_languages": list(model.languages),
        "model_repository_revision": model.repository_revision,
        "pretrained_tag": model.pretrained_tag,
        "adapter_name": adapter_metadata.get("adapter_name", model.adapter),
        "model_license": model.license,
        "model_compatibility_note": model.compatibility_note,
        "checkpoint_sha256": adapter_metadata.get("checkpoint_sha256"),
        "checkpoint_files": adapter_metadata.get("checkpoint_files"),
        "parameter_count": adapter_metadata.get("parameter_count"),
        "model_implementation_version": adapter_metadata.get("implementation_version"),
        "preprocessing_version": preprocessing_version,
        "preprocessing_config": preprocessing_config,
        "preprocessing_config_sha256": content_hash(preprocessing_config),
        "tokenizer_implementation_version": adapter_metadata.get(
            "tokenizer_implementation_version",
            adapter_metadata.get("implementation_version"),
        ),
        "tokenizer_repository": adapter_metadata.get(
            "tokenizer_repository",
            model.tokenizer_repository or model.hf_repository,
        ),
        "tokenizer_revision": adapter_metadata.get(
            "tokenizer_revision",
            model.tokenizer_revision or model.checkpoint_revision,
        ),
        "tokenizer_license": adapter_metadata.get("tokenizer_license", model.tokenizer_license),
        "dataset_id": dataset.key,
        "dataset_revision": record.get("version"),
        "dataset_split": dataset.split,
        "dataset_checksum": record.get("dataset_checksum"),
        "dataset_license": dataset.license,
        "dataset_license_status": dataset.license_status,
        "dataset_terms_confirmed": data_terms_confirmed,
        "dataset_source": dataset.source,
        "dataset_integrity_status": record.get("data_integrity_status"),
        "annotation_source": record.get("annotation_source"),
        "images_archive_sha256": record.get("images_archive_sha256"),
        "captions_archive_sha256": record.get("captions_archive_sha256"),
        "image_files_sha256": record.get("image_files_sha256"),
        "image_attributions_sha256": record.get("image_attributions_sha256"),
        "image_count": evaluation.get(
            "image_count",
            record.get("actual_sample_count", dataset.expected_images),
        ),
        "text_count": evaluation.get(
            "text_count",
            record.get(
                "text_count",
                dataset.expected_text_count(normalized_language, retrieval_variant),
            ),
        ),
        "class_count": evaluation.get(
            "class_count",
            record.get("class_count", dataset.expected_classes),
        ),
        "retrieval_variant": retrieval_variant,
        "retrieval_variant_reference_commit": evaluation.get(
            "retrieval_variant_reference_commit"
        ),
        "retrieval_variant_source": (
            "run_configuration" if retrieval_variant is not None else None
        ),
        "caption_selection": evaluation.get("caption_selection"),
        "one_caption_per_image": evaluation.get("one_caption_per_image"),
        "candidate_counts": evaluation.get("candidate_counts"),
        "official_image_filtering": evaluation.get("official_image_filtering"),
        "caption_count_per_image_min": evaluation.get("caption_count_per_image_min"),
        "caption_count_per_image_max": evaluation.get("caption_count_per_image_max"),
        "text_content_sha256": evaluation.get("text_content_sha256"),
        "language": normalized_language,
        "original_language_code": language_code,
        "prompt_policy": evaluation.get("prompt_policy"),
        "prompt_name": evaluation.get("prompt_name", default_prompt_name),
        "task_prompt": evaluation.get("task_prompt"),
        "template_hash": evaluation.get("template_hash"),
        "classnames_hash": evaluation.get("classnames_hash"),
        "image_id_scheme": evaluation.get("image_id_scheme"),
        "image_ids": evaluation.get("image_ids"),
        "text_ids": evaluation.get("text_ids"),
        "image_id_order_hash": evaluation.get("image_id_order_hash"),
        "text_id_order_hash": evaluation.get("text_id_order_hash"),
        "text_id_scheme": evaluation.get("text_id_scheme"),
        "positive_pair_count": evaluation.get("positive_pair_count"),
        "device": actual_device,
        "requested_device": requested_device,
        **hardware,
        "precision": precision,
        "amp_enabled": precision == "bf16",
        "batch_size_requested": batch_size,
        "batch_size_actual": evaluation.get("batch_size_actual"),
        "image_batch_size_actual": evaluation.get("image_batch_size_actual"),
        "text_batch_size_actual": evaluation.get("text_batch_size_actual"),
        "seed": seed,
        "model_download_seconds": (
            adapter_metadata.get("model_download_seconds") if status == "completed" else None
        ),
        "model_load_seconds": run_timings.get("model_load_seconds"),
        "dataset_preparation_seconds": run_timings.get("dataset_preparation_seconds"),
        "dataset_loading_seconds": run_timings.get("dataset_loading_seconds"),
        "data_loading_seconds": evaluation.get("data_loading_seconds"),
        "preprocessing_seconds": evaluation.get("preprocessing_seconds"),
        "inference_seconds": evaluation.get("inference_seconds"),
        "image_inference_seconds": evaluation.get("image_inference_seconds"),
        "text_inference_seconds": evaluation.get("text_inference_seconds"),
        "retrieval_aggregation_seconds": evaluation.get("retrieval_aggregation_seconds"),
        "evaluation_wall_seconds": run_timings.get("evaluation_wall_seconds"),
        "throughput_images_per_sec": run_timings.get("throughput_images_per_sec"),
        "throughput_texts_per_sec": run_timings.get("throughput_texts_per_sec"),
        "inference_ms_median": run_timings.get("inference_ms_median"),
        "inference_ms_iqr": run_timings.get("inference_ms_iqr"),
        "image_inference_ms_median": run_timings.get("image_inference_ms_median"),
        "text_inference_ms_median": run_timings.get("text_inference_ms_median"),
        "gpu_peak_allocated_bytes": run_timings.get("gpu_peak_allocated_bytes"),
        "gpu_peak_reserved_bytes": run_timings.get("gpu_peak_reserved_bytes"),
        "oom_history": oom_history or evaluation.get("oom_history", []),
        "training_data_overlap_risk": (
            "The model pretraining corpus is not independently verified to exclude this "
            "evaluation dataset; training-data contamination cannot be ruled out."
        ),
        "run_configuration": run_configuration,
    }
    if embedding_dimension is not None:
        metadata["embedding_dimension"] = embedding_dimension
        dimension_strategy = adapter_metadata.get("embedding_dimension_strategy")
        if dimension_strategy is None:
            dimension_strategy = preprocessing_config.get(
                "embedding_dimension_strategy",
                EMBEDDING_DIMENSION_STRATEGY,
            )
        metadata["embedding_dimension_strategy"] = dimension_strategy
        native_dimension = adapter_metadata.get("native_embedding_dimension")
        if native_dimension is None:
            native_dimension = preprocessing_config.get(
                "native_embedding_dimension",
                NATIVE_EMBEDDING_DIMENSION,
            )
        metadata["native_embedding_dimension"] = native_dimension
        generation_mode = adapter_metadata.get("embedding_generation_mode")
        if generation_mode is None:
            generation_mode = preprocessing_config.get("embedding_generation_mode")
        if generation_mode is not None:
            metadata["embedding_generation_mode"] = generation_mode
    if reproducibility_mode is not None:
        metadata.update(
            {
                "reproducibility_mode": reproducibility_mode,
                "expected_source_commit_sha": expected_source_commit_sha,
                "staging_directory": staging_directory,
            }
        )
    result: dict[str, Any] = {
        "dataset": dataset.key,
        "model": model.model_id,
        "pretrained": model.pretrained_tag,
        "task": dataset.task,
        "language": normalized_language,
        "status": status,
        "benchmark_metadata": metadata,
    }
    if metrics is not None:
        result["metrics"] = metrics
    return result


def _persist_result(
    result: dict[str, Any],
    results_root: Path,
    manifest: RunManifest,
    configuration_hash: str,
) -> Path:
    validate_result(result)
    metadata = result["benchmark_metadata"]
    run_id = metadata["run_id"]
    raw_dir = results_root / "raw"
    raw_path = raw_dir / f"{run_id}.json"
    atomic_write_json(raw_path, result)
    relative_path = raw_path.relative_to(results_root).as_posix()
    manifest.transition(
        run_id,
        result["status"],
        {
            "configuration_hash": configuration_hash,
            "raw_path": relative_path,
            "model": result["model"],
            "dataset": result["dataset"],
            "language": result["language"],
            "protocol": metadata["run_configuration"]["protocol"],
            "retrieval_variant": metadata.get("retrieval_variant"),
            "reason_if_not_success": metadata["reason_if_not_success"],
        },
    )
    return raw_path


def _write_skipped_or_blocked(
    *,
    model_key: str,
    dataset_key: str,
    protocol: ProtocolName,
    language: str,
    precision: str,
    device: str,
    batch_size: int,
    seed: int,
    status: str,
    reason: str,
    results_root: Path,
    source_sha: str,
    lock_sha: str | None,
    dirty: bool,
    hardware: dict[str, Any],
    confirm_data_terms: bool = False,
    retrieval_variant: RetrievalVariant | None = None,
    embedding_dimension: int | None = None,
    dataset_record: dict[str, Any] | None = None,
    configuration_hash: str | None = None,
    run_id: str | None = None,
    reproducibility_mode: str | None = None,
    expected_source_commit_sha: str | None = None,
    staging_directory: str | None = None,
) -> dict[str, Any]:
    configuration = {
        "model": model_key,
        "dataset": dataset_key,
        "protocol": protocol,
        "language": canonical_language(language),
        "precision": precision,
        "device": device,
        "batch_size": batch_size,
        "seed": seed,
        "data_terms_confirmed": confirm_data_terms,
        "dataset_checksum": (dataset_record or {}).get("dataset_checksum"),
        "source_commit_sha": source_sha,
        "uv_lock_sha256": lock_sha,
    }
    if embedding_dimension is not None:
        configuration["embedding_dimension"] = embedding_dimension
    if retrieval_variant is not None:
        configuration["retrieval_variant"] = retrieval_variant
    config = configuration_hash or content_hash(configuration)
    actual_run_id = run_id or config[:16]
    manifest = RunManifest(results_root / "run-manifest.json")
    entry = manifest.get(actual_run_id)
    if entry is not None:
        actual_run_id = f"{actual_run_id}-{uuid.uuid4().hex[:8]}"
    result = _make_result(
        model_key=model_key,
        dataset_key=dataset_key,
        protocol=protocol,
        language=language,
        precision=precision,
        requested_device=device,
        actual_device=device,
        batch_size=batch_size,
        seed=seed,
        embedding_dimension=embedding_dimension,
        status=status,
        reason=reason,
        source_commit_sha=source_sha,
        lock_sha256=lock_sha,
        dirty=dirty,
        hardware=hardware,
        configuration_hash=config,
        run_id=actual_run_id,
        data_terms_confirmed=confirm_data_terms,
        retrieval_variant=retrieval_variant,
        reproducibility_mode=reproducibility_mode,
        expected_source_commit_sha=expected_source_commit_sha,
        staging_directory=staging_directory,
        dataset_record=dataset_record,
    )
    manifest.transition(
        actual_run_id,
        "running",
        {
            "configuration_hash": config,
            "model": result["model"],
            "dataset": result["dataset"],
            "language": result["language"],
            "protocol": protocol,
            "retrieval_variant": retrieval_variant,
        },
    )
    _persist_result(result, results_root, manifest, config)
    return result


def _seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_id(configuration_hash: str, results_root: Path, resume: bool, force: bool) -> str:
    base_id = configuration_hash[:16]
    entry = RunManifest(results_root / "run-manifest.json").get(base_id)
    if resume and not force and entry is not None and entry.get("status") == "completed":
        if entry.get("configuration_hash") == configuration_hash:
            raw_path = results_root / str(entry.get("raw_path", ""))
            if raw_path.is_file():
                previous = json.loads(raw_path.read_text(encoding="utf-8"))
                validate_result(previous)
                if (
                    previous.get("benchmark_metadata", {}).get("configuration_hash")
                    == configuration_hash
                    and previous.get("status") == "completed"
                ):
                    return base_id
    if entry is not None or force:
        return f"{base_id}-{uuid.uuid4().hex[:8]}"
    return base_id


def _reuse_result(run_id: str, results_root: Path) -> dict[str, Any] | None:
    manifest = RunManifest(results_root / "run-manifest.json")
    entry = manifest.get(run_id)
    if entry is None or entry.get("status") != "completed":
        return None
    raw_path = results_root / str(entry.get("raw_path", ""))
    if not raw_path.is_file():
        return None
    result = json.loads(raw_path.read_text(encoding="utf-8"))
    validate_result(result)
    return result


def _record_invalid_attempt(
    result: dict[str, Any],
    results_root: Path,
    manifest: RunManifest,
    configuration_hash: str,
    reason: str,
    raw_path: Path | None = None,
) -> None:
    metadata = result["benchmark_metadata"]
    run_id = str(metadata["run_id"])
    attempts_dir = results_root / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    git_state = inspect_git_repository(PROJECT_ROOT)
    worktree_dirty = None
    if git_state["worktree_status_available"]:
        worktree_dirty = bool(
            git_state["staged_changes"]
            or git_state["unstaged_changes"]
            or git_state["untracked_files"]
        )
    record: dict[str, Any] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "invalid",
        "reason": reason,
        "source_commit_sha": metadata.get("source_commit_sha"),
        "expected_source_commit_sha": metadata.get("expected_source_commit_sha"),
        "working_tree_dirty": worktree_dirty,
        "source_commit_sha_at_validation": git_state.get("source_commit_sha"),
        "result_status": result["status"],
        "metrics": result.get("metrics"),
    }
    if raw_path is not None and raw_path.is_file():
        preserved_path = attempts_dir / f"{run_id}.measurement.json"
        raw_path.replace(preserved_path)
        record["measurement_path"] = preserved_path.relative_to(results_root).as_posix()
    attempt_path = attempts_dir / f"{run_id}.json"
    atomic_write_json(attempt_path, record)

    entry = manifest.get(run_id)
    if entry is not None and entry.get("status") in {
        "completed",
        "skipped",
        "failed",
        "blocked",
    }:
        manifest.transition(
            run_id,
            "running",
            {"invalid_for_publication": True},
            restart=True,
        )
    manifest.transition(
        run_id,
        "failed",
        {
            "configuration_hash": configuration_hash,
            "invalid_for_publication": True,
            "reason_if_not_success": reason,
            "attempt_path": attempt_path.relative_to(results_root).as_posix(),
        },
    )


def _final_metrics(metadata: dict[str, Any]) -> dict[str, Any]:
    inference_times = metadata.get("batch_inference_ms", [])
    image_times = metadata.get("image_batch_inference_ms", [])
    text_times = metadata.get("text_batch_inference_ms", [])

    def median(values: list[float]) -> float | None:
        return float(statistics.median(values)) if values else None

    inference_median = median(inference_times)
    inference_iqr = (
        float(np.percentile(inference_times, 75) - np.percentile(inference_times, 25))
        if inference_times
        else None
    )
    image_seconds = float(metadata.get("image_inference_seconds", 0.0))
    text_seconds = float(metadata.get("text_inference_seconds", 0.0))
    image_count = int(metadata.get("image_count", 0))
    text_count = int(metadata.get("text_count", 0))
    return {
        "throughput_images_per_sec": image_count / image_seconds if image_seconds > 0 else None,
        "throughput_texts_per_sec": text_count / text_seconds if text_seconds > 0 else None,
        "inference_ms_median": inference_median,
        "inference_ms_iqr": inference_iqr,
        "image_inference_ms_median": median(image_times),
        "text_inference_ms_median": median(text_times),
    }


def run_benchmark(
    model_key: str,
    dataset_key: str,
    *,
    protocol: ProtocolName = "p0",
    language: str = "en",
    precision: str = "fp32",
    device: str = "cuda",
    batch_size: int = 16,
    seed: int = 0,
    results_root: Path = RESULTS_ROOT,
    resume: bool = False,
    force: bool = False,
    allow_cpu_fallback: bool = False,
    confirm_data_terms: bool = False,
    retrieval_variant: RetrievalVariant | None = None,
    embedding_dimension: int | None = None,
    strict_reproducibility: bool = False,
    expected_source_commit_sha: str | None = None,
    uv_executable: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    model = get_model(model_key)
    dataset = get_dataset(dataset_key)
    resolved_embedding_dimension = resolve_embedding_dimension(
        model.key,
        embedding_dimension,
    )
    resolved_retrieval_variant = normalize_retrieval_variant(dataset_key, retrieval_variant)
    if protocol not in {"p0", "p1", "p2"}:
        raise ValueError(f"Unsupported protocol: {protocol!r}.")
    if precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be 'fp32' or 'bf16'.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("device must be 'cpu', 'cuda', or 'cuda:<index>'.")
    if strict_reproducibility and device == "cpu":
        raise ValueError("Strict reproducibility requires a CUDA device.")
    if strict_reproducibility and allow_cpu_fallback:
        raise ValueError("Strict reproducibility does not allow CPU fallback.")
    if protocol == "p2" and dataset.task != "retrieval":
        raise ValueError("P2 is reserved for paired-language retrieval datasets.")
    if dataset.key == "crossmodal3600" and protocol != "p2":
        raise ValueError("XM3600 retrieval variants are an additional dimension of P2.")
    normalized_language = canonical_language(language)
    preflight = None
    reproducibility_metadata = {
        "reproducibility_mode": "exploratory",
        "embedding_dimension": resolved_embedding_dimension,
    }
    if strict_reproducibility:
        preflight = require_clean_source_preflight(
            PROJECT_ROOT,
            results_root,
            expected_source_commit_sha,
            uv_executable=uv_executable,
        )
        source_sha = preflight.source_commit_sha
        reproducibility_metadata = {
            "reproducibility_mode": "strict",
            "expected_source_commit_sha": preflight.source_commit_sha,
            "staging_directory": preflight.staging_directory,
            "embedding_dimension": resolved_embedding_dimension,
        }
    else:
        source_sha = _source_commit_sha()
    lock_sha = _lock_sha256()
    dirty = False if strict_reproducibility else _working_tree_dirty()
    hardware_report = doctor_report(device="cuda" if device.startswith("cuda") else "cpu")
    if strict_reproducibility and not hardware_report.get("torch", {}).get(
        "cuda_available"
    ):
        raise DataIntegrityError(
            "Strict reproducibility requires CUDA, but torch.cuda.is_available() is false."
        )
    hardware = _hardware_values(hardware_report, device)

    if dataset.access_status == "blocked" or dataset.license_status == "blocked":
        return _write_skipped_or_blocked(
            model_key=model_key,
            dataset_key=dataset_key,
            protocol=protocol,
            language=language,
            precision=precision,
            device=device,
            batch_size=batch_size,
            seed=seed,
            status="blocked",
            reason=dataset.access_reason,
            results_root=results_root,
            source_sha=source_sha,
            lock_sha=lock_sha,
            dirty=dirty,
            hardware=hardware,
            confirm_data_terms=confirm_data_terms,
            retrieval_variant=resolved_retrieval_variant,
            **reproducibility_metadata,
        )
    if (
        dataset.license_status == "requires_user_confirmation"
        and not confirm_data_terms
    ):
        return _write_skipped_or_blocked(
            model_key=model_key,
            dataset_key=dataset_key,
            protocol=protocol,
            language=language,
            precision=precision,
            device=device,
            batch_size=batch_size,
            seed=seed,
            status="blocked",
            reason=(
                f"Usage terms for {dataset.name} are not verified. Review {dataset.source} "
                "and rerun with confirm_data_terms=True only if you are authorized to use it."
            ),
            results_root=results_root,
            source_sha=source_sha,
            lock_sha=lock_sha,
            dirty=dirty,
            hardware=hardware,
            confirm_data_terms=confirm_data_terms,
            retrieval_variant=resolved_retrieval_variant,
            **reproducibility_metadata,
        )
    if normalized_language not in dataset.languages:
        return _write_skipped_or_blocked(
            model_key=model_key,
            dataset_key=dataset_key,
            protocol=protocol,
            language=language,
            precision=precision,
            device=device,
            batch_size=batch_size,
            seed=seed,
            status="skipped",
            reason=(
                f"Dataset {dataset.key!r} does not provide {normalized_language!r}; "
                f"available languages: {', '.join(dataset.languages)}."
            ),
            results_root=results_root,
            source_sha=source_sha,
            lock_sha=lock_sha,
            dirty=dirty,
            hardware=hardware,
            confirm_data_terms=confirm_data_terms,
            retrieval_variant=resolved_retrieval_variant,
            **reproducibility_metadata,
        )
    if not language_is_supported(model, normalized_language):
        return _write_skipped_or_blocked(
            model_key=model_key,
            dataset_key=dataset_key,
            protocol=protocol,
            language=language,
            precision=precision,
            device=device,
            batch_size=batch_size,
            seed=seed,
            status="skipped",
            reason=f"Model {model.name!r} does not support language {normalized_language!r}.",
            results_root=results_root,
            source_sha=source_sha,
            lock_sha=lock_sha,
            dirty=dirty,
            hardware=hardware,
            confirm_data_terms=confirm_data_terms,
            retrieval_variant=resolved_retrieval_variant,
            **reproducibility_metadata,
        )
    if dataset.task not in model.supported_tasks:
        return _write_skipped_or_blocked(
            model_key=model_key,
            dataset_key=dataset_key,
            protocol=protocol,
            language=language,
            precision=precision,
            device=device,
            batch_size=batch_size,
            seed=seed,
            status="skipped",
            reason=f"Model {model.name!r} does not support task {dataset.task!r}.",
            results_root=results_root,
            source_sha=source_sha,
            lock_sha=lock_sha,
            dirty=dirty,
            hardware=hardware,
            confirm_data_terms=confirm_data_terms,
            retrieval_variant=resolved_retrieval_variant,
            **reproducibility_metadata,
        )

    if device.startswith("cuda") and not hardware_report.get("torch", {}).get("cuda_available"):
        if allow_cpu_fallback and precision == "fp32":
            if progress:
                progress("CUDA is unavailable; using CPU because --allow-cpu-fallback was set.")
            actual_device = "cpu"
        else:
            return _write_skipped_or_blocked(
                model_key=model_key,
                dataset_key=dataset_key,
                protocol=protocol,
                language=language,
                precision=precision,
                device=device,
                batch_size=batch_size,
                seed=seed,
                status="skipped",
                reason=(
                    "CUDA was requested but is unavailable; CPU fallback is disabled. "
                    "Run `vlbench doctor --device cuda` or pass --allow-cpu-fallback."
                ),
                results_root=results_root,
                source_sha=source_sha,
                lock_sha=lock_sha,
                dirty=dirty,
                hardware=hardware,
                confirm_data_terms=confirm_data_terms,
                retrieval_variant=resolved_retrieval_variant,
                **reproducibility_metadata,
            )
    else:
        actual_device = device

    prepare_started = time.perf_counter()
    try:
        dataset_record = prepare_dataset(
            dataset_key,
            progress=progress,
            confirm_data_terms=confirm_data_terms,
        )
    except KNOWN_RUN_ERRORS as error:
        return _write_skipped_or_blocked(
            model_key=model_key,
            dataset_key=dataset_key,
            protocol=protocol,
            language=language,
            precision=precision,
            device=actual_device,
            batch_size=batch_size,
            seed=seed,
            status="failed",
            reason=f"Dataset preparation failed: {type(error).__name__}: {error}",
            results_root=results_root,
            source_sha=source_sha,
            lock_sha=lock_sha,
            dirty=dirty,
            hardware=hardware,
            confirm_data_terms=confirm_data_terms,
            retrieval_variant=resolved_retrieval_variant,
            **reproducibility_metadata,
        )
    dataset_preparation_seconds = time.perf_counter() - prepare_started
    configuration = {
        "model_key": model.key,
        "model_repository": model.hf_repository or model.model_id,
        "checkpoint_revision": model.checkpoint_revision,
        "pretrained_tag": model.pretrained_tag,
        "dataset_key": dataset.key,
        "dataset_checksum": dataset_record.get("dataset_checksum"),
        "protocol": protocol,
        "language": normalized_language,
        "data_terms_confirmed": confirm_data_terms,
        "precision": precision,
        "requested_device": device,
        "actual_device": actual_device,
        "batch_size": batch_size,
        "seed": seed,
        "source_commit_sha": source_sha,
        "uv_lock_sha256": lock_sha,
    }
    if resolved_embedding_dimension is not None:
        configuration["embedding_dimension"] = resolved_embedding_dimension
    if resolved_retrieval_variant is not None:
        configuration["retrieval_variant"] = resolved_retrieval_variant
    if strict_reproducibility:
        configuration.update(
            {
                "reproducibility_mode": "strict",
                "expected_source_commit_sha": source_sha,
            }
        )
    configuration_hash = content_hash(configuration)
    selected_run_id = _run_id(configuration_hash, results_root, resume, force)
    if resume and not force:
        cached = _reuse_result(selected_run_id, results_root)
        if (
            cached is not None
            and cached["benchmark_metadata"].get("configuration_hash") == configuration_hash
        ):
            if progress:
                progress(f"Reusing completed run {selected_run_id}.")
            return cached

    results_root.mkdir(parents=True, exist_ok=True)
    manifest = RunManifest(results_root / "run-manifest.json")
    manifest.transition(
        selected_run_id,
        "running",
        {
            "configuration_hash": configuration_hash,
            "model": model.model_id,
            "dataset": dataset.key,
            "language": normalized_language,
            "protocol": protocol,
            "retrieval_variant": resolved_retrieval_variant,
            "embedding_dimension": resolved_embedding_dimension,
            "dataset_checksum": dataset_record.get("dataset_checksum"),
        },
    )

    model_started = time.perf_counter()
    adapter = None
    model_metadata: dict[str, Any] = {}
    evaluation_metadata: dict[str, Any] = {}
    oom_history: list[dict[str, Any]] = []
    try:
        adapter = create_adapter(
            model.key,
            actual_device,
            precision,
            embedding_dimension=resolved_embedding_dimension,
        )
        model_initialization_seconds = time.perf_counter() - model_started
        torch = adapter.torch
        _seed_everything(seed, torch)
        if actual_device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(adapter.device_object)

        if dataset.task == "classification":
            evaluation = evaluate_classification(
                adapter,
                dataset.key,
                protocol,
                normalized_language,
                batch_size,
                progress=progress,
            )
        else:
            evaluation = evaluate_retrieval(
                adapter,
                dataset.key,
                normalized_language,
                batch_size,
                retrieval_variant=resolved_retrieval_variant,
                progress=progress,
            )
        evaluation_metadata = evaluation.metadata
        dataset_loading_seconds = float(
            evaluation_metadata.get("dataset_manifest_loading_seconds", 0.0)
        )
        oom_history = evaluation_metadata.get("oom_history", [])
        model_metadata = adapter.metadata()
        if actual_device.startswith("cuda"):
            torch.cuda.synchronize(adapter.device_object)
            memory_metadata = {
                "gpu_peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(adapter.device_object)
                ),
                "gpu_peak_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(adapter.device_object)
                ),
            }
        else:
            memory_metadata = {
                "gpu_peak_allocated_bytes": None,
                "gpu_peak_reserved_bytes": None,
            }
        model_download_seconds = float(model_metadata.get("model_download_seconds", 0.0))
        run_timings = {
            "model_load_seconds": max(
                0.0,
                model_initialization_seconds - model_download_seconds,
            ),
            "dataset_preparation_seconds": dataset_preparation_seconds,
            "dataset_loading_seconds": dataset_loading_seconds,
            "evaluation_wall_seconds": time.perf_counter() - model_started,
            **_final_metrics(evaluation_metadata),
            **memory_metadata,
        }
        result = _make_result(
            model_key=model.key,
            dataset_key=dataset.key,
            protocol=protocol,
            language=normalized_language,
            precision=precision,
            requested_device=device,
            actual_device=actual_device,
            batch_size=batch_size,
            seed=seed,
            embedding_dimension=resolved_embedding_dimension,
            status="completed",
            reason=None,
            source_commit_sha=source_sha,
            lock_sha256=lock_sha,
            dirty=dirty,
            hardware=hardware,
            configuration_hash=configuration_hash,
            run_id=selected_run_id,
            data_terms_confirmed=confirm_data_terms,
            retrieval_variant=resolved_retrieval_variant,
            reproducibility_mode=(
                "strict" if strict_reproducibility else "exploratory"
            ),
            expected_source_commit_sha=(
                source_sha if strict_reproducibility else None
            ),
            staging_directory=(
                preflight.staging_directory if preflight is not None else None
            ),
            dataset_record=dataset_record,
            model_metadata=model_metadata,
            evaluation_metadata=evaluation_metadata,
            metrics=evaluation.metrics,
            timings=run_timings,
            oom_history=oom_history,
        )
    except KNOWN_RUN_ERRORS as error:
        if isinstance(error, BatchSizeExhaustedError):
            oom_history = error.oom_history
        reason = f"{type(error).__name__}: {error}"
        status = "skipped" if isinstance(error, DependencyUnavailableError) else "failed"
        if progress:
            progress(f"Run {selected_run_id} {status}: {reason}")
        result = _make_result(
            model_key=model.key,
            dataset_key=dataset.key,
            protocol=protocol,
            language=normalized_language,
            precision=precision,
            requested_device=device,
            actual_device=actual_device,
            batch_size=batch_size,
            seed=seed,
            embedding_dimension=resolved_embedding_dimension,
            status=status,
            reason=reason,
            source_commit_sha=source_sha,
            lock_sha256=lock_sha,
            dirty=dirty,
            hardware=hardware,
            configuration_hash=configuration_hash,
            run_id=selected_run_id,
            data_terms_confirmed=confirm_data_terms,
            retrieval_variant=resolved_retrieval_variant,
            reproducibility_mode=(
                "strict" if strict_reproducibility else "exploratory"
            ),
            expected_source_commit_sha=(
                source_sha if strict_reproducibility else None
            ),
            staging_directory=(
                preflight.staging_directory if preflight is not None else None
            ),
            dataset_record=dataset_record,
            model_metadata=model_metadata,
            evaluation_metadata=evaluation_metadata,
            oom_history=oom_history,
        )
    finally:
        if adapter is not None:
            adapter.close()
            del adapter
            gc.collect()

    if strict_reproducibility:
        try:
            require_clean_source_preflight(
                PROJECT_ROOT,
                results_root,
                source_sha,
                uv_executable=uv_executable,
                check_lockfile=False,
            )
        except DataIntegrityError as error:
            validate_result(result)
            _record_invalid_attempt(
                result,
                results_root,
                manifest,
                configuration_hash,
                str(error),
            )
            raise DataIntegrityError(
                f"Run {selected_run_id} is invalid for publication: {error}"
            ) from error

    if progress:
        progress(f"Run {selected_run_id}: {result['status']}.")
    raw_path = _persist_result(result, results_root, manifest, configuration_hash)
    if strict_reproducibility:
        try:
            require_clean_source_preflight(
                PROJECT_ROOT,
                results_root,
                source_sha,
                uv_executable=uv_executable,
                check_lockfile=False,
            )
        except DataIntegrityError as error:
            _record_invalid_attempt(
                result,
                results_root,
                manifest,
                configuration_hash,
                str(error),
                raw_path,
            )
            raise DataIntegrityError(
                f"Run {selected_run_id} is invalid for publication: {error}"
            ) from error
    return result
