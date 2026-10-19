from __future__ import annotations

from typing import Any

from vision_language_benchmarks.evaluation.runner import _make_result
from vision_language_benchmarks.registry import get_dataset, get_model


def make_result(
    *,
    model_key: str = "openai_clip_b32",
    dataset_key: str = "cifar10",
    language: str = "en",
    protocol: str = "p0",
    status: str = "completed",
    run_id: str | None = None,
    data_terms_confirmed: bool | None = None,
    image_id_order_hash: str = "synthetic-image-order",
) -> dict[str, Any]:
    model = get_model(model_key)
    dataset = get_dataset(dataset_key)
    selected_run_id = run_id or f"synthetic-{dataset_key}-{language}-{status}"
    selected_terms_confirmation = (
        dataset.license_status == "requires_user_confirmation" and status == "completed"
        if data_terms_confirmed is None
        else data_terms_confirmed
    )
    if status != "completed":
        metrics = None
        reason = "Synthetic unsuccessful result."
    elif dataset.task == "classification":
        metrics = {"acc1": 0.5, "acc5": 0.9}
        reason = None
    else:
        metrics = {
            "image_retrieval_recall@1": 0.5,
            "image_retrieval_recall@5": 0.7,
            "image_retrieval_recall@10": 0.9,
            "text_retrieval_recall@1": 0.4,
            "text_retrieval_recall@5": 0.6,
            "text_retrieval_recall@10": 0.8,
        }
        reason = None

    image_count = dataset.expected_images or 2
    if dataset.task == "classification":
        class_count = dataset.expected_classes or 2
        text_count = class_count * 18
    else:
        class_count = None
        text_count = dataset.expected_texts or image_count
    dataset_record = {
        "version": "synthetic-v1",
        "dataset_checksum": "b" * 64,
        "actual_sample_count": image_count,
        "text_count": text_count,
        "class_count": class_count,
        "data_integrity_status": "verified",
        "annotation_source": "synthetic test fixture",
    }
    evaluation_metadata = {
        "image_count": image_count,
        "text_count": text_count,
        "class_count": class_count,
        "prompt_policy": "synthetic-test",
        "prompt_name": None,
        "template_hash": "c" * 64,
        "classnames_hash": "d" * 64 if class_count is not None else None,
        "image_id_order_hash": image_id_order_hash,
        "text_id_order_hash": "e" * 64,
        "positive_pair_count": text_count if dataset.task == "retrieval" else None,
        "batch_size_actual": 2,
        "image_batch_size_actual": 2,
        "text_batch_size_actual": 2,
        "oom_history": [],
    }
    return _make_result(
        model_key=model_key,
        dataset_key=dataset_key,
        protocol=protocol,
        language=language,
        precision="fp32",
        requested_device="cpu",
        actual_device="cpu",
        batch_size=2,
        seed=0,
        status=status,
        reason=reason,
        source_commit_sha="a" * 40,
        lock_sha256="f" * 64,
        dirty=False,
        hardware={
            "gpu_name": None,
            "gpu_vram_bytes": None,
            "driver_version": None,
            "torch_version": None,
            "torch_cuda_version": None,
        },
        configuration_hash=f"configuration-{selected_run_id}",
        run_id=selected_run_id,
        data_terms_confirmed=selected_terms_confirmation,
        dataset_record=dataset_record,
        model_metadata={
            "adapter_name": model.adapter,
            "implementation_version": "test",
            "checkpoint_sha256": "1" * 64,
            "checkpoint_files": [],
            "parameter_count": 123_456_789,
            "model_download_seconds": 1.0,
            "tokenizer_repository": model.tokenizer_repository or f"open_clip:{model.model_id}",
            "tokenizer_revision": model.tokenizer_revision,
            "tokenizer_license": model.tokenizer_license,
        },
        evaluation_metadata=evaluation_metadata,
        metrics=metrics,
        timings={
            "throughput_images_per_sec": 12.5,
            "throughput_texts_per_sec": 24.5,
            "inference_ms_median": 5.0,
            "inference_ms_iqr": 1.0,
            "image_inference_ms_median": 4.0,
            "text_inference_ms_median": 6.0,
            "gpu_peak_allocated_bytes": None,
            "gpu_peak_reserved_bytes": None,
        },
    )
