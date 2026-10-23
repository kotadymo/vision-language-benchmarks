from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from vision_language_benchmarks.adapters.base import VisionLanguageAdapter
from vision_language_benchmarks.atomic import content_hash
from vision_language_benchmarks.datasets.prepare import load_cifar_dataset, retrieval_items
from vision_language_benchmarks.errors import BatchSizeExhaustedError, DataIntegrityError
from vision_language_benchmarks.evaluation.metrics import (
    classification_metrics,
    normalize_features,
    retrieval_metrics,
)
from vision_language_benchmarks.prompts import (
    CLIP_BENCHMARK_COMMIT,
    classification_prompt_groups,
    classification_prompt_name,
    retrieval_prompt_name,
)
from vision_language_benchmarks.registry import (
    ProtocolName,
    RetrievalVariant,
    get_dataset,
    normalize_retrieval_variant,
)


@dataclass(slots=True)
class EncodedBatches:
    features: NDArray[np.float32]
    batch_size_actual: int
    oom_history: list[dict[str, Any]]
    data_loading_seconds: float
    preprocessing_seconds: float
    inference_seconds: float
    batch_inference_ms: list[float]
    batch_metadata: list[dict[str, Any]]


@dataclass(slots=True)
class EvaluationOutput:
    metrics: dict[str, float | None]
    metadata: dict[str, Any]


@dataclass(slots=True)
class RetrievalCandidates:
    image_ids: list[str]
    image_paths: list[Path]
    text_values: list[str]
    text_ids: list[str]
    positive_images_per_text: list[int]
    caption_counts_per_image: list[int]


def _batch_sizes(initial: int) -> list[int]:
    if initial < 1:
        raise ValueError("batch_size must be positive.")
    sizes = [initial]
    while sizes[-1] > 1:
        sizes.append(max(1, sizes[-1] // 2))
    return sizes


def _synchronize(adapter: VisionLanguageAdapter) -> None:
    torch = getattr(adapter, "torch", None)
    device = getattr(adapter, "device_object", None)
    if torch is not None and device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def _close_images(batch: Sequence[Any]) -> None:
    for item in batch:
        if isinstance(item, Image.Image):
            item.close()


def _image_batch_size_actual(adapter: VisionLanguageAdapter, default: int) -> int:
    shape_group_size = getattr(adapter, "image_shape_group_max_batch_size", None)
    if isinstance(shape_group_size, int) and shape_group_size > 0:
        return min(default, shape_group_size)
    return default


def _encode_batched(
    *,
    count: int,
    load_batch: Callable[[int, int], Sequence[Any]],
    preprocess_batch: Callable[[Sequence[Any]], Any],
    encode_batch: Callable[[Any], NDArray[np.float32]],
    batch_size: int,
    adapter: VisionLanguageAdapter,
    modality: str,
    progress: Callable[[str], None] | None = None,
) -> EncodedBatches:
    if count < 1:
        raise ValueError(f"Cannot encode an empty {modality} collection.")
    torch = getattr(adapter, "torch", None)
    if torch is None:
        raise TypeError("The adapter does not expose its PyTorch runtime.")

    oom_history: list[dict[str, Any]] = []
    for candidate_size in _batch_sizes(batch_size):
        reset_image_batch_stats = getattr(adapter, "reset_image_batch_stats", None)
        if modality == "images" and callable(reset_image_batch_stats):
            reset_image_batch_stats()
        blocks: list[NDArray[np.float32]] = []
        batch_inference_ms: list[float] = []
        batch_metadata: list[dict[str, Any]] = []
        data_loading_seconds = 0.0
        preprocessing_seconds = 0.0
        inference_seconds = 0.0
        stage = "warmup"
        warmup: Sequence[Any] = []
        prepared: Any = None
        try:
            warmup = load_batch(0, min(candidate_size, count))
            try:
                prepared = preprocess_batch(warmup)
                _synchronize(adapter)
                warmup_features = encode_batch(prepared)
                _synchronize(adapter)
                del warmup_features
            finally:
                _close_images(warmup)
                prepared = None

            stage = "evaluation"
            for start in range(0, count, candidate_size):
                end = min(start + candidate_size, count)
                load_started = time.perf_counter()
                batch = load_batch(start, end)
                data_loading_seconds += time.perf_counter() - load_started
                try:
                    preprocess_started = time.perf_counter()
                    prepared = preprocess_batch(batch)
                    if modality == "images":
                        collect_batch_metadata = getattr(
                            adapter,
                            "image_batch_metadata",
                            None,
                        )
                        if callable(collect_batch_metadata):
                            details = collect_batch_metadata(prepared)
                            if not isinstance(details, dict):
                                raise DataIntegrityError(
                                    "Image batch metadata must be a dictionary."
                                )
                            if details:
                                batch_metadata.append(details)
                    _synchronize(adapter)
                    preprocessing_seconds += time.perf_counter() - preprocess_started

                    _synchronize(adapter)
                    inference_started = time.perf_counter()
                    features = encode_batch(prepared)
                    _synchronize(adapter)
                    elapsed = time.perf_counter() - inference_started
                    inference_seconds += elapsed
                    batch_inference_ms.append(elapsed * 1000)
                    if features.ndim != 2 or features.shape[0] != end - start:
                        raise DataIntegrityError(
                            f"{modality} encoder returned shape {features.shape} for "
                            f"{end - start} inputs."
                        )
                    blocks.append(features)
                finally:
                    _close_images(batch)
                    prepared = None
                    batch = []
                if progress and (end == count or end % max(candidate_size * 64, 1) == 0):
                    progress(f"Encoded {end}/{count} {modality} items.")
        except torch.cuda.OutOfMemoryError as error:
            oom_history.append(
                {
                    "modality": modality,
                    "batch_size": candidate_size,
                    "stage": stage,
                    "message": str(error).splitlines()[0],
                }
            )
            blocks.clear()
            warmup = []
            prepared = None
            torch.cuda.empty_cache()
            if candidate_size == 1:
                raise BatchSizeExhaustedError(
                    f"CUDA ran out of memory while encoding {modality} at batch size 1.",
                    oom_history,
                ) from error
            if progress:
                progress(
                    f"CUDA OOM while encoding {modality} at batch size {candidate_size}; "
                    f"retrying with {max(1, candidate_size // 2)}."
                )
            continue

        if not blocks:
            raise DataIntegrityError(f"The {modality} encoder returned no feature batches.")
        features = np.concatenate(blocks, axis=0)
        if features.shape[0] != count:
            raise DataIntegrityError(
                f"Encoded {features.shape[0]} of {count} requested {modality} items."
            )
        return EncodedBatches(
            features=features,
            batch_size_actual=candidate_size,
            oom_history=oom_history,
            data_loading_seconds=data_loading_seconds,
            preprocessing_seconds=preprocessing_seconds,
            inference_seconds=inference_seconds,
            batch_inference_ms=batch_inference_ms,
            batch_metadata=batch_metadata,
        )
    raise RuntimeError("The batch-size fallback loop ended unexpectedly.")


def _image_batch_metadata(encoded: EncodedBatches) -> dict[str, Any]:
    patch_counts = [
        count
        for details in encoded.batch_metadata
        for count in details.get("valid_patch_counts", [])
    ]
    if not patch_counts:
        return {}
    if len(patch_counts) != encoded.features.shape[0]:
        raise DataIntegrityError(
            "Image preprocessing metadata does not contain one patch count per image."
        )
    return {
        "valid_patch_counts": patch_counts,
        "valid_patch_counts_sha256": content_hash(patch_counts),
    }


def evaluate_classification(
    adapter: VisionLanguageAdapter,
    dataset_key: str,
    protocol: ProtocolName,
    language: str,
    batch_size: int,
    *,
    progress: Callable[[str], None] | None = None,
) -> EvaluationOutput:
    dataset_loading_started = time.perf_counter()
    dataset = load_cifar_dataset(dataset_key)
    dataset_manifest_loading_seconds = time.perf_counter() - dataset_loading_started
    classnames = list(dataset.classes)
    class_groups, prompt_policy, template_hash = classification_prompt_groups(
        classnames,
        protocol,
        adapter.spec.key,
    )
    flattened_prompts = [prompt for group in class_groups for prompt in group]
    text_encoded = _encode_batched(
        count=len(flattened_prompts),
        load_batch=lambda start, end: flattened_prompts[start:end],
        preprocess_batch=lambda texts: adapter.preprocess_texts(
            texts,
            "classification",
            language,
        ),
        encode_batch=lambda tokens: adapter.encode_texts(tokens, "classification", language),
        batch_size=batch_size,
        adapter=adapter,
        modality="class prompts",
        progress=progress,
    )

    template_count = len(class_groups[0])
    normalized_text_features = normalize_features(text_encoded.features, "class text features")
    class_weights = normalize_features(
        normalized_text_features.reshape(
            len(classnames),
            template_count,
            normalized_text_features.shape[1],
        ).mean(axis=1),
        "class text prototypes",
    )
    image_count = len(dataset)
    scores = np.empty((image_count, len(classnames)), dtype=np.float32)
    labels = np.asarray(dataset.targets, dtype=np.int64)

    def load_images(start: int, end: int) -> list[Image.Image]:
        return [dataset[index][0] for index in range(start, end)]

    image_encoded = _encode_batched(
        count=image_count,
        load_batch=load_images,
        preprocess_batch=adapter.preprocess_images,
        encode_batch=adapter.encode_images,
        batch_size=batch_size,
        adapter=adapter,
        modality="images",
        progress=progress,
    )
    image_batch_size_actual = _image_batch_size_actual(
        adapter,
        image_encoded.batch_size_actual,
    )
    normalized_image_features = normalize_features(image_encoded.features, "image features")
    scores[:] = normalized_image_features @ class_weights.T
    metrics = classification_metrics(scores, labels)
    metadata = {
        "image_count": image_count,
        "text_count": len(flattened_prompts),
        "text_encoding_count": len(flattened_prompts),
        "prompt_count": len(flattened_prompts),
        "class_count": len(classnames),
        "dataset_manifest_loading_seconds": dataset_manifest_loading_seconds,
        "class_names": classnames,
        "image_id_scheme": "CIFAR test-split row index",
        "image_id_order_hash": content_hash(list(range(image_count))),
        "text_id_order_hash": content_hash(flattened_prompts),
        "prompt_policy": prompt_policy,
        "prompt_name": classification_prompt_name(adapter.spec.key),
        "template_hash": template_hash,
        "classnames_hash": content_hash(classnames),
        "batch_size_actual": min(text_encoded.batch_size_actual, image_batch_size_actual),
        "image_batch_size_actual": image_batch_size_actual,
        "text_batch_size_actual": text_encoded.batch_size_actual,
        "oom_history": text_encoded.oom_history + image_encoded.oom_history,
        "data_loading_seconds": image_encoded.data_loading_seconds,
        "preprocessing_seconds": (
            text_encoded.preprocessing_seconds + image_encoded.preprocessing_seconds
        ),
        "inference_seconds": text_encoded.inference_seconds + image_encoded.inference_seconds,
        "image_inference_seconds": image_encoded.inference_seconds,
        "text_inference_seconds": text_encoded.inference_seconds,
        "batch_inference_ms": (
            text_encoded.batch_inference_ms + image_encoded.batch_inference_ms
        ),
        "image_batch_inference_ms": image_encoded.batch_inference_ms,
        "text_batch_inference_ms": text_encoded.batch_inference_ms,
        "retrieval_aggregation_seconds": 0.0,
        **_image_batch_metadata(image_encoded),
    }
    return EvaluationOutput(metrics=metrics, metadata=metadata)


def evaluate_retrieval(
    adapter: VisionLanguageAdapter,
    dataset_key: str,
    language: str,
    batch_size: int,
    *,
    retrieval_variant: RetrievalVariant | None = None,
    progress: Callable[[str], None] | None = None,
) -> EvaluationOutput:
    dataset_loading_started = time.perf_counter()
    dataset_items = retrieval_items(dataset_key, language)
    dataset_manifest_loading_seconds = time.perf_counter() - dataset_loading_started
    retrieval_variant = normalize_retrieval_variant(dataset_key, retrieval_variant)
    candidates = _retrieval_candidates(dataset_key, dataset_items, retrieval_variant)
    dataset = get_dataset(dataset_key)
    expected_image_count = dataset.expected_images
    expected_text_count = dataset.expected_text_count(language, retrieval_variant)
    if (
        expected_image_count is not None
        and len(candidates.image_ids) != expected_image_count
    ):
        raise DataIntegrityError(
            f"{dataset_key} contains {len(candidates.image_ids)} image candidates; "
            f"expected {expected_image_count}."
        )
    if expected_text_count is not None and len(candidates.text_values) != expected_text_count:
        raise DataIntegrityError(
            f"{dataset_key} {language} {retrieval_variant or 'retrieval'} contains "
            f"{len(candidates.text_values)} text candidates; expected {expected_text_count}."
        )

    def load_images(start: int, end: int) -> list[Image.Image]:
        images: list[Image.Image] = []
        for path in candidates.image_paths[start:end]:
            with Image.open(path) as image:
                image.load()
                images.append(image.convert("RGB"))
        return images

    image_encoded = _encode_batched(
        count=len(candidates.image_paths),
        load_batch=load_images,
        preprocess_batch=adapter.preprocess_images,
        encode_batch=adapter.encode_images,
        batch_size=batch_size,
        adapter=adapter,
        modality="images",
        progress=progress,
    )
    image_batch_size_actual = _image_batch_size_actual(
        adapter,
        image_encoded.batch_size_actual,
    )
    preprocess_retrieval_texts = getattr(adapter, "preprocess_retrieval_texts", None)
    encode_retrieval_texts = getattr(adapter, "encode_retrieval_texts", None)
    directional_retrieval = (
        getattr(adapter, "supports_directional_retrieval", False)
        and callable(preprocess_retrieval_texts)
        and callable(encode_retrieval_texts)
    )
    query_text_encoded: EncodedBatches | None = None
    if directional_retrieval:
        query_text_encoded = _encode_batched(
            count=len(candidates.text_values),
            load_batch=lambda start, end: candidates.text_values[start:end],
            preprocess_batch=lambda texts: preprocess_retrieval_texts(
                texts,
                "query",
                language,
            ),
            encode_batch=lambda tokens: encode_retrieval_texts(
                tokens,
                "query",
                language,
            ),
            batch_size=batch_size,
            adapter=adapter,
            modality="retrieval queries",
            progress=progress,
        )
        text_encoded = _encode_batched(
            count=len(candidates.text_values),
            load_batch=lambda start, end: candidates.text_values[start:end],
            preprocess_batch=lambda texts: preprocess_retrieval_texts(
                texts,
                "document",
                language,
            ),
            encode_batch=lambda tokens: encode_retrieval_texts(
                tokens,
                "document",
                language,
            ),
            batch_size=batch_size,
            adapter=adapter,
            modality="retrieval documents",
            progress=progress,
        )
    else:
        text_encoded = _encode_batched(
            count=len(candidates.text_values),
            load_batch=lambda start, end: candidates.text_values[start:end],
            preprocess_batch=lambda texts: adapter.preprocess_texts(
                texts,
                "retrieval",
                language,
            ),
            encode_batch=lambda tokens: adapter.encode_texts(
                tokens,
                "retrieval",
                language,
            ),
            batch_size=batch_size,
            adapter=adapter,
            modality="captions",
            progress=progress,
        )
    text_encoded_runs = [text_encoded]
    if query_text_encoded is not None:
        text_encoded_runs.append(query_text_encoded)

    aggregation_started = time.perf_counter()
    metrics = retrieval_metrics(
        image_encoded.features,
        text_encoded.features,
        candidates.positive_images_per_text,
        ks=(1, 5, 10),
        block_size=256,
        text_query_features=(
            query_text_encoded.features if query_text_encoded is not None else None
        ),
    )
    aggregation_seconds = time.perf_counter() - aggregation_started
    text_preprocessing_seconds = sum(
        encoded.preprocessing_seconds for encoded in text_encoded_runs
    )
    text_inference_seconds = sum(
        encoded.inference_seconds for encoded in text_encoded_runs
    )
    text_batch_inference_ms = [
        value
        for encoded in text_encoded_runs
        for value in encoded.batch_inference_ms
    ]
    text_oom_history = [
        item for encoded in text_encoded_runs for item in encoded.oom_history
    ]
    metadata = {
        "retrieval_variant": retrieval_variant,
        "retrieval_variant_reference_commit": (
            CLIP_BENCHMARK_COMMIT
            if retrieval_variant == "official_first_caption"
            else None
        ),
        "caption_selection": (
            "captions[0], matching the pinned CLIP Benchmark XM3600 loader"
            if retrieval_variant == "official_first_caption"
            else "all captions in source order; multiple positives retained"
        ),
        "one_caption_per_image": (
            all(count == 1 for count in candidates.caption_counts_per_image)
            if retrieval_variant == "official_first_caption"
            else None
        ),
        "image_count": len(candidates.image_ids),
        "text_count": len(candidates.text_values),
        "text_encoding_count": sum(
            encoded.features.shape[0] for encoded in text_encoded_runs
        ),
        "candidate_counts": {
            "images": len(candidates.image_ids),
            "texts": len(candidates.text_values),
        },
        "official_image_filtering": (
            {
                "rule": "include an image only when its image file exists",
                "reference_commit": CLIP_BENCHMARK_COMMIT,
                "included_count": len(candidates.image_ids),
                "excluded_count": 0,
                "excluded_image_ids": [],
            }
            if retrieval_variant == "official_first_caption"
            else None
        ),
        "class_count": None,
        "dataset_manifest_loading_seconds": dataset_manifest_loading_seconds,
        "image_id_scheme": (
            "XM3600 captions.jsonl source order using image/key"
            if dataset_key == "crossmodal3600"
            else "dataset annotation order using image_id"
        ),
        "image_ids": candidates.image_ids,
        "text_ids": candidates.text_ids,
        "image_id_order_hash": content_hash(candidates.image_ids),
        "text_id_order_hash": content_hash(candidates.text_ids),
        "text_content_sha256": content_hash(candidates.text_values),
        "text_id_scheme": "image_id#caption_index, preserving source order",
        "caption_count_per_image_min": min(candidates.caption_counts_per_image),
        "caption_count_per_image_max": max(candidates.caption_counts_per_image),
        "positive_pair_count": len(candidates.positive_images_per_text),
        "prompt_policy": (
            "qwen3_vl_directional_retrieval_instructions"
            if query_text_encoded is not None
            else (
                "embeddinggemma_searchquery_task"
                if retrieval_prompt_name(adapter.spec.key)
                else "model_native_caption_text"
            )
        ),
        "prompt_name": retrieval_prompt_name(adapter.spec.key),
        "task_prompt": retrieval_prompt_name(adapter.spec.key),
        "template_hash": None,
        "classnames_hash": None,
        "batch_size_actual": min(
            image_batch_size_actual,
            *(encoded.batch_size_actual for encoded in text_encoded_runs),
        ),
        "image_batch_size_actual": image_batch_size_actual,
        "text_batch_size_actual": min(
            encoded.batch_size_actual for encoded in text_encoded_runs
        ),
        "oom_history": image_encoded.oom_history + text_oom_history,
        "data_loading_seconds": image_encoded.data_loading_seconds,
        "preprocessing_seconds": (
            image_encoded.preprocessing_seconds + text_preprocessing_seconds
        ),
        "inference_seconds": image_encoded.inference_seconds + text_inference_seconds,
        "image_inference_seconds": image_encoded.inference_seconds,
        "text_inference_seconds": text_inference_seconds,
        "batch_inference_ms": image_encoded.batch_inference_ms + text_batch_inference_ms,
        "image_batch_inference_ms": image_encoded.batch_inference_ms,
        "text_batch_inference_ms": text_batch_inference_ms,
        "retrieval_aggregation_seconds": aggregation_seconds,
        **_image_batch_metadata(image_encoded),
    }
    return EvaluationOutput(metrics=metrics, metadata=metadata)


def _retrieval_candidates(
    dataset_key: str,
    dataset_items: Sequence[dict[str, Any]],
    retrieval_variant: RetrievalVariant | None,
) -> RetrievalCandidates:
    if not dataset_items:
        raise DataIntegrityError(f"{dataset_key} contains no retrieval items.")
    image_ids = [str(item["image_id"]) for item in dataset_items]
    if len(set(image_ids)) != len(image_ids):
        raise DataIntegrityError(f"{dataset_key} retrieval items contain duplicate image IDs.")
    image_index = {image_id: index for index, image_id in enumerate(image_ids)}
    image_paths = [Path(item["image_path"]) for item in dataset_items]
    text_values: list[str] = []
    text_ids: list[str] = []
    positive_images_per_text: list[int] = []
    caption_counts_per_image: list[int] = []
    for item in dataset_items:
        image_id = str(item["image_id"])
        captions = item.get("captions")
        if not isinstance(captions, list) or not captions:
            raise DataIntegrityError(f"Retrieval image {image_id!r} has no captions.")
        selected_captions = (
            captions[:1]
            if retrieval_variant == "official_first_caption"
            else captions
        )
        caption_counts_per_image.append(len(selected_captions))
        for caption_index, caption in enumerate(selected_captions):
            text = str(caption).strip()
            if not text:
                raise DataIntegrityError(
                    f"Retrieval image {image_id!r} has an empty caption at index "
                    f"{caption_index}."
                )
            text_values.append(text)
            text_ids.append(f"{image_id}#{caption_index}")
            positive_images_per_text.append(image_index[image_id])
    if retrieval_variant == "official_first_caption" and any(
        count != 1 for count in caption_counts_per_image
    ):
        raise DataIntegrityError(
            "The official_first_caption XM3600 variant must select exactly one caption "
            "for every included image."
        )
    return RetrievalCandidates(
        image_ids=image_ids,
        image_paths=image_paths,
        text_values=text_values,
        text_ids=text_ids,
        positive_images_per_text=positive_images_per_text,
        caption_counts_per_image=caption_counts_per_image,
    )
