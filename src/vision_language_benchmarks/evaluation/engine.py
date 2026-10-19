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
    classification_prompt_groups,
    classification_prompt_name,
    retrieval_prompt_name,
)
from vision_language_benchmarks.registry import ProtocolName


@dataclass(slots=True)
class EncodedBatches:
    features: NDArray[np.float32]
    batch_size_actual: int
    oom_history: list[dict[str, Any]]
    data_loading_seconds: float
    preprocessing_seconds: float
    inference_seconds: float
    batch_inference_ms: list[float]


@dataclass(slots=True)
class EvaluationOutput:
    metrics: dict[str, float | None]
    metadata: dict[str, Any]


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
        )
    raise RuntimeError("The batch-size fallback loop ended unexpectedly.")


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
    }
    return EvaluationOutput(metrics=metrics, metadata=metadata)


def evaluate_retrieval(
    adapter: VisionLanguageAdapter,
    dataset_key: str,
    language: str,
    batch_size: int,
    *,
    progress: Callable[[str], None] | None = None,
) -> EvaluationOutput:
    dataset_loading_started = time.perf_counter()
    dataset_items = retrieval_items(dataset_key, language)
    dataset_manifest_loading_seconds = time.perf_counter() - dataset_loading_started
    image_ids = [str(item["image_id"]) for item in dataset_items]
    if len(set(image_ids)) != len(image_ids):
        raise DataIntegrityError(f"{dataset_key} retrieval items contain duplicate image IDs.")
    image_index = {image_id: index for index, image_id in enumerate(image_ids)}
    image_paths = [Path(item["image_path"]) for item in dataset_items]

    text_values: list[str] = []
    positive_images_per_text: list[int] = []
    text_ids: list[str] = []
    for item in dataset_items:
        image_id = str(item["image_id"])
        captions = item.get("captions")
        if not isinstance(captions, list) or not captions:
            raise DataIntegrityError(f"Retrieval image {image_id!r} has no captions.")
        for caption_index, caption in enumerate(captions):
            text = str(caption).strip()
            if not text:
                raise DataIntegrityError(
                    f"Retrieval image {image_id!r} has an empty caption at index {caption_index}."
                )
            text_values.append(text)
            text_ids.append(f"{image_id}#{caption_index}")
            positive_images_per_text.append(image_index[image_id])

    def load_images(start: int, end: int) -> list[Image.Image]:
        images: list[Image.Image] = []
        for path in image_paths[start:end]:
            with Image.open(path) as image:
                image.load()
                images.append(image.convert("RGB"))
        return images

    image_encoded = _encode_batched(
        count=len(image_paths),
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
    text_encoded = _encode_batched(
        count=len(text_values),
        load_batch=lambda start, end: text_values[start:end],
        preprocess_batch=lambda texts: adapter.preprocess_texts(
            texts,
            "retrieval",
            language,
        ),
        encode_batch=lambda tokens: adapter.encode_texts(tokens, "retrieval", language),
        batch_size=batch_size,
        adapter=adapter,
        modality="captions",
        progress=progress,
    )

    aggregation_started = time.perf_counter()
    metrics = retrieval_metrics(
        image_encoded.features,
        text_encoded.features,
        positive_images_per_text,
        ks=(1, 5, 10),
        block_size=256,
    )
    aggregation_seconds = time.perf_counter() - aggregation_started
    metadata = {
        "image_count": len(image_paths),
        "text_count": len(text_values),
        "class_count": None,
        "dataset_manifest_loading_seconds": dataset_manifest_loading_seconds,
        "image_ids": image_ids,
        "text_ids": text_ids,
        "image_id_order_hash": content_hash(image_ids),
        "text_id_order_hash": content_hash(text_ids),
        "positive_pair_count": len(positive_images_per_text),
        "prompt_policy": (
            "embeddinggemma_searchquery_task"
            if retrieval_prompt_name(adapter.spec.key)
            else "model_native_caption_text"
        ),
        "prompt_name": retrieval_prompt_name(adapter.spec.key),
        "template_hash": None,
        "classnames_hash": None,
        "batch_size_actual": min(image_batch_size_actual, text_encoded.batch_size_actual),
        "image_batch_size_actual": image_batch_size_actual,
        "text_batch_size_actual": text_encoded.batch_size_actual,
        "oom_history": image_encoded.oom_history + text_encoded.oom_history,
        "data_loading_seconds": image_encoded.data_loading_seconds,
        "preprocessing_seconds": (
            image_encoded.preprocessing_seconds + text_encoded.preprocessing_seconds
        ),
        "inference_seconds": image_encoded.inference_seconds + text_encoded.inference_seconds,
        "image_inference_seconds": image_encoded.inference_seconds,
        "text_inference_seconds": text_encoded.inference_seconds,
        "batch_inference_ms": (
            image_encoded.batch_inference_ms + text_encoded.batch_inference_ms
        ),
        "image_batch_inference_ms": image_encoded.batch_inference_ms,
        "text_batch_inference_ms": text_encoded.batch_inference_ms,
        "retrieval_aggregation_seconds": aggregation_seconds,
    }
    return EvaluationOutput(metrics=metrics, metadata=metadata)
