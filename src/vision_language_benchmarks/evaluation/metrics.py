from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def _finite_matrix(value: object, name: str) -> NDArray[np.float32]:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return matrix


def normalize_features(value: object, name: str = "features") -> NDArray[np.float32]:
    matrix = _finite_matrix(value, name)
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if np.any(norms == 0):
        raise ValueError(f"{name} contains a zero-norm feature.")
    return (matrix / norms.astype(np.float32)[:, None]).astype(np.float32, copy=False)


def classification_metrics(
    logits: object,
    targets: Sequence[int] | NDArray[np.integer],
    topk: tuple[int, int] = (1, 5),
) -> dict[str, float | None]:
    scores = _finite_matrix(logits, "logits")
    labels = np.asarray(targets)
    if labels.ndim != 1 or labels.shape[0] != scores.shape[0]:
        raise ValueError("targets must be a one-dimensional label for every score row.")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("targets must contain integer class indices.")
    if np.any(labels < 0) or np.any(labels >= scores.shape[1]):
        raise ValueError("targets contain a class index outside the candidate range.")
    if (
        len(topk) != 2
        or any(
            isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer))
            for k in topk
        )
        or topk[0] != 1
        or topk[1] < 1
    ):
        raise ValueError("topk must request Top-1 and a positive second cutoff.")

    rankings = np.argsort(-scores, axis=1, kind="stable")
    acc1 = float(np.mean(rankings[:, :1] == labels[:, None]))
    if scores.shape[1] < topk[1]:
        acc5 = None
    else:
        acc5 = float(np.mean(np.any(rankings[:, : topk[1]] == labels[:, None], axis=1)))
    return {"acc1": acc1, "acc5": acc5}


def _positive_sets(
    positive_images_per_text: Sequence[int | Sequence[int]],
    image_count: int,
) -> list[set[int]]:
    positives: list[set[int]] = []
    for text_index, value in enumerate(positive_images_per_text):
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            indices = {int(value)}
        else:
            if isinstance(value, (str, bytes)):
                raise ValueError(
                    f"Positive-image mapping for text {text_index} must contain integer indices."
                )
            try:
                values = list(value)  # type: ignore[arg-type]
            except TypeError as error:
                raise ValueError(
                    f"Positive-image mapping for text {text_index} is not an index or sequence."
                ) from error
            if any(
                isinstance(index, (bool, np.bool_))
                or not isinstance(index, (int, np.integer))
                for index in values
            ):
                raise ValueError(
                    f"Positive-image mapping for text {text_index} must contain integer indices."
                )
            indices = {int(index) for index in values}
        if not indices:
            raise ValueError(f"Text query {text_index} has no positive image.")
        if any(index < 0 or index >= image_count for index in indices):
            raise ValueError(f"Text query {text_index} has an out-of-range positive image.")
        positives.append(indices)
    return positives


def _blocked_hit_rate(
    queries: NDArray[np.float32],
    candidates: NDArray[np.float32],
    positives: Sequence[set[int]],
    ks: tuple[int, ...],
    block_size: int,
) -> dict[int, float]:
    if block_size < 1:
        raise ValueError("block_size must be positive.")
    hit_counts = {k: 0 for k in ks}
    candidate_count = candidates.shape[0]
    for start in range(0, queries.shape[0], block_size):
        end = min(start + block_size, queries.shape[0])
        scores = queries[start:end] @ candidates.T
        order = np.argsort(-scores, axis=1, kind="stable")
        for local_index, ranking in enumerate(order):
            positive_indices = positives[start + local_index]
            for k in ks:
                effective_k = min(k, candidate_count)
                if any(int(candidate) in positive_indices for candidate in ranking[:effective_k]):
                    hit_counts[k] += 1
    return {k: count / queries.shape[0] for k, count in hit_counts.items()}


def retrieval_metrics(
    image_features: object,
    text_features: object,
    positive_images_per_text: Sequence[int | Sequence[int]],
    ks: tuple[int, ...] = (1, 5, 10),
    block_size: int = 256,
) -> dict[str, float]:
    images = normalize_features(image_features, "image_features")
    texts = normalize_features(text_features, "text_features")
    if images.shape[1] != texts.shape[1]:
        raise ValueError("Image and text features must have the same embedding dimension.")
    if len(positive_images_per_text) != texts.shape[0]:
        raise ValueError("Provide one positive-image mapping for every text feature.")
    if block_size < 1:
        raise ValueError("block_size must be positive.")
    if not ks or any(
        isinstance(k, (bool, np.bool_))
        or not isinstance(k, (int, np.integer))
        or k < 1
        for k in ks
    ):
        raise ValueError("ks must contain positive integers.")
    if len(set(ks)) != len(ks):
        raise ValueError("ks must not contain duplicate values.")
    ks = tuple(int(k) for k in ks)

    text_positives = _positive_sets(positive_images_per_text, images.shape[0])
    image_positives: list[set[int]] = [set() for _ in range(images.shape[0])]
    for text_index, image_indices in enumerate(text_positives):
        for image_index in image_indices:
            image_positives[image_index].add(text_index)
    missing = [index for index, positives in enumerate(image_positives) if not positives]
    if missing:
        raise ValueError(
            f"{len(missing)} image queries have no positive caption; "
            f"first missing index {missing[0]}."
        )

    text_to_image = _blocked_hit_rate(texts, images, text_positives, ks, block_size)
    image_to_text = _blocked_hit_rate(images, texts, image_positives, ks, block_size)
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"image_retrieval_recall@{k}"] = text_to_image[k]
        metrics[f"text_retrieval_recall@{k}"] = image_to_text[k]
    return metrics
