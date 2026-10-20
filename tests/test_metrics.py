from __future__ import annotations

import numpy as np
import pytest

from vision_language_benchmarks.evaluation.metrics import (
    classification_metrics,
    normalize_features,
    retrieval_metrics,
)


def test_classification_metrics_preserve_stable_tie_order_and_topk() -> None:
    logits = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)

    result = classification_metrics(logits, [0, 2], topk=(1, 2))

    assert result == {"acc1": 0.5, "acc5": 0.5}


def test_classification_topk_is_null_when_candidate_pool_is_too_small() -> None:
    result = classification_metrics([[0.9, 0.1]], [0])

    assert result == {"acc1": 1.0, "acc5": None}


@pytest.mark.parametrize(
    ("logits", "targets", "topk"),
    [
        ([[1.0, 0.0]], [0.0], (1, 1)),
        ([[1.0, 0.0]], [2], (1, 1)),
        ([[1.0, 0.0]], [0], (2, 1)),
        ([[float("nan"), 0.0]], [0], (1, 1)),
    ],
)
def test_classification_rejects_invalid_inputs(
    logits: object,
    targets: list[float] | list[int],
    topk: tuple[int, int],
) -> None:
    with pytest.raises(ValueError):
        classification_metrics(logits, targets, topk=topk)


def test_normalize_features_rejects_zero_and_nonfinite_rows() -> None:
    with pytest.raises(ValueError, match="zero-norm"):
        normalize_features([[0.0, 0.0]])
    with pytest.raises(ValueError, match="NaN or infinite"):
        normalize_features([[1.0, float("inf")]])


def test_retrieval_metrics_measure_both_directions_and_multiple_captions() -> None:
    images = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    texts = np.array([[1.0, 0.0], [0.1, 0.99], [0.0, 1.0]], dtype=np.float32)

    result = retrieval_metrics(images, texts, [0, 1, 0], ks=(1, 5, 10), block_size=1)

    assert result["image_retrieval_recall@1"] == pytest.approx(2 / 3)
    assert result["text_retrieval_recall@1"] == pytest.approx(0.5)
    assert result["image_retrieval_recall@10"] == 1.0
    assert result["text_retrieval_recall@10"] == 1.0


@pytest.mark.parametrize(
    ("positives", "ks", "block_size"),
    [
        ([0], (1,), 1),
        ([0, 1], (0,), 1),
        ([0, 1], (1, 1), 1),
        ([0, 1], (1,), 0),
        ([True, 1], (1,), 1),
    ],
)
def test_retrieval_rejects_invalid_positive_mappings_and_options(
    positives: list[object],
    ks: tuple[int, ...],
    block_size: int,
) -> None:
    with pytest.raises(ValueError):
        retrieval_metrics(
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            positives,  # type: ignore[arg-type]
            ks=ks,
            block_size=block_size,
        )
