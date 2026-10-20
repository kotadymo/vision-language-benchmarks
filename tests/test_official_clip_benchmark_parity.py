from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from vision_language_benchmarks.evaluation.metrics import (
    normalize_features,
    retrieval_metrics,
)

REFERENCE_METRICS = (
    Path(__file__).parents[1]
    / ".cache"
    / "clip-benchmark-reference"
    / "clip_benchmark"
    / "metrics"
    / "zeroshot_retrieval.py"
)


@pytest.mark.skipif(
    not REFERENCE_METRICS.is_file(),
    reason="The pinned CLIP Benchmark checkout is only available in local validation.",
)
def test_retrieval_metrics_match_pinned_clip_benchmark() -> None:
    torch = pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location(
        "clip_benchmark_zeroshot_retrieval_reference",
        REFERENCE_METRICS,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load pinned reference metric: {REFERENCE_METRICS}")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)

    generator = np.random.default_rng(41)
    image_features = generator.normal(size=(12, 16)).astype(np.float32)
    text_features = generator.normal(size=(17, 16)).astype(np.float32)
    positive_images = [index % 12 for index in range(text_features.shape[0])]
    image_tensor = torch.from_numpy(normalize_features(image_features))
    text_tensor = torch.from_numpy(normalize_features(text_features))
    scores = text_tensor @ image_tensor.T
    positive_pairs = torch.zeros_like(scores, dtype=torch.bool)
    positive_pairs[
        torch.arange(len(positive_images)),
        torch.tensor(positive_images),
    ] = True

    official: dict[str, float] = {}
    for k in (1, 5, 10):
        text_to_image_recall = reference.batchify(
            reference.recall_at_k,
            scores,
            positive_pairs,
            4,
            torch.device("cpu"),
            k=k,
        )
        image_to_text_recall = reference.batchify(
            reference.recall_at_k,
            scores.T,
            positive_pairs.T,
            4,
            torch.device("cpu"),
            k=k,
        )
        official[f"image_retrieval_recall@{k}"] = float(
            (text_to_image_recall > 0).float().mean().item()
        )
        official[f"text_retrieval_recall@{k}"] = float(
            (image_to_text_recall > 0).float().mean().item()
        )

    actual = retrieval_metrics(
        image_features,
        text_features,
        positive_images,
        ks=(1, 5, 10),
        block_size=4,
    )

    assert actual == pytest.approx(official, abs=1e-7)
