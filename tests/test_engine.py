from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from vision_language_benchmarks.evaluation import engine
from vision_language_benchmarks.evaluation.engine import _encode_batched, evaluate_classification
from vision_language_benchmarks.registry import get_model


class FakeOutOfMemoryError(Exception):
    pass


class FakeCuda:
    OutOfMemoryError = FakeOutOfMemoryError

    def __init__(self) -> None:
        self.empty_cache_calls = 0
        self.synchronized_devices: list[object] = []

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1

    def synchronize(self, device: object) -> None:
        self.synchronized_devices.append(device)


def test_encoder_retries_smaller_batches_after_cuda_oom() -> None:
    cuda = FakeCuda()
    adapter = SimpleNamespace(
        torch=SimpleNamespace(cuda=cuda),
        device_object=SimpleNamespace(type="cuda"),
    )

    def encode(values: list[int]) -> np.ndarray:
        if len(values) == 4:
            raise FakeOutOfMemoryError("synthetic OOM")
        return np.asarray([[value, 1] for value in values], dtype=np.float32)

    result = _encode_batched(
        count=5,
        load_batch=lambda start, end: list(range(start, end)),
        preprocess_batch=lambda values: values,
        encode_batch=encode,
        batch_size=4,
        adapter=adapter,
        modality="images",
    )

    assert result.batch_size_actual == 2
    assert result.features.shape == (5, 2)
    assert result.oom_history == [
        {
            "modality": "images",
            "batch_size": 4,
            "stage": "warmup",
            "message": "synthetic OOM",
        }
    ]
    assert cuda.empty_cache_calls == 1


def test_classification_normalizes_mean_prompt_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SyntheticDataset:
        classes = ["class-zero", "class-one"]
        targets = [1]

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> tuple[object, int]:
            return object(), self.targets[index]

    angle = np.arccos(0.2)
    class_one_prompts = np.array(
        [[np.cos(angle), np.sin(angle)], [np.cos(angle), -np.sin(angle)]] * 9,
        dtype=np.float32,
    )
    class_zero_prompts = np.tile(
        np.array([[0.8, 0.6]], dtype=np.float32),
        (18, 1),
    )

    def fake_encode_batched(**kwargs: Any) -> engine.EncodedBatches:
        if kwargs["modality"] == "class prompts":
            features = np.concatenate((class_zero_prompts, class_one_prompts))
        else:
            features = np.array([[1.0, 0.0]], dtype=np.float32)
        return engine.EncodedBatches(
            features=features,
            batch_size_actual=1,
            oom_history=[],
            data_loading_seconds=0.0,
            preprocessing_seconds=0.0,
            inference_seconds=0.0,
            batch_inference_ms=[],
        )

    def fake_load_cifar_dataset(dataset_key: str) -> SyntheticDataset:
        assert dataset_key == "cifar10"
        return dataset

    dataset = SyntheticDataset()
    adapter = SimpleNamespace(
        spec=get_model("openai_clip_b32"),
        preprocess_images=None,
        encode_images=None,
        preprocess_texts=None,
        encode_texts=None,
    )
    monkeypatch.setattr(engine, "load_cifar_dataset", fake_load_cifar_dataset)
    monkeypatch.setattr(engine, "_encode_batched", fake_encode_batched)
    result = evaluate_classification(
        adapter,
        "cifar10",
        "p0",
        "en",
        1,
    )

    assert result.metrics["acc1"] == 1.0
