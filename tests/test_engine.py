from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from vision_language_benchmarks.errors import DataIntegrityError
from vision_language_benchmarks.evaluation import engine
from vision_language_benchmarks.evaluation.engine import (
    _encode_batched,
    _retrieval_candidates,
    evaluate_classification,
    evaluate_retrieval,
)
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


def test_xm3600_retrieval_variants_preserve_order_and_positive_mappings(
    tmp_path,
) -> None:
    items = [
        {
            "image_id": "image-b",
            "image_path": str(tmp_path / "b.jpg"),
            "captions": [" B first caption ", "B second caption"],
        },
        {
            "image_id": "image-a",
            "image_path": str(tmp_path / "a.jpg"),
            "captions": ["A first caption"],
        },
    ]

    official = _retrieval_candidates(
        "crossmodal3600",
        items,
        "official_first_caption",
    )
    all_captions = _retrieval_candidates("crossmodal3600", items, "all_captions")

    assert official.image_ids == ["image-b", "image-a"]
    assert official.text_ids == ["image-b#0", "image-a#0"]
    assert official.text_values == ["B first caption", "A first caption"]
    assert official.positive_images_per_text == [0, 1]
    assert official.caption_counts_per_image == [1, 1]
    assert all_captions.image_ids == ["image-b", "image-a"]
    assert all_captions.text_ids == ["image-b#0", "image-b#1", "image-a#0"]
    assert all_captions.positive_images_per_text == [0, 0, 1]
    assert all_captions.caption_counts_per_image == [2, 1]


def test_official_caption_candidates_match_order_across_languages(
    tmp_path,
) -> None:
    english_items = [
        {
            "image_id": "image-b",
            "image_path": str(tmp_path / "b.jpg"),
            "captions": ["B caption"],
        },
        {
            "image_id": "image-a",
            "image_path": str(tmp_path / "a.jpg"),
            "captions": ["A caption"],
        },
    ]
    japanese_items = [
        {
            "image_id": "image-b",
            "image_path": str(tmp_path / "b.jpg"),
            "captions": ["B の説明"],
        },
        {
            "image_id": "image-a",
            "image_path": str(tmp_path / "a.jpg"),
            "captions": ["A の説明"],
        },
    ]

    english = _retrieval_candidates(
        "crossmodal3600",
        english_items,
        "official_first_caption",
    )
    japanese = _retrieval_candidates(
        "crossmodal3600",
        japanese_items,
        "official_first_caption",
    )

    assert english.image_ids == japanese.image_ids
    assert english.text_ids == japanese.text_ids
    assert len(english.text_values) == len(japanese.text_values) == 2


def test_official_first_caption_rejects_candidate_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        engine,
        "retrieval_items",
        lambda _dataset_key, _language: [
            {
                "image_id": "image-1",
                "image_path": "image-1.jpg",
                "captions": ["caption"],
            }
        ],
    )
    monkeypatch.setattr(
        engine,
        "get_dataset",
        lambda _dataset_key: SimpleNamespace(
            expected_images=2,
            expected_text_count=lambda _language, _retrieval_variant: 2,
        ),
    )

    with pytest.raises(DataIntegrityError, match="contains 1 image candidates"):
        evaluate_retrieval(
            SimpleNamespace(),
            "crossmodal3600",
            "en",
            16,
            retrieval_variant="official_first_caption",
        )
