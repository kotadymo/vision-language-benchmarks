from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tests.helpers import make_result
from vision_language_benchmarks.adapters import create_adapter, openclip_gemma
from vision_language_benchmarks.embedding_dimensions import (
    EMBEDDING_DIMENSION_STRATEGY,
    NATIVE_EMBEDDING_DIMENSION,
    SUPPORTED_EMBEDDING_DIMENSIONS,
    resolve_embedding_dimension,
)
from vision_language_benchmarks.errors import DataIntegrityError
from vision_language_benchmarks.evaluation import runner
from vision_language_benchmarks.reporting.schema import validate_result


@pytest.mark.parametrize("dimension", SUPPORTED_EMBEDDING_DIMENSIONS)
def test_embeddinggemma_accepts_supported_native_dimensions(dimension: int) -> None:
    assert resolve_embedding_dimension("embeddinggemma2", dimension) == dimension


def test_embeddinggemma_defaults_to_native_dimension() -> None:
    assert resolve_embedding_dimension("embeddinggemma2", None) == 768


@pytest.mark.parametrize("dimension", (0, 64, 1024, True, 256.0))
def test_embeddinggemma_rejects_unsupported_dimensions(dimension: Any) -> None:
    with pytest.raises(ValueError, match="Unsupported EmbeddingGemma"):
        resolve_embedding_dimension("embeddinggemma2", dimension)


def test_embedding_dimension_is_not_available_for_other_models() -> None:
    with pytest.raises(ValueError, match="only for the embeddinggemma2"):
        create_adapter("openai_clip_b32", "cpu", "fp32", embedding_dimension=128)


def test_embedding_dimension_changes_skipped_run_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(runner, "_source_commit_sha", lambda: "a" * 40)
    monkeypatch.setattr(runner, "_working_tree_dirty", lambda: False)
    monkeypatch.setattr(runner, "_lock_sha256", lambda: "b" * 64)
    monkeypatch.setattr(
        runner,
        "doctor_report",
        lambda device: {
            "torch": {
                "cuda_available": False,
                "devices": [],
                "version": "test",
                "cuda_runtime": None,
            },
            "nvidia_smi": {"devices": []},
        },
    )

    results = [
        runner.run_benchmark(
            "embeddinggemma2",
            "crossmodal3600",
            protocol="p2",
            language="fr",
            device="cpu",
            embedding_dimension=dimension,
            results_root=tmp_path,
        )
        for dimension in (128, 256)
    ]

    assert all(result["status"] == "skipped" for result in results)
    assert len({result["benchmark_metadata"]["configuration_hash"] for result in results}) == 2
    assert len({result["benchmark_metadata"]["run_id"] for result in results}) == 2
    assert {
        result["benchmark_metadata"]["run_configuration"]["embedding_dimension"]
        for result in results
    } == {128, 256}


class _Tensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.shape = values.shape

    def to(self, _device: object) -> _Tensor:
        return self

    def detach(self) -> _Tensor:
        return self

    def float(self) -> _Tensor:
        return self

    def cpu(self) -> _Tensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class _Model:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.image_calls: list[dict[str, Any]] = []
        self.text_calls: list[dict[str, Any]] = []

    def _features(self, count: int) -> _Tensor:
        values = np.tile(
            np.arange(1, self.dimension + 1, dtype=np.float32),
            (count, 1),
        )
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        return _Tensor(values)

    def encode_image(self, batch: _Tensor, **kwargs: Any) -> _Tensor:
        self.image_calls.append(kwargs)
        return self._features(batch.shape[0])

    def encode_text(self, batch: _Tensor, **kwargs: Any) -> _Tensor:
        self.text_calls.append(kwargs)
        return self._features(batch.shape[0])


def _adapter(dimension: int) -> openclip_gemma.Adapter:
    adapter = object.__new__(openclip_gemma.Adapter)
    adapter.embedding_dimension = dimension
    adapter.model = _Model(dimension)
    adapter.torch = SimpleNamespace(stack=lambda tensors: _Tensor(
        np.stack([tensor.values for tensor in tensors])
    ))
    adapter.device_object = object()
    adapter.inference_context = nullcontext
    adapter.autocast_context = nullcontext
    adapter.image_shape_group_max_batch_size = 0
    return adapter


@pytest.mark.parametrize("dimension", SUPPORTED_EMBEDDING_DIMENSIONS)
def test_adapter_calls_native_truncate_dim_and_checks_normalized_features(
    dimension: int,
) -> None:
    adapter = _adapter(dimension)
    image_batch = [
        _Tensor(np.ones((3, 2, 2), dtype=np.float32)),
        _Tensor(np.full((3, 2, 2), 2, dtype=np.float32)),
    ]

    image_features = adapter.encode_images(image_batch)
    text_features = adapter.encode_texts(
        _Tensor(np.ones((2, 4), dtype=np.int64)),
        "retrieval",
        "en",
    )

    assert image_features.shape == (2, dimension)
    assert text_features.shape == (2, dimension)
    assert adapter.model.image_calls == [
        {"normalize": False, "truncate_dim": dimension}
    ]
    assert adapter.model.text_calls == [
        {"normalize": False, "truncate_dim": dimension}
    ]
    assert np.allclose(np.linalg.norm(image_features, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(text_features, axis=1), 1.0)


def test_adapter_rejects_non_normalized_native_embeddings() -> None:
    adapter = _adapter(128)
    adapter.model._features = lambda count: _Tensor(
        np.ones((count, 128), dtype=np.float32)
    )

    with pytest.raises(DataIntegrityError, match="non-normalized image"):
        adapter.encode_images([_Tensor(np.ones((3, 2, 2), dtype=np.float32))])


def test_dimension_metadata_is_optional_for_legacy_results() -> None:
    legacy_result = make_result(model_key="embeddinggemma2")

    validate_result(legacy_result)


def test_dimension_results_require_matching_native_v04_metadata() -> None:
    result = make_result(model_key="embeddinggemma2", embedding_dimension=128)

    validate_result(result)

    metadata = result["benchmark_metadata"]
    assert metadata["embedding_dimension"] == 128
    assert metadata["native_embedding_dimension"] == NATIVE_EMBEDDING_DIMENSION
    assert metadata["embedding_dimension_strategy"] == EMBEDDING_DIMENSION_STRATEGY
    assert metadata["embedding_generation_mode"] == "native_truncate_dim"
    assert metadata["run_configuration"]["embedding_dimension"] == 128

    metadata["run_configuration"]["embedding_dimension"] = 256
    with pytest.raises(ValueError, match="embedding dimensions must match"):
        validate_result(result)
