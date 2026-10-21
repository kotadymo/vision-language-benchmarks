from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pytest

from vision_language_benchmarks.reporting.mrl_equivalence import (
    compare_embedding_arrays,
    compare_native_mrl_batch,
    render_native_mrl_api_equivalence_markdown,
)


def _normalized(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values, axis=1, keepdims=True)


class _Tensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.shape = values.shape
        self.ndim = values.ndim

    def __getitem__(self, index: object) -> _Tensor:
        return _Tensor(self.values[index])

    def detach(self) -> _Tensor:
        return self

    def float(self) -> _Tensor:
        return self

    def cpu(self) -> _Tensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class _FakeTorch:
    @staticmethod
    def inference_mode():
        return nullcontext()

    class nn:
        class functional:
            @staticmethod
            def normalize(values: _Tensor, p: int, dim: int) -> _Tensor:
                norms = np.linalg.norm(
                    values.values,
                    ord=p,
                    axis=dim,
                    keepdims=True,
                )
                return _Tensor(values.values / norms)


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, bool]] = []

    def encode_image(
        self,
        inputs: _Tensor,
        *,
        normalize: bool,
        truncate_dim: int,
    ) -> _Tensor:
        return self._encode("image", inputs, normalize, truncate_dim)

    def encode_text(
        self,
        inputs: _Tensor,
        *,
        normalize: bool,
        truncate_dim: int,
    ) -> _Tensor:
        return self._encode("text", inputs, normalize, truncate_dim)

    def _encode(
        self,
        modality: str,
        inputs: _Tensor,
        normalize: bool,
        truncate_dim: int,
    ) -> _Tensor:
        self.calls.append((modality, truncate_dim, normalize))
        prefix = inputs.values[..., :truncate_dim]
        normalized = prefix / np.linalg.norm(prefix, ord=2, axis=-1, keepdims=True)
        return _Tensor(normalized)


def test_compare_embedding_arrays_reports_close_normalized_vectors() -> None:
    reference = _normalized(
        np.asarray([[3.0, 4.0, 1.0], [1.0, -2.0, 2.0]], dtype=np.float32)
    )

    result = compare_embedding_arrays(reference.copy(), reference)

    assert result["compared_embedding_count"] == 2
    assert result["valid_comparison_count"] == 2
    assert result["invalid_embedding_count"] == 0
    assert result["max_absolute_difference"] == 0.0
    assert result["mean_absolute_difference"] == 0.0
    assert result["minimum_cosine_similarity"] == pytest.approx(1.0)
    assert result["maximum_l2_norm_deviation_from_one"] < 1e-7
    assert result["tolerance"] == {"rtol": 1e-5, "atol": 1e-6}
    assert result["passed"] is True


@pytest.mark.parametrize("modality", ["image", "text"])
def test_compare_native_mrl_batch_measures_native_dimension_order(
    modality: str,
) -> None:
    inputs = _Tensor(np.asarray([[3.0, 4.0, 12.0]], dtype=np.float32))
    model = _FakeModel()

    results = compare_native_mrl_batch(
        model,
        inputs,
        modality,
        (3, 2, 1),
        _FakeTorch,
        reference_dimension=3,
    )

    assert list(results) == [3, 2, 1]
    assert [results[dimension]["native_embeddings"].shape for dimension in results] == [
        (1, 3),
        (1, 2),
        (1, 1),
    ]
    assert all(results[dimension]["comparison"]["passed"] for dimension in results)
    assert model.calls == [
        (modality, 3, False),
        (modality, 3, False),
        (modality, 2, False),
        (modality, 1, False),
    ]


def test_compare_embedding_arrays_fails_outside_tolerance() -> None:
    reference = _normalized(np.asarray([[1.0, 2.0]], dtype=np.float32))
    native = reference.copy()
    native[0, 0] += 1e-3

    result = compare_embedding_arrays(native, reference)

    assert result["max_absolute_difference"] > 1e-4
    assert result["within_tolerance"] is False
    assert result["passed"] is False


def test_compare_embedding_arrays_excludes_near_zero_coordinates_from_relative_error() -> None:
    reference = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    native = np.asarray([[1.0, 5e-7], [5e-7, 1.0]], dtype=np.float32)

    result = compare_embedding_arrays(native, reference)

    assert result["relative_error_reference_abs_threshold"] == 1e-6
    assert result["relative_error_coordinate_count"] == 2
    assert result["max_relative_difference"] == 0.0
    assert result["passed"] is True


def test_compare_embedding_arrays_counts_nonfinite_and_zero_norm_vectors() -> None:
    native = np.asarray([[np.nan, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    reference = np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    result = compare_embedding_arrays(native, reference)

    assert result["non_finite_embedding_count"] == 1
    assert result["zero_norm_embedding_count"] == 1
    assert result["invalid_embedding_count"] == 2
    assert result["valid_comparison_count"] == 1
    assert result["passed"] is False


@pytest.mark.parametrize(
    ("native", "reference"),
    [
        (np.ones(3), np.ones((1, 3))),
        (np.ones((1, 3)), np.ones((1, 2))),
        (np.empty((0, 3)), np.empty((0, 3))),
    ],
)
def test_compare_embedding_arrays_rejects_invalid_shapes(
    native: np.ndarray,
    reference: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        compare_embedding_arrays(native, reference)


def test_render_native_mrl_api_equivalence_markdown_keeps_scope_and_tolerance_clear() -> None:
    report = {
        "status": "completed",
        "source_commit_sha": "a" * 40,
        "timestamp_utc": "2026-10-21T00:00:00Z",
        "library_version": "0.4.0",
        "model_id": "google/embeddinggemma-2",
        "model_revision": "b" * 40,
        "checkpoint": {"sha256": "c" * 64},
        "environment": {
            "gpu_name": "GPU",
            "cuda_runtime": "13.0",
            "pytorch_version": "2.14.1",
            "python_version": "3.12.10",
        },
        "working_tree_dirty": False,
        "tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "sample_selection": {
            "cifar10": {
                "selection_description": "indices 0 and 1",
                "dataset_checksum": "d" * 64,
                "text_samples": {
                    "en": {
                        "sample_count": 2,
                        "text_ids_sha256": "e" * 64,
                        "text_content_sha256": "f" * 64,
                    }
                },
            }
        },
        "measurements": [
            {
                "embedding_dimension": 128,
                "modality": "classification_text",
                "dataset_ids": ["cifar10"],
                "protocol": "p1",
                "language": "en",
                "native_vs_reference": {
                    "compared_embedding_count": 2,
                    "max_absolute_difference": 0.0,
                    "mean_absolute_difference": 0.0,
                    "max_relative_difference": 0.0,
                    "minimum_cosine_similarity": 1.0,
                    "maximum_l2_norm_deviation_from_one": 0.0,
                    "invalid_embedding_count": 0,
                    "passed": True,
                },
                "adapter_output": {
                    "dtype": "float32",
                    "vs_direct_native": {
                        "max_absolute_difference": 0.0,
                        "mean_absolute_difference": 0.0,
                        "maximum_l2_norm_deviation_from_one": 0.0,
                        "invalid_embedding_count": 0,
                        "passed": True,
                    },
                },
            }
        ],
    }

    markdown = render_native_mrl_api_equivalence_markdown(report)

    assert "does not reproduce their classification or retrieval accuracy scores" in markdown
    assert "rtol=1e-05" in markdown
    assert "classification_text" in markdown
    assert "Ranking agreement is not reported" in markdown
