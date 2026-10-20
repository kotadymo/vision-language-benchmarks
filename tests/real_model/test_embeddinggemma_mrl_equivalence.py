from __future__ import annotations

import importlib
import os

import pytest
from PIL import Image

from vision_language_benchmarks.adapters import create_adapter
from vision_language_benchmarks.embedding_dimensions import SUPPORTED_EMBEDDING_DIMENSIONS

if os.environ.get("VLBENCH_RUN_MRL_INTEGRATION") != "1":
    pytest.skip(
        "Set VLBENCH_RUN_MRL_INTEGRATION=1 to verify native CUDA truncation.",
        allow_module_level=True,
    )


@pytest.mark.real_model
def test_native_image_and_text_truncation_matches_manual_prefix_normalization() -> None:
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        pytest.skip("A CUDA-enabled PyTorch runtime and NVIDIA GPU are required.")

    adapter = create_adapter(
        "embeddinggemma2",
        "cuda",
        "fp32",
        embedding_dimension=768,
    )
    try:
        image = Image.new("RGB", (256, 256), color=(73, 124, 201))
        pixels = torch.stack(adapter.preprocess_images([image])).to(adapter.device_object)
        tokens = adapter.preprocess_texts(
            [
                "A blue object in a simple image.",
                "A short text used to validate the embedding output.",
            ],
            "retrieval",
            "en",
        )
        with adapter.inference_context():
            full_image_features = adapter.model.encode_image(pixels, normalize=False)
            full_text_features = adapter.model.encode_text(tokens, normalize=False)
            assert full_image_features.shape == (1, 768)
            assert full_text_features.shape == (2, 768)

            for dimension in SUPPORTED_EMBEDDING_DIMENSIONS:
                image_features = adapter.model.encode_image(
                    pixels,
                    normalize=False,
                    truncate_dim=dimension,
                )
                text_features = adapter.model.encode_text(
                    tokens,
                    normalize=False,
                    truncate_dim=dimension,
                )
                expected_image = torch.nn.functional.normalize(
                    full_image_features[..., :dimension],
                    p=2,
                    dim=-1,
                )
                expected_text = torch.nn.functional.normalize(
                    full_text_features[..., :dimension],
                    p=2,
                    dim=-1,
                )

                assert image_features.shape == (1, dimension)
                assert text_features.shape == (2, dimension)
                assert torch.isfinite(image_features).all()
                assert torch.isfinite(text_features).all()
                torch.testing.assert_close(
                    image_features,
                    expected_image,
                    rtol=1e-5,
                    atol=1e-6,
                )
                torch.testing.assert_close(
                    text_features,
                    expected_text,
                    rtol=1e-5,
                    atol=1e-6,
                )
                torch.testing.assert_close(
                    torch.linalg.vector_norm(image_features, dim=-1),
                    torch.ones(1, device=adapter.device_object),
                    rtol=1e-5,
                    atol=1e-6,
                )
                torch.testing.assert_close(
                    torch.linalg.vector_norm(text_features, dim=-1),
                    torch.ones(2, device=adapter.device_object),
                    rtol=1e-5,
                    atol=1e-6,
                )
    finally:
        adapter.close()
