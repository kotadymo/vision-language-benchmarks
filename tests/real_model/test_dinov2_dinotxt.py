from __future__ import annotations

import importlib
import os

import numpy as np
import pytest
from PIL import Image

from vision_language_benchmarks.adapters import create_adapter
from vision_language_benchmarks.errors import ModelAccessBlockedError
from vision_language_benchmarks.evaluation.metrics import normalize_features
from vision_language_benchmarks.registry import get_model

if os.environ.get("VLBENCH_RUN_DINOTXT_REAL_MODEL_TESTS") != "1":
    pytest.skip(
        "Set VLBENCH_RUN_DINOTXT_REAL_MODEL_TESTS=1 only after weight-use rights are confirmed.",
        allow_module_level=True,
    )


def test_dinotxt_adapter_cuda_image_and_bilingual_text_smoke() -> None:
    spec = get_model("dinov2_dinotxt")
    if spec.weight_license_status != "verified":
        raise ModelAccessBlockedError(
            "dino.txt real-model inference is blocked until all component licenses are verified."
        )
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        pytest.skip("A CUDA-enabled PyTorch runtime and NVIDIA GPU are required.")

    image_array = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
    image = Image.fromarray(image_array).convert("RGB")
    adapter = create_adapter(spec.key, "cuda:0", "fp32")
    try:
        image_features = adapter.encode_images(adapter.preprocess_images([image]))
        image_features = normalize_features(image_features, "dino.txt smoke image")
        assert image_features.shape == (1, 2048)
        assert np.isfinite(image_features).all()
        assert np.allclose(np.linalg.norm(image_features, axis=1), 1.0)

        for language, text in (
            ("en", "A red bicycle beside a tree."),
            ("ja", "木の横に停められた赤い自転車。"),
        ):
            tokens = adapter.preprocess_texts([text], "retrieval", language)
            text_features = adapter.encode_texts(tokens, "retrieval", language)
            text_features = normalize_features(
                text_features,
                f"dino.txt smoke {language} text",
            )
            assert text_features.shape == (1, 2048)
            assert np.isfinite(text_features).all()
            assert np.allclose(np.linalg.norm(text_features, axis=1), 1.0)
    finally:
        adapter.close()
