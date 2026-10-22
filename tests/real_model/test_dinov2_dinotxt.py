from __future__ import annotations

import importlib
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from vision_language_benchmarks.adapters import create_adapter
from vision_language_benchmarks.atomic import atomic_write_json
from vision_language_benchmarks.dinotxt_provenance import (
    DINOTXT_MODEL_ID,
    DINOV2_SOURCE_REVISION,
    verify_cached_components,
)
from vision_language_benchmarks.errors import ModelAccessBlockedError
from vision_language_benchmarks.evaluation.metrics import normalize_features
from vision_language_benchmarks.registry import get_model

if os.environ.get("VLBENCH_RUN_DINOTXT_REAL_MODEL_TESTS") != "1":
    pytest.skip(
        "Set VLBENCH_RUN_DINOTXT_REAL_MODEL_TESTS=1 to run CUDA model inference.",
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
        assert adapter.device_object.type == "cuda"
        image_batch = adapter.preprocess_images([image])
        assert tuple(image_batch.shape) == (1, 3, 224, 224)
        assert image_batch.device.type == "cuda"
        image_batch_cpu = image_batch.detach().cpu()
        torch.cuda.reset_peak_memory_stats(adapter.device_object)
        image_started = time.perf_counter()
        image_features = adapter.encode_images(image_batch)
        torch.cuda.synchronize(adapter.device_object)
        image_inference_seconds = time.perf_counter() - image_started
        del image_batch

        language_results: dict[str, dict[str, object]] = {}
        wrapper_text_features: dict[str, np.ndarray] = {}
        english_tokens_cpu: Any | None = None
        for language, text in (
            ("en", "A red bicycle beside a tree."),
            ("ja", "木の横に停められた赤い自転車。"),
        ):
            tokens = adapter.preprocess_texts([text], "retrieval", language)
            assert tuple(tokens.shape) == (1, 77)
            assert tokens.device.type == "cuda"
            if language == "en":
                english_tokens_cpu = tokens.detach().cpu()
            text_started = time.perf_counter()
            text_features = adapter.encode_texts(tokens, "retrieval", language)
            torch.cuda.synchronize(adapter.device_object)
            text_inference_seconds = time.perf_counter() - text_started
            wrapper_text_features[language] = text_features
            normalized_image = normalize_features(image_features, "dino.txt smoke image")
            normalized_text = normalize_features(
                text_features,
                f"dino.txt smoke {language} text",
            )
            assert image_features.shape == (1, 2048)
            assert text_features.shape == (1, 2048)
            assert np.isfinite(image_features).all()
            assert np.isfinite(text_features).all()
            assert np.allclose(np.linalg.norm(normalized_image, axis=1), 1.0)
            assert np.allclose(np.linalg.norm(normalized_text, axis=1), 1.0)
            cosine_similarity = float(np.sum(normalized_image * normalized_text))
            assert np.isfinite(cosine_similarity)
            assert -1.00001 <= cosine_similarity <= 1.00001
            language_results[language] = {
                "text": text,
                "token_shape": list(tokens.shape),
                "embedding_dimension": int(text_features.shape[1]),
                "cosine_similarity": cosine_similarity,
                "text_inference_seconds": text_inference_seconds,
            }

        adapter_metadata = adapter.metadata()
        assert adapter_metadata["model_library_version"] == "0.6.0"
        assert adapter_metadata["global_embedding_dimension"] == 2048
        assert adapter_metadata["checkpoint_components"]
        assert all(
            component["verified"]
            and component["license_status"] == "verified"
            for component in adapter_metadata["checkpoint_components"]
        )
        wrapper_memory = {
            "current_allocated_bytes": int(
                torch.cuda.memory_allocated(adapter.device_object)
            ),
            "current_reserved_bytes": int(
                torch.cuda.memory_reserved(adapter.device_object)
            ),
            "peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(adapter.device_object)
            ),
            "peak_reserved_bytes": int(
                torch.cuda.max_memory_reserved(adapter.device_object)
            ),
        }
        assert wrapper_memory["peak_allocated_bytes"] > 0
        assert wrapper_memory["peak_reserved_bytes"] > 0
    finally:
        adapter.close()
    del adapter

    official_model = None
    try:
        official_model = torch.hub.load(
            f"facebookresearch/dinov2:{DINOV2_SOURCE_REVISION}",
            DINOTXT_MODEL_ID,
            trust_repo=True,
        ).to("cuda:0").eval()
        official_image_batch = image_batch_cpu.to("cuda:0")
        if english_tokens_cpu is None:
            raise AssertionError("English tokenizer output was not retained for comparison.")
        official_text_tokens = english_tokens_cpu.to("cuda:0")
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            official_image_features = (
                official_model.encode_image(official_image_batch, normalize=False)
                .float()
                .cpu()
                .numpy()
            )
            official_text_features = (
                official_model.encode_text(official_text_tokens, normalize=False)
                .float()
                .cpu()
                .numpy()
            )
        torch.cuda.synchronize()
        official_memory = {
            "current_allocated_bytes": int(torch.cuda.memory_allocated()),
            "current_reserved_bytes": int(torch.cuda.memory_reserved()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        assert next(official_model.parameters()).device.type == "cuda"
        assert np.allclose(
            image_features,
            official_image_features,
            rtol=1e-5,
            atol=1e-6,
        )
        assert np.allclose(
            wrapper_text_features["en"],
            official_text_features,
            rtol=1e-5,
            atol=1e-6,
        )

        verified_components = verify_cached_components(
            Path(torch.hub.get_dir()) / "checkpoints"
        )
        assert [
            component["sha256"] for component in adapter_metadata["checkpoint_components"]
        ] == [component["sha256"] for component in verified_components]
        smoke_record = {
            "schema_version": "1.0.0",
            "status": "completed",
            "source_revision": DINOV2_SOURCE_REVISION,
            "model_key": spec.key,
            "model_id": spec.model_id,
            "model_library": "openvlclip",
            "model_library_version": adapter_metadata["model_library_version"],
            "precision": "fp32",
            "device": "cuda:0",
            "gpu_name": torch.cuda.get_device_name(0),
            "input_shape": list(image_batch_cpu.shape),
            "token_context_length": 77,
            "embedding_dimension": 2048,
            "checkpoint_sha256": adapter_metadata["checkpoint_sha256"],
            "checkpoint_components": verified_components,
            "wrapper_image_inference_seconds": image_inference_seconds,
            "wrapper_gpu_memory": wrapper_memory,
            "official_gpu_memory": official_memory,
            "official_wrapper_max_abs_error": {
                "image": float(np.max(np.abs(image_features - official_image_features))),
                "text": float(
                    np.max(
                        np.abs(
                            wrapper_text_features["en"] - official_text_features
                        )
                    )
                ),
            },
            "languages": language_results,
        }
        smoke_output = os.environ.get("VLBENCH_DINOTXT_SMOKE_OUTPUT")
        if smoke_output:
            atomic_write_json(Path(smoke_output), smoke_record)
    finally:
        if official_model is not None:
            del official_model
        torch.cuda.empty_cache()
