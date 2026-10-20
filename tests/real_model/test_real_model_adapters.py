from __future__ import annotations

import gc
import importlib
import os
import platform
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pytest
from PIL import Image

from vision_language_benchmarks.adapters import create_adapter
from vision_language_benchmarks.atomic import atomic_write_json
from vision_language_benchmarks.evaluation.metrics import normalize_features
from vision_language_benchmarks.paths import CACHE_ROOT
from vision_language_benchmarks.registry import get_model

if os.environ.get("VLBENCH_RUN_REAL_MODEL_TESTS") != "1":
    pytest.skip(
        "Set VLBENCH_RUN_REAL_MODEL_TESTS=1 to load pretrained CUDA models.",
        allow_module_level=True,
    )

MODEL_KEYS = (
    "openai_clip_b32",
    "openclip_b32_laion2b",
    "openclip_l14_laion2b",
    "embeddinggemma2",
    "siglip2_base_224",
    "openclip_multilingual_b32",
    "rinna_ja_clip_b16",
)

TEXT_EXAMPLES = {
    "en": (
        "A red bicycle parked beside a tree.",
        "A dog running across a grassy field.",
    ),
    "ja": (
        "木の横に停められた赤い自転車。",
        "草原を走っている犬。",
    ),
}


def _run_inference(adapter: Any, languages: tuple[str, ...]) -> dict[str, Any]:
    image_values = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
    image = Image.fromarray(image_values).convert("RGB")
    image_batch = adapter.preprocess_images([image])
    try:
        image_features = adapter.encode_images(image_batch)
    finally:
        del image_batch

    assert image_features.shape[0] == 1
    assert image_features.shape[1] > 0
    assert np.isfinite(image_features).all()
    normalized_image_features = normalize_features(image_features, "smoke image features")
    assert np.allclose(np.linalg.norm(normalized_image_features, axis=1), 1.0)

    language_results: dict[str, Any] = {}
    for language in languages:
        text_batch = adapter.preprocess_texts(
            TEXT_EXAMPLES[language],
            "retrieval",
            language,
        )
        try:
            text_features = adapter.encode_texts(text_batch, "retrieval", language)
        finally:
            del text_batch

        assert text_features.shape[0] == len(TEXT_EXAMPLES[language])
        assert text_features.shape[1] == image_features.shape[1]
        assert np.isfinite(text_features).all()
        normalized_text_features = normalize_features(
            text_features,
            f"smoke {language} text features",
        )
        assert np.allclose(np.linalg.norm(normalized_text_features, axis=1), 1.0)
        cosine_similarities = normalized_image_features @ normalized_text_features.T
        assert cosine_similarities.shape == (1, len(TEXT_EXAMPLES[language]))
        assert np.isfinite(cosine_similarities).all()
        assert np.all(np.abs(cosine_similarities) <= 1.0 + 1e-5)
        language_results[language] = {
            "text_embedding_shape": list(text_features.shape),
            "normalized_cosine_similarities": cosine_similarities[0].tolist(),
        }

    return {
        "image_embedding_shape": list(image_features.shape),
        "languages": language_results,
    }


@pytest.mark.real_model
@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_pretrained_adapter_cuda_smoke(model_key: str) -> None:
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        pytest.skip("A CUDA-enabled PyTorch runtime and NVIDIA GPU are required.")

    spec = get_model(model_key)
    languages = tuple(language for language in ("en", "ja") if language in spec.languages)
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "running",
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": {
            "key": spec.key,
            "repository": spec.hf_repository or spec.model_id,
            "checkpoint_revision": spec.checkpoint_revision,
            "pretrained_tag": spec.pretrained_tag,
            "tokenizer_repository": spec.tokenizer_repository,
            "tokenizer_revision": spec.tokenizer_revision,
            "languages": list(spec.languages),
            "license": spec.license,
        },
        "runtime": {
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "device": "cuda:0",
            "precision": "fp32",
        },
    }
    adapter = None
    torch.cuda.synchronize(0)
    torch.cuda.reset_peak_memory_stats(0)
    try:
        initialization_started = time.perf_counter()
        adapter = create_adapter(spec.key, "cuda:0", "fp32")
        result["model_initialization_seconds"] = time.perf_counter() - initialization_started

        model = getattr(adapter, "model", None)
        assert model is not None
        first_parameter = next(model.parameters(), None)
        assert first_parameter is not None
        assert first_parameter.device.type == "cuda"

        inference_started = time.perf_counter()
        result["inference"] = _run_inference(adapter, languages)
        torch.cuda.synchronize(0)
        result["inference_seconds"] = time.perf_counter() - inference_started
        adapter_metadata = adapter.metadata()
        result["adapter_metadata"] = {
            key: adapter_metadata[key]
            for key in (
                "adapter_name",
                "implementation_version",
                "checkpoint_sha256",
                "checkpoint_files",
                "model_download_seconds",
                "max_text_length",
                "parameter_count",
                "tokenizer_repository",
                "tokenizer_revision",
                "tokenizer_license",
            )
            if key in adapter_metadata
        }
        result["gpu_memory"] = {
            "allocated_bytes": int(torch.cuda.memory_allocated(0)),
            "reserved_bytes": int(torch.cuda.memory_reserved(0)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        }
        assert len(result["adapter_metadata"]["checkpoint_sha256"]) == 64
        result["status"] = "passed"
    except Exception as error:
        result["status"] = "failed"
        result["error_type"] = type(error).__name__
        result["error_message"] = str(error)
        raise
    finally:
        if adapter is not None:
            adapter.close()
            del adapter
        gc.collect()
        torch.cuda.empty_cache()
        log_path = CACHE_ROOT / "real-smoke" / f"{model_key}.json"
        atomic_write_json(log_path, result)
