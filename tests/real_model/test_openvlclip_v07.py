from __future__ import annotations

import gc
import hashlib
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
from vision_language_benchmarks.datasets.prepare import load_cifar_dataset, retrieval_items
from vision_language_benchmarks.evaluation.metrics import normalize_features
from vision_language_benchmarks.paths import CACHE_ROOT
from vision_language_benchmarks.registry import get_model

if os.environ.get("VLBENCH_RUN_REAL_MODEL_TESTS") != "1":
    pytest.skip(
        "Set VLBENCH_RUN_REAL_MODEL_TESTS=1 to load pretrained CUDA models.",
        allow_module_level=True,
    )

MODEL_KEYS = (
    "openvlclip_siglip2_base",
    "openvlclip_siglip2_naflex_base",
    "openvlclip_qwen3_vl_embedding_2b",
)


def _cifar_sample() -> tuple[Image.Image, str]:
    dataset = load_cifar_dataset("cifar10")
    image, target = dataset[0]
    return image.convert("RGB"), dataset.classes[int(target)]


def _naflex_samples(
    square_image: Image.Image,
) -> tuple[list[Image.Image], list[str], list[int]]:
    selected: dict[str, tuple[Image.Image, str]] = {}
    for item in retrieval_items("crossmodal3600", "en"):
        path = item["image_path"]
        with Image.open(path) as image:
            ratio = image.width / image.height
            category = (
                "landscape"
                if ratio > 1.05
                else "portrait"
                if ratio < 0.95
                else None
            )
            if category is not None and category not in selected:
                selected[category] = (
                    image.convert("RGB"),
                    str(item["image_id"]),
                )
        if len(selected) == 2:
            break
    assert set(selected) == {"landscape", "portrait"}
    images = [
        square_image,
        selected["landscape"][0],
        selected["portrait"][0],
    ]
    image_ids = ["cifar10:test:0", selected["landscape"][1], selected["portrait"][1]]
    shapes = [(image.width, image.height) for image in images]
    assert shapes[0][0] == shapes[0][1]
    assert shapes[1][0] > shapes[1][1]
    assert shapes[2][0] < shapes[2][1]
    return images, image_ids, [image.width * image.height for image in images]


def _caption_examples() -> dict[str, str]:
    examples: dict[str, str] = {}
    for language in ("en", "ja"):
        items = retrieval_items("crossmodal3600", language)
        examples[language] = str(items[0]["captions"][0])
    return examples


def _inference_smoke(adapter: Any, model_key: str) -> dict[str, Any]:
    square_image, _ = _cifar_sample()
    if model_key == "openvlclip_siglip2_naflex_base":
        images, image_ids, image_areas = _naflex_samples(square_image)
    else:
        images = [square_image]
        image_ids = ["cifar10:test:0"]
        image_areas = [square_image.width * square_image.height]

    image_batch = adapter.preprocess_images(images)
    image_features = adapter.encode_images(image_batch)
    assert image_features.shape == (
        len(images),
        get_model(model_key).global_embedding_dimension,
    )
    assert np.isfinite(image_features).all()
    normalized_images = normalize_features(image_features, "smoke image features")

    captions = _caption_examples()
    text_results: dict[str, Any] = {}
    for language, caption in captions.items():
        if model_key == "openvlclip_qwen3_vl_embedding_2b":
            query_batch = adapter.preprocess_retrieval_texts(
                [caption],
                "query",
                language,
            )
            document_batch = adapter.preprocess_retrieval_texts(
                [caption],
                "document",
                language,
            )
            query_features = adapter.encode_retrieval_texts(
                query_batch,
                "query",
                language,
            )
            document_features = adapter.encode_retrieval_texts(
                document_batch,
                "document",
                language,
            )
            text_features = query_features
            role_features = {
                "query": query_features,
                "document": document_features,
            }
        else:
            text_batch = adapter.preprocess_texts([caption], "retrieval", language)
            text_features = adapter.encode_texts(text_batch, "retrieval", language)
            role_features = {"shared": text_features}
        assert text_features.shape == (1, image_features.shape[1])
        assert np.isfinite(text_features).all()
        cosine = normalized_images @ normalize_features(
            text_features,
            f"smoke {language} text features",
        ).T
        assert cosine.shape == (len(images), 1)
        assert np.isfinite(cosine).all()
        for _role, features in role_features.items():
            assert features.shape == (1, image_features.shape[1])
            assert np.isfinite(features).all()
        text_results[language] = {
            "caption_sha256": hashlib.sha256(caption.encode("utf-8")).hexdigest(),
            "embedding_shape": list(text_features.shape),
            "normalized_cosine_similarities": cosine[:, 0].tolist(),
            "role_embedding_shapes": {
                role: list(features.shape)
                for role, features in role_features.items()
            },
        }

    image_metadata = adapter.image_batch_metadata(image_batch)
    result = {
        "image_ids": image_ids,
        "image_dimensions_and_areas": [
            {
                "width": image.width,
                "height": image.height,
                "area": image_areas[index],
            }
            for index, image in enumerate(images)
        ],
        "image_embedding_shape": list(image_features.shape),
        "text_examples": text_results,
    }
    if image_metadata:
        result["valid_patch_counts"] = image_metadata["valid_patch_counts"]
        assert len(image_metadata["valid_patch_counts"]) == len(images)
        assert max(image_metadata["valid_patch_counts"]) <= 256
        individual_features = []
        for image in images:
            individual_features.append(
                adapter.encode_images(adapter.preprocess_images([image]))
            )
        assert np.allclose(
            image_features,
            np.concatenate(individual_features, axis=0),
            rtol=1e-4,
            atol=1e-4,
        )
        result["single_image_order_check"] = "passed"
    else:
        result["single_image_order_check"] = "not_applicable"
    return result


@pytest.mark.real_model
@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_openvlclip_v07_pretrained_cuda_smoke(model_key: str) -> None:
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        pytest.skip("A CUDA-enabled PyTorch runtime and NVIDIA GPU are required.")

    spec = get_model(model_key)
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "running",
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": {
            "key": spec.key,
            "repository": spec.hf_repository,
            "checkpoint_revision": spec.checkpoint_revision,
            "openvlclip_source_revision": spec.repository_revision,
            "native_embedding_dimension": spec.global_embedding_dimension,
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
    try:
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)
        initialization_started = time.perf_counter()
        adapter = create_adapter(model_key, "cuda:0", "fp32")
        result["model_initialization_seconds"] = (
            time.perf_counter() - initialization_started
        )
        inference_started = time.perf_counter()
        result["inference"] = _inference_smoke(adapter, model_key)
        torch.cuda.synchronize(0)
        result["inference_seconds"] = time.perf_counter() - inference_started
        result["adapter_metadata"] = {
            key: value
            for key, value in adapter.metadata().items()
            if key
            in {
                "adapter_name",
                "implementation_version",
                "model_library_version",
                "checkpoint_sha256",
                "checkpoint_files",
                "tokenizer_config_sha256",
                "model_checkpoint_size_bytes",
                "preprocessing_identity",
                "max_num_patches",
                "model_download_seconds",
            }
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
        atomic_write_json(
            CACHE_ROOT / "real-smoke" / f"openvlclip-v0.7.0-{model_key}.json",
            result,
        )


@pytest.mark.real_model
def test_openvlclip_v070_masked_text_path_differs_from_official_siglip2() -> None:
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        pytest.skip("A CUDA-enabled PyTorch runtime and NVIDIA GPU are required.")

    image, _ = _cifar_sample()
    classes = load_cifar_dataset("cifar10").classes
    prompts = [f"This is a photo of a {name}." for name in classes]
    openvlclip_adapter = None
    direct_adapter = None
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "running",
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "input": {
            "dataset": "CIFAR-10 test",
            "image_id": "cifar10:test:0",
            "class_prompt_policy": "This is a photo of a {c}.",
            "precision": "fp32",
        },
    }
    try:
        openvlclip_adapter = create_adapter(
            "openvlclip_siglip2_base",
            "cuda:0",
            "fp32",
        )
        direct_adapter = create_adapter("siglip2_base_224", "cuda:0", "fp32")
        assert (
            get_model("openvlclip_siglip2_base").checkpoint_revision
            == get_model("siglip2_base_224").checkpoint_revision
        )

        open_image_batch = openvlclip_adapter.preprocess_images([image])
        direct_image_batch = direct_adapter.preprocess_images([image])
        open_pixels = open_image_batch[0].detach().cpu().numpy()
        direct_pixels = direct_image_batch["pixel_values"][0].detach().cpu().numpy()
        image_preprocess_max_abs_difference = float(
            np.max(np.abs(open_pixels - direct_pixels))
        )
        assert np.allclose(open_pixels, direct_pixels, rtol=1e-6, atol=1e-6)

        open_image_features = openvlclip_adapter.encode_images(open_image_batch)
        direct_image_features = direct_adapter.encode_images(direct_image_batch)
        open_text_batch = openvlclip_adapter.preprocess_texts(
            prompts,
            "classification",
            "en",
        )
        direct_text_batch = direct_adapter.preprocess_texts(
            prompts,
            "classification",
            "en",
        )
        open_text_features = openvlclip_adapter.encode_texts(
            open_text_batch,
            "classification",
            "en",
        )
        direct_text_features = direct_adapter.encode_texts(
            direct_text_batch,
            "classification",
            "en",
        )
        image_feature_max_abs_difference = float(
            np.max(np.abs(open_image_features - direct_image_features))
        )
        text_feature_max_abs_difference = float(
            np.max(np.abs(open_text_features - direct_text_features))
        )
        assert np.allclose(
            open_image_features,
            direct_image_features,
            rtol=1e-4,
            atol=1e-4,
        )
        assert not np.allclose(
            open_text_features,
            direct_text_features,
            rtol=1e-4,
            atol=1e-4,
        )
        open_scores = normalize_features(open_image_features) @ normalize_features(
            open_text_features
        ).T
        direct_scores = normalize_features(direct_image_features) @ normalize_features(
            direct_text_features
        ).T
        top1_ranking_identical = np.array_equal(
            np.argsort(-open_scores, axis=1),
            np.argsort(-direct_scores, axis=1),
        )
        result["status"] = "passed"
        result["checkpoint_revision"] = get_model(
            "openvlclip_siglip2_base"
        ).checkpoint_revision
        result["comparison"] = {
            "image_preprocess_max_abs_difference": image_preprocess_max_abs_difference,
            "image_feature_max_abs_difference": image_feature_max_abs_difference,
            "text_feature_max_abs_difference": text_feature_max_abs_difference,
            "text_features_match_official_default": False,
            "top1_ranking_identical": top1_ranking_identical,
        }
    except Exception as error:
        result["status"] = "failed"
        result["error_type"] = type(error).__name__
        result["error_message"] = str(error)
        raise
    finally:
        if openvlclip_adapter is not None:
            openvlclip_adapter.close()
        if direct_adapter is not None:
            direct_adapter.close()
        del openvlclip_adapter, direct_adapter
        gc.collect()
        torch.cuda.empty_cache()
        atomic_write_json(
            CACHE_ROOT
            / "real-smoke"
            / "openvlclip-v0.7.0-fixres-transformers-parity.json",
            result,
        )
