from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from vision_language_benchmarks.adapters import create_adapter
from vision_language_benchmarks.atomic import atomic_write_json, content_hash
from vision_language_benchmarks.datasets.prepare import load_cifar_dataset, retrieval_items
from vision_language_benchmarks.embedding_dimensions import SUPPORTED_EMBEDDING_DIMENSIONS
from vision_language_benchmarks.evaluation.engine import _retrieval_candidates
from vision_language_benchmarks.paths import DATA_ROOT, PROJECT_ROOT
from vision_language_benchmarks.prompts import (
    classification_prompt_groups,
    classification_prompt_name,
    retrieval_prompt_name,
)
from vision_language_benchmarks.registry import get_model
from vision_language_benchmarks.reporting.mrl_equivalence import (
    compare_embedding_arrays,
    compare_native_mrl_batch,
    render_native_mrl_api_equivalence_markdown,
)

VALIDATION_DIMENSIONS = (768, 512, 256, 128)
IMAGE_SAMPLE_INDICES = (0, 1)
RETRIEVAL_SAMPLE_COUNT = 2
CLASSIFICATION_CLASS_COUNT = 2
MODEL_KEY = "embeddinggemma2"
MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
LIBRARY_VERSION = "0.4.0"
RTOL = 1e-5
ATOL = 1e-6

if os.environ.get("VLBENCH_RUN_MRL_INTEGRATION") != "1":
    pytest.skip(
        "Set VLBENCH_RUN_MRL_INTEGRATION=1 to verify native CUDA truncation.",
        allow_module_level=True,
    )


def _git_provenance() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, bool(status.strip())


def _dataset_manifest(dataset_key: str) -> dict[str, Any]:
    manifest_path = DATA_ROOT / dataset_key / ".vlbench-dataset.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _text_sample_record(
    texts: list[str],
    text_ids: list[str],
    *,
    task_prompt_name: str,
    prompt_policy: str | None = None,
    prompt_set_sha256: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "sample_count": len(texts),
        "text_ids_sha256": content_hash(text_ids),
        "text_content_sha256": content_hash(texts),
        "task_prompt_name": task_prompt_name,
    }
    if prompt_policy is not None:
        record["prompt_policy"] = prompt_policy
    if prompt_set_sha256 is not None:
        record["prompt_set_sha256"] = prompt_set_sha256
    return record


def _prepare_validation_samples() -> tuple[
    list[Image.Image],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    cifar = load_cifar_dataset("cifar10", DATA_ROOT)
    if len(cifar.classes) < CLASSIFICATION_CLASS_COUNT:
        raise ValueError("CIFAR-10 does not contain enough class labels for validation.")
    images: list[Image.Image] = []
    for index in IMAGE_SAMPLE_INDICES:
        image, _ = cifar[index]
        if not isinstance(image, Image.Image):
            raise TypeError(f"CIFAR-10 sample {index} is not a PIL image.")
        images.append(image.convert("RGB"))

    classification_task_prompt = classification_prompt_name(MODEL_KEY)
    retrieval_task_prompt = retrieval_prompt_name(MODEL_KEY)
    if classification_task_prompt is None or retrieval_task_prompt is None:
        raise ValueError("EmbeddingGemma task prompt names are not configured.")

    categories: dict[str, dict[str, Any]] = {
        "image": {
            "modality": "image",
            "dataset_ids": ["cifar10"],
            "texts": None,
            "task": None,
            "language": None,
        }
    }
    cifar_text_samples: dict[str, Any] = {}
    class_names = list(cifar.classes[:CLASSIFICATION_CLASS_COUNT])
    for protocol in ("p0", "p1"):
        prompt_groups, prompt_policy, prompt_set_sha256 = classification_prompt_groups(
            class_names,
            protocol,
            MODEL_KEY,
        )
        texts = [prompt for group in prompt_groups for prompt in group]
        text_ids = [
            f"{class_name}#prompt-{prompt_index}"
            for class_name, prompts in zip(class_names, prompt_groups, strict=True)
            for prompt_index in range(len(prompts))
        ]
        category_key = f"classification_{protocol}_en"
        categories[category_key] = {
            "modality": "classification_text",
            "dataset_ids": ["cifar10"],
            "texts": texts,
            "text_ids": text_ids,
            "task": "classification",
            "language": "en",
            "protocol": protocol,
            "task_prompt_name": classification_task_prompt,
            "prompt_policy": prompt_policy,
            "prompt_set_sha256": prompt_set_sha256,
            "prompt_content_sha256": content_hash(texts),
        }
        cifar_text_samples[category_key] = _text_sample_record(
            texts,
            text_ids,
            task_prompt_name=classification_task_prompt,
            prompt_policy=prompt_policy,
            prompt_set_sha256=prompt_set_sha256,
        )
        cifar_text_samples[category_key]["selected_class_names"] = class_names
        cifar_text_samples[category_key]["prompt_content_sha256"] = content_hash(texts)

    cifar_selection = {
        "selection_description": (
            "CIFAR-10 test images at indices [0, 1]; classification prompts use the "
            "first two class names and the existing P0/P1 prompt generators."
        ),
        "dataset_checksum": _dataset_manifest("cifar10")["dataset_checksum"],
        "image_sample_count": len(images),
        "image_sample_indices": list(IMAGE_SAMPLE_INDICES),
        "image_sample_ids_sha256": content_hash(
            [f"test-index:{index}" for index in IMAGE_SAMPLE_INDICES]
        ),
        "text_samples": cifar_text_samples,
    }

    xm_manifest = _dataset_manifest("crossmodal3600")
    xm_text_samples: dict[str, Any] = {}
    for language in ("en", "ja"):
        items = retrieval_items("crossmodal3600", language, DATA_ROOT)
        candidates = _retrieval_candidates(
            "crossmodal3600",
            items,
            "official_first_caption",
        )
        texts = candidates.text_values[:RETRIEVAL_SAMPLE_COUNT]
        text_ids = candidates.text_ids[:RETRIEVAL_SAMPLE_COUNT]
        if len(texts) != RETRIEVAL_SAMPLE_COUNT:
            raise ValueError(
                f"XM3600 {language} provides fewer than {RETRIEVAL_SAMPLE_COUNT} "
                "official-first-caption samples."
            )
        category_key = f"retrieval_{language}"
        categories[category_key] = {
            "modality": "retrieval_text",
            "dataset_ids": ["crossmodal3600"],
            "texts": texts,
            "text_ids": text_ids,
            "task": "retrieval",
            "language": language,
            "protocol": "p2",
            "task_prompt_name": retrieval_task_prompt,
            "prompt_content_sha256": content_hash(texts),
        }
        xm_text_samples[language] = _text_sample_record(
            texts,
            text_ids,
            task_prompt_name=retrieval_task_prompt,
        )

    xm_selection = {
        "selection_description": (
            "The first two candidate records in canonical annotation order, using "
            "`official_first_caption` (caption index 0)."
        ),
        "dataset_checksum": xm_manifest["dataset_checksum"],
        "text_samples": xm_text_samples,
    }
    return images, categories, {
        "sample_selection": {
            "cifar10": cifar_selection,
            "crossmodal3600": xm_selection,
        },
        "sample_counts": {
            "images": len(images),
            "classification_text": {
                key: len(category["texts"])
                for key, category in categories.items()
                if category["modality"] == "classification_text"
            },
            "retrieval_text": {
                key: len(category["texts"])
                for key, category in categories.items()
                if category["modality"] == "retrieval_text"
            },
        },
    }


def _measurement(
    category: dict[str, Any],
    dimension: int,
    native_result: dict[str, Any],
    adapter_embeddings: np.ndarray,
) -> dict[str, Any]:
    native_embeddings = native_result["native_embeddings"]
    shape_matches = adapter_embeddings.shape == native_embeddings.shape
    dtype_matches = adapter_embeddings.dtype == np.float32
    adapter_comparison = compare_embedding_arrays(
        adapter_embeddings,
        native_embeddings,
        rtol=RTOL,
        atol=ATOL,
    )
    adapter_comparison["passed"] = bool(
        adapter_comparison["passed"] and shape_matches and dtype_matches
    )
    measurement: dict[str, Any] = {
        "embedding_dimension": dimension,
        "modality": category["modality"],
        "dataset_ids": category["dataset_ids"],
        "sample_count": int(native_embeddings.shape[0]),
        "native_output_shape": list(native_embeddings.shape),
        "native_vs_reference": native_result["comparison"],
        "adapter_output": {
            "dtype": str(adapter_embeddings.dtype),
            "shape": list(adapter_embeddings.shape),
            "shape_matches_direct_native": shape_matches,
            "dtype_is_float32": dtype_matches,
            "vs_direct_native": adapter_comparison,
        },
        "task_prompt_name": category.get("task_prompt_name"),
        "prompt_content_sha256": category.get("prompt_content_sha256"),
        "prompt_policy": category.get("prompt_policy"),
        "prompt_set_sha256": category.get("prompt_set_sha256"),
        "passed": bool(
            native_result["comparison"]["passed"] and adapter_comparison["passed"]
        ),
    }
    for field in ("protocol", "language"):
        if category.get(field) is not None:
            measurement[field] = category[field]
    return measurement


@pytest.mark.real_model
def test_native_mrl_equivalence_is_measured_and_published() -> None:
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        pytest.skip("A CUDA-enabled PyTorch runtime and NVIDIA GPU are required.")
    if tuple(SUPPORTED_EMBEDDING_DIMENSIONS) != VALIDATION_DIMENSIONS:
        raise AssertionError(
            "Validation dimensions must match the supported native MRL dimensions "
            "in descending order."
        )

    model_spec = get_model(MODEL_KEY)
    if model_spec.model_id != MODEL_ID or model_spec.checkpoint_revision != MODEL_REVISION:
        raise AssertionError("The registered model does not match the pinned checkpoint.")
    timestamp_utc = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    source_commit_sha, working_tree_dirty = _git_provenance()
    images, categories, sample_provenance = _prepare_validation_samples()

    adapter = create_adapter(
        MODEL_KEY,
        "cuda",
        "fp32",
        embedding_dimension=VALIDATION_DIMENSIONS[0],
    )
    try:
        adapter_metadata = adapter.metadata()
        if adapter_metadata["implementation_version"] != LIBRARY_VERSION:
            raise AssertionError(
                f"Expected openclip-gemma=={LIBRARY_VERSION}, received "
                f"{adapter_metadata['implementation_version']}."
            )
        checkpoint = {
            "sha256": adapter.checkpoint_sha256,
            "files": adapter.checkpoint_files,
        }
        prepared_images = adapter.preprocess_images(images)
        image_batch = torch.stack(prepared_images).to(adapter.device_object)
        prepared_texts = {
            key: adapter.preprocess_texts(
                category["texts"],
                category["task"],
                category["language"],
            )
            for key, category in categories.items()
            if category["modality"] != "image"
        }

        native_results: dict[str, dict[int, dict[str, Any]]] = {
            "image": compare_native_mrl_batch(
                adapter.model,
                image_batch,
                "image",
                VALIDATION_DIMENSIONS,
                torch,
                rtol=RTOL,
                atol=ATOL,
            )
        }
        for key, tokens in prepared_texts.items():
            native_results[key] = compare_native_mrl_batch(
                adapter.model,
                tokens,
                "text",
                VALIDATION_DIMENSIONS,
                torch,
                rtol=RTOL,
                atol=ATOL,
            )
        initial_adapter_outputs = {"image": adapter.encode_images(prepared_images)}
        for key, category in categories.items():
            if category["modality"] == "image":
                continue
            initial_adapter_outputs[key] = adapter.encode_texts(
                prepared_texts[key],
                category["task"],
                category["language"],
            )
        adapter_outputs: dict[int, dict[str, np.ndarray]] = {
            VALIDATION_DIMENSIONS[0]: initial_adapter_outputs
        }
    finally:
        adapter.close()

    checkpoint_hash = checkpoint["sha256"]
    for dimension in VALIDATION_DIMENSIONS[1:]:
        adapter = create_adapter(
            MODEL_KEY,
            "cuda",
            "fp32",
            embedding_dimension=dimension,
        )
        try:
            if adapter.metadata()["implementation_version"] != LIBRARY_VERSION:
                raise AssertionError(
                    f"Expected openclip-gemma=={LIBRARY_VERSION} for {dimension}D."
                )
            if adapter.checkpoint_sha256 != checkpoint_hash:
                raise AssertionError("Checkpoint hash changed between adapter instances.")
            prepared_images = adapter.preprocess_images(images)
            dimension_outputs = {"image": adapter.encode_images(prepared_images)}
            for key, category in categories.items():
                if category["modality"] == "image":
                    continue
                tokens = adapter.preprocess_texts(
                    category["texts"],
                    category["task"],
                    category["language"],
                )
                dimension_outputs[key] = adapter.encode_texts(
                    tokens,
                    category["task"],
                    category["language"],
                )
            adapter_outputs[dimension] = dimension_outputs
        finally:
            adapter.close()

    measurements = [
        _measurement(
            categories[key],
            dimension,
            native_results[key][dimension],
            adapter_outputs[dimension][key],
        )
        for dimension in VALIDATION_DIMENSIONS
        for key in categories
    ]
    failures = [
        {
            "dimension": measurement["embedding_dimension"],
            "modality": measurement["modality"],
            "dataset_ids": measurement["dataset_ids"],
            "protocol": measurement.get("protocol"),
            "language": measurement.get("language"),
        }
        for measurement in measurements
        if not measurement["passed"]
    ]
    dataset_checksums = {
        dataset_id: selection["dataset_checksum"]
        for dataset_id, selection in sample_provenance["sample_selection"].items()
    }
    report = {
        "schema_version": "1.0.0",
        "validation_type": "native_mrl_api_equivalence",
        "status": "failed" if failures else "completed",
        "aggregate_validation_status": "failed" if failures else "passed",
        "overall_passed": not failures,
        "validation_source": "tests/real_model/test_embeddinggemma_mrl_equivalence.py",
        "source_commit_sha": source_commit_sha,
        "timestamp_utc": timestamp_utc,
        "working_tree_dirty": working_tree_dirty,
        "library": "openclip-gemma",
        "library_version": LIBRARY_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "device": "cuda",
        "precision": "fp32",
        "dimensions": list(VALIDATION_DIMENSIONS),
        "tolerance": {"rtol": RTOL, "atol": ATOL},
        "checkpoint": checkpoint,
        "dataset_identifiers": list(dataset_checksums),
        "dataset_checksums": dataset_checksums,
        "sample_counts": sample_provenance["sample_counts"],
        "sample_selection": sample_provenance["sample_selection"],
        "environment": {
            "python_version": platform.python_version(),
            "pytorch_version": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "gpu_name": torch.cuda.get_device_name(0),
        },
        "measurements": measurements,
        "failures": failures,
    }

    default_output_directory = (
        PROJECT_ROOT
        / "results"
        / "mrl-openclip-gemma-v0.4.0"
        / "validation"
    )
    output_directory = Path(
        os.environ.get(
            "VLBENCH_MRL_VALIDATION_OUTPUT_DIR",
            str(default_output_directory),
        )
    ).resolve()
    atomic_write_json(output_directory / "native-mrl-api-equivalence.json", report)
    with (output_directory / "native-mrl-api-equivalence.md").open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as stream:
        stream.write(render_native_mrl_api_equivalence_markdown(report))

    assert report["overall_passed"], (
        f"Native MRL equivalence failed for {len(failures)} measurement(s); "
        f"see {output_directory / 'native-mrl-api-equivalence.json'}."
    )
