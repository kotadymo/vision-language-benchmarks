from __future__ import annotations

import importlib
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from vision_language_benchmarks.adapters.base import (
    TextTask,
    TorchAdapter,
    checkpoint_fingerprint,
    configure_huggingface_cache,
    download_huggingface_snapshot,
    package_version,
    parameter_count,
    tensor_to_numpy,
)
from vision_language_benchmarks.errors import DataIntegrityError, DependencyUnavailableError
from vision_language_benchmarks.registry import ModelSpec


class Adapter(TorchAdapter):
    def __init__(self, spec: ModelSpec, device: str, precision: str) -> None:
        super().__init__(spec, device, precision)
        cache_dir = configure_huggingface_cache()
        if spec.hf_repository is None or spec.checkpoint_revision is None:
            raise ValueError(f"Model {spec.key!r} has no pinned Hugging Face revision.")
        download_started = time.perf_counter()
        snapshot = download_huggingface_snapshot(
            spec.hf_repository,
            spec.checkpoint_revision,
            cache_dir=cache_dir,
        )
        model_download_seconds = time.perf_counter() - download_started
        fingerprint = checkpoint_fingerprint(snapshot)

        try:
            gemma = importlib.import_module("openclip_gemma")
        except ImportError as error:
            raise DependencyUnavailableError(
                "EmbeddingGemma inference requires openclip-gemma. "
                "Install the project 'gemma' extra."
            ) from error
        self.model, _, self.preprocess = gemma.create_model_and_transforms(
            model_name="embeddinggemma-2",
            pretrained=spec.pretrained_tag,
            precision="fp32",
            device=str(self.device_object),
            cache_dir=str(cache_dir),
        )
        self.gemma = gemma
        self.snapshot_path = snapshot
        self.checkpoint_sha256 = fingerprint["checkpoint_sha256"]
        self.checkpoint_files = fingerprint["checkpoint_files"]
        self.model_download_seconds = model_download_seconds
        self.cache_dir = cache_dir
        self.tokenizers: dict[str, Any] = {}
        self.image_shape_group_max_batch_size = 0
        self.model.eval()

    def preprocess_images(self, images: Sequence[Image.Image]) -> list[Any]:
        return [self.preprocess(image.convert("RGB")) for image in images]

    def reset_image_batch_stats(self) -> None:
        self.image_shape_group_max_batch_size = 0

    def encode_images(self, batch: Any) -> NDArray:
        if not isinstance(batch, list) or not batch:
            raise ValueError("EmbeddingGemma expects a non-empty list of preprocessed images.")

        groups: dict[tuple[int, ...], list[tuple[int, Any]]] = {}
        for index, pixels in enumerate(batch):
            shape = tuple(pixels.shape)
            if len(shape) != 3 or shape[0] != 3:
                raise DataIntegrityError(
                    "EmbeddingGemma preprocessing must return CHW image tensors; "
                    f"received shape {shape}."
                )
            groups.setdefault(shape, []).append((index, pixels))

        features_by_index: list[NDArray[np.float32] | None] = [None] * len(batch)
        for group in groups.values():
            pixel_batch = self.torch.stack([pixels for _, pixels in group]).to(
                self.device_object
            )
            with self.inference_context(), self.autocast_context():
                features = self.model.encode_image(pixel_batch, normalize=False)
            encoded = tensor_to_numpy(features)
            if encoded.shape[0] != len(group):
                raise DataIntegrityError(
                    "EmbeddingGemma returned "
                    f"{encoded.shape[0]} image features for {len(group)} images."
                )
            for offset, (index, _) in enumerate(group):
                features_by_index[index] = encoded[offset]
            self.image_shape_group_max_batch_size = max(
                self.image_shape_group_max_batch_size,
                len(group),
            )

        if any(features is None for features in features_by_index):
            raise DataIntegrityError("EmbeddingGemma did not return features for every image.")
        return np.stack([features for features in features_by_index if features is not None])

    def preprocess_texts(
        self,
        texts: Sequence[str],
        task: TextTask,
        language: str,
    ) -> Any:
        del language
        prompt_name = "Classification" if task == "classification" else "SearchQuery"
        tokenizer = self.tokenizers.get(prompt_name)
        if tokenizer is None:
            tokenizer = self.gemma.get_tokenizer(
                "embeddinggemma-2",
                prompt_name=prompt_name,
                cache_dir=str(self.cache_dir),
            )
            self.tokenizers[prompt_name] = tokenizer
        return tokenizer(list(texts)).to(self.device_object)

    def encode_texts(self, tokens: Any, task: TextTask, language: str) -> NDArray:
        del task, language
        with self.inference_context(), self.autocast_context():
            features = self.model.encode_text(tokens, normalize=False)
        return tensor_to_numpy(features)

    def metadata(self) -> dict[str, Any]:
        vision = getattr(self.model, "vision_model", None)
        config = getattr(vision, "config", None)
        text_config = getattr(getattr(self.model, "config", None), "text_config", None)
        return {
            "adapter_name": "openclip_gemma",
            "implementation_version": package_version("openclip-gemma"),
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_files": self.checkpoint_files,
            "model_download_seconds": self.model_download_seconds,
            "input_resolution": getattr(config, "image_size", None),
            "image_batching_strategy": "grouped_by_preprocessed_shape",
            "image_shape_group_max_batch_size": self.image_shape_group_max_batch_size,
            "max_text_length": getattr(text_config, "max_position_embeddings", None),
            "parameter_count": parameter_count(self.model),
            "tokenizer_repository": self.spec.hf_repository,
            "tokenizer_revision": self.spec.checkpoint_revision,
        }
