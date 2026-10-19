from __future__ import annotations

import importlib
import time
from collections.abc import Sequence
from typing import Any

from numpy.typing import NDArray
from PIL import Image

from vision_language_benchmarks.adapters.base import (
    TextTask,
    TorchAdapter,
    checkpoint_fingerprint,
    configure_huggingface_cache,
    download_huggingface_snapshot,
    move_to_device,
    package_version,
    parameter_count,
    tensor_to_numpy,
)
from vision_language_benchmarks.errors import DependencyUnavailableError
from vision_language_benchmarks.registry import ModelSpec


def _pooler_features(output: Any) -> Any:
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if isinstance(output, tuple) and output:
        return output[0]
    return output


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
            transformers = importlib.import_module("transformers")
        except ImportError as error:
            raise DependencyUnavailableError(
                "SigLIP 2 inference requires Transformers. "
                "Install the project 'siglip2' extra."
            ) from error
        self.model = transformers.AutoModel.from_pretrained(str(snapshot))
        self.processor = transformers.AutoProcessor.from_pretrained(str(snapshot))
        self.model.to(self.device_object).eval()
        self.snapshot_path = snapshot
        self.checkpoint_sha256 = fingerprint["checkpoint_sha256"]
        self.checkpoint_files = fingerprint["checkpoint_files"]
        self.model_download_seconds = model_download_seconds
        self.max_text_length = int(self.model.config.text_config.max_position_embeddings)

    def preprocess_images(self, images: Sequence[Image.Image]) -> Any:
        inputs = self.processor(
            images=[image.convert("RGB") for image in images],
            return_tensors="pt",
        )
        return move_to_device(inputs, str(self.device_object))

    def encode_images(self, inputs: Any) -> NDArray:
        with self.inference_context(), self.autocast_context():
            features = self.model.get_image_features(**inputs, return_dict=True)
        return tensor_to_numpy(_pooler_features(features))

    def preprocess_texts(
        self,
        texts: Sequence[str],
        task: TextTask,
        language: str,
    ) -> Any:
        del task, language
        inputs = self.processor(
            text=list(texts),
            padding="max_length",
            max_length=self.max_text_length,
            truncation=True,
            return_tensors="pt",
        )
        return move_to_device(inputs, str(self.device_object))

    def encode_texts(
        self,
        inputs: Any,
        task: TextTask,
        language: str,
    ) -> NDArray:
        del task, language
        with self.inference_context(), self.autocast_context():
            features = self.model.get_text_features(**inputs, return_dict=True)
        return tensor_to_numpy(_pooler_features(features))

    def metadata(self) -> dict[str, Any]:
        image_processor = getattr(self.processor, "image_processor", None)
        input_resolution = getattr(image_processor, "size", None)
        return {
            "adapter_name": "transformers_siglip2",
            "implementation_version": package_version("transformers"),
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_files": self.checkpoint_files,
            "model_download_seconds": self.model_download_seconds,
            "input_resolution": input_resolution,
            "max_text_length": self.max_text_length,
            "parameter_count": parameter_count(self.model),
            "tokenizer_repository": self.spec.hf_repository,
            "tokenizer_revision": self.spec.checkpoint_revision,
        }
