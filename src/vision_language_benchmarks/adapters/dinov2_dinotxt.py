from __future__ import annotations

import importlib
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from numpy.typing import NDArray
from PIL import Image

from vision_language_benchmarks.adapters.base import (
    TextTask,
    TorchAdapter,
    move_to_device,
    package_version,
    parameter_count,
    tensor_to_numpy,
)
from vision_language_benchmarks.dinotxt_provenance import (
    DINOTXT_MODEL_ID,
    DINOTXT_OPENVLCLIP_MODEL_NAME,
    DINOTXT_OPENVLCLIP_VERSION,
    checkpoint_fingerprint,
    verify_cached_components,
)
from vision_language_benchmarks.errors import (
    DataIntegrityError,
    DependencyUnavailableError,
    ModelAccessBlockedError,
)
from vision_language_benchmarks.registry import ModelSpec


class Adapter(TorchAdapter):
    def __init__(self, spec: ModelSpec, device: str, precision: str) -> None:
        if spec.weight_license_status != "verified":
            raise ModelAccessBlockedError(
                f"Loading {spec.name!r} is blocked until its model-weight license is "
                f"verified. {spec.weight_license_reason or ''}".strip()
            )
        if spec.model_id != DINOTXT_MODEL_ID:
            raise DataIntegrityError(
                f"Unexpected dino.txt model identifier: {spec.model_id!r}."
            )
        super().__init__(spec, device, precision)

        implementation_version = package_version("openvlclip")
        if implementation_version != DINOTXT_OPENVLCLIP_VERSION:
            raise DataIntegrityError(
                f"Expected openvlclip=={DINOTXT_OPENVLCLIP_VERSION}, "
                f"found {implementation_version}."
            )
        try:
            openvlclip = importlib.import_module("openvlclip")
        except ImportError as error:
            raise DependencyUnavailableError(
                "DINOv2+dino.txt inference requires openvlclip. "
                "Install the project 'dinotxt' extra."
            ) from error

        started = time.perf_counter()
        self.model, _, self.preprocess = openvlclip.create_model_and_transforms(
            model_name=DINOTXT_OPENVLCLIP_MODEL_NAME,
            pretrained="official",
            precision=precision,
            device=str(self.device_object),
            logit_scale_mode="cosine",
        )
        self.embedding_space_metadata = getattr(
            self.model,
            "embedding_space_metadata",
            None,
        )
        if (
            self.embedding_space_metadata is None
            or self.embedding_space_metadata.model_id != spec.model_id
            or self.embedding_space_metadata.model_revision != spec.checkpoint_revision
            or self.embedding_space_metadata.embedding_dimension
            != spec.global_embedding_dimension
        ):
            raise DataIntegrityError(
                "OpenVLCLIP returned an embedding space that does not match the "
                "registered dino.txt model identity."
            )
        self.model.to(self.device_object).eval()
        self.tokenizer = openvlclip.get_tokenizer(
            DINOTXT_OPENVLCLIP_MODEL_NAME,
            prompt_name=None,
        )
        self.model_download_seconds = time.perf_counter() - started
        checkpoint_directory = Path(self.torch.hub.get_dir()) / "checkpoints"
        self.checkpoint_components = verify_cached_components(checkpoint_directory)
        fingerprint = checkpoint_fingerprint(self.checkpoint_components)
        self.checkpoint_sha256 = fingerprint["checkpoint_sha256"]
        self.checkpoint_files = fingerprint["checkpoint_files"]
        self.implementation_version = implementation_version

    def preprocess_images(self, images: Sequence[Image.Image]) -> Any:
        return self.torch.stack(
            [self.preprocess(image.convert("RGB")) for image in images]
        ).to(self.device_object)

    def encode_images(self, batch: Any) -> NDArray:
        with self.inference_context(), self.autocast_context():
            features = self.model.encode_image(batch, normalize=False)
        values = tensor_to_numpy(features)
        self._validate_feature_shape(values, batch.shape[0], "image")
        return values

    def preprocess_texts(
        self,
        texts: Sequence[str],
        task: TextTask,
        language: str,
    ) -> Any:
        del task, language
        tokens = self.tokenizer(list(texts))
        return move_to_device(tokens, str(self.device_object))

    def encode_texts(
        self,
        tokens: Any,
        task: TextTask,
        language: str,
    ) -> NDArray:
        del task, language
        with self.inference_context(), self.autocast_context():
            features = self.model.encode_text(tokens, normalize=False)
        values = tensor_to_numpy(features)
        self._validate_feature_shape(values, len(tokens), "text")
        return values

    def _validate_feature_shape(
        self,
        features: NDArray,
        expected_count: int,
        modality: str,
    ) -> None:
        expected_dimension = self.spec.global_embedding_dimension
        if features.shape[0] != expected_count:
            raise DataIntegrityError(
                f"Model returned {features.shape[0]} {modality} embeddings for "
                f"{expected_count} inputs."
            )
        if expected_dimension is None or features.shape[1] != expected_dimension:
            raise DataIntegrityError(
                f"Model returned {features.shape[1]}D {modality} embeddings; "
                f"expected {expected_dimension}D."
            )

    def metadata(self) -> dict[str, Any]:
        preprocessing_config = {
            "transform": "openvlclip.create_model_and_transforms(pretrained='official')",
            "color_mode": "RGB",
            "resize_shorter": 256,
            "resize_interpolation": "bicubic",
            "crop": "center",
            "to_tensor_scale": "float32 in [0, 1]",
            "input_resolution": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "max_text_length": 77,
            "task_prompt": None,
        }
        return {
            "adapter_name": "openvlclip_dinotxt",
            "implementation_version": self.implementation_version,
            "model_library": "openvlclip",
            "model_library_version": self.implementation_version,
            "model_repository_revision": self.spec.repository_revision,
            "checkpoint_revision": self.spec.checkpoint_revision,
            "preprocessing_version": self.implementation_version,
            "preprocessing_config": preprocessing_config,
            "tokenizer_implementation_version": self.implementation_version,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_files": self.checkpoint_files,
            "checkpoint_components": self.checkpoint_components,
            "model_download_seconds": self.model_download_seconds,
            "global_embedding_dimension": self.spec.global_embedding_dimension,
            "embedding_generation_mode": "native_global_embeddings",
            "embedding_space_strategy": self.embedding_space_metadata.embedding_strategy,
            "embedding_normalization": self.embedding_space_metadata.normalization,
            "input_resolution": 224,
            "max_text_length": 77,
            "parameter_count": parameter_count(self.model),
            "tokenizer_repository": self.spec.tokenizer_repository,
            "tokenizer_revision": self.spec.tokenizer_revision,
            "tokenizer_license": self.spec.tokenizer_license,
        }
