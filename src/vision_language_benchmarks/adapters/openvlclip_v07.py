from __future__ import annotations

import importlib
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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
from vision_language_benchmarks.atomic import content_hash, sha256_file
from vision_language_benchmarks.errors import DataIntegrityError
from vision_language_benchmarks.prompts import (
    QWEN3_VL_DEFAULT_INSTRUCTION,
    QWEN3_VL_RETRIEVAL_QUERY_INSTRUCTION,
)
from vision_language_benchmarks.registry import ModelSpec

_OPENVLCLIP_MODEL_NAMES = {
    "openvlclip_siglip2_base": "siglip2-base",
    "openvlclip_siglip2_naflex_base": "siglip2-naflex-base",
    "openvlclip_qwen3_vl_embedding_2b": "qwen3-vl-embedding-2b",
}
_QWEN_KEY = "openvlclip_qwen3_vl_embedding_2b"
_NAFLEX_KEY = "openvlclip_siglip2_naflex_base"
_TOKENIZER_FILES = {
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
}

RetrievalTextRole = Literal["query", "document"]


@dataclass(frozen=True, slots=True)
class _PreparedTextBatch:
    texts: tuple[str, ...]
    instruction: str
    role: str


def _plain_metadata(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_metadata(item) for item in value]
    if callable(getattr(value, "tolist", None)):
        return _plain_metadata(value.tolist())
    return str(value)


def _tokenizer_configuration_sha256(snapshot: Path) -> str:
    files = sorted(
        path
        for path in snapshot.rglob("*")
        if path.is_file() and path.name in _TOKENIZER_FILES
    )
    if not files:
        raise DataIntegrityError(
            f"No tokenizer or chat-template files were found in {snapshot}."
        )
    return content_hash(
        [
            (path.relative_to(snapshot).as_posix(), sha256_file(path))
            for path in files
        ]
    )


class Adapter(TorchAdapter):
    supports_directional_retrieval = True

    def __init__(self, spec: ModelSpec, device: str, precision: str) -> None:
        super().__init__(spec, device, precision)
        try:
            model_name = _OPENVLCLIP_MODEL_NAMES[spec.key]
        except KeyError as error:
            raise ValueError(f"Unsupported OpenVLCLIP v0.7.0 model: {spec.key!r}.") from error
        if spec.hf_repository is None or spec.checkpoint_revision is None:
            raise DataIntegrityError(f"Model {spec.key!r} has no pinned checkpoint revision.")
        if spec.global_embedding_dimension is None:
            raise DataIntegrityError(f"Model {spec.key!r} has no native embedding dimension.")

        self.openvlclip_version = package_version("openvlclip")
        if self.openvlclip_version != "0.7.0":
            raise DataIntegrityError(
                f"Expected openvlclip==0.7.0, found {self.openvlclip_version}."
            )
        self.transformers_version = package_version("transformers")
        self.cache_dir = configure_huggingface_cache()
        started = time.perf_counter()
        self.snapshot_path = download_huggingface_snapshot(
            spec.hf_repository,
            spec.checkpoint_revision,
            cache_dir=self.cache_dir,
        )
        fingerprint = checkpoint_fingerprint(self.snapshot_path)
        openvlclip = importlib.import_module("openvlclip")
        self.model, _, self.preprocess = openvlclip.create_model_and_transforms(
            model_name=model_name,
            pretrained=spec.pretrained_tag,
            precision=precision,
            device=str(self.device_object),
            cache_dir=str(self.cache_dir),
            logit_scale_mode="cosine",
        )
        if not callable(self.preprocess):
            raise DataIntegrityError("OpenVLCLIP did not return an image preprocessor.")

        embedding_space = getattr(self.model, "embedding_space_metadata", None)
        if (
            embedding_space is None
            or getattr(embedding_space, "model_id", None) != spec.model_id
            or getattr(embedding_space, "model_revision", None) != spec.checkpoint_revision
        ):
            raise DataIntegrityError(
                "OpenVLCLIP returned an embedding space that does not match the "
                "registered checkpoint identity."
            )
        dimension = self.model.validate_embedding_dimension(
            None,
            name="global embedding dimension",
        )
        if dimension != spec.global_embedding_dimension:
            raise DataIntegrityError(
                f"OpenVLCLIP returned {dimension}D features; "
                f"expected {spec.global_embedding_dimension}D."
            )

        self.model.to(self.device_object).eval()
        self.embedding_space_metadata = embedding_space
        self.embedding_dimension = dimension
        self.model_download_seconds = time.perf_counter() - started
        self.checkpoint_sha256 = fingerprint["checkpoint_sha256"]
        self.checkpoint_files = fingerprint["checkpoint_files"]
        self.model_checkpoint_size_bytes = sum(
            (self.snapshot_path / item["path"]).stat().st_size
            for item in self.checkpoint_files
        )
        self.tokenizer_config_sha256 = _tokenizer_configuration_sha256(
            self.snapshot_path
        )
        self.processor = getattr(self.model, "processor", None)
        if self.processor is None:
            raise DataIntegrityError("OpenVLCLIP did not expose its official processor.")
        self.image_processor = getattr(self.processor, "image_processor", None)
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.image_processor is None or self.tokenizer is None:
            raise DataIntegrityError(
                "The official processor must expose image and text processing."
            )
        self.preprocessing_identity = getattr(
            embedding_space,
            "preprocessing_id",
            None,
        )
        if not isinstance(self.preprocessing_identity, str) or not self.preprocessing_identity:
            raise DataIntegrityError("OpenVLCLIP did not identify its preprocessing contract.")

        self.max_num_patches: int | None = None
        self.image_shape_group_max_batch_size = 0
        if spec.key == _NAFLEX_KEY:
            configured_patch_count = getattr(self.image_processor, "max_num_patches", None)
            self.max_num_patches = (
                configured_patch_count
                if isinstance(configured_patch_count, int)
                and not isinstance(configured_patch_count, bool)
                else 256
            )
            if self.max_num_patches != 256:
                raise DataIntegrityError(
                    "The registered NaFlex checkpoint must use the documented "
                    "256-patch maximum."
                )

        self.preprocessing_config = self._preprocessing_config()
        self.tokenizer_max_length = self._tokenizer_max_length()
        self.supports_directional_retrieval = spec.key == _QWEN_KEY

    def _tokenizer_max_length(self) -> int | None:
        value = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(value, int) and not isinstance(value, bool) and 0 < value < 100_000:
            return value
        return None

    def _preprocessing_config(self) -> dict[str, Any]:
        image_fields = (
            "size",
            "crop_size",
            "do_resize",
            "resample",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "patch_size",
            "max_num_patches",
            "min_pixels",
            "max_pixels",
            "min_total_pixels",
            "max_total_pixels",
        )
        tokenizer = self.tokenizer
        tokenizer_config = {
            name: _plain_metadata(getattr(tokenizer, name, None))
            for name in (
                "vocab_size",
                "model_max_length",
                "padding_side",
                "truncation_side",
            )
        }
        chat_template = getattr(self.processor, "chat_template", None)
        if not isinstance(chat_template, str):
            chat_template = getattr(tokenizer, "chat_template", None)
        if isinstance(chat_template, str):
            tokenizer_config["chat_template_sha256"] = content_hash(chat_template)
        image_processor_config = {
            name: _plain_metadata(getattr(self.image_processor, name, None))
            for name in image_fields
        }
        if self.max_num_patches is not None:
            image_processor_config["max_num_patches"] = self.max_num_patches
        return {
            "openvlclip_model_name": _OPENVLCLIP_MODEL_NAMES[self.spec.key],
            "image_preprocessor_class": type(self.preprocess).__name__,
            "image_processor_class": type(self.image_processor).__name__,
            "image_processor": image_processor_config,
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer": tokenizer_config,
            "preprocessing_identity": self.preprocessing_identity,
            "image_batching_strategy": self._image_batching_strategy(),
            "max_num_patches": self.max_num_patches,
        }

    def _image_batching_strategy(self) -> str:
        if self.spec.key == _NAFLEX_KEY:
            return "grouped_by_processor_mapping_shape_preserving_order"
        if self.spec.key == _QWEN_KEY:
            return "official_processor_pil_sequence"
        return "official_fixres_processed_tensor_batch"

    def preprocess_images(self, images: Sequence[Image.Image]) -> list[Any]:
        if not images:
            raise ValueError("Cannot preprocess an empty image batch.")
        return [self.preprocess(image) for image in images]

    def _validate_features(
        self,
        features: Any,
        expected_count: int,
        modality: str,
    ) -> NDArray:
        values = tensor_to_numpy(features)
        if values.shape != (expected_count, self.embedding_dimension):
            raise DataIntegrityError(
                f"OpenVLCLIP returned {modality} features with shape {values.shape}; "
                f"expected ({expected_count}, {self.embedding_dimension})."
            )
        return values

    def encode_images(self, batch: Any) -> NDArray:
        if not isinstance(batch, Sequence) or isinstance(batch, (str, bytes)) or not batch:
            raise ValueError("OpenVLCLIP expects a non-empty sequence of preprocessed images.")
        if self.spec.key != _NAFLEX_KEY:
            with self.inference_context():
                features = self.model.encode_image(batch, normalize=False)
            return self._validate_features(features, len(batch), "image")

        groups: dict[tuple[Any, ...], list[tuple[int, Any]]] = {}
        for index, image in enumerate(batch):
            if not isinstance(image, Mapping) or set(image) != {
                "pixel_values",
                "pixel_attention_mask",
                "spatial_shapes",
            }:
                raise DataIntegrityError(
                    "NaFlex preprocessing must retain pixel_values, "
                    "pixel_attention_mask, and spatial_shapes."
                )
            shape_key: list[Any] = []
            for name in ("pixel_values", "pixel_attention_mask", "spatial_shapes"):
                value = image[name]
                if not isinstance(value, self.torch.Tensor):
                    raise DataIntegrityError(f"NaFlex {name} must be a PyTorch tensor.")
                shape_key.append((name, tuple(value.shape), value.dtype, value.device))
            groups.setdefault(tuple(shape_key), []).append((index, image))

        ordered_features: list[NDArray | None] = [None] * len(batch)
        self.image_shape_group_max_batch_size = 0
        for group in groups.values():
            inputs = [image for _, image in group]
            self.image_shape_group_max_batch_size = max(
                self.image_shape_group_max_batch_size,
                len(inputs),
            )
            with self.inference_context():
                features = self.model.encode_image(inputs, normalize=False)
            values = self._validate_features(features, len(inputs), "image")
            for offset, (original_index, _) in enumerate(group):
                ordered_features[original_index] = values[offset]

        if any(features is None for features in ordered_features):
            raise DataIntegrityError("NaFlex did not return features for every image.")
        return np.stack(
            [features for features in ordered_features if features is not None]
        )

    def reset_image_batch_stats(self) -> None:
        self.image_shape_group_max_batch_size = 0

    def image_batch_metadata(self, batch: Any) -> dict[str, Any]:
        if self.spec.key != _NAFLEX_KEY:
            return {}
        max_num_patches = self.max_num_patches
        if max_num_patches is None:
            raise DataIntegrityError("The NaFlex patch limit was not initialized.")
        valid_patch_counts: list[int] = []
        for image in batch:
            mask = image.get("pixel_attention_mask") if isinstance(image, Mapping) else None
            if not isinstance(mask, self.torch.Tensor) or mask.numel() == 0:
                raise DataIntegrityError("NaFlex returned an empty or invalid patch mask.")
            valid_count = int(mask.to(dtype=self.torch.bool).sum().item())
            if not 0 < valid_count <= max_num_patches:
                raise DataIntegrityError(
                    f"NaFlex returned {valid_count} valid patches; expected 1.."
                    f"{max_num_patches}."
                )
            valid_patch_counts.append(valid_count)
        return {"valid_patch_counts": valid_patch_counts}

    def _text_values(self, texts: Sequence[str]) -> tuple[str, ...]:
        if not texts:
            raise ValueError("Cannot preprocess an empty text batch.")
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("Text inputs must be strings.")
        return tuple(texts)

    def preprocess_texts(
        self,
        texts: Sequence[str],
        task: TextTask,
        language: str,
    ) -> Any:
        del language
        values = self._text_values(texts)
        if self.spec.key == _QWEN_KEY:
            role = "classification" if task == "classification" else "query"
            instruction = (
                QWEN3_VL_DEFAULT_INSTRUCTION
                if role == "classification"
                else QWEN3_VL_RETRIEVAL_QUERY_INSTRUCTION
            )
            return _PreparedTextBatch(values, instruction, role)
        return values

    def _encode_text_batch(self, batch: _PreparedTextBatch) -> NDArray:
        with self.inference_context():
            features = self.model.encode_texts(
                batch.texts,
                instruction=batch.instruction,
                normalize=False,
            )
        return self._validate_features(features, len(batch.texts), "text")

    def encode_texts(
        self,
        batch: Any,
        task: TextTask,
        language: str,
    ) -> NDArray:
        del task, language
        if self.spec.key == _QWEN_KEY:
            if not isinstance(batch, _PreparedTextBatch):
                raise TypeError("Qwen text batches must preserve their instruction and role.")
            return self._encode_text_batch(batch)
        if not isinstance(batch, Sequence) or isinstance(batch, (str, bytes)) or not batch:
            raise ValueError("OpenVLCLIP expects a non-empty sequence of text inputs.")
        with self.inference_context():
            features = self.model.encode_texts(batch, normalize=False)
        return self._validate_features(features, len(batch), "text")

    def preprocess_retrieval_texts(
        self,
        texts: Sequence[str],
        role: RetrievalTextRole,
        language: str,
    ) -> _PreparedTextBatch:
        del language
        if not self.supports_directional_retrieval:
            raise ValueError("Directional retrieval instructions are only used for Qwen3-VL.")
        instruction = (
            QWEN3_VL_RETRIEVAL_QUERY_INSTRUCTION
            if role == "query"
            else QWEN3_VL_DEFAULT_INSTRUCTION
        )
        return _PreparedTextBatch(self._text_values(texts), instruction, role)

    def encode_retrieval_texts(
        self,
        batch: _PreparedTextBatch,
        role: RetrievalTextRole,
        language: str,
    ) -> NDArray:
        del language
        if not self.supports_directional_retrieval:
            raise ValueError("Directional retrieval instructions are only used for Qwen3-VL.")
        if not isinstance(batch, _PreparedTextBatch) or batch.role != role:
            raise DataIntegrityError(
                f"Qwen retrieval features must be encoded with the {role!r} instruction."
            )
        return self._encode_text_batch(batch)

    def metadata(self) -> dict[str, Any]:
        metadata = {
            "adapter_name": f"openvlclip_v07_{self.spec.key.removeprefix('openvlclip_')}",
            "implementation_version": self.openvlclip_version,
            "preprocessing_version": self.openvlclip_version,
            "preprocessing_config": self.preprocessing_config,
            "preprocessing_identity": self.preprocessing_identity,
            "tokenizer_implementation_version": self.transformers_version,
            "tokenizer_config_sha256": self.tokenizer_config_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_files": self.checkpoint_files,
            "model_checkpoint_size_bytes": self.model_checkpoint_size_bytes,
            "model_download_seconds": self.model_download_seconds,
            "model_library": "openvlclip",
            "model_library_version": self.openvlclip_version,
            "global_embedding_dimension": self.embedding_dimension,
            "native_embedding_dimension": self.embedding_dimension,
            "embedding_generation_mode": "native_global_embeddings",
            "embedding_space_strategy": self.embedding_space_metadata.embedding_strategy,
            "embedding_normalization": self.embedding_space_metadata.normalization,
            "image_batching_strategy": self._image_batching_strategy(),
            "image_shape_group_max_batch_size": (
                self.image_shape_group_max_batch_size
                if self.spec.key == _NAFLEX_KEY
                else None
            ),
            "max_num_patches": self.max_num_patches,
            "input_resolution": 224 if self.spec.key == "openvlclip_siglip2_base" else None,
            "max_text_length": self.tokenizer_max_length,
            "parameter_count": parameter_count(self.model),
            "tokenizer_repository": self.spec.tokenizer_repository,
            "tokenizer_revision": self.spec.tokenizer_revision,
            "tokenizer_license": self.spec.tokenizer_license,
        }
        return metadata
