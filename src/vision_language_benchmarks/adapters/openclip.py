from __future__ import annotations

import copy
import importlib
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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
from vision_language_benchmarks.atomic import sha256_file
from vision_language_benchmarks.errors import DataIntegrityError, DependencyUnavailableError
from vision_language_benchmarks.paths import CACHE_ROOT
from vision_language_benchmarks.registry import ModelSpec


class Adapter(TorchAdapter):
    def __init__(self, spec: ModelSpec, device: str, precision: str) -> None:
        super().__init__(spec, device, precision)
        configure_huggingface_cache()
        try:
            self.open_clip = importlib.import_module("open_clip")
        except ImportError as error:
            raise DependencyUnavailableError(
                "OpenCLIP inference requires open-clip-torch. "
                "Install the project 'openclip' extra."
            ) from error

        self.cache_dir = CACHE_ROOT / "open_clip"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        download_started = time.perf_counter()
        tokenizer_snapshot: Path | None = None
        if spec.hf_repository:
            if spec.checkpoint_revision is None or spec.checkpoint_filename is None:
                raise ValueError(
                    f"Model {spec.key!r} has no pinned checkpoint revision and filename."
                )
            if spec.pretrained_tag is None:
                raise ValueError(f"Model {spec.key!r} has no OpenCLIP pretrained tag.")
            if spec.uses_openclip_builtin_config:
                if spec.tokenizer_repository is None or spec.tokenizer_revision is None:
                    raise ValueError(
                        f"Model {spec.key!r} has no pinned Hugging Face tokenizer."
                    )
                snapshot = download_huggingface_snapshot(
                    spec.hf_repository,
                    spec.checkpoint_revision,
                    allow_patterns=(spec.checkpoint_filename,),
                )
                fingerprint = checkpoint_fingerprint(
                    snapshot,
                    filenames=(spec.checkpoint_filename,),
                )
                tokenizer_snapshot = download_huggingface_snapshot(
                    spec.tokenizer_repository,
                    spec.tokenizer_revision,
                    allow_patterns=(
                        "config.json",
                        "special_tokens_map.json",
                        "sentencepiece.bpe.model",
                        "tokenizer.json",
                        "tokenizer_config.json",
                    ),
                )
                model_config = self.open_clip.get_model_config(spec.model_id)
                if not isinstance(model_config, dict):
                    raise DataIntegrityError(
                        f"No built-in OpenCLIP configuration exists for {spec.model_id!r}."
                    )
                model_config = copy.deepcopy(model_config)
                text_config = model_config.get("text_cfg")
                if not isinstance(text_config, dict):
                    raise DataIntegrityError(
                        f"OpenCLIP text configuration is missing for {spec.model_id!r}."
                    )
                text_config.update(
                    {
                        "hf_model_name": str(tokenizer_snapshot),
                        "hf_model_pretrained": False,
                        "hf_tokenizer_name": str(tokenizer_snapshot),
                    }
                )
                pretrained_config = self.open_clip.get_pretrained_cfg(
                    spec.model_id,
                    spec.pretrained_tag,
                )
                if not isinstance(pretrained_config, dict):
                    raise DataIntegrityError(
                        f"OpenCLIP pretrained tag {spec.pretrained_tag!r} is unavailable "
                        f"for {spec.model_id!r}."
                    )
                image_mean = pretrained_config["mean"]
                image_std = pretrained_config["std"]
                image_interpolation = pretrained_config["interpolation"]
                image_resize_mode = pretrained_config["resize_mode"]
                self.preprocessing_config = _image_preprocessing_config(
                    image_mean,
                    image_std,
                    image_interpolation,
                    image_resize_mode,
                )
                checkpoint_path = snapshot / spec.checkpoint_filename
                self.model, _, self.preprocess = self.open_clip.create_model_and_transforms(
                    spec.model_id,
                    pretrained=str(checkpoint_path),
                    precision="fp32",
                    device=self.device_object,
                    cache_dir=str(self.cache_dir),
                    force_quick_gelu=bool(pretrained_config.get("quick_gelu", False)),
                    image_mean=image_mean,
                    image_std=image_std,
                    image_interpolation=image_interpolation,
                    image_resize_mode=image_resize_mode,
                    **model_config,
                )
            else:
                snapshot = download_huggingface_snapshot(
                    spec.hf_repository,
                    spec.checkpoint_revision,
                    allow_patterns=("open_clip_config.json", spec.checkpoint_filename),
                )
                fingerprint = checkpoint_fingerprint(
                    snapshot,
                    filenames=(spec.checkpoint_filename,),
                )
                config_path = snapshot / "open_clip_config.json"
                if not config_path.is_file():
                    raise DataIntegrityError(
                        f"Pinned OpenCLIP repository {spec.hf_repository!r} has no "
                        "open_clip_config.json."
                    )
                with config_path.open("r", encoding="utf-8") as stream:
                    repository_config = json.load(stream)
                model_config = repository_config.get("model_cfg")
                preprocess_config = repository_config.get("preprocess_cfg", {})
                if not isinstance(model_config, dict) or not isinstance(
                    preprocess_config,
                    dict,
                ):
                    raise DataIntegrityError(
                        f"Invalid OpenCLIP configuration in {config_path}."
                    )
                pretrained_config = self.open_clip.get_pretrained_cfg(
                    spec.model_id,
                    spec.pretrained_tag,
                )
                if not isinstance(pretrained_config, dict):
                    raise DataIntegrityError(
                        f"OpenCLIP pretrained tag {spec.pretrained_tag!r} is unavailable "
                        f"for {spec.model_id!r}."
                    )
                image_mean = preprocess_config.get("mean", pretrained_config["mean"])
                image_std = preprocess_config.get("std", pretrained_config["std"])
                image_interpolation = preprocess_config.get(
                    "interpolation",
                    pretrained_config["interpolation"],
                )
                image_resize_mode = preprocess_config.get(
                    "resize_mode",
                    pretrained_config["resize_mode"],
                )
                self.preprocessing_config = _image_preprocessing_config(
                    image_mean,
                    image_std,
                    image_interpolation,
                    image_resize_mode,
                )
                checkpoint_path = snapshot / spec.checkpoint_filename
                self.model, _, self.preprocess = self.open_clip.create_model_and_transforms(
                    spec.model_id,
                    pretrained=str(checkpoint_path),
                    precision="fp32",
                    device=self.device_object,
                    cache_dir=str(self.cache_dir),
                    force_quick_gelu=bool(
                        model_config.get(
                            "quick_gelu",
                            pretrained_config.get("quick_gelu", False),
                        )
                    ),
                    image_mean=image_mean,
                    image_std=image_std,
                    image_interpolation=image_interpolation,
                    image_resize_mode=image_resize_mode,
                    **model_config,
                )
            self.model_download_seconds = time.perf_counter() - download_started
            self.checkpoint_sha256 = fingerprint["checkpoint_sha256"]
            self.checkpoint_files = fingerprint["checkpoint_files"]
            self.snapshot_path = snapshot
            tokenizer_name = spec.model_id
        else:
            if spec.pretrained_tag is None:
                raise ValueError(f"Model {spec.key!r} has no OpenCLIP pretrained tag.")
            pretrained_config = self.open_clip.get_pretrained_cfg(
                spec.model_id,
                spec.pretrained_tag,
            )
            if not isinstance(pretrained_config, dict):
                raise DataIntegrityError(
                    f"OpenCLIP pretrained tag {spec.pretrained_tag!r} is unavailable "
                    f"for {spec.model_id!r}."
                )
            self.preprocessing_config = _image_preprocessing_config(
                pretrained_config["mean"],
                pretrained_config["std"],
                pretrained_config["interpolation"],
                pretrained_config["resize_mode"],
            )
            checkpoint_path = Path(
                self.open_clip.download_pretrained(
                    pretrained_config,
                    cache_dir=str(self.cache_dir),
                )
            )
            self.model_download_seconds = time.perf_counter() - download_started
            self.checkpoint_sha256 = sha256_file(checkpoint_path)
            self.checkpoint_files = [
                {"path": checkpoint_path.name, "sha256": self.checkpoint_sha256}
            ]
            self.snapshot_path = None
            self.model, _, self.preprocess = self.open_clip.create_model_and_transforms(
                spec.model_id,
                pretrained=spec.pretrained_tag,
                precision="fp32",
                device=self.device_object,
                cache_dir=str(self.cache_dir),
                force_quick_gelu=bool(pretrained_config.get("quick_gelu", False)),
            )
            tokenizer_name = spec.model_id

        if tokenizer_snapshot is None:
            self.tokenizer = self.open_clip.get_tokenizer(
                tokenizer_name,
                cache_dir=str(self.cache_dir),
            )
        else:
            tokenizer_class = importlib.import_module("open_clip.tokenizer").HFTokenizer
            self.tokenizer = tokenizer_class(
                str(tokenizer_snapshot),
                context_length=int(self.model.context_length),
                cache_dir=str(self.cache_dir),
                local_files_only=True,
            )
        self.model.eval()

    def preprocess_images(self, images: Sequence[Image.Image]) -> Any:
        return self.torch.stack(
            [self.preprocess(image.convert("RGB")) for image in images]
        ).to(self.device_object)

    def encode_images(self, batch: Any) -> NDArray:
        with self.inference_context(), self.autocast_context():
            features = self.model.encode_image(batch, normalize=False)
        return tensor_to_numpy(features)

    def preprocess_texts(
        self,
        texts: Sequence[str],
        task: TextTask,
        language: str,
    ) -> Any:
        del task, language
        return self.tokenizer(list(texts)).to(self.device_object)

    def encode_texts(self, tokens: Any, task: TextTask, language: str) -> NDArray:
        del task, language
        with self.inference_context(), self.autocast_context():
            features = self.model.encode_text(tokens, normalize=False)
        return tensor_to_numpy(features)

    def metadata(self) -> dict[str, Any]:
        visual = getattr(self.model, "visual", None)
        image_size = getattr(visual, "image_size", None)
        if hasattr(image_size, "tolist"):
            image_size = image_size.tolist()
        if isinstance(image_size, tuple):
            image_size = list(image_size)
        return {
            "adapter_name": "open_clip",
            "implementation_version": package_version("open-clip-torch"),
            "preprocessing_version": package_version("open-clip-torch"),
            "preprocessing_config": {
                "transform": "open_clip.create_model_and_transforms",
                "input_resolution": image_size,
                **self.preprocessing_config,
            },
            "tokenizer_implementation_version": (
                package_version("transformers")
                if self.spec.tokenizer_repository
                else package_version("open-clip-torch")
            ),
            "architecture": self.spec.architecture or self.spec.model_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_files": self.checkpoint_files,
            "model_download_seconds": self.model_download_seconds,
            "input_resolution": image_size,
            "max_text_length": int(getattr(self.model, "context_length", 0)) or None,
            "parameter_count": parameter_count(self.model),
            "tokenizer_repository": (
                self.spec.tokenizer_repository or f"open_clip:{self.spec.model_id}"
            ),
            "tokenizer_revision": self.spec.tokenizer_revision,
            "tokenizer_license": self.spec.tokenizer_license,
        }


def _image_preprocessing_config(
    mean: Sequence[float],
    std: Sequence[float],
    interpolation: Any,
    resize_mode: Any,
) -> dict[str, Any]:
    return {
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "interpolation": getattr(interpolation, "name", str(interpolation)),
        "resize_mode": str(resize_mode),
    }
