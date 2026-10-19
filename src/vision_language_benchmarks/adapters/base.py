from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from vision_language_benchmarks.atomic import content_hash, sha256_file
from vision_language_benchmarks.errors import (
    DataIntegrityError,
    DependencyUnavailableError,
)
from vision_language_benchmarks.paths import CACHE_ROOT
from vision_language_benchmarks.registry import ModelSpec

TextTask = Literal["classification", "retrieval"]


class VisionLanguageAdapter(Protocol):
    spec: ModelSpec
    device: str
    precision: str
    torch: Any
    device_object: Any

    def preprocess_images(self, images: Sequence[Image.Image]) -> Any: ...

    def encode_images(self, batch: Any) -> NDArray[np.float32]: ...

    def preprocess_texts(
        self,
        texts: Sequence[str],
        task: TextTask,
        language: str,
    ) -> Any: ...

    def encode_texts(
        self,
        batch: Any,
        task: TextTask,
        language: str,
    ) -> NDArray[np.float32]: ...

    def metadata(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


class TorchAdapter:
    def __init__(self, spec: ModelSpec, device: str, precision: str) -> None:
        self.spec = spec
        self.device = device
        self.precision = precision
        try:
            self.torch = importlib.import_module("torch")
        except ImportError as error:
            raise DependencyUnavailableError(
                "Model inference requires PyTorch. Install the project 'cuda' extra."
            ) from error

        self.device_object = self.torch.device(device)
        if self.device_object.type == "cuda" and not self.torch.cuda.is_available():
            raise DependencyUnavailableError(
                "CUDA was requested, but torch.cuda.is_available() is false. "
                "CPU execution requires an explicit --device cpu selection."
            )
        if precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be 'fp32' or 'bf16'.")
        if precision == "bf16":
            if self.device_object.type != "cuda":
                raise ValueError("BF16 benchmark runs require a CUDA device.")
            if not spec.supports_bf16:
                raise ValueError(f"Model {spec.key!r} is not marked as BF16-capable.")
            if not self.torch.cuda.is_bf16_supported():
                raise DependencyUnavailableError("The selected CUDA device does not support BF16.")

    def inference_context(self) -> Any:
        return self.torch.inference_mode()

    def autocast_context(self) -> Any:
        if self.precision == "bf16":
            return self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16)
        return contextlib.nullcontext()

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
        if self.device_object.type == "cuda":
            self.torch.cuda.empty_cache()


def tensor_to_numpy(features: Any) -> NDArray[np.float32]:
    if not hasattr(features, "detach"):
        raise TypeError("An adapter must return a tensor-like feature matrix.")
    values = features.detach().float()
    array = values.cpu().numpy()
    if array.ndim != 2:
        raise DataIntegrityError(f"Model returned a {array.ndim}D feature array; expected 2D.")
    if not np.isfinite(array).all():
        raise DataIntegrityError("Model returned NaN or infinite feature values.")
    return np.asarray(array, dtype=np.float32)


def configure_huggingface_cache() -> Path:
    cache = CACHE_ROOT / "huggingface"
    os.environ.setdefault("HF_HOME", str(cache))
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def download_huggingface_snapshot(
    repository: str,
    revision: str,
    *,
    cache_dir: Path | None = None,
    allow_patterns: Sequence[str] | None = None,
) -> Path:
    try:
        hub = importlib.import_module("huggingface_hub")
    except ImportError as error:
        raise DependencyUnavailableError(
            "Hugging Face model loading requires huggingface-hub. Install a model extra."
        ) from error
    resolved_cache = cache_dir or configure_huggingface_cache()
    snapshot = hub.snapshot_download(
        repo_id=repository,
        revision=revision,
        cache_dir=str(resolved_cache),
        allow_patterns=list(
            allow_patterns
            or (
                "*.json",
                "*.merges",
                "*.model",
                "model*.safetensors",
                "*.tiktoken",
                "*.txt",
                "*.vocab",
            )
        ),
    )
    return Path(snapshot)


def checkpoint_fingerprint(
    root: Path,
    *,
    filenames: Sequence[str] | None = None,
) -> dict[str, Any]:
    supported_suffixes = {".bin", ".pt", ".pth", ".safetensors"}
    if filenames is None:
        weight_files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in supported_suffixes
            and path.name.lower().startswith(("model", "pytorch_model", "open_clip"))
        )
    else:
        relative_paths = [Path(filename) for filename in filenames]
        if any(path.is_absolute() or ".." in path.parts for path in relative_paths):
            raise ValueError("Checkpoint filenames must be relative to the snapshot root.")
        if len(set(relative_paths)) != len(relative_paths):
            raise ValueError("Checkpoint filenames must not contain duplicates.")
        if any(path.suffix.lower() not in supported_suffixes for path in relative_paths):
            raise ValueError("Checkpoint filenames must use a supported weight-file extension.")
        weight_files = sorted(root / path for path in relative_paths)
        missing = [path.relative_to(root).as_posix() for path in weight_files if not path.is_file()]
        if missing:
            raise DataIntegrityError(
                f"Expected checkpoint file(s) missing under {root}: {', '.join(missing)}."
            )
    if not weight_files:
        raise DataIntegrityError(f"No supported model checkpoint files found under {root}.")
    hashes = [(path.relative_to(root).as_posix(), sha256_file(path)) for path in weight_files]
    return {
        "checkpoint_files": [{"path": name, "sha256": digest} for name, digest in hashes],
        "checkpoint_sha256": content_hash(hashes),
    }


def parameter_count(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise DependencyUnavailableError(
            f"Package metadata for {distribution!r} is unavailable."
        ) from error


def move_to_device(value: Any, device: str) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, dict):
        return {
            key: tensor.to(device) if hasattr(tensor, "to") else tensor
            for key, tensor in value.items()
        }
    raise TypeError(f"Cannot move {type(value).__name__} to device {device!r}.")
