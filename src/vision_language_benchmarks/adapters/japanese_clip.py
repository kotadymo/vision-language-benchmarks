from __future__ import annotations

import importlib
import threading
import time
from collections.abc import Sequence
from typing import Any
from urllib.parse import unquote, urlparse

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
from vision_language_benchmarks.errors import DependencyUnavailableError
from vision_language_benchmarks.registry import ModelSpec

_IMPORT_LOCK = threading.RLock()


def _import_japanese_clip() -> Any:
    try:
        hub = importlib.import_module("huggingface_hub")
        modeling_utils = importlib.import_module("transformers.modeling_utils")
        transformers_utils = importlib.import_module("transformers.utils")
    except ImportError as error:
        raise DependencyUnavailableError(
            "Japanese CLIP inference requires huggingface-hub and Transformers. "
            "Install the project 'japanese-clip' extra."
        ) from error

    with _IMPORT_LOCK:
        patched: list[tuple[Any, str]] = []
        # The pinned upstream adapter imports APIs removed by current Transformers/Hub versions.
        if not hasattr(modeling_utils, "ModelOutput"):
            modeling_utils.ModelOutput = transformers_utils.ModelOutput
            patched.append((modeling_utils, "ModelOutput"))
        if not hasattr(hub, "cached_download"):
            def cached_download_compat(url: str, cache_dir: str | None = None, **_: Any) -> str:
                parts = unquote(urlparse(url).path).strip("/").split("/")
                try:
                    resolve_index = parts.index("resolve")
                except ValueError as error:
                    raise ValueError(f"Unsupported legacy Hugging Face URL: {url!r}.") from error
                if resolve_index < 2 or len(parts) <= resolve_index + 2:
                    raise ValueError(f"Malformed legacy Hugging Face URL: {url!r}.")
                return hub.hf_hub_download(
                    repo_id="/".join(parts[:resolve_index]),
                    filename="/".join(parts[resolve_index + 2 :]),
                    revision=parts[resolve_index + 1],
                    cache_dir=cache_dir,
                )

            hub.cached_download = cached_download_compat
            patched.append((hub, "cached_download"))
        try:
            return importlib.import_module("japanese_clip")
        finally:
            for module, attribute in reversed(patched):
                delattr(module, attribute)


class Adapter(TorchAdapter):
    def __init__(self, spec: ModelSpec, device: str, precision: str) -> None:
        super().__init__(spec, device, precision)
        cache_dir = configure_huggingface_cache()
        if (
            spec.hf_repository is None
            or spec.checkpoint_revision is None
            or spec.tokenizer_repository is None
            or spec.tokenizer_revision is None
        ):
            raise ValueError(f"Model {spec.key!r} has incomplete model/tokenizer revisions.")
        download_started = time.perf_counter()
        snapshot = download_huggingface_snapshot(
            spec.hf_repository,
            spec.checkpoint_revision,
            cache_dir=cache_dir,
        )
        tokenizer_snapshot = download_huggingface_snapshot(
            spec.tokenizer_repository,
            spec.tokenizer_revision,
            cache_dir=cache_dir,
            allow_patterns=(
                "config.json",
                "special_tokens_map.json",
                "spiece.model",
                "tokenizer_config.json",
            ),
        )
        model_download_seconds = time.perf_counter() - download_started
        fingerprint = checkpoint_fingerprint(snapshot)
        self.ja_clip = _import_japanese_clip()
        self.model, self.preprocess = self.ja_clip.load(
            str(snapshot),
            device=str(self.device_object),
            cache_dir=str(cache_dir),
        )
        transformers = importlib.import_module("transformers")
        self.tokenizer = transformers.T5Tokenizer.from_pretrained(str(tokenizer_snapshot))
        self.tokenizer.do_lower_case = True
        self.model.eval()
        self.snapshot_path = snapshot
        self.tokenizer_snapshot_path = tokenizer_snapshot
        self.checkpoint_sha256 = fingerprint["checkpoint_sha256"]
        self.checkpoint_files = fingerprint["checkpoint_files"]
        self.model_download_seconds = model_download_seconds

    def preprocess_images(self, images: Sequence[Image.Image]) -> Any:
        return self.torch.stack(
            [self.preprocess(image.convert("RGB")) for image in images]
        ).to(self.device_object)

    def encode_images(self, batch: Any) -> NDArray:
        with self.inference_context(), self.autocast_context():
            features = self.model.get_image_features(pixel_values=batch)
        return tensor_to_numpy(features)

    def preprocess_texts(
        self,
        texts: Sequence[str],
        task: TextTask,
        language: str,
    ) -> Any:
        del task, language
        return self.ja_clip.tokenize(
            list(texts),
            tokenizer=self.tokenizer,
            max_seq_len=77,
            device=str(self.device_object),
        )

    def encode_texts(
        self,
        tokens: Any,
        task: TextTask,
        language: str,
    ) -> NDArray:
        del task, language
        with self.inference_context(), self.autocast_context():
            features = self.model.get_text_features(**tokens)
        return tensor_to_numpy(features)

    def metadata(self) -> dict[str, Any]:
        vision = getattr(getattr(self.model, "config", None), "vision_config", None)
        return {
            "adapter_name": "japanese_clip",
            "implementation_version": package_version("japanese-clip"),
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_files": self.checkpoint_files,
            "model_download_seconds": self.model_download_seconds,
            "input_resolution": getattr(vision, "image_size", None),
            "max_text_length": 77,
            "parameter_count": parameter_count(self.model),
            "tokenizer_repository": self.spec.tokenizer_repository,
            "tokenizer_revision": self.spec.tokenizer_revision,
            "tokenizer_license": self.spec.tokenizer_license,
        }
