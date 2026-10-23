from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from vision_language_benchmarks.adapters import japanese_clip, openclip_gemma, siglip2
from vision_language_benchmarks.atomic import content_hash
from vision_language_benchmarks.registry import ModelSpec, get_model


class FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(max_position_embeddings=64),
            vision_config=SimpleNamespace(image_size=224),
        )
        self.vision_model = SimpleNamespace(config=SimpleNamespace(image_size=224))
        self.evaluated = False
        self.device: object | None = None

    def eval(self) -> FakeModel:
        self.evaluated = True
        return self

    def to(self, device: object) -> FakeModel:
        self.device = device
        return self

    def parameters(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(numel=lambda: 128)]


class FakeTokens:
    def __init__(self) -> None:
        self.device: object | None = None

    def to(self, device: object) -> FakeTokens:
        self.device = device
        return self


class FakeDevice:
    def __init__(self, device: str) -> None:
        self.type = device.split(":", maxsplit=1)[0]

    def __str__(self) -> str:
        return self.type


def patch_torch_adapter(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
) -> None:
    def initialize(
        adapter: Any,
        spec: ModelSpec,
        device: str,
        precision: str,
    ) -> None:
        adapter.spec = spec
        adapter.device = device
        adapter.precision = precision
        adapter.device_object = FakeDevice(device)
        adapter.torch = SimpleNamespace()

    monkeypatch.setattr(module.TorchAdapter, "__init__", initialize)


def prepare_snapshot(path: Path, filename: str = "model.safetensors") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / filename).write_bytes(b"synthetic checkpoint")
    return path


def test_embeddinggemma_adapter_uses_pinned_factory_and_task_tokenizers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = get_model("embeddinggemma2")
    snapshot = prepare_snapshot(tmp_path / "snapshot")
    cache_dir = tmp_path / "hf-cache"
    model = FakeModel()
    factory_calls: list[dict[str, Any]] = []
    tokenizer_calls: list[tuple[str, str, str]] = []

    class FakeGemmaTokenizer:
        def __init__(self, prompt_name: str) -> None:
            self.prompt_name = prompt_name
            self.tokens = FakeTokens()

        def __call__(self, texts: list[str]) -> FakeTokens:
            assert texts == ["a test prompt"]
            return self.tokens

    def create_model_and_transforms(**kwargs: Any) -> tuple[FakeModel, None, object]:
        factory_calls.append(kwargs)
        return model, None, object()

    def get_tokenizer(
        model_name: str,
        *,
        prompt_name: str,
        cache_dir: str,
    ) -> FakeGemmaTokenizer:
        tokenizer_calls.append((model_name, prompt_name, cache_dir))
        return FakeGemmaTokenizer(prompt_name)

    fake_gemma = SimpleNamespace(
        create_model_and_transforms=create_model_and_transforms,
        get_tokenizer=get_tokenizer,
    )

    def fake_import_module(name: str) -> object:
        if name == "openclip_gemma":
            return fake_gemma
        raise AssertionError(f"Unexpected dependency import: {name}")

    def fake_package_version(name: str) -> str:
        versions = {
            "openclip-gemma": "0.3.0",
            "transformers": "4.45.0",
        }
        assert name in versions
        return versions[name]

    def download_snapshot(
        repository: str,
        revision: str,
        *,
        cache_dir: Path | None = None,
        allow_patterns: object = None,
    ) -> Path:
        assert repository == spec.hf_repository
        assert revision == spec.checkpoint_revision
        assert cache_dir == cache_dir_expected
        assert allow_patterns is None
        return snapshot

    cache_dir_expected = cache_dir
    monkeypatch.setattr(
        openclip_gemma,
        "importlib",
        SimpleNamespace(import_module=fake_import_module),
    )
    monkeypatch.setattr(
        openclip_gemma,
        "configure_huggingface_cache",
        lambda: cache_dir,
    )
    monkeypatch.setattr(
        openclip_gemma,
        "download_huggingface_snapshot",
        download_snapshot,
    )
    monkeypatch.setattr(openclip_gemma, "package_version", fake_package_version)
    patch_torch_adapter(monkeypatch, openclip_gemma)

    adapter = openclip_gemma.Adapter(spec, "cpu", "fp32")
    first_tokens = adapter.preprocess_texts(["a test prompt"], "classification", "en")
    cached_tokens = adapter.preprocess_texts(["a test prompt"], "classification", "ja")
    search_tokens = adapter.preprocess_texts(["a test prompt"], "retrieval", "ja")
    metadata = adapter.metadata()

    assert model.evaluated
    assert factory_calls == [
        {
            "model_name": "embeddinggemma-2",
            "pretrained": "official",
            "precision": "fp32",
            "device": "cpu",
            "cache_dir": str(cache_dir),
        }
    ]
    assert first_tokens is cached_tokens
    assert first_tokens.device == adapter.device_object
    assert search_tokens.device == adapter.device_object
    assert [call[1] for call in tokenizer_calls] == ["Classification", "SearchQuery"]
    assert metadata["implementation_version"] == "0.3.0"
    assert metadata["tokenizer_implementation_version"] == "4.45.0"
    assert metadata["tokenizer_repository"] == spec.hf_repository
    assert metadata["tokenizer_revision"] == spec.checkpoint_revision


def test_embeddinggemma_batches_images_by_preprocessed_shape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = get_model("embeddinggemma2")
    snapshot = prepare_snapshot(tmp_path / "snapshot")
    cache_dir = tmp_path / "hf-cache"

    class FakePixelTensor:
        def __init__(self, shape: tuple[int, int, int], value: int) -> None:
            self.shape = shape
            self.value = value

        def to(self, device: object) -> FakePixelTensor:
            del device
            return self

    class FakeImageBatch:
        def __init__(self, images: list[FakePixelTensor]) -> None:
            self.images = images
            self.shape = (len(images), *images[0].shape)

        def to(self, device: object) -> FakeImageBatch:
            del device
            return self

    class FakeFeatures:
        def __init__(self, values: np.ndarray) -> None:
            self.values = values

        def detach(self) -> FakeFeatures:
            return self

        def float(self) -> FakeFeatures:
            return self

        def cpu(self) -> FakeFeatures:
            return self

        def numpy(self) -> np.ndarray:
            return self.values

    class GroupedFakeModel(FakeModel):
        def __init__(self) -> None:
            super().__init__()
            self.image_batch_shapes: list[tuple[int, ...]] = []

        def encode_image(
            self,
            batch: FakeImageBatch,
            *,
            normalize: bool,
            truncate_dim: int,
        ) -> FakeFeatures:
            assert not normalize
            self.image_batch_shapes.append(batch.shape)
            values = np.zeros((len(batch.images), truncate_dim), dtype=np.float32)
            values[:, 0] = [image.value for image in batch.images]
            values[:, 1] = 1.0
            values /= np.linalg.norm(values, axis=1, keepdims=True)
            return FakeFeatures(values)

    model = GroupedFakeModel()

    def preprocess(image: Image.Image) -> FakePixelTensor:
        width, height = image.size
        return FakePixelTensor(
            (3, height, width),
            image.getpixel((0, 0))[0],
        )

    def stack(images: list[FakePixelTensor]) -> FakeImageBatch:
        return FakeImageBatch(images)

    def create_model_and_transforms(**kwargs: Any) -> tuple[GroupedFakeModel, None, Any]:
        assert kwargs == {
            "model_name": "embeddinggemma-2",
            "pretrained": "official",
            "precision": "fp32",
            "device": "cpu",
            "cache_dir": str(cache_dir),
        }
        return model, None, preprocess

    def download_snapshot(
        repository: str,
        revision: str,
        *,
        cache_dir: Path | None = None,
        allow_patterns: object = None,
    ) -> Path:
        assert repository == spec.hf_repository
        assert revision == spec.checkpoint_revision
        assert cache_dir == cache_dir_expected
        assert allow_patterns is None
        return snapshot

    fake_gemma = SimpleNamespace(create_model_and_transforms=create_model_and_transforms)

    def import_module(name: str) -> object:
        assert name == "openclip_gemma"
        return fake_gemma

    def package_version(name: str) -> str:
        versions = {
            "openclip-gemma": "0.4.0",
            "transformers": "4.45.0",
        }
        assert name in versions
        return versions[name]

    monkeypatch.setattr(
        openclip_gemma,
        "importlib",
        SimpleNamespace(import_module=import_module),
    )
    monkeypatch.setattr(
        openclip_gemma,
        "configure_huggingface_cache",
        lambda: cache_dir,
    )
    monkeypatch.setattr(openclip_gemma, "download_huggingface_snapshot", download_snapshot)
    monkeypatch.setattr(openclip_gemma, "package_version", package_version)
    patch_torch_adapter(monkeypatch, openclip_gemma)
    cache_dir_expected = cache_dir

    adapter = openclip_gemma.Adapter(spec, "cpu", "fp32")
    adapter.torch.stack = stack
    adapter.inference_context = nullcontext
    adapter.autocast_context = nullcontext
    images = [
        Image.new("RGB", (4, 2), (11, 0, 0)),
        Image.new("RGB", (4, 2), (22, 0, 0)),
        Image.new("RGB", (2, 4), (33, 0, 0)),
    ]

    prepared = adapter.preprocess_images(images)
    features = adapter.encode_images(prepared)

    assert model.image_batch_shapes == [(2, 3, 2, 4), (1, 3, 4, 2)]
    expected_first_components = [
        value / np.sqrt(value**2 + 1.0)
        for value in (11.0, 22.0, 33.0)
    ]
    assert np.allclose(features[:, 0], expected_first_components)
    assert features.shape == (3, 768)
    assert adapter.image_shape_group_max_batch_size == 2
    assert adapter.metadata()["image_shape_group_max_batch_size"] == 2


def test_siglip_adapter_loads_pinned_local_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = get_model("siglip2_base_224")
    snapshot = prepare_snapshot(tmp_path / "snapshot")
    model = FakeModel()
    model_paths: list[str] = []
    processor_paths: list[str] = []
    text_calls: list[dict[str, Any]] = []

    class FakeProcessor:
        def __init__(self) -> None:
            self.image_processor = SimpleNamespace(size=SimpleNamespace(height=224, width=224))

        def __call__(self, **kwargs: Any) -> dict[str, torch.Tensor]:
            text_calls.append(kwargs)
            texts = kwargs["text"]
            max_length = kwargs["max_length"]
            assert isinstance(texts, list)
            assert isinstance(max_length, int)
            input_ids = torch.zeros((len(texts), max_length), dtype=torch.long)
            input_ids[:, 0] = 1
            return {"input_ids": input_ids}

    processor = FakeProcessor()

    def load_model(path: str) -> FakeModel:
        model_paths.append(path)
        return model

    def load_processor(path: str) -> FakeProcessor:
        processor_paths.append(path)
        return processor

    def fake_import_module(name: str) -> object:
        if name == "transformers":
            return SimpleNamespace(
                AutoModel=SimpleNamespace(from_pretrained=load_model),
                AutoProcessor=SimpleNamespace(from_pretrained=load_processor),
            )
        raise AssertionError(f"Unexpected dependency import: {name}")

    def download_snapshot(repository: str, revision: str, *, cache_dir: Path) -> Path:
        assert repository == spec.hf_repository
        assert revision == spec.checkpoint_revision
        assert cache_dir == tmp_path / "hf-cache"
        return snapshot

    def fake_package_version(name: str) -> str:
        assert name == "transformers"
        return "5.20.1"

    monkeypatch.setattr(
        siglip2,
        "importlib",
        SimpleNamespace(import_module=fake_import_module),
    )
    monkeypatch.setattr(
        siglip2,
        "configure_huggingface_cache",
        lambda: tmp_path / "hf-cache",
    )
    monkeypatch.setattr(siglip2, "download_huggingface_snapshot", download_snapshot)
    monkeypatch.setattr(siglip2, "package_version", fake_package_version)
    patch_torch_adapter(monkeypatch, siglip2)

    adapter = siglip2.Adapter(spec, "cpu", "fp32")
    adapter.torch.Tensor = torch.Tensor
    metadata = adapter.metadata()
    text_inputs = adapter.preprocess_texts(
        ["a test prompt"],
        "classification",
        "en",
    )

    assert model.evaluated
    assert model_paths == [str(snapshot)]
    assert processor_paths == [str(snapshot)]
    assert adapter.max_text_length == 64
    assert metadata["input_resolution"] == {"height": 224, "width": 224}
    assert content_hash(metadata["preprocessing_config"])
    assert metadata["tokenizer_repository"] == spec.hf_repository
    assert metadata["tokenizer_revision"] == spec.checkpoint_revision
    assert text_calls == [
        {
            "text": ["a test prompt"],
            "padding": "max_length",
            "max_length": 64,
            "truncation": True,
            "return_tensors": "pt",
        }
    ]
    assert set(text_inputs) == {"input_ids"}
    assert text_inputs["input_ids"].shape == (1, 64)


def test_japanese_clip_adapter_uses_pinned_model_and_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = get_model("rinna_ja_clip_b16")
    model_snapshot = prepare_snapshot(tmp_path / "model")
    tokenizer_snapshot = tmp_path / "tokenizer"
    tokenizer_snapshot.mkdir()
    model = FakeModel()
    downloaded: list[tuple[str, str, object]] = []
    tokenizer_paths: list[str] = []
    tokenization_calls: list[dict[str, Any]] = []

    class FakeT5Tokenizer:
        do_lower_case = False

        @classmethod
        def from_pretrained(cls, path: str) -> FakeT5Tokenizer:
            tokenizer_paths.append(path)
            return cls()

    def load_model(path: str, *, device: str, cache_dir: str) -> tuple[FakeModel, object]:
        assert path == str(model_snapshot)
        assert device == "cpu"
        assert cache_dir == str(tmp_path / "hf-cache")
        return model, object()

    def tokenize(
        texts: list[str],
        *,
        tokenizer: FakeT5Tokenizer,
        max_seq_len: int,
        device: str,
    ) -> dict[str, object]:
        tokenization_calls.append(
            {
                "texts": texts,
                "tokenizer": tokenizer,
                "max_seq_len": max_seq_len,
                "device": device,
            }
        )
        return {"input_ids": texts}

    fake_japanese_clip = SimpleNamespace(load=load_model, tokenize=tokenize)

    def fake_import_module(name: str) -> object:
        if name == "transformers":
            return SimpleNamespace(T5Tokenizer=FakeT5Tokenizer)
        raise AssertionError(f"Unexpected dependency import: {name}")

    def download_snapshot(
        repository: str,
        revision: str,
        *,
        cache_dir: Path,
        allow_patterns: tuple[str, ...] | None = None,
    ) -> Path:
        assert cache_dir == tmp_path / "hf-cache"
        downloaded.append((repository, revision, allow_patterns))
        if repository == spec.hf_repository:
            return model_snapshot
        return tokenizer_snapshot

    def fake_package_version(name: str) -> str:
        versions = {
            "japanese-clip": "0.2.0",
            "transformers": "4.45.0",
        }
        assert name in versions
        return versions[name]

    monkeypatch.setattr(
        japanese_clip,
        "importlib",
        SimpleNamespace(import_module=fake_import_module),
    )
    monkeypatch.setattr(
        japanese_clip,
        "configure_huggingface_cache",
        lambda: tmp_path / "hf-cache",
    )
    monkeypatch.setattr(japanese_clip, "download_huggingface_snapshot", download_snapshot)
    monkeypatch.setattr(japanese_clip, "_import_japanese_clip", lambda: fake_japanese_clip)
    monkeypatch.setattr(japanese_clip, "package_version", fake_package_version)
    patch_torch_adapter(monkeypatch, japanese_clip)

    adapter = japanese_clip.Adapter(spec, "cpu", "fp32")
    tokens = adapter.preprocess_texts(["猫"], "retrieval", "ja")
    metadata = adapter.metadata()

    assert model.evaluated
    assert [call[:2] for call in downloaded] == [
        (spec.hf_repository, spec.checkpoint_revision),
        (spec.tokenizer_repository, spec.tokenizer_revision),
    ]
    assert downloaded[1][2] == (
        "config.json",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer_config.json",
    )
    assert tokenizer_paths == [str(tokenizer_snapshot)]
    assert adapter.tokenizer.do_lower_case
    assert tokens == {"input_ids": ["猫"]}
    assert tokenization_calls[0]["max_seq_len"] == 77
    assert tokenization_calls[0]["device"] == "cpu"
    assert metadata["tokenizer_implementation_version"] == "4.45.0"
    assert metadata["tokenizer_license"] == spec.tokenizer_license


def test_japanese_clip_compatibility_shim_preserves_repository_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub_calls: list[dict[str, str | None]] = []
    hub = SimpleNamespace(
        hf_hub_download=lambda **kwargs: hub_calls.append(kwargs) or "cached-config"
    )
    modeling_utils = SimpleNamespace()
    model_output = object()
    transformers_utils = SimpleNamespace(ModelOutput=model_output)
    japanese_clip_object = object()
    observed: dict[str, object] = {}

    def import_module(name: str) -> object:
        if name == "huggingface_hub":
            return hub
        if name == "transformers.modeling_utils":
            return modeling_utils
        if name == "transformers.utils":
            return transformers_utils
        if name == "japanese_clip":
            observed["model_output"] = modeling_utils.ModelOutput
            observed["cached_path"] = hub.cached_download(
                "https://huggingface.co/rinna/japanese-clip-vit-b-16/resolve/revision/config.json",
                cache_dir="cache",
            )
            observed["cached_download"] = hub.cached_download
            return japanese_clip_object
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.setattr(
        japanese_clip,
        "importlib",
        SimpleNamespace(import_module=import_module),
    )

    result = japanese_clip._import_japanese_clip()

    assert result is japanese_clip_object
    assert observed["model_output"] is model_output
    assert observed["cached_path"] == "cached-config"
    assert hub_calls == [
        {
            "repo_id": "rinna/japanese-clip-vit-b-16",
            "filename": "config.json",
            "revision": "revision",
            "cache_dir": "cache",
        }
    ]
    assert not hasattr(modeling_utils, "ModelOutput")
    assert not hasattr(hub, "cached_download")
