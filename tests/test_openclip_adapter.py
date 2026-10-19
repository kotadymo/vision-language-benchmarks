from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vision_language_benchmarks.adapters import openclip as openclip_adapter
from vision_language_benchmarks.registry import ModelSpec, get_model


class FakeModel:
    context_length = 77
    visual = SimpleNamespace(image_size=(224, 224))

    def __init__(self) -> None:
        self.evaluated = False

    def eval(self) -> FakeModel:
        self.evaluated = True
        return self

    def parameters(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(numel=lambda: 512)]


class FakeHFTokenizer:
    def __init__(
        self,
        path: str,
        *,
        context_length: int,
        cache_dir: str,
        local_files_only: bool,
    ) -> None:
        self.path = path
        self.context_length = context_length
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only


@pytest.mark.parametrize(
    "model_key",
    [
        "openclip_b32_laion2b",
        "openclip_l14_laion2b",
        "openclip_multilingual_b32",
    ],
)
def test_huggingface_openclip_models_use_pinned_files_and_configs(
    model_key: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = get_model(model_key)
    model_snapshot = tmp_path / "model"
    tokenizer_snapshot = tmp_path / "tokenizer"
    downloads: list[tuple[str, str, tuple[str, ...] | None]] = []
    model_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    tokenizer_calls: list[tuple[str, str]] = []
    created_model = FakeModel()

    def model_config() -> dict[str, Any]:
        return {
            "embed_dim": 512,
            "vision_cfg": {"image_size": 224},
            "text_cfg": {"context_length": 77, "vocab_size": 49_408},
        }

    def download_snapshot(
        repository: str,
        revision: str,
        *,
        allow_patterns: tuple[str, ...] | None = None,
    ) -> Path:
        downloads.append((repository, revision, allow_patterns))
        if repository == spec.hf_repository:
            model_snapshot.mkdir(parents=True, exist_ok=True)
            checkpoint_filename = spec.checkpoint_filename
            assert checkpoint_filename is not None
            (model_snapshot / checkpoint_filename).write_bytes(b"synthetic checkpoint")
            if not spec.uses_openclip_builtin_config:
                (model_snapshot / "open_clip_config.json").write_text(
                    json.dumps(
                        {
                            "model_cfg": model_config(),
                            "preprocess_cfg": {
                                "mean": [0.48, 0.46, 0.41],
                                "std": [0.27, 0.26, 0.28],
                                "interpolation": "bicubic",
                                "resize_mode": "shortest",
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            return model_snapshot

        tokenizer_snapshot.mkdir(parents=True, exist_ok=True)
        for filename in (
            "config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
            "tokenizer.json",
            "tokenizer_config.json",
        ):
            (tokenizer_snapshot / filename).write_bytes(b"tokenizer fixture")
        return tokenizer_snapshot

    def create_model_and_transforms(
        *args: Any,
        **kwargs: Any,
    ) -> tuple[FakeModel, None, Any]:
        model_calls.append((args, kwargs))
        return created_model, None, lambda image: image

    def get_tokenizer(name: str, *, cache_dir: str) -> object:
        tokenizer_calls.append((name, cache_dir))
        return object()

    def get_model_config(model_name: str) -> dict[str, Any]:
        assert model_name == spec.model_id
        return model_config()

    def get_pretrained_config(model_name: str, pretrained: str) -> dict[str, Any]:
        assert model_name == spec.model_id
        assert pretrained == spec.pretrained_tag
        return {
            "mean": [0.48, 0.46, 0.41],
            "std": [0.27, 0.26, 0.28],
            "interpolation": "bicubic",
            "resize_mode": "shortest",
            "quick_gelu": False,
        }

    fake_open_clip = SimpleNamespace(
        create_model_and_transforms=create_model_and_transforms,
        get_model_config=get_model_config,
        get_pretrained_cfg=get_pretrained_config,
        get_tokenizer=get_tokenizer,
    )
    fake_tokenizer_module = SimpleNamespace(HFTokenizer=FakeHFTokenizer)

    def fake_import_module(name: str) -> object:
        if name == "open_clip":
            return fake_open_clip
        if name == "open_clip.tokenizer":
            return fake_tokenizer_module
        raise AssertionError(f"Unexpected optional dependency import: {name}")

    def fake_package_version(name: str) -> str:
        assert name == "open-clip-torch"
        return "3.3.0"

    def fake_torch_adapter_init(
        adapter: Any,
        model_spec: ModelSpec,
        device: str,
        precision: str,
    ) -> None:
        adapter.spec = model_spec
        adapter.device = device
        adapter.precision = precision
        adapter.device_object = SimpleNamespace(type="cpu")

    monkeypatch.setattr(
        openclip_adapter,
        "importlib",
        SimpleNamespace(import_module=fake_import_module),
    )
    monkeypatch.setattr(openclip_adapter, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(
        openclip_adapter,
        "configure_huggingface_cache",
        lambda: tmp_path / "hf-cache",
    )
    monkeypatch.setattr(
        openclip_adapter,
        "download_huggingface_snapshot",
        download_snapshot,
    )
    monkeypatch.setattr(openclip_adapter, "package_version", fake_package_version)
    monkeypatch.setattr(
        openclip_adapter.TorchAdapter,
        "__init__",
        fake_torch_adapter_init,
    )

    adapter = openclip_adapter.Adapter(spec, "cpu", "fp32")
    metadata = adapter.metadata()

    assert created_model.evaluated
    assert downloads[0][0] == spec.hf_repository
    assert downloads[0][1] == spec.checkpoint_revision
    assert spec.checkpoint_filename in (downloads[0][2] or ())
    assert model_calls[0][1]["pretrained"] == str(
        model_snapshot / spec.checkpoint_filename
    )
    assert metadata["checkpoint_files"][0]["path"] == spec.checkpoint_filename
    assert metadata["checkpoint_sha256"]

    if spec.uses_openclip_builtin_config:
        assert len(downloads) == 2
        assert downloads[0][2] == (spec.checkpoint_filename,)
        assert downloads[1][0] == spec.tokenizer_repository
        assert downloads[1][1] == spec.tokenizer_revision
        assert adapter.tokenizer.path == str(tokenizer_snapshot)
        assert adapter.tokenizer.context_length == FakeModel.context_length
        assert adapter.tokenizer.local_files_only
        assert tokenizer_calls == []
        text_config = model_calls[0][1]["text_cfg"]
        assert text_config["hf_model_name"] == str(tokenizer_snapshot)
        assert text_config["hf_tokenizer_name"] == str(tokenizer_snapshot)
        assert text_config["hf_model_pretrained"] is False
        assert metadata["tokenizer_repository"] == spec.tokenizer_repository
        assert metadata["tokenizer_revision"] == spec.tokenizer_revision
    else:
        assert len(downloads) == 1
        assert downloads[0][2] == (
            "open_clip_config.json",
            spec.checkpoint_filename,
        )
        assert tokenizer_calls[0][0] == spec.model_id
        assert model_calls[0][1]["image_mean"] == [0.48, 0.46, 0.41]
        assert metadata["tokenizer_repository"] == f"open_clip:{spec.model_id}"


def test_openai_checkpoint_uses_quick_gelu_from_pretrained_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = get_model("openai_clip_b32")
    checkpoint = tmp_path / "ViT-B-32.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    created_model = FakeModel()
    model_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def create_model_and_transforms(
        *args: Any,
        **kwargs: Any,
    ) -> tuple[FakeModel, None, Any]:
        model_calls.append((args, kwargs))
        return created_model, None, lambda image: image

    def download_pretrained(config: dict[str, Any], *, cache_dir: str) -> str:
        assert config["quick_gelu"] is True
        assert cache_dir == str(tmp_path / "cache" / "open_clip")
        return str(checkpoint)

    def get_pretrained_config(model_name: str, pretrained: str) -> dict[str, Any]:
        assert model_name == spec.model_id
        assert pretrained == spec.pretrained_tag
        return {
            "quick_gelu": True,
            "mean": [0.48, 0.46, 0.41],
            "std": [0.27, 0.26, 0.28],
            "interpolation": "bicubic",
            "resize_mode": "shortest",
        }

    def get_tokenizer(model_name: str, *, cache_dir: str) -> object:
        assert model_name == spec.model_id
        assert cache_dir == str(tmp_path / "cache" / "open_clip")
        return object()

    fake_open_clip = SimpleNamespace(
        create_model_and_transforms=create_model_and_transforms,
        download_pretrained=download_pretrained,
        get_pretrained_cfg=get_pretrained_config,
        get_tokenizer=get_tokenizer,
    )

    def fake_import_module(name: str) -> object:
        assert name == "open_clip"
        return fake_open_clip

    def fake_torch_adapter_init(
        adapter: Any,
        model_spec: ModelSpec,
        device: str,
        precision: str,
    ) -> None:
        adapter.spec = model_spec
        adapter.device = device
        adapter.precision = precision
        adapter.device_object = SimpleNamespace(type="cpu")

    def fake_package_version(distribution: str) -> str:
        assert distribution == "open-clip-torch"
        return "3.3.0"

    monkeypatch.setattr(
        openclip_adapter,
        "importlib",
        SimpleNamespace(import_module=fake_import_module),
    )
    monkeypatch.setattr(openclip_adapter, "package_version", fake_package_version)
    monkeypatch.setattr(openclip_adapter, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(
        openclip_adapter.TorchAdapter,
        "__init__",
        fake_torch_adapter_init,
    )

    adapter = openclip_adapter.Adapter(spec, "cpu", "fp32")

    assert created_model.evaluated
    assert model_calls[0][1]["force_quick_gelu"] is True
    assert adapter.metadata()["checkpoint_sha256"]
