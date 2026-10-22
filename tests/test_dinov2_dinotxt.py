from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from tests.helpers import make_result
from vision_language_benchmarks.adapters import dinov2_dinotxt
from vision_language_benchmarks.cli import main
from vision_language_benchmarks.dinotxt_provenance import (
    DINOTXT_CHECKPOINT_REVISION,
    DinoTxtArtifact,
    checkpoint_fingerprint,
    expected_component_records,
    verify_cached_components,
)
from vision_language_benchmarks.embedding_dimensions import resolve_embedding_dimension
from vision_language_benchmarks.errors import (
    DataIntegrityError,
    ModelAccessBlockedError,
)
from vision_language_benchmarks.evaluation import runner
from vision_language_benchmarks.prompts import (
    classification_prompt_groups,
    classification_prompt_name,
    retrieval_prompt_name,
)
from vision_language_benchmarks.registry import get_model
from vision_language_benchmarks.reporting.aggregate import _status_rows, render_report
from vision_language_benchmarks.reporting.schema import validate_result


def test_cached_dinotxt_components_are_hash_verified(tmp_path: Path) -> None:
    payload = b"test checkpoint"
    component = DinoTxtArtifact(
        component="test_weight",
        kind="weight",
        filename="test.pth",
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license="test",
        license_status="verified",
    )
    (tmp_path / component.filename).write_bytes(payload)

    records = verify_cached_components(tmp_path, (component,))

    assert records == [
        {
            "component": "test_weight",
            "kind": "weight",
            "filename": "test.pth",
            "sha256": component.sha256,
            "size_bytes": len(payload),
            "license": "test",
            "license_status": "verified",
            "verified": True,
        }
    ]
    assert checkpoint_fingerprint(records)["checkpoint_files"] == [
        {"path": "test.pth", "sha256": component.sha256}
    ]


def test_cached_dinotxt_component_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    component = DinoTxtArtifact(
        component="test_weight",
        kind="weight",
        filename="test.pth",
        sha256="0" * 64,
        size_bytes=4,
        license="test",
        license_status="verified",
    )
    (tmp_path / component.filename).write_bytes(b"data")

    with pytest.raises(DataIntegrityError, match="SHA-256 mismatch"):
        verify_cached_components(tmp_path, (component,))


def test_dinotxt_adapter_blocks_before_loading_unlicensed_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dinov2_dinotxt,
        "package_version",
        lambda _: pytest.fail("The blocked adapter must not inspect or load the package."),
    )

    with pytest.raises(ModelAccessBlockedError, match="license is verified"):
        dinov2_dinotxt.Adapter(get_model("dinov2_dinotxt"), "cuda", "fp32")


class FakeTensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.shape = values.shape

    def to(self, _device: object) -> FakeTensor:
        return self

    def detach(self) -> FakeTensor:
        return self

    def float(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.values

    def __len__(self) -> int:
        return self.shape[0]


class FakeDinoTxtModel:
    def __init__(self) -> None:
        self.evaluated = False
        self.device: object | None = None
        self.embedding_space_metadata: Any = None

    def to(self, device: object) -> FakeDinoTxtModel:
        self.device = device
        return self

    def eval(self) -> FakeDinoTxtModel:
        self.evaluated = True
        return self

    def encode_image(self, batch: FakeTensor, *, normalize: bool) -> FakeTensor:
        assert normalize is False
        return FakeTensor(np.ones((batch.shape[0], 2048), dtype=np.float32))

    def encode_text(self, tokens: FakeTensor, *, normalize: bool) -> FakeTensor:
        assert normalize is False
        return FakeTensor(np.ones((len(tokens), 2048), dtype=np.float32))

    def parameters(self) -> list[object]:
        return []


def test_dinotxt_adapter_uses_pinned_openvlclip_api_and_fixed_global_dimension(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = replace(
        get_model("dinov2_dinotxt"),
        weight_license_status="verified",
        weight_license_reason=None,
    )
    model = FakeDinoTxtModel()
    factory_calls: list[dict[str, Any]] = []
    tokenizer_calls: list[tuple[str, str | None, list[str]]] = []

    def create_model_and_transforms(**kwargs: Any) -> tuple[FakeDinoTxtModel, None, Any]:
        factory_calls.append(kwargs)
        model.embedding_space_metadata = SimpleNamespace(
            model_id=spec.model_id,
            model_revision=spec.checkpoint_revision,
            embedding_dimension=2048,
            embedding_strategy="dinotxt-global-cls-plus-mean-patches-v1",
            normalization="none-or-l2-global-2048-v1",
        )
        return model, None, lambda _image: FakeTensor(
            np.ones((3, 224, 224), dtype=np.float32)
        )

    def get_tokenizer(model_name: str, *, prompt_name: str | None) -> Any:
        def tokenize(texts: list[str]) -> FakeTensor:
            tokenizer_calls.append((model_name, prompt_name, texts))
            return FakeTensor(np.ones((len(texts), 77), dtype=np.int64))

        return tokenize

    fake_package = SimpleNamespace(
        create_model_and_transforms=create_model_and_transforms,
        get_tokenizer=get_tokenizer,
    )
    fake_torch = SimpleNamespace(
        hub=SimpleNamespace(get_dir=lambda: str(tmp_path)),
        stack=lambda tensors: FakeTensor(
            np.stack([tensor.values for tensor in tensors])
        ),
    )
    component_records = [
        {**record, "verified": True} for record in expected_component_records()
    ]

    def initialize_adapter(
        adapter: Any,
        selected_spec: Any,
        device: str,
        precision: str,
    ) -> None:
        adapter.spec = selected_spec
        adapter.device = device
        adapter.precision = precision
        adapter.device_object = "cpu"
        adapter.torch = fake_torch
        adapter.inference_context = nullcontext
        adapter.autocast_context = nullcontext

    monkeypatch.setattr(
        dinov2_dinotxt.TorchAdapter,
        "__init__",
        initialize_adapter,
    )
    monkeypatch.setattr(
        dinov2_dinotxt.importlib,
        "import_module",
        lambda name: fake_package
        if name == "openvlclip"
        else pytest.fail(f"Unexpected import: {name}"),
    )
    monkeypatch.setattr(dinov2_dinotxt, "package_version", lambda _: "0.6.0")
    monkeypatch.setattr(
        dinov2_dinotxt,
        "verify_cached_components",
        lambda _directory: component_records,
    )

    adapter = dinov2_dinotxt.Adapter(spec, "cpu", "fp32")
    image_batch = adapter.preprocess_images([Image.new("RGB", (32, 32))])
    image_features = adapter.encode_images(image_batch)
    text_tokens = adapter.preprocess_texts(
        ["a photo of a cat."],
        "classification",
        "en",
    )
    text_features = adapter.encode_texts(text_tokens, "retrieval", "ja")
    metadata = adapter.metadata()

    assert model.evaluated
    assert factory_calls == [
        {
            "model_name": "dinov2-dinotxt",
            "pretrained": "official",
            "precision": "fp32",
            "device": "cpu",
            "logit_scale_mode": "cosine",
        }
    ]
    assert tokenizer_calls == [
        ("dinov2-dinotxt", None, ["a photo of a cat."])
    ]
    assert image_features.shape == (1, 2048)
    assert text_features.shape == (1, 2048)
    assert metadata["adapter_name"] == "openvlclip_dinotxt"
    assert metadata["global_embedding_dimension"] == 2048
    assert metadata["model_library_version"] == "0.6.0"
    assert all(component["verified"] for component in metadata["checkpoint_components"])
    assert metadata["preprocessing_config"]["task_prompt"] is None
    assert {
        component["license_status"] for component in metadata["checkpoint_components"]
    } == {"verified", "unverified"}


def test_dinotxt_prompt_policy_and_dimension_are_explicit() -> None:
    groups, policy, _ = classification_prompt_groups(
        ["cat"],
        "p1",
        "dinov2_dinotxt",
    )

    assert groups == [["a photo of a cat."]]
    assert policy == "dinotxt_p1_single_photo_template"
    assert classification_prompt_name("dinov2_dinotxt") is None
    assert retrieval_prompt_name("dinov2_dinotxt") is None
    with pytest.raises(ValueError, match="fixed 2048D"):
        resolve_embedding_dimension("dinov2_dinotxt", 128)


def test_dinotxt_model_revision_matches_pinned_upstream_identity() -> None:
    assert (
        get_model("dinov2_dinotxt").checkpoint_revision
        == DINOTXT_CHECKPOINT_REVISION
    )
    assert DINOTXT_CHECKPOINT_REVISION.startswith(
        "7764ea0f912e53c92e82eb78a2a1631e92725fc8+backbone-"
    )


def test_dinotxt_benchmark_is_persisted_as_blocked_without_dataset_or_model_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runner, "_source_commit_sha", lambda: "a" * 40)
    monkeypatch.setattr(runner, "_working_tree_dirty", lambda: False)
    monkeypatch.setattr(runner, "_lock_sha256", lambda: "b" * 64)
    monkeypatch.setattr(
        runner,
        "doctor_report",
        lambda device: {
            "torch": {
                "cuda_available": False,
                "devices": [],
                "version": "test",
                "cuda_runtime": None,
            },
            "nvidia_smi": {"devices": []},
        },
    )
    monkeypatch.setattr(
        runner,
        "prepare_dataset",
        lambda *args, **kwargs: pytest.fail("A blocked model must not prepare the dataset."),
    )
    monkeypatch.setattr(
        runner,
        "create_adapter",
        lambda *args, **kwargs: pytest.fail("A blocked model must not load weights."),
    )

    result = runner.run_benchmark(
        "dinov2_dinotxt",
        "cifar10",
        protocol="p0",
        language="en",
        device="cpu",
        results_root=tmp_path,
    )

    validate_result(result)
    metadata = result["benchmark_metadata"]
    assert result["status"] == "blocked"
    assert metadata["model_weight_license_status"] == "blocked"
    assert metadata["global_embedding_dimension"] == 2048
    assert metadata.get("embedding_dimension") is None
    assert metadata["checkpoint_components"] is None
    assert metadata["expected_checkpoint_components"]
    assert "external dino.txt Vision Head" in metadata["reason_if_not_success"]


def test_cli_dry_run_exposes_model_license_block(tmp_path: Path, capsys) -> None:
    output = tmp_path / "blocked-plan"
    exit_code = main(
        [
            "run",
            "--model",
            "dinov2_dinotxt",
            "--dataset",
            "cifar10",
            "--device",
            "cpu",
            "--output",
            str(output),
            "--dry-run",
        ]
    )
    plan = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert plan["model"]["access"] == "blocked"
    assert plan["model"]["weight_license_status"] == "blocked"
    assert plan["model"]["global_embedding_dimension"] == 2048
    assert plan["dataset"]["access"] == "available"
    assert len(plan["model"]["expected_checkpoint_components"]) == 4
    assert not output.exists()


def test_cli_records_blocked_model_without_download_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys,
) -> None:
    monkeypatch.setattr(runner, "_source_commit_sha", lambda: "a" * 40)
    monkeypatch.setattr(runner, "_working_tree_dirty", lambda: False)
    monkeypatch.setattr(runner, "_lock_sha256", lambda: "b" * 64)
    monkeypatch.setattr(
        runner,
        "doctor_report",
        lambda device: {
            "torch": {
                "cuda_available": False,
                "devices": [],
                "version": "test",
                "cuda_runtime": None,
            },
            "nvidia_smi": {"devices": []},
        },
    )
    monkeypatch.setattr(
        runner,
        "prepare_dataset",
        lambda *args, **kwargs: pytest.fail("A blocked model must not prepare the dataset."),
    )
    monkeypatch.setattr(
        "vision_language_benchmarks.cli._authorize_downloads",
        lambda *args, **kwargs: pytest.fail("A blocked model must not request downloads."),
    )

    output = tmp_path / "blocked-run"
    exit_code = main(
        [
            "run",
            "--model",
            "dinov2_dinotxt",
            "--dataset",
            "cifar10",
            "--device",
            "cpu",
            "--output",
            str(output),
        ]
    )
    result = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert result["status"] == "blocked"
    assert result["benchmark_metadata"]["model_weight_license_status"] == "blocked"
    assert list(output.glob("*.json"))


def test_dinotxt_non_success_reporting_surfaces_license_status() -> None:
    result = make_result(
        model_key="dinov2_dinotxt",
        status="blocked",
        run_id="dinotxt-blocked-report",
    )
    validate_result(result)
    status_row = _status_rows([result])[0]

    assert result["benchmark_metadata"]["model_weight_license_status"] == "blocked"
    assert result["benchmark_metadata"]["global_embedding_dimension"] == 2048
    assert status_row["model_weight_license_status"] == "blocked"
    assert "external dino.txt Vision Head" in status_row["model_weight_license_reason"]

    report = render_report([result])
    assert "| Model weight license |" in report
    assert status_row["model"] in report
    assert "| blocked |" in report
