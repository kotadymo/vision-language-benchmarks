from __future__ import annotations

import json

import pytest

from vision_language_benchmarks.datasets import prepare
from vision_language_benchmarks.errors import DataIntegrityError, DatasetUnavailableError
from vision_language_benchmarks.registry import get_dataset


def test_cifar_preparation_requires_terms_before_creating_files(tmp_path) -> None:
    data_root = tmp_path / "not-created"

    with pytest.raises(DatasetUnavailableError, match="confirm-data-terms"):
        prepare.prepare_dataset("cifar10", data_root)

    assert not data_root.exists()


def test_xm3600_preparation_requires_terms_before_network_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    def unexpected_network_request(*args: object, **kwargs: object) -> None:
        pytest.fail(
            "XM3600 metadata request must not happen before terms confirmation: "
            f"{args!r}, {kwargs!r}"
        )

    monkeypatch.setattr(prepare, "_remote_size", unexpected_network_request)
    data_root = tmp_path / "not-created"

    with pytest.raises(DatasetUnavailableError, match="confirm-data-terms"):
        prepare.prepare_xm3600(data_root)

    assert not data_root.exists()


def test_blocked_coco_preparation_never_requests_downloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    def unexpected_download(*args: object, **kwargs: object) -> None:
        pytest.fail(f"Blocked COCO data must not be downloaded: {args!r}, {kwargs!r}")

    monkeypatch.setattr(prepare, "download_file", unexpected_download)
    with pytest.raises(DatasetUnavailableError, match="per-image rights"):
        prepare.prepare_dataset("mscoco_captions", tmp_path)


def test_cifar_marker_does_not_bypass_terms_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    spec = get_dataset("cifar10")
    dataset_root = tmp_path / "cifar10"
    dataset_root.mkdir()
    (dataset_root / ".vlbench-dataset.json").write_text(
        json.dumps(
            {
                "dataset": "cifar10",
                "split": spec.split,
                "actual_sample_count": spec.expected_images,
                "data_integrity_status": "verified",
                "dataset_terms_confirmed": False,
            }
        ),
        encoding="utf-8",
    )

    def unexpected_cifar_load(*args: object, **kwargs: object) -> None:
        pytest.fail(f"CIFAR must not load without confirmed terms: {args!r}, {kwargs!r}")

    monkeypatch.setattr(prepare, "_load_torchvision_cifar", unexpected_cifar_load)
    with pytest.raises(DataIntegrityError, match="metadata is inconsistent"):
        prepare.load_cifar_dataset("cifar10", tmp_path)
