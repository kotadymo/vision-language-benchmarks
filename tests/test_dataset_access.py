from __future__ import annotations

import csv
import json

import pytest

from vision_language_benchmarks.datasets import prepare
from vision_language_benchmarks.errors import DataIntegrityError, DatasetUnavailableError
from vision_language_benchmarks.registry import get_dataset


def test_cifar_official_download_needs_citation_but_no_acceptance_gate() -> None:
    for dataset_key in ("cifar10", "cifar100"):
        spec = get_dataset(dataset_key)

        assert spec.access_status == "available"
        assert spec.license_status == "public_download_citation_required"
        prepare._require_dataset_access(spec, confirm_data_terms=False)


def test_xm3600_requires_per_image_attribution_validation_not_terms_acceptance() -> None:
    spec = get_dataset("crossmodal3600")

    assert spec.access_status == "available"
    assert spec.license_status == "verified_attribution_required"
    assert spec.expected_text_count("en") == 7_200
    assert spec.expected_text_count("ja") == 7_185
    prepare._require_dataset_access(spec, confirm_data_terms=False)


def test_flickr30k_requires_verified_official_access() -> None:
    spec = get_dataset("flickr30k")

    assert spec.access_status == "blocked"
    assert "access request" in spec.access_reason
    assert "Flickr Terms of Use" in spec.access_reason
    with pytest.raises(DatasetUnavailableError, match="access was approved"):
        prepare._require_dataset_access(spec, confirm_data_terms=False)


def test_coco_karpathy_split_caption_count_matches_archive() -> None:
    spec = get_dataset("mscoco_captions")

    assert spec.expected_images == 5_000
    assert spec.expected_texts == 25_010
    assert spec.access_status == "blocked"
    assert "no user acceptance" in spec.access_reason


def test_xm3600_records_all_captions_per_image(tmp_path) -> None:
    captions = tmp_path / "captions.jsonl"
    captions.write_text(
        json.dumps(
            {
                "image/key": "image-1",
                "en": {"caption": ["caption one", "caption two"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = prepare._xm3600_records(captions, None, "en")

    assert records == [
        {
            "image_id": "image-1",
            "image_path": "image-1.jpg",
            "captions": ["caption one", "caption two"],
        }
    ]


def test_xm3600_attribution_manifest_validates_ids_and_licenses(tmp_path) -> None:
    path = tmp_path / "image_attributions.csv"
    rows = [
        {
            "ImageID": image_id,
            "Subset": "test",
            "OriginalURL": f"https://example.com/{image_id}.jpg",
            "OriginalLandingURL": f"https://example.com/{image_id}",
            "License": "https://creativecommons.org/licenses/by/2.0/",
            "Author": "Example photographer",
            "Title": image_id,
        }
        for image_id in ("image-1", "image-2")
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = prepare._validate_xm3600_attributions(
        path,
        {"image-1", "image-2"},
        expected_image_count=2,
    )

    assert metadata["image_license_count"] == 2
    assert metadata["image_license_counts"] == {
        "https://creativecommons.org/licenses/by/2.0/": 2
    }
    assert metadata["image_attribution_subset_counts"] == {"test": 2}

    rows[0]["License"] = "https://example.com/unknown-license"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(DataIntegrityError, match="unverified license"):
        prepare._validate_xm3600_attributions(
            path,
            {"image-1", "image-2"},
            expected_image_count=2,
        )


def test_blocked_coco_preparation_never_requests_downloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    def unexpected_download(*args: object, **kwargs: object) -> None:
        pytest.fail(f"Blocked COCO data must not be downloaded: {args!r}, {kwargs!r}")

    monkeypatch.setattr(prepare, "download_file", unexpected_download)
    with pytest.raises(DatasetUnavailableError, match="per-image rights"):
        prepare.prepare_dataset("mscoco_captions", tmp_path)


def test_cifar_marker_loads_without_terms_acceptance(
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

    class FakeCIFAR:
        classes = [f"class-{index}" for index in range(spec.expected_classes or 0)]

        def __len__(self) -> int:
            return spec.expected_images or 0

    dataset = FakeCIFAR()

    def load_cifar(dataset_key: str, root) -> FakeCIFAR:
        assert dataset_key == "cifar10"
        assert root == dataset_root
        return dataset

    monkeypatch.setattr(prepare, "_load_torchvision_cifar", load_cifar)

    assert prepare.load_cifar_dataset("cifar10", tmp_path) is dataset
