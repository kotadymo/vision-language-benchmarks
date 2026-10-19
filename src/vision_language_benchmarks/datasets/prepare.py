from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from vision_language_benchmarks.atomic import (
    atomic_write_json,
    content_hash,
    md5_file,
    sha256_file,
)
from vision_language_benchmarks.datasets.downloader import (
    download_file,
    safe_extract_tar,
    safe_extract_zip,
)
from vision_language_benchmarks.errors import (
    DataIntegrityError,
    DatasetUnavailableError,
)
from vision_language_benchmarks.paths import DATA_ROOT
from vision_language_benchmarks.registry import DatasetSpec, get_dataset

MIN_FREE_SPACE_BYTES = 20 * 1024**3
CIFAR_ARCHIVES = {
    "cifar10": {
        "url": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        "md5": "c58f30108f718f92721af3b95e74349a",
        "directory": "cifar-10-batches-py",
        "files": ("test_batch", "batches.meta"),
    },
    "cifar100": {
        "url": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
        "md5": "eb9058c3a382ffc7106e4002c42a8d85",
        "directory": "cifar-100-python",
        "files": ("test", "meta"),
    },
}
XM3600_IMAGES_URL = "https://open-images-dataset.s3.amazonaws.com/crossmodal-3600/images.tgz"
XM3600_CAPTIONS_URL = "https://google.github.io/crossmodal-3600/web-data/captions.zip"
KARPATHY_ANNOTATIONS_URL = (
    "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
)
COCO_IMAGE_BASE_URL = "https://images.cocodataset.org"


def _ensure_space(path: Path, estimated_bytes: int) -> dict[str, int]:
    if estimated_bytes < 0:
        raise ValueError("estimated_bytes must not be negative.")
    existing = path
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    minimum_required = estimated_bytes + MIN_FREE_SPACE_BYTES
    if free < minimum_required:
        raise DatasetUnavailableError(
            f"Insufficient disk space at {existing}: {free} bytes free, "
            f"{estimated_bytes} bytes estimated for this dataset, and "
            f"{MIN_FREE_SPACE_BYTES} bytes must remain free."
        )
    return {"free_bytes_before": free, "estimated_download_and_extract_bytes": estimated_bytes}


def _require_dataset_access(spec: DatasetSpec, confirm_data_terms: bool) -> None:
    if spec.access_status == "blocked" or spec.license_status == "blocked":
        raise DatasetUnavailableError(spec.access_reason or f"{spec.key} is blocked.")
    if spec.license_status == "requires_user_confirmation" and not confirm_data_terms:
        raise DatasetUnavailableError(
            f"{spec.key} has unverified usage terms. Review {spec.source} and rerun with "
            "--confirm-data-terms only if you are authorized to use it."
        )


def _remote_size(url: str, timeout: int = 20) -> int | None:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "vision-language-benchmarks/0.1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = response.headers.get("Content-Length")
    except urllib.error.HTTPError as error:
        if error.code in {400, 403, 405, 501}:
            return None
        raise
    if value is None:
        return None
    size = int(value)
    if size < 0:
        raise DataIntegrityError(f"Invalid Content-Length for {url}: {value!r}.")
    return size


def _atomic_write_json_lines(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
            newline="\n",
        ) as stream:
            temporary = Path(stream.name)
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _load_torchvision_cifar(dataset_key: str, root: Path) -> Any:
    try:
        from torchvision.datasets import CIFAR10, CIFAR100
    except ImportError as error:
        raise DatasetUnavailableError(
            "CIFAR loading requires torchvision. Install the project 'cuda' extra."
        ) from error

    dataset_class = CIFAR10 if dataset_key == "cifar10" else CIFAR100
    return dataset_class(root=str(root), train=False, download=False)


def prepare_cifar(
    dataset_key: str,
    data_root: Path = DATA_ROOT,
    *,
    confirm_data_terms: bool = False,
) -> dict[str, Any]:
    if dataset_key not in CIFAR_ARCHIVES:
        raise ValueError(f"Not a supported CIFAR dataset: {dataset_key!r}.")
    spec = get_dataset(dataset_key)
    _require_dataset_access(spec, confirm_data_terms)
    config = CIFAR_ARCHIVES[dataset_key]
    dataset_root = data_root / dataset_key
    dataset_root.mkdir(parents=True, exist_ok=True)
    archive = dataset_root / Path(config["url"]).name
    target = dataset_root / config["directory"]
    marker_path = dataset_root / ".vlbench-dataset.json"

    if not target.exists():
        _ensure_space(data_root, 200 * 1024**2)
        download_file(config["url"], archive, md5=config["md5"])
        staging = dataset_root / ".extract-cifar"
        if staging.exists():
            raise DatasetUnavailableError(
                f"An incomplete extraction exists at {staging}; preserve it and move it "
                "aside before preparing a clean dataset."
            )
        safe_extract_tar(archive, staging)
        extracted = staging / config["directory"]
        if not extracted.is_dir():
            raise DataIntegrityError(
                f"Expected directory {config['directory']!r} in {archive.name}."
            )
        if target.exists():
            raise FileExistsError(f"Refusing to replace existing dataset directory {target}.")
        os.replace(extracted, target)
        shutil.rmtree(staging)
    elif not archive.exists():
        _ensure_space(data_root, 200 * 1024**2)
        download_file(config["url"], archive, md5=config["md5"])

    archive_md5 = md5_file(archive)
    if archive_md5 != config["md5"]:
        raise DataIntegrityError(
            f"{dataset_key} archive MD5 mismatch: expected {config['md5']}, got {archive_md5}."
        )
    dataset = _load_torchvision_cifar(dataset_key, dataset_root)
    expected_images = spec.expected_images
    actual_images = len(dataset)
    if expected_images != actual_images:
        raise DataIntegrityError(
            f"{dataset_key} test split has {actual_images} samples; expected {expected_images}."
        )
    if len(dataset.classes) != spec.expected_classes:
        raise DataIntegrityError(
            f"{dataset_key} has {len(dataset.classes)} classes; expected {spec.expected_classes}."
        )

    record = {
        "dataset": dataset_key,
        "version": "CIFAR official python archive",
        "download_source": config["url"],
        "license": spec.license,
        "license_status": spec.license_status,
        "dataset_terms_confirmed": confirm_data_terms,
        "split": spec.split,
        "expected_sample_count": expected_images,
        "actual_sample_count": actual_images,
        "class_count": len(dataset.classes),
        "annotation_source": "Official CIFAR Python archive",
        "archive_md5": archive_md5,
        "archive_sha256": sha256_file(archive),
        "data_integrity_status": "verified",
        "missing_or_excluded_samples": [],
        "dataset_checksum": content_hash(
            {
                "archive_sha256": sha256_file(archive),
                "split": spec.split,
                "dataset": dataset_key,
            }
        ),
    }
    atomic_write_json(marker_path, record)
    return record


def load_cifar_dataset(dataset_key: str, data_root: Path = DATA_ROOT) -> Any:
    spec = get_dataset(dataset_key)
    dataset_root = data_root / dataset_key
    marker_path = dataset_root / ".vlbench-dataset.json"
    if not marker_path.is_file():
        raise DatasetUnavailableError(
            f"{dataset_key} is not prepared; run `vlbench datasets prepare "
            f"--dataset {dataset_key}` first."
        )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        marker.get("dataset") != dataset_key
        or marker.get("split") != spec.split
        or marker.get("actual_sample_count") != spec.expected_images
        or marker.get("data_integrity_status") != "verified"
        or (
            spec.license_status == "requires_user_confirmation"
            and marker.get("dataset_terms_confirmed") is not True
        )
    ):
        raise DataIntegrityError(f"Prepared {dataset_key} metadata is inconsistent.")
    dataset = _load_torchvision_cifar(dataset_key, dataset_root)
    if len(dataset) != spec.expected_images:
        raise DataIntegrityError(
            f"{dataset_key} test split changed after preparation: {len(dataset)} samples."
        )
    return dataset


def _xm3600_records(
    captions_path: Path,
    images_path: Path,
    language_code: str,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    with captions_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            image_id = str(raw["image/key"])
            if image_id in seen_ids:
                raise DataIntegrityError(f"Duplicate XM3600 image ID {image_id!r}.")
            seen_ids.add(image_id)
            language = raw.get(language_code)
            if not isinstance(language, dict) or not language.get("caption"):
                raise DataIntegrityError(
                    f"XM3600 line {line_number} has no {language_code!r} caption."
                )
            image_name = f"{image_id}.jpg"
            image_path = images_path / image_name
            if not image_path.is_file():
                raise DataIntegrityError(f"XM3600 image is missing: {image_name}.")
            records.append(
                {
                    "image_id": image_id,
                    "image_path": image_name,
                    "caption": str(language["caption"][0]),
                }
            )
    return records


def prepare_xm3600(
    data_root: Path = DATA_ROOT,
    *,
    progress: Callable[[str], None] | None = None,
    confirm_data_terms: bool = False,
) -> dict[str, Any]:
    spec = get_dataset("crossmodal3600")
    _require_dataset_access(spec, confirm_data_terms)
    dataset_root = data_root / "crossmodal3600"
    dataset_root.mkdir(parents=True, exist_ok=True)
    image_archive = dataset_root / "images.tgz"
    caption_archive = dataset_root / "captions.zip"
    images_path = dataset_root / "images"
    captions_path = dataset_root / "captions.jsonl"
    marker_path = dataset_root / ".vlbench-dataset.json"

    if marker_path.is_file():
        existing = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            existing.get("dataset") == "crossmodal3600"
            and str(existing.get("data_integrity_status", "")).startswith("verified")
            and existing.get("actual_sample_count") == spec.expected_images
        ):
            cached_records: dict[str, list[dict[str, str]]] = {}
            for language in spec.languages:
                language_code = spec.source_language_code(language)
                cached_records[language] = _xm3600_records(
                    captions_path,
                    images_path,
                    language_code,
                )
            annotations_exist = all(
                (
                    dataset_root
                    / f"annotations-{spec.source_language_code(language)}.jsonl"
                ).is_file()
                for language in spec.languages
            )
            if (
                annotations_exist
                and all(len(records) == spec.expected_images for records in cached_records.values())
            ):
                existing["dataset_terms_confirmed"] = confirm_data_terms
                existing["license_status"] = spec.license_status
                atomic_write_json(marker_path, existing)
                return existing

    image_size = _remote_size(XM3600_IMAGES_URL)
    caption_size = _remote_size(XM3600_CAPTIONS_URL)
    estimate = (image_size or 2 * 1024**3) * 2 + (caption_size or 100 * 1024**2) * 2
    space = _ensure_space(data_root, estimate)
    if progress:
        progress(
            f"Preparing {spec.name}: approximately {estimate / 1024**3:.2f} GiB "
            f"download/extraction space estimated; {space['free_bytes_before'] / 1024**3:.2f} "
            f"GiB currently free. Terms status: {spec.license_status}. {spec.license} "
            "Images will not be published."
        )

    if not captions_path.is_file():
        download_file(XM3600_CAPTIONS_URL, caption_archive, retries=3)
        if not zipfile.is_zipfile(caption_archive):
            raise DataIntegrityError("XM3600 captions download is not a valid ZIP archive.")
        caption_stage = dataset_root / ".captions-extract"
        if caption_stage.exists():
            raise DatasetUnavailableError(
                f"Incomplete XM3600 caption extraction exists at {caption_stage}."
            )
        safe_extract_zip(caption_archive, caption_stage)
        matches = list(caption_stage.rglob("captions.jsonl"))
        if len(matches) != 1:
            raise DataIntegrityError(
                f"Expected one captions.jsonl in XM3600 archive, found {len(matches)}."
            )
        os.replace(matches[0], captions_path)
        shutil.rmtree(caption_stage)
    elif not caption_archive.exists():
        download_file(XM3600_CAPTIONS_URL, caption_archive, retries=3)

    if not images_path.is_dir():
        download_file(XM3600_IMAGES_URL, image_archive, retries=3)
        if not tarfile.is_tarfile(image_archive):
            raise DataIntegrityError("XM3600 images download is not a valid TAR archive.")
        safe_extract_tar(image_archive, images_path)
    elif not image_archive.exists():
        download_file(XM3600_IMAGES_URL, image_archive, retries=3)

    language_records = {
        language: _xm3600_records(
            captions_path,
            images_path,
            spec.source_language_code(language),
        )
        for language in spec.languages
    }
    if any(len(records) != spec.expected_images for records in language_records.values()):
        raise DataIntegrityError(
            f"XM3600 expected {spec.expected_images} records per language."
        )
    image_hashes = [
        (path.name, sha256_file(path))
        for path in sorted(images_path.iterdir())
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    ]
    if len(image_hashes) != spec.expected_images:
        raise DataIntegrityError(
            f"XM3600 image folder has {len(image_hashes)} JPEG files; "
            f"expected {spec.expected_images}."
        )
    images_archive_sha = sha256_file(image_archive)
    captions_archive_sha = sha256_file(caption_archive)
    captions_sha = sha256_file(captions_path)
    record = {
        "dataset": "crossmodal3600",
        "version": "Crossmodal-3600 official web-data release",
        "download_sources": {
            "images": XM3600_IMAGES_URL,
            "captions": XM3600_CAPTIONS_URL,
        },
        "license": spec.license,
        "license_status": spec.license_status,
        "dataset_terms_confirmed": confirm_data_terms,
        "split": spec.split,
        "expected_sample_count": spec.expected_images,
        "actual_sample_count": len(image_hashes),
        "text_count_by_language": {
            language: len(records) for language, records in language_records.items()
        },
        "annotation_source": captions_sha,
        "images_archive_sha256": images_archive_sha,
        "captions_archive_sha256": captions_archive_sha,
        "image_files_sha256": content_hash(image_hashes),
        "dataset_checksum": content_hash(
            {
                "images_archive_sha256": images_archive_sha,
                "captions_sha256": captions_sha,
                "image_files_sha256": content_hash(image_hashes),
            }
        ),
        "data_integrity_status": "verified; upstream per-file checksums not published",
        "missing_or_excluded_samples": [],
        "language_codes": {
            language: spec.source_language_code(language) for language in spec.languages
        },
    }
    for language, records in language_records.items():
        language_code = spec.source_language_code(language)
        _atomic_write_json_lines(dataset_root / f"annotations-{language_code}.jsonl", records)
    atomic_write_json(marker_path, record)
    return record


def _download_coco_image(filename: str, image_directory: Path) -> tuple[str, str]:
    if filename.startswith("COCO_val2014_"):
        split_directory = "val2014"
    elif filename.startswith("COCO_train2014_"):
        split_directory = "train2014"
    else:
        raise DataIntegrityError(f"Unexpected Karpathy COCO image filename: {filename!r}.")
    destination = image_directory / filename
    url = f"{COCO_IMAGE_BASE_URL}/{split_directory}/{filename}"
    download_file(url, destination, retries=3)
    try:
        with Image.open(destination) as image:
            image.verify()
    except OSError as error:
        raise DataIntegrityError(f"Downloaded COCO image is invalid: {filename}.") from error
    return filename, sha256_file(destination)


def _coco_test_items(annotations: dict[str, Any]) -> list[dict[str, Any]]:
    images = annotations.get("images")
    if not isinstance(images, list):
        raise DataIntegrityError("Karpathy COCO annotations have no images list.")
    test_items = [item for item in images if item.get("split") == "test"]
    if len(test_items) != 5_000:
        raise DataIntegrityError(
            f"Karpathy COCO test split has {len(test_items)} images; expected 5000."
        )
    result: list[dict[str, Any]] = []
    ids: set[str] = set()
    for item in test_items:
        filename = str(item.get("filename", ""))
        captions = [str(sentence.get("raw", "")).strip() for sentence in item.get("sentences", [])]
        if not filename or any(not caption for caption in captions) or len(captions) != 5:
            raise DataIntegrityError(
                f"Karpathy COCO item {filename!r} must have exactly five nonempty captions."
            )
        if filename in ids:
            raise DataIntegrityError(f"Duplicate Karpathy COCO image: {filename}.")
        ids.add(filename)
        result.append(
            {
                "image_id": filename.removesuffix(".jpg"),
                "image_path": filename,
                "captions": captions,
            }
        )
    return result


def prepare_coco(
    data_root: Path = DATA_ROOT,
    *,
    progress: Callable[[str], None] | None = None,
    workers: int = 6,
    confirm_data_terms: bool = False,
) -> dict[str, Any]:
    spec = get_dataset("mscoco_captions")
    _require_dataset_access(spec, confirm_data_terms)
    if workers < 1 or workers > 16:
        raise ValueError("workers must be between 1 and 16.")
    dataset_root = data_root / "mscoco_captions"
    dataset_root.mkdir(parents=True, exist_ok=True)
    image_directory = dataset_root / "images"
    annotation_archive = dataset_root / "caption_datasets.zip"
    annotation_path = dataset_root / "dataset_coco.json"
    evaluation_items_path = dataset_root / "evaluation-items.json"
    marker_path = dataset_root / ".vlbench-dataset.json"
    if marker_path.is_file():
        existing = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            existing.get("dataset") == "mscoco_captions"
            and existing.get("data_integrity_status") == "verified"
            and existing.get("actual_sample_count") == spec.expected_images
            and evaluation_items_path.is_file()
        ):
            cached_items = json.loads(evaluation_items_path.read_text(encoding="utf-8")).get(
                "items"
            )
            if (
                isinstance(cached_items, list)
                and len(cached_items) == spec.expected_images
                and all((image_directory / item["image_path"]).is_file() for item in cached_items)
            ):
                return existing

    _ensure_space(data_root, 5 * 1024**3)
    if progress:
        progress(
            "Preparing the Karpathy 5K test images only; estimated image storage is "
            "5 GiB, in addition to the required 20 GiB free-space reserve. "
            f"Terms status: {spec.license_status}. {spec.license} "
            "Images will not be published."
        )
    if not annotation_path.is_file():
        download_file(KARPATHY_ANNOTATIONS_URL, annotation_archive, retries=3)
        if not zipfile.is_zipfile(annotation_archive):
            raise DataIntegrityError("Karpathy annotations download is not a ZIP archive.")
        annotation_stage = dataset_root / ".annotations-extract"
        if annotation_stage.exists():
            raise DatasetUnavailableError(
                f"Incomplete Karpathy annotation extraction exists at {annotation_stage}."
            )
        safe_extract_zip(annotation_archive, annotation_stage)
        candidates = list(annotation_stage.rglob("dataset_coco.json"))
        if len(candidates) != 1:
            raise DataIntegrityError(
                f"Expected one dataset_coco.json in annotation archive, found {len(candidates)}."
            )
        os.replace(candidates[0], annotation_path)
        shutil.rmtree(annotation_stage)
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    items = _coco_test_items(annotations)
    if not image_directory.exists():
        image_directory.mkdir(parents=True)
    if progress:
        progress(f"Downloading exactly {len(items)} Karpathy test images from COCO.")
    image_directory.mkdir(parents=True, exist_ok=True)
    failures: list[Exception] = []
    hashes: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_coco_image, item["image_path"], image_directory): item[
                "image_path"
            ]
            for item in items
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                filename, digest = future.result()
            except (
                DataIntegrityError,
                OSError,
                TimeoutError,
                urllib.error.HTTPError,
                urllib.error.URLError,
                ConnectionError,
            ) as error:
                failures.append(error)
            else:
                hashes[filename] = digest
    if failures:
        raise DataIntegrityError(
            f"{len(failures)} of {len(items)} COCO image downloads failed; "
            f"first error: {failures[0]}"
        ) from failures[0]
    if len(hashes) != len(items):
        raise DataIntegrityError(
            f"Downloaded {len(hashes)} COCO images; expected {len(items)}."
        )

    captions_count = sum(len(item["captions"]) for item in items)
    annotation_sha = sha256_file(annotation_path)
    item_sha = content_hash(sorted(hashes.items()))
    record = {
        "dataset": "mscoco_captions",
        "version": "Karpathy caption split from Stanford caption_datasets.zip",
        "download_sources": {
            "annotations": KARPATHY_ANNOTATIONS_URL,
            "images": COCO_IMAGE_BASE_URL,
        },
        "license": spec.license,
        "split": spec.split,
        "expected_sample_count": spec.expected_images,
        "actual_sample_count": len(hashes),
        "text_count": captions_count,
        "annotation_source_sha256": annotation_sha,
        "annotation_archive_sha256": sha256_file(annotation_archive),
        "image_files_sha256": item_sha,
        "dataset_checksum": content_hash(
            {
                "annotation_sha256": annotation_sha,
                "image_files_sha256": item_sha,
                "split": spec.split,
            }
        ),
        "data_integrity_status": "verified; upstream per-image checksums not published",
        "missing_or_excluded_samples": [],
    }
    atomic_write_json(evaluation_items_path, {"items": items})
    atomic_write_json(marker_path, record)
    return record


def retrieval_items(
    dataset_key: str,
    language: str,
    data_root: Path = DATA_ROOT,
) -> list[dict[str, Any]]:
    spec = get_dataset(dataset_key)
    if spec.access_status == "blocked" or spec.license_status == "blocked":
        raise DatasetUnavailableError(spec.access_reason or f"{dataset_key} is blocked.")
    if dataset_key == "crossmodal3600":
        marker_path = data_root / dataset_key / ".vlbench-dataset.json"
        if not marker_path.is_file():
            raise DatasetUnavailableError(
                "XM3600 is not fully prepared; run `vlbench datasets prepare "
                "--dataset crossmodal3600` first."
            )
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            not str(marker.get("data_integrity_status", "")).startswith("verified")
            or marker.get("actual_sample_count") != spec.expected_images
            or (
                spec.license_status == "requires_user_confirmation"
                and marker.get("dataset_terms_confirmed") is not True
            )
        ):
            raise DataIntegrityError("XM3600 preparation metadata is not verified.")
        code = spec.source_language_code(language)
        path = data_root / dataset_key / f"annotations-{code}.jsonl"
        if not path.is_file():
            raise DatasetUnavailableError(
                "XM3600 is not prepared; run `vlbench datasets prepare "
                "--dataset crossmodal3600` first."
            )
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(records) != spec.expected_texts:
            raise DataIntegrityError(
                f"XM3600 {language} annotations contain {len(records)} texts; "
                f"expected {spec.expected_texts}."
            )
        return [
            {
                "image_id": item["image_id"],
                "image_path": data_root / dataset_key / "images" / item["image_path"],
                "captions": [item["caption"]],
            }
            for item in records
        ]
    if dataset_key == "mscoco_captions":
        dataset_root = data_root / dataset_key
        marker_path = dataset_root / ".vlbench-dataset.json"
        path = dataset_root / "evaluation-items.json"
        if not marker_path.is_file() or not path.is_file():
            raise DatasetUnavailableError(
                "MS COCO Karpathy test data is not prepared; run `vlbench datasets prepare "
                "--dataset mscoco_captions` first."
            )
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            not str(marker.get("data_integrity_status", "")).startswith("verified")
            or marker.get("actual_sample_count") != get_dataset(dataset_key).expected_images
            or (
                get_dataset(dataset_key).license_status == "requires_user_confirmation"
                and marker.get("dataset_terms_confirmed") is not True
            )
        ):
            raise DataIntegrityError("MS COCO preparation metadata is not verified.")
        items = json.loads(path.read_text(encoding="utf-8"))["items"]
        if len(items) != get_dataset(dataset_key).expected_images:
            raise DataIntegrityError(
                f"MS COCO item manifest contains {len(items)} images; "
                f"expected {get_dataset(dataset_key).expected_images}."
            )
        return [
            {
                "image_id": item["image_id"],
                "image_path": data_root / dataset_key / "images" / item["image_path"],
                "captions": item["captions"],
            }
            for item in items
        ]
    raise ValueError(f"No retrieval loader is implemented for dataset {dataset_key!r}.")


def prepare_dataset(
    dataset_key: str,
    data_root: Path = DATA_ROOT,
    *,
    progress: Callable[[str], None] | None = None,
    confirm_data_terms: bool = False,
) -> dict[str, Any]:
    spec = get_dataset(dataset_key)
    _require_dataset_access(spec, confirm_data_terms)
    if dataset_key in CIFAR_ARCHIVES:
        return prepare_cifar(
            dataset_key,
            data_root,
            confirm_data_terms=confirm_data_terms,
        )
    if dataset_key == "crossmodal3600":
        return prepare_xm3600(
            data_root,
            progress=progress,
            confirm_data_terms=confirm_data_terms,
        )
    if dataset_key == "mscoco_captions":
        return prepare_coco(
            data_root,
            progress=progress,
            confirm_data_terms=confirm_data_terms,
        )
    raise DatasetUnavailableError(f"No acquisition method is registered for {dataset_key!r}.")
