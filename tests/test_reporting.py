from __future__ import annotations

import csv
import json

import jsonschema
import pytest

from tests.helpers import make_result
from vision_language_benchmarks.atomic import atomic_write_json
from vision_language_benchmarks.reporting.aggregate import generate_reports
from vision_language_benchmarks.reporting.comparison import paired_language_results
from vision_language_benchmarks.reporting.schema import validate_result


def test_result_status_and_measured_metrics_must_agree() -> None:
    blocked = make_result(status="blocked")
    validate_result(blocked)
    blocked["metrics"] = {"acc1": 0.0, "acc5": 0.0}

    with pytest.raises(jsonschema.ValidationError):
        validate_result(blocked)


def test_json_results_regenerate_csv_markdown_metadata_and_all_chart_types(tmp_path) -> None:
    results = [
        make_result(run_id="cifar-classification"),
        make_result(protocol="p1", run_id="cifar-classification-p1"),
        make_result(
            model_key="openclip_multilingual_b32",
            dataset_key="crossmodal3600",
            language="en",
            protocol="p2",
            run_id="xm3600-en",
            image_id_order_hash="same-image-order",
        ),
        make_result(
            model_key="openclip_multilingual_b32",
            dataset_key="crossmodal3600",
            language="ja",
            protocol="p2",
            run_id="xm3600-ja",
            image_id_order_hash="same-image-order",
        ),
        make_result(
            model_key="openclip_multilingual_b32",
            dataset_key="crossmodal3600",
            language="en",
            protocol="p2",
            status="blocked",
            run_id="xm3600-blocked",
            data_terms_confirmed=False,
        ),
    ]
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    for result in results:
        atomic_write_json(
            input_dir / f"{result['benchmark_metadata']['run_id']}.json",
            result,
        )

    output_dir = tmp_path / "reports"
    files = generate_reports(input_dir, output_dir)
    for path in files.values():
        if path.suffix == ".svg":
            assert all(
                line == line.rstrip()
                for line in path.read_text(encoding="utf-8").splitlines()
            )

    with files["summary_csv"].open(encoding="utf-8", newline="") as stream:
        summary = list(csv.DictReader(stream))
    with files["non_success_csv"].open(encoding="utf-8", newline="") as stream:
        unsuccessful = list(csv.DictReader(stream))
    metadata = json.loads(files["metadata_json"].read_text(encoding="utf-8"))

    assert len(summary) == 4
    multilingual = next(row for row in summary if row["model_key"] == "openclip_multilingual_b32")
    assert multilingual["model_checkpoint_size_bytes"] == "1464635263"
    assert multilingual["model_languages"] == "en, ja"
    assert multilingual["image_count"] == "3600"
    assert multilingual["text_count"] == "7200"
    assert multilingual["batch_size_requested"] == "2"
    assert multilingual["batch_size_actual"] == "2"
    assert multilingual["image_batch_size_actual"] == "2"
    assert multilingual["text_batch_size_actual"] == "2"
    assert multilingual["oom_retry_count"] == "0"
    assert multilingual["image_retrieval_recall@10_fraction"] == "0.9"
    assert [row["status"] for row in unsuccessful] == ["blocked"]
    xm3600 = next(
        dataset
        for dataset in metadata["datasets"]
        if dataset["dataset_id"] == "crossmodal3600"
    )
    assert {item["confirmed"] for item in xm3600["terms_confirmations"]} == {False}
    report = files["report_markdown"].read_text(encoding="utf-8")
    assert "No values are estimated or fabricated." in report
    assert "Top-1 | Top-5" in report
    assert "T→I R@10" in report
    assert "Requested / image / text batch" in report
    assert "OOM retries" in report
    assert "2 / 2 / 2" in report
    assert "T→I R@1 (EN → JA; Δ pp)" in report
    assert "[xm3600-en](raw/xm3600-en.json)" in report
    assert (output_dir / "plots" / "classification_accuracy.png").is_file()
    assert (output_dir / "plots" / "classification_accuracy_p1.svg").is_file()
    assert (output_dir / "plots" / "retrieval_recall.png").is_file()
    assert (output_dir / "plots" / "multilingual_comparison.svg").is_file()
    assert (output_dir / "plots" / "performance_comparison.png").is_file()
    assert (output_dir / "plots" / "performance_comparison_p1.svg").is_file()
    first_generation = {
        name: path.read_bytes()
        for name, path in files.items()
    }

    regenerated_files = generate_reports(input_dir, output_dir)
    second_generation = {
        name: path.read_bytes()
        for name, path in regenerated_files.items()
    }

    assert second_generation == first_generation


def test_paired_language_results_require_matching_data_and_accept_cuda_aliases() -> None:
    english = make_result(
        model_key="openclip_multilingual_b32",
        dataset_key="crossmodal3600",
        language="en",
        protocol="p2",
        run_id="paired-en",
    )
    japanese = make_result(
        model_key="openclip_multilingual_b32",
        dataset_key="crossmodal3600",
        language="ja",
        protocol="p2",
        run_id="paired-ja",
    )
    english_metadata = english["benchmark_metadata"]
    japanese_metadata = japanese["benchmark_metadata"]
    english_metadata.update(
        {
            "device": "cuda",
            "gpu_name": "NVIDIA GeForce RTX 4070 Ti SUPER",
            "gpu_vram_bytes": 17_170_956_288,
        }
    )
    japanese_metadata.update(
        {
            "device": "cuda:0",
            "gpu_name": "NVIDIA GeForce RTX 4070 Ti SUPER",
            "gpu_vram_bytes": 17_170_956_288,
        }
    )

    assert paired_language_results([english, japanese]) == [(english, japanese)]

    japanese_metadata["dataset_checksum"] = "f" * 64
    assert paired_language_results([english, japanese]) == []


def test_report_generator_does_not_create_empty_chart_placeholders(tmp_path) -> None:
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    result = make_result(status="blocked", run_id="blocked-only")
    atomic_write_json(input_dir / "blocked-only.json", result)

    files = generate_reports(input_dir, tmp_path / "reports")

    assert not any(key.endswith("_png") or key.endswith("_svg") for key in files)
    assert not (tmp_path / "reports" / "plots").exists()
