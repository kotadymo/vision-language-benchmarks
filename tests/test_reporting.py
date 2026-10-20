from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import jsonschema
import pytest

from tests.helpers import make_result
from vision_language_benchmarks.atomic import atomic_write_json
from vision_language_benchmarks.reporting.aggregate import generate_reports
from vision_language_benchmarks.reporting.comparison import paired_language_results
from vision_language_benchmarks.reporting.schema import validate_result


def _mark_official_first_caption(result: dict) -> dict:
    metadata = result["benchmark_metadata"]
    image_ids = [f"image-{index}" for index in range(3_600)]
    metadata["run_configuration"]["retrieval_variant"] = "official_first_caption"
    metadata.update(
        {
            "retrieval_variant": "official_first_caption",
            "retrieval_variant_source": "run_configuration",
            "retrieval_variant_reference_commit": (
                "ee14a9a8a9bcde09d51f4a4b1dff5a75182a3596"
            ),
            "one_caption_per_image": True,
            "candidate_counts": {"images": 3_600, "texts": 3_600},
            "image_count": 3_600,
            "text_count": 3_600,
            "positive_pair_count": 3_600,
            "image_ids": image_ids,
            "text_ids": [f"{image_id}#0" for image_id in image_ids],
            "image_id_order_hash": "a" * 64,
            "text_id_order_hash": "b" * 64,
            "text_content_sha256": ("c" if result["language"] == "en" else "d") * 64,
        }
    )
    return result


def test_result_status_and_measured_metrics_must_agree() -> None:
    blocked = make_result(status="blocked")
    validate_result(blocked)
    blocked["metrics"] = {"acc1": 0.0, "acc5": 0.0}

    with pytest.raises(jsonschema.ValidationError):
        validate_result(blocked)


def test_historical_results_match_audit_hashes_and_keep_all_caption_assignment() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    audit_path = (
        repository_root
        / "results"
        / "v0.1.0"
        / "validation"
        / "historical-results-audit.json"
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    assert audit["historical_result_count"] == 31
    assert audit["all_historical_raw_files_unchanged"] is True
    for record in audit["raw_results"]:
        raw_path = repository_root / "results" / "v0.1.0" / record["file"]
        raw_bytes = raw_path.read_bytes()
        assert hashlib.sha256(raw_bytes).hexdigest() == record["sha256"]
        validate_result(json.loads(raw_bytes))

    assignment = audit["historical_xm3600_variant_assignment"]
    assert assignment["retrieval_variant"] == "all_captions"
    assert assignment["assignment_source"] == "legacy_candidate_counts"
    assert assignment["raw_results_modified_or_relabelled"] is False
    assert len(assignment["runs"]) == 7
    assert Counter(run["language"] for run in assignment["runs"]) == {
        "en": 3,
        "ja": 4,
    }
    assert all(
        run["assigned_variant"] == "all_captions"
        and run["assignment_source"] == "legacy_candidate_counts"
        and run["raw_metadata_precedes_variant_field"] is True
        and run["image_count"] == 3_600
        and run["text_count"] == (7_200 if run["language"] == "en" else 7_185)
        for run in assignment["runs"]
    )


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
        ),
        make_result(
            model_key="openclip_multilingual_b32",
            dataset_key="crossmodal3600",
            language="ja",
            protocol="p2",
            run_id="xm3600-ja",
        ),
        make_result(
            model_key="openclip_multilingual_b32",
            dataset_key="crossmodal3600",
            language="en",
            protocol="p2",
            run_id="xm3600-official-en",
        ),
        make_result(
            model_key="openclip_multilingual_b32",
            dataset_key="crossmodal3600",
            language="ja",
            protocol="p2",
            run_id="xm3600-official-ja",
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
    results[4] = _mark_official_first_caption(results[4])
    results[5] = _mark_official_first_caption(results[5])
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    for result in results:
        atomic_write_json(
            input_dir / f"{result['benchmark_metadata']['run_id']}.json",
            result,
        )

    output_dir = tmp_path / "reports"
    validation_dir = output_dir / "validation"
    validation_dir.mkdir(parents=True)
    for filename in (
        "historical-results-audit.json",
        "official-cifar10-parity.json",
        "official-xm3600-parity.json",
    ):
        atomic_write_json(validation_dir / filename, {"status": "completed"})
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

    assert len(summary) == 6
    multilingual = next(
        row
        for row in summary
        if row["model_key"] == "openclip_multilingual_b32"
        and row["retrieval_variant"] == "all_captions"
        and row["language"] == "en"
    )
    assert multilingual["model_checkpoint_size_bytes"] == "1464635263"
    assert multilingual["model_languages"] == "en, ja"
    assert multilingual["image_count"] == "3600"
    assert multilingual["text_count"] == "7200"
    assert multilingual["retrieval_variant_source"] == "legacy_candidate_counts"
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
    assert "English zero-shot classification" in report
    assert "Japanese/English official-compatible XM3600 retrieval" in report
    assert "XM3600 all-caption retrieval (`all_captions`)" in report
    assert "T→I R@10" in report
    assert "Requested / image / text batch" in report
    assert "OOM retries" in report
    assert "2 / 2 / 2" in report
    assert "T→I R@1 (EN → JA; Δ pp)" in report
    assert "[xm3600-official-en](raw/xm3600-official-en.json)" in report
    assert "[xm3600-en](raw/xm3600-en.json)" in report
    assert "## Validation and reproducibility" in report
    assert "[official-xm3600-parity.json](validation/official-xm3600-parity.json)" in report
    assert "## Known limitations" in report
    assert (output_dir / "plots" / "classification_accuracy.png").is_file()
    assert (output_dir / "plots" / "classification_accuracy_p1.svg").is_file()
    assert (output_dir / "plots" / "xm3600_official_retrieval.png").is_file()
    assert (output_dir / "plots" / "xm3600_all_captions_retrieval.svg").is_file()
    assert (
        output_dir
        / "plots"
        / "xm3600_language_comparison_official_first_caption.svg"
    ).is_file()
    official_chart = (
        output_dir / "plots" / "xm3600_official_retrieval.svg"
    ).read_text(encoding="utf-8")
    all_captions_chart = (
        output_dir / "plots" / "xm3600_all_captions_retrieval.svg"
    ).read_text(encoding="utf-8")
    comparison_chart = (
        output_dir
        / "plots"
        / "xm3600_language_comparison_official_first_caption.svg"
    ).read_text(encoding="utf-8")
    assert "Bidirectional Recall@1/5/10" in official_chart
    assert "Crossmodal-3600 (XM3600)" in official_chart
    assert "P2 / FP32" in official_chart
    assert "variant: official_first_caption" in official_chart
    assert "EN" in official_chart and "JA" in official_chart
    assert "variant: all_captions" in all_captions_chart
    assert "Crossmodal-3600 (XM3600)" in all_captions_chart
    assert "P2 / FP32" in all_captions_chart
    assert "Japanese − English recall (percentage points)" in comparison_chart
    assert "official_first_caption" in comparison_chart
    assert "3,600 candidates per language" in comparison_chart
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
    english = _mark_official_first_caption(
        make_result(
            model_key="openclip_multilingual_b32",
            dataset_key="crossmodal3600",
            language="en",
            protocol="p2",
            run_id="paired-en",
        )
    )
    japanese = _mark_official_first_caption(
        make_result(
            model_key="openclip_multilingual_b32",
            dataset_key="crossmodal3600",
            language="ja",
            protocol="p2",
            run_id="paired-ja",
        )
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


def test_historical_all_caption_runs_with_unequal_candidates_are_not_paired() -> None:
    english = make_result(
        model_key="openclip_multilingual_b32",
        dataset_key="crossmodal3600",
        language="en",
        protocol="p2",
        run_id="all-captions-en",
    )
    japanese = make_result(
        model_key="openclip_multilingual_b32",
        dataset_key="crossmodal3600",
        language="ja",
        protocol="p2",
        run_id="all-captions-ja",
    )

    assert paired_language_results([english, japanese]) == []


def test_result_schema_rejects_inconsistent_official_retrieval_metadata() -> None:
    result = _mark_official_first_caption(
        make_result(
            model_key="openclip_multilingual_b32",
            dataset_key="crossmodal3600",
            language="en",
            protocol="p2",
            run_id="official-schema",
        )
    )
    validate_result(result)

    result["benchmark_metadata"]["one_caption_per_image"] = False

    with pytest.raises(ValueError, match="one caption per image"):
        validate_result(result)


def test_report_generator_does_not_create_empty_chart_placeholders(tmp_path) -> None:
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    result = make_result(status="blocked", run_id="blocked-only")
    atomic_write_json(input_dir / "blocked-only.json", result)

    files = generate_reports(input_dir, tmp_path / "reports")

    assert not any(key.endswith("_png") or key.endswith("_svg") for key in files)
    assert not (tmp_path / "reports" / "plots").exists()
