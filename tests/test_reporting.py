from __future__ import annotations

import csv
import json

import jsonschema
import pytest

from tests.helpers import make_result
from vision_language_benchmarks.atomic import atomic_write_json
from vision_language_benchmarks.reporting.aggregate import generate_reports
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

    with files["summary_csv"].open(encoding="utf-8", newline="") as stream:
        summary = list(csv.DictReader(stream))
    with files["non_success_csv"].open(encoding="utf-8", newline="") as stream:
        unsuccessful = list(csv.DictReader(stream))
    metadata = json.loads(files["metadata_json"].read_text(encoding="utf-8"))

    assert len(summary) == 3
    multilingual = next(row for row in summary if row["model_key"] == "openclip_multilingual_b32")
    assert multilingual["model_checkpoint_size_bytes"] == "1464635263"
    assert multilingual["model_languages"] == "en, ja"
    assert [row["status"] for row in unsuccessful] == ["blocked"]
    xm3600 = next(
        dataset
        for dataset in metadata["datasets"]
        if dataset["dataset_id"] == "crossmodal3600"
    )
    assert {item["confirmed"] for item in xm3600["terms_confirmations"]} == {False, True}
    assert "No values are estimated or fabricated." in files["report_markdown"].read_text(
        encoding="utf-8"
    )
    assert (output_dir / "plots" / "classification-top1-p0-fp32-en.png").is_file()
    assert (output_dir / "plots" / "retrieval-recall-p2-fp32-ja.svg").is_file()
    assert (
        output_dir / "plots" / "english-japanese-recall-paired-p2-fp32.png"
    ).is_file()
    assert (output_dir / "plots" / "accuracy-throughput-p0-fp32-en.svg").is_file()
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
