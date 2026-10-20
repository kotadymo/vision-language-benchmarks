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
from vision_language_benchmarks.errors import DataIntegrityError
from vision_language_benchmarks.reporting.aggregate import generate_reports
from vision_language_benchmarks.reporting.comparison import paired_language_results
from vision_language_benchmarks.reporting.lifecycle import build_result_lifecycle
from vision_language_benchmarks.reporting.reproducibility_audit import (
    build_clean_official_xm3600_parity,
    build_clean_xm3600_reproducibility_audit,
)
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


def _mark_strict_clean_source(result: dict, source_commit_sha: str) -> dict:
    metadata = result["benchmark_metadata"]
    metadata["source_commit_sha"] = source_commit_sha
    metadata["working_tree_dirty"] = False
    metadata["reproducibility_mode"] = "strict"
    metadata["expected_source_commit_sha"] = source_commit_sha
    metadata["staging_directory"] = ".cache/benchmark-staging/test-suite"
    metadata["run_configuration"].update(
        {
            "reproducibility_mode": "strict",
            "expected_source_commit_sha": source_commit_sha,
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


def test_publishable_report_excludes_superseded_results_and_labels_active_runs(
    tmp_path,
) -> None:
    frozen_source_commit = "a" * 40
    active = _mark_strict_clean_source(
        make_result(run_id="active-clean"), frozen_source_commit
    )
    historical = make_result(run_id="historical")
    superseded = _mark_official_first_caption(
        make_result(
            model_key="openclip_multilingual_b32",
            dataset_key="crossmodal3600",
            language="en",
            protocol="p2",
            run_id="superseded-dirty",
        )
    )
    superseded["benchmark_metadata"]["working_tree_dirty"] = True
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    for result in (active, historical, superseded):
        atomic_write_json(
            input_dir / f"{result['benchmark_metadata']['run_id']}.json",
            result,
        )

    output_dir = tmp_path / "reports"
    lifecycle_path = output_dir / "validation" / "result-lifecycle.json"
    atomic_write_json(
        lifecycle_path,
        {
            "schema_version": "1.0.0",
            "frozen_source_commit_sha": frozen_source_commit,
            "entries": [
                {
                    "run_id": "active-clean",
                    "state": "active",
                    "leaderboard_eligible": True,
                },
                {
                    "run_id": "historical",
                    "state": "historical",
                    "leaderboard_eligible": True,
                },
                {
                    "run_id": "superseded-dirty",
                    "state": "superseded",
                    "leaderboard_eligible": False,
                },
            ],
            "attempts": [],
        },
    )

    files = generate_reports(
        input_dir,
        output_dir,
        publishable=True,
        lifecycle_path=lifecycle_path,
        expected_source_commit_sha=frozen_source_commit,
    )

    with files["summary_csv"].open(encoding="utf-8", newline="") as stream:
        summary = list(csv.DictReader(stream))
    metadata = json.loads(files["metadata_json"].read_text(encoding="utf-8"))
    report = files["report_markdown"].read_text(encoding="utf-8")

    assert {row["run_id"]: row["publication_state"] for row in summary} == {
        "active-clean": "active",
        "historical": "historical",
    }
    assert metadata["result_file_count"] == 2
    assert metadata["publication_lifecycle"]["state_counts"]["superseded"] == 1
    assert "Superseded measurements excluded from primary results: 1" in report
    assert "superseded-dirty" not in report


def test_publishable_report_rejects_dirty_leaderboard_result(tmp_path) -> None:
    result = make_result(run_id="dirty-active")
    result["benchmark_metadata"]["working_tree_dirty"] = True
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    atomic_write_json(input_dir / "dirty-active.json", result)

    frozen_source_commit = "a" * 40
    lifecycle_path = tmp_path / "result-lifecycle.json"
    atomic_write_json(
        lifecycle_path,
        {
            "schema_version": "1.0.0",
            "frozen_source_commit_sha": frozen_source_commit,
            "entries": [
                {
                    "run_id": "dirty-active",
                    "state": "active",
                    "leaderboard_eligible": True,
                }
            ],
            "attempts": [],
        },
    )

    with pytest.raises(DataIntegrityError, match="dirty or unknown worktree"):
        generate_reports(
            input_dir,
            tmp_path / "reports",
            publishable=True,
            lifecycle_path=lifecycle_path,
            expected_source_commit_sha=frozen_source_commit,
        )


def test_result_lifecycle_marks_clean_replacements_and_preserves_old_records(
    tmp_path,
) -> None:
    frozen_source_commit = "a" * 40
    active = _mark_strict_clean_source(
        _mark_official_first_caption(
            make_result(
                model_key="openclip_multilingual_b32",
                dataset_key="crossmodal3600",
                language="en",
                protocol="p2",
                run_id="replacement-official",
            )
        ),
        frozen_source_commit,
    )
    superseded = _mark_official_first_caption(
        make_result(
            model_key="openclip_multilingual_b32",
            dataset_key="crossmodal3600",
            language="en",
            protocol="p2",
            run_id="previous-official",
        )
    )
    superseded["benchmark_metadata"]["working_tree_dirty"] = True
    historical = make_result(run_id="historical-cifar")
    original_superseded = json.loads(json.dumps(superseded))

    lifecycle = build_result_lifecycle(
        [active, superseded, historical],
        frozen_source_commit,
    )
    states = {entry["run_id"]: entry for entry in lifecycle["entries"]}

    assert states["replacement-official"]["state"] == "active"
    assert states["replacement-official"]["leaderboard_eligible"] is True
    assert states["previous-official"]["state"] == "superseded"
    assert states["previous-official"]["leaderboard_eligible"] is False
    assert states["historical-cifar"]["state"] == "historical"
    assert superseded == original_superseded


def test_result_lifecycle_keeps_failed_and_incomplete_attempts(tmp_path) -> None:
    result = make_result(run_id="completed-run")
    manifest_path = tmp_path / "run-manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": "1.0.0",
            "runs": {
                "failed-run": {
                    "status": "failed",
                    "reason_if_not_success": "Synthetic failure.",
                },
                "running-run": {"status": "running"},
                "completed-run": {"status": "completed"},
            },
        },
    )

    lifecycle = build_result_lifecycle(
        [result],
        "a" * 40,
        (manifest_path,),
    )

    attempts = {attempt["run_id"]: attempt for attempt in lifecycle["attempts"]}
    assert attempts["failed-run"]["state"] == "failed"
    assert attempts["running-run"]["state"] == "incomplete"
    assert "completed-run" not in attempts
    assert lifecycle["state_counts"] == {
        "failed": 1,
        "historical": 1,
        "incomplete": 1,
    }


def test_clean_xm3600_audit_reports_previous_and_replacement_metrics() -> None:
    frozen_source_commit = "f" * 40
    expected_runs = (
        ("embeddinggemma2", "en"),
        ("embeddinggemma2", "ja"),
        ("openclip_multilingual_b32", "en"),
        ("openclip_multilingual_b32", "ja"),
        ("siglip2_base_224", "en"),
        ("siglip2_base_224", "ja"),
        ("rinna_ja_clip_b16", "ja"),
    )
    results = []
    states = {}
    for index, (model_key, language) in enumerate(expected_runs):
        previous_id = f"previous-{model_key}-{language}"
        replacement_id = f"replacement-{model_key}-{language}"
        previous = _mark_official_first_caption(
            make_result(
                model_key=model_key,
                dataset_key="crossmodal3600",
                language=language,
                protocol="p2",
                run_id=previous_id,
            )
        )
        previous["benchmark_metadata"]["source_commit_sha"] = "b" * 40
        previous["benchmark_metadata"]["working_tree_dirty"] = index != 0
        replacement = _mark_strict_clean_source(
            _mark_official_first_caption(
                make_result(
                    model_key=model_key,
                    dataset_key="crossmodal3600",
                    language=language,
                    protocol="p2",
                    run_id=replacement_id,
                )
            ),
            frozen_source_commit,
        )
        results.extend((previous, replacement))
        states[previous_id] = "superseded"
        states[replacement_id] = "active"

    audit = build_clean_xm3600_reproducibility_audit(
        results,
        states,
        frozen_source_commit,
    )

    assert audit["previous_runs"]["count"] == 7
    assert audit["previous_runs"]["clean_worktree_count"] == 1
    assert audit["previous_runs"]["dirty_worktree_count"] == 6
    assert audit["replacement_runs"]["count"] == 7
    assert audit["replacement_runs"]["clean_worktree_count"] == 7
    assert audit["replacement_runs"]["source_commit_shas"] == [frozen_source_commit]
    assert audit["maximum_absolute_recall_difference"] == 0.0
    assert all(
        all(pair["checks"].values())
        and all(
            metric["absolute_difference"] == 0.0
            for metric in pair["metrics"].values()
        )
        for pair in audit["pairs"]
    )
    assert all(
        all(pair["checks"].values())
        for pair in audit["language_pairs"].values()
    )


def test_clean_official_parity_record_uses_saved_reference_and_clean_run() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    validation_dir = repository_root / "results" / "v0.1.0" / "validation"
    parity = json.loads(
        (validation_dir / "official-xm3600-parity.json").read_text(encoding="utf-8")
    )
    official_reference = json.loads(
        (validation_dir / "official-xm3600-openclip-ja.json").read_text(
            encoding="utf-8"
        )
    )
    clean_result = json.loads(
        (
            repository_root
            / "results"
            / "v0.1.0"
            / "raw"
            / f"{parity['vlbench']['run_id']}.json"
        ).read_text(encoding="utf-8")
    )
    frozen_source_commit = "f" * 40
    metadata = clean_result["benchmark_metadata"]
    metadata["run_id"] = "synthetic-clean-openclip-ja"
    metadata["source_commit_sha"] = frozen_source_commit
    metadata["working_tree_dirty"] = False
    metadata["reproducibility_mode"] = "strict"
    metadata["expected_source_commit_sha"] = frozen_source_commit
    metadata["staging_directory"] = ".cache/benchmark-staging/test-suite"
    metadata["run_configuration"].update(
        {
            "reproducibility_mode": "strict",
            "expected_source_commit_sha": frozen_source_commit,
        }
    )

    clean_parity = build_clean_official_xm3600_parity(
        parity,
        official_reference,
        clean_result,
    )

    assert clean_parity["status"] == "completed"
    assert clean_parity["vlbench"]["run_id"] == "synthetic-clean-openclip-ja"
    assert clean_parity["vlbench"]["source_commit"] == frozen_source_commit
    assert clean_parity["reference"]["result_reused"] is True
    assert clean_parity["comparison"]["within_tolerance"] is True
    assert clean_parity["comparison"]["same_integer_hit_counts"] is True
