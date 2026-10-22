from __future__ import annotations

import json

from tests.helpers import make_result
from vision_language_benchmarks.atomic import atomic_write_json
from vision_language_benchmarks.cli import main
from vision_language_benchmarks.errors import DataIntegrityError


def test_dry_run_displays_pinned_model_size_and_verified_access_without_downloads(
    capsys,
    tmp_path,
) -> None:
    output_path = tmp_path / "results"

    exit_code = main(
        [
            "run",
            "--model",
            "openclip_multilingual_b32",
            "--dataset",
            "crossmodal3600",
            "--protocol",
            "p2",
            "--language",
            "ja",
            "--retrieval-variant",
            "official_first_caption",
            "--device",
            "cpu",
            "--output",
            str(output_path),
            "--dry-run",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["model"]["checkpoint_size_bytes"] == 1_464_635_263
    assert report["model"]["checkpoint_filename"] == "open_clip_pytorch_model.bin"
    assert report["model"]["supported_languages"] == ["en", "ja"]
    assert report["dataset"]["access"] == "available"
    assert report["dataset"]["license_status"] == "verified_attribution_required"
    assert report["dataset"]["expected_text_counts_by_language"] == {
        "en": 7_200,
        "ja": 7_185,
    }
    assert report["dataset"]["expected_text_candidates"] == 3_600
    assert report["evaluation"]["retrieval_variant"] == "official_first_caption"
    assert not output_path.exists()


def test_dry_run_exposes_embeddinggemma_dimension(capsys, tmp_path) -> None:
    exit_code = main(
        [
            "run",
            "--model",
            "embeddinggemma2",
            "--dataset",
            "cifar10",
            "--embedding-dimension",
            "128",
            "--device",
            "cpu",
            "--output",
            str(tmp_path / "staging"),
            "--dry-run",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["evaluation"]["embedding_dimension"] == 128


def test_dry_run_rejects_dimensions_for_non_embeddinggemma_models(
    capsys,
    tmp_path,
) -> None:
    exit_code = main(
        [
            "run",
            "--model",
            "openai_clip_b32",
            "--dataset",
            "cifar10",
            "--embedding-dimension",
            "128",
            "--output",
            str(tmp_path / "staging"),
            "--dry-run",
        ]
    )

    assert exit_code == 1
    assert "only for the embeddinggemma2 model" in capsys.readouterr().err


def test_report_accepts_historical_baseline_inputs(monkeypatch, capsys, tmp_path) -> None:
    received: dict[str, object] = {}

    def fake_generate_reports(input_path, output_dir, **kwargs):
        received.update(
            {
                "input_path": input_path,
                "output_dir": output_dir,
                **kwargs,
            }
        )
        return {}

    monkeypatch.setattr(
        "vision_language_benchmarks.cli.generate_reports",
        fake_generate_reports,
    )
    current_input = tmp_path / "current-raw"
    current_output = tmp_path / "current-reports"
    current_lifecycle = tmp_path / "current-lifecycle.json"
    baseline_input = tmp_path / "baseline-raw"
    baseline_lifecycle = tmp_path / "baseline-lifecycle.json"
    comparison_input = tmp_path / "comparison-raw"
    comparison_lifecycle = tmp_path / "comparison-lifecycle.json"

    exit_code = main(
        [
            "report",
            "--input",
            str(current_input),
            "--output",
            str(current_output),
            "--publishable",
            "--lifecycle",
            str(current_lifecycle),
            "--baseline-input",
            str(baseline_input),
            "--baseline-lifecycle",
            str(baseline_lifecycle),
            "--comparison-input",
            str(comparison_input),
            "--comparison-lifecycle",
            str(comparison_lifecycle),
        ]
    )
    capsys.readouterr()

    assert exit_code == 0
    assert received == {
        "input_path": current_input,
        "output_dir": current_output,
        "publishable": True,
        "lifecycle_path": current_lifecycle,
        "expected_source_commit_sha": None,
        "baseline_input_path": baseline_input,
        "baseline_lifecycle_path": baseline_lifecycle,
        "comparison_input_paths": (comparison_input,),
        "comparison_lifecycle_paths": (comparison_lifecycle,),
    }


def test_data_terms_confirmation_does_not_override_verified_access(capsys, tmp_path) -> None:
    exit_code = main(
        [
            "run",
            "--model",
            "embeddinggemma2",
            "--dataset",
            "crossmodal3600",
            "--protocol",
            "p2",
            "--language",
            "ja",
            "--device",
            "cpu",
            "--output",
            str(tmp_path / "results"),
            "--confirm-data-terms",
            "--dry-run",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["dataset"]["access"] == "available"
    assert report["dataset"]["license_status"] == "verified_attribution_required"
    assert report["evaluation"]["retrieval_variant"] == "all_captions"


def test_force_does_not_bypass_strict_clean_source_preflight(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    download_authorized = False

    def reject_dirty_source(*args, **kwargs) -> None:
        raise DataIntegrityError("tracked source changes are present")

    def authorize_downloads(*args, **kwargs) -> bool:
        nonlocal download_authorized
        download_authorized = True
        return True

    monkeypatch.setattr(
        "vision_language_benchmarks.cli.require_clean_source_preflight",
        reject_dirty_source,
    )
    monkeypatch.setattr(
        "vision_language_benchmarks.cli._authorize_downloads",
        authorize_downloads,
    )

    exit_code = main(
        [
            "run",
            "--model",
            "openclip_multilingual_b32",
            "--dataset",
            "crossmodal3600",
            "--protocol",
            "p2",
            "--language",
            "ja",
            "--retrieval-variant",
            "official_first_caption",
            "--output",
            str(tmp_path / "staging"),
            "--strict-reproducibility",
            "--expected-source-commit",
            "a" * 40,
            "--force",
            "--yes",
        ]
    )
    error = capsys.readouterr().err

    assert exit_code == 1
    assert "tracked source changes are present" in error
    assert download_authorized is False


def test_doctor_reproducibility_validates_requested_cuda_configuration(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        "vision_language_benchmarks.cli.reproducibility_report",
        lambda *args, **kwargs: {"ok": True, "errors": []},
    )
    monkeypatch.setattr(
        "vision_language_benchmarks.cli.doctor_report",
        lambda device: {"ok": True, "torch": {"cuda_available": True}},
    )

    exit_code = main(
        [
            "doctor",
            "--reproducibility",
            "--model",
            "openclip_multilingual_b32",
            "--dataset",
            "crossmodal3600",
            "--protocol",
            "p2",
            "--language",
            "ja",
            "--retrieval-variant",
            "official_first_caption",
            "--output",
            str(tmp_path / "staging"),
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["ok"] is True
    assert report["requested_configuration"]["ok"] is True
    assert report["requested_configuration"]["evaluation"]["retrieval_variant"] == (
        "official_first_caption"
    )
    assert report["cuda_required"] is True
    assert report["cuda_available"] is True


def test_lifecycle_command_writes_publication_state_manifest(
    capsys,
    tmp_path,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    result = make_result(run_id="historical-cifar")
    atomic_write_json(raw_dir / "historical-cifar.json", result)
    lifecycle_path = tmp_path / "validation" / "result-lifecycle.json"

    exit_code = main(
        [
            "lifecycle",
            "--input",
            str(raw_dir),
            "--frozen-source-commit",
            "a" * 40,
            "--output",
            str(lifecycle_path),
        ]
    )
    output = json.loads(capsys.readouterr().out)
    manifest = json.loads(lifecycle_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert output["result_count"] == 1
    assert output["state_counts"] == {"historical": 1}
    assert manifest["entries"][0]["state"] == "historical"
