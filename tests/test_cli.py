from __future__ import annotations

import json

from vision_language_benchmarks.cli import main


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
            "--language",
            "ja",
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
    assert not output_path.exists()


def test_data_terms_confirmation_does_not_override_verified_access(capsys, tmp_path) -> None:
    exit_code = main(
        [
            "run",
            "--model",
            "embeddinggemma2",
            "--dataset",
            "crossmodal3600",
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
