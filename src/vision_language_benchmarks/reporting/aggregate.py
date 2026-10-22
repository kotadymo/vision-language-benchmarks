from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import jsonschema

from vision_language_benchmarks.atomic import atomic_write_json
from vision_language_benchmarks.errors import DataIntegrityError
from vision_language_benchmarks.registry import get_dataset, get_model
from vision_language_benchmarks.reporting.comparison import paired_language_results
from vision_language_benchmarks.reporting.mrl import (
    build_embedding_dimension_audit,
    build_embedding_dimension_baseline_comparison,
    render_embedding_dimension_section,
    write_embedding_dimension_analysis,
)
from vision_language_benchmarks.reporting.plots import write_plots
from vision_language_benchmarks.reporting.reproducibility_audit import (
    build_clean_official_xm3600_parity,
    build_clean_xm3600_reproducibility_audit,
)
from vision_language_benchmarks.reporting.retrieval_variant import (
    retrieval_variant_info,
    retrieval_variant_label,
)
from vision_language_benchmarks.reporting.schema import validate_result

SUMMARY_FIELDS = (
    "run_id",
    "publication_state",
    "reproducibility_mode",
    "model_key",
    "embedding_dimension",
    "global_embedding_dimension",
    "model",
    "model_checkpoint_filename",
    "model_checkpoint_size_bytes",
    "model_parameter_count",
    "model_architecture",
    "model_languages",
    "model_license",
    "model_weight_license_status",
    "model_weight_license_reason",
    "experimental_languages",
    "model_library",
    "model_library_version",
    "tokenizer_license",
    "dataset_key",
    "dataset",
    "task",
    "split",
    "language",
    "protocol",
    "retrieval_variant",
    "retrieval_variant_source",
    "precision",
    "device",
    "gpu_name",
    "image_count",
    "text_count",
    "class_count",
    "batch_size_requested",
    "batch_size_actual",
    "image_batch_size_actual",
    "text_batch_size_actual",
    "oom_retry_count",
    "dataset_checksum",
    "dataset_license_status",
    "dataset_terms_confirmed",
    "prompt_name",
    "acc1_fraction",
    "acc5_fraction",
    "image_retrieval_recall@1_fraction",
    "image_retrieval_recall@5_fraction",
    "image_retrieval_recall@10_fraction",
    "text_retrieval_recall@1_fraction",
    "text_retrieval_recall@5_fraction",
    "text_retrieval_recall@10_fraction",
    "throughput_images_per_sec",
    "throughput_texts_per_sec",
    "inference_ms_median",
    "inference_ms_iqr",
    "gpu_peak_allocated_bytes",
    "gpu_peak_reserved_bytes",
    "source_commit_sha",
    "working_tree_dirty",
)
_LIFECYCLE_STATES = {
    "active",
    "historical",
    "superseded",
    "failed",
    "incomplete",
    "invalid",
    "blocked",
    "skipped",
}
NON_SUCCESS_FIELDS = (
    "run_id",
    "status",
    "model_key",
    "embedding_dimension",
    "global_embedding_dimension",
    "model",
    "model_weight_license_status",
    "model_weight_license_reason",
    "dataset_key",
    "dataset",
    "language",
    "protocol",
    "retrieval_variant",
    "precision",
    "reason",
)
RETRIEVAL_KEYS = (
    "image_retrieval_recall@1",
    "image_retrieval_recall@5",
    "image_retrieval_recall@10",
    "text_retrieval_recall@1",
    "text_retrieval_recall@5",
    "text_retrieval_recall@10",
)


def load_results(input_path: Path) -> list[dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Result input does not exist: {input_path}")
    if input_path.is_file():
        paths = [input_path]
    elif input_path.is_dir():
        paths = sorted(input_path.glob("*.json"))
    else:
        raise ValueError(f"Result input is not a regular file or directory: {input_path}")

    results: list[dict[str, Any]] = []
    for path in paths:
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise DataIntegrityError(f"Cannot read result JSON {path}: {error}") from error
        if not isinstance(result, dict):
            raise DataIntegrityError(f"Result JSON must contain an object: {path}")
        try:
            validate_result(result)
        except (jsonschema.ValidationError, ValueError) as error:
            raise DataIntegrityError(f"Result JSON failed validation ({path}): {error}") from error
        results.append(result)
    return sorted(
        results,
        key=lambda result: (
            str(result["dataset"]),
            str(result["benchmark_metadata"]["run_configuration"].get("protocol", "")),
            str(result["language"]),
            str(result["benchmark_metadata"].get("precision", "")),
            str(result["benchmark_metadata"]["run_id"]),
        ),
    )


def _load_publication_lifecycle(path: Path) -> dict[str, Any]:
    try:
        lifecycle = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DataIntegrityError(
            f"Cannot read publication lifecycle manifest {path}: {error}"
        ) from error
    if not isinstance(lifecycle, dict) or lifecycle.get("schema_version") != "1.0.0":
        raise DataIntegrityError(
            f"Publication lifecycle manifest has an unsupported structure: {path}."
        )
    frozen_source_commit_sha = lifecycle.get("frozen_source_commit_sha")
    if (
        not isinstance(frozen_source_commit_sha, str)
        or len(frozen_source_commit_sha) != 40
        or any(character not in "0123456789abcdef" for character in frozen_source_commit_sha)
    ):
        raise DataIntegrityError(
            "Publication lifecycle manifest must identify the frozen full source commit."
        )
    entries = lifecycle.get("entries")
    attempts = lifecycle.get("attempts", [])
    if not isinstance(entries, list) or not isinstance(attempts, list):
        raise DataIntegrityError(
            "Publication lifecycle entries and attempts must be arrays."
        )

    entry_map: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise DataIntegrityError("Publication lifecycle entries must be objects.")
        run_id = entry.get("run_id")
        state = entry.get("state")
        leaderboard_eligible = entry.get("leaderboard_eligible")
        if not isinstance(run_id, str) or not run_id:
            raise DataIntegrityError("Every publication lifecycle entry needs a run ID.")
        if run_id in entry_map:
            raise DataIntegrityError(f"Duplicate publication lifecycle run ID: {run_id}.")
        if (
            not isinstance(state, str)
            or state not in _LIFECYCLE_STATES
            or not isinstance(leaderboard_eligible, bool)
        ):
            raise DataIntegrityError(
                f"Publication lifecycle entry {run_id} has an invalid state."
            )
        if leaderboard_eligible != (state in {"active", "historical"}):
            raise DataIntegrityError(
                f"Publication lifecycle state and leaderboard eligibility disagree for {run_id}."
            )
        entry_map[run_id] = entry

    for attempt in attempts:
        if (
            not isinstance(attempt, dict)
            or not isinstance(attempt.get("state"), str)
            or attempt.get("state")
            not in {"failed", "incomplete", "invalid", "blocked", "skipped"}
        ):
            raise DataIntegrityError(
                "Publication lifecycle attempts must have failed, incomplete, invalid, "
                "blocked, or skipped states."
            )
    return {
        **lifecycle,
        "entries_by_run_id": entry_map,
    }


def _select_publishable_results(
    results: list[dict[str, Any]],
    lifecycle: dict[str, Any],
    expected_source_commit_sha: str | None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    frozen_source_commit_sha = lifecycle["frozen_source_commit_sha"]
    if (
        expected_source_commit_sha is not None
        and expected_source_commit_sha != frozen_source_commit_sha
    ):
        raise DataIntegrityError(
            "The requested publication source commit does not match the lifecycle manifest."
        )
    entries = lifecycle["entries_by_run_id"]
    result_ids = [str(result["benchmark_metadata"]["run_id"]) for result in results]
    if len(result_ids) != len(set(result_ids)):
        raise DataIntegrityError("Raw result files contain duplicate run IDs.")
    missing = sorted(set(result_ids).difference(entries))
    extra = sorted(set(entries).difference(result_ids))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing lifecycle entries for {', '.join(missing)}")
        if extra:
            details.append(f"lifecycle entries without raw results for {', '.join(extra)}")
        raise DataIntegrityError(
            "Publication lifecycle does not match raw results: " + "; ".join(details)
        )

    selected: list[dict[str, Any]] = []
    publication_states: dict[str, str] = {}
    for result in results:
        metadata = result["benchmark_metadata"]
        run_id = str(metadata["run_id"])
        entry = entries[run_id]
        state = entry["state"]
        publication_states[run_id] = state
        variant, _ = retrieval_variant_info(result)
        if variant == "official_first_caption" and state not in {
            "active",
            "superseded",
        }:
            raise DataIntegrityError(
                "Official-first-caption results must be active or explicitly superseded; "
                "invalid, failed, or incomplete records cannot be published."
            )
        if not entry["leaderboard_eligible"]:
            continue
        if result["status"] != "completed":
            raise DataIntegrityError(
                f"Leaderboard-eligible run {run_id} is not completed."
            )
        if metadata.get("working_tree_dirty") is not False:
            raise DataIntegrityError(
                f"Leaderboard-eligible run {run_id} has a dirty or unknown worktree."
            )
        if state == "active":
            if metadata.get("source_commit_sha") != frozen_source_commit_sha:
                raise DataIntegrityError(
                    f"Active run {run_id} does not use the frozen source commit."
                )
            if (
                metadata.get("reproducibility_mode") != "strict"
                or metadata.get("expected_source_commit_sha") != frozen_source_commit_sha
                or _configuration(result).get("reproducibility_mode") != "strict"
            ):
                raise DataIntegrityError(
                    f"Active run {run_id} lacks strict clean-source provenance."
                )
        selected.append(result)
    return selected, publication_states


def _configuration(result: dict[str, Any]) -> dict[str, Any]:
    value = result["benchmark_metadata"].get("run_configuration", {})
    return value if isinstance(value, dict) else {}


def _model_name(result: dict[str, Any]) -> str:
    model_key = _configuration(result).get("model_key")
    if isinstance(model_key, str):
        try:
            return get_model(model_key).name
        except ValueError:
            pass
    return str(result["model"])


def _dataset_name(result: dict[str, Any]) -> str:
    dataset_key = str(result["dataset"])
    try:
        return get_dataset(dataset_key).name
    except ValueError:
        return dataset_key


def summary_rows(
    results: list[dict[str, Any]],
    publication_states: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        if result["status"] != "completed":
            continue
        metadata = result["benchmark_metadata"]
        configuration = _configuration(result)
        metrics = result["metrics"]
        retrieval_variant, retrieval_variant_source = retrieval_variant_info(result)
        row: dict[str, Any] = {
            "run_id": metadata["run_id"],
            "publication_state": (publication_states or {}).get(metadata["run_id"], ""),
            "reproducibility_mode": metadata.get("reproducibility_mode", ""),
            "model_key": configuration.get("model_key", ""),
            "embedding_dimension": metadata.get(
                "embedding_dimension",
                configuration.get("embedding_dimension"),
            ),
            "global_embedding_dimension": metadata.get("global_embedding_dimension"),
            "model": _model_name(result),
            "model_checkpoint_filename": metadata.get("model_checkpoint_filename"),
            "model_checkpoint_size_bytes": metadata.get("model_checkpoint_size_bytes"),
            "model_parameter_count": metadata.get("parameter_count"),
            "model_architecture": metadata.get("model_architecture"),
            "model_languages": ", ".join(metadata.get("model_languages") or []),
            "model_license": metadata.get("model_license"),
            "model_weight_license_status": metadata.get("model_weight_license_status"),
            "model_weight_license_reason": metadata.get("model_weight_license_reason"),
            "experimental_languages": ", ".join(
                metadata.get("experimental_languages") or []
            ),
            "model_library": metadata.get("model_library"),
            "model_library_version": metadata.get("model_library_version"),
            "tokenizer_license": metadata.get("tokenizer_license"),
            "dataset_key": result["dataset"],
            "dataset": _dataset_name(result),
            "task": result["task"],
            "split": metadata["dataset_split"],
            "language": result["language"],
            "protocol": configuration.get("protocol", ""),
            "retrieval_variant": retrieval_variant or "",
            "retrieval_variant_source": retrieval_variant_source or "",
            "precision": metadata["precision"],
            "device": metadata["device"],
            "gpu_name": metadata.get("gpu_name"),
            "image_count": metadata.get("image_count"),
            "text_count": metadata.get("text_count"),
            "class_count": metadata.get("class_count"),
            "batch_size_requested": configuration.get("batch_size_requested"),
            "batch_size_actual": metadata.get("batch_size_actual"),
            "image_batch_size_actual": metadata.get("image_batch_size_actual"),
            "text_batch_size_actual": metadata.get("text_batch_size_actual"),
            "oom_retry_count": len(metadata.get("oom_history") or []),
            "dataset_checksum": metadata.get("dataset_checksum"),
            "dataset_license_status": metadata.get("dataset_license_status"),
            "dataset_terms_confirmed": metadata.get("dataset_terms_confirmed"),
            "prompt_name": metadata.get("prompt_name"),
            "acc1_fraction": metrics.get("acc1"),
            "acc5_fraction": metrics.get("acc5"),
            "throughput_images_per_sec": metadata.get("throughput_images_per_sec"),
            "throughput_texts_per_sec": metadata.get("throughput_texts_per_sec"),
            "inference_ms_median": metadata.get("inference_ms_median"),
            "inference_ms_iqr": metadata.get("inference_ms_iqr"),
            "gpu_peak_allocated_bytes": metadata.get("gpu_peak_allocated_bytes"),
            "gpu_peak_reserved_bytes": metadata.get("gpu_peak_reserved_bytes"),
            "source_commit_sha": metadata.get("source_commit_sha"),
            "working_tree_dirty": metadata.get("working_tree_dirty"),
        }
        for metric_name in RETRIEVAL_KEYS:
            row[f"{metric_name}_fraction"] = metrics.get(metric_name)
        rows.append(row)
    return rows


def _status_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        if result["status"] == "completed":
            continue
        metadata = result["benchmark_metadata"]
        configuration = _configuration(result)
        retrieval_variant, _ = retrieval_variant_info(result)
        rows.append(
            {
                "run_id": metadata["run_id"],
                "status": result["status"],
                "model_key": configuration.get("model_key", ""),
                "embedding_dimension": metadata.get(
                    "embedding_dimension",
                    configuration.get("embedding_dimension"),
                ),
                "global_embedding_dimension": metadata.get("global_embedding_dimension"),
                "model": _model_name(result),
                "model_weight_license_status": metadata.get(
                    "model_weight_license_status"
                ),
                "model_weight_license_reason": metadata.get(
                    "model_weight_license_reason"
                ),
                "dataset_key": result["dataset"],
                "dataset": _dataset_name(result),
                "language": result["language"],
                "protocol": configuration.get("protocol", ""),
                "retrieval_variant": retrieval_variant or "",
                "precision": metadata.get("precision", ""),
                "reason": metadata.get("reason_if_not_success") or "",
            }
        )
    return rows


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(
                stream,
                fieldnames=fields,
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {key: "" if row.get(key) is None else row.get(key, "") for key in fields}
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _protocol_precision_groups(
    results: list[dict[str, Any]],
    *,
    task: str,
    language: str,
) -> list[tuple[str, str, list[dict[str, Any]]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in results:
        if (
            result["status"] != "completed"
            or result["task"] != task
            or result["language"] != language
        ):
            continue
        configuration = _configuration(result)
        key = (
            str(configuration.get("protocol", "")),
            str(result["benchmark_metadata"].get("precision", "")),
        )
        grouped.setdefault(key, []).append(result)
    return [
        (protocol, precision, sorted(items, key=lambda item: item["benchmark_metadata"]["run_id"]))
        for (protocol, precision), items in sorted(grouped.items())
    ]


def _markdown_cell(value: Any) -> str:
    return str(value if value is not None else "—").replace("|", "\\|").replace("\n", " ")


def _percentage(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    return "—" if value is None else f"{float(value) * 100:.2f}%"


def _sample_count(value: Any) -> str:
    return "—" if not isinstance(value, int) else f"{value:,}"


def _rate(value: Any) -> str:
    return "—" if not isinstance(value, (int, float)) else f"{float(value):.1f}"


def _milliseconds(value: Any) -> str:
    return "—" if not isinstance(value, (int, float)) else f"{float(value):.2f}"


def _gibibytes(value: Any) -> str:
    return "—" if not isinstance(value, (int, float)) else f"{float(value) / 1024**3:.2f}"


def _peak_memory(metadata: dict[str, Any]) -> str:
    allocated = _gibibytes(metadata.get("gpu_peak_allocated_bytes"))
    reserved = _gibibytes(metadata.get("gpu_peak_reserved_bytes"))
    return f"{allocated} / {reserved}"


def _batch_sizes(metadata: dict[str, Any]) -> str:
    keys = (
        "batch_size_requested",
        "image_batch_size_actual",
        "text_batch_size_actual",
    )
    return " / ".join(_sample_count(metadata.get(key)) for key in keys)


def _raw_result_link(run_id: Any) -> str:
    value = str(run_id)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        return _markdown_cell(value)
    return f"[{_markdown_cell(value)}](raw/{value}.json)"


def _render_measurement_table(results: list[dict[str, Any]]) -> list[str]:
    if not results:
        return []

    task = results[0]["task"]
    if task == "classification":
        lines = [
            "| Model | Embedding dim | Parameters | Checkpoint | "
            "Dataset | Split | Images | Classes | "
            "Requested / image / text batch | Top-1 | Top-5 | Images/s | Median (ms) | "
            "Peak allocated / reserved (GiB) | OOM retries | Raw result |",
            "|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for result in results:
            metadata = result["benchmark_metadata"]
            metrics = result["metrics"]
            parameter_count = metadata.get("parameter_count")
            parameter_size = (
                "—" if parameter_count is None else f"{int(parameter_count) / 1_000_000:.1f} M"
            )
            checkpoint_size = metadata.get("model_checkpoint_size_bytes")
            checkpoint_size_text = (
                "—"
                if checkpoint_size is None
                else f"{int(checkpoint_size) / 1024**3:.2f} GiB"
            )
            values = (
                _model_name(result),
                _sample_count(metadata.get("embedding_dimension")),
                parameter_size,
                checkpoint_size_text,
                _dataset_name(result),
                metadata["dataset_split"],
                _sample_count(metadata.get("image_count")),
                _sample_count(metadata.get("class_count")),
                _batch_sizes(metadata),
                _percentage(metrics, "acc1"),
                _percentage(metrics, "acc5"),
                _rate(metadata.get("throughput_images_per_sec")),
                _milliseconds(metadata.get("inference_ms_median")),
                _peak_memory(metadata),
                _sample_count(len(metadata.get("oom_history") or [])),
                _raw_result_link(metadata["run_id"]),
            )
            lines.append("| " + " | ".join(_markdown_cell(value) for value in values) + " |")
        return lines

    lines = [
        "| Model | Embedding dim | Parameters | Checkpoint | Dataset | Split | "
        "Language | Retrieval variant | "
        "Images / captions | "
        "Requested / image / text batch | "
        "T→I R@1 | T→I R@5 | T→I R@10 | I→T R@1 | I→T R@5 | I→T R@10 | "
        "Images/s | Texts/s | Median (ms) | Peak allocated / reserved (GiB) | "
        "OOM retries | Raw result |",
        "|---|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        metadata = result["benchmark_metadata"]
        metrics = result["metrics"]
        parameter_count = metadata.get("parameter_count")
        parameter_size = (
            "—" if parameter_count is None else f"{int(parameter_count) / 1_000_000:.1f} M"
        )
        checkpoint_size = metadata.get("model_checkpoint_size_bytes")
        checkpoint_size_text = (
            "—"
            if checkpoint_size is None
            else f"{int(checkpoint_size) / 1024**3:.2f} GiB"
        )
        image_caption_count = (
            f"{_sample_count(metadata.get('image_count'))} / "
            f"{_sample_count(metadata.get('text_count'))}"
        )
        variant, _ = retrieval_variant_info(result)
        values = (
            _model_name(result),
            _sample_count(metadata.get("embedding_dimension")),
            parameter_size,
            checkpoint_size_text,
            _dataset_name(result),
            metadata["dataset_split"],
            result["language"],
            variant or "—",
            image_caption_count,
            _batch_sizes(metadata),
            _percentage(metrics, "image_retrieval_recall@1"),
            _percentage(metrics, "image_retrieval_recall@5"),
            _percentage(metrics, "image_retrieval_recall@10"),
            _percentage(metrics, "text_retrieval_recall@1"),
            _percentage(metrics, "text_retrieval_recall@5"),
            _percentage(metrics, "text_retrieval_recall@10"),
            _rate(metadata.get("throughput_images_per_sec")),
            _rate(metadata.get("throughput_texts_per_sec")),
            _milliseconds(metadata.get("inference_ms_median")),
            _peak_memory(metadata),
            _sample_count(len(metadata.get("oom_history") or [])),
            _raw_result_link(metadata["run_id"]),
        )
        lines.append("| " + " | ".join(_markdown_cell(value) for value in values) + " |")
    return lines


def _paired_metric_cell(
    english_metrics: dict[str, Any],
    japanese_metrics: dict[str, Any],
    metric_key: str,
) -> str:
    english = english_metrics.get(metric_key)
    japanese = japanese_metrics.get(metric_key)
    if not isinstance(english, (int, float)) or not isinstance(japanese, (int, float)):
        return "—"
    delta = (float(japanese) - float(english)) * 100
    return f"{float(english) * 100:.2f}% → {float(japanese) * 100:.2f}% ({delta:+.2f} pp)"


def _render_paired_language_section(results: list[dict[str, Any]]) -> list[str]:
    grouped: dict[
        tuple[str, str, str, str],
        list[tuple[dict[str, Any], dict[str, Any]]],
    ] = {}
    for english, japanese in paired_language_results(results):
        configuration = _configuration(english)
        key = (
            str(english["dataset"]),
            str(configuration.get("protocol", "")),
            str(english["benchmark_metadata"].get("precision", "")),
            str(retrieval_variant_info(english)[0]),
        )
        grouped.setdefault(key, []).append((english, japanese))

    if not grouped:
        return [
            "## Matched English/Japanese retrieval",
            "",
            "No unique, condition-matched English/Japanese result pairs are available.",
            "",
        ]

    lines = ["## Matched English/Japanese retrieval", ""]
    for (_, protocol, precision, variant), pairs in sorted(grouped.items()):
        lines.extend(
            [
                f"### {_dataset_name(pairs[0][0])} · {protocol.upper()} · "
                f"{precision.upper()} · {variant}",
                "",
                "| Model | Embedding dim | Split | Images | Captions (EN / JA) | "
                "T→I R@1 (EN → JA; Δ pp) | "
                "T→I R@5 (EN → JA; Δ pp) | T→I R@10 (EN → JA; Δ pp) | "
                "I→T R@1 (EN → JA; Δ pp) | I→T R@5 (EN → JA; Δ pp) | "
                "I→T R@10 (EN → JA; Δ pp) |",
                "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for english, japanese in sorted(pairs, key=lambda pair: _model_name(pair[0])):
            english_metadata = english["benchmark_metadata"]
            japanese_metadata = japanese["benchmark_metadata"]
            english_metrics = english["metrics"]
            japanese_metrics = japanese["metrics"]
            values = (
                _model_name(english),
                _sample_count(english_metadata.get("embedding_dimension")),
                english_metadata["dataset_split"],
                _sample_count(english_metadata.get("image_count")),
                (
                    f"{_sample_count(english_metadata.get('text_count'))} / "
                    f"{_sample_count(japanese_metadata.get('text_count'))}"
                ),
                _paired_metric_cell(
                    english_metrics, japanese_metrics, "image_retrieval_recall@1"
                ),
                _paired_metric_cell(
                    english_metrics, japanese_metrics, "image_retrieval_recall@5"
                ),
                _paired_metric_cell(
                    english_metrics, japanese_metrics, "image_retrieval_recall@10"
                ),
                _paired_metric_cell(
                    english_metrics, japanese_metrics, "text_retrieval_recall@1"
                ),
                _paired_metric_cell(
                    english_metrics, japanese_metrics, "text_retrieval_recall@5"
                ),
                _paired_metric_cell(
                    english_metrics, japanese_metrics, "text_retrieval_recall@10"
                ),
            )
            lines.append("| " + " | ".join(_markdown_cell(value) for value in values) + " |")
        lines.append("")
    return lines


def _render_task_language_section(
    results: list[dict[str, Any]],
    *,
    task: str,
    language: str,
    title: str,
) -> list[str]:
    lines = [f"## {title}", ""]
    groups = _protocol_precision_groups(results, task=task, language=language)
    if not groups:
        lines.extend(["No completed measured runs are available.", ""])
        return lines
    for protocol, precision, items in groups:
        lines.extend([f"### {protocol.upper()} · {precision.upper()}", ""])
        lines.extend(_render_measurement_table(items))
        lines.append("")
    return lines


def _render_retrieval_variant_section(
    results: list[dict[str, Any]],
    *,
    variant: str,
) -> list[str]:
    title = (
        "Japanese/English official-compatible XM3600 retrieval"
        if variant == "official_first_caption"
        else "XM3600 all-caption retrieval"
    )
    lines = [f"## {title} (`{retrieval_variant_label(variant)}`)", ""]
    variant_results = [
        result
        for result in results
        if result["task"] == "retrieval"
        and retrieval_variant_info(result)[0] == variant
    ]
    if not variant_results:
        lines.extend(["No completed measured runs are available.", ""])
        return lines
    for language in ("en", "ja"):
        language_results = [
            result for result in variant_results if result["language"] == language
        ]
        if not language_results:
            continue
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for result in language_results:
            configuration = _configuration(result)
            key = (
                str(configuration.get("protocol", "")),
                str(result["benchmark_metadata"].get("precision", "")),
            )
            grouped.setdefault(key, []).append(result)
        for (protocol, precision), items in sorted(grouped.items()):
            lines.extend(
                [
                    f"### {language.upper()} · {protocol.upper()} · {precision.upper()}",
                    "",
                ]
            )
            lines.extend(_render_measurement_table(items))
            lines.append("")
    return lines


def _render_performance_section(results: list[dict[str, Any]]) -> list[str]:
    rows = [
        result
        for result in results
        if result["status"] == "completed"
        and (
            isinstance(
                result["benchmark_metadata"].get("throughput_images_per_sec"),
                (int, float),
            )
            or isinstance(
                result["benchmark_metadata"].get("throughput_texts_per_sec"),
                (int, float),
            )
        )
    ]
    if not rows:
        return [
            "## Model throughput and VRAM",
            "",
            "No throughput measurements are available.",
            "",
        ]
    lines = [
        "## Model throughput and VRAM",
        "",
        "Throughput is computed from model inference only; model downloads, initialization, "
        "dataset preparation, and report generation are excluded.",
        "",
        "| Model | Dataset | Task | Language | Protocol | Retrieval variant | Precision | "
        "Images/s | Texts/s | Median (ms) | Peak allocated / reserved (GiB) | Raw result |",
        "|---|---|---|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for result in rows:
        metadata = result["benchmark_metadata"]
        configuration = _configuration(result)
        variant, _ = retrieval_variant_info(result)
        values = (
            _model_name(result),
            _dataset_name(result),
            result["task"],
            result["language"],
            configuration.get("protocol", ""),
            variant or "—",
            metadata.get("precision", "—"),
            _rate(metadata.get("throughput_images_per_sec")),
            _rate(metadata.get("throughput_texts_per_sec")),
            _milliseconds(metadata.get("inference_ms_median")),
            _peak_memory(metadata),
            _raw_result_link(metadata["run_id"]),
        )
        lines.append("| " + " | ".join(_markdown_cell(value) for value in values) + " |")
    lines.append("")
    return lines


def render_report(
    results: list[dict[str, Any]],
    plot_files: dict[str, Path] | None = None,
    validation_dir: Path | None = None,
    publication_lifecycle: dict[str, Any] | None = None,
    native_validation_report: dict[str, Any] | None = None,
) -> str:
    completed = [result for result in results if result["status"] == "completed"]
    non_success = _status_rows(results)
    status_counts = Counter(result["status"] for result in results)
    source_shas = sorted(
        {
            str(result["benchmark_metadata"]["source_commit_sha"])
            for result in results
            if result["benchmark_metadata"].get("source_commit_sha")
        }
    )
    runtime_profiles = sorted(
        {
            tuple(
                str(result["benchmark_metadata"].get(key) or "")
                for key in (
                    "gpu_name",
                    "gpu_vram_bytes",
                    "driver_version",
                    "torch_version",
                    "torch_cuda_version",
                    "precision",
                )
            )
            for result in completed
        }
    )
    lines = [
        "# Vision-Language Benchmark Report",
        "",
        "This report is regenerated from schema-validated raw JSON. "
        "Only completed runs contribute numeric metrics; failed, skipped, and blocked runs "
        "are listed separately. No values are estimated or fabricated.",
        "",
        "## Conditions",
        "",
        f"- Validated result files: {len(results)}",
        f"- Completed measurements: {status_counts.get('completed', 0)}",
        f"- Skipped runs: {status_counts.get('skipped', 0)}",
        f"- Failed runs: {status_counts.get('failed', 0)}",
        f"- Blocked runs: {status_counts.get('blocked', 0)}",
        f"- Source commit(s): {', '.join(source_shas) if source_shas else 'not recorded'}",
        "- Runtime profile(s): "
        + (
            "; ".join(
                f"{gpu or 'GPU unknown'} ({vram or 'VRAM unknown'} bytes, driver "
                f"{driver or 'unknown'}, PyTorch {torch or 'unknown'}, CUDA "
                f"{cuda or 'unknown'}, {precision or 'precision unknown'})"
                for gpu, vram, driver, torch, cuda, precision in runtime_profiles
            )
            if runtime_profiles
            else "not recorded"
        ),
        "- Protocols and precisions are kept in separate tables and chart files.",
        "- Accuracy and recall values in JSON are fractions; report tables display percentages.",
        "- Checkpoint sizes are pinned weight-file sizes; tokenizer and configuration files "
        "may add to the download. Peak memory is reported as allocated / reserved GiB.",
        "- Each completed table row links to its schema-validated raw result JSON.",
        "",
    ]
    if publication_lifecycle is not None:
        counts = publication_lifecycle["state_counts"]
        lines.extend(
            [
                "## Measurement lifecycle",
                "",
                f"- Frozen benchmark source commit: "
                f"`{publication_lifecycle['frozen_source_commit_sha']}`",
                f"- Active validated measurements: {counts.get('active', 0)}",
                f"- Retained historical measurements: {counts.get('historical', 0)}",
                f"- Superseded measurements excluded from primary results: "
                f"{counts.get('superseded', 0)}",
                f"- Failed attempts: {counts.get('failed', 0)}",
                f"- Incomplete attempts: {counts.get('incomplete', 0)}",
                f"- Invalid measurements: {counts.get('invalid', 0)}",
                f"- Blocked runs: {counts.get('blocked', 0)}",
                f"- Skipped runs: {counts.get('skipped', 0)}",
                "- Per-run lifecycle assignments are recorded in "
                "[result-lifecycle.json](validation/result-lifecycle.json).",
                "",
            ]
        )
    lines.extend(["## English zero-shot classification", ""])
    lines.extend(
        _render_task_language_section(
            completed,
            task="classification",
            language="en",
            title="English zero-shot classification",
        )[2:]
    )
    lines.extend(
        _render_retrieval_variant_section(
            completed,
            variant="official_first_caption",
        )
    )
    lines.extend(
        _render_task_language_section(
            completed,
            task="classification",
            language="ja",
            title="Japanese zero-shot classification",
        )
    )
    lines.extend(
        _render_retrieval_variant_section(
            completed,
            variant="all_captions",
        )
    )
    lines.extend(_render_paired_language_section(completed))
    lines.extend(_render_performance_section(completed))
    lines.extend(
        render_embedding_dimension_section(
            completed,
            native_validation_report,
        )
    )

    lines.extend(
        [
            "## Non-successful runs",
            "",
        ]
    )
    if non_success:
        for status in ("blocked", "skipped", "failed"):
            status_items = [row for row in non_success if row["status"] == status]
            if not status_items:
                continue
            lines.extend(
                [
                    f"### {status.capitalize()}",
                    "",
                    "| Model | Dataset | Language | Protocol | Retrieval variant | "
                    "Model weight license | Reason | Run ID |",
                    "|---|---|---|---|---|---|---|---|",
                ]
            )
            for row in status_items:
                lines.append(
                    "| "
                    + " | ".join(
                        _markdown_cell(row[key])
                        for key in (
                            "model",
                            "dataset",
                            "language",
                            "protocol",
                            "retrieval_variant",
                            "model_weight_license_status",
                            "reason",
                            "run_id",
                        )
                    )
                    + " |"
                )
            lines.append("")
    else:
        lines.extend(
            [
                "No unsuccessful-run JSON records are included in the published raw set. "
                "See the machine-readable run status for blocked evaluations and access "
                "constraints.",
                "",
            ]
        )

    validation_artifacts = (
        "historical-results-audit.json",
        "model-smoke-tests.json",
        "official-cifar10-parity.json",
        "official-xm3600-openclip-ja.json",
        "official-xm3600-parity.json",
        "official-xm3600-parity-clean.json",
        "result-lifecycle.json",
        "clean-xm3600-reproducibility-audit.json",
        "embedding-dimension-audit.json",
    )
    available_validation_artifacts = [
        f"[{name}](validation/{name})"
        for name in validation_artifacts
        if validation_dir is not None and (validation_dir / name).is_file()
    ]
    lines.extend(
        [
            "## Validation and reproducibility",
            "",
            "Each raw result is schema-validated before aggregation. Per-run JSON records retain "
            "the source commit, model checkpoint fingerprint, dataset checksum, ordered "
            "candidate IDs and hashes, prompt policy, runtime, and measured metrics.",
        ]
    )
    if available_validation_artifacts:
        lines.append(
            "- Validation artifacts: " + ", ".join(available_validation_artifacts) + "."
        )
    else:
        lines.append("No separate validation artifacts are present in this report output.")
    lines.append("")

    lines.extend(
        [
            "## Known limitations",
            "",
            "- Official numerical parity applies only to the exact combinations documented "
            "in the available validation records. It does not establish parity for other "
            "models, languages, datasets, precisions, or retrieval variants.",
            "- Training-data contamination cannot be ruled out unless a model's pretraining "
            "corpus is independently verified.",
            "- Dataset and model access restrictions remain in force. See "
            "[model-and-dataset-licenses.md](../../docs/model-and-dataset-licenses.md).",
            "- Missing metrics remain missing. Unsupported languages and unavailable datasets "
            "are not represented as zero scores.",
            "",
            "## Reproduction",
            "",
            "```powershell",
            "uv sync --locked --group dev",
            "uv run vlbench doctor --device cuda",
            "uv run vlbench report --input results/v0.1.0/raw --output results/v0.1.0",
            "```",
            "",
            "To reproduce a measurement, use the exact model revision, dataset checksum, split, "
            "protocol, language, precision, batch size, seed, source commit, and lockfile hash "
            "stored in its raw JSON.",
            "",
            "## Figures",
            "",
        ]
    )
    if plot_files:
        stems = sorted(
            {key.removesuffix("_png") for key in plot_files if key.endswith("_png")}
        )
        for stem in stems:
            png = plot_files.get(f"{stem}_png")
            svg = plot_files.get(f"{stem}_svg")
            links = []
            if png is not None:
                links.append(f"[PNG](plots/{png.name})")
            if svg is not None:
                links.append(f"[SVG](plots/{svg.name})")
            lines.append(f"- {stem.replace('_', ' ').title()}: {' · '.join(links)}")
        lines.append("")
    else:
        lines.extend(["No meaningful figures are available from completed measurements.", ""])
    return "\n".join(lines)


def _metadata_document(
    results: list[dict[str, Any]],
    publication_states: dict[str, str] | None = None,
    publication_lifecycle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_map: dict[str, dict[str, Any]] = {}
    model_map: dict[str, dict[str, Any]] = {}
    environments: set[tuple[str, ...]] = set()
    source_shas: set[str] = set()
    for result in results:
        metadata = result["benchmark_metadata"]
        configuration = _configuration(result)
        dataset_key = str(result["dataset"])
        model_key = str(configuration.get("model_key", result["model"]))
        try:
            dataset_spec = get_dataset(dataset_key)
        except ValueError:
            dataset_spec = None
        dataset = dataset_map.setdefault(
            dataset_key,
            {
                "dataset_id": metadata["dataset_id"],
                "name": _dataset_name(result),
                "split": metadata["dataset_split"],
                "revision": metadata.get("dataset_revision"),
                "checksum": metadata.get("dataset_checksum"),
                "license": metadata.get("dataset_license"),
                "license_status": metadata.get("dataset_license_status"),
                "terms_confirmations": [],
                "image_count": (
                    dataset_spec.expected_images
                    if dataset_spec is not None
                    else metadata.get("image_count")
                ),
                "expected_text_counts_by_language": (
                    {
                        language: dataset_spec.expected_text_count(language)
                        for language in dataset_spec.languages
                    }
                    if dataset_spec is not None
                    else {}
                ),
                "class_count": metadata.get("class_count"),
                "retrieval_variant_evaluations": [],
            },
        )
        dataset["terms_confirmations"].append(
            {
                "run_id": metadata["run_id"],
                "confirmed": metadata.get("dataset_terms_confirmed"),
            }
        )
        if result["task"] == "retrieval":
            variant, variant_source = retrieval_variant_info(result)
            dataset["retrieval_variant_evaluations"].append(
                {
                    "run_id": metadata["run_id"],
                    "status": result["status"],
                    "protocol": configuration.get("protocol"),
                    "precision": metadata.get("precision"),
                    "embedding_dimension": metadata.get(
                        "embedding_dimension",
                        configuration.get("embedding_dimension"),
                    ),
                    "language": result["language"],
                    "retrieval_variant": variant,
                    "retrieval_variant_source": variant_source,
                    "image_count": metadata.get("image_count"),
                    "text_count": metadata.get("text_count"),
                    "candidate_counts": metadata.get("candidate_counts"),
                    "positive_pair_count": metadata.get("positive_pair_count"),
                    "image_id_order_hash": metadata.get("image_id_order_hash"),
                    "text_id_order_hash": metadata.get("text_id_order_hash"),
                    "text_content_sha256": metadata.get("text_content_sha256"),
                    "source_commit_sha": metadata.get("source_commit_sha"),
                    "checkpoint_revision": metadata.get("checkpoint_revision"),
                    "checkpoint_sha256": metadata.get("checkpoint_sha256"),
                    "preprocessing_config_sha256": metadata.get(
                        "preprocessing_config_sha256"
                    ),
                }
            )
        model = model_map.setdefault(model_key, {})
        model_values = {
            "name": _model_name(result),
            "model_id": result["model"],
            "repository": metadata.get("model_repo"),
            "checkpoint_revision": metadata.get("checkpoint_revision"),
            "checkpoint_filename": metadata.get("model_checkpoint_filename"),
            "checkpoint_size_bytes": metadata.get("model_checkpoint_size_bytes"),
            "supported_languages": metadata.get("model_languages"),
            "checkpoint_sha256": metadata.get("checkpoint_sha256"),
            "pretrained_tag": result["pretrained"],
            "license": metadata.get("model_license"),
            "tokenizer_license": metadata.get("tokenizer_license"),
            "parameter_count": metadata.get("parameter_count"),
        }
        model.update(
            {
                key: value
                for key, value in model_values.items()
                if value is not None
            }
        )
        environments.add(
            tuple(
                str(metadata.get(key) or "")
                for key in (
                    "python_version",
                    "torch_version",
                    "torch_cuda_version",
                    "device",
                    "gpu_name",
                    "driver_version",
                    "precision",
                )
            )
        )
        commit = metadata.get("source_commit_sha")
        if commit:
            source_shas.add(str(commit))
    status_counts = Counter(result["status"] for result in results)
    for dataset in dataset_map.values():
        dataset["terms_confirmations"].sort(key=lambda entry: entry["run_id"])
        dataset["retrieval_variant_evaluations"].sort(
            key=lambda entry: (
                str(entry["retrieval_variant"] or ""),
                str(entry["language"]),
                str(entry["precision"] or ""),
                str(entry["run_id"]),
            )
        )
    document = {
        "schema_version": "1.0.0",
        "result_file_count": len(results),
        "status_counts": {key: status_counts[key] for key in sorted(status_counts)},
        "source_commit_shas": sorted(source_shas),
        "datasets": [dataset_map[key] for key in sorted(dataset_map)],
        "models": [model_map[key] for key in sorted(model_map)],
        "environments": [
            {
                key: value or None
                for key, value in zip(
                    (
                        "python_version",
                        "torch_version",
                        "torch_cuda_version",
                        "device",
                        "gpu_name",
                        "driver_version",
                        "precision",
                    ),
                    values,
                    strict=True,
                )
            }
            for values in sorted(environments)
        ],
        "runs": [
            {
                "run_id": result["benchmark_metadata"]["run_id"],
                "publication_state": (publication_states or {}).get(
                    result["benchmark_metadata"]["run_id"],
                    "",
                ),
                "reproducibility_mode": result["benchmark_metadata"].get(
                    "reproducibility_mode"
                ),
                "status": result["status"],
                "model": result["model"],
                "dataset": result["dataset"],
                "language": result["language"],
                "protocol": _configuration(result).get("protocol"),
                "precision": result["benchmark_metadata"].get("precision"),
                "embedding_dimension": result["benchmark_metadata"].get(
                    "embedding_dimension",
                    _configuration(result).get("embedding_dimension"),
                ),
            }
            for result in results
        ],
    }
    if publication_lifecycle is not None:
        document["publication_lifecycle"] = publication_lifecycle
    return document


def _load_native_mrl_validation_report(output_dir: Path) -> dict[str, Any] | None:
    report_path = output_dir / "validation" / "native-mrl-api-equivalence.json"
    if not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DataIntegrityError(
            f"Cannot read native MRL validation report {report_path}: {error}"
        ) from error
    if (
        not isinstance(report, dict)
        or report.get("validation_type") != "native_mrl_api_equivalence"
        or not isinstance(report.get("measurements"), list)
    ):
        raise DataIntegrityError(f"Invalid native MRL validation report at {report_path}.")
    return report


def generate_reports(
    input_path: Path,
    output_dir: Path,
    *,
    publishable: bool = False,
    lifecycle_path: Path | None = None,
    expected_source_commit_sha: str | None = None,
    baseline_input_path: Path | None = None,
    baseline_lifecycle_path: Path | None = None,
) -> dict[str, Path]:
    if (baseline_input_path is None) != (baseline_lifecycle_path is None):
        raise ValueError(
            "Historical baseline comparison requires both baseline input and lifecycle paths."
        )
    if baseline_input_path is not None and not publishable:
        raise ValueError(
            "Historical baseline comparison is supported only for publishable reports."
        )

    all_results = load_results(input_path)
    results = all_results
    publication_states: dict[str, str] | None = None
    publication_lifecycle: dict[str, Any] | None = None
    validation_documents: dict[str, dict[str, Any]] = {}
    dimension_results: list[dict[str, Any]] = []
    baseline_comparison: dict[str, Any] | None = None
    if publishable:
        selected_lifecycle_path = lifecycle_path or (
            output_dir / "validation" / "result-lifecycle.json"
        )
        lifecycle = _load_publication_lifecycle(selected_lifecycle_path)
        results, publication_states = _select_publishable_results(
            results,
            lifecycle,
            expected_source_commit_sha,
        )
        active_official_runs = [
            result
            for result in results
            if publication_states.get(result["benchmark_metadata"]["run_id"]) == "active"
            and retrieval_variant_info(result)[0] == "official_first_caption"
        ]
        dimension_results = [
            result
            for result in results
            if result["benchmark_metadata"].get("embedding_dimension") is not None
        ]
        if dimension_results:
            dimension_states = {
                result["benchmark_metadata"]["run_id"]: publication_states[
                    result["benchmark_metadata"]["run_id"]
                ]
                for result in dimension_results
            }
            validation_documents["embedding-dimension-audit.json"] = (
                build_embedding_dimension_audit(
                    dimension_results,
                    dimension_states,
                    lifecycle["frozen_source_commit_sha"],
                )
            )

        historical_suite_results = [
            result
            for result in all_results
            if result["benchmark_metadata"].get("embedding_dimension") is None
        ]
        historical_suite_states = {
            result["benchmark_metadata"]["run_id"]: publication_states[
                result["benchmark_metadata"]["run_id"]
            ]
            for result in historical_suite_results
        }
        active_historical_official_runs = [
            result
            for result in active_official_runs
            if result["benchmark_metadata"].get("embedding_dimension") is None
        ]
        if active_historical_official_runs:
            audit = build_clean_xm3600_reproducibility_audit(
                historical_suite_results,
                historical_suite_states,
                lifecycle["frozen_source_commit_sha"],
            )
            validation_dir = output_dir / "validation"
            historical_parity_path = validation_dir / "official-xm3600-parity.json"
            official_reference_path = validation_dir / "official-xm3600-openclip-ja.json"
            try:
                historical_parity = json.loads(
                    historical_parity_path.read_text(encoding="utf-8")
                )
                official_reference = json.loads(
                    official_reference_path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise DataIntegrityError(
                    f"Cannot read the pinned XM3600 parity reference: {error}"
                ) from error
            clean_openclip_japanese = next(
                result
                for result in active_historical_official_runs
                if _configuration(result).get("model_key") == "openclip_multilingual_b32"
                and result["language"] == "ja"
            )
            clean_parity = build_clean_official_xm3600_parity(
                historical_parity,
                official_reference,
                clean_openclip_japanese,
            )
            if clean_parity["status"] != "completed":
                failed_checks = [
                    name
                    for name, passed in clean_parity["validation_checks"].items()
                    if not passed
                ]
                raise DataIntegrityError(
                    "Clean XM3600 parity validation failed: "
                    + ", ".join(failed_checks)
                )
            validation_documents.update({
                "clean-xm3600-reproducibility-audit.json": audit,
                "official-xm3600-parity-clean.json": clean_parity,
            })
        state_counts = Counter(
            entry["state"] for entry in lifecycle["entries"]
        )
        state_counts.update(attempt["state"] for attempt in lifecycle["attempts"])
        publication_lifecycle = {
            "frozen_source_commit_sha": lifecycle["frozen_source_commit_sha"],
            "state_counts": {
                state: state_counts[state]
                for state in sorted(_LIFECYCLE_STATES)
            },
            "attempt_count": len(lifecycle["attempts"]),
        }
        if baseline_input_path is not None and baseline_lifecycle_path is not None:
            if not dimension_results:
                raise DataIntegrityError(
                    "Historical baseline comparison requires EmbeddingGemma dimension results."
                )
            previous_results = load_results(baseline_input_path)
            previous_lifecycle = _load_publication_lifecycle(baseline_lifecycle_path)
            previous_ids = {
                str(result["benchmark_metadata"]["run_id"])
                for result in previous_results
            }
            previous_lifecycle_ids = set(previous_lifecycle["entries_by_run_id"])
            if previous_ids != previous_lifecycle_ids:
                raise DataIntegrityError(
                    "Historical baseline lifecycle does not match its raw results."
                )
            baseline_comparison = build_embedding_dimension_baseline_comparison(
                dimension_results,
                previous_results,
                {
                    run_id: entry["state"]
                    for run_id, entry in previous_lifecycle["entries_by_run_id"].items()
                },
                lifecycle["frozen_source_commit_sha"],
                previous_lifecycle["frozen_source_commit_sha"],
            )
            validation_documents["embedding-dimension-baseline-comparison.json"] = (
                baseline_comparison
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    if validation_documents:
        validation_dir = output_dir / "validation"
        validation_dir.mkdir(parents=True, exist_ok=True)
        for filename, document in validation_documents.items():
            atomic_write_json(validation_dir / filename, document)
    native_validation_report = _load_native_mrl_validation_report(output_dir)
    completed_rows = summary_rows(results, publication_states)
    completed_rows.sort(
        key=lambda row: (
            str(row["dataset_key"]),
            str(row["protocol"]),
            str(row["language"]),
            str(row["precision"]),
            str(row["model_key"]),
            -int(row["embedding_dimension"] or 0),
            str(row["run_id"]),
        )
    )
    status_rows = _status_rows(results)
    status_rows.sort(
        key=lambda row: (
            str(row["status"]),
            str(row["dataset_key"]),
            str(row["protocol"]),
            str(row["language"]),
            str(row["model_key"]),
            -int(row["embedding_dimension"] or 0),
            str(row["run_id"]),
        )
    )
    summary_path = output_dir / "summary.csv"
    non_success_path = output_dir / "non-success.csv"
    metadata_path = output_dir / "metadata.json"
    report_path = output_dir / "report.md"
    _write_csv(summary_path, SUMMARY_FIELDS, completed_rows)
    _write_csv(non_success_path, NON_SUCCESS_FIELDS, status_rows)
    atomic_write_json(
        metadata_path,
        _metadata_document(results, publication_states, publication_lifecycle),
    )
    plots = write_plots(results, output_dir / "plots")
    dimension_files = write_embedding_dimension_analysis(
        results,
        output_dir,
        baseline_comparison,
        native_validation_report,
    )
    report_path.write_text(
        render_report(
            results,
            plots,
            output_dir / "validation",
            publication_lifecycle,
            native_validation_report,
        ),
        encoding="utf-8",
        newline="\n",
    )
    return {
        "summary_csv": summary_path,
        "non_success_csv": non_success_path,
        "metadata_json": metadata_path,
        "report_markdown": report_path,
        **dimension_files,
        **plots,
    }
