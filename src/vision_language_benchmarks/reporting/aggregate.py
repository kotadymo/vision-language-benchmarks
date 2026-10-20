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
from vision_language_benchmarks.reporting.plots import write_plots
from vision_language_benchmarks.reporting.retrieval_variant import (
    retrieval_variant_info,
    retrieval_variant_label,
)
from vision_language_benchmarks.reporting.schema import validate_result

SUMMARY_FIELDS = (
    "run_id",
    "model_key",
    "model",
    "model_checkpoint_filename",
    "model_checkpoint_size_bytes",
    "model_parameter_count",
    "model_languages",
    "model_license",
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
NON_SUCCESS_FIELDS = (
    "run_id",
    "status",
    "model_key",
    "model",
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


def summary_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
            "model_key": configuration.get("model_key", ""),
            "model": _model_name(result),
            "model_checkpoint_filename": metadata.get("model_checkpoint_filename"),
            "model_checkpoint_size_bytes": metadata.get("model_checkpoint_size_bytes"),
            "model_parameter_count": metadata.get("parameter_count"),
            "model_languages": ", ".join(metadata.get("model_languages") or []),
            "model_license": metadata.get("model_license"),
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
                "model": _model_name(result),
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
            "| Model | Parameters | Checkpoint | Dataset | Split | Images | Classes | "
            "Requested / image / text batch | Top-1 | Top-5 | Images/s | Median (ms) | "
            "Peak allocated / reserved (GiB) | OOM retries | Raw result |",
            "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
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
        "| Model | Parameters | Checkpoint | Dataset | Split | Language | Retrieval variant | "
        "Images / captions | "
        "Requested / image / text batch | "
        "T→I R@1 | T→I R@5 | T→I R@10 | I→T R@1 | I→T R@5 | I→T R@10 | "
        "Images/s | Texts/s | Median (ms) | Peak allocated / reserved (GiB) | "
        "OOM retries | Raw result |",
        "|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
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
                "| Model | Split | Images | Captions (EN / JA) | T→I R@1 (EN → JA; Δ pp) | "
                "T→I R@5 (EN → JA; Δ pp) | T→I R@10 (EN → JA; Δ pp) | "
                "I→T R@1 (EN → JA; Δ pp) | I→T R@5 (EN → JA; Δ pp) | "
                "I→T R@10 (EN → JA; Δ pp) |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for english, japanese in sorted(pairs, key=lambda pair: _model_name(pair[0])):
            english_metadata = english["benchmark_metadata"]
            japanese_metadata = japanese["benchmark_metadata"]
            english_metrics = english["metrics"]
            japanese_metrics = japanese["metrics"]
            values = (
                _model_name(english),
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
        "## English zero-shot classification",
        "",
    ]
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
                    "Reason | Run ID |",
                    "|---|---|---|---|---|---|---|",
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


def _metadata_document(results: list[dict[str, Any]]) -> dict[str, Any]:
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
    return {
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
                "status": result["status"],
                "model": result["model"],
                "dataset": result["dataset"],
                "language": result["language"],
                "protocol": _configuration(result).get("protocol"),
                "precision": result["benchmark_metadata"].get("precision"),
            }
            for result in results
        ],
    }


def generate_reports(input_path: Path, output_dir: Path) -> dict[str, Path]:
    results = load_results(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_rows = summary_rows(results)
    completed_rows.sort(
        key=lambda row: (
            str(row["dataset_key"]),
            str(row["protocol"]),
            str(row["language"]),
            str(row["precision"]),
            str(row["model_key"]),
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
            str(row["run_id"]),
        )
    )
    summary_path = output_dir / "summary.csv"
    non_success_path = output_dir / "non-success.csv"
    metadata_path = output_dir / "metadata.json"
    report_path = output_dir / "report.md"
    _write_csv(summary_path, SUMMARY_FIELDS, completed_rows)
    _write_csv(non_success_path, NON_SUCCESS_FIELDS, status_rows)
    atomic_write_json(metadata_path, _metadata_document(results))
    plots = write_plots(results, output_dir / "plots")
    report_path.write_text(
        render_report(results, plots, output_dir / "validation"),
        encoding="utf-8",
        newline="\n",
    )
    return {
        "summary_csv": summary_path,
        "non_success_csv": non_success_path,
        "metadata_json": metadata_path,
        "report_markdown": report_path,
        **plots,
    }
