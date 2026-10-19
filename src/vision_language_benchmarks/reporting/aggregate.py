from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import jsonschema

from vision_language_benchmarks.atomic import atomic_write_json
from vision_language_benchmarks.errors import DataIntegrityError
from vision_language_benchmarks.registry import get_dataset, get_model
from vision_language_benchmarks.reporting.plots import write_plots
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
    "precision",
    "device",
    "gpu_name",
    "image_count",
    "text_count",
    "class_count",
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
            "precision": metadata["precision"],
            "device": metadata["device"],
            "gpu_name": metadata.get("gpu_name"),
            "image_count": metadata.get("image_count"),
            "text_count": metadata.get("text_count"),
            "class_count": metadata.get("class_count"),
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


def _render_measurement_table(results: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Model | Parameters | Checkpoint | Supported languages | Dataset | Split | Language | "
        "Top-1 / T→I R@1 | I→T R@1 | Images/s | Run ID |",
        "|---|---:|---:|---|---|---|---|---:|---:|---:|---|",
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
        supported_languages = ", ".join(metadata.get("model_languages") or [])
        acc1 = _percentage(metrics, "acc1")
        text_to_image = _percentage(metrics, "image_retrieval_recall@1")
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(value)
                for value in (
                    _model_name(result),
                    parameter_size,
                    checkpoint_size_text,
                    supported_languages,
                    _dataset_name(result),
                    metadata["dataset_split"],
                    result["language"],
                    acc1 if acc1 != "—" else text_to_image,
                    _percentage(metrics, "text_retrieval_recall@1"),
                    metadata.get("throughput_images_per_sec"),
                    metadata["run_id"],
                )
            )
            + " |"
        )
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


def render_report(results: list[dict[str, Any]]) -> str:
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
        "- Protocols and precisions are kept in separate tables and chart files.",
        "- Accuracy and recall values in JSON are fractions; report tables display percentages.",
        "- Checkpoint sizes are pinned weight-file sizes; tokenizer and configuration files "
        "may add to the download.",
        "",
        "## English classification",
        "",
    ]
    lines.extend(
        _render_task_language_section(
            completed,
            task="classification",
            language="en",
            title="English classification",
        )[2:]
    )
    lines.extend(
        _render_task_language_section(
            completed,
            task="retrieval",
            language="en",
            title="English retrieval",
        )
    )
    lines.extend(
        _render_task_language_section(
            completed,
            task="classification",
            language="ja",
            title="Japanese classification",
        )
    )
    lines.extend(
        _render_task_language_section(
            completed,
            task="retrieval",
            language="ja",
            title="Japanese retrieval",
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
                    "| Model | Dataset | Language | Protocol | Reason | Run ID |",
                    "|---|---|---|---|---|---|",
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
                            "reason",
                            "run_id",
                        )
                    )
                    + " |"
                )
            lines.append("")
    else:
        lines.extend(["No unsuccessful runs were recorded.", ""])

    lines.extend(
        [
            "## Limitations",
            "",
            "- A completed run is not evidence of numerical parity with CLIP Benchmark; "
            "parity must be tested under identical model, weights, inputs, preprocessing, "
            "templates, and aggregation.",
            "- Training-data contamination cannot be ruled out unless a model's pretraining "
            "corpus is independently verified.",
            "- Dataset and model access restrictions remain in force. See "
            "[datasets-and-licenses.md](../../docs/datasets-and-licenses.md).",
            "- Missing metrics remain missing. Unsupported languages and unavailable datasets "
            "are not represented as zero scores.",
            "",
            "## Reproduction",
            "",
            "```powershell",
            "uv sync --locked --all-extras --group dev",
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
            "- [Classification Top-1](plots/)",
            "- [Bidirectional retrieval recall](plots/)",
            "- [English/Japanese paired retrieval](plots/)",
            "- [Accuracy versus throughput](plots/)",
            "",
        ]
    )
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
                "image_count": metadata.get("image_count"),
                "text_count": metadata.get("text_count"),
                "class_count": metadata.get("class_count"),
            },
        )
        dataset["terms_confirmations"].append(
            {
                "run_id": metadata["run_id"],
                "confirmed": metadata.get("dataset_terms_confirmed"),
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
    report_path.write_text(render_report(results), encoding="utf-8", newline="\n")
    plots = write_plots(results, output_dir / "plots")
    return {
        "summary_csv": summary_path,
        "non_success_csv": non_success_path,
        "metadata_json": metadata_path,
        "report_markdown": report_path,
        **plots,
    }
