from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from vision_language_benchmarks.embedding_dimensions import EMBEDDINGGEMMA_MODEL_KEY
from vision_language_benchmarks.registry import get_dataset, get_model
from vision_language_benchmarks.reporting.comparison import paired_language_results
from vision_language_benchmarks.reporting.retrieval_variant import (
    retrieval_variant_info,
    retrieval_variant_label,
)

matplotlib.rcParams["svg.hashsalt"] = "vision-language-benchmarks-v0.1.0"

Result = dict[str, Any]
RETRIEVAL_SERIES = (
    ("image_retrieval_recall@1", "T→I R@1"),
    ("image_retrieval_recall@5", "T→I R@5"),
    ("image_retrieval_recall@10", "T→I R@10"),
    ("text_retrieval_recall@1", "I→T R@1"),
    ("text_retrieval_recall@5", "I→T R@5"),
    ("text_retrieval_recall@10", "I→T R@10"),
)


def _metadata(result: Result) -> dict[str, Any]:
    return result["benchmark_metadata"]


def _configuration(result: Result) -> dict[str, Any]:
    value = _metadata(result).get("run_configuration", {})
    return value if isinstance(value, dict) else {}


def _model_label(result: Result) -> str:
    configuration = _configuration(result)
    key = configuration.get("model_key")
    dimension = configuration.get(
        "embedding_dimension",
        _metadata(result).get("embedding_dimension"),
    )
    if dimension is None and key == EMBEDDINGGEMMA_MODEL_KEY:
        dimension = 768
    if dimension is None and isinstance(key, str):
        try:
            dimension = get_model(key).global_embedding_dimension
        except ValueError:
            pass
    suffix = f" · {dimension}D" if isinstance(dimension, int) else ""
    if isinstance(key, str):
        try:
            return f"{get_model(key).name}{suffix}"
        except ValueError:
            pass
    return f"{result['model']}{suffix}"


def _dataset_label(result: Result) -> str:
    key = str(result["dataset"])
    try:
        return get_dataset(key).name
    except ValueError:
        return key


def _condition_label(result: Result) -> str:
    metadata = _metadata(result)
    configuration = _configuration(result)
    variant, _ = retrieval_variant_info(result)
    variant_label = (
        f" · {retrieval_variant_label(variant)}"
        if result.get("task") == "retrieval"
        else ""
    )
    return (
        f"{_model_label(result)} · {_dataset_label(result)} "
        f"({metadata.get('dataset_split', 'split unknown')}) · "
        f"{result.get('language', 'language unknown')} · "
        f"{str(configuration.get('protocol', 'protocol unknown')).upper()} · "
        f"{str(metadata.get('precision', 'precision unknown')).upper()}"
        f"{variant_label}"
    )


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "unspecified"


def _save_figure(
    figure: Figure,
    output_dir: Path,
    stem: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"
    figure.savefig(
        png_path,
        dpi=150,
        bbox_inches="tight",
        metadata={"Software": "vision-language-benchmarks"},
    )
    figure.savefig(
        svg_path,
        bbox_inches="tight",
        metadata={"Creator": "vision-language-benchmarks", "Date": None},
    )
    svg_lines = svg_path.read_text(encoding="utf-8").splitlines()
    with svg_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(line.rstrip() for line in svg_lines) + "\n")
    plt.close(figure)
    return {f"{stem}_png": png_path, f"{stem}_svg": svg_path}


def _condition_groups(
    results: list[Result],
    *,
    task: str,
    language: str | None = None,
) -> list[tuple[tuple[str, ...], list[Result]]]:
    grouped: dict[tuple[str, ...], list[Result]] = defaultdict(list)
    for result in results:
        if result.get("status") != "completed" or result.get("task") != task:
            continue
        if language is not None and result.get("language") != language:
            continue
        metadata = _metadata(result)
        configuration = _configuration(result)
        key = (
            str(configuration.get("protocol", "")),
            str(metadata.get("precision", "")),
            str(result.get("language", "")),
        )
        grouped[key].append(result)
    return [
        (key, sorted(items, key=lambda item: _metadata(item).get("run_id", "")))
        for key, items in sorted(grouped.items())
    ]


def _classification_plots(
    results: list[Result],
    output_dir: Path,
    *,
    filename_prefix: str = "",
) -> dict[str, Path]:
    groups = _condition_groups(results, task="classification")
    files: dict[str, Path] = {}
    for (protocol, precision, language), items in groups:
        eligible = [
            result
            for result in items
            if isinstance(result.get("metrics", {}).get("acc1"), (int, float))
        ]
        if not eligible:
            continue
        if protocol == "p0" and precision == "fp32" and language == "en":
            stem = "classification_accuracy"
        elif protocol == "p1" and precision == "fp32" and language == "en":
            stem = "classification_accuracy_p1"
        else:
            stem = (
                f"classification_accuracy_{_slug(protocol)}_"
                f"{_slug(precision)}_{_slug(language)}"
            )
        stem = f"{filename_prefix}{stem}"
        datasets = [_dataset_label(result) for result in eligible]
        unique_datasets = list(dict.fromkeys(datasets))
        title = (
            f"Zero-shot classification Top-1 — {protocol.upper()} / "
            f"{precision.upper()} / {language.upper()} · "
            f"datasets: {', '.join(unique_datasets)} · retrieval variant: N/A"
        )
        labels = [_condition_label(result) for result in eligible]
        values = [float(result["metrics"]["acc1"]) * 100 for result in eligible]
        palette = plt.get_cmap("tab10")
        colors = [palette(unique_datasets.index(dataset) % 10) for dataset in datasets]
        figure_height = max(4.5, 0.42 * len(eligible) + 1.5)
        figure, axis = plt.subplots(figsize=(12, figure_height))
        positions = np.arange(len(eligible))
        axis.barh(positions, values, color=colors)
        axis.set_yticks(positions, labels=labels, fontsize=8)
        axis.invert_yaxis()
        axis.set_xlim(0, 100)
        axis.set_xlabel("Top-1 accuracy (%)")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
        if len(unique_datasets) > 1:
            handles = [
                plt.Rectangle((0, 0), 1, 1, color=palette(index % 10))
                for index in range(len(unique_datasets))
            ]
            axis.legend(handles, unique_datasets, title="Dataset", loc="lower right")
        figure.tight_layout()
        files.update(_save_figure(figure, output_dir, stem))
    return files


def _retrieval_plots(
    results: list[Result],
    output_dir: Path,
    *,
    filename_prefix: str = "",
) -> dict[str, Path]:
    grouped: dict[tuple[str, str, str, str], dict[str, list[Result]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for result in results:
        if result.get("status") != "completed" or result.get("task") != "retrieval":
            continue
        configuration = _configuration(result)
        metrics = result.get("metrics", {})
        if not any(
            isinstance(metrics.get(key), (int, float))
            for key, _ in RETRIEVAL_SERIES
        ):
            continue
        key = (
            str(result["dataset"]),
            str(configuration.get("protocol", "")),
            str(_metadata(result).get("precision", "")),
            retrieval_variant_info(result)[0] or "unspecified",
        )
        grouped[key][str(result.get("language", ""))].append(result)

    files: dict[str, Path] = {}
    for (dataset_key, protocol, precision, variant), language_results in sorted(
        grouped.items()
    ):
        languages = [
            (
                language,
                sorted(items, key=lambda item: _model_label(item)),
            )
            for language, items in sorted(language_results.items())
            if items
        ]
        if not languages:
            continue

        figure_height = max(5.0, 0.65 * max(len(items) for _, items in languages) + 3.0)
        figure, axes = plt.subplots(
            1,
            len(languages),
            figsize=(8.0 * len(languages), figure_height),
            sharex=True,
            squeeze=False,
        )
        palette = plt.get_cmap("tab10")
        available_series = [
            (metric_key, label)
            for metric_key, label in RETRIEVAL_SERIES
            if any(
                isinstance(result.get("metrics", {}).get(metric_key), (int, float))
                for _, items in languages
                for result in items
            )
        ]
        for axis, (language, items) in zip(axes[0], languages, strict=True):
            positions = np.arange(len(items))
            bar_height = 0.8 / len(available_series)
            for series_index, (metric_key, series_label) in enumerate(available_series):
                offset = (series_index - (len(available_series) - 1) / 2) * bar_height
                for result_index, result in enumerate(items):
                    value = result.get("metrics", {}).get(metric_key)
                    if not isinstance(value, (int, float)):
                        continue
                    axis.barh(
                        positions[result_index] + offset,
                        float(value) * 100,
                        height=bar_height,
                        color=palette(RETRIEVAL_SERIES.index((metric_key, series_label))),
                    )

            metadata = _metadata(items[0])
            image_count = metadata.get("image_count")
            text_count = metadata.get("text_count")
            count_label = (
                f"{int(image_count):,} images"
                if isinstance(image_count, int)
                else "image count unavailable"
            )
            if isinstance(text_count, int):
                count_label += f" · {text_count:,} captions"
            axis.set_yticks(
                positions,
                labels=[_model_label(result) for result in items],
                fontsize=8,
            )
            axis.invert_yaxis()
            axis.set_xlim(0, 100)
            axis.set_xlabel("Recall (%)")
            axis.set_title(f"{language.upper()} · {count_label}")
            axis.grid(axis="x", alpha=0.25)

        dataset_name = _dataset_label(languages[0][1][0])
        split = _metadata(languages[0][1][0]).get("dataset_split", "split unknown")
        figure.suptitle(
            f"Bidirectional Recall@1/5/10 — {dataset_name} ({split}) · "
            f"{protocol.upper()} / {precision.upper()} · variant: "
            f"{retrieval_variant_label(None if variant == 'unspecified' else variant)}"
        )
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=palette(RETRIEVAL_SERIES.index(item)))
            for item in available_series
        ]
        figure.legend(
            handles,
            [label for _, label in available_series],
            title="Query direction / cutoff",
            ncol=3,
            loc="lower center",
        )
        figure.tight_layout(rect=(0, 0.1, 1, 0.93))
        if dataset_key == "crossmodal3600" and protocol == "p2" and precision == "fp32":
            stem = {
                "official_first_caption": "xm3600_official_retrieval",
                "all_captions": "xm3600_all_captions_retrieval",
            }.get(
                variant,
                f"xm3600_retrieval_{_slug(variant)}",
            )
        else:
            stem = (
                f"retrieval_recall_{_slug(dataset_key)}_{_slug(protocol)}_"
                f"{_slug(precision)}_{_slug(variant)}"
            )
        stem = f"{filename_prefix}{stem}"
        files.update(_save_figure(figure, output_dir, stem))
    return files


def _language_comparison_plots(results: list[Result], output_dir: Path) -> dict[str, Path]:
    grouped: dict[tuple[str, str, str, str], list[tuple[Result, Result]]] = defaultdict(list)
    for english, japanese in paired_language_results(results):
        configuration = _configuration(english)
        key = (
            str(english["dataset"]),
            str(configuration.get("protocol", "")),
            str(_metadata(english).get("precision", "")),
            retrieval_variant_info(english)[0] or "unspecified",
        )
        grouped[key].append((english, japanese))

    files: dict[str, Path] = {}
    for (dataset_key, protocol, precision, variant), pairs in sorted(grouped.items()):
        metric_keys = [metric_key for metric_key, _ in RETRIEVAL_SERIES]
        figure, axis = plt.subplots(figsize=(max(10, 1.8 * len(pairs) + 6), 7))
        positions = np.arange(len(pairs))
        bar_width = 0.8 / len(metric_keys)
        palette = plt.get_cmap("tab10")
        maximum_absolute_delta = 0.0
        plotted_series: list[tuple[int, str]] = []
        for metric_index, (metric_key, label) in enumerate(RETRIEVAL_SERIES):
            plotted = False
            offset = (metric_index - (len(metric_keys) - 1) / 2) * bar_width
            for pair_index, (english, japanese) in enumerate(pairs):
                en_value = english.get("metrics", {}).get(metric_key)
                ja_value = japanese.get("metrics", {}).get(metric_key)
                if not isinstance(en_value, (int, float)) or not isinstance(
                    ja_value, (int, float)
                ):
                    continue
                delta = (float(ja_value) - float(en_value)) * 100
                maximum_absolute_delta = max(maximum_absolute_delta, abs(delta))
                x_position = positions[pair_index] + offset
                axis.bar(
                    x_position,
                    delta,
                    width=bar_width,
                    color=palette(metric_index),
                )
                axis.annotate(
                    f"{delta:+.1f}",
                    (x_position, delta),
                    xytext=(0, 3 if delta >= 0 else -3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if delta >= 0 else "top",
                    fontsize=7,
                )
                plotted = True
            if plotted:
                plotted_series.append((metric_index, label))

        if not plotted_series:
            plt.close(figure)
            continue

        model_labels = [_model_label(english) for english, _ in pairs]
        dataset_name = _dataset_label(pairs[0][0])
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(positions, labels=model_labels, fontsize=8)
        axis.set_ylabel("Japanese − English recall (percentage points)")
        candidate_count = _metadata(pairs[0][0]).get("text_count")
        axis.set_title(
            f"{dataset_name} — {protocol.upper()} / {precision.upper()} "
            f"· variant: {retrieval_variant_label(None if variant == 'unspecified' else variant)} "
            f"· {candidate_count:,} candidates per language "
            "(Japanese − English; matched image IDs and run conditions)"
        )
        axis.grid(axis="y", alpha=0.25)
        limit = max(2.0, maximum_absolute_delta * 1.25)
        axis.set_ylim(-limit, limit)
        axis.legend(
            [
                plt.Rectangle((0, 0), 1, 1, color=palette(index))
                for index, _ in plotted_series
            ],
            [label for _, label in plotted_series],
            title="Query direction / cutoff",
            ncol=3,
            loc="best",
        )
        figure.tight_layout()
        if dataset_key == "crossmodal3600" and protocol == "p2" and precision == "fp32":
            stem = {
                "official_first_caption": (
                    "xm3600_language_comparison_official_first_caption"
                ),
                "all_captions": "xm3600_language_comparison_all_captions",
            }.get(
                variant,
                f"xm3600_language_comparison_{_slug(variant)}",
            )
        else:
            stem = (
                f"multilingual_comparison_{_slug(dataset_key)}_{_slug(protocol)}_"
                f"{_slug(precision)}_{_slug(variant)}"
            )
        files.update(_save_figure(figure, output_dir, stem))
    return files


def _throughput_plots(
    results: list[Result],
    output_dir: Path,
    *,
    filename_prefix: str = "",
) -> dict[str, Path]:
    groups = _condition_groups(results, task="classification")
    files: dict[str, Path] = {}
    for (protocol, precision, language), items in groups:
        eligible = [
            result
            for result in items
            if isinstance(result.get("metrics", {}).get("acc1"), (int, float))
            and isinstance(_metadata(result).get("throughput_images_per_sec"), (int, float))
            and _metadata(result)["throughput_images_per_sec"] > 0
        ]
        if not eligible:
            continue
        if protocol == "p0" and precision == "fp32" and language == "en":
            stem = "performance_comparison"
        elif protocol == "p1" and precision == "fp32" and language == "en":
            stem = "performance_comparison_p1"
        else:
            stem = (
                f"performance_comparison_{_slug(protocol)}_"
                f"{_slug(precision)}_{_slug(language)}"
            )
        stem = f"{filename_prefix}{stem}"
        dataset_labels = list(dict.fromkeys(_dataset_label(result) for result in eligible))
        title = (
            f"Zero-shot accuracy versus image throughput — {protocol.upper()} / "
            f"{precision.upper()} / {language.upper()} · "
            f"datasets: {', '.join(dataset_labels)} · retrieval variant: N/A"
        )
        figure, axis = plt.subplots(figsize=(11, 7))
        for result in eligible:
            metadata = _metadata(result)
            axis.scatter(
                float(metadata["throughput_images_per_sec"]),
                float(result["metrics"]["acc1"]) * 100,
                label=_condition_label(result),
            )
        axis.set_xlabel("Image encoder throughput (images/s)")
        axis.set_ylabel("Zero-shot Top-1 accuracy (%)")
        axis.set_title(title)
        axis.set_ylim(0, 100)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, loc="best")
        figure.tight_layout()
        files.update(_save_figure(figure, output_dir, stem))
    return files


def _embedding_dimension_plots(
    results: list[Result],
    output_dir: Path,
) -> dict[str, Path]:
    dimension_results = [
        result
        for result in results
        if result.get("status") == "completed"
        and isinstance(_metadata(result).get("embedding_dimension"), int)
    ]
    if not dimension_results:
        return {}

    files: dict[str, Path] = {}
    classification_groups: dict[tuple[str, str, str, str, str], list[Result]] = defaultdict(
        list
    )
    retrieval_groups: dict[tuple[str, str, str, str, str, str], list[Result]] = defaultdict(
        list
    )
    for result in dimension_results:
        configuration = _configuration(result)
        metadata = _metadata(result)
        group_key = (
            str(configuration.get("model_key", result["model"])),
            str(result["dataset"]),
            str(configuration.get("protocol", "")),
            str(result["language"]),
            str(metadata.get("precision", "")),
        )
        if result.get("task") == "classification":
            classification_groups[group_key].append(result)
        elif result.get("task") == "retrieval":
            retrieval_groups[
                (
                    *group_key,
                    retrieval_variant_info(result)[0] or "unspecified",
                )
            ].append(result)

    for (model_key, dataset_key, protocol, language, precision), items in sorted(
        classification_groups.items()
    ):
        ordered = sorted(
            items,
            key=lambda result: int(_metadata(result)["embedding_dimension"]),
            reverse=True,
        )
        dimensions = [
            int(_metadata(result)["embedding_dimension"])
            for result in ordered
        ]
        figure, axis = plt.subplots(figsize=(8, 5))
        for metric, label in (("acc1", "Top-1"), ("acc5", "Top-5")):
            values = [
                result.get("metrics", {}).get(metric)
                for result in ordered
            ]
            if all(isinstance(value, (int, float)) for value in values):
                axis.plot(
                    dimensions,
                    [float(value) * 100 for value in values],
                    marker="o",
                    label=label,
                )
        axis.set_xticks(dimensions)
        axis.set_xlabel("Embedding dimension (FP32)")
        axis.set_ylabel("Accuracy (%)")
        axis.set_ylim(0, 100)
        axis.set_title(
            f"{get_model(model_key).name} · {_dataset_label(ordered[0])} · "
            f"{protocol.upper()} · {language.upper()} · {precision.upper()}"
        )
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        files.update(
            _save_figure(
                figure,
                output_dir,
                f"embedding_dimension_{_slug(dataset_key)}_{_slug(protocol)}_"
                f"{_slug(language)}_classification",
            )
        )

    for (
        model_key,
        dataset_key,
        protocol,
        language,
        precision,
        variant,
    ), items in sorted(retrieval_groups.items()):
        ordered = sorted(
            items,
            key=lambda result: int(_metadata(result)["embedding_dimension"]),
            reverse=True,
        )
        dimensions = [
            int(_metadata(result)["embedding_dimension"])
            for result in ordered
        ]
        figure, axis = plt.subplots(figsize=(10, 6))
        plotted = False
        for metric, label in RETRIEVAL_SERIES:
            values = [
                result.get("metrics", {}).get(metric)
                for result in ordered
            ]
            if not all(isinstance(value, (int, float)) for value in values):
                continue
            axis.plot(
                dimensions,
                [float(value) * 100 for value in values],
                marker="o",
                label=label,
            )
            plotted = True
        if not plotted:
            plt.close(figure)
            continue
        axis.set_xticks(dimensions)
        axis.set_xlabel("Embedding dimension (FP32)")
        axis.set_ylabel("Recall (%)")
        axis.set_ylim(0, 100)
        axis.set_title(
            f"{get_model(model_key).name} · {_dataset_label(ordered[0])} · "
            f"{language.upper()} · {protocol.upper()} · {precision.upper()} · "
            f"{retrieval_variant_label(None if variant == 'unspecified' else variant)}"
        )
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
        figure.tight_layout()
        files.update(
            _save_figure(
                figure,
                output_dir,
                f"embedding_dimension_{_slug(dataset_key)}_{_slug(language)}_"
                f"{_slug(variant)}",
            )
        )

    dimensions = list(
        dict.fromkeys(
            int(_metadata(result)["embedding_dimension"])
            for result in dimension_results
        )
    )
    dimensions.sort(reverse=True)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(
        [str(dimension) for dimension in dimensions],
        [dimension / 768 * 100 for dimension in dimensions],
        color=plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(dimensions))),
    )
    axis.set_xlabel("Embedding dimension")
    axis.set_ylabel("FP32 vector storage relative to 768D (%)")
    axis.set_title("Embedding vector size (excluding weights and index overhead)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    files.update(_save_figure(figure, output_dir, "embedding_dimension_storage"))
    return files


def write_plots(results: list[Result], output_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    files.update(_classification_plots(results, output_dir))
    files.update(_retrieval_plots(results, output_dir))
    files.update(_language_comparison_plots(results, output_dir))
    files.update(_throughput_plots(results, output_dir))
    files.update(_embedding_dimension_plots(results, output_dir))
    return files


def write_model_comparison_plots(
    results: list[Result],
    output_dir: Path,
) -> dict[str, Path]:
    filename_prefix = "model_comparison_"
    files: dict[str, Path] = {}
    files.update(
        _classification_plots(
            results,
            output_dir,
            filename_prefix=filename_prefix,
        )
    )
    files.update(
        _retrieval_plots(
            results,
            output_dir,
            filename_prefix=filename_prefix,
        )
    )
    files.update(
        _throughput_plots(
            results,
            output_dir,
            filename_prefix=filename_prefix,
        )
    )
    return files
