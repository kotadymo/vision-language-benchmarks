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

from vision_language_benchmarks.registry import get_dataset, get_model
from vision_language_benchmarks.reporting.comparison import paired_language_results

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
    key = _configuration(result).get("model_key")
    if isinstance(key, str):
        try:
            return get_model(key).name
        except ValueError:
            pass
    return str(result["model"])


def _dataset_label(result: Result) -> str:
    key = str(result["dataset"])
    try:
        return get_dataset(key).name
    except ValueError:
        return key


def _condition_label(result: Result) -> str:
    metadata = _metadata(result)
    configuration = _configuration(result)
    return (
        f"{_model_label(result)} · {_dataset_label(result)} "
        f"({metadata.get('dataset_split', 'split unknown')}) · "
        f"{result.get('language', 'language unknown')} · "
        f"{str(configuration.get('protocol', 'protocol unknown')).upper()} · "
        f"{str(metadata.get('precision', 'precision unknown')).upper()}"
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


def _classification_plots(results: list[Result], output_dir: Path) -> dict[str, Path]:
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
        title = (
            f"Zero-shot classification Top-1 — {protocol.upper()} / "
            f"{precision.upper()} / {language.upper()}"
        )
        labels = [_condition_label(result) for result in eligible]
        values = [float(result["metrics"]["acc1"]) * 100 for result in eligible]
        datasets = [_dataset_label(result) for result in eligible]
        unique_datasets = list(dict.fromkeys(datasets))
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


def _retrieval_plots(results: list[Result], output_dir: Path) -> dict[str, Path]:
    grouped: dict[tuple[str, str, str], dict[str, list[Result]]] = defaultdict(
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
        )
        grouped[key][str(result.get("language", ""))].append(result)

    files: dict[str, Path] = {}
    for (dataset_key, protocol, precision), language_results in sorted(grouped.items()):
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
            f"Bidirectional retrieval recall — {dataset_name} ({split}) · "
            f"{protocol.upper()} / {precision.upper()}"
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
        stem = (
            "retrieval_recall"
            if dataset_key == "crossmodal3600" and protocol == "p2" and precision == "fp32"
            else (
                f"retrieval_recall_{_slug(dataset_key)}_"
                f"{_slug(protocol)}_{_slug(precision)}"
            )
        )
        files.update(_save_figure(figure, output_dir, stem))
    return files


def _language_comparison_plots(results: list[Result], output_dir: Path) -> dict[str, Path]:
    grouped: dict[tuple[str, str, str], list[tuple[Result, Result]]] = defaultdict(list)
    for english, japanese in paired_language_results(results):
        configuration = _configuration(english)
        key = (
            str(english["dataset"]),
            str(configuration.get("protocol", "")),
            str(_metadata(english).get("precision", "")),
        )
        grouped[key].append((english, japanese))

    files: dict[str, Path] = {}
    for (dataset_key, protocol, precision), pairs in sorted(grouped.items()):
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
        axis.set_title(
            f"{dataset_name} — {protocol.upper()} / {precision.upper()} "
            "(matched image IDs and run conditions)"
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
        stem = (
            "multilingual_comparison"
            if dataset_key == "crossmodal3600" and protocol == "p2" and precision == "fp32"
            else (
                f"multilingual_comparison_{_slug(dataset_key)}_"
                f"{_slug(protocol)}_{_slug(precision)}"
            )
        )
        files.update(_save_figure(figure, output_dir, stem))
    return files


def _throughput_plots(results: list[Result], output_dir: Path) -> dict[str, Path]:
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
        title = (
            f"Zero-shot accuracy versus image throughput — {protocol.upper()} / "
            f"{precision.upper()} / {language.upper()}"
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


def write_plots(results: list[Result], output_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    files.update(_classification_plots(results, output_dir))
    files.update(_retrieval_plots(results, output_dir))
    files.update(_language_comparison_plots(results, output_dir))
    files.update(_throughput_plots(results, output_dir))
    return files
