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
    plt.close(figure)
    return {f"{stem}_png": png_path, f"{stem}_svg": svg_path}


def _empty_figure(title: str, message: str) -> Figure:
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    axis.set_axis_off()
    figure.tight_layout()
    return figure


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
        stem = f"classification-top1-{_slug(protocol)}-{_slug(precision)}-{_slug(language)}"
        title = (
            f"Zero-shot classification Top-1 — {protocol.upper()} / "
            f"{precision.upper()} / {language.upper()}"
        )
        if not eligible:
            files.update(
                _save_figure(
                    _empty_figure(
                        title,
                        "No completed Top-1 measurements are available. "
                        "Dataset, split, and model are not represented as zero.",
                    ),
                    output_dir,
                    stem,
                )
            )
            continue
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
    if not groups:
        stem = "classification-top1-no-results"
        files.update(
            _save_figure(
                _empty_figure(
                    "Zero-shot classification Top-1",
                    "No completed classification runs. Dataset, split, model, protocol, "
                    "and precision are missing, not zero.",
                ),
                output_dir,
                stem,
            )
        )
    return files


def _retrieval_plots(results: list[Result], output_dir: Path) -> dict[str, Path]:
    groups = _condition_groups(results, task="retrieval")
    files: dict[str, Path] = {}
    for (protocol, precision, language), items in groups:
        eligible = [
            result
            for result in items
            if any(
                isinstance(result.get("metrics", {}).get(key), (int, float))
                for key, _ in RETRIEVAL_SERIES
            )
        ]
        stem = f"retrieval-recall-{_slug(protocol)}-{_slug(precision)}-{_slug(language)}"
        title = (
            f"Bidirectional retrieval recall — {protocol.upper()} / "
            f"{precision.upper()} / {language.upper()}"
        )
        if not eligible:
            files.update(
                _save_figure(
                    _empty_figure(
                        title,
                        "No completed retrieval measurements are available. "
                        "Missing values are not plotted as zero.",
                    ),
                    output_dir,
                    stem,
                )
            )
            continue
        labels = [_condition_label(result) for result in eligible]
        group_positions = np.arange(len(eligible))
        bar_height = 0.8 / len(RETRIEVAL_SERIES)
        figure_height = max(5.0, 0.48 * len(eligible) + 1.5)
        figure, axis = plt.subplots(figsize=(13, figure_height))
        palette = plt.get_cmap("tab10")
        for series_index, (metric_key, series_label) in enumerate(RETRIEVAL_SERIES):
            offset = (series_index - (len(RETRIEVAL_SERIES) - 1) / 2) * bar_height
            label_added = False
            for result_index, result in enumerate(eligible):
                value = result.get("metrics", {}).get(metric_key)
                if not isinstance(value, (int, float)):
                    continue
                axis.barh(
                    group_positions[result_index] + offset,
                    float(value) * 100,
                    height=bar_height,
                    color=palette(series_index),
                    label=series_label if not label_added else None,
                )
                label_added = True
        axis.set_yticks(group_positions, labels=labels, fontsize=8)
        axis.invert_yaxis()
        axis.set_xlim(0, 100)
        axis.set_xlabel("Recall (%)")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
        axis.legend(title="Query direction / cutoff", ncol=3, loc="lower right")
        figure.tight_layout()
        files.update(_save_figure(figure, output_dir, stem))
    if not groups:
        stem = "retrieval-recall-no-results"
        files.update(
            _save_figure(
                _empty_figure(
                    "Bidirectional retrieval recall",
                    "No completed retrieval runs. Dataset, split, model, protocol, "
                    "precision, and recall values are missing, not zero.",
                ),
                output_dir,
                stem,
            )
        )
    return files


def _paired_language_groups(
    results: list[Result],
) -> list[tuple[tuple[str, str], list[tuple[Result, Result]]]]:
    grouped: dict[tuple[str, ...], dict[str, list[Result]]] = defaultdict(
        lambda: {"en": [], "ja": []}
    )
    for result in results:
        if (
            result.get("status") != "completed"
            or result.get("task") != "retrieval"
            or result.get("language") not in {"en", "ja"}
        ):
            continue
        metadata = _metadata(result)
        image_order_hash = metadata.get("image_id_order_hash")
        if not isinstance(image_order_hash, str) or not image_order_hash:
            continue
        configuration = _configuration(result)
        key = (
            str(result["dataset"]),
            str(configuration.get("model_key", result["model"])),
            str(configuration.get("protocol", "")),
            str(metadata.get("precision", "")),
            image_order_hash,
            str(metadata.get("seed", "")),
            str(metadata.get("device", "")),
            str(metadata.get("source_commit_sha", "")),
        )
        grouped[key][result["language"]].append(result)

    result_groups: dict[tuple[str, str], list[tuple[Result, Result]]] = defaultdict(list)
    for key, languages in grouped.items():
        english = languages["en"]
        japanese = languages["ja"]
        if len(english) != 1 or len(japanese) != 1:
            continue
        protocol, precision = key[2], key[3]
        result_groups[(protocol, precision)].append((english[0], japanese[0]))
    return [
        (key, sorted(pairs, key=lambda pair: _condition_label(pair[0])))
        for key, pairs in sorted(result_groups.items())
    ]


def _language_comparison_plots(results: list[Result], output_dir: Path) -> dict[str, Path]:
    groups = _paired_language_groups(results)
    files: dict[str, Path] = {}
    for (protocol, precision), pairs in groups:
        stem = f"english-japanese-recall-paired-{_slug(protocol)}-{_slug(precision)}"
        title = (
            f"English versus Japanese retrieval — {protocol.upper()} / {precision.upper()} "
            "(identical dataset and image-ID order required)"
        )
        figure, axis = plt.subplots(figsize=(max(9, 0.85 * len(pairs) + 4), 6))
        directions = (
            ("image_retrieval_recall@1", "T→I R@1", "o"),
            ("text_retrieval_recall@1", "I→T R@1", "s"),
        )
        plotted = 0
        for english, japanese in pairs:
            if (
                _metadata(english).get("image_id_order_hash")
                != _metadata(japanese).get("image_id_order_hash")
            ):
                continue
            for metric_key, direction, marker in directions:
                en_value = english.get("metrics", {}).get(metric_key)
                ja_value = japanese.get("metrics", {}).get(metric_key)
                if not isinstance(en_value, (int, float)) or not isinstance(
                    ja_value, (int, float)
                ):
                    continue
                axis.plot(
                    (0, 1),
                    (float(en_value) * 100, float(ja_value) * 100),
                    marker=marker,
                    label=f"{_condition_label(english)} · {direction}",
                )
                plotted += 1
        axis.set_xticks((0, 1), labels=("English", "Japanese"))
        axis.set_ylim(0, 100)
        axis.set_ylabel("Recall@1 (%)")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        if plotted:
            axis.legend(fontsize=7, loc="best")
        else:
            axis.text(
                0.5,
                0.5,
                "No complete paired retrieval measurements are available.",
                ha="center",
                va="center",
            )
        figure.tight_layout()
        files.update(_save_figure(figure, output_dir, stem))
    if not groups:
        stem = "english-japanese-recall-paired-no-matching-image-ids"
        files.update(
            _save_figure(
                _empty_figure(
                    "English versus Japanese paired retrieval",
                    "No unique completed English/Japanese result pair has an identical "
                    "dataset, protocol, precision, and image-ID order hash. "
                    "No comparison values are inferred.",
                ),
                output_dir,
                stem,
            )
        )
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
        stem = f"accuracy-throughput-{_slug(protocol)}-{_slug(precision)}-{_slug(language)}"
        title = (
            f"Zero-shot accuracy versus image throughput — {protocol.upper()} / "
            f"{precision.upper()} / {language.upper()}"
        )
        if not eligible:
            files.update(
                _save_figure(
                    _empty_figure(
                        title,
                        "No completed classification result contains both measured "
                        "Top-1 accuracy and positive image throughput.",
                    ),
                    output_dir,
                    stem,
                )
            )
            continue
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
    if not groups:
        stem = "accuracy-throughput-no-results"
        files.update(
            _save_figure(
                _empty_figure(
                    "Zero-shot accuracy versus throughput",
                    "No completed classification runs contain comparable measured "
                    "accuracy and image throughput.",
                ),
                output_dir,
                stem,
            )
        )
    return files


def write_plots(results: list[Result], output_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    files.update(_classification_plots(results, output_dir))
    files.update(_retrieval_plots(results, output_dir))
    files.update(_language_comparison_plots(results, output_dir))
    files.update(_throughput_plots(results, output_dir))
    return files
