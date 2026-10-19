from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vision_language_benchmarks.datasets.prepare import prepare_dataset
from vision_language_benchmarks.errors import BenchmarkError
from vision_language_benchmarks.evaluation.runner import run_benchmark
from vision_language_benchmarks.hardware import doctor_report
from vision_language_benchmarks.paths import RESULTS_ROOT
from vision_language_benchmarks.registry import (
    DATASETS,
    MODELS,
    ProtocolName,
    canonical_language,
    dataset_registry_dict,
    get_dataset,
    get_model,
    language_is_supported,
    model_registry_dict,
)
from vision_language_benchmarks.reporting.aggregate import generate_reports

SUITES: dict[str, dict[str, Any]] = {
    "core-en": {
        "models": (
            "openai_clip_b32",
            "embeddinggemma2",
            "openclip_b32_laion2b",
            "openclip_l14_laion2b",
            "siglip2_base_224",
        ),
        "datasets": ("cifar10", "cifar100", "mscoco_captions"),
        "languages": ("en",),
        "protocol": "p0",
    },
    "core-ja": {
        "models": (
            "embeddinggemma2",
            "rinna_ja_clip_b16",
            "openclip_multilingual_b32",
            "siglip2_base_224",
        ),
        "datasets": ("crossmodal3600",),
        "languages": ("en", "ja"),
        "protocol": "p2",
    },
}


def _write_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "Unknown"
    size = float(value)
    for unit in ("bytes", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:,.0f} {unit}" if unit == "bytes" else f"{size:.2f} {unit}"
        size /= 1024
    raise ValueError("Could not format checkpoint size.")


def _dataset_download_estimate(dataset_key: str) -> str:
    if dataset_key in {"cifar10", "cifar100"}:
        return "The preparation check budgets 200 MiB for the archive and extraction."
    if dataset_key == "mscoco_captions":
        return "Approximately 5 GiB for the 5K test images, plus annotation files."
    if dataset_key == "crossmodal3600":
        return (
            "The official page lists 302 MiB of images, 17 MiB of captions, and a 1.3 MiB "
            "image-attribution CSV; temporary extraction space is also required."
        )
    return "No download size is available; this dataset is currently blocked."


def _run_plan(
    model_key: str,
    dataset_key: str,
    *,
    protocol: ProtocolName,
    language: str,
    precision: str,
    device: str,
    batch_size: int,
    seed: int,
    output: Path,
    confirm_data_terms: bool = False,
) -> dict[str, Any]:
    model = get_model(model_key)
    dataset = get_dataset(dataset_key)
    normalized_language = canonical_language(language)
    if dataset.access_status == "blocked" or dataset.license_status == "blocked":
        access = "blocked"
        access_reason = dataset.access_reason
    elif (
        dataset.license_status == "requires_user_confirmation"
        and not confirm_data_terms
    ):
        access = "requires_data_terms_confirmation"
        access_reason = (
            f"Review {dataset.source}; pass --confirm-data-terms only if authorized "
            "to use this dataset."
        )
    elif normalized_language not in dataset.languages:
        access = "unsupported_language"
        access_reason = (
            f"{dataset.key} does not provide {normalized_language}; "
            f"available: {', '.join(dataset.languages)}."
        )
    elif not language_is_supported(model, normalized_language):
        access = "unsupported_model_language"
        access_reason = f"{model.name} does not support {normalized_language}."
    elif dataset.task not in model.supported_tasks:
        access = "unsupported_task"
        access_reason = f"{model.name} does not support {dataset.task}."
    else:
        access = "available"
        access_reason = None
    return {
        "model": {
            "key": model.key,
            "name": model.name,
            "repository": model.hf_repository or model.model_id,
            "revision": model.checkpoint_revision or model.repository_revision,
            "pretrained_tag": model.pretrained_tag,
            "license": model.license,
            "supported_languages": list(model.languages),
            "checkpoint_filename": model.checkpoint_filename,
            "checkpoint_size_bytes": model.checkpoint_size_bytes,
            "checkpoint_size_human": _format_bytes(model.checkpoint_size_bytes),
            "download_size": (
                f"{_format_bytes(model.checkpoint_size_bytes)} for the pinned checkpoint "
                "file; configuration and tokenizer files may add more. Cached files are reused."
                if model.checkpoint_size_bytes is not None
                else "Unknown; cached files are reused, but the uncached checkpoint size "
                "has not been established."
            ),
            "tokenizer_repository": (
                model.tokenizer_repository or f"open_clip:{model.model_id}"
            ),
            "tokenizer_revision": model.tokenizer_revision,
            "tokenizer_license": model.tokenizer_license,
        },
        "dataset": {
            "key": dataset.key,
            "name": dataset.name,
            "split": dataset.split,
            "task": dataset.task,
            "expected_images": dataset.expected_images,
            "expected_texts": dataset.expected_texts,
            "expected_text_counts_by_language": dict(
                dataset.expected_text_counts_by_language
            ),
            "download_estimate": _dataset_download_estimate(dataset.key),
            "license": dataset.license,
            "license_status": dataset.license_status,
            "access": access,
            "access_reason": access_reason,
        },
        "evaluation": {
            "protocol": protocol,
            "language": normalized_language,
            "precision": precision,
            "device": device,
            "batch_size": batch_size,
            "seed": seed,
            "output": str(output),
        },
    }


def _authorize_downloads(plan: dict[str, Any], yes: bool) -> bool:
    _write_json(plan)
    if yes:
        return True
    if not sys.stdin.isatty():
        print(
            "No download or evaluation started. Review the plan and pass --yes to authorize it.",
            file=sys.stderr,
        )
        return False
    answer = input("Proceed with any required downloads and evaluation? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def _command_doctor(args: argparse.Namespace) -> int:
    report = doctor_report(args.device)
    _write_json(report)
    return 0 if report["ok"] else 2


def _command_list(args: argparse.Namespace) -> int:
    _write_json(model_registry_dict() if args.kind == "models" else dataset_registry_dict())
    return 0


def _command_prepare(args: argparse.Namespace) -> int:
    dataset = get_dataset(args.dataset)
    plan = {
        "dataset": dataset.name,
        "key": dataset.key,
        "split": dataset.split,
        "expected_images": dataset.expected_images,
        "expected_texts": dataset.expected_texts,
        "expected_text_counts_by_language": dict(dataset.expected_text_counts_by_language),
        "download_estimate": _dataset_download_estimate(dataset.key),
        "license": dataset.license,
        "license_status": dataset.license_status,
        "access_status": dataset.access_status,
        "access_reason": dataset.access_reason or None,
    }
    if dataset.access_status == "blocked" or dataset.license_status == "blocked":
        _write_json(plan)
        return 2
    if dataset.license_status == "requires_user_confirmation" and not args.confirm_data_terms:
        _write_json({"preparation": plan})
        print(
            "No dataset download started. Review the source terms and pass "
            "--confirm-data-terms only if you are authorized to use this dataset.",
            file=sys.stderr,
        )
        return 2
    if not _authorize_downloads({"preparation": plan}, args.yes):
        return 2
    record = prepare_dataset(
        dataset.key,
        progress=_progress,
        confirm_data_terms=args.confirm_data_terms,
    )
    _write_json(record)
    return 0


def _command_run(args: argparse.Namespace) -> int:
    dataset = get_dataset(args.dataset)
    if args.split is not None and args.split != dataset.split:
        raise ValueError(
            f"Dataset {dataset.key!r} uses split {dataset.split!r}, not {args.split!r}."
        )
    plan = _run_plan(
        args.model,
        args.dataset,
        protocol=args.protocol,
        language=args.language,
        precision=args.precision,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        output=args.output,
        confirm_data_terms=args.confirm_data_terms,
    )
    if args.dry_run:
        _write_json({"dry_run": True, **plan})
        return 0
    if plan["dataset"]["access"] == "available" and not _authorize_downloads(
        plan,
        args.yes,
    ):
        return 2
    result = run_benchmark(
        args.model,
        args.dataset,
        protocol=args.protocol,
        language=args.language,
        precision=args.precision,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        results_root=args.output,
        resume=args.resume,
        force=args.force,
        allow_cpu_fallback=args.allow_cpu_fallback,
        confirm_data_terms=args.confirm_data_terms,
        progress=_progress,
    )
    _write_json(result)
    return _result_exit_code(result)


def _suite_plans(args: argparse.Namespace) -> list[dict[str, Any]]:
    suite = SUITES[args.name]
    model_keys = args.models or suite["models"]
    dataset_keys = args.datasets or suite["datasets"]
    languages = args.languages or suite["languages"]
    protocol: ProtocolName = args.protocol or suite["protocol"]
    plans: list[dict[str, Any]] = []
    for model_key in model_keys:
        get_model(model_key)
    for dataset_key in dataset_keys:
        get_dataset(dataset_key)
    for model_key in model_keys:
        for dataset_key in dataset_keys:
            for language in languages:
                plans.append(
                    _run_plan(
                        model_key,
                        dataset_key,
                        protocol=protocol,
                        language=language,
                        precision=args.precision,
                        device=args.device,
                        batch_size=args.batch_size,
                        seed=args.seed,
                        output=args.output,
                        confirm_data_terms=args.confirm_data_terms,
                    )
                )
    return plans


def _command_suite(args: argparse.Namespace) -> int:
    plans = _suite_plans(args)
    if args.dry_run:
        _write_json({"dry_run": True, "suite": args.name, "plans": plans})
        return 0
    _write_json({"suite": args.name, "plans": plans})
    if not args.yes:
        if not sys.stdin.isatty():
            print(
                "No suite started. Review all plans and pass --yes to authorize downloads.",
                file=sys.stderr,
            )
            return 2
        answer = input("Proceed with the suite and any required downloads? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            return 2
    suite = SUITES[args.name]
    model_keys = args.models or suite["models"]
    dataset_keys = args.datasets or suite["datasets"]
    languages = args.languages or suite["languages"]
    protocol: ProtocolName = args.protocol or suite["protocol"]
    results = [
        run_benchmark(
            model_key,
            dataset_key,
            protocol=protocol,
            language=language,
            precision=args.precision,
            device=args.device,
            batch_size=args.batch_size,
            seed=args.seed,
            results_root=args.output,
            resume=args.resume,
            force=args.force,
            allow_cpu_fallback=args.allow_cpu_fallback,
            confirm_data_terms=args.confirm_data_terms,
            progress=_progress,
        )
        for model_key in model_keys
        for dataset_key in dataset_keys
        for language in languages
    ]
    _write_json(
        {
            "suite": args.name,
            "results": [
                {
                    "run_id": item["benchmark_metadata"]["run_id"],
                    "model": item["model"],
                    "dataset": item["dataset"],
                    "language": item["language"],
                    "status": item["status"],
                    "reason": item["benchmark_metadata"]["reason_if_not_success"],
                }
                for item in results
            ],
        }
    )
    codes = [_result_exit_code(result) for result in results]
    return 1 if 1 in codes else 2 if any(code == 2 for code in codes) else 0


def _result_exit_code(result: dict[str, Any]) -> int:
    status = result["status"]
    if status == "completed":
        return 0
    return 1 if status == "failed" else 2


def _command_report(args: argparse.Namespace) -> int:
    files = generate_reports(args.input, args.output)
    _write_json({name: str(path) for name, path in files.items()})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vlbench",
        description="Reproducible vision-language model benchmark tooling.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="Inspect Python, CUDA, GPU, storage, and auth.")
    doctor.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    doctor.set_defaults(handler=_command_doctor)

    listing = commands.add_parser("list", help="List registered models or datasets.")
    listing.add_argument("kind", choices=("models", "datasets"))
    listing.set_defaults(handler=_command_list)

    datasets = commands.add_parser("datasets", help="Prepare benchmark datasets.")
    dataset_commands = datasets.add_subparsers(dest="dataset_command", required=True)
    prepare = dataset_commands.add_parser("prepare", help="Download and verify a dataset.")
    prepare.add_argument("--dataset", choices=tuple(sorted(DATASETS)), required=True)
    prepare.add_argument(
        "--yes",
        action="store_true",
        help="Authorize any required network downloads after reviewing the plan.",
    )
    prepare.add_argument(
        "--confirm-data-terms",
        action="store_true",
        help="Confirm you reviewed and are authorized to use a dataset with unverified terms.",
    )
    prepare.set_defaults(handler=_command_prepare)

    run = commands.add_parser("run", help="Run one model/dataset evaluation.")
    run.add_argument("--model", choices=tuple(sorted(MODELS)), required=True)
    run.add_argument("--dataset", choices=tuple(sorted(DATASETS)), required=True)
    run.add_argument("--split", help="Assert the registry's pinned dataset split.")
    run.add_argument("--language", default="en")
    run.add_argument("--protocol", choices=("p0", "p1", "p2"), default="p0")
    run.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    run.add_argument("--device", default="cuda")
    run.add_argument("--batch-size", type=int, default=16)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--output", type=Path, default=RESULTS_ROOT)
    run_reuse = run.add_mutually_exclusive_group()
    run_reuse.add_argument("--resume", action="store_true")
    run_reuse.add_argument("--force", action="store_true")
    run.add_argument("--allow-cpu-fallback", action="store_true")
    run.add_argument(
        "--confirm-data-terms",
        action="store_true",
        help="Confirm you reviewed and are authorized to use a dataset with unverified terms.",
    )
    run.add_argument("--dry-run", action="store_true")
    run.add_argument(
        "--yes",
        action="store_true",
        help="Authorize any required network downloads after reviewing the plan.",
    )
    run.set_defaults(handler=_command_run)

    suite = commands.add_parser("suite", help="Run a predefined benchmark suite.")
    suite.add_argument("--name", choices=tuple(sorted(SUITES)), required=True)
    suite.add_argument("--models", nargs="+", choices=tuple(sorted(MODELS)))
    suite.add_argument("--datasets", nargs="+", choices=tuple(sorted(DATASETS)))
    suite.add_argument("--languages", nargs="+")
    suite.add_argument("--protocol", choices=("p0", "p1", "p2"))
    suite.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    suite.add_argument("--device", default="cuda")
    suite.add_argument("--batch-size", type=int, default=16)
    suite.add_argument("--seed", type=int, default=0)
    suite.add_argument("--output", type=Path, default=RESULTS_ROOT)
    suite_reuse = suite.add_mutually_exclusive_group()
    suite_reuse.add_argument("--resume", action="store_true")
    suite_reuse.add_argument("--force", action="store_true")
    suite.add_argument("--allow-cpu-fallback", action="store_true")
    suite.add_argument(
        "--confirm-data-terms",
        action="store_true",
        help="Confirm review and authorization for datasets with unverified terms.",
    )
    suite.add_argument("--dry-run", action="store_true")
    suite.add_argument(
        "--yes",
        action="store_true",
        help="Authorize all displayed suite downloads and evaluations.",
    )
    suite.set_defaults(handler=_command_suite)

    report = commands.add_parser("report", help="Regenerate summaries and plots from raw JSON.")
    report.add_argument("--input", type=Path, default=RESULTS_ROOT / "raw")
    report.add_argument("--output", type=Path, default=RESULTS_ROOT)
    report.set_defaults(handler=_command_report)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (BenchmarkError, OSError, ValueError) as error:
        print(f"vlbench: error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
