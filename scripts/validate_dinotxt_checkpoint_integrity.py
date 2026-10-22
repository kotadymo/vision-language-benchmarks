from __future__ import annotations

import argparse
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from vision_language_benchmarks.adapters.base import package_version
from vision_language_benchmarks.atomic import atomic_write_json
from vision_language_benchmarks.dinotxt_provenance import (
    DINOTXT_CHECKPOINT_REVISION,
    DINOTXT_MODEL_ID,
    DINOTXT_OPENVLCLIP_MODEL_NAME,
    DINOTXT_OPENVLCLIP_VERSION,
    DINOV2_SOURCE_REVISION,
    checkpoint_fingerprint,
    verify_cached_components,
)
from vision_language_benchmarks.paths import PROJECT_ROOT


def _git_source_state() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    if commit.returncode != 0:
        raise RuntimeError(f"Cannot determine source commit: {commit.stderr.strip()}.")
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    if status.returncode != 0:
        raise RuntimeError(f"Cannot inspect worktree status: {status.stderr.strip()}.")
    return commit.stdout.strip(), bool(status.stdout.strip())


def _gpu_name() -> str | None:
    if not torch.cuda.is_available():
        return None
    return str(torch.cuda.get_device_name(0))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify cached dino.txt artifact hashes without loading model weights "
            "or running inference."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "dinov2-dinotxt-v0.6.0"
            / "validation"
            / "checkpoint-integrity.json"
        ),
    )
    args = parser.parse_args()

    openvlclip_version = package_version("openvlclip")
    if openvlclip_version != DINOTXT_OPENVLCLIP_VERSION:
        raise RuntimeError(
            f"Expected openvlclip=={DINOTXT_OPENVLCLIP_VERSION}, "
            f"found {openvlclip_version}."
        )
    checkpoint_directory = Path(torch.hub.get_dir()) / "checkpoints"
    components = verify_cached_components(checkpoint_directory)
    fingerprint = checkpoint_fingerprint(components)
    source_sha, dirty = _git_source_state()
    runtime: dict[str, Any] = {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": _gpu_name(),
    }
    report = {
        "schema_version": "1.0.0",
        "validation_type": "dinotxt_checkpoint_integrity",
        "status": "completed",
        "model_inference_performed": False,
        "source_commit_sha": source_sha,
        "working_tree_dirty": dirty,
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds").replace(
            "+00:00",
            "Z",
        ),
        "library": "openvlclip",
        "library_version": openvlclip_version,
        "library_version_expected": DINOTXT_OPENVLCLIP_VERSION,
        "model_name": DINOTXT_OPENVLCLIP_MODEL_NAME,
        "model_id": DINOTXT_MODEL_ID,
        "model_revision": DINOTXT_CHECKPOINT_REVISION,
        "model_source_revision": DINOV2_SOURCE_REVISION,
        "cache_location": "torch.hub/checkpoints",
        "checkpoint_sha256": fingerprint["checkpoint_sha256"],
        "runtime": runtime,
        "components": components,
    }
    atomic_write_json(args.output, report)
    print(f"Verified {len(components)} dino.txt cached artifacts; no inference was run.")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
