from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vision_language_benchmarks.errors import DataIntegrityError
from vision_language_benchmarks.reporting.retrieval_variant import retrieval_variant_info

_ATTEMPT_STATES = {
    "failed": "failed",
    "blocked": "blocked",
    "skipped": "skipped",
    "running": "incomplete",
}


def _is_frozen_strict_result(
    result: dict[str, Any],
    frozen_source_commit_sha: str,
) -> bool:
    metadata = result["benchmark_metadata"]
    configuration = metadata["run_configuration"]
    return (
        result["status"] == "completed"
        and metadata.get("reproducibility_mode") == "strict"
        and metadata.get("working_tree_dirty") is False
        and metadata.get("source_commit_sha") == frozen_source_commit_sha
        and metadata.get("expected_source_commit_sha") == frozen_source_commit_sha
        and configuration.get("reproducibility_mode") == "strict"
        and configuration.get("expected_source_commit_sha") == frozen_source_commit_sha
    )


def _attempt_state(status: Any) -> str:
    try:
        return _ATTEMPT_STATES[status]
    except (KeyError, TypeError) as error:
        raise DataIntegrityError(
            f"Run manifest contains an unsupported attempt status: {status!r}."
        ) from error


def _read_attempt_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DataIntegrityError(f"Cannot read run manifest {path}: {error}") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "1.0.0"
        or not isinstance(manifest.get("runs"), dict)
    ):
        raise DataIntegrityError(f"Run manifest has an unsupported structure: {path}.")
    return manifest


def build_result_lifecycle(
    results: list[dict[str, Any]],
    frozen_source_commit_sha: str,
    attempt_manifest_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    if (
        len(frozen_source_commit_sha) != 40
        or any(character not in "0123456789abcdef" for character in frozen_source_commit_sha)
    ):
        raise ValueError("Frozen source commit must be a full 40-character Git SHA.")

    result_ids = [
        str(result["benchmark_metadata"]["run_id"])
        for result in results
    ]
    if len(result_ids) != len(set(result_ids)):
        raise DataIntegrityError("Raw result files contain duplicate run IDs.")

    active_official_keys = {
        (
            result["benchmark_metadata"]["run_configuration"]["model_key"],
            result["language"],
        )
        for result in results
        if retrieval_variant_info(result)[0] == "official_first_caption"
        and _is_frozen_strict_result(result, frozen_source_commit_sha)
    }

    entries: list[dict[str, Any]] = []
    for result in results:
        metadata = result["benchmark_metadata"]
        configuration = metadata["run_configuration"]
        variant, _ = retrieval_variant_info(result)
        status = result["status"]
        if status != "completed":
            state = status if status in {"failed", "blocked", "skipped"} else "invalid"
        elif metadata.get("reproducibility_mode") == "strict":
            state = (
                "active"
                if _is_frozen_strict_result(result, frozen_source_commit_sha)
                else "invalid"
            )
        elif variant == "official_first_caption" and (
            configuration["model_key"],
            result["language"],
        ) in active_official_keys:
            state = "superseded"
        else:
            state = "historical"

        entries.append(
            {
                "run_id": metadata["run_id"],
                "state": state,
                "leaderboard_eligible": state in {"active", "historical"},
                "status": status,
                "model_key": configuration["model_key"],
                "dataset_key": configuration["dataset_key"],
                "language": result["language"],
                "protocol": configuration["protocol"],
                "retrieval_variant": variant,
                "source_commit_sha": metadata.get("source_commit_sha"),
                "working_tree_dirty": metadata.get("working_tree_dirty"),
            }
        )

    attempts: list[dict[str, Any]] = []
    raw_ids = set(result_ids)
    attempt_ids: set[str] = set()
    for path in attempt_manifest_paths:
        manifest = _read_attempt_manifest(path)
        for run_id, entry in manifest["runs"].items():
            if not isinstance(entry, dict):
                raise DataIntegrityError(
                    f"Run manifest entry {run_id!r} must be an object."
                )
            if run_id in raw_ids or entry.get("status") == "completed":
                continue
            if not isinstance(run_id, str) or not run_id:
                raise DataIntegrityError("Run manifest entries need non-empty run IDs.")
            if run_id in attempt_ids:
                raise DataIntegrityError(f"Duplicate run attempt ID: {run_id}.")
            attempt_ids.add(run_id)
            attempts.append(
                {
                    "run_id": run_id,
                    "state": _attempt_state(entry.get("status")),
                    "status": entry["status"],
                    "model_key": entry.get("model_key"),
                    "dataset_key": entry.get("dataset_key"),
                    "language": entry.get("language"),
                    "reason": entry.get("reason_if_not_success"),
                }
            )

    state_counts = Counter(entry["state"] for entry in entries)
    state_counts.update(attempt["state"] for attempt in attempts)
    return {
        "schema_version": "1.0.0",
        "frozen_source_commit_sha": frozen_source_commit_sha,
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "entries": sorted(entries, key=lambda entry: entry["run_id"]),
        "attempts": sorted(attempts, key=lambda attempt: attempt["run_id"]),
        "state_counts": dict(sorted(state_counts.items())),
    }
