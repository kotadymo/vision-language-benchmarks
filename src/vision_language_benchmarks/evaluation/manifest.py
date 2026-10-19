from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vision_language_benchmarks.atomic import atomic_write_json
from vision_language_benchmarks.errors import DataIntegrityError
from vision_language_benchmarks.registry import RunStatus

TERMINAL_STATUSES = {"completed", "skipped", "failed", "blocked"}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class RunManifest:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"schema_version": "1.0.0", "runs": {}}
        manifest = json.loads(self.path.read_text(encoding="utf-8"))
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != "1.0.0"
            or not isinstance(manifest.get("runs"), dict)
        ):
            raise DataIntegrityError(f"Run manifest has an unsupported structure: {self.path}.")
        return manifest

    def get(self, run_id: str) -> dict[str, Any] | None:
        entry = self._read()["runs"].get(run_id)
        return entry if isinstance(entry, dict) else None

    def transition(
        self,
        run_id: str,
        status: RunStatus,
        fields: dict[str, Any],
        *,
        restart: bool = False,
    ) -> dict[str, Any]:
        manifest = self._read()
        previous = manifest["runs"].get(run_id)
        previous_status = previous.get("status") if isinstance(previous, dict) else None
        if previous_status is None:
            if status not in {"running", "skipped", "failed", "blocked"}:
                raise ValueError(f"Cannot start a run in status {status!r}.")
        elif previous_status == "running":
            if status not in TERMINAL_STATUSES:
                raise ValueError(f"Invalid run transition {previous_status!r} -> {status!r}.")
        elif previous_status in TERMINAL_STATUSES:
            if status != "running" or not restart:
                raise ValueError(
                    f"Run {run_id} is already terminal ({previous_status}); "
                    "restart must be explicit."
                )
        else:
            raise DataIntegrityError(
                f"Run {run_id} has an unknown manifest status {previous_status!r}."
            )

        now = utc_now()
        entry = dict(previous) if isinstance(previous, dict) else {"run_id": run_id}
        entry.update(fields)
        entry["status"] = status
        entry["updated_at"] = now
        if status == "running":
            entry["started_at"] = now
            entry.pop("finished_at", None)
        elif status in TERMINAL_STATUSES:
            entry["finished_at"] = now
        manifest["runs"][run_id] = entry
        atomic_write_json(self.path, manifest)
        return entry
