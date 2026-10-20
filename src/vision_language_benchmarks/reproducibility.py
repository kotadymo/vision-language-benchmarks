from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vision_language_benchmarks.errors import DataIntegrityError

_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_STAGING_PREFIX = Path(".cache") / "benchmark-staging"


@dataclass(frozen=True)
class SourcePreflight:
    source_commit_sha: str
    staging_directory: str


def _run_git(project_root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )


def _parse_worktree_status(status: str) -> dict[str, list[str]]:
    changes: dict[str, list[str]] = {
        "staged": [],
        "unstaged": [],
        "untracked": [],
    }
    records = status.split("\0")
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if len(record) < 3:
            continue
        state = record[:2]
        path = record[3:]
        if state == "??":
            changes["untracked"].append(path)
            continue
        if state[0] != " ":
            changes["staged"].append(f"{state} {path}")
        if state[1] != " ":
            changes["unstaged"].append(f"{state} {path}")
        if "R" in state or "C" in state:
            index += 1
    return changes


def inspect_git_repository(
    project_root: Path,
    expected_source_commit_sha: str | None = None,
) -> dict[str, Any]:
    root = project_root.resolve()
    errors: list[str] = []
    repository_root: str | None = None
    source_commit_sha: str | None = None
    worktree_status_available = False
    changes = {"staged": [], "unstaged": [], "untracked": []}

    try:
        root_result = _run_git(root, "rev-parse", "--show-toplevel")
    except (OSError, subprocess.SubprocessError) as error:
        root_result = None
        errors.append(f"Could not inspect the Git repository: {error}")
    if root_result is not None:
        if root_result.returncode != 0:
            detail = root_result.stderr.strip() or "the directory is not a Git checkout"
            errors.append(f"Invalid Git checkout: {detail}")
        else:
            repository_root = str(Path(root_result.stdout.strip()).resolve())
            if Path(repository_root) != root:
                errors.append(
                    f"Expected project root {root}, but Git reports {repository_root}."
                )

    try:
        head_result = _run_git(root, "rev-parse", "--verify", "HEAD^{commit}")
    except (OSError, subprocess.SubprocessError) as error:
        head_result = None
        errors.append(f"Could not resolve HEAD to a commit: {error}")
    if head_result is not None:
        if head_result.returncode != 0:
            detail = head_result.stderr.strip() or "HEAD does not identify a commit"
            errors.append(f"Missing Git commit: {detail}")
        else:
            source_commit_sha = head_result.stdout.strip()

    try:
        status_result = _run_git(
            root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "-z",
        )
    except (OSError, subprocess.SubprocessError) as error:
        status_result = None
        errors.append(f"Could not inspect Git worktree status: {error}")
    if status_result is not None:
        if status_result.returncode != 0:
            detail = status_result.stderr.strip() or "git status failed"
            errors.append(f"Could not inspect Git worktree status: {detail}")
        else:
            worktree_status_available = True
            changes = _parse_worktree_status(status_result.stdout)
            if changes["staged"]:
                errors.append("Staged changes are present.")
            if changes["unstaged"]:
                errors.append("Tracked worktree changes are present.")
            if changes["untracked"]:
                errors.append("Untracked files are present.")

    if expected_source_commit_sha is not None:
        if not _COMMIT_PATTERN.fullmatch(expected_source_commit_sha):
            errors.append(
                "Expected source revision must be a full 40-character lowercase Git SHA."
            )
        elif (
            source_commit_sha is not None
            and source_commit_sha != expected_source_commit_sha
        ):
            errors.append(
                "HEAD does not match the expected source revision "
                f"{expected_source_commit_sha}."
            )

    return {
        "ok": not errors,
        "repository_root": repository_root,
        "source_commit_sha": source_commit_sha,
        "worktree_status_available": worktree_status_available,
        "expected_source_commit_sha": expected_source_commit_sha,
        "staged_changes": changes["staged"],
        "unstaged_changes": changes["unstaged"],
        "untracked_files": changes["untracked"],
        "errors": errors,
    }


def inspect_staging_directory(project_root: Path, output_dir: Path) -> dict[str, Any]:
    root = project_root.resolve()
    target = output_dir.expanduser().resolve()
    errors: list[str] = []
    if target == root or root.is_relative_to(target):
        errors.append("Benchmark output must not be the project root or one of its parents.")
        return {
            "ok": False,
            "location": str(target),
            "ignored": False,
            "external": False,
            "tracked_paths": [],
            "errors": errors,
        }

    if not target.is_relative_to(root):
        return {
            "ok": True,
            "location": "external",
            "ignored": None,
            "external": True,
            "tracked_paths": [],
            "errors": [],
        }

    relative = target.relative_to(root)
    location = relative.as_posix()
    if relative.parts[:2] != _STAGING_PREFIX.parts:
        errors.append(
            "In-repository benchmark output must be under "
            f"{_STAGING_PREFIX.as_posix()}."
        )

    try:
        ignore_result = _run_git(
            root,
            "check-ignore",
            "--quiet",
            "--no-index",
            "--",
            location,
        )
    except (OSError, subprocess.SubprocessError) as error:
        ignore_result = None
        errors.append(f"Could not verify the staging ignore rule: {error}")
    if ignore_result is not None and ignore_result.returncode != 0:
        detail = ignore_result.stderr.strip()
        errors.append(
            "Benchmark output is not ignored by Git."
            + (f" {detail}" if detail else "")
        )

    try:
        tracked_result = _run_git(root, "ls-files", "--", location)
    except (OSError, subprocess.SubprocessError) as error:
        tracked_result = None
        errors.append(f"Could not verify that staging output is untracked: {error}")
    tracked_paths: list[str] = []
    if tracked_result is not None:
        if tracked_result.returncode != 0:
            detail = tracked_result.stderr.strip()
            errors.append(
                "Could not verify that staging output is untracked."
                + (f" {detail}" if detail else "")
            )
        else:
            tracked_paths = [
                path for path in tracked_result.stdout.splitlines() if path.strip()
            ]
            if tracked_paths:
                errors.append(
                    "Benchmark output contains tracked paths: "
                    + ", ".join(tracked_paths)
                )

    return {
        "ok": not errors,
        "location": location,
        "ignored": ignore_result is not None and ignore_result.returncode == 0,
        "external": False,
        "tracked_paths": tracked_paths,
        "errors": errors,
    }


def _resolve_uv_executable(
    project_root: Path,
    uv_executable: Path | None,
) -> str | None:
    if uv_executable is not None:
        candidate = uv_executable.expanduser()
        if not candidate.is_absolute():
            candidate = project_root / candidate
        if candidate.is_file():
            return str(candidate.resolve())
        return shutil.which(str(uv_executable))

    found = shutil.which("uv")
    if found is not None:
        return found
    for candidate in (
        Path.home() / ".local" / "bin" / "uv.exe",
        Path.home() / ".local" / "bin" / "uv",
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def inspect_lockfile(
    project_root: Path,
    uv_executable: Path | None = None,
) -> dict[str, Any]:
    root = project_root.resolve()
    lock_path = root / "uv.lock"
    errors: list[str] = []
    if not lock_path.is_file():
        errors.append(f"Lockfile is missing: {lock_path.name}.")
        return {"ok": False, "lockfile": lock_path.name, "errors": errors}

    executable = _resolve_uv_executable(root, uv_executable)
    if executable is None:
        errors.append(
            "uv was not found. Install uv or pass --uv-executable with its path."
        )
        return {"ok": False, "lockfile": lock_path.name, "errors": errors}

    try:
        completed = subprocess.run(
            [executable, "lock", "--check"],
            cwd=root,
            capture_output=True,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as error:
        errors.append(f"Could not verify uv.lock: {error}")
        return {
            "ok": False,
            "lockfile": lock_path.name,
            "uv_executable": Path(executable).name,
            "errors": errors,
        }

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        errors.append(
            "uv.lock is inconsistent."
            + (f" {detail}" if detail else f" uv lock --check exited {completed.returncode}.")
        )
    return {
        "ok": not errors,
        "lockfile": lock_path.name,
        "uv_executable": Path(executable).name,
        "errors": errors,
    }


def reproducibility_report(
    project_root: Path,
    output_dir: Path,
    *,
    expected_source_commit_sha: str | None = None,
    uv_executable: Path | None = None,
) -> dict[str, Any]:
    git = inspect_git_repository(project_root, expected_source_commit_sha)
    staging = inspect_staging_directory(project_root, output_dir)
    lockfile = inspect_lockfile(project_root, uv_executable)
    errors = [
        error
        for check in (git, staging, lockfile)
        for error in check["errors"]
    ]
    return {
        "ok": not errors,
        "git": git,
        "staging": staging,
        "lockfile": lockfile,
        "errors": errors,
    }


def require_clean_source_preflight(
    project_root: Path,
    output_dir: Path,
    expected_source_commit_sha: str | None,
    *,
    uv_executable: Path | None = None,
    check_lockfile: bool = True,
) -> SourcePreflight:
    if expected_source_commit_sha is None:
        raise DataIntegrityError(
            "Strict reproducibility requires --expected-source-commit with the full "
            "40-character frozen Git SHA."
        )

    git = inspect_git_repository(project_root, expected_source_commit_sha)
    staging = inspect_staging_directory(project_root, output_dir)
    lockfile = (
        inspect_lockfile(project_root, uv_executable)
        if check_lockfile
        else {"ok": True, "errors": []}
    )
    errors = [
        error
        for check in (git, staging, lockfile)
        for error in check["errors"]
    ]
    if errors:
        raise DataIntegrityError(
            "Strict reproducibility preflight failed: " + " ".join(errors)
        )

    source_commit_sha = git["source_commit_sha"]
    if not isinstance(source_commit_sha, str):
        raise DataIntegrityError(
            "Strict reproducibility preflight could not resolve a committed source revision."
        )
    return SourcePreflight(
        source_commit_sha=source_commit_sha,
        staging_directory=str(staging["location"]),
    )
