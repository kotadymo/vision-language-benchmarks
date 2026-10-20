from __future__ import annotations

import subprocess
from pathlib import Path

from vision_language_benchmarks.reproducibility import (
    inspect_git_repository,
    inspect_staging_directory,
)


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        check=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _repository(tmp_path: Path, *, committed: bool = True) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.email", "benchmark-test@example.invalid")
    _git(repository, "config", "user.name", "Benchmark Test")
    (repository / ".gitignore").write_text(".cache/\n", encoding="utf-8")
    (repository / "src").mkdir()
    (repository / "src" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    if committed:
        _git(repository, "add", "--all")
        _git(repository, "commit", "-m", "Initial source")
    return repository


def test_preflight_accepts_clean_repository_and_ignored_staging_output(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    source_commit_sha = _git(repository, "rev-parse", "HEAD")
    staging = repository / ".cache" / "benchmark-staging" / "suite"
    staging.mkdir(parents=True)
    (staging / "temporary.json").write_text("{}", encoding="utf-8")

    git_report = inspect_git_repository(repository, source_commit_sha)
    staging_report = inspect_staging_directory(repository, staging)

    assert git_report["ok"] is True
    assert git_report["source_commit_sha"] == source_commit_sha
    assert git_report["staged_changes"] == []
    assert git_report["unstaged_changes"] == []
    assert git_report["untracked_files"] == []
    assert staging_report["ok"] is True
    assert staging_report["ignored"] is True
    assert staging_report["tracked_paths"] == []


def test_preflight_rejects_modified_tracked_source(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    source_file = repository / "src" / "module.py"
    source_file.write_text("VALUE = 2\n", encoding="utf-8")

    report = inspect_git_repository(repository)

    assert report["ok"] is False
    assert report["unstaged_changes"]
    assert "Tracked worktree changes are present." in report["errors"]


def test_preflight_rejects_staged_changes(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    source_file = repository / "src" / "module.py"
    source_file.write_text("VALUE = 2\n", encoding="utf-8")
    _git(repository, "add", "src/module.py")

    report = inspect_git_repository(repository)

    assert report["ok"] is False
    assert report["staged_changes"]
    assert "Staged changes are present." in report["errors"]


def test_preflight_rejects_untracked_source_files(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    (repository / "src" / "untracked.py").write_text("VALUE = 3\n", encoding="utf-8")

    report = inspect_git_repository(repository)

    assert report["ok"] is False
    assert report["untracked_files"] == ["src/untracked.py"]
    assert "Untracked files are present." in report["errors"]


def test_preflight_rejects_missing_git_commit(tmp_path: Path) -> None:
    repository = _repository(tmp_path, committed=False)

    report = inspect_git_repository(repository)

    assert report["ok"] is False
    assert report["source_commit_sha"] is None
    assert any(error.startswith("Missing Git commit:") for error in report["errors"])


def test_preflight_rejects_incorrect_source_revision(tmp_path: Path) -> None:
    repository = _repository(tmp_path)

    report = inspect_git_repository(repository, "0" * 40)

    assert report["ok"] is False
    assert any("does not match the expected source revision" in error for error in report["errors"])


def test_staging_preflight_rejects_nonignored_or_tracked_output(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    unignored_report = inspect_staging_directory(repository, repository / "temporary-output")

    assert unignored_report["ok"] is False
    assert any(
        "must be under .cache/benchmark-staging" in error
        for error in unignored_report["errors"]
    )

    tracked_staging = repository / ".cache" / "benchmark-staging" / "tracked"
    tracked_staging.mkdir(parents=True)
    (tracked_staging / "result.json").write_text("{}", encoding="utf-8")
    _git(repository, "add", "-f", ".cache/benchmark-staging/tracked/result.json")
    tracked_report = inspect_staging_directory(repository, tracked_staging)

    assert tracked_report["ok"] is False
    assert tracked_report["tracked_paths"] == [
        ".cache/benchmark-staging/tracked/result.json"
    ]
