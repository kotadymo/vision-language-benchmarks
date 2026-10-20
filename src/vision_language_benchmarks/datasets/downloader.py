from __future__ import annotations

import os
import shutil
import stat
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

from vision_language_benchmarks.atomic import md5_file, sha256_file
from vision_language_benchmarks.errors import DataIntegrityError


def download_file(
    url: str,
    destination: Path,
    *,
    sha256: str | None = None,
    md5: str | None = None,
    retries: int = 3,
    timeout: int = 90,
) -> Path:
    if retries < 1:
        raise ValueError("retries must be at least 1.")
    if sha256 is not None and md5 is not None:
        raise ValueError("Specify only one expected checksum.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")

    if destination.exists():
        if sha256 is None and md5 is None:
            if destination.stat().st_size == 0:
                raise DataIntegrityError(f"Existing download is empty: {destination}")
            return destination
        actual = sha256_file(destination) if sha256 else md5_file(destination)
        expected = sha256 or md5
        if actual.lower() == expected.lower():
            return destination
        raise DataIntegrityError(
            f"Existing file checksum mismatch for {destination}: "
            f"expected {expected}, got {actual}."
        )

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "vision-language-benchmarks/0.1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                append = offset > 0 and response.status == 206
                if offset > 0 and not append:
                    offset = 0
                mode = "ab" if append else "wb"
                with partial.open(mode) as stream:
                    shutil.copyfileobj(response, stream, length=1024 * 1024)
                    stream.flush()
                    os.fsync(stream.fileno())
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in {408, 425, 429, 500, 502, 503, 504} or attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            last_error = error
            if attempt == retries:
                raise

        if last_error is not None and attempt < retries:
            time.sleep(min(2 ** (attempt - 1), 8))
            last_error = None
            continue

        if not partial.exists() or partial.stat().st_size == 0:
            if attempt == retries:
                raise DataIntegrityError(f"Download produced no data for {url}.")
            time.sleep(min(2 ** (attempt - 1), 8))
            continue

        actual = sha256_file(partial) if sha256 else md5_file(partial) if md5 else None
        expected = sha256 or md5
        if expected is not None and actual is not None and actual.lower() != expected.lower():
            partial.unlink(missing_ok=True)
            if attempt == retries:
                raise DataIntegrityError(
                    f"Checksum mismatch downloading {url}: expected {expected}, got {actual}."
                )
            time.sleep(min(2 ** (attempt - 1), 8))
            continue

        os.replace(partial, destination)
        return destination

    raise RuntimeError(f"Download failed after {retries} attempts: {url}") from last_error


def _safe_member_path(root: Path, name: str) -> Path:
    if not name or "\\" in name:
        raise DataIntegrityError(f"Unsafe archive member path: {name!r}.")
    posix_path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or any(part in {"..", ""} for part in posix_path.parts)
    ):
        raise DataIntegrityError(f"Unsafe archive member path: {name!r}.")
    target = root.joinpath(*posix_path.parts)
    try:
        target.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise DataIntegrityError(f"Archive member escapes extraction root: {name!r}.") from error
    return target


def safe_extract_tar(archive: Path, destination: Path) -> Path:
    if destination.exists():
        raise FileExistsError(f"Extraction destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.extract-",
        dir=destination.parent,
    ) as temporary_name:
        staging = Path(temporary_name)
        seen: set[str] = set()
        with tarfile.open(archive, mode="r:*") as tar:
            for member in tar:
                target = _safe_member_path(staging, member.name)
                normalized = str(PurePosixPath(member.name)).casefold()
                if normalized in seen:
                    raise DataIntegrityError(f"Duplicate archive member: {member.name!r}.")
                seen.add(normalized)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = tar.extractfile(member)
                    if source is None:
                        raise DataIntegrityError(f"Cannot read archive member {member.name!r}.")
                    with source, target.open("xb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                else:
                    raise DataIntegrityError(
                        f"Unsupported non-regular archive member: {member.name!r}."
                    )
        os.replace(staging, destination)
    return destination


def safe_extract_zip(
    archive: Path,
    destination: Path,
    *,
    members: set[str] | None = None,
) -> Path:
    if destination.exists():
        raise FileExistsError(f"Extraction destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.extract-",
        dir=destination.parent,
    ) as temporary_name:
        staging = Path(temporary_name)
        seen: set[str] = set()
        with zipfile.ZipFile(archive) as zipped:
            for info in zipped.infolist():
                if members is not None and info.filename not in members:
                    continue
                target = _safe_member_path(staging, info.filename)
                normalized = str(PurePosixPath(info.filename)).casefold()
                if normalized in seen:
                    raise DataIntegrityError(f"Duplicate archive member: {info.filename!r}.")
                seen.add(normalized)
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise DataIntegrityError(
                        f"Symbolic links are not allowed in archives: {info.filename!r}."
                    )
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zipped.open(info) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
        os.replace(staging, destination)
    return destination
