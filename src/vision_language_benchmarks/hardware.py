from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from vision_language_benchmarks.paths import CACHE_ROOT, DATA_ROOT, PROJECT_ROOT


def _nvidia_smi() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "reason": "nvidia-smi was not found on PATH."}
    completed = subprocess.run(
        [
            executable,
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    if completed.returncode != 0:
        return {
            "available": False,
            "reason": completed.stderr.strip() or f"nvidia-smi exited {completed.returncode}.",
        }
    devices = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", maxsplit=2)]
        if len(fields) == 3:
            devices.append(
                {
                    "name": fields[0],
                    "memory_total": fields[1],
                    "driver_version": fields[2],
                }
            )
    return {"available": bool(devices), "devices": devices}


def _torch_info() -> dict[str, Any]:
    try:
        torch = importlib.import_module("torch")
    except ImportError as error:
        return {"installed": False, "reason": str(error)}

    cuda_available = bool(torch.cuda.is_available())
    info: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "bf16_supported": False,
        "devices": [],
    }
    if cuda_available:
        info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            info["devices"].append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    return info


def _free_bytes(path: Path) -> int:
    existing = path
    while not existing.exists():
        if existing.parent == existing:
            raise FileNotFoundError(f"No existing parent directory for {path}.")
        existing = existing.parent
    return shutil.disk_usage(existing).free


def doctor_report(device: str = "cuda") -> dict[str, Any]:
    if device not in {"cuda", "cpu"}:
        raise ValueError("device must be 'cuda' or 'cpu'.")
    torch = _torch_info()
    smi = _nvidia_smi()
    token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    try:
        hub = importlib.import_module("huggingface_hub")
    except ImportError:
        hub_available = False
    else:
        hub_available = True
        token_present = token_present or bool(hub.get_token())

    cuda_ok = device != "cuda" or (
        torch["installed"] and bool(torch.get("cuda_available"))
    )
    report = {
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "requested_device": device,
        "torch": torch,
        "nvidia_smi": smi,
        "huggingface_hub_installed": hub_available,
        "huggingface_auth_present": token_present,
        "storage": {
            "project_free_bytes": _free_bytes(PROJECT_ROOT),
            "data_free_bytes": _free_bytes(DATA_ROOT),
            "cache_free_bytes": _free_bytes(CACHE_ROOT),
        },
        "ok": bool(cuda_ok),
    }
    if not cuda_ok:
        if not torch["installed"]:
            report["error"] = (
                "PyTorch is not installed. Install the cuda extra and the selected "
                "model extra before requesting CUDA inference."
            )
        else:
            report["error"] = (
                "CUDA was requested but torch.cuda.is_available() is false; no CPU "
                "fallback was performed."
            )
    return report
