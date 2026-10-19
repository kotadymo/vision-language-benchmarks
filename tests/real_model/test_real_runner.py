from __future__ import annotations

import importlib
import math
import os
from pathlib import Path

import pytest

if os.environ.get("VLBENCH_RUN_REAL_BENCHMARK_TESTS") != "1":
    pytest.skip(
        "Set VLBENCH_RUN_REAL_BENCHMARK_TESTS=1 to run the full XM3600 CUDA benchmark.",
        allow_module_level=True,
    )


@pytest.mark.real_benchmark
def test_runner_completes_full_xm3600_japanese_retrieval(tmp_path: Path) -> None:
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        pytest.skip("A CUDA-enabled PyTorch runtime and NVIDIA GPU are required.")

    from vision_language_benchmarks.evaluation.runner import run_benchmark

    results_root = tmp_path / "results"
    result = run_benchmark(
        "openclip_multilingual_b32",
        "crossmodal3600",
        protocol="p2",
        language="ja",
        precision="fp32",
        device="cuda:0",
        batch_size=16,
        results_root=results_root,
        force=True,
    )

    assert result["status"] == "completed", result["benchmark_metadata"].get(
        "reason_if_not_success"
    )
    metadata = result["benchmark_metadata"]
    assert metadata["image_count"] == 3_600
    assert metadata["text_count"] == 7_185
    assert metadata["positive_pair_count"] == 7_185
    assert metadata["checkpoint_revision"] == "506d40eb551f4801a1c27fc20a31c7b8f590deda"
    metrics = result.get("metrics")
    assert isinstance(metrics, dict)
    expected_metrics = {
        f"{metric_prefix}_retrieval_recall@{cutoff}"
        for metric_prefix in ("image", "text")
        for cutoff in (1, 5, 10)
    }
    assert set(metrics) == expected_metrics
    assert all(
        isinstance(value, (int, float))
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
        for value in metrics.values()
    )
    raw_path = results_root / "raw" / f"{metadata['run_id']}.json"
    assert raw_path.is_file()
    assert (results_root / "run-manifest.json").is_file()
