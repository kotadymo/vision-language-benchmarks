# Vision-Language Benchmarks

A reproducible evaluation framework for zero-shot image classification and image-text
retrieval with CLIP, OpenCLIP, SigLIP 2, EmbeddingGemma 2, and Japanese or multilingual
models.

## Status

The v0.1.0 framework, result schema, report generator, and offline tests are implemented.
All seven registered pretrained-model adapters have passed opt-in CUDA image/text smoke tests
on an NVIDIA RTX 4070 Ti SUPER. Smoke tests are not benchmark results; full-dataset CIFAR and
XM3600 evaluations and numerical parity with CLIP Benchmark remain outstanding. See the
[implementation status](docs/implementation-status.md) for verified milestones and blockers.

Dataset access is deliberately conservative: unverified or restricted sources are blocked or
require explicit user review. No dataset terms have been accepted for this project. See
[dataset and model licenses](docs/model-and-dataset-licenses.md) before preparing data or
running an evaluation.

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync --locked --group dev
uv run vlbench list models
uv run vlbench list datasets
uv run vlbench run --model openai_clip_b32 --dataset cifar10 --device cuda --dry-run
```

The dry run prints the planned model, checkpoint, dataset, and access status without
downloading anything. To install inference dependencies for a model family, choose only the
required extras. For example:

```powershell
uv sync --locked --group dev --extra cuda --extra openclip
uv run vlbench doctor --device cuda
```

Installing a Python extra does not download model weights or benchmark datasets. A real run
may download them and is separately gated by the CLI. `--yes` authorizes the reviewed network
plan; it does not accept legal terms. CIFAR-10/100 use official downloadable test archives
with report attribution, and XM3600 uses CC BY 4.0 annotations plus verified per-image
CC BY 2.0 attribution records; none of these files are redistributed. Use
`--confirm-data-terms` only when a dataset's exact terms require separate user acceptance.
See [reproducibility](docs/reproducibility.md) for model extras and run procedures.

## Documentation

- [Benchmark specification](docs/benchmark-specification.md)
- [Methodology](docs/methodology.md)
- [Model and dataset licenses](docs/model-and-dataset-licenses.md)
- [Development environment](docs/environment.md)
- [Reproducibility and CLI](docs/reproducibility.md)
- [Implementation status](docs/implementation-status.md)
- [Results status](results/README.md)
