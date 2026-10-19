# Vision-Language Benchmarks

A reproducible evaluation framework for zero-shot image classification and image-text
retrieval with CLIP, OpenCLIP, SigLIP 2, EmbeddingGemma 2, and Japanese or multilingual
models.

## Status

The v0.1.0 framework, result schema, report generator, and offline tests are implemented.
No real model evaluations or measured results have been published. The tests use synthetic
features and mocked model loading; they do not download datasets or model weights. Numerical
parity with CLIP Benchmark and the Phase 1–3 dataset evaluations remain outstanding.

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
may download them and is separately gated by the CLI. `--yes` authorizes planned network
downloads; `--confirm-data-terms` records that you reviewed and are authorized to use a
dataset. Do not pass either flag until you have reviewed the exact plan and applicable terms.
See [reproducibility](docs/reproducibility.md) for other model extras and run procedures.

## Documentation

- [Benchmark specification](docs/benchmark-specification.md)
- [Methodology](docs/methodology.md)
- [Model and dataset licenses](docs/model-and-dataset-licenses.md)
- [Development environment](docs/environment.md)
- [Reproducibility and CLI](docs/reproducibility.md)
- [Results status](results/README.md)
