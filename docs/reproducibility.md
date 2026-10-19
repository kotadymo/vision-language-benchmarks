# Reproducibility and CLI

## Set up

Use Python 3.12 and uv. The base environment is sufficient for linting, schema validation,
synthetic tests, and report generation:

```powershell
uv lock --check
uv sync --locked --group dev
uv run --locked ruff check .
uv run --locked pytest
```

Install only the inference dependencies needed for the model family:

```powershell
# OpenAI CLIP, LAION OpenCLIP, and multilingual OpenCLIP
uv sync --locked --group dev --extra cuda --extra openclip

# EmbeddingGemma 2
uv sync --locked --group dev --extra cuda --extra gemma

# SigLIP 2
uv sync --locked --group dev --extra cuda --extra siglip2

# rinna Japanese CLIP
uv sync --locked --group dev --extra cuda --extra japanese-clip
```

These commands install Python packages, not model weights or datasets. Changing the selected
extras with `uv sync` may remove extras that are not named in that command. Check the
environment before a planned run:

```powershell
uv run vlbench doctor --device cuda
```

## Optional real-model integration tests

The default test suite skips these network- and GPU-dependent tests. Install all model
families and run the seven real adapter smoke tests sequentially with:

```powershell
uv sync --locked --group dev --extra cuda --extra openclip --extra gemma --extra siglip2 --extra japanese-clip
$env:VLBENCH_RUN_REAL_MODEL_TESTS = "1"
uv run --locked pytest -q -s tests/real_model/test_real_model_adapters.py
Remove-Item Env:VLBENCH_RUN_REAL_MODEL_TESTS
```

Each smoke test downloads or reuses the pinned pretrained checkpoint, performs CUDA image and
language-appropriate text inference, validates finite and dimension-compatible embeddings and
normalized cosine similarities, records checkpoint/runtime/GPU-memory metadata, then releases
the model before the next test. Local execution logs are written under `.cache/real-smoke/` and
are excluded from Git.

To exercise the complete benchmark runner against all 3,600 authorized XM3600 images and the
7,185 Japanese captions, run this separate, longer integration test:

```powershell
$env:VLBENCH_RUN_REAL_BENCHMARK_TESTS = "1"
uv run --locked pytest -q -s tests/real_model/test_real_runner.py
Remove-Item Env:VLBENCH_RUN_REAL_BENCHMARK_TESTS
```

It writes its schema-validated result and run manifest into pytest's temporary directory.
Publish benchmark results only from deliberate `vlbench run` commands after verifying the
recorded source commit, dataset integrity, and run status.

## Inspect plans before acquisition

List the registered models and datasets, then run a dry plan:

```powershell
uv run vlbench list models
uv run vlbench list datasets
uv run vlbench run --model openai_clip_b32 --dataset cifar10 --device cuda --dry-run
```

`--dry-run` performs no dataset or checkpoint downloads. Review the output for the pinned
checkpoint revision and size, dataset split and access state, language, protocol, precision,
device, and output path.

For a dataset marked `requires_user_confirmation`, review the exact terms and obtain any
required user acceptance before passing `--confirm-data-terms`; this flag is not legal
acceptance by itself. CIFAR-10/100 use the official University of Toronto archives with
attribution to the dataset report and no redistribution. XM3600 is published with CC BY 4.0
annotations and per-image CC BY 2.0 attribution metadata; preparation validates all image
licenses and caption/image IDs. The separate `--yes` flag authorizes the reviewed network
plan. Blocked datasets remain unavailable regardless of flags.

## Run records and reports

Runs default to `results/v0.1.0/`; use `--output` to select another result root. The runner
records a schema-validated JSON file per run and a manifest of run status. `--resume` reuses a
matching completed configuration; `--force` creates a new run. These flags are mutually
exclusive.

Regenerate all report artifacts from raw JSON:

```powershell
uv run vlbench report --input results/v0.1.0/raw --output results/v0.1.0
```

The report command validates each JSON file before aggregation. Completed results contribute
to numeric tables and plots; blocked, skipped, and failed runs remain visible in the status
report. The report generator uses a fixed SVG hash salt and fixed metadata so repeated
generation from the same raw JSON is byte-identical. Automated tests cover CSV, Markdown,
metadata JSON, PNG, and SVG regeneration.

Validated, sanitized per-run metric JSON under `results/v0.1.0/raw/` may be committed. The
private `run-manifest.json` is git-ignored. Do not commit datasets, model weights, caches,
tokens, credentials, local filesystem paths, or embedding arrays. Reports should be published
only when they derive from complete, genuine, authorized runs. Current result status is
documented in [implementation status](implementation-status.md) and
[results/README](../results/README.md).

## Data and cache locations

By default, data and caches are stored under the project root and excluded from Git. Set
`VLBENCH_DATA_DIR` or `VLBENCH_CACHE_DIR` before running the CLI to move those locations.
Output locations are selected with `--output`.
