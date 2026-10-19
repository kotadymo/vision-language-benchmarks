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

For a dataset marked `requires_user_confirmation`, inspect the original terms and confirm you
are authorized before passing `--confirm-data-terms`. The separate `--yes` flag authorizes
network downloads after reviewing the plan. Example commands that acquire data or weights
must only be run after those checks; blocked datasets remain unavailable regardless of flags.

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

Raw result JSON and run manifests are git-ignored. Do not commit data, weights, caches, tokens,
or embedding arrays. Reports should be published only when they derive from genuine,
authorized runs. Current result status is documented in [results/README](../results/README.md).

## Data and cache locations

By default, data and caches are stored under the project root and excluded from Git. Set
`VLBENCH_DATA_DIR` or `VLBENCH_CACHE_DIR` before running the CLI to move those locations.
Output locations are selected with `--output`.
