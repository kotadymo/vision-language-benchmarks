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
uv run --locked --extra cuda --extra openclip --extra gemma --extra siglip2 --extra japanese-clip pytest -q -s tests/real_model/test_real_model_adapters.py
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
$env:VLBENCH_REAL_BENCHMARK_OUTPUT = "results/v0.1.0"
uv run --locked --extra cuda --extra openclip pytest -q -s tests/real_model/test_real_runner.py
Remove-Item Env:VLBENCH_REAL_BENCHMARK_OUTPUT
Remove-Item Env:VLBENCH_RUN_REAL_BENCHMARK_TESTS
```

By default it writes its schema-validated result and run manifest into pytest's temporary
directory. Setting `VLBENCH_REAL_BENCHMARK_OUTPUT` instead stores the completed run under that
result root, where the raw JSON can be validated and considered for publication. Verify the
recorded source commit, dataset integrity, and run status before publishing.
This existing opt-in test follows the backwards-compatible `all_captions` default. To exercise
the official first-caption selection, use `--retrieval-variant official_first_caption` in a
`vlbench run` command; do not infer one variant from the other.

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

Exploratory runs default to `results/v0.1.0/`; use `--output` to select another result root.
The runner records a schema-validated JSON file per run and a manifest of run status. For
publishable measurements, use strict reproducibility mode and an ignored staging directory;
never write inference outputs directly into the tracked publication directory. `--resume`
reuses a matching completed configuration; `--force` creates a new run. These flags are
mutually exclusive, and neither bypasses strict source checks.

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

## Reproduce the published measurements

The current v0.1.0 baseline contains 38 completed full-dataset CUDA FP32 raw JSON records: 24
CIFAR classification runs (CIFAR-10 and CIFAR-100, six English-compatible models, P0 and P1),
seven historical XM3600 `all_captions` runs, and seven earlier `official_first_caption` runs.
The seven historical files predate the variant field and are identified as `all_captions`
from their candidate counts; their SHA-256 hashes and audit evidence are in
[`historical-results-audit.json`](../results/v0.1.0/validation/historical-results-audit.json).
They remain byte-for-byte unchanged and must not be silently relabeled or replaced.

The previous seven official-compatible runs evaluated the complete 3,600-image XM3600 split
in FP32 with seed 0 and requested batch size 16: EmbeddingGemma 2, multilingual OpenCLIP
ViT-B/32, and SigLIP 2 Base in English and Japanese, plus rinna Japanese CLIP ViT-B/16 in
Japanese. The audit found one clean worktree and six runs with `working_tree_dirty: true`.
The run-status record attributes the observed dirty state to generated raw results being
untracked during later runs; the exact historical uncommitted file set is not recoverable.
Actual batch sizes, OOM history, throughput, latency, peak memory, source commit, model and
tokenizer revisions, checkpoint hashes, dataset checksum, ordered candidate hashes, and
caption-content hashes remain in each unchanged raw result. The
[generated report](../results/v0.1.0/report.md) links every measured row to its raw JSON.

For XM3600, pass `--retrieval-variant` explicitly. The CLI keeps `all_captions` as the
backwards-compatible default, but explicit selection records the variant in the run
configuration, configuration hash, run ID, raw metadata, and resume identity. `P2` remains
the multilingual comparison protocol; it is distinct from the caption-selection variant.

The pinned CLIP Benchmark loader selects `captions[0]` in source order and excludes only
records whose image files are missing. Local validation found 3,600 images, zero exclusions,
one caption per image, and identical ordered image IDs in English and Japanese. Thus
`official_first_caption` has 3,600 candidates in each language. Pair language results only
when variant, candidate count and ordered IDs, dataset/checkpoint/source revisions, precision,
preprocessing, tokenizer, prompts, and runtime conditions match. Equal candidate counts do
not equalize caption content, difficulty, translation quality, or dataset composition; report
English/Japanese differences as observed performance under those conditions, not as a causal
measurement of language understanding.

## Strict clean-source XM3600 reruns

Publishable measurements require a committed, clean Git revision, CUDA, a matching lockfile,
and an ignored staging directory. Finish all code and documentation changes, run offline tests,
and commit them before benchmarking. Capture the full commit SHA and verify that all Git
status and diff checks are clean:

```powershell
$sourceCommit = (git rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw "Cannot read the source commit." }
if (git status --porcelain --untracked-files=all) { throw "The Git worktree is not clean." }
git diff --exit-code
if ($LASTEXITCODE -ne 0) { throw "Tracked source changes are present." }
git diff --cached --exit-code
if ($LASTEXITCODE -ne 0) { throw "Staged changes are present." }

$stage = ".cache\benchmark-staging\clean-xm3600"
uv run --locked --extra cuda --extra gemma vlbench doctor `
  --reproducibility --model embeddinggemma2 --dataset crossmodal3600 `
  --protocol p2 --language en --retrieval-variant official_first_caption `
  --precision fp32 --device cuda --batch-size 16 --seed 0 `
  --output $stage --expected-source-commit $sourceCommit
if ($LASTEXITCODE -ne 0) { throw "Strict reproducibility preflight failed." }
```

The runner repeats strict Git and staging checks for every measurement, verifies the frozen
source SHA after inference and persistence, and rejects a dirty or mismatched run. This
PowerShell loop covers the seven required model/language combinations. Each command writes
only to the ignored staging area. `--yes` authorizes the reviewed download plan; it does not
accept legal terms.

```powershell
$runs = @(
  @{ model = "embeddinggemma2"; language = "en"; extra = "gemma" },
  @{ model = "embeddinggemma2"; language = "ja"; extra = "gemma" },
  @{ model = "openclip_multilingual_b32"; language = "en"; extra = "openclip" },
  @{ model = "openclip_multilingual_b32"; language = "ja"; extra = "openclip" },
  @{ model = "siglip2_base_224"; language = "en"; extra = "siglip2" },
  @{ model = "siglip2_base_224"; language = "ja"; extra = "siglip2" },
  @{ model = "rinna_ja_clip_b16"; language = "ja"; extra = "japanese-clip" }
)

foreach ($run in $runs) {
  uv run --locked --extra cuda --extra $run.extra vlbench run `
    --model $run.model --dataset crossmodal3600 --protocol p2 `
    --language $run.language --retrieval-variant official_first_caption `
    --precision fp32 --device cuda:0 --batch-size 16 --seed 0 `
    --output $stage --strict-reproducibility `
    --expected-source-commit $sourceCommit --yes
  if ($LASTEXITCODE -ne 0) {
    throw "Strict XM3600 run failed: $($run.model) / $($run.language)."
  }
}

if ((git rev-parse HEAD).Trim() -ne $sourceCommit) {
  throw "The source revision changed during the benchmark suite."
}
if (git status --porcelain --untracked-files=all) {
  throw "The worktree became dirty during the benchmark suite."
}
```

After all seven results validate, copy them into the publication raw directory without
overwriting existing records. Generate a lifecycle manifest that marks matching strict runs
active, replaced first-caption runs superseded, other completed measurements historical, and
failed/incomplete attempts separately. Then generate the publishable artifacts from the
combined raw results:

```powershell
$destination = "results\v0.1.0\raw"
Get-ChildItem (Join-Path $stage "raw") -Filter "*.json" | ForEach-Object {
  $target = Join-Path $destination $_.Name
  if (Test-Path $target) { throw "Refusing to overwrite historical result: $target" }
  Copy-Item -LiteralPath $_.FullName -Destination $target
}

uv run --locked vlbench lifecycle --input $destination `
  --frozen-source-commit $sourceCommit `
  --attempt-manifest (Join-Path $stage "run-manifest.json")
if ($LASTEXITCODE -ne 0) { throw "Result lifecycle generation failed." }

uv run --locked vlbench report --publishable --input $destination `
  --output results\v0.1.0 `
  --lifecycle results\v0.1.0\validation\result-lifecycle.json `
  --expected-source-commit $sourceCommit
if ($LASTEXITCODE -ne 0) { throw "Publishable report validation failed." }
```

Do not pass `--confirm-data-terms` without the required user decision, or run blocked datasets.
For exact reproduction, use the recorded model, tokenizer, dataset, checkpoint, and source
revisions plus every setting in the raw record. Never alter historical raw metadata to make
it appear clean; retain it and use the generated lifecycle state to distinguish it from
validated replacements.

The official parity evidence is deliberately scoped: OpenAI CLIP ViT-B/32 on CIFAR-10 P0 is
exact at `1e-12` tolerance. The Japanese XM3600 OpenCLIP comparison is tied to the exact
measurement ID and parity artifact recorded in
[validation](../results/v0.1.0/validation/). It does not establish end-to-end parity for
other model/language/variant combinations.

## Data and cache locations

By default, data and caches are stored under the project root and excluded from Git. Set
`VLBENCH_DATA_DIR` or `VLBENCH_CACHE_DIR` before running the CLI to move those locations.
Output locations are selected with `--output`.
