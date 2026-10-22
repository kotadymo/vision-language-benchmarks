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

# DINOv2 + dino.txt (installing code does not authorize model-weight use)
uv sync --locked --group dev --extra cuda --extra dinotxt

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

The DINOv2+dino.txt CUDA smoke test is separately opt-in. The model-level gate is
`verified_for_benchmark_use` for local, non-redistributive scientific evaluation under the
pinned Meta terms; review the [license determination](dinotxt-license-review.md) before
running inference. Dataset safeguards and checkpoint hash verification remain active:

```powershell
$env:VLBENCH_RUN_DINOTXT_REAL_MODEL_TESTS = "1"
$env:VLBENCH_DINOTXT_SMOKE_OUTPUT = ".cache\real-smoke\dinotxt-cuda-smoke.json"
uv run --locked --extra cuda --extra dinotxt pytest -q -s tests/real_model/test_dinov2_dinotxt.py
Remove-Item Env:VLBENCH_DINOTXT_SMOKE_OUTPUT
Remove-Item Env:VLBENCH_RUN_DINOTXT_REAL_MODEL_TESTS
```

This opt-in test loads the official model and performs actual CUDA inference. It validates
preprocessing and tokenizer shapes, finite and normalized 2048D image/text features,
image-text cosine similarity, GPU memory, checkpoint hashes, and direct numerical consistency
with the pinned official PyTorch Hub model. Japanese encoding does not establish Japanese
semantic quality. See the
[license and integration record](dinotxt-integration.md).

To exercise the complete benchmark runner against all 3,600 authorized XM3600 images and the
7,185 Japanese captions, run this separate, longer integration test:

```powershell
$env:VLBENCH_RUN_REAL_BENCHMARK_TESTS = "1"
$env:VLBENCH_REAL_BENCHMARK_OUTPUT = ".cache\benchmark-staging\real-runner-test"
uv run --locked --extra cuda --extra openclip pytest -q -s tests/real_model/test_real_runner.py
Remove-Item Env:VLBENCH_REAL_BENCHMARK_OUTPUT
Remove-Item Env:VLBENCH_RUN_REAL_BENCHMARK_TESTS
```

By default it writes its schema-validated result and run manifest into pytest's temporary
directory. If `VLBENCH_REAL_BENCHMARK_OUTPUT` is set, use an ignored staging directory as
shown above; never point this integration test at the tracked publication directory. This
opt-in test is not a substitute for strict clean-source checks and its output is not
publishable without the full validation and lifecycle workflow below.
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
For `dinov2_dinotxt`, the plan shows model access as `available` for the documented local
benchmark use. Dataset access and attribution checks remain independent; neither `--yes` nor
`--confirm-data-terms` bypasses those checks.

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

EmbeddingGemma Matryoshka runs use the independent native dimensions 768/512/256/128 and
include the selected dimension in run identity and publication metadata. See the
[EmbeddingGemma dimension workflow](embedding-dimensions.md) for v0.4.0 inference, equivalence
checks, and separate publication commands.

Generate lifecycle assignments, then regenerate only publishable report artifacts:

```powershell
$sourceCommit = "9d170b6071eecd75d7100ca70092d44d86650f01"
uv run --locked vlbench lifecycle --input results\v0.1.0\raw `
  --frozen-source-commit $sourceCommit

uv run --locked vlbench report --publishable `
  --input results\v0.1.0\raw --output results\v0.1.0 `
  --lifecycle results\v0.1.0\validation\result-lifecycle.json `
  --expected-source-commit $sourceCommit
```

The report command validates each JSON file before aggregation. Validated active and
historical results contribute to primary numeric tables and plots; superseded measurements
are excluded, while blocked, skipped, failed, invalid, and incomplete attempts remain visible
in lifecycle/status artifacts. The report generator uses a fixed SVG hash salt and fixed
metadata so repeated generation from the same eligible raw JSON is byte-identical.
Automated tests cover CSV, Markdown, metadata JSON, PNG, and SVG regeneration.

Validated, sanitized per-run metric JSON under `results/v0.1.0/raw/` may be committed. The
private `run-manifest.json` is git-ignored. Do not commit datasets, model weights, caches,
tokens, credentials, local filesystem paths, or embedding arrays. Reports should be published
only when they derive from complete, genuine, authorized runs. Current result status is
documented in [implementation status](implementation-status.md) and
[results/README](../results/README.md).

## Reproduce the published measurements

The v0.1.0 raw archive contains 45 completed full-dataset CUDA FP32 JSON records: 24 CIFAR
classification runs (CIFAR-10 and CIFAR-100, six English-compatible models, P0 and P1), seven
historical XM3600 `all_captions` runs, and two sets of seven
`official_first_caption` runs. The primary report contains 38 eligible rows: the CIFAR runs,
the all-caption runs, and the seven active replacements. The earlier seven first-caption
records are preserved as superseded evidence and excluded from aggregation. The lifecycle
manifest distinguishes active, historical, superseded, invalid, failed, incomplete, blocked,
and skipped records. The 31 original CIFAR and all-caption raw files have their historical
audit; SHA-256 hashes for all 38 raw files that predated the replacement runs are recorded in
the separate [pre-publication preservation manifest](../results/v0.1.0/validation/preexisting-raw-results-preservation.json).

The earlier first-caption audit found one clean worktree and six runs with
`working_tree_dirty: true`. The recorded cause was generated raw result files being
untracked during later runs. The exact historical uncommitted file set cannot be recovered;
no contents are inferred and none of the historical raw result or parity JSON files was
rewritten. Per-run source SHAs and original dirty flags remain in their raw records.

The seven clean replacements evaluated the complete XM3600 split in FP32 with seed 0 and
requested batch size 16: EmbeddingGemma 2, multilingual OpenCLIP ViT-B/32, and SigLIP 2 Base
in English and Japanese, plus rinna Japanese CLIP ViT-B/16 in Japanese. All seven identify
frozen source commit `9d170b6071eecd75d7100ca70092d44d86650f01`, report
`working_tree_dirty: false`, and use 3,600 images and 3,600 first-caption candidates per
language. The maximum absolute difference from each earlier corresponding run is zero for all
six recall metrics. Exact per-run batch sizes, OOM history, throughput, latency, peak memory,
checkpoint/tokenizer revisions, hashes, candidate order, and source metadata remain in the
raw JSON and the [comparison audit](../results/v0.1.0/validation/clean-xm3600-reproducibility-audit.json).

The frozen measurement environment was Python 3.12.10, PyTorch 2.14.1+cu130, CUDA 13.0,
driver 610.88, and an NVIDIA RTX 4070 Ti SUPER with 16 GiB VRAM. The XM3600 dataset checksum
was `ed934a9782286a688ca79d4594aec679ca4accb8d738186f986c831711545fce`; all replacement
runs used the same 3,600-image ordering (SHA-256
`da0f6363692e680b85303ad510a7296fffb78b137744169fbef92e040f50adf5`) and text-ID order
(SHA-256 `544f56d541aec33ff83763f1636f2273214a8c02ab7f77c1aca58db061a79afb`). The model checkpoint
revisions and SHA-256 fingerprints were:

| Model | Checkpoint revision | Checkpoint SHA-256 |
|---|---|---|
| EmbeddingGemma 2 | `914f7f89142e33e77833254d9c9b90c3cef7303b` | `1ff16e54c7410c6a1c8a8fddd076136fba78368eec798b45a0f9532e83b2f510` |
| Multilingual OpenCLIP ViT-B/32 | `506d40eb551f4801a1c27fc20a31c7b8f590deda` | `f8bd344cf9398b5869c9001efca0f26b8c13d6f0d072f8830de5a992bac4a298` |
| SigLIP 2 Base | `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` | `5a074c08ae416f7af68e3e8e8e8fea5396d6dd359b978795f22f1ea112d64180` |
| rinna Japanese CLIP ViT-B/16 | `ce22bc39e6f5dcb449e082954211acfb00c2227b` | `043cae46a1214011b90081233f50e86c086bf31f40929e57a569bfb334feb4e8` |

All paired languages use the same checkpoint revision and hash, tokenizer revision,
preprocessing, prompts, precision, seed, dataset split, candidate count/order, and runtime
environment. EmbeddingGemma used actual image/text batch sizes 11/16 because of
aspect-ratio-aware shape grouping; other replacements used 16/16. No run required an OOM
retry. The [generated report](../results/v0.1.0/report.md) links every published row to its
raw JSON.

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

To reproduce the published source revision exactly, first check out the frozen measurement
commit rather than the later publication commit:

```powershell
git switch --detach 9d170b6071eecd75d7100ca70092d44d86650f01
```

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
exact at `1e-12` tolerance. Clean Japanese XM3600 parity was checked for multilingual
OpenCLIP ViT-B/32 run `ae4e1863504978e0` against CLIP Benchmark commit
`ee14a9a8a9bcde09d51f4a4b1dff5a75182a3596`. Candidate fingerprints and integer hit counts
match; the maximum absolute metric difference is `2.9404958135081927e-8`, within the
`5e-8` tolerance. The historical parity record is retained unchanged, and the clean result is
stored separately in
[official-xm3600-parity-clean.json](../results/v0.1.0/validation/official-xm3600-parity-clean.json).
This evidence does not establish end-to-end parity for other model/language/variant
combinations.

## Data and cache locations

By default, data and caches are stored under the project root and excluded from Git. Set
`VLBENCH_DATA_DIR` or `VLBENCH_CACHE_DIR` before running the CLI to move those locations.
Output locations are selected with `--output`.

## Published EmbeddingGemma Matryoshka suite

The separate [v0.4.0 MRL publication](../results/mrl-openclip-gemma-v0.4.0/report.md) contains
24 complete runs at 768D, 512D, 256D, and 128D, all recorded clean from source commit
`a9909fbe1b15dc66b44bca51a41651f8f3a5d30c`. The model revision, checkpoint hash, dataset
checksums, full run settings, actual batch sizes, OOM history, and XM3600 candidate-order
checks are linked from its [dimension audit](../results/mrl-openclip-gemma-v0.4.0/validation/embedding-dimension-audit.json)
and [run-status manifest](../results/mrl-openclip-gemma-v0.4.0/run-status.json).

The separate [native MRL equivalence report](../results/mrl-openclip-gemma-v0.4.0/validation/native-mrl-api-equivalence.md)
records a small-sample CUDA FP32 validation from source commit
`dfda41a311f7a6b86a89aec95b7dd2aeebfe3073`. With `openclip-gemma==0.4.0`, it covered image,
classification P0/P1, and English/Japanese retrieval text at all four dimensions. The maximum
absolute difference from the mathematical reference was `2.980e-08` with `rtol=1e-5` and
`atol=1e-6`; this validates API numerics, not the 24 full-dataset scores. The
[machine-readable report](../results/mrl-openclip-gemma-v0.4.0/validation/native-mrl-api-equivalence.json)
contains the sample hashes and environment provenance. Ranking agreement is deferred because
the published raw results do not contain per-query rankings or embeddings.

To reproduce it, check out the frozen measurement commit, use the strict staging workflow and
per-condition commands in [EmbeddingGemma dimension workflow](embedding-dimensions.md), and
keep output under an ignored directory such as `.cache\benchmark-staging\embeddinggemma-mrl`.
Do not run from the later publication commit if an exact source revision is required. The
earlier schema-invalid 768D pilot is recorded as an excluded failed attempt; it has no persisted
measurement and is not part of the report.
