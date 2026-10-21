# EmbeddingGemma Matryoshka dimensions

`vlbench` supports native Matryoshka dimension inference for `embeddinggemma2` through
`openclip-gemma==0.4.0`. Supported dimensions are 768, 512, 256, and 128; omitting
`--embedding-dimension` selects the existing 768D output. Passing the option to another model
is rejected.

## Inference semantics

Every dimension is a separate evaluation. The adapter calls
`encode_image(..., truncate_dim=<dimension>)` and
`encode_text(..., truncate_dim=<dimension>)` for that run. It does not reuse 768D features,
slice embeddings in the benchmark engine, or reuse rankings across dimensions. The library
uses the leading coordinates and L2-normalizes the truncated vectors
(`leading-dimensions-l2-v1`). The adapter checks the returned dimension, finiteness, and unit
norm before metric computation.

The selected dimension is part of the run configuration hash and run ID. Result JSON and
summary CSV files retain the dimension and inference strategy; historical result files without
these fields remain valid and unchanged. Dimension results belong in a separate publication
directory rather than `results/v0.1.0/`.

## Verified v0.4.0 suite

The [published matrix](../results/mrl-openclip-gemma-v0.4.0/embedding-dimension-analysis.md)
contains 24 independent full-dataset runs from frozen source commit
`a9909fbe1b15dc66b44bca51a41651f8f3a5d30c`: CIFAR-10/100 P0/P1 at all four dimensions, plus
XM3600 `official_first_caption` in English and Japanese at all four dimensions. The CIFAR
conditions use all 10,000 official test images; each XM3600 condition uses 3,600 images and
3,600 first-caption candidates. The exact dataset checksums, candidate-order checks, and run
provenance are in the [dimension audit](../results/mrl-openclip-gemma-v0.4.0/validation/embedding-dimension-audit.json).

All runs use EmbeddingGemma revision `914f7f89142e33e77833254d9c9b90c3cef7303b`,
checkpoint SHA-256 `1ff16e54c7410c6a1c8a8fddd076136fba78368eec798b45a0f9532e83b2f510`,
FP32, seed 0, requested batch size 16, Python 3.12.10, PyTorch 2.14.1+cu130, CUDA 13.0,
driver 610.88, and an NVIDIA RTX 4070 Ti SUPER. Each records a clean worktree and the same
source commit. Actual image/text batches were 16/16 for CIFAR and 11/16 for XM3600: the image
adapter groups aspect-ratio-aware preprocessed tensors by shape. No run needed an OOM retry.
The [run-status manifest](../results/mrl-openclip-gemma-v0.4.0/run-status.json) records a
separate pre-fix schema-invalid pilot; it produced no persisted raw measurement and is excluded
from the 24 active conditions.

All 20 measured values in the fresh 768D versus v0.3.0 comparison are numerically identical.
This is not a claim that every old execution condition is fully verified: the historical CIFAR
records lack preprocessing provenance, and v0.4.0 adds native-dimension metadata to the
preprocessing configuration while retaining the same XM3600 transform, resolution, and
shape-grouping fields. See the [baseline audit](../results/mrl-openclip-gemma-v0.4.0/validation/embedding-dimension-baseline-comparison.json)
for the per-condition status.

## Strict run example

Publishable runs require a clean, committed checkout and an ignored staging directory:

```powershell
$sourceCommit = "<full frozen source commit>"
$staging = ".cache\benchmark-staging\embeddinggemma-mrl"

uv run --locked vlbench doctor --reproducibility `
  --model embeddinggemma2 --dataset cifar10 --protocol p0 `
  --embedding-dimension 128 --device cuda --output $staging `
  --expected-source-commit $sourceCommit

uv run --locked vlbench run --model embeddinggemma2 --dataset cifar10 `
  --protocol p0 --language en --embedding-dimension 128 `
  --device cuda --precision fp32 --batch-size 16 --seed 0 `
  --output $staging --strict-reproducibility `
  --expected-source-commit $sourceCommit --yes
```

For the full matrix, start with a fresh ignored staging directory and run each condition
separately:

```powershell
$sourceCommit = "a9909fbe1b15dc66b44bca51a41651f8f3a5d30c"
$staging = ".cache\benchmark-staging\embeddinggemma-mrl-repro"

uv run --locked --extra cuda --extra gemma vlbench doctor `
  --reproducibility --model embeddinggemma2 --dataset cifar10 `
  --protocol p0 --language en --embedding-dimension 768 `
  --precision fp32 --device cuda --batch-size 16 --seed 0 `
  --output $staging --expected-source-commit $sourceCommit
if ($LASTEXITCODE -ne 0) { throw "Strict reproducibility preflight failed." }

$conditions = @(
  @{ dataset = "cifar10"; protocol = "p0"; language = "en"; variant = $null },
  @{ dataset = "cifar10"; protocol = "p1"; language = "en"; variant = $null },
  @{ dataset = "cifar100"; protocol = "p0"; language = "en"; variant = $null },
  @{ dataset = "cifar100"; protocol = "p1"; language = "en"; variant = $null },
  @{ dataset = "crossmodal3600"; protocol = "p2"; language = "en"; variant = "official_first_caption" },
  @{ dataset = "crossmodal3600"; protocol = "p2"; language = "ja"; variant = "official_first_caption" }
)

foreach ($dimension in @(768, 512, 256, 128)) {
  foreach ($condition in $conditions) {
    $variantArguments = @()
    if ($condition.variant) {
      $variantArguments = @("--retrieval-variant", $condition.variant)
    }
    uv run --locked --extra cuda --extra gemma vlbench run `
      --model embeddinggemma2 --dataset $condition.dataset `
      --protocol $condition.protocol --language $condition.language `
      @variantArguments --embedding-dimension $dimension `
      --precision fp32 --device cuda --batch-size 16 --seed 0 `
      --output $staging --strict-reproducibility `
      --expected-source-commit $sourceCommit --yes
    if ($LASTEXITCODE -ne 0) {
      throw "Strict MRL run failed: $($condition.dataset) / $($condition.protocol) / $($condition.language) / ${dimension}D."
    }
  }
}

if ((git rev-parse HEAD).Trim() -ne $sourceCommit) {
  throw "The source revision changed during the benchmark suite."
}
if (git status --porcelain --untracked-files=all) {
  throw "The source worktree became dirty during the benchmark suite."
}
```

This runs 24 full-dataset conditions. The strict runner validates the source and staging area
before and after each run. For publication, copy validated raw JSON into the separate results
directory only after all conditions complete; never write inference output directly into a
tracked publication directory.

## Validation and comparison

The offline test suite verifies configuration, result-schema, lifecycle, and reporting
behavior. The opt-in CUDA equivalence test uses real CIFAR-10 test images and prompts plus
XM3600 English/Japanese `official_first_caption` texts. It tests image, classification P0/P1,
and retrieval text paths at 768D, 512D, 256D, and 128D. For validation only, each native
`truncate_dim=N` output is compared with the L2-normalized leading N coordinates of a separate
native 768D output; the full-dataset benchmark results continue to use native inference.

```powershell
$env:VLBENCH_RUN_MRL_INTEGRATION = "1"
$env:VLBENCH_MRL_VALIDATION_OUTPUT_DIR = ".cache\mrl-validation-staging"
try {
  uv run --locked --extra cuda --extra gemma pytest -q `
    tests/real_model/test_embeddinggemma_mrl_equivalence.py
  if ($LASTEXITCODE -ne 0) {
    throw "Native MRL equivalence validation failed."
  }
}
finally {
  Remove-Item Env:VLBENCH_RUN_MRL_INTEGRATION -ErrorAction SilentlyContinue
  Remove-Item Env:VLBENCH_MRL_VALIDATION_OUTPUT_DIR -ErrorAction SilentlyContinue
}
```

The opt-in test writes a measured JSON and Markdown report to the selected output directory.
The published validation covers 20 dimension/category checks over 176 embeddings (2 images,
38 classification prompts, and 4 retrieval texts). Its maximum native-versus-reference
absolute difference was `2.980e-08`; the adapter's NumPy FP32 output matched direct native
output exactly. All checks passed with `rtol=1e-5`, `atol=1e-6`, and zero invalid embeddings.
See the [detailed numerical report](../results/mrl-openclip-gemma-v0.4.0/validation/native-mrl-api-equivalence.md)
and [machine-readable results](../results/mrl-openclip-gemma-v0.4.0/validation/native-mrl-api-equivalence.json).
This small-sample result does not reproduce the 24 full-dataset accuracy measurements.

Dimension reports include the measured classification/retrieval scores, throughput, latency,
VRAM, OOM retries, and score changes versus the matching 768D condition. The estimated
FP32 vector payload is `dimension × 4` bytes (3,072 bytes at 768D and 512 bytes at 128D); this
does not include model weights, indexes, metadata, or storage-system overhead.

To add the fresh 768D versus historical v0.3.0 comparison, publish a lifecycle manifest for the
new runs, then pass both the old raw directory and its matching lifecycle manifest:

```powershell
uv run --locked vlbench lifecycle `
  --input results\mrl-openclip-gemma-v0.4.0\raw `
  --frozen-source-commit $sourceCommit `
  --output results\mrl-openclip-gemma-v0.4.0\validation\result-lifecycle.json

uv run --locked vlbench report `
  --input results\mrl-openclip-gemma-v0.4.0\raw `
  --output results\mrl-openclip-gemma-v0.4.0 `
  --publishable `
  --lifecycle results\mrl-openclip-gemma-v0.4.0\validation\result-lifecycle.json `
  --expected-source-commit $sourceCommit `
  --baseline-input results\v0.1.0\raw `
  --baseline-lifecycle results\v0.1.0\validation\result-lifecycle.json
```

Only completed v0.3.0 EmbeddingGemma results marked active or historical by the baseline
lifecycle are compared; superseded runs are excluded. Metric differences are retained as
measured. Missing historical provenance is marked unverified in
`validation\embedding-dimension-baseline-comparison.json` rather than treated as a match.
