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

Run the other dimensions and dataset/protocol/language combinations as separate invocations.
For XM3600, specify `--dataset crossmodal3600 --protocol p2 --retrieval-variant
official_first_caption` and run English and Japanese separately. Do not point inference output
at a tracked results directory. Copy validated raw JSON to the dedicated publication
directory only after all required runs have completed.

## Validation and comparison

The offline test suite verifies configuration, result-schema, lifecycle, and reporting
behavior. The opt-in real-model test checks that each native image and text output matches
manual prefix truncation followed by FP32 L2 normalization:

```powershell
$env:VLBENCH_RUN_MRL_INTEGRATION = "1"
uv run --locked --extra cuda --extra gemma pytest -q `
  tests/real_model/test_embeddinggemma_mrl_equivalence.py
Remove-Item Env:VLBENCH_RUN_MRL_INTEGRATION
```

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
