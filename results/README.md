# Benchmark Results

## EmbeddingGemma 2 native Matryoshka dimensions

The separate [v0.4.0 report](mrl-openclip-gemma-v0.4.0/report.md) and
[dimension analysis](mrl-openclip-gemma-v0.4.0/embedding-dimension-analysis.md) summarize 24
completed full-dataset FP32 CUDA evaluations through `openclip-gemma==0.4.0`: four dimensions
(768/512/256/128) across CIFAR-10/100 P0/P1 and XM3600 `official_first_caption` English/Japanese.
All use frozen source commit `a9909fbe1b15dc66b44bca51a41651f8f3a5d30c`, the pinned
EmbeddingGemma 2 checkpoint, seed 0, and requested batch size 16. All runs record
`working_tree_dirty: false`; there were no OOM retries. CIFAR image/text batches were 16/16.
XM3600 image/text batches were 11/16 because the aspect-ratio-aware image transform groups
preprocessed tensors by shape. The [run-status manifest](mrl-openclip-gemma-v0.4.0/run-status.json)
also records one earlier schema-invalid pilot attempt from commit
`737b77848c52987652bcc4ddfbb883dede2ba79b`; no raw measurement was persisted and it is excluded
from the 24 active results.

The [summary CSV](mrl-openclip-gemma-v0.4.0/embedding-dimension-summary.csv),
[raw JSON records](mrl-openclip-gemma-v0.4.0/raw/), [dimension audit](mrl-openclip-gemma-v0.4.0/validation/embedding-dimension-audit.json),
[baseline comparison](mrl-openclip-gemma-v0.4.0/validation/embedding-dimension-baseline-comparison.json),
[lifecycle](mrl-openclip-gemma-v0.4.0/validation/result-lifecycle.json), and PNG/SVG charts are
published with the report. Every one of the 20 matched 768D metric values is identical to the
v0.3.0 baseline. The audit preserves the provenance limitation: the old CIFAR preprocessing
configuration is unavailable, while the XM3600 v0.4.0 configuration adds native-dimension
fields to the old shared image-transform settings.

## DINOv2 + dino.txt v0.6.0 integration

The [campaign report](dinov2-dinotxt-v0.6.0/report.md) publishes six full-dataset CUDA FP32
measurements from `openvlclip==0.6.0`, using the fixed 2048D DINOv2 ViT-L/14 + dino.txt
global embedding. All six used frozen source commit
`28519573b78f0dcb7ab6515ead02bc6b403a92bd` and a clean working tree. The
[license review](../docs/dinotxt-license-review.md) documents why the pinned Meta terms support
the local scientific benchmark; the original six blocked attempts remain unchanged as
historical records and contribute no metrics. The report includes matched CIFAR and XM3600
comparisons, exact hit counts, throughput, peak GPU memory, and PNG/SVG figures. The
[summary CSV](dinov2-dinotxt-v0.6.0/summary.csv),
[model-comparison CSV](dinov2-dinotxt-v0.6.0/model-comparison.csv),
[raw results](dinov2-dinotxt-v0.6.0/raw/),
[run-status manifest](dinov2-dinotxt-v0.6.0/run-status.json),
[publication lifecycle](dinov2-dinotxt-v0.6.0/validation/result-lifecycle.json),
[comparison audit](dinov2-dinotxt-v0.6.0/validation/model-comparison-audit.json),
[checkpoint integrity record](dinov2-dinotxt-v0.6.0/validation/checkpoint-integrity.json), and
[CUDA smoke-validation record](dinov2-dinotxt-v0.6.0/validation/dinotxt-cuda-smoke.json) retain
the associated provenance.

## v0.1.0 benchmark results

The raw archive contains 45 schema-validated, completed full-dataset CUDA FP32 measurements:
24 CIFAR-10/100 classification runs, seven historical XM3600 `all_captions` runs, seven
superseded earlier `official_first_caption` runs, and seven clean replacement runs. The
[v0.1.0 report](v0.1.0/report.md) and [CSV summary](v0.1.0/summary.csv) contain 38
leaderboard-eligible rows: 31 historical measurements and seven active replacements. The
superseded first-caption records remain in the raw archive and are excluded from primary
aggregations by the [publication lifecycle](v0.1.0/validation/result-lifecycle.json). All
seven active replacements use source commit
`9d170b6071eecd75d7100ca70092d44d86650f01`, 3,600 images, and 3,600 first-caption candidates
per language; every replacement records a clean worktree. The [metadata](v0.1.0/metadata.json),
[run-status manifest](v0.1.0/run-status.json), and [raw per-run JSON](v0.1.0/raw/) are
published alongside the report. Sanitized real-model CUDA smoke-test execution evidence is in
[model-smoke-tests.json](v0.1.0/validation/model-smoke-tests.json). The
[official CIFAR-10 parity record](v0.1.0/validation/official-cifar10-parity.json) compares
OpenAI CLIP P0 with the pinned CLIP Benchmark implementation. The
[historical XM3600 parity record](v0.1.0/validation/official-xm3600-parity.json) is retained
unchanged. The separate
[clean XM3600 parity record](v0.1.0/validation/official-xm3600-parity-clean.json) compares the
active Japanese multilingual OpenCLIP replacement with the pinned official reference; the
[official CLI output](v0.1.0/validation/official-xm3600-openclip-ja.json) is retained.
The [historical-results audit](v0.1.0/validation/historical-results-audit.json) records the
31 original raw-file hashes, provenance checks, and legacy XM3600 variant assignment without
modifying the measured JSON. The
[pre-publication preservation manifest](v0.1.0/validation/preexisting-raw-results-preservation.json)
records SHA-256 hashes for all 38 raw files that existed before the replacements were added.
The [clean-suite comparison audit](v0.1.0/validation/clean-xm3600-reproducibility-audit.json)
records previous/new metrics, revisions, checksums, candidate-order checks, and worktree states.

The seven historical XM3600 runs retain all captions (3,600 images; 7,200 English or 7,185
Japanese candidates). They are not directly comparable with the pinned official XM3600
selection, which uses the first caption per image, and their unequal language candidate counts
do not support a matched-candidate English/Japanese delta. These original raw files are
preserved unchanged and remain a separate `all_captions` result set.

The seven clean replacements use P2, FP32, seed 0, requested batch size 16, and the pinned
pretrained checkpoints. EmbeddingGemma 2, multilingual OpenCLIP ViT-B/32, and SigLIP 2 Base
were evaluated in English and Japanese; rinna Japanese CLIP ViT-B/16 was evaluated in
Japanese. Every recall metric exactly matches its prior corresponding measurement. The report
presents the two variants in separate tables and plots, and reports matched-language deltas
only for the three models with paired runs. Language differences are observed performance
under each caption condition, not causal estimates of language understanding. Every result
table row links to its raw JSON.

Regenerate the lifecycle and publishable artifacts from raw JSON with:

```powershell
uv run --locked vlbench lifecycle --input results/v0.1.0/raw `
  --frozen-source-commit 9d170b6071eecd75d7100ca70092d44d86650f01

uv run --locked vlbench report --publishable `
  --input results/v0.1.0\raw --output results\v0.1.0 `
  --lifecycle results\v0.1.0\validation\result-lifecycle.json `
  --expected-source-commit 9d170b6071eecd75d7100ca70092d44d86650f01
```

Only complete and authorized measurements contribute numeric results. Failed attempts and
unmeasured/access-blocked combinations are not converted into scores; their statuses and
blockers are recorded in the run-status manifest and implementation status. The private
`run-manifest.json` remains git-ignored. Dataset images, model weights, embeddings, caches,
credentials, and machine-specific paths must never be committed. See
[reproducibility](../docs/reproducibility.md) for exact commands.
