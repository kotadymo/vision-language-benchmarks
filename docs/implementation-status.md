# Implementation and Evaluation Status

Last updated: 2026-10-20
Branch: `feature/benchmark-framework-v0.1.0`
Base branch head before the XM3600 protocol work: `5b512ecb1d0177d88817a4e07d54f86810b7f86f`
Measurement source commits: `2993187c2b960e6e6690c97205b22838d18a36a3`,
`60a61259c95e7c387e896b23084e3c3d0fc375f5`,
`d72e4e462d137e9b77581a507e42897ff427eaf1`,
`27f2ac614a1e6043fec8f16beceda202bb9272ef`, and
`cec36f68a1523062f44216d4568fe2612208e913` (recorded per run).
The seven clean XM3600 replacements use frozen source commit
`9d170b6071eecd75d7100ca70092d44d86650f01`.

## Verification milestones

| Milestone | Status | Evidence |
| --- | --- | --- |
| Offline unit suite | completed for current implementation | `uv run --locked pytest -q`: 64 passed, two opt-in CUDA modules skipped |
| Dependency lock, lint, and patch checks | completed | `uv lock --check`, `uv run --locked ruff check .`, and `git diff --check` passed |
| Pretrained checkpoints loaded on CUDA | completed | All seven registered adapters passed |
| Real image and text inference on CUDA | completed | All supported smoke-test languages passed; embeddings were finite and dimension-compatible |
| Full CIFAR evaluations | completed, historical | 24 runs: CIFAR-10 and CIFAR-100 full 10,000-image test splits, six English-compatible models, P0 and P1 reported separately |
| Historical XM3600 evaluations | completed, `all_captions` | Seven original P2 runs: EmbeddingGemma, multilingual OpenCLIP, and SigLIP in English/Japanese; rinna Japanese-only; 3,600 images per run; raw files hash-audited and unchanged |
| XM3600 variant implementation | implemented and unit-tested | Explicit `official_first_caption` and `all_captions`; variant-aware run identity, resume, result schema, reporting, and pairing |
| Official XM3600 loader validation | completed | Pinned loader selects `captions[0]` in annotation order; local annotations contain 3,600 images per language, zero missing-image exclusions, one caption per image, and identical ordered English/Japanese image IDs |
| Official XM3600 metric differential | completed for synthetic candidates | Both direction keys and R@1/5/10 compared to the pinned metric implementation; this is not a full-model evaluation |
| Prior `official_first_caption` CUDA evaluations | completed, superseded | Seven FP32, seed-0, requested-batch-16 runs; the audit found one clean run and six with `working_tree_dirty: true`; all seven raw records remain unchanged |
| Strict clean-source execution | implemented and offline-tested | `doctor --reproducibility`, strict CUDA runs, frozen-SHA and ignored-staging validation, post-run checks, lifecycle-aware publication, and superseded-result exclusion |
| Clean `official_first_caption` CUDA reruns | completed | Seven complete FP32 runs, all from frozen source `9d170b6071eecd75d7100ca70092d44d86650f01`; every run records `working_tree_dirty: false`; no OOM retries |
| Paired English/Japanese XM3600 comparison | completed for three multilingual models | All replacements use the same 3,600 ordered images/candidates and pinned checkpoints; every old/new recall metric is exactly equal, with zero absolute difference |
| End-to-end official XM3600 evaluation | completed for clean OpenCLIP ViT-B/32 Japanese run | Clean run `ae4e1863504978e0` matched the pinned reference candidates, checkpoint, tokenizer, preprocessing, and positive pairs |
| Official CLIP Benchmark parity | CIFAR-10 P0 and clean XM3600 OpenCLIP Japanese completed | Reference commit `ee14a9a8a9bcde09d51f4a4b1dff5a75182a3596`; CIFAR-10 P0 is exact at `1e-12`; clean XM3600 maximum absolute difference is `2.9404958135081927e-8` at `5e-8` tolerance |
| Historical raw-result audit and preservation | completed | The 31 original files remain schema-valid and byte-identical to baseline commit `5b512ecb1d0177d88817a4e07d54f86810b7f86f`; SHA-256 hashes for all 38 pre-replacement files are in the preservation manifest |

The prior seven-run audit found one clean first-caption run and six runs with
`working_tree_dirty: true`. The recorded run status attributes the observed dirty state to
generated raw result files being untracked during later runs. The exact historical uncommitted
file set cannot be reconstructed; no contents are inferred, and no historical raw metadata has
been changed. The exact historical parity JSON was also preserved. All seven replacements ran
from the same frozen source commit, used ignored staging, passed post-run clean-worktree
checks, and remain clean in the published raw metadata. Every previous/replacement metric
difference is zero. The
[comparison audit](../results/v0.1.0/validation/clean-xm3600-reproducibility-audit.json),
[lifecycle manifest](../results/v0.1.0/validation/result-lifecycle.json), and
[pre-publication preservation manifest](../results/v0.1.0/validation/preexisting-raw-results-preservation.json)
provide per-run evidence.

The smoke tests use one generated image and two text examples per supported language. They
validate the actual pretrained image and text encoders, feature shape and finiteness, normalized
cosine similarity, and CUDA placement. These are adapter checks, not benchmark measurements.
All seven adapters passed these smoke checks. Sanitized execution records are published in
[`model-smoke-tests.json`](../results/v0.1.0/validation/model-smoke-tests.json); the original
local logs remain under the ignored `.cache/real-smoke/` directory. Machine-readable outcomes
are also in [`run-status.json`](../results/v0.1.0/run-status.json).

## Real CUDA model smoke tests

| Model | Pinned checkpoint | Languages and embedding dimension | Peak allocated / reserved VRAM |
| --- | --- | --- | --- |
| `openai_clip_b32` | OpenCLIP `ViT-B-32`, tag `openai`; checkpoint SHA-256 recorded in the status JSON | English, 512 | 0.58 / 0.64 GiB |
| `openclip_b32_laion2b` | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`, revision `1a25a446712ba5ee05982a381eed697ef9b435cf` | English, 512 | 0.58 / 0.72 GiB |
| `openclip_l14_laion2b` | `laion/CLIP-ViT-L-14-laion2B-s32B-b82K`, revision `1627032197142fbe2a7cfec626f4ced3ae60d07a` | English, 768 | 1.62 / 1.67 GiB |
| `embeddinggemma2` | `google/embeddinggemma-2`, revision `914f7f89142e33e77833254d9c9b90c3cef7303b` | English and Japanese, 768 | 2.92 / 2.99 GiB |
| `siglip2_base_224` | `google/siglip2-base-patch16-224`, revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` | English and Japanese, 768 | 1.43 / 3.55 GiB |
| `openclip_multilingual_b32` | `laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k`, revision `506d40eb551f4801a1c27fc20a31c7b8f590deda` | English and Japanese, 512 | 1.42 / 1.54 GiB |
| `rinna_ja_clip_b16` | `rinna/japanese-clip-vit-b-16`, revision `ce22bc39e6f5dcb449e082954211acfb00c2227b` | Japanese, 512 | 0.76 / 1.53 GiB |

The VRAM figures are peak values during smoke loading and inference, not full-benchmark
measurements. Checkpoint hashes, tokenizer revisions, exact byte counts, and numerical smoke
outputs are recorded in [`model-smoke-tests.json`](../results/v0.1.0/validation/model-smoke-tests.json)
and [`run-status.json`](../results/v0.1.0/run-status.json). The
Japanese CLIP extra now includes `protobuf`, required by the pinned Japanese SentencePiece
tokenizer with the locked Transformers version.

## Dataset preparation

| Dataset | Status | Verified facts or remaining work |
| --- | --- | --- |
| CIFAR-10 | completed | Official archive MD5 matched and SHA-256 recorded; full 10,000-image test split; dataset checksum `744e4b130af776f7a1c7a19427963e1f6d1c63984a1bb2bc2ceb4da2ba54c470`. |
| CIFAR-100 | completed | Official archive MD5 matched and SHA-256 recorded; full 10,000-image test split; dataset checksum `e5b40f8042fe89ad338c43b208464e0001bcb4ffba8a6827ed0da83562323a1f`. |
| XM3600 | both retrieval variants fully evaluated | All 3,600 official image IDs match captions and the attribution CSV. English has 7,200 captions; Japanese has 7,185. The pinned official loader yields 3,600 first-caption candidates in each language, with zero missing-image exclusions and matching image order. All 3,600 image entries list CC BY 2.0; annotations are CC BY 4.0. Data is local-only and is not redistributed. |

The machine-readable run matrix, data checksums, and access blockers are maintained in
[`results/v0.1.0/run-status.json`](../results/v0.1.0/run-status.json).

## Completed full evaluations

The raw archive contains 45 completed measurements using CUDA FP32, seed 0, the complete
registered split, and requested batch size 16. The primary report contains 38 eligible
measurements: 24 CIFAR runs, seven historical XM3600 `all_captions` runs, and seven active
clean `official_first_caption` replacements. P0 (18 CLIP Benchmark templates) and P1
(model-native prompts) remain separate. The earlier seven first-caption runs are preserved as
superseded evidence and excluded from primary aggregation. Actual image/text batch sizes and
OOM retry counts are recorded per run; replacement EmbeddingGemma image sub-batches were 11
images due to dynamic-resolution shape grouping, with text batch size 16. The other six
replacements used 16/16 image/text batches, and no replacement required an OOM retry.

The seven clean replacements used dataset checksum
`ed934a9782286a688ca79d4594aec679ca4accb8d738186f986c831711545fce`, 3,600 images, and 3,600
official first-caption candidates per language. Model checkpoint revisions and hashes,
tokenizer revisions, candidate ordering, exact metrics, throughput, latency, and peak VRAM
are recorded in the raw JSON and comparison audit. The frozen runtime was Python 3.12.10,
PyTorch 2.14.1+cu130, CUDA 13.0, driver 610.88, and NVIDIA GeForce RTX 4070 Ti SUPER (16 GiB).

The CIFAR matrix contains all six English-compatible models on both test sets under P0 and P1.
Each XM3600 variant contains English/Japanese runs for EmbeddingGemma, multilingual OpenCLIP,
and SigLIP 2, plus Japanese-only rinna CLIP. The first-caption runs use the same 3,600 image
IDs and exactly one caption per image in both languages. English/Japanese pairs are compared
only under matching run conditions; unequal candidate counts in the historical all-caption
variant are not used to calculate language deltas. Caption content and translation difficulty
remain uncontrolled, so score differences describe observed performance under the specified
conditions rather than a causal language-understanding effect.

The published primary results under [`results/v0.1.0`](../results/v0.1.0/) contain 38 rows;
all 45 raw records remain archived. The directory includes raw metric JSON, summary CSV,
generated Markdown report, metadata, run status, lifecycle evidence, and non-empty plots.
The earlier seven first-caption results are superseded and excluded from the primary
leaderboard. Numerical parity evidence is in
[`official-cifar10-parity.json`](../results/v0.1.0/validation/official-cifar10-parity.json)
and the clean replacement
[`official-xm3600-parity-clean.json`](../results/v0.1.0/validation/official-xm3600-parity-clean.json).
The historical
[`official-xm3600-parity.json`](../results/v0.1.0/validation/official-xm3600-parity.json)
and official CLI output are retained unchanged in
[`official-xm3600-openclip-ja.json`](../results/v0.1.0/validation/official-xm3600-openclip-ja.json).
The run-status record links the six prior dirty flags to generated raw results being untracked
during later runs. The exact uncommitted historical file set is unrecoverable.

## Access blockers

| Dataset | Status | Blocker |
| --- | --- | --- |
| XTD10 | blocked | The Adobe Research License requires explicit user acceptance; no terms were accepted. |
| Multilingual MS COCO Captions | blocked | It reuses XTD10 assets and has the same acceptance requirement. |
| Flickr30k | blocked | No authorized local Karpathy test copy or clearly permitted official image download has been established. |
| MS COCO Captions | blocked | The current preparer does not preserve or verify per-image rights for the selected test images. |
| ImageNet-1K | blocked | Registration/authorization is required; no authorized local validation archive was established. |
| Babel-ImageNet | blocked | It requires authorized ImageNet images and verified Japanese-label coverage. |

No dataset agreement has been accepted. See [model and dataset licenses](model-and-dataset-licenses.md)
for source details and the current access policy.
