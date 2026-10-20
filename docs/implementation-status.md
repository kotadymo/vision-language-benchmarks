# Implementation and Evaluation Status

Last updated: 2026-10-20
Branch: `feature/benchmark-framework-v0.1.0`
Base branch head before the XM3600 protocol work: `5b512ecb1d0177d88817a4e07d54f86810b7f86f`
Measurement source commits: `2993187c2b960e6e6690c97205b22838d18a36a3`,
`60a61259c95e7c387e896b23084e3c3d0fc375f5`,
`d72e4e462d137e9b77581a507e42897ff427eaf1`,
`27f2ac614a1e6043fec8f16beceda202bb9272ef`, and
`cec36f68a1523062f44216d4568fe2612208e913` (recorded per run).

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
| Prior `official_first_caption` CUDA evaluations | completed historical set; clean-source replacement pending | Seven FP32, seed-0, requested-batch-16 runs; prior audit found one clean run and six with `working_tree_dirty: true` |
| Strict clean-source execution | implemented and offline-tested | `doctor --reproducibility`, strict CUDA runs, frozen-SHA and ignored-staging validation, post-run checks, lifecycle-aware publication, and superseded-result exclusion |
| Clean `official_first_caption` CUDA reruns | pending | Seven complete FP32 runs must use one clean committed source SHA; inference has not started |
| Paired English/Japanese XM3600 comparison | previously completed for three multilingual models; replacement comparison pending | Prior results used matched image IDs, candidate counts, checkpoints, and P2 protocol; replacement differences will be computed from actual rerun records |
| End-to-end official XM3600 evaluation | prior historical OpenCLIP ViT-B/32 Japanese comparison completed; replacement parity pending | Prior pinned CLI and `vlbench` results used the same 3,600 candidates; clean-run parity must be revalidated |
| Official CLIP Benchmark parity | CIFAR-10 P0 and one historical XM3600 condition completed | Reference commit `ee14a9a8a9bcde09d51f4a4b1dff5a75182a3596`; CIFAR-10 P0 is exact at `1e-12`; historical XM3600 maximum absolute metric difference is `2.9404958135081927e-8` at `5e-8` tolerance |
| Historical raw-result audit | completed | All 31 original files are schema-valid and byte-identical to baseline commit `5b512ecb1d0177d88817a4e07d54f86810b7f86f`; provenance, metrics, batch sizes, throughput, prompts, and sensitive-payload checks are recorded in the audit JSON |

The prior seven-run audit found one clean first-caption run and six runs with
`working_tree_dirty: true`. The recorded run status attributes the observed dirty state to
generated raw result files being untracked during later runs. The exact historical uncommitted
file set cannot be reconstructed; no contents are inferred, and no historical raw metadata has
been changed. The new strict workflow writes intermediate results to ignored staging and
rejects dirty source state rather than bypassing it.

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

The existing 38-measurement baseline uses CUDA FP32, seed 0, the complete registered split, and
requested batch size 16. P0 (18 CLIP Benchmark templates) and P1 (model-native prompts) remain
separate. Actual image/text batch sizes and OOM retry counts are recorded per run; the
EmbeddingGemma XM3600 image sub-batches were 11 images due to dynamic-resolution shape
grouping, with text batch size 16. No published run required an OOM retry. The 24 CIFAR runs
and seven historical XM3600 `all_captions` runs are unchanged. The seven earlier
`official_first_caption` runs use 3,600 candidates per language and remain separate from the
all-caption scores; the prior audit found one clean run and six dirty-worktree records. Their
replacement evaluations and numerical comparison are pending.

The CIFAR matrix contains all six English-compatible models on both test sets under P0 and P1.
Each XM3600 variant contains English/Japanese runs for EmbeddingGemma, multilingual OpenCLIP,
and SigLIP 2, plus Japanese-only rinna CLIP. The first-caption runs use the same 3,600 image
IDs and exactly one caption per image in both languages. English/Japanese pairs are compared
only under matching run conditions; unequal candidate counts in the historical all-caption
variant are not used to calculate language deltas. Caption content and translation difficulty
remain uncontrolled, so score differences describe observed performance under the specified
conditions rather than a causal language-understanding effect.

The existing results are published under [`results/v0.1.0`](../results/v0.1.0/): raw metric
JSON, summary CSV, generated Markdown report, metadata, and non-empty plots. Numerical parity
evidence is in
[`official-cifar10-parity.json`](../results/v0.1.0/validation/official-cifar10-parity.json)
and the historical
[`official-xm3600-parity.json`](../results/v0.1.0/validation/official-xm3600-parity.json).
The official CLI output is retained in
[`official-xm3600-openclip-ja.json`](../results/v0.1.0/validation/official-xm3600-openclip-ja.json).
The run-status record links the six dirty flags to generated raw results being untracked during
later runs. The previous raw JSON and parity records remain untouched; the source and dirty
metadata are authoritative, while the exact uncommitted historical file set is unrecoverable.

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
