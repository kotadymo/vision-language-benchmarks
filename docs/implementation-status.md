# Implementation and Evaluation Status

Last updated: 2026-10-19
Current branch source commit: `60a61259c95e7c387e896b23084e3c3d0fc375f5`
Measurement source commits: `2993187c2b960e6e6690c97205b22838d18a36a3`,
`60a61259c95e7c387e896b23084e3c3d0fc375f5`, and
`d72e4e462d137e9b77581a507e42897ff427eaf1` (recorded per run).

## Verification milestones

| Milestone | Status | Evidence |
| --- | --- | --- |
| Offline unit suite | completed | `uv run --locked pytest -q`: 40 passed, two opt-in GPU modules skipped by default |
| Dependency lock, lint, and patch checks | completed | `uv lock --check`, `uv run --locked ruff check .`, and `git diff --check` passed |
| Pretrained checkpoints loaded on CUDA | completed | All seven registered adapters passed |
| Real image and text inference on CUDA | completed | All supported smoke-test languages passed; embeddings were finite and dimension-compatible |
| Full CIFAR evaluations | completed | 24 runs: CIFAR-10 and CIFAR-100 full 10,000-image test splits, six English-compatible models, P0 and P1 reported separately |
| Full XM3600 evaluations | completed | Seven P2 runs: EmbeddingGemma, multilingual OpenCLIP, and SigLIP in English/Japanese; rinna Japanese-only; 3,600 images per run |
| Official CLIP Benchmark parity | completed | Reference commit `ee14a9a8a9bcde09d51f4a4b1dff5a75182a3596`; CIFAR-10 P0 metrics match exactly at tolerance `1e-12` |

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
| XM3600 | completed | All 3,600 official image IDs match captions and the attribution CSV. English has 7,200 captions; Japanese has 7,185. All 3,600 image entries list CC BY 2.0; annotations are CC BY 4.0. Data is local-only and is not redistributed. |

The machine-readable run matrix, data checksums, and access blockers are maintained in
[`results/v0.1.0/run-status.json`](../results/v0.1.0/run-status.json).

## Completed full evaluations

All measurements use CUDA FP32, seed 0, the complete registered split, and requested batch size
16. P0 (18 CLIP Benchmark templates) and P1 (model-native prompts) remain separate. Actual
image/text batch sizes and OOM retry counts are recorded per run; the EmbeddingGemma XM3600
image sub-batches were 11 images due to dynamic-resolution shape grouping, with text batch size
16. No published run required an OOM retry.

The CIFAR matrix contains all six English-compatible models on both test sets under P0 and P1.
The XM3600 matrix contains matched English/Japanese runs for EmbeddingGemma, multilingual
OpenCLIP, and SigLIP 2, plus Japanese-only rinna CLIP. English/Japanese pairs use the same
3,600 images and image-ID order. Within each language, caption candidate-ID order and
positive-pair counts match across models; comparisons are limited to condition-matched pairs.

The results are published under [`results/v0.1.0`](../results/v0.1.0/): raw metric JSON,
summary CSV, generated Markdown report, metadata, and non-empty plots. Numerical parity evidence
is in [`official-cifar10-parity.json`](../results/v0.1.0/validation/official-cifar10-parity.json).
The README summarizes measured values and links the artifacts.

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
