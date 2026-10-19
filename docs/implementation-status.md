# Implementation and Evaluation Status

Last updated: 2026-10-19 15:09 UTC
Source commit for the current implementation: `4a15c9dea3e841606c2009db990f0d738934def8`

## Verification milestones

| Milestone | Status | Evidence |
| --- | --- | --- |
| Offline unit suite | completed | 35 passed; two opt-in GPU modules skipped by default |
| Pretrained checkpoints loaded on CUDA | completed | All seven registered adapters passed |
| Real image and text inference on CUDA | completed | All supported smoke-test languages passed; embeddings were finite and dimension-compatible |
| Full-dataset evaluations | pending | No CIFAR or XM3600 benchmark metrics have been produced yet |
| Official CLIP Benchmark parity | pending | Reference commit `ee14a9a8a9bcde09d51f4a4b1dff5a75182a3596` is pinned; numerical comparison remains outstanding |

The smoke tests use one generated image and two text examples per supported language. They
validate the actual pretrained image and text encoders, feature shape and finiteness, normalized
cosine similarity, and CUDA placement. These are adapter checks, not benchmark measurements.
The detailed local execution logs are under the ignored `.cache/real-smoke/` directory.

## Real CUDA model smoke tests

| Model | Pinned checkpoint | Languages and embedding dimension | Peak allocated / reserved VRAM |
| --- | --- | --- | --- |
| `openai_clip_b32` | OpenCLIP `ViT-B-32`, tag `openai`; checkpoint SHA-256 recorded in the status JSON | English, 512 | 0.58 / 0.64 GiB |
| `openclip_b32_laion2b` | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`, revision `1a25a446712ba5ee05982a381eed697ef9b435cf` | English, 512 | 0.59 / 0.72 GiB |
| `openclip_l14_laion2b` | `laion/CLIP-ViT-L-14-laion2B-s32B-b82K`, revision `1627032197142fbe2a7cfec626f4ced3ae60d07a` | English, 768 | 1.62 / 1.67 GiB |
| `embeddinggemma2` | `google/embeddinggemma-2`, revision `914f7f89142e33e77833254d9c9b90c3cef7303b` | English and Japanese, 768 | 2.94 / 3.02 GiB |
| `siglip2_base_224` | `google/siglip2-base-patch16-224`, revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` | English and Japanese, 768 | 1.43 / 3.55 GiB |
| `openclip_multilingual_b32` | `laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k`, revision `506d40eb551f4801a1c27fc20a31c7b8f590deda` | English and Japanese, 512 | 1.42 / 1.54 GiB |
| `rinna_ja_clip_b16` | `rinna/japanese-clip-vit-b-16`, revision `ce22bc39e6f5dcb449e082954211acfb00c2227b` | Japanese, 512 | 0.77 / 1.53 GiB |

The VRAM figures are peak values during smoke loading and inference, not full-benchmark
measurements. Checkpoint hashes, tokenizer revisions, exact byte counts, and numerical smoke
outputs are recorded in [`run-status.json`](../results/v0.1.0/run-status.json). The
Japanese CLIP extra now includes `protobuf`, required by the pinned Japanese SentencePiece
tokenizer with the locked Transformers version.

## Dataset preparation

| Dataset | Status | Verified facts or remaining work |
| --- | --- | --- |
| CIFAR-10 | pending | Official Python archive download is in progress. Archive integrity and the 10,000-image test split are not yet verified. |
| CIFAR-100 | pending | Official Python archive has not yet been acquired; integrity and its 10,000-image test split remain to be verified. |
| XM3600 | completed | All 3,600 official image IDs match captions and the attribution CSV. English has 7,200 captions; Japanese has 7,185. All 3,600 image entries list CC BY 2.0; annotations are CC BY 4.0. Data is local-only and is not redistributed. |

The machine-readable run matrix, data checksums, and access blockers are maintained in
[`results/v0.1.0/run-status.json`](../results/v0.1.0/run-status.json).

## Planned full evaluations

The initial classification matrix is CIFAR-10 and CIFAR-100, FP32, English, P0 and P1, for
all six English-compatible models. P0 and P1 are kept as separate protocols. The initial
retrieval matrix is XM3600 P2 for EmbeddingGemma 2, multilingual OpenCLIP, and SigLIP 2 in
English and Japanese, plus rinna Japanese CLIP in Japanese. Each planned model/dataset/language/
protocol entry has an explicit status in the run matrix.

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
