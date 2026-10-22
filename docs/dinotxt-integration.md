# DINOv2 + dino.txt integration status

The `dinov2_dinotxt` adapter uses OpenVLCLIP v0.6.0's official `dinov2-dinotxt` factory,
preprocessing transform, and tokenizer. Image and text outputs share a fixed 2048D global
embedding space; this model does not support EmbeddingGemma-style MRL truncation.

## Status: campaign completed

The pinned Meta DINOv2 README lists the dino.txt Vision Head and Text Encoder checkpoints as
pretrained heads, provides the official Hub loader, and states that DINOv2 code and model
weights are Apache-2.0. The official BPE vocabulary is MIT-licensed. Based on this evidence,
local, non-redistributive scientific benchmark use is registered as `verified`. The detailed
scope, reasoning, and remaining wording limitation are documented in the
[DINO.txt license review](dinotxt-license-review.md). This is an engineering determination,
not formal legal advice; it does not establish dataset rights or a commercial-use determination.

The model-level access gate remains active and checks the registered state before importing
OpenVLCLIP or loading weights. Dataset access and attribution controls remain independent.
The six original blocked run records are retained unchanged as historical attempts; they do
not contribute metrics and have not been relabeled as completed.

## Completed benchmark campaign

Six full-dataset CUDA FP32 evaluations completed from frozen source commit
`28519573b78f0dcb7ab6515ead02bc6b403a92bd`, all with `working_tree_dirty=false`. They use
`openvlclip==0.6.0`, seed 0, requested batch size 16, and the official checkpoint components.
CIFAR-10 and CIFAR-100 each use all 10,000 official test images. XM3600 uses 3,600 images and
3,600 first-caption candidates per language.

| Condition | Top-1 | Top-5 |
|---|---:|---:|
| CIFAR-10 P0 English | 98.24% (9,824/10,000) | 99.95% (9,995/10,000) |
| CIFAR-10 P1 English | 98.12% (9,812/10,000) | 99.94% (9,994/10,000) |
| CIFAR-100 P0 English | 89.71% (8,971/10,000) | 98.17% (9,817/10,000) |
| CIFAR-100 P1 English | 89.14% (8,914/10,000) | 98.13% (9,813/10,000) |

XM3600 English T→I Recall@1/5/10 is 51.50% / 77.31% / 84.64%; I→T is 53.03% / 77.33% /
85.44%. Japanese T→I is 2.03% / 5.14% / 7.75%; I→T is 3.17% / 8.00% / 11.97%. Japanese
retrieval is exploratory and uses the original Japanese captions without translation; it is
not evidence of Japanese semantic capability.

See the [campaign report](../results/dinov2-dinotxt-v0.6.0/report.md),
[summary CSV](../results/dinov2-dinotxt-v0.6.0/summary.csv),
[matched comparison CSV](../results/dinov2-dinotxt-v0.6.0/model-comparison.csv),
[raw results](../results/dinov2-dinotxt-v0.6.0/raw/),
[run-status manifest](../results/dinov2-dinotxt-v0.6.0/run-status.json), and
[publication lifecycle](../results/dinov2-dinotxt-v0.6.0/validation/result-lifecycle.json).
The initial blocked attempts remain separate from the six active measurements.

## CUDA and checkpoint validation

CUDA smoke validation confirmed the official 224×224 preprocessing, 77-token text context,
finite normalized 2048D image/text embeddings, finite cosine similarity, and English and
Japanese text encoding. Maximum absolute difference between official and wrapper image/text
outputs was zero. The full measurements record CUDA FP32 timing and memory.

The composite checkpoint fingerprint and all four component SHA-256 values are published in
the [checkpoint-integrity report](../results/dinov2-dinotxt-v0.6.0/validation/checkpoint-integrity.json)
and [smoke record](../results/dinov2-dinotxt-v0.6.0/validation/dinotxt-cuda-smoke.json).
Hash verification establishes file identity; the separate license review records the basis
for local use. No model weights or dataset files are redistributed.

## Evaluation protocol and comparison scope

- **P0:** CIFAR zero-shot classification with the pinned CLIP Benchmark class names and 18
  common templates per class.
- **P1:** a separate model-native protocol. DINO.txt uses the single explicit template
  `a photo of a {class}.`; EmbeddingGemma uses its task prefix, SigLIP 2 its model-card template,
  and other CLIP-family adapters their single photo template. P0 and P1 remain separate.
- **P2:** XM3600 `official_first_caption` retrieval, with one original caption per image and
  no model task prefix. English and Japanese results remain separate.
- Historical comparisons match dataset checksum, split, task, protocol, language, precision,
  sample/candidate counts, and retrieval variant. Historical package/source revisions and
  batch sizes may differ, so throughput differences are descriptive rather than controlled
  performance claims.

## Reproduction

The [report reproduction section](../results/dinov2-dinotxt-v0.6.0/report.md#reproduction)
separates `vlbench run` commands that perform inference from `vlbench report` commands that
only regenerate summaries and figures from existing raw JSON.

### Upstream references

- [OpenVLCLIP v0.6.0](https://github.com/kotadymo/openvlclip/tree/v0.6.0)
- [Pinned DINOv2 source revision](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8)
- [Official DINOv2 README and license statements](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/README.md)
- [Official dino.txt Hub loader](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/hub/dinotxt.py)
- [DINOv2 LICENSE](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/LICENSE)
