# DINOv2 + dino.txt integration status

The `dinov2_dinotxt` adapter is wired to OpenVLCLIP v0.6.0 for zero-shot classification and
image-text retrieval. It uses OpenVLCLIP's official `dinov2-dinotxt` factory, its returned
preprocessing transform, and raw CLIP-tokenized text without task prefixes. The global image
and text embedding space is fixed at 2048 dimensions; this model does not expose the
EmbeddingGemma MRL dimension option.

## Current status: local benchmark use verified

The pinned Meta DINOv2 README directly lists the Vision Head and Text Encoder as pretrained
dino.txt heads, provides the official Hub loader, and states that DINOv2 code and model
weights are Apache-2.0. Separate terms are explicitly called out for Cell-DINO and XRay-DINO,
but not for dino.txt. The official BPE vocabulary license is MIT. Based on this source chain,
local scientific benchmarking with no weight redistribution is registered as `verified`.
The evidence, scope, and remaining limitation are documented in the
[DINO.txt licensing review](dinotxt-license-review.md).

The model-level gate has been updated only for this verified use. It still checks the
registered status before package import or weight loading, and it does not override dataset
access or attribution checks. The six original blocked run records remain historical records
of the first attempt; they are not measurements and are not relabeled as completed.

CUDA smoke validation and the six requested full-dataset measurements are pending execution
from a clean frozen source revision. No inference results are claimed by this status update.

The pinned cache artifacts can be checked without loading them or running inference:
[checkpoint integrity validation](../results/dinov2-dinotxt-v0.6.0/validation/checkpoint-integrity.json).
SHA-256 verification establishes file identity; the separate
[license review](dinotxt-license-review.md) records the basis for the permitted local use.

## Evaluation protocol when rights are resolved

- **Classification:** P0 uses the existing 18 CLIP Benchmark CIFAR templates. P1 uses the
  explicit single template `a photo of a {class}.`
- **Retrieval:** XM3600 uses P2 and the `official_first_caption` variant, with no model task
  prefix. The existing English and Japanese captions are evaluated separately.
- **Language:** Japanese is marked experimental; listing Japanese in the tokenizer/model
  interface is not evidence of Japanese accuracy.
- **Precision and hardware:** the requested campaign is CUDA FP32 on the available RTX 4070 Ti
  SUPER. No model inference has been performed for this integration.
- **Provenance:** the three weight files and BPE vocabulary are pinned separately by filename,
  byte size, and SHA-256. Successful future runs will record the actual component hashes.

Run the real-model smoke test and all six full-dataset conditions from a clean committed
source revision. The existing CLIP and EmbeddingGemma result files are not modified by this
integration.

### Upstream references

- [OpenVLCLIP v0.6.0](https://github.com/kotadymo/openvlclip/tree/v0.6.0)
- [DINOv2 source revision](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8)
- [DINOv2 license](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/LICENSE)
