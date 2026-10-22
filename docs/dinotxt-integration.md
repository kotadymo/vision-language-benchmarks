# DINOv2 + dino.txt integration status

The `dinov2_dinotxt` adapter is wired to OpenVLCLIP v0.6.0 for zero-shot classification and
image-text retrieval. It uses OpenVLCLIP's official `dinov2-dinotxt` factory, its returned
preprocessing transform, and raw CLIP-tokenized text without task prefixes. The global image
and text embedding space is fixed at 2048 dimensions; this model does not expose the
EmbeddingGemma MRL dimension option.

## Current status: model use blocked

The reviewed OpenVLCLIP and DINOv2 source materials identify Apache-2.0 for the library/code
and DINOv2 backbone, while the bundled CLIP BPE vocabulary is MIT-licensed. The separately
distributed dino.txt Vision Head and Text Encoder weights do not have a license stated in the
reviewed upstream materials. Those component rights cannot be inferred from the package or
backbone license. OpenVLCLIP's pinned
[dino.txt v0.6.0 reference](https://github.com/kotadymo/openvlclip/blob/v0.6.0/docs/dinotxt-v0.6.0.md#license-and-dependency-findings)
also records that no dino.txt-specific checkpoint license is linked and requests explicit
licensing confirmation before release.

Accordingly, the model is registered as `blocked`. The benchmark runner stops before dataset
preparation, OpenVLCLIP import, checkpoint loading, and inference. CLI `--yes` and
`--confirm-data-terms` do not override this model-level block. The requested CIFAR-10/100 P0/P1
and XM3600 English/Japanese `official_first_caption` evaluations therefore have no scores or
charts. Their blocked status is recorded in the
[run report](../results/dinov2-dinotxt-v0.6.0/report.md) and
[machine-readable status](../results/dinov2-dinotxt-v0.6.0/run-status.json).

The pinned cache artifacts can be checked without loading them or running inference:
[checkpoint integrity validation](../results/dinov2-dinotxt-v0.6.0/validation/checkpoint-integrity.json).
That SHA-256 check establishes file identity only; it does not establish permission to use the
two unlicensed dino.txt components.

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

Before enabling inference, obtain and document clear license terms or permission for both
external dino.txt weight files, update the model access state, and then run the real-model
smoke test and all six full-dataset conditions from a clean committed source revision. The
existing CLIP and EmbeddingGemma result files are not modified by this integration.

### Upstream references

- [OpenVLCLIP v0.6.0](https://github.com/kotadymo/openvlclip/tree/v0.6.0)
- [OpenVLCLIP dino.txt license findings](https://github.com/kotadymo/openvlclip/blob/v0.6.0/docs/dinotxt-v0.6.0.md#license-and-dependency-findings)
- [DINOv2 source revision](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8)
- [DINOv2 license](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/LICENSE)
