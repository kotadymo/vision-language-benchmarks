# DINO.txt checkpoint license review

**Outcome: `verified_for_benchmark_use`**

This review covers local, non-redistributive inference with the official DINOv2 ViT-L/14
reg4 backbone, dino.txt Vision Head, dino.txt Text Encoder, and BPE vocabulary for a
scientific benchmark. It includes publication of aggregate evaluation metrics, but not
redistribution of checkpoint files or commercial deployment. This is an evidence-based
engineering determination, not formal legal advice.

## Determination

At the pinned Meta DINOv2 revision
[`7764ea0f912e53c92e82eb78a2a1631e92725fc8`](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8),
the official README places the exact dino.txt Vision Head and Text Encoder files in its
“Pretrained heads - Zero-shot tasks with dino.txt” section, and gives an official
PyTorch Hub loader for the full model. The same README states: “DINOv2 code and model
weights are released under the Apache License 2.0.” It does not identify a separate
dino.txt checkpoint license or carve the dino.txt weights out of that statement.

This is affirmative evidence connecting the two listed checkpoint files to the
publisher's Apache-2.0 model-weight statement; the conclusion does not rely merely on
the absence of a prohibition. The same README gives separate license notices for
XRay-DINO and Cell-DINO, while no corresponding exception is stated for dino.txt.
Accordingly, the published terms support local use of both dino.txt weight files for
the benchmark described above, subject to Apache-2.0 terms and applicable law.

The checkpoint files will not be redistributed. Dataset terms remain independent:
the CIFAR archives and XM3600 captions/images continue to be handled under their own
documented access, citation, and attribution safeguards. The BPE vocabulary is
separately MIT-licensed.

## Source evidence

All Meta repository sources below are pinned to
[`7764ea0f912e53c92e82eb78a2a1631e92725fc8`](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8).

| Source | Relevant published language or behavior |
| --- | --- |
| [DINOv2 README, dino.txt pretrained-head section](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/README.md#L395-L425) | Lists the exact [Vision Head](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_vision_head.pth) and [Text Encoder](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_text_encoder.pth) URLs as pretrained dino.txt heads, and provides the full-model Hub entry. |
| [DINOv2 README, license section](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/README.md#L658-L660) | States that “DINOv2 code and model weights are released under the Apache License 2.0” and links the repository [LICENSE](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/LICENSE). |
| [DINOv2 Hub loader](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/hub/dinotxt.py) | Defines `dinov2_vitl14_reg4_dinotxt_tet1280d20h24l` and loads both externally hosted state dictionaries from the same official URLs. Its source code carries the Apache-2.0 header. |
| [DINOv2 Hub exports](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/hubconf.py) | Exports the exact dino.txt Hub entry. |
| [DINOv2 model card](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/MODEL_CARD.md#model-description) | Lists Apache-2.0 for the DINOv2 ViT-S/B/L/g backbone family. This corroborates the backbone license but is not used alone to establish the dino.txt head or Text Encoder license. |
| [DINOv2 BPE vocabulary license](https://dl.fbaipublicfiles.com/dinov2/thirdparty/LICENSE) | The official README links this license with the dino.txt vocabulary; it is the MIT License, copyright OpenAI 2021. |
| [DINOv2 Meets Text paper](https://arxiv.org/abs/2412.16334) | Describes the dino.txt work. The paper's arXiv distribution terms apply to the paper and are not treated as a license for model checkpoints. No separate checkpoint license was identified in the paper. |
| [OpenVLCLIP v0.6.0 dino.txt documentation](https://github.com/kotadymo/openvlclip/blob/v0.6.0/docs/dinotxt-v0.6.0.md#license-and-dependency-findings) | A downstream integration note that conservatively requests clarification because it did not identify a file-specific license. It is not a Meta grant and is not treated as independent evidence against the pinned Meta README's explicit model-weight statement. |

The pinned README also gives explicit, separate notices for
[XRay-DINO](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/README.md#L159-L160)
and
[Cell-DINO](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/README.md#L754-L757).
Those notices identify their own FAIR Noncommercial Research and CC BY-NC terms.
They concern those named model families; they do not identify dino.txt as an
exception. This distinction is relevant because it shows that the repository
explicitly calls out model-specific licensing departures where applicable.

## Component-by-component assessment

| Component or activity | License / terms found | Assessment for this benchmark |
| --- | --- | --- |
| DINOv2 source code | Apache-2.0 repository `LICENSE`; Apache header on the Hub loader | Covered by Apache-2.0, subject to its terms. |
| DINOv2 ViT-L/14 reg4 backbone weights | README model-weight statement and model-card license | Apache-2.0. |
| dino.txt Vision Head weights | Exact file is listed as a dino.txt pretrained head in the same README; the general statement licenses DINOv2 model weights under Apache-2.0; no dino.txt exception is stated | Covered by the published Apache-2.0 model-weight statement for the specified local benchmark use. |
| dino.txt Text Encoder weights | Exact file is listed alongside the Vision Head in the same dino.txt section; the general statement licenses DINOv2 model weights under Apache-2.0; no dino.txt exception is stated | Covered by the published Apache-2.0 model-weight statement for the specified local benchmark use. |
| BPE vocabulary | Officially linked separate MIT license | MIT; retain required notice if distributing the vocabulary. It will not be redistributed here. |
| OpenVLCLIP software | Its own v0.6.0 package/repository terms | Separate from the Meta checkpoint grant; the benchmark pins `openvlclip==0.6.0`. |
| Download and local inference | Official public download URLs and official Hub loader, considered together with the Apache-2.0 model-weight statement | Downloading and locally loading the listed checkpoints for this benchmark is supported. Public availability alone would not establish a license. |
| Research benchmark use | Apache-2.0 provides broad use rights and contains no noncommercial-only restriction for these identified DINOv2 model weights | The requested noncommercial scientific inference and evaluation are supported. |
| Commercial use | Apache-2.0 is not itself a research-only license | Commercial deployment is outside this determination; review all other model, dataset, patent, privacy, and applicable-law issues separately. |
| Weight redistribution | Apache-2.0 includes reproduction and distribution rights subject to its conditions | No weights will be redistributed. This review does not publish or package any checkpoint files. |
| Publication of derived numerical results | No dino.txt-specific restriction on outputs or benchmark metrics was found in the applicable Meta terms | Publishing aggregate accuracy, recall, throughput, and memory measurements is supported for this benchmark; cite the model and paper as scientific practice. |

## Remaining scope limitation

The README's Apache-2.0 sentence is repository-level rather than a separate
per-filename notice, and the two weight files are hosted outside the Git repository.
The same pinned official README, however, expressly classifies both URLs as dino.txt
pretrained heads and links the official loader; the broad model-weight statement
follows in the same document. Unlike the named Cell-DINO and XRay-DINO exceptions,
there is no stated dino.txt-specific exception or additional agreement. On that
concrete evidence, the remaining wording limitation is not considered material to
the specified local, non-redistributive benchmark.

This determination is limited to model access for the described benchmark. It does
not grant dataset rights, change checkpoint integrity requirements, or authorize
redistribution of weights. The benchmark continues to pin and verify all four
component hashes, preserve dataset access checks, and record the license basis in
its provenance.
