# Model and dataset licenses

This registry snapshot records source metadata and access safeguards; it is not legal advice.
Review the linked license and usage terms for the exact revision and intended use before
downloading or using any checkpoint or dataset. A CLI confirmation records a user's assertion
and does not grant rights.

## Models

| Registry key | Checkpoint and pinned revision | Recorded license | Notes |
| --- | --- | --- | --- |
| `embeddinggemma2` | [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2), revision `914f7f89142e33e77833254d9c9b90c3cef7303b` | Apache-2.0 | Official model card reports Apache-2.0. The pinned model revision is recorded in each run. |
| `openai_clip_b32` | OpenCLIP `ViT-B-32`, tag `openai` ([upstream](https://github.com/openai/CLIP)) | MIT | Not revision-addressable as a Hugging Face snapshot; the actual checkpoint hash is recorded at runtime. |
| `openclip_b32_laion2b` | [`laion/CLIP-ViT-B-32-laion2B-s34B-b79K`](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K), revision `1a25a446712ba5ee05982a381eed697ef9b435cf` | MIT | Checkpoint and model configuration are pinned. |
| `openclip_l14_laion2b` | [`laion/CLIP-ViT-L-14-laion2B-s32B-b82K`](https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K), revision `1627032197142fbe2a7cfec626f4ced3ae60d07a` | MIT | Checkpoint and model configuration are pinned. |
| `openclip_multilingual_b32` | [`laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k`](https://huggingface.co/laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k), revision `506d40eb551f4801a1c27fc20a31c7b8f590deda` | MIT | Uses the pinned [XLM-R tokenizer](https://huggingface.co/FacebookAI/xlm-roberta-base), revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089` (MIT). |
| `siglip2_base_224` | [`google/siglip2-base-patch16-224`](https://huggingface.co/google/siglip2-base-patch16-224), revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` | Apache-2.0 | Uses the pinned Transformers processor and model files. |
| `rinna_ja_clip_b16` | [`rinna/japanese-clip-vit-b-16`](https://huggingface.co/rinna/japanese-clip-vit-b-16), revision `ce22bc39e6f5dcb449e082954211acfb00c2227b` | Apache-2.0 | Uses the separately pinned [Japanese RoBERTa tokenizer](https://huggingface.co/rinna/japanese-roberta-base), revision `f93982a6f5a8bb9fd483b9deedae693091114b33` (MIT). |

All listed model files are pinned where the hosting repository supports immutable revisions.
The OpenAI tag is fingerprinted after download because its upstream checkpoint is not pinned
to a Hugging Face commit. File sizes in the registry describe checkpoint files only;
configuration and tokenizer files may add to the download.

The OpenAI CLIP model card, which the LAION CLIP cards also reference, describes research
use and lists deployment as out of scope. Review the model cards' intended-use guidance in
addition to the license before using results outside this benchmark.

## Datasets and access policy

| Dataset | Registry state | Current safeguard |
| --- | --- | --- |
| CIFAR-10 / CIFAR-100 | `public_download_citation_required` | The [official University of Toronto page](https://www.cs.toronto.edu/~kriz/cifar.html) publishes the Python archives, defines the test splits, and requests citation of the dataset technical report. It displays no separate acceptance agreement. This benchmark downloads only the official archives for non-redistributive evaluation and cites the report; this is not a claim of a formal open-source license. |
| Crossmodal-3600 (XM3600) | `verified_attribution_required` | The [official project page](https://google.github.io/crossmodal-3600/) states that Google LLC licenses annotations under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Its [image-attribution CSV](https://google.github.io/crossmodal-3600/web-data/image_attributions.csv) was checked on 2026-10-19: all 3,600 unique image IDs match the caption set, all entries list [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/), and author, title, and source fields are present. Preparation repeats these checks before image inference. Images are not redistributed. |
| Flickr30k | `blocked` | No authorized local Karpathy test copy or permitted image download was established. |
| MS COCO Captions | `blocked` | The current preparer does not preserve/verify per-image third-party licenses for the selected test images. |
| ImageNet-1K | `blocked` | Requires authorized ImageNet access; the framework will not bypass registration or access controls. |
| XTD10 | `blocked` | The Adobe Research License requires the user's acceptance of its terms; no acceptance was provided. |
| Multilingual MS COCO Captions | `blocked` | Reuses XTD10 assets and has the same access constraint; it is not treated as an independent dataset. |
| Babel-ImageNet | `blocked` | Requires authorized ImageNet images and verified Japanese class coverage. |

The registry and CLI plan show the access state before an evaluation. `--confirm-data-terms`
does not override a `blocked` status and is not used for the reviewed CIFAR or XM3600 sources.
`--yes` authorizes the reviewed network plan; it does not accept a license or grant dataset
rights. CIFAR use is limited here to official-source research evaluation with citation and no
redistribution. XM3600 attribution metadata is validated against the downloaded captions
before images are acquired. Do not use either flag as a substitute for any acceptance that a
different source may require.

No dataset terms have been accepted. The authoritative access states are in
[`registry.py`](../src/vision_language_benchmarks/registry.py); check the original sources
again before any acquisition or publication. The benchmark code does not redistribute
dataset files.
