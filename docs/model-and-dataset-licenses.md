# Model and dataset licenses

This registry snapshot records source metadata and access safeguards; it is not legal advice.
Review the linked license and usage terms for the exact revision and intended use before
downloading or using any checkpoint or dataset. A CLI confirmation records a user's assertion
and does not grant rights.

## Models

| Registry key | Checkpoint and pinned revision | Recorded license | Notes |
| --- | --- | --- | --- |
| `embeddinggemma2` | [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2), revision `914f7f89142e33e77833254d9c9b90c3cef7303b` | Apache-2.0 | Official model card reports Apache-2.0. The pinned model revision is recorded in each run. |
| `dinov2_dinotxt` | OpenVLCLIP v0.6.0 official `dinov2-dinotxt` configuration; DINOv2 source revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8` | Apache-2.0 for DINOv2 code and model weights, including both dino.txt weight components; MIT for the BPE vocabulary | The pinned Meta README lists both dino.txt weights as pretrained DINOv2 heads and states that DINOv2 model weights are Apache-2.0. Local scientific benchmark use without weight redistribution is verified; see the [evidence-based license review](dinotxt-license-review.md). |
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

For the dino.txt license determination and the per-component artifact pins, see the
[DINOv2+dino.txt license review](dinotxt-license-review.md) and
[integration record](dinotxt-integration.md). Matching locally cached checkpoint hashes is
an integrity check, not a license grant.

## Datasets and access policy

| Dataset | Registry state | Current safeguard |
| --- | --- | --- |
| CIFAR-10 / CIFAR-100 | `public_download_citation_required` | The [official University of Toronto page](https://www.cs.toronto.edu/~kriz/cifar.html) publishes the Python archives, defines the test splits, and requests citation of the dataset technical report. It displays no separate acceptance agreement. This benchmark downloads only the official archives for non-redistributive evaluation and cites the report; this is not a claim of a formal open-source license. |
| Crossmodal-3600 (XM3600) | `verified_attribution_required` | The [official project page](https://google.github.io/crossmodal-3600/) states that Google LLC licenses annotations under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Its [image-attribution CSV](https://google.github.io/crossmodal-3600/web-data/image_attributions.csv) was checked on 2026-10-19: all 3,600 unique image IDs match the caption set, all entries list [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/), and author, title, and source fields are present. Preparation repeats these checks before image inference. Images are not redistributed. |
| Flickr30k | `blocked` | The official project requires an access request and states that images are from Flickr and subject to Flickr Terms of Use. No approved access grant has been verified for the local archives. |
| MS COCO Captions | `blocked` | The official COCO terms state that downloading signifies agreement; no acceptance was provided. The current preparer also does not preserve/verify per-image image rights for the Karpathy test split. |
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

### Published XM3600 evaluations

The 38-run v0.1.0 publication reuses the existing local XM3600 copy after validating the
official image IDs, captions, and per-image attribution metadata. It includes the seven
historical `all_captions` runs and seven new `official_first_caption` runs; these variants are
kept separate in the results. The new runs use one attributed image-caption pair per image.
No new XM3600 files were downloaded for these evaluations, no dataset terms were accepted,
and no images or annotations are redistributed by this repository. The
[run-status manifest](../results/v0.1.0/run-status.json) and per-run raw JSON retain the
source checksum and attribution provenance.

### Flickr30k and COCO access blockers

The Flickr30k block is based on unresolved acquisition and rights evidence, not on third-party
image copyright alone. The linked [UIUC project page](https://shannon.cs.illinois.edu/DenotationGraph/)
is the official entry point available in this review; its current response exposes Denotation
Graph files but no unambiguous current Flickr30k image-access form or reuse terms. Prior access
descriptions refer to a request process, but no approved request is on record. Local archive
files are not evidence of authorization. The caption documentation also describes an
Attribution-ShareAlike license while linking to [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/);
this discrepancy must be resolved before relying on the captions' license. To unblock
evaluation, obtain current written access/usage instructions from the dataset maintainers and
review the applicable Flickr and caption terms.

The [official COCO terms](https://cocodataset.org/#termsofuse) say that downloading the
dataset signifies agreement to those terms. No such acceptance was provided, so neither the
official images nor the local Karpathy split were evaluated. The Karpathy split contains
5,000 test images and 25,010 caption entries in the available annotation archive; the
registry records that count but remains blocked. Unblocking evaluation requires the user to
review and accept the official terms and the preparer to preserve and validate the
image-specific rights for every selected test image. An existing local copy does not bypass
either requirement.

### Dataset feasibility audit

The following feasibility review uses the linked project sources and the pinned CLIP
Benchmark dataset loaders. No new dataset files were downloaded, and no access or legal terms
were accepted. None of the six datasets below currently has an authorized local copy available
for this benchmark. A public URL or repository alone is not treated as permission to use the
underlying data.

#### Flickr30k Karpathy test

- **Source and acquisition:** Start with the
  [UIUC Flickr30k/Denotation Graph project page](https://shannon.cs.illinois.edu/DenotationGraph/)
  and obtain the original image/caption materials through an authorized project-maintainer
  route. The page available during this review exposes Denotation Graph files but does not
  provide an unambiguous current Flickr30k image download and reuse process. Do not substitute
  a mirror or assume a public split JSON authorizes image use.
- **Terms and access:** Images remain subject to the original Flickr uploaders' rights and
  [Flickr Terms of Use](https://www.flickr.com/help/terms); the Flickr30k caption documentation
  has a stated-license/link discrepancy (Attribution-ShareAlike description versus a CC BY 3.0
  link). The current official page does not establish whether a new registration or specific
  agreement is required. That access process and the applicable caption license remain
  unresolved, so permission has not been established.
- **Split and annotations:** The Karpathy community split is not the dataset creators'
  original split. Its retrieval test is 1,000 images with five captions per image (5,000
  caption candidates). It requires the split JSON/IDs, caption text, and original image files;
  validate image IDs, order, and exactly five references per selected image.
- **Local status and implementation:** No authorized local Karpathy test copy is verified.
  `vlbench` still needs a loader that maps the split annotations to local images, validates
  per-image correspondence, and records image-level rights/attribution. The registry and CLI
  intentionally keep this dataset blocked.
- **Next action:** The user must obtain current written access/usage instructions from the
  dataset maintainers, resolve the caption-license discrepancy, and provide a lawful local
  copy with its source terms and split annotation. Only then can the adapter and full test
  evaluation proceed.

#### MS COCO Captions Karpathy 5K test

- **Source and acquisition:** The
  [official COCO download page](https://cocodataset.org/#download) provides the image and
  annotation archives; the [official terms](https://cocodataset.org/#termsofuse) govern
  downloading and use. The Karpathy split JSON is a separate community split and must be
  obtained from an authorized source; it is not the official COCO evaluation split.
- **Terms and access:** COCO states that downloading signifies agreement to its terms. The
  annotations (including captions) are CC BY 4.0; image rights remain with their original
  owners and must be checked per image. The terms page does not establish a separate account
  requirement, but downloading/use entails the agreement. No one has accepted it for this
  task.
- **Split and annotations:** The Karpathy test selection is 5,000 images, conventionally five
  captions per image (25,000 expected references). The current repository registry records
  25,010 caption rows for its candidate annotation archive; that discrepancy has not been
  reconciled against an authorized source and must be resolved before evaluation. Required
  files are the authorized image archive, COCO caption annotations, Karpathy split JSON, and
  image IDs/rights metadata sufficient to validate each selected file.
- **Local status and implementation:** No authorized local COCO image/annotation copy is
  established. The loader must map Karpathy IDs to COCO image filenames, reject missing or
  duplicated records, validate caption-to-image positives and per-image rights, and record
  the exact caption count. The registry and CLI keep this dataset blocked.
- **Next action:** The user must review and personally accept the official COCO terms, then
  provide the authorized files and split source. Resolve the 25,000-versus-25,010 count and
  preserve/check original image rights before implementing or running the adapter.

#### ImageNet-1K validation

- **Source and acquisition:** The
  [official ImageNet download page](https://image-net.org/download.php) says the
  ILSVRC 2012–2017 subset contains 1,000 classes, 1,281,167 training images, and 50,000
  validation images. The validation image archive and the
  [ILSVRC 2012 challenge page](https://image-net.org/challenges/LSVRC/2012/) are the
  authoritative acquisition path; the official page also points to the Kaggle challenge.
- **Terms and access:** Access requires logging in or requesting permission and agreeing to
  ImageNet's terms. The terms limit use to non-commercial research and education and restrict
  redistribution. Registration/authorization and explicit acceptance are required; none was
  performed for this task.
- **Split and annotations:** The target split is all 50,000 validation images across 1,000
  classes. Evaluation needs the validation JPEG archive plus the ILSVRC devkit/class synset
  mapping and the validation ground-truth labels.
- **Local status and implementation:** No authorized local validation archive or devkit is
  verified. A loader must read the local validation tree, join labels through the official
  synset mapping, preserve deterministic order, and expose the complete split to the
  classification engine.
- **Next action:** The user must create/request an authorized account, personally accept the
  applicable ImageNet or Kaggle challenge terms, and provide the validation archive and
  matching label/devkit files. Do not use an unofficial mirror or a presumed pre-existing
  local copy.

#### XTD10

- **Source and acquisition:** The
  [Adobe Research XTD10 repository](https://github.com/adobe-research/Cross-lingual-Test-Dataset-XTD10)
  provides its index and caption files; its README describes a 1K MS COCO 2014 caption test
  set in seven additional languages. The pinned CLIP Benchmark loader also supports English,
  German, French, Italian, Japanese (`jp`), and the XTD10 languages, using MIC and STAIR
  caption files for some language codes. It selects 1,000 image IDs and one caption per image.
  The associated images are COCO images and have a separate COCO access/terms requirement.
- **Terms and access:** The XTD10
  [Adobe Research License](https://github.com/adobe-research/Cross-lingual-Test-Dataset-XTD10/blob/main/LICENSE.md)
  grants non-commercial research use and states that it is effective upon acceptance.
  Viewing the public repository does not require a GitHub account, but using the licensed
  annotations requires accepting the license. No acceptance was provided. COCO image terms
  also remain applicable.
- **Local status and implementation:** No authorized local annotation or image copy is
  verified. `vlbench` needs a loader for the official image index, language-specific caption
  file, `jp` source language identifier, and authorized COCO image paths; validate the same
  1,000 ordered image IDs and exactly one caption per image.
- **Next action:** The user must review and accept the Adobe Research License and the
  applicable COCO terms, then provide the authorized local image and annotation files. Do not
  download/use the data before those decisions.

#### Multilingual MS COCO Captions (CLIP Benchmark wrapper)

- **Source and acquisition:** The pinned CLIP Benchmark
  [`multilingual_mscoco.py`](https://github.com/LAION-AI/CLIP_benchmark/blob/ee14a9a8a9bcde09d51f4a4b1dff5a75182a3596/clip_benchmark/datasets/multilingual_mscoco.py)
  reads the same XTD10 repository image index and image archive and the language-specific
  XTD10/MIC/STAIR caption files. It uses 1,000 test image IDs and one caption per selected
  image. This is an alias over overlapping XTD10 assets, not an independent 1K evaluation;
  results and counts must not be added as if they were separate datasets.
- **Terms and access:** The Adobe Research License applies to the caption materials; COCO
  terms apply to the image material. The same explicit acceptance requirements and lack of
  user acceptance as XTD10 apply.
- **Local status and implementation:** No authorized local files are verified. The same
  language-aware loader can serve both dataset identifiers while preserving the requested
  dataset key, original `jp` code, caption source, and duplicate-of relationship.
- **Next action:** Unblock through the same user decisions and authorized files as XTD10.
  First document the overlap and use one representative result when reporting shared image
  IDs and annotations.

#### Babel-ImageNet Japanese classification

- **Source and acquisition:** The
  [Babel-ImageNet repository](https://github.com/gregor-ge/Babel-ImageNet) publishes
  multilingual label mappings for the 1,000 ImageNet classes (including Japanese) but no
  images. Its README says the released labels are derived from BabelNet 5.2 and are made
  available under the
  [BabelNet Non-Commercial License](https://babelnet.org/full-license).
- **Terms and access:** BabelNet's
  [registration page](https://babelnet.org/register) requires registration and license
  acceptance for access to its licensed data. The label set is for non-commercial use; the
  ImageNet images separately require ImageNet authorization and terms acceptance. Neither
  license has been accepted for this task.
- **Split and annotations:** The image source would be the 50,000-image ImageNet-1K
  validation split, filtered to classes with Japanese labels. Label coverage is partial; the
  retained Japanese class and image counts have not been verified because the licensed label
  data and authorized images are unavailable. Required annotations are the BabelNet-derived
  Japanese labels keyed to ImageNet class/synset IDs and the ILSVRC validation ground-truth
  mapping. English comparisons must use the identical retained classes and image IDs.
- **Local status and implementation:** No authorized label file, ImageNet archive, or devkit
  is verified. `vlbench` needs a class-filtering loader that validates Japanese coverage and
  creates matched English/Japanese class lists on exactly the same validation examples.
- **Next action:** The user must register with BabelNet and personally accept the
  non-commercial license, obtain authorized ImageNet access and accept its terms, then
  provide the label file and validation images/devkit. Compute Japanese label coverage from
  those authorized files before choosing or reporting an image count.
