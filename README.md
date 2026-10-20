# Vision-Language Benchmarks

A reproducible framework for evaluating pretrained vision-language models on zero-shot
classification and bilingual image-text retrieval.

## Verified v0.1.0 results

The v0.1.0 publication contains 38 full-dataset CUDA FP32 measurements on an NVIDIA GeForce
RTX 4070 Ti SUPER: 24 CIFAR classification runs, the seven preserved historical XM3600
`all_captions` runs, and seven new XM3600 `official_first_caption` runs. The two retrieval
variants are reported separately. Seven pretrained model adapters also passed real image/text
CUDA smoke tests; these smoke tests are distinct from the full evaluations.

- [Full report](results/v0.1.0/report.md) · [CSV summary](results/v0.1.0/summary.csv) ·
  [metadata](results/v0.1.0/metadata.json)
- [Validated per-run JSON results](results/v0.1.0/raw/) ·
  [machine-readable run status](results/v0.1.0/run-status.json)
- [Historical raw-result audit and preservation hashes](results/v0.1.0/validation/historical-results-audit.json)
- [Real CUDA model smoke-test evidence](results/v0.1.0/validation/model-smoke-tests.json)
- Official CLIP Benchmark parity records:
  [CIFAR-10](results/v0.1.0/validation/official-cifar10-parity.json) ·
  [XM3600 end-to-end](results/v0.1.0/validation/official-xm3600-parity.json) ·
  [pinned reference run](results/v0.1.0/validation/official-xm3600-openclip-ja.json)
- Charts: [classification P0](results/v0.1.0/plots/classification_accuracy.png),
  [classification P1](results/v0.1.0/plots/classification_accuracy_p1.png),
  [official first-caption XM3600 retrieval](results/v0.1.0/plots/xm3600_official_retrieval.png),
  [historical all-caption XM3600 retrieval](results/v0.1.0/plots/xm3600_all_captions_retrieval.png),
  [matched-language comparison](results/v0.1.0/plots/xm3600_language_comparison_official_first_caption.png),
  [performance P0](results/v0.1.0/plots/performance_comparison.png), and
  [performance P1](results/v0.1.0/plots/performance_comparison_p1.png).
  Corresponding SVG files are in the same plots folder.

### CIFAR-10 and CIFAR-100 classification

Each result uses the complete 10,000-image official test split. All scores are percentages,
with Top-1 / Top-5 shown separately for each protocol. P0 uses CLIP Benchmark class names and
18 templates per class; P1 uses one model-native prompt. **P0 and P1 are separate protocols
and must not be combined into one ranking.**

**P0 · CLIP Benchmark-compatible · CUDA FP32**

| Model | CIFAR-10 Top-1 / Top-5 | CIFAR-100 Top-1 / Top-5 | Raw results (CIFAR-10 · CIFAR-100) |
|---|---:|---:|---|
| OpenAI CLIP ViT-B/32 | 89.83% / 99.61% | 64.20% / 88.73% | [JSON](results/v0.1.0/raw/35cef7e32f8afb13.json) · [JSON](results/v0.1.0/raw/5a8886b0fab54729-30d1b401.json) |
| OpenCLIP ViT-B/32 LAION-2B | 93.56% / 99.81% | 75.56% / 93.87% | [JSON](results/v0.1.0/raw/649c7b44763ebc72.json) · [JSON](results/v0.1.0/raw/35dfb13b0aa46c57-54e65389.json) |
| OpenCLIP ViT-L/14 LAION-2B | 96.61% / 99.85% | 83.34% / 96.63% | [JSON](results/v0.1.0/raw/c8faef3bbd4c2779.json) · [JSON](results/v0.1.0/raw/1de2d2b7854b5e80-30c3f626.json) |
| EmbeddingGemma 2 | 94.66% / 99.84% | 76.08% / 92.92% | [JSON](results/v0.1.0/raw/ab5d599956168b86.json) · [JSON](results/v0.1.0/raw/4bf47c3e534537b0-fb1914c0.json) |
| SigLIP 2 Base/16 224 | 93.62% / 99.85% | 73.68% / 91.89% | [JSON](results/v0.1.0/raw/38b1a8be68ae509c.json) · [JSON](results/v0.1.0/raw/d59181f45a441ab9-1a33071f.json) |
| Multilingual OpenCLIP ViT-B/32 | 93.64% / 99.88% | 76.55% / 94.99% | [JSON](results/v0.1.0/raw/931adb237a72e7b9.json) · [JSON](results/v0.1.0/raw/e4c4922a8b419fc2-ac4a43b4.json) |

**P1 · model-native prompts · CUDA FP32**

| Model | CIFAR-10 Top-1 / Top-5 | CIFAR-100 Top-1 / Top-5 | Raw results (CIFAR-10 · CIFAR-100) |
|---|---:|---:|---|
| OpenAI CLIP ViT-B/32 | 88.30% / 99.24% | 62.32% / 87.00% | [JSON](results/v0.1.0/raw/c542f4a0ea8ac9a8.json) · [JSON](results/v0.1.0/raw/c08f92332ebe01a4-2815d62c.json) |
| OpenCLIP ViT-B/32 LAION-2B | 93.68% / 99.84% | 75.59% / 93.65% | [JSON](results/v0.1.0/raw/c3350bf4370a58b8.json) · [JSON](results/v0.1.0/raw/a54f8b8dfb17227b-8e3ad59c.json) |
| OpenCLIP ViT-L/14 LAION-2B | 96.92% / 99.93% | 83.07% / 96.57% | [JSON](results/v0.1.0/raw/9c075a8286dd1604.json) · [JSON](results/v0.1.0/raw/5cd03bacdf1e5b06-43791242.json) |
| EmbeddingGemma 2 | 92.58% / 99.64% | 62.84% / 84.04% | [JSON](results/v0.1.0/raw/beac7b7ac03ee242.json) · [JSON](results/v0.1.0/raw/0c75f5095220051c-f8285c69.json) |
| SigLIP 2 Base/16 224 | 93.85% / 99.88% | 73.25% / 91.71% | [JSON](results/v0.1.0/raw/ddcae56da141ed7f.json) · [JSON](results/v0.1.0/raw/38a8be7d4b997457-8dbcaf34.json) |
| Multilingual OpenCLIP ViT-B/32 | 93.54% / 99.86% | 76.46% / 95.02% | [JSON](results/v0.1.0/raw/013541e887452604.json) · [JSON](results/v0.1.0/raw/d4721f7c31f9d23c-e083b306.json) |

EmbeddingGemma 2 uses its `Classification` task prefix in both classification protocols.
The raw audit confirms that P0 encodes 18 templates per class (180 CIFAR-10 prompts and 1,800
CIFAR-100 prompts), while P1 encodes one model-native prompt per class (10 and 100 prompts).
The substantial P0/P1 accuracy differences are therefore consistent with the implemented
prompt protocols; no prompt-path defect was found and no historical accuracy was rewritten.

### XM3600 retrieval — historical `all_captions`

These seven original runs use all 3,600 images and the same image-ID ordering. English uses
7,200 captions; Japanese uses 7,185. The pinned CLIP Benchmark XM3600 loader selects only the
first caption per image, so these `all_captions` scores must not be described as directly
comparable to its XM3600 retrieval scores. `T→I` is text-to-image and `I→T` is image-to-text.
Values are Recall@1 / Recall@5 / Recall@10 percentages. The pinned metric keys are
`image_retrieval_recall@k` for T→I and `text_retrieval_recall@k` for I→T.

| Model | Language | Captions | T→I R@1 / R@5 / R@10 | I→T R@1 / R@5 / R@10 | Raw result |
|---|---|---:|---:|---:|---|
| EmbeddingGemma 2 | English | 7,200 | 61.79% / 83.76% / 90.08% | 70.36% / 90.81% / 95.69% | [JSON](results/v0.1.0/raw/3284f6a2ba1863cc.json) |
| EmbeddingGemma 2 | Japanese | 7,185 | 76.23% / 92.86% / 96.10% | 84.25% / 96.78% / 98.64% | [JSON](results/v0.1.0/raw/5a7dc3bfc3d618c1.json) |
| Multilingual OpenCLIP ViT-B/32 | English | 7,200 | 48.72% / 76.18% / 84.35% | 57.75% / 83.50% / 91.06% | [JSON](results/v0.1.0/raw/8f9f79539fb877d6.json) |
| Multilingual OpenCLIP ViT-B/32 | Japanese | 7,185 | 59.19% / 84.05% / 91.20% | 71.56% / 92.83% / 96.56% | [JSON](results/v0.1.0/raw/028043fb5a078d1e-214521ad.json) |
| SigLIP 2 Base/16 224 | English | 7,200 | 52.44% / 76.92% / 84.08% | 60.67% / 83.89% / 89.72% | [JSON](results/v0.1.0/raw/36baa9edd5e9fa9a-53dd822c.json) |
| SigLIP 2 Base/16 224 | Japanese | 7,185 | 37.51% / 62.85% / 72.12% | 52.58% / 79.11% / 86.83% | [JSON](results/v0.1.0/raw/405472b594821180-783ab865.json) |
| rinna Japanese CLIP ViT-B/16 | Japanese | 7,185 | 50.33% / 77.27% / 85.21% | 58.00% / 84.83% / 91.61% | [JSON](results/v0.1.0/raw/221fdf4bab042e89.json) |

All historical runs use the same dataset checksum and image-ID ordering, and each
English/Japanese model pair was run from the same source commit on the same GPU. However, the
language candidate counts differ (7,200 versus 7,185); these legacy runs are not a
matched-candidate language comparison and no English/Japanese deltas should be inferred from
them. Per-run source SHAs, checkpoint revisions, checksums, prompts, timings, and memory
measurements are in the raw JSON. The seven raw files remain byte-for-byte unchanged; the
audit records their hashes and identifies the legacy variant assignment.

The seven historical all-caption results remain unchanged and are not directly comparable to
the pinned official XM3600 loader. The separate `official_first_caption` variant follows its
first-caption selection and yields 3,600 candidates per language on the same ordered 3,600
images. P2 remains the language-comparison protocol; caption selection is an additional,
hash- and resume-aware result dimension.

### XM3600 retrieval — official-compatible `official_first_caption`

These seven new complete-split evaluations use FP32, seed 0, requested batch size 16, and
P2. Each included image has exactly one caption. The pinned metric keys are
`image_retrieval_recall@k` for text-to-image (T→I) and `text_retrieval_recall@k` for
image-to-text (I→T). Values are percentages; each row links to its raw JSON result.

| Model | Language | Candidates | T→I R@1 / R@5 / R@10 | I→T R@1 / R@5 / R@10 | Raw result |
|---|---|---:|---:|---:|---|
| EmbeddingGemma 2 | English | 3,600 | 58.61% / 82.06% / 89.19% | 59.50% / 85.17% / 91.28% | [JSON](results/v0.1.0/raw/d5aaa9c64a02d8a7-f1d55790.json) |
| EmbeddingGemma 2 | Japanese | 3,600 | 80.14% / 95.03% / 97.50% | 79.61% / 95.28% / 97.61% | [JSON](results/v0.1.0/raw/311d4e3975436530-c5bef931.json) |
| Multilingual OpenCLIP ViT-B/32 | English | 3,600 | 47.08% / 74.86% / 83.03% | 48.31% / 76.64% / 85.19% | [JSON](results/v0.1.0/raw/3090789513e09ab4-ce9c0347.json) |
| Multilingual OpenCLIP ViT-B/32 | Japanese | 3,600 | 63.83% / 86.86% / 92.94% | 67.92% / 90.58% / 95.39% | [JSON](results/v0.1.0/raw/6bce751107f3fd39-86ab99c9.json) |
| SigLIP 2 Base/16 224 | English | 3,600 | 51.47% / 77.03% / 84.72% | 50.67% / 76.67% / 85.00% | [JSON](results/v0.1.0/raw/908af0749569fc22-7d1f73a4.json) |
| SigLIP 2 Base/16 224 | Japanese | 3,600 | 37.86% / 63.33% / 72.47% | 45.06% / 69.97% / 78.92% | [JSON](results/v0.1.0/raw/5b893cd6e37d6266-85b30816.json) |
| rinna Japanese CLIP ViT-B/16 | Japanese | 3,600 | 54.00% / 79.86% / 87.22% | 53.94% / 80.67% / 87.69% | [JSON](results/v0.1.0/raw/878e9c60981784ee-79118ade.json) |

### Matched English/Japanese comparison — `official_first_caption`

The three multilingual models below use the same 3,600 ordered image IDs and 3,600
caption candidates per language, with matching checkpoint and run conditions. Cells show
English → Japanese recall and the Japanese-minus-English difference in percentage points.
These are observed scores under the two caption conditions, not a causal estimate of
language understanding: caption content, translation quality, difficulty, and dataset
composition are not controlled to semantic equivalence.

| Model | T→I R@1 (EN → JA; Δ pp) | T→I R@5 (EN → JA; Δ pp) | T→I R@10 (EN → JA; Δ pp) | I→T R@1 (EN → JA; Δ pp) | I→T R@5 (EN → JA; Δ pp) | I→T R@10 (EN → JA; Δ pp) | Raw EN / JA |
|---|---:|---:|---:|---:|---:|---:|---|
| EmbeddingGemma 2 | 58.61% → 80.14% (+21.53 pp) | 82.06% → 95.03% (+12.97 pp) | 89.19% → 97.50% (+8.31 pp) | 59.50% → 79.61% (+20.11 pp) | 85.17% → 95.28% (+10.11 pp) | 91.28% → 97.61% (+6.33 pp) | [EN](results/v0.1.0/raw/d5aaa9c64a02d8a7-f1d55790.json) · [JA](results/v0.1.0/raw/311d4e3975436530-c5bef931.json) |
| Multilingual OpenCLIP ViT-B/32 | 47.08% → 63.83% (+16.75 pp) | 74.86% → 86.86% (+12.00 pp) | 83.03% → 92.94% (+9.92 pp) | 48.31% → 67.92% (+19.61 pp) | 76.64% → 90.58% (+13.94 pp) | 85.19% → 95.39% (+10.19 pp) | [EN](results/v0.1.0/raw/3090789513e09ab4-ce9c0347.json) · [JA](results/v0.1.0/raw/6bce751107f3fd39-86ab99c9.json) |
| SigLIP 2 Base/16 224 | 51.47% → 37.86% (-13.61 pp) | 77.03% → 63.33% (-13.69 pp) | 84.72% → 72.47% (-12.25 pp) | 50.67% → 45.06% (-5.61 pp) | 76.67% → 69.97% (-6.69 pp) | 85.00% → 78.92% (-6.08 pp) | [EN](results/v0.1.0/raw/908af0749569fc22-7d1f73a4.json) · [JA](results/v0.1.0/raw/5b893cd6e37d6266-85b30816.json) |

### CLIP Benchmark numerical parity

Against the pinned official CLIP Benchmark commit
[`ee14a9a8a9bcde09d51f4a4b1dff5a75182a3596`](https://github.com/LAION-AI/CLIP_benchmark/commit/ee14a9a8a9bcde09d51f4a4b1dff5a75182a3596),
OpenAI CLIP ViT-B/32 on CIFAR-10 P0 produced identical Top-1 (**89.83%**) and Top-5
(**99.61%**) results. Absolute differences were 0.0 at a `1e-12` tolerance. The parity
record documents the official `ViT-B-32-quickgelu` architecture, checkpoint SHA-256, complete
test split, 18 prompts, FP32, and matching data/labels.

A separate end-to-end XM3600 comparison used multilingual OpenCLIP ViT-B/32 on Japanese
`official_first_caption` candidates. All six retrieval metrics have matching integer hit
counts; the maximum absolute numeric difference is `2.9404958135081927e-8`, within the
recorded `5e-8` tolerance. The official run and `vlbench` used matching candidate order,
checkpoint, preprocessing, and token IDs. The small numerical difference is due to the
official implementation's float32 mean aggregation. This evidence applies only to that exact
model, language, checkpoint, precision, preprocessing, and candidate set; it does not imply
universal end-to-end parity.

### Hardware and measured performance

| Component | Verified value |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER, 16 GiB |
| Driver | 610.88 |
| Python / PyTorch | 3.12.10 / 2.14.1+cu130 |
| CUDA runtime | 13.0 |
| BF16 support | Available; not used for these measurements |
| Benchmark precision | FP32; CUDA inference; requested batch size 16 |

The full report and CSV give per-run image/text throughput, median inference latency, and
peak allocated/reserved VRAM. For example, the OpenAI CLIP CIFAR-10 P0 run measured
**745.5 images/s** and **0.62 / 0.68 GiB** peak allocated / reserved VRAM. Throughput and
memory are model- and run-specific; smoke-test memory figures are not substituted for full
benchmark measurements. Actual image/text batch sizes and OOM retries are reported per run.
EmbeddingGemma's aspect-ratio-aware XM3600 image sub-batches had an actual size of 11, while
its text batch size was 16; no selected run needed an OOM retry.

## Reproduction

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync --locked --group dev
uv run vlbench list models
uv run vlbench list datasets
uv run vlbench run --model openai_clip_b32 --dataset cifar10 --device cuda --dry-run
```

Example full-data commands:

```powershell
uv run --locked --extra cuda --extra openclip vlbench run `
  --model openai_clip_b32 --dataset cifar10 --protocol p0 --language en `
  --precision fp32 --device cuda:0 --batch-size 16 --seed 0 `
  --output results/v0.1.0 --force --yes

uv run --locked --extra cuda --extra gemma vlbench run `
  --model embeddinggemma2 --dataset crossmodal3600 --protocol p2 --language ja `
  --retrieval-variant official_first_caption `
  --precision fp32 --device cuda:0 --batch-size 16 --seed 0 `
  --output results/v0.1.0 --force --yes

uv run --locked vlbench report --input results/v0.1.0/raw --output results/v0.1.0
```

`--yes` authorizes the reviewed network plan; it does not accept dataset or model terms.
Do not use `--confirm-data-terms` without genuine user authorization. Exact per-run
reproduction settings and optional real-model test commands are in
[reproducibility documentation](docs/reproducibility.md).

## Limitations and blocked datasets

- No BF16 accuracy runs or statistical confidence intervals have been published.
- Pretraining data overlap cannot be ruled out unless independently verified.
- English/Japanese differences are observed performance under each caption condition, not
  causal estimates of language understanding; caption meaning, translation quality, and
  difficulty are not controlled to semantic equivalence.
- End-to-end XM3600 parity has been established only for multilingual OpenCLIP ViT-B/32 on
  Japanese `official_first_caption` candidates; it does not extend to other models, languages,
  datasets, or the `all_captions` variant.
- Flickr30k remains blocked until approved official access is confirmed; the caption license
  description and linked license also conflict.
- COCO remains blocked because downloading entails acceptance of official terms, which has
  not been provided, and image-specific rights are not yet preserved and validated.
- XTD10 and the XTD10-based multilingual COCO path require acceptance of the Adobe Research
  License. ImageNet requires authorized access; Babel-ImageNet additionally needs verified
  Japanese-label coverage.
- Dataset images, model weights, caches, and credentials are not published.
- Default GitHub CI remains CPU-only; passing tests are not a substitute for the real CUDA
  measurements reported above.

See [implementation status](docs/implementation-status.md), [methodology](docs/methodology.md),
[model and dataset licenses](docs/model-and-dataset-licenses.md), and
[results status](results/README.md) for detailed evidence and limitations.
