# Vision-Language Benchmarks

A reproducible framework for evaluating pretrained vision-language models on zero-shot
classification and bilingual image-text retrieval.

## Verified v0.1.0 results

The existing release contains 31 full-dataset CUDA FP32 measurements on an NVIDIA GeForce RTX
4070 Ti SUPER: 24 CIFAR classification runs and seven historical XM3600
`all_captions` retrieval runs. The separate `official_first_caption` XM3600 protocol is
implemented and tested; its full-dataset CUDA measurements are not included until completed
and validated. Seven pretrained model adapters also passed real image/text CUDA smoke tests.
The smoke tests are distinct from the full evaluations.

- [Full report](results/v0.1.0/report.md) · [CSV summary](results/v0.1.0/summary.csv) ·
  [metadata](results/v0.1.0/metadata.json)
- [Validated per-run JSON results](results/v0.1.0/raw/) ·
  [machine-readable run status](results/v0.1.0/run-status.json)
- [Historical raw-result audit and preservation hashes](results/v0.1.0/validation/historical-results-audit.json)
- [Real CUDA model smoke-test evidence](results/v0.1.0/validation/model-smoke-tests.json)
- [Official CLIP Benchmark parity record](results/v0.1.0/validation/official-cifar10-parity.json)
- Charts: [classification P0](results/v0.1.0/plots/classification_accuracy.png),
  [classification P1](results/v0.1.0/plots/classification_accuracy_p1.png),
  [historical all-caption XM3600 retrieval](results/v0.1.0/plots/xm3600_all_captions_retrieval.png),
  [performance P0](results/v0.1.0/plots/performance_comparison.png), and
  [performance P1](results/v0.1.0/plots/performance_comparison_p1.png).
  Corresponding SVG files are in the same plots folder.

### CIFAR-10 and CIFAR-100 classification

Each result uses the complete 10,000-image official test split. All scores are percentages,
with Top-1 / Top-5 shown separately for each protocol. P0 uses CLIP Benchmark class names and
18 templates per class; P1 uses one model-native prompt. **P0 and P1 are separate protocols
and must not be combined into one ranking.**

**P0 · CLIP Benchmark-compatible · CUDA FP32**

| Model | CIFAR-10 Top-1 / Top-5 | CIFAR-100 Top-1 / Top-5 |
|---|---:|---:|
| OpenAI CLIP ViT-B/32 | 89.83% / 99.61% | 64.20% / 88.73% |
| OpenCLIP ViT-B/32 LAION-2B | 93.56% / 99.81% | 75.56% / 93.87% |
| OpenCLIP ViT-L/14 LAION-2B | 96.61% / 99.85% | 83.34% / 96.63% |
| EmbeddingGemma 2 | 94.66% / 99.84% | 76.08% / 92.92% |
| SigLIP 2 Base/16 224 | 93.62% / 99.85% | 73.68% / 91.89% |
| Multilingual OpenCLIP ViT-B/32 | 93.64% / 99.88% | 76.55% / 94.99% |

**P1 · model-native prompts · CUDA FP32**

| Model | CIFAR-10 Top-1 / Top-5 | CIFAR-100 Top-1 / Top-5 |
|---|---:|---:|
| OpenAI CLIP ViT-B/32 | 88.30% / 99.24% | 62.32% / 87.00% |
| OpenCLIP ViT-B/32 LAION-2B | 93.68% / 99.84% | 75.59% / 93.65% |
| OpenCLIP ViT-L/14 LAION-2B | 96.92% / 99.93% | 83.07% / 96.57% |
| EmbeddingGemma 2 | 92.58% / 99.64% | 62.84% / 84.04% |
| SigLIP 2 Base/16 224 | 93.85% / 99.88% | 73.25% / 91.71% |
| Multilingual OpenCLIP ViT-B/32 | 93.54% / 99.86% | 76.46% / 95.02% |

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

| Model | Language | Captions | T→I R@1 / R@5 / R@10 | I→T R@1 / R@5 / R@10 |
|---|---|---:|---:|---:|
| EmbeddingGemma 2 | English | 7,200 | 61.79% / 83.76% / 90.08% | 70.36% / 90.81% / 95.69% |
| EmbeddingGemma 2 | Japanese | 7,185 | 76.23% / 92.86% / 96.10% | 84.25% / 96.78% / 98.64% |
| Multilingual OpenCLIP ViT-B/32 | English | 7,200 | 48.72% / 76.18% / 84.35% | 57.75% / 83.50% / 91.06% |
| Multilingual OpenCLIP ViT-B/32 | Japanese | 7,185 | 59.19% / 84.05% / 91.20% | 71.56% / 92.83% / 96.56% |
| SigLIP 2 Base/16 224 | English | 7,200 | 52.44% / 76.92% / 84.08% | 60.67% / 83.89% / 89.72% |
| SigLIP 2 Base/16 224 | Japanese | 7,185 | 37.51% / 62.85% / 72.12% | 52.58% / 79.11% / 86.83% |
| rinna Japanese CLIP ViT-B/16 | Japanese | 7,185 | 50.33% / 77.27% / 85.21% | 58.00% / 84.83% / 91.61% |

All historical runs use the same dataset checksum and image-ID ordering, and each
English/Japanese model pair was run from the same source commit on the same GPU. However, the
language candidate counts differ (7,200 versus 7,185); these legacy runs are not a
matched-candidate language comparison and no English/Japanese deltas should be inferred from
them. Per-run source SHAs, checkpoint revisions, checksums, prompts, timings, and memory
measurements are in the raw JSON. The seven raw files remain byte-for-byte unchanged; the
audit records their hashes and identifies the legacy variant assignment.

The new `official_first_caption` variant selects one caption per image using the pinned
reference's dataset behavior, yielding 3,600 candidates in each language with identical
ordered image IDs. P2 remains the language-comparison protocol; caption selection is an
additional, hash- and resume-aware result dimension. Its full CUDA evaluation and publication
status is tracked in [implementation status](docs/implementation-status.md).

### CLIP Benchmark numerical parity

Against the pinned official CLIP Benchmark commit
[`ee14a9a8a9bcde09d51f4a4b1dff5a75182a3596`](https://github.com/LAION-AI/CLIP_benchmark/commit/ee14a9a8a9bcde09d51f4a4b1dff5a75182a3596),
OpenAI CLIP ViT-B/32 on CIFAR-10 P0 produced identical Top-1 (**89.83%**) and Top-5
(**99.61%**) results. Absolute differences were 0.0 at a `1e-12` tolerance. The parity
record documents the official `ViT-B-32-quickgelu` architecture, checkpoint SHA-256, complete
test split, 18 prompts, FP32, and matching data/labels. This parity claim does not extend to
other models or datasets.

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
