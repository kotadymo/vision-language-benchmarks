# Vision-Language Benchmark Report

> **Source identity notice:** This is a measurement using the **unpublished correction source based on v0.7.0** (`openvlclip` Git commit `8596295c1f519506eda5842e0322b2fbc8ad8daa`), not the official PyPI `openvlclip==0.7.0` distribution. The unchanged package-version metadata is not source identity. See [source provenance](validation/source-provenance.json) and the [exact source diff](validation/openvlclip-v0.7.0-to-corrected.patch).

This report is regenerated from schema-validated raw JSON. Only completed runs contribute numeric metrics; failed, skipped, and blocked runs are listed separately. No values are estimated or fabricated.

## Conditions

- Validated result files: 12
- Completed measurements: 12
- Skipped runs: 0
- Failed runs: 0
- Blocked runs: 0
- Source commit(s): 60bb3c6a10fa827e190954b4f74c6dd82b74e2ae
- Runtime profile(s): NVIDIA GeForce RTX 4070 Ti SUPER (17170956288 bytes, driver 610.88, PyTorch 2.14.1+cu130, CUDA 13.0, fp32)
- Protocols and precisions are kept in separate tables and chart files.
- Accuracy and recall values in JSON are fractions; report tables display percentages.
- Checkpoint sizes are pinned weight-file sizes; tokenizer and configuration files may add to the download. Peak memory is reported as allocated / reserved GiB.
- Each completed table row links to its schema-validated raw result JSON.

## Measurement lifecycle

- Frozen benchmark source commit: `60bb3c6a10fa827e190954b4f74c6dd82b74e2ae`
- Active validated measurements: 12
- Retained historical measurements: 0
- Superseded measurements excluded from primary results: 0
- Failed attempts: 0
- Incomplete attempts: 0
- Invalid measurements: 0
- Blocked runs: 0
- Skipped runs: 0
- Per-run lifecycle assignments are recorded in [result-lifecycle.json](validation/result-lifecycle.json).

## English zero-shot classification

### P0 · FP32

| Model | Embedding dim | Parameters | Checkpoint | Dataset | Split | Images | Classes | Requested / image / text batch | Top-1 (correct/total) | Top-5 (correct/total) | Images/s | Median (ms) | Peak allocated / reserved (GiB) | OOM retries | Raw result |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| OpenVLCLIP SigLIP 2 Base/16 FixRes | 768 | 375.2 M | 1.40 GiB | CIFAR-10 | test | 10,000 | 10 | 16 / 16 / 16 | 93.62% (9,362/10,000) | 99.85% (9,985/10,000) | 306.3 | 52.47 | 1.52 / 1.67 | 0 | [04f85d8307897f41](raw/04f85d8307897f41.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | 768 | 375.2 M | 1.40 GiB | CIFAR-100 | test | 10,000 | 100 | 16 / 16 / 16 | 75.43% (7,543/10,000) | 92.57% (9,257/10,000) | 258.6 | 62.35 | 1.56 / 1.70 | 0 | [2c8d507fcc3fa129](raw/2c8d507fcc3fa129.json) |
| OpenVLCLIP SigLIP 2 Base/16 FixRes | 768 | 375.2 M | 1.40 GiB | CIFAR-100 | test | 10,000 | 100 | 16 / 16 / 16 | 73.68% (7,368/10,000) | 91.89% (9,189/10,000) | 304.3 | 53.36 | 1.52 / 1.66 | 0 | [8bda121d4a8e9e0c](raw/8bda121d4a8e9e0c.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | 768 | 375.2 M | 1.40 GiB | CIFAR-10 | test | 10,000 | 10 | 16 / 16 / 16 | 94.05% (9,405/10,000) | 99.70% (9,970/10,000) | 214.8 | 66.43 | 1.56 / 1.70 | 0 | [adea66ba17455b4b](raw/adea66ba17455b4b.json) |

### P1 · FP32

| Model | Embedding dim | Parameters | Checkpoint | Dataset | Split | Images | Classes | Requested / image / text batch | Top-1 (correct/total) | Top-5 (correct/total) | Images/s | Median (ms) | Peak allocated / reserved (GiB) | OOM retries | Raw result |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | 768 | 375.2 M | 1.40 GiB | CIFAR-100 | test | 10,000 | 100 | 16 / 16 / 16 | 73.23% (7,323/10,000) | 91.64% (9,164/10,000) | 264.5 | 60.63 | 1.56 / 1.70 | 0 | [3ca50611538ca5da](raw/3ca50611538ca5da.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | 768 | 375.2 M | 1.40 GiB | CIFAR-10 | test | 10,000 | 10 | 16 / 16 / 16 | 93.69% (9,369/10,000) | 99.71% (9,971/10,000) | 245.0 | 64.00 | 1.56 / 1.70 | 0 | [98b0802263ec6f4c](raw/98b0802263ec6f4c.json) |
| OpenVLCLIP SigLIP 2 Base/16 FixRes | 768 | 375.2 M | 1.40 GiB | CIFAR-100 | test | 10,000 | 100 | 16 / 16 / 16 | 73.25% (7,325/10,000) | 91.71% (9,171/10,000) | 305.3 | 52.63 | 1.52 / 1.67 | 0 | [b1cc0c8b4149e1b1](raw/b1cc0c8b4149e1b1.json) |
| OpenVLCLIP SigLIP 2 Base/16 FixRes | 768 | 375.2 M | 1.40 GiB | CIFAR-10 | test | 10,000 | 10 | 16 / 16 / 16 | 93.85% (9,385/10,000) | 99.88% (9,988/10,000) | 305.4 | 52.80 | 1.52 / 1.65 | 0 | [d3ef005049f56d7c](raw/d3ef005049f56d7c.json) |

## Japanese/English official-compatible XM3600 retrieval (`official_first_caption`)

### EN · P2 · FP32

| Model | Embedding dim | Parameters | Checkpoint | Dataset | Split | Language | Retrieval variant | Images / captions | Requested / image / text batch | T→I R@1 (hits/queries) | T→I R@5 (hits/queries) | T→I R@10 (hits/queries) | I→T R@1 (hits/queries) | I→T R@5 (hits/queries) | I→T R@10 (hits/queries) | Images/s | Texts/s | Median (ms) | Peak allocated / reserved (GiB) | OOM retries | Raw result |
|---|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | 768 | 375.2 M | 1.40 GiB | Crossmodal-3600 (XM3600) | all 3,600 images | en | official_first_caption | 3,600 / 3,600 | 16 / 16 / 16 | 53.86% (1,939/3,600) | 80.08% (2,883/3,600) | 87.56% (3,152/3,600) | 51.06% (1,838/3,600) | 78.28% (2,818/3,600) | 85.75% (3,087/3,600) | 221.0 | 5.3 | 918.11 | 1.56 / 1.69 | 0 | [1aab08bebc0b122f](raw/1aab08bebc0b122f.json) |
| OpenVLCLIP SigLIP 2 Base/16 FixRes | 768 | 375.2 M | 1.40 GiB | Crossmodal-3600 (XM3600) | all 3,600 images | en | official_first_caption | 3,600 / 3,600 | 16 / 16 / 16 | 51.47% (1,853/3,600) | 77.03% (2,773/3,600) | 84.72% (3,050/3,600) | 50.67% (1,824/3,600) | 76.67% (2,760/3,600) | 85.00% (3,060/3,600) | 177.4 | 5.4 | 940.44 | 1.52 / 1.65 | 0 | [97b6d69758693acc](raw/97b6d69758693acc.json) |

### JA · P2 · FP32

| Model | Embedding dim | Parameters | Checkpoint | Dataset | Split | Language | Retrieval variant | Images / captions | Requested / image / text batch | T→I R@1 (hits/queries) | T→I R@5 (hits/queries) | T→I R@10 (hits/queries) | I→T R@1 (hits/queries) | I→T R@5 (hits/queries) | I→T R@10 (hits/queries) | Images/s | Texts/s | Median (ms) | Peak allocated / reserved (GiB) | OOM retries | Raw result |
|---|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| OpenVLCLIP SigLIP 2 Base/16 FixRes | 768 | 375.2 M | 1.40 GiB | Crossmodal-3600 (XM3600) | all 3,600 images | ja | official_first_caption | 3,600 / 3,600 | 16 / 16 / 16 | 37.86% (1,363/3,600) | 63.33% (2,280/3,600) | 72.47% (2,609/3,600) | 45.06% (1,622/3,600) | 69.97% (2,519/3,600) | 78.92% (2,841/3,600) | 264.4 | 6.9 | 923.76 | 1.52 / 1.65 | 0 | [1f68a1ca616f9819-0544d6b3](raw/1f68a1ca616f9819-0544d6b3.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | 768 | 375.2 M | 1.40 GiB | Crossmodal-3600 (XM3600) | all 3,600 images | ja | official_first_caption | 3,600 / 3,600 | 16 / 16 / 16 | 36.86% (1,327/3,600) | 62.25% (2,241/3,600) | 72.17% (2,598/3,600) | 40.64% (1,463/3,600) | 66.08% (2,379/3,600) | 74.89% (2,696/3,600) | 173.5 | 6.0 | 957.96 | 1.56 / 1.69 | 0 | [f0417644516402eb](raw/f0417644516402eb.json) |

## Japanese zero-shot classification

No completed measured runs are available.

## XM3600 all-caption retrieval (`all_captions`)

No completed measured runs are available.

## Matched English/Japanese retrieval

### Crossmodal-3600 (XM3600) · P2 · FP32 · official_first_caption

| Model | Embedding dim | Split | Images | Captions (EN / JA) | T→I R@1 (EN → JA; Δ pp) | T→I R@5 (EN → JA; Δ pp) | T→I R@10 (EN → JA; Δ pp) | I→T R@1 (EN → JA; Δ pp) | I→T R@5 (EN → JA; Δ pp) | I→T R@10 (EN → JA; Δ pp) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenVLCLIP SigLIP 2 Base/16 FixRes | 768 | all 3,600 images | 3,600 | 3,600 / 3,600 | 51.47% → 37.86% (-13.61 pp) | 77.03% → 63.33% (-13.69 pp) | 84.72% → 72.47% (-12.25 pp) | 50.67% → 45.06% (-5.61 pp) | 76.67% → 69.97% (-6.69 pp) | 85.00% → 78.92% (-6.08 pp) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | 768 | all 3,600 images | 3,600 | 3,600 / 3,600 | 53.86% → 36.86% (-17.00 pp) | 80.08% → 62.25% (-17.83 pp) | 87.56% → 72.17% (-15.39 pp) | 51.06% → 40.64% (-10.42 pp) | 78.28% → 66.08% (-12.19 pp) | 85.75% → 74.89% (-10.86 pp) |

## Model throughput and VRAM

Throughput is computed from model inference only; model downloads, initialization, dataset preparation, and report generation are excluded.

| Model | Dataset | Task | Language | Protocol | Retrieval variant | Precision | Images/s | Texts/s | Median (ms) | Peak allocated / reserved (GiB) | Raw result |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---|
| OpenVLCLIP SigLIP 2 Base/16 FixRes | CIFAR-10 | classification | en | p0 | — | fp32 | 306.3 | 8.2 | 52.47 | 1.52 / 1.67 | [04f85d8307897f41](raw/04f85d8307897f41.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | CIFAR-10 | classification | en | p0 | — | fp32 | 214.8 | 4.6 | 66.43 | 1.56 / 1.70 | [adea66ba17455b4b](raw/adea66ba17455b4b.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | CIFAR-10 | classification | en | p1 | — | fp32 | 245.0 | 3.8 | 64.00 | 1.56 / 1.70 | [98b0802263ec6f4c](raw/98b0802263ec6f4c.json) |
| OpenVLCLIP SigLIP 2 Base/16 FixRes | CIFAR-10 | classification | en | p1 | — | fp32 | 305.4 | 6.1 | 52.80 | 1.52 / 1.65 | [d3ef005049f56d7c](raw/d3ef005049f56d7c.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | CIFAR-100 | classification | en | p0 | — | fp32 | 258.6 | 5.4 | 62.35 | 1.56 / 1.70 | [2c8d507fcc3fa129](raw/2c8d507fcc3fa129.json) |
| OpenVLCLIP SigLIP 2 Base/16 FixRes | CIFAR-100 | classification | en | p0 | — | fp32 | 304.3 | 8.5 | 53.36 | 1.52 / 1.66 | [8bda121d4a8e9e0c](raw/8bda121d4a8e9e0c.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | CIFAR-100 | classification | en | p1 | — | fp32 | 264.5 | 7.4 | 60.63 | 1.56 / 1.70 | [3ca50611538ca5da](raw/3ca50611538ca5da.json) |
| OpenVLCLIP SigLIP 2 Base/16 FixRes | CIFAR-100 | classification | en | p1 | — | fp32 | 305.3 | 7.9 | 52.63 | 1.52 / 1.67 | [b1cc0c8b4149e1b1](raw/b1cc0c8b4149e1b1.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | Crossmodal-3600 (XM3600) | retrieval | en | p2 | official_first_caption | fp32 | 221.0 | 5.3 | 918.11 | 1.56 / 1.69 | [1aab08bebc0b122f](raw/1aab08bebc0b122f.json) |
| OpenVLCLIP SigLIP 2 Base/16 FixRes | Crossmodal-3600 (XM3600) | retrieval | en | p2 | official_first_caption | fp32 | 177.4 | 5.4 | 940.44 | 1.52 / 1.65 | [97b6d69758693acc](raw/97b6d69758693acc.json) |
| OpenVLCLIP SigLIP 2 Base/16 FixRes | Crossmodal-3600 (XM3600) | retrieval | ja | p2 | official_first_caption | fp32 | 264.4 | 6.9 | 923.76 | 1.52 / 1.65 | [1f68a1ca616f9819-0544d6b3](raw/1f68a1ca616f9819-0544d6b3.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | Crossmodal-3600 (XM3600) | retrieval | ja | p2 | official_first_caption | fp32 | 173.5 | 6.0 | 957.96 | 1.56 / 1.69 | [f0417644516402eb](raw/f0417644516402eb.json) |

## Matched historical model comparisons

Rows are matched to each new benchmark run by dataset checksum, split, task, protocol, language, precision, sample/candidate counts, and retrieval variant. Only lifecycle-eligible completed historical results are included. Matching protocol labels do not imply identical model-side preprocessing or instructions.
P0 keeps the common 18-template candidate-text protocol; P1 remains model-specific.
The historical direct Transformers SigLIP 2 CIFAR results predate the PAD-derived attention mask used by OpenVLCLIP v0.7.0. On the same first 128 CIFAR-10 P0 images and 180 prompts, OpenVLCLIP features matched the current masked direct adapter exactly, while the legacy unmasked text features differed (maximum absolute difference 6.9498; mean prompt cosine 0.6882). The historical 93.62% Top-1 result is retained unchanged as context, not a controlled like-for-like quality comparison. The subset diagnostic is `validation/siglip2-p0-parity.json`.
Qwen's fixed classification instruction and separate retrieval query/document instructions are recorded in the comparison CSV and raw JSON. Qwen retrieval throughput counts both text-encoding passes.
Machine-readable rows: [model-comparison.csv](model-comparison.csv); selection details: [model-comparison-audit.json](validation/model-comparison-audit.json).

### CIFAR10 · P0 · EN · FP32

| Model | Top-1 (correct/total) | Top-5 (correct/total) | Images/s | Peak allocated/reserved (GiB) | Embedding/vector bytes | FP32 image+text payload | Package | Source commit | Raw result |
|---|---:|---:|---:|---:|---|---:|---|---|---|
| OpenVLCLIP SigLIP 2 Base/16 FixRes | 93.62% (9,362/10,000) | 99.85% (9,985/10,000) | 306.33 | 1.52 / 1.67 | 768D · 3,072 B/vector | 29.82 MiB | openvlclip==0.7.0 | 60bb3c6a10fa | [04f85d8307897f41](raw/04f85d8307897f41.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | 94.05% (9,405/10,000) | 99.70% (9,970/10,000) | 214.84 | 1.56 / 1.70 | 768D · 3,072 B/vector | 29.82 MiB | openvlclip==0.7.0 | 60bb3c6a10fa | [adea66ba17455b4b](raw/adea66ba17455b4b.json) |
| DINOv2 ViT-L/14 + dino.txt | 98.24% (9,824/10,000) | 99.95% (9,995/10,000) | 80.99 | 3.55 / 3.71 | 2,048D · 8,192 B/vector | 79.53 MiB | openvlclip==0.6.0 | 28519573b78f | [7d2fd70c52399af1](../dinov2-dinotxt-v0.6.0/raw/7d2fd70c52399af1.json) |
| EmbeddingGemma 2 | 94.66% (9,466/10,000) | 99.84% (9,984/10,000) | 10.21 | 4.97 / 5.90 | 768D · 3,072 B/vector | 29.82 MiB | openclip-gemma==0.4.0 | a9909fbe1b15 | [b8214ecb970362c9](../mrl-openclip-gemma-v0.4.0/raw/b8214ecb970362c9.json) |
| OpenAI CLIP ViT-B/32 | 89.83% (8,983/10,000) | 99.61% (9,961/10,000) | 745.47 | 0.62 / 0.68 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [35cef7e32f8afb13](../v0.1.0/raw/35cef7e32f8afb13.json) |
| OpenCLIP ViT-B/32 LAION-2B | 93.56% (9,356/10,000) | 99.81% (9,981/10,000) | 749.63 | 0.61 / 0.68 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [649c7b44763ebc72](../v0.1.0/raw/649c7b44763ebc72.json) |
| OpenCLIP ViT-L/14 LAION-2B | 96.61% (9,661/10,000) | 99.85% (9,985/10,000) | 106.59 | 1.80 / 1.95 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [c8faef3bbd4c2779](../v0.1.0/raw/c8faef3bbd4c2779.json) |
| OpenCLIP Multilingual ViT-B/32 | 93.64% (9,364/10,000) | 99.88% (9,988/10,000) | 701.99 | 1.64 / 1.90 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [931adb237a72e7b9](../v0.1.0/raw/931adb237a72e7b9.json) |
| OpenVLCLIP Qwen3-VL-Embedding-2B | 87.58% (8,758/10,000) | 98.68% (9,868/10,000) | 5.26 | 7.94 / 8.06 | 2,048D · 8,192 B/vector | 79.53 MiB | openvlclip==0.7.0 | 23b469da312e | [44ca47e7559432f0](../openvlclip-v0.7.0/raw/44ca47e7559432f0.json) |
| SigLIP 2 Base/16 224 | 93.62% (9,362/10,000) | 99.85% (9,985/10,000) | 338.26 | 1.52 / 1.67 | not recorded | not recorded | transformers==5.20.1 | 60a61259c95e | [38b1a8be68ae509c](../v0.1.0/raw/38b1a8be68ae509c.json) |

### CIFAR10 · P1 · EN · FP32

| Model | Top-1 (correct/total) | Top-5 (correct/total) | Images/s | Peak allocated/reserved (GiB) | Embedding/vector bytes | FP32 image+text payload | Package | Source commit | Raw result |
|---|---:|---:|---:|---:|---|---:|---|---|---|
| OpenVLCLIP SigLIP 2 Base/16 FixRes | 93.85% (9,385/10,000) | 99.88% (9,988/10,000) | 305.42 | 1.52 / 1.65 | 768D · 3,072 B/vector | 29.33 MiB | openvlclip==0.7.0 | 60bb3c6a10fa | [d3ef005049f56d7c](raw/d3ef005049f56d7c.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | 93.69% (9,369/10,000) | 99.71% (9,971/10,000) | 245.03 | 1.56 / 1.70 | 768D · 3,072 B/vector | 29.33 MiB | openvlclip==0.7.0 | 60bb3c6a10fa | [98b0802263ec6f4c](raw/98b0802263ec6f4c.json) |
| DINOv2 ViT-L/14 + dino.txt | 98.12% (9,812/10,000) | 99.94% (9,994/10,000) | 92.03 | 3.55 / 3.72 | 2,048D · 8,192 B/vector | 78.20 MiB | openvlclip==0.6.0 | 28519573b78f | [0f885e751a90a8ef](../dinov2-dinotxt-v0.6.0/raw/0f885e751a90a8ef.json) |
| EmbeddingGemma 2 | 92.58% (9,258/10,000) | 99.64% (9,964/10,000) | 10.18 | 4.97 / 5.88 | 768D · 3,072 B/vector | 29.33 MiB | openclip-gemma==0.4.0 | a9909fbe1b15 | [6950817bdd99d973](../mrl-openclip-gemma-v0.4.0/raw/6950817bdd99d973.json) |
| OpenAI CLIP ViT-B/32 | 88.30% (8,830/10,000) | 99.24% (9,924/10,000) | 688.74 | 0.62 / 0.68 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [c542f4a0ea8ac9a8](../v0.1.0/raw/c542f4a0ea8ac9a8.json) |
| OpenCLIP ViT-B/32 LAION-2B | 93.68% (9,368/10,000) | 99.84% (9,984/10,000) | 717.66 | 0.61 / 0.66 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [c3350bf4370a58b8](../v0.1.0/raw/c3350bf4370a58b8.json) |
| OpenCLIP ViT-L/14 LAION-2B | 96.92% (9,692/10,000) | 99.93% (9,993/10,000) | 106.76 | 1.80 / 1.90 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [9c075a8286dd1604](../v0.1.0/raw/9c075a8286dd1604.json) |
| OpenCLIP Multilingual ViT-B/32 | 93.54% (9,354/10,000) | 99.86% (9,986/10,000) | 739.21 | 1.54 / 1.71 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [013541e887452604](../v0.1.0/raw/013541e887452604.json) |
| OpenVLCLIP Qwen3-VL-Embedding-2B | 87.08% (8,708/10,000) | 98.68% (9,868/10,000) | 4.95 | 7.94 / 8.06 | 2,048D · 8,192 B/vector | 78.20 MiB | openvlclip==0.7.0 | 23b469da312e | [b602850fd3d77ccd](../openvlclip-v0.7.0/raw/b602850fd3d77ccd.json) |
| SigLIP 2 Base/16 224 | 93.85% (9,385/10,000) | 99.88% (9,988/10,000) | 336.90 | 1.52 / 1.65 | not recorded | not recorded | transformers==5.20.1 | 60a61259c95e | [ddcae56da141ed7f](../v0.1.0/raw/ddcae56da141ed7f.json) |

### CIFAR100 · P0 · EN · FP32

| Model | Top-1 (correct/total) | Top-5 (correct/total) | Images/s | Peak allocated/reserved (GiB) | Embedding/vector bytes | FP32 image+text payload | Package | Source commit | Raw result |
|---|---:|---:|---:|---:|---|---:|---|---|---|
| OpenVLCLIP SigLIP 2 Base/16 FixRes | 73.68% (7,368/10,000) | 91.89% (9,189/10,000) | 304.29 | 1.52 / 1.66 | 768D · 3,072 B/vector | 34.57 MiB | openvlclip==0.7.0 | 60bb3c6a10fa | [8bda121d4a8e9e0c](raw/8bda121d4a8e9e0c.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | 75.43% (7,543/10,000) | 92.57% (9,257/10,000) | 258.57 | 1.56 / 1.70 | 768D · 3,072 B/vector | 34.57 MiB | openvlclip==0.7.0 | 60bb3c6a10fa | [2c8d507fcc3fa129](raw/2c8d507fcc3fa129.json) |
| DINOv2 ViT-L/14 + dino.txt | 89.71% (8,971/10,000) | 98.17% (9,817/10,000) | 91.73 | 3.55 / 3.71 | 2,048D · 8,192 B/vector | 92.19 MiB | openvlclip==0.6.0 | 28519573b78f | [f0b3f208c3f8f151](../dinov2-dinotxt-v0.6.0/raw/f0b3f208c3f8f151.json) |
| EmbeddingGemma 2 | 76.08% (7,608/10,000) | 92.92% (9,292/10,000) | 10.19 | 4.97 / 5.94 | 768D · 3,072 B/vector | 34.57 MiB | openclip-gemma==0.4.0 | a9909fbe1b15 | [b08c9f0fbe719961](../mrl-openclip-gemma-v0.4.0/raw/b08c9f0fbe719961.json) |
| OpenAI CLIP ViT-B/32 | 64.20% (6,420/10,000) | 88.73% (8,873/10,000) | 668.75 | 0.62 / 0.68 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [5a8886b0fab54729-30d1b401](../v0.1.0/raw/5a8886b0fab54729-30d1b401.json) |
| OpenCLIP ViT-B/32 LAION-2B | 75.56% (7,556/10,000) | 93.87% (9,387/10,000) | 701.07 | 0.61 / 0.68 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [35dfb13b0aa46c57-54e65389](../v0.1.0/raw/35dfb13b0aa46c57-54e65389.json) |
| OpenCLIP ViT-L/14 LAION-2B | 83.34% (8,334/10,000) | 96.63% (9,663/10,000) | 106.97 | 1.80 / 1.94 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [1de2d2b7854b5e80-30c3f626](../v0.1.0/raw/1de2d2b7854b5e80-30c3f626.json) |
| OpenCLIP Multilingual ViT-B/32 | 76.55% (7,655/10,000) | 94.99% (9,499/10,000) | 750.93 | 1.64 / 1.90 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [e4c4922a8b419fc2-ac4a43b4](../v0.1.0/raw/e4c4922a8b419fc2-ac4a43b4.json) |
| OpenVLCLIP Qwen3-VL-Embedding-2B | 57.66% (5,766/10,000) | 80.99% (8,099/10,000) | 8.45 | 7.94 / 8.06 | 2,048D · 8,192 B/vector | 92.19 MiB | openvlclip==0.7.0 | 23b469da312e | [1e814e0b64df8a09](../openvlclip-v0.7.0/raw/1e814e0b64df8a09.json) |
| SigLIP 2 Base/16 224 | 73.68% (7,368/10,000) | 91.89% (9,189/10,000) | 342.26 | 1.52 / 1.66 | not recorded | not recorded | transformers==5.20.1 | 60a61259c95e | [d59181f45a441ab9-1a33071f](../v0.1.0/raw/d59181f45a441ab9-1a33071f.json) |

### CIFAR100 · P1 · EN · FP32

| Model | Top-1 (correct/total) | Top-5 (correct/total) | Images/s | Peak allocated/reserved (GiB) | Embedding/vector bytes | FP32 image+text payload | Package | Source commit | Raw result |
|---|---:|---:|---:|---:|---|---:|---|---|---|
| OpenVLCLIP SigLIP 2 Base/16 FixRes | 73.25% (7,325/10,000) | 91.71% (9,171/10,000) | 305.27 | 1.52 / 1.67 | 768D · 3,072 B/vector | 29.59 MiB | openvlclip==0.7.0 | 60bb3c6a10fa | [b1cc0c8b4149e1b1](raw/b1cc0c8b4149e1b1.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | 73.23% (7,323/10,000) | 91.64% (9,164/10,000) | 264.46 | 1.56 / 1.70 | 768D · 3,072 B/vector | 29.59 MiB | openvlclip==0.7.0 | 60bb3c6a10fa | [3ca50611538ca5da](raw/3ca50611538ca5da.json) |
| DINOv2 ViT-L/14 + dino.txt | 89.14% (8,914/10,000) | 98.13% (9,813/10,000) | 93.22 | 3.55 / 3.71 | 2,048D · 8,192 B/vector | 78.91 MiB | openvlclip==0.6.0 | 28519573b78f | [37ce26e54de28445](../dinov2-dinotxt-v0.6.0/raw/37ce26e54de28445.json) |
| EmbeddingGemma 2 | 62.84% (6,284/10,000) | 84.04% (8,404/10,000) | 10.25 | 4.97 / 5.90 | 768D · 3,072 B/vector | 29.59 MiB | openclip-gemma==0.4.0 | a9909fbe1b15 | [820f9a84ba7355e3](../mrl-openclip-gemma-v0.4.0/raw/820f9a84ba7355e3.json) |
| OpenAI CLIP ViT-B/32 | 62.32% (6,232/10,000) | 87.00% (8,700/10,000) | 723.23 | 0.62 / 0.68 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [c08f92332ebe01a4-2815d62c](../v0.1.0/raw/c08f92332ebe01a4-2815d62c.json) |
| OpenCLIP ViT-B/32 LAION-2B | 75.59% (7,559/10,000) | 93.65% (9,365/10,000) | 773.59 | 0.61 / 0.68 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [a54f8b8dfb17227b-8e3ad59c](../v0.1.0/raw/a54f8b8dfb17227b-8e3ad59c.json) |
| OpenCLIP ViT-L/14 LAION-2B | 83.07% (8,307/10,000) | 96.57% (9,657/10,000) | 106.15 | 1.80 / 1.95 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [5cd03bacdf1e5b06-43791242](../v0.1.0/raw/5cd03bacdf1e5b06-43791242.json) |
| OpenCLIP Multilingual ViT-B/32 | 76.46% (7,646/10,000) | 95.02% (9,502/10,000) | 748.05 | 1.64 / 1.90 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [d4721f7c31f9d23c-e083b306](../v0.1.0/raw/d4721f7c31f9d23c-e083b306.json) |
| OpenVLCLIP Qwen3-VL-Embedding-2B | 57.30% (5,730/10,000) | 79.31% (7,931/10,000) | 4.12 | 7.94 / 8.06 | 2,048D · 8,192 B/vector | 78.91 MiB | openvlclip==0.7.0 | 23b469da312e | [10a1bb028a2f5318](../openvlclip-v0.7.0/raw/10a1bb028a2f5318.json) |
| SigLIP 2 Base/16 224 | 73.25% (7,325/10,000) | 91.71% (9,171/10,000) | 338.32 | 1.52 / 1.67 | not recorded | not recorded | transformers==5.20.1 | 60a61259c95e | [38a8be7d4b997457-8dbcaf34](../v0.1.0/raw/38a8be7d4b997457-8dbcaf34.json) |

### CROSSMODAL3600 · P2 · EN · FP32 · official_first_caption

| Model | T→I R@1 | T→I R@5 | T→I R@10 | I→T R@1 | I→T R@5 | I→T R@10 | Images/s | Texts/s | Peak allocated/reserved (GiB) | Embedding/vector bytes | FP32 image+text payload | Package | Source commit | Raw result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|
| OpenVLCLIP SigLIP 2 Base/16 FixRes | 51.47% (1,853/3,600) | 77.03% (2,773/3,600) | 84.72% (3,050/3,600) | 50.67% (1,824/3,600) | 76.67% (2,760/3,600) | 85.00% (3,060/3,600) | 177.37 | 5.43 | 1.52 / 1.65 | 768D · 3,072 B/vector | 21.09 MiB | openvlclip==0.7.0 | 60bb3c6a10fa | [97b6d69758693acc](raw/97b6d69758693acc.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | 53.86% (1,939/3,600) | 80.08% (2,883/3,600) | 87.56% (3,152/3,600) | 51.06% (1,838/3,600) | 78.28% (2,818/3,600) | 85.75% (3,087/3,600) | 220.99 | 5.28 | 1.56 / 1.69 | 768D · 3,072 B/vector | 21.09 MiB | openvlclip==0.7.0 | 60bb3c6a10fa | [1aab08bebc0b122f](raw/1aab08bebc0b122f.json) |
| DINOv2 ViT-L/14 + dino.txt | 51.50% (1,854/3,600) | 77.31% (2,783/3,600) | 84.64% (3,047/3,600) | 53.03% (1,909/3,600) | 77.33% (2,784/3,600) | 85.44% (3,076/3,600) | 97.47 | 206.63 | 3.55 / 3.69 | 2,048D · 8,192 B/vector | 56.25 MiB | openvlclip==0.6.0 | 28519573b78f | [d8c51bd44857ee04](../dinov2-dinotxt-v0.6.0/raw/d8c51bd44857ee04.json) |
| EmbeddingGemma 2 | 58.61% (2,110/3,600) | 82.06% (2,954/3,600) | 89.19% (3,211/3,600) | 59.50% (2,142/3,600) | 85.17% (3,066/3,600) | 91.28% (3,286/3,600) | 7.52 | 208.51 | 4.29 / 6.31 | 768D · 3,072 B/vector | 21.09 MiB | openclip-gemma==0.4.0 | a9909fbe1b15 | [02b17361f1a3a0ef](../mrl-openclip-gemma-v0.4.0/raw/02b17361f1a3a0ef.json) |
| OpenCLIP Multilingual ViT-B/32 | 47.08% (1,695/3,600) | 74.86% (2,695/3,600) | 83.03% (2,989/3,600) | 48.31% (1,739/3,600) | 76.64% (2,759/3,600) | 85.19% (3,067/3,600) | 324.41 | 148.03 | 1.64 / 1.91 | not recorded | not recorded | open-clip-torch==3.3.0 | 9d170b6071ee | [b5b51e1810e20bc1](../v0.1.0/raw/b5b51e1810e20bc1.json) |
| OpenVLCLIP Qwen3-VL-Embedding-2B | 55.78% (2,008/3,600) | 79.58% (2,865/3,600) | 86.67% (3,120/3,600) | 57.94% (2,086/3,600) | 82.31% (2,963/3,600) | 89.25% (3,213/3,600) | 4.33 | 17.21 | 8.01 / 8.15 | 2,048D · 8,192 B/vector | 56.25 MiB | openvlclip==0.7.0 | 23b469da312e | [026e74e2db5b6144](../openvlclip-v0.7.0/raw/026e74e2db5b6144.json) |
| SigLIP 2 Base/16 224 | 51.47% (1,853/3,600) | 77.03% (2,773/3,600) | 84.72% (3,050/3,600) | 50.67% (1,824/3,600) | 76.67% (2,760/3,600) | 85.00% (3,060/3,600) | 304.09 | 703.03 | 1.52 / 1.65 | not recorded | not recorded | transformers==5.20.1 | 9d170b6071ee | [25b6a90b6a8e0c82](../v0.1.0/raw/25b6a90b6a8e0c82.json) |

### CROSSMODAL3600 · P2 · JA · FP32 · official_first_caption · exploratory

| Model | T→I R@1 | T→I R@5 | T→I R@10 | I→T R@1 | I→T R@5 | I→T R@10 | Images/s | Texts/s | Peak allocated/reserved (GiB) | Embedding/vector bytes | FP32 image+text payload | Package | Source commit | Raw result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|
| OpenVLCLIP SigLIP 2 Base/16 FixRes | 37.86% (1,363/3,600) | 63.33% (2,280/3,600) | 72.47% (2,609/3,600) | 45.06% (1,622/3,600) | 69.97% (2,519/3,600) | 78.92% (2,841/3,600) | 264.44 | 6.86 | 1.52 / 1.65 | 768D · 3,072 B/vector | 21.09 MiB | openvlclip==0.7.0 | 60bb3c6a10fa | [1f68a1ca616f9819-0544d6b3](raw/1f68a1ca616f9819-0544d6b3.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | 36.86% (1,327/3,600) | 62.25% (2,241/3,600) | 72.17% (2,598/3,600) | 40.64% (1,463/3,600) | 66.08% (2,379/3,600) | 74.89% (2,696/3,600) | 173.50 | 5.98 | 1.56 / 1.69 | 768D · 3,072 B/vector | 21.09 MiB | openvlclip==0.7.0 | 60bb3c6a10fa | [f0417644516402eb](raw/f0417644516402eb.json) |
| DINOv2 ViT-L/14 + dino.txt | 2.03% (73/3,600) | 5.14% (185/3,600) | 7.75% (279/3,600) | 3.17% (114/3,600) | 8.00% (288/3,600) | 11.97% (431/3,600) | 96.74 | 211.50 | 3.55 / 3.69 | 2,048D · 8,192 B/vector | 56.25 MiB | openvlclip==0.6.0 | 28519573b78f | [347bf1161cfb1180](../dinov2-dinotxt-v0.6.0/raw/347bf1161cfb1180.json) |
| EmbeddingGemma 2 | 80.14% (2,885/3,600) | 95.03% (3,421/3,600) | 97.50% (3,510/3,600) | 79.61% (2,866/3,600) | 95.28% (3,430/3,600) | 97.61% (3,514/3,600) | 7.54 | 204.05 | 4.29 / 6.31 | 768D · 3,072 B/vector | 21.09 MiB | openclip-gemma==0.4.0 | a9909fbe1b15 | [5e1276cab5ecb4e8](../mrl-openclip-gemma-v0.4.0/raw/5e1276cab5ecb4e8.json) |
| OpenCLIP Multilingual ViT-B/32 | 63.83% (2,298/3,600) | 86.86% (3,127/3,600) | 92.94% (3,346/3,600) | 67.92% (2,445/3,600) | 90.58% (3,261/3,600) | 95.39% (3,434/3,600) | 319.62 | 148.01 | 1.64 / 1.91 | not recorded | not recorded | open-clip-torch==3.3.0 | 9d170b6071ee | [ae4e1863504978e0](../v0.1.0/raw/ae4e1863504978e0.json) |
| OpenVLCLIP Qwen3-VL-Embedding-2B | 77.81% (2,801/3,600) | 93.58% (3,369/3,600) | 96.47% (3,473/3,600) | 72.25% (2,601/3,600) | 91.22% (3,284/3,600) | 95.17% (3,426/3,600) | 4.21 | 15.13 | 8.01 / 8.15 | 2,048D · 8,192 B/vector | 56.25 MiB | openvlclip==0.7.0 | 23b469da312e | [a123010f08059e2f](../openvlclip-v0.7.0/raw/a123010f08059e2f.json) |
| rinna Japanese CLIP ViT-B/16 | 54.00% (1,944/3,600) | 79.86% (2,875/3,600) | 87.22% (3,140/3,600) | 53.94% (1,942/3,600) | 80.67% (2,904/3,600) | 87.69% (3,157/3,600) | 308.22 | 525.83 | 0.86 / 0.97 | not recorded | not recorded | japanese-clip==0.2.0 | 9d170b6071ee | [8bcce7f11a428163](../v0.1.0/raw/8bcce7f11a428163.json) |
| SigLIP 2 Base/16 224 | 37.86% (1,363/3,600) | 63.33% (2,280/3,600) | 72.47% (2,609/3,600) | 45.06% (1,622/3,600) | 69.97% (2,519/3,600) | 78.92% (2,841/3,600) | 297.47 | 688.32 | 1.52 / 1.65 | not recorded | not recorded | transformers==5.20.1 | 9d170b6071ee | [f001095d25c36cb3](../v0.1.0/raw/f001095d25c36cb3.json) |

### Interpretation and comparability notes

- Each recall/accuracy percentage includes the exact hit count and query denominator.
- The payload column is derived as `(image count + text/prompt count) × embedding dimension × 4 bytes`; it is an FP32 vector payload calculation, not a measured on-disk file size, and excludes indexes/metadata.
- Equal embedding dimensions do not make model spaces interchangeable. Storage calculations describe vector payload only, not cross-model index compatibility.
- Legacy raw results that do not record an embedding dimension are left blank for storage calculations rather than inferred from the model name.
- Throughput and memory are actual recorded values. The historical campaigns used the same GPU and FP32 here, but batch sizes, software revisions, and source commits differ; consult the CSV and raw records for details.
- Japanese captions were evaluated as supplied, without translation. Japanese XM3600 retrieval is exploratory and is not evidence of Japanese semantic quality.

## Non-successful runs

No unsuccessful-run JSON records are included in the published raw set. See the machine-readable run status for blocked evaluations and access constraints.

## Validation and reproducibility

Each raw result is schema-validated before aggregation. Per-run JSON records retain status-appropriate provenance. Checkpoint fingerprints, dataset checksums, candidate IDs, and metrics are available only when a run reaches those stages; runs blocked earlier intentionally leave them unset.
- Validation artifacts: [result-lifecycle.json](validation/result-lifecycle.json), [model-comparison-audit.json](validation/model-comparison-audit.json).

## Known limitations

- Official numerical parity applies only to the exact combinations documented in the available validation records. It does not establish parity for other models, languages, datasets, precisions, or retrieval variants.
- Training-data contamination cannot be ruled out unless a model's pretraining corpus is independently verified.
- Dataset-specific access and attribution requirements remain in force; model use remains subject to applicable component terms. See [model-and-dataset-licenses.md](../../docs/model-and-dataset-licenses.md). DINO.txt use in this local benchmark is covered by the specific determination linked above; this is not a commercial-deployment or dataset-rights grant.
- Missing metrics remain missing. Unsupported languages and unavailable datasets are not represented as zero scores.

## Reproduction

### Execute benchmark inference

The following `vlbench run` commands execute model inference and create new raw runs. They do not regenerate this report. Use the recorded clean source revision and review dataset terms before execution.

```powershell
uv sync --locked --group dev --extra cuda --extra openvlclip07
uv run --locked vlbench doctor --device cuda
uv run --locked vlbench run --model openvlclip_siglip2_base --dataset cifar10 --split test --protocol p0 --language en --precision fp32 --device cuda:0 --batch-size 16 --seed 0 --strict-reproducibility --expected-source-commit 60bb3c6a10fa827e190954b4f74c6dd82b74e2ae --yes --output .cache/benchmark-staging/openvlclip-v0.7.0-siglip2-corrected-source
uv run --locked vlbench run --model openvlclip_siglip2_naflex_base --dataset cifar10 --split test --protocol p0 --language en --precision fp32 --device cuda:0 --batch-size 16 --seed 0 --strict-reproducibility --expected-source-commit 60bb3c6a10fa827e190954b4f74c6dd82b74e2ae --yes --output .cache/benchmark-staging/openvlclip-v0.7.0-siglip2-corrected-source
uv run --locked vlbench run --model openvlclip_siglip2_naflex_base --dataset cifar10 --split test --protocol p1 --language en --precision fp32 --device cuda:0 --batch-size 16 --seed 0 --strict-reproducibility --expected-source-commit 60bb3c6a10fa827e190954b4f74c6dd82b74e2ae --yes --output .cache/benchmark-staging/openvlclip-v0.7.0-siglip2-corrected-source
uv run --locked vlbench run --model openvlclip_siglip2_base --dataset cifar10 --split test --protocol p1 --language en --precision fp32 --device cuda:0 --batch-size 16 --seed 0 --strict-reproducibility --expected-source-commit 60bb3c6a10fa827e190954b4f74c6dd82b74e2ae --yes --output .cache/benchmark-staging/openvlclip-v0.7.0-siglip2-corrected-source
uv run --locked vlbench run --model openvlclip_siglip2_naflex_base --dataset cifar100 --split test --protocol p0 --language en --precision fp32 --device cuda:0 --batch-size 16 --seed 0 --strict-reproducibility --expected-source-commit 60bb3c6a10fa827e190954b4f74c6dd82b74e2ae --yes --output .cache/benchmark-staging/openvlclip-v0.7.0-siglip2-corrected-source
uv run --locked vlbench run --model openvlclip_siglip2_base --dataset cifar100 --split test --protocol p0 --language en --precision fp32 --device cuda:0 --batch-size 16 --seed 0 --strict-reproducibility --expected-source-commit 60bb3c6a10fa827e190954b4f74c6dd82b74e2ae --yes --output .cache/benchmark-staging/openvlclip-v0.7.0-siglip2-corrected-source
uv run --locked vlbench run --model openvlclip_siglip2_naflex_base --dataset cifar100 --split test --protocol p1 --language en --precision fp32 --device cuda:0 --batch-size 16 --seed 0 --strict-reproducibility --expected-source-commit 60bb3c6a10fa827e190954b4f74c6dd82b74e2ae --yes --output .cache/benchmark-staging/openvlclip-v0.7.0-siglip2-corrected-source
uv run --locked vlbench run --model openvlclip_siglip2_base --dataset cifar100 --split test --protocol p1 --language en --precision fp32 --device cuda:0 --batch-size 16 --seed 0 --strict-reproducibility --expected-source-commit 60bb3c6a10fa827e190954b4f74c6dd82b74e2ae --yes --output .cache/benchmark-staging/openvlclip-v0.7.0-siglip2-corrected-source
uv run --locked vlbench run --model openvlclip_siglip2_naflex_base --dataset crossmodal3600 --split 'all 3,600 images' --protocol p2 --language en --precision fp32 --device cuda:0 --batch-size 16 --seed 0 --retrieval-variant official_first_caption --strict-reproducibility --expected-source-commit 60bb3c6a10fa827e190954b4f74c6dd82b74e2ae --yes --output .cache/benchmark-staging/openvlclip-v0.7.0-siglip2-corrected-source
uv run --locked vlbench run --model openvlclip_siglip2_base --dataset crossmodal3600 --split 'all 3,600 images' --protocol p2 --language en --precision fp32 --device cuda:0 --batch-size 16 --seed 0 --retrieval-variant official_first_caption --strict-reproducibility --expected-source-commit 60bb3c6a10fa827e190954b4f74c6dd82b74e2ae --yes --output .cache/benchmark-staging/openvlclip-v0.7.0-siglip2-corrected-source
uv run --locked vlbench run --model openvlclip_siglip2_base --dataset crossmodal3600 --split 'all 3,600 images' --protocol p2 --language ja --precision fp32 --device cuda:0 --batch-size 16 --seed 0 --retrieval-variant official_first_caption --strict-reproducibility --expected-source-commit 60bb3c6a10fa827e190954b4f74c6dd82b74e2ae --yes --output .cache/benchmark-staging/openvlclip-v0.7.0-siglip2-corrected-source
uv run --locked vlbench run --model openvlclip_siglip2_naflex_base --dataset crossmodal3600 --split 'all 3,600 images' --protocol p2 --language ja --precision fp32 --device cuda:0 --batch-size 16 --seed 0 --retrieval-variant official_first_caption --strict-reproducibility --expected-source-commit 60bb3c6a10fa827e190954b4f74c6dd82b74e2ae --yes --output .cache/benchmark-staging/openvlclip-v0.7.0-siglip2-corrected-source
```

### Regenerate this report only (no inference)

This command reads existing raw JSON and regenerates summaries, plots, and validation outputs; it does not load model weights or run inference.

```powershell
uv run --locked vlbench report --input results/openvlclip-v0.7.0-siglip2-corrected-source/raw --output results/openvlclip-v0.7.0-siglip2-corrected-source --publishable --lifecycle results/openvlclip-v0.7.0-siglip2-corrected-source/validation/result-lifecycle.json --expected-source-commit 60bb3c6a10fa827e190954b4f74c6dd82b74e2ae --comparison-input results/openvlclip-v0.7.0/raw --comparison-lifecycle results/openvlclip-v0.7.0/validation/result-lifecycle.json --comparison-input results/v0.1.0/raw --comparison-lifecycle results/v0.1.0/validation/result-lifecycle.json --comparison-input results/mrl-openclip-gemma-v0.4.0/raw --comparison-lifecycle results/mrl-openclip-gemma-v0.4.0/validation/result-lifecycle.json --comparison-input results/dinov2-dinotxt-v0.6.0/raw --comparison-lifecycle results/dinov2-dinotxt-v0.6.0/validation/result-lifecycle.json
```

Use the exact model revision, dataset checksum, split, protocol, language, precision, batch size, seed, source commit, and lockfile hash recorded in each raw result when reproducing a measurement.

## Figures

- Classification Accuracy: [PNG](plots/classification_accuracy.png) · [SVG](plots/classification_accuracy.svg)
- Classification Accuracy P1: [PNG](plots/classification_accuracy_p1.png) · [SVG](plots/classification_accuracy_p1.svg)
- Model Comparison Classification Accuracy: [PNG](plots/model_comparison_classification_accuracy.png) · [SVG](plots/model_comparison_classification_accuracy.svg)
- Model Comparison Classification Accuracy P1: [PNG](plots/model_comparison_classification_accuracy_p1.png) · [SVG](plots/model_comparison_classification_accuracy_p1.svg)
- Model Comparison Performance Comparison: [PNG](plots/model_comparison_performance_comparison.png) · [SVG](plots/model_comparison_performance_comparison.svg)
- Model Comparison Performance Comparison P1: [PNG](plots/model_comparison_performance_comparison_p1.png) · [SVG](plots/model_comparison_performance_comparison_p1.svg)
- Model Comparison Xm3600 Official Retrieval: [PNG](plots/model_comparison_xm3600_official_retrieval.png) · [SVG](plots/model_comparison_xm3600_official_retrieval.svg)
- Performance Comparison: [PNG](plots/performance_comparison.png) · [SVG](plots/performance_comparison.svg)
- Performance Comparison P1: [PNG](plots/performance_comparison_p1.png) · [SVG](plots/performance_comparison_p1.svg)
- Xm3600 Language Comparison Official First Caption: [PNG](plots/xm3600_language_comparison_official_first_caption.png) · [SVG](plots/xm3600_language_comparison_official_first_caption.svg)
- Xm3600 Official Retrieval: [PNG](plots/xm3600_official_retrieval.png) · [SVG](plots/xm3600_official_retrieval.svg)
