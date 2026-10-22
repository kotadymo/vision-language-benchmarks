# Vision-Language Benchmark Report

This report is regenerated from schema-validated raw JSON. Only completed runs contribute numeric metrics; failed, skipped, and blocked runs are listed separately. No values are estimated or fabricated.

## Conditions

- Validated result files: 12
- Completed measurements: 6
- Skipped runs: 0
- Failed runs: 0
- Blocked runs: 6
- Source commit(s): 28519573b78f0dcb7ab6515ead02bc6b403a92bd, 53eeb22790fc8f135c3a94d589e2e422f8b6566a
- Runtime profile(s): NVIDIA GeForce RTX 4070 Ti SUPER (17170956288 bytes, driver 610.88, PyTorch 2.14.1+cu130, CUDA 13.0, fp32)
- Protocols and precisions are kept in separate tables and chart files.
- Accuracy and recall values in JSON are fractions; report tables display percentages.
- Checkpoint sizes are pinned weight-file sizes; tokenizer and configuration files may add to the download. Peak memory is reported as allocated / reserved GiB.
- Each completed table row links to its schema-validated raw result JSON.

## Measurement lifecycle

- Frozen benchmark source commit: `28519573b78f0dcb7ab6515ead02bc6b403a92bd`
- Active validated measurements: 6
- Retained historical measurements: 0
- Superseded measurements excluded from primary results: 0
- Failed attempts: 0
- Incomplete attempts: 0
- Invalid measurements: 0
- Blocked runs: 6
- Skipped runs: 0
- Per-run lifecycle assignments are recorded in [result-lifecycle.json](validation/result-lifecycle.json).

## DINO.txt campaign notes

- Completed DINO.txt measurements: 6.
- Preserved blocked DINO.txt attempts: 6; these are separate historical records and contribute no numeric metrics.
- The local scientific-use licensing determination is documented in [dinotxt-license-review.md](../../docs/dinotxt-license-review.md).
- Japanese DINO.txt retrieval is exploratory; the original Japanese captions were evaluated without translation.

## English zero-shot classification

### P0 · FP32

| Model | Embedding dim | Parameters | Checkpoint | Dataset | Split | Images | Classes | Requested / image / text batch | Top-1 (correct/total) | Top-5 (correct/total) | Images/s | Median (ms) | Peak allocated / reserved (GiB) | OOM retries | Raw result |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| DINOv2 ViT-L/14 + dino.txt | 2,048 | 867.8 M | — | CIFAR-10 | test | 10,000 | 10 | 16 / 16 / 16 | 98.24% (9,824/10,000) | 99.95% (9,995/10,000) | 81.0 | 191.24 | 3.55 / 3.71 | 0 | [7d2fd70c52399af1](raw/7d2fd70c52399af1.json) |
| DINOv2 ViT-L/14 + dino.txt | 2,048 | 867.8 M | — | CIFAR-100 | test | 10,000 | 100 | 16 / 16 / 16 | 89.71% (8,971/10,000) | 98.17% (9,817/10,000) | 91.7 | 173.99 | 3.55 / 3.71 | 0 | [f0b3f208c3f8f151](raw/f0b3f208c3f8f151.json) |

### P1 · FP32

| Model | Embedding dim | Parameters | Checkpoint | Dataset | Split | Images | Classes | Requested / image / text batch | Top-1 (correct/total) | Top-5 (correct/total) | Images/s | Median (ms) | Peak allocated / reserved (GiB) | OOM retries | Raw result |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| DINOv2 ViT-L/14 + dino.txt | 2,048 | 867.8 M | — | CIFAR-10 | test | 10,000 | 10 | 16 / 16 / 16 | 98.12% (9,812/10,000) | 99.94% (9,994/10,000) | 92.0 | 173.80 | 3.55 / 3.72 | 0 | [0f885e751a90a8ef](raw/0f885e751a90a8ef.json) |
| DINOv2 ViT-L/14 + dino.txt | 2,048 | 867.8 M | — | CIFAR-100 | test | 10,000 | 100 | 16 / 16 / 16 | 89.14% (8,914/10,000) | 98.13% (9,813/10,000) | 93.2 | 171.52 | 3.55 / 3.71 | 0 | [37ce26e54de28445](raw/37ce26e54de28445.json) |

## Japanese/English official-compatible XM3600 retrieval (`official_first_caption`)

### EN · P2 · FP32

| Model | Embedding dim | Parameters | Checkpoint | Dataset | Split | Language | Retrieval variant | Images / captions | Requested / image / text batch | T→I R@1 (hits/queries) | T→I R@5 (hits/queries) | T→I R@10 (hits/queries) | I→T R@1 (hits/queries) | I→T R@5 (hits/queries) | I→T R@10 (hits/queries) | Images/s | Texts/s | Median (ms) | Peak allocated / reserved (GiB) | OOM retries | Raw result |
|---|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| DINOv2 ViT-L/14 + dino.txt | 2,048 | 867.8 M | — | Crossmodal-3600 (XM3600) | all 3,600 images | en | official_first_caption | 3,600 / 3,600 | 16 / 16 / 16 | 51.50% (1,854/3,600) | 77.31% (2,783/3,600) | 84.64% (3,047/3,600) | 53.03% (1,909/3,600) | 77.33% (2,784/3,600) | 85.44% (3,076/3,600) | 97.5 | 206.6 | 118.70 | 3.55 / 3.69 | 0 | [d8c51bd44857ee04](raw/d8c51bd44857ee04.json) |

### JA · P2 · FP32

| Model | Embedding dim | Parameters | Checkpoint | Dataset | Split | Language | Retrieval variant | Images / captions | Requested / image / text batch | T→I R@1 (hits/queries) | T→I R@5 (hits/queries) | T→I R@10 (hits/queries) | I→T R@1 (hits/queries) | I→T R@5 (hits/queries) | I→T R@10 (hits/queries) | Images/s | Texts/s | Median (ms) | Peak allocated / reserved (GiB) | OOM retries | Raw result |
|---|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| DINOv2 ViT-L/14 + dino.txt | 2,048 | 867.8 M | — | Crossmodal-3600 (XM3600) | all 3,600 images | ja | official_first_caption | 3,600 / 3,600 | 16 / 16 / 16 | 2.03% (73/3,600) | 5.14% (185/3,600) | 7.75% (279/3,600) | 3.17% (114/3,600) | 8.00% (288/3,600) | 11.97% (431/3,600) | 96.7 | 211.5 | 124.28 | 3.55 / 3.69 | 0 | [347bf1161cfb1180](raw/347bf1161cfb1180.json) |

## Japanese zero-shot classification

No completed measured runs are available.

## XM3600 all-caption retrieval (`all_captions`)

No completed measured runs are available.

## Matched English/Japanese retrieval

### Crossmodal-3600 (XM3600) · P2 · FP32 · official_first_caption

| Model | Embedding dim | Split | Images | Captions (EN / JA) | T→I R@1 (EN → JA; Δ pp) | T→I R@5 (EN → JA; Δ pp) | T→I R@10 (EN → JA; Δ pp) | I→T R@1 (EN → JA; Δ pp) | I→T R@5 (EN → JA; Δ pp) | I→T R@10 (EN → JA; Δ pp) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DINOv2 ViT-L/14 + dino.txt | 2,048 | all 3,600 images | 3,600 | 3,600 / 3,600 | 51.50% → 2.03% (-49.47 pp) | 77.31% → 5.14% (-72.17 pp) | 84.64% → 7.75% (-76.89 pp) | 53.03% → 3.17% (-49.86 pp) | 77.33% → 8.00% (-69.33 pp) | 85.44% → 11.97% (-73.47 pp) |

## Model throughput and VRAM

Throughput is computed from model inference only; model downloads, initialization, dataset preparation, and report generation are excluded.

| Model | Dataset | Task | Language | Protocol | Retrieval variant | Precision | Images/s | Texts/s | Median (ms) | Peak allocated / reserved (GiB) | Raw result |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---|
| DINOv2 ViT-L/14 + dino.txt | CIFAR-10 | classification | en | p0 | — | fp32 | 81.0 | 168.8 | 191.24 | 3.55 / 3.71 | [7d2fd70c52399af1](raw/7d2fd70c52399af1.json) |
| DINOv2 ViT-L/14 + dino.txt | CIFAR-10 | classification | en | p1 | — | fp32 | 92.0 | 183.1 | 173.80 | 3.55 / 3.72 | [0f885e751a90a8ef](raw/0f885e751a90a8ef.json) |
| DINOv2 ViT-L/14 + dino.txt | CIFAR-100 | classification | en | p0 | — | fp32 | 91.7 | 203.6 | 173.99 | 3.55 / 3.71 | [f0b3f208c3f8f151](raw/f0b3f208c3f8f151.json) |
| DINOv2 ViT-L/14 + dino.txt | CIFAR-100 | classification | en | p1 | — | fp32 | 93.2 | 205.5 | 171.52 | 3.55 / 3.71 | [37ce26e54de28445](raw/37ce26e54de28445.json) |
| DINOv2 ViT-L/14 + dino.txt | Crossmodal-3600 (XM3600) | retrieval | en | p2 | official_first_caption | fp32 | 97.5 | 206.6 | 118.70 | 3.55 / 3.69 | [d8c51bd44857ee04](raw/d8c51bd44857ee04.json) |
| DINOv2 ViT-L/14 + dino.txt | Crossmodal-3600 (XM3600) | retrieval | ja | p2 | official_first_caption | fp32 | 96.7 | 211.5 | 124.28 | 3.55 / 3.69 | [347bf1161cfb1180](raw/347bf1161cfb1180.json) |

## Matched historical model comparisons

Rows are matched to each new DINO.txt run by dataset checksum, split, task, protocol, language, precision, sample/candidate counts, and retrieval variant. Only lifecycle-eligible completed historical results are included.
Machine-readable rows: [model-comparison.csv](model-comparison.csv); selection details: [model-comparison-audit.json](validation/model-comparison-audit.json).

### CIFAR10 · P0 · EN · FP32

| Model | Top-1 (correct/total) | Top-5 (correct/total) | Images/s | Peak allocated/reserved (GiB) | Embedding/vector bytes | FP32 image+text payload | Package | Source commit | Raw result |
|---|---:|---:|---:|---:|---|---:|---|---|---|
| DINOv2 ViT-L/14 + dino.txt | 98.24% (9,824/10,000) | 99.95% (9,995/10,000) | 80.99 | 3.55 / 3.71 | 2,048D · 8,192 B/vector | 79.53 MiB | openvlclip==0.6.0 | 28519573b78f | [7d2fd70c52399af1](raw/7d2fd70c52399af1.json) |
| EmbeddingGemma 2 | 94.66% (9,466/10,000) | 99.84% (9,984/10,000) | 10.21 | 4.97 / 5.90 | 768D · 3,072 B/vector | 29.82 MiB | openclip-gemma==0.4.0 | a9909fbe1b15 | [b8214ecb970362c9](../mrl-openclip-gemma-v0.4.0/raw/b8214ecb970362c9.json) |
| OpenAI CLIP ViT-B/32 | 89.83% (8,983/10,000) | 99.61% (9,961/10,000) | 745.47 | 0.62 / 0.68 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [35cef7e32f8afb13](../v0.1.0/raw/35cef7e32f8afb13.json) |
| OpenCLIP ViT-B/32 LAION-2B | 93.56% (9,356/10,000) | 99.81% (9,981/10,000) | 749.63 | 0.61 / 0.68 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [649c7b44763ebc72](../v0.1.0/raw/649c7b44763ebc72.json) |
| OpenCLIP ViT-L/14 LAION-2B | 96.61% (9,661/10,000) | 99.85% (9,985/10,000) | 106.59 | 1.80 / 1.95 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [c8faef3bbd4c2779](../v0.1.0/raw/c8faef3bbd4c2779.json) |
| OpenCLIP Multilingual ViT-B/32 | 93.64% (9,364/10,000) | 99.88% (9,988/10,000) | 701.99 | 1.64 / 1.90 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [931adb237a72e7b9](../v0.1.0/raw/931adb237a72e7b9.json) |
| SigLIP 2 Base/16 224 | 93.62% (9,362/10,000) | 99.85% (9,985/10,000) | 338.26 | 1.52 / 1.67 | not recorded | not recorded | transformers==5.20.1 | 60a61259c95e | [38b1a8be68ae509c](../v0.1.0/raw/38b1a8be68ae509c.json) |

### CIFAR10 · P1 · EN · FP32

| Model | Top-1 (correct/total) | Top-5 (correct/total) | Images/s | Peak allocated/reserved (GiB) | Embedding/vector bytes | FP32 image+text payload | Package | Source commit | Raw result |
|---|---:|---:|---:|---:|---|---:|---|---|---|
| DINOv2 ViT-L/14 + dino.txt | 98.12% (9,812/10,000) | 99.94% (9,994/10,000) | 92.03 | 3.55 / 3.72 | 2,048D · 8,192 B/vector | 78.20 MiB | openvlclip==0.6.0 | 28519573b78f | [0f885e751a90a8ef](raw/0f885e751a90a8ef.json) |
| EmbeddingGemma 2 | 92.58% (9,258/10,000) | 99.64% (9,964/10,000) | 10.18 | 4.97 / 5.88 | 768D · 3,072 B/vector | 29.33 MiB | openclip-gemma==0.4.0 | a9909fbe1b15 | [6950817bdd99d973](../mrl-openclip-gemma-v0.4.0/raw/6950817bdd99d973.json) |
| OpenAI CLIP ViT-B/32 | 88.30% (8,830/10,000) | 99.24% (9,924/10,000) | 688.74 | 0.62 / 0.68 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [c542f4a0ea8ac9a8](../v0.1.0/raw/c542f4a0ea8ac9a8.json) |
| OpenCLIP ViT-B/32 LAION-2B | 93.68% (9,368/10,000) | 99.84% (9,984/10,000) | 717.66 | 0.61 / 0.66 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [c3350bf4370a58b8](../v0.1.0/raw/c3350bf4370a58b8.json) |
| OpenCLIP ViT-L/14 LAION-2B | 96.92% (9,692/10,000) | 99.93% (9,993/10,000) | 106.76 | 1.80 / 1.90 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [9c075a8286dd1604](../v0.1.0/raw/9c075a8286dd1604.json) |
| OpenCLIP Multilingual ViT-B/32 | 93.54% (9,354/10,000) | 99.86% (9,986/10,000) | 739.21 | 1.54 / 1.71 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [013541e887452604](../v0.1.0/raw/013541e887452604.json) |
| SigLIP 2 Base/16 224 | 93.85% (9,385/10,000) | 99.88% (9,988/10,000) | 336.90 | 1.52 / 1.65 | not recorded | not recorded | transformers==5.20.1 | 60a61259c95e | [ddcae56da141ed7f](../v0.1.0/raw/ddcae56da141ed7f.json) |

### CIFAR100 · P0 · EN · FP32

| Model | Top-1 (correct/total) | Top-5 (correct/total) | Images/s | Peak allocated/reserved (GiB) | Embedding/vector bytes | FP32 image+text payload | Package | Source commit | Raw result |
|---|---:|---:|---:|---:|---|---:|---|---|---|
| DINOv2 ViT-L/14 + dino.txt | 89.71% (8,971/10,000) | 98.17% (9,817/10,000) | 91.73 | 3.55 / 3.71 | 2,048D · 8,192 B/vector | 92.19 MiB | openvlclip==0.6.0 | 28519573b78f | [f0b3f208c3f8f151](raw/f0b3f208c3f8f151.json) |
| EmbeddingGemma 2 | 76.08% (7,608/10,000) | 92.92% (9,292/10,000) | 10.19 | 4.97 / 5.94 | 768D · 3,072 B/vector | 34.57 MiB | openclip-gemma==0.4.0 | a9909fbe1b15 | [b08c9f0fbe719961](../mrl-openclip-gemma-v0.4.0/raw/b08c9f0fbe719961.json) |
| OpenAI CLIP ViT-B/32 | 64.20% (6,420/10,000) | 88.73% (8,873/10,000) | 668.75 | 0.62 / 0.68 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [5a8886b0fab54729-30d1b401](../v0.1.0/raw/5a8886b0fab54729-30d1b401.json) |
| OpenCLIP ViT-B/32 LAION-2B | 75.56% (7,556/10,000) | 93.87% (9,387/10,000) | 701.07 | 0.61 / 0.68 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [35dfb13b0aa46c57-54e65389](../v0.1.0/raw/35dfb13b0aa46c57-54e65389.json) |
| OpenCLIP ViT-L/14 LAION-2B | 83.34% (8,334/10,000) | 96.63% (9,663/10,000) | 106.97 | 1.80 / 1.94 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [1de2d2b7854b5e80-30c3f626](../v0.1.0/raw/1de2d2b7854b5e80-30c3f626.json) |
| OpenCLIP Multilingual ViT-B/32 | 76.55% (7,655/10,000) | 94.99% (9,499/10,000) | 750.93 | 1.64 / 1.90 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [e4c4922a8b419fc2-ac4a43b4](../v0.1.0/raw/e4c4922a8b419fc2-ac4a43b4.json) |
| SigLIP 2 Base/16 224 | 73.68% (7,368/10,000) | 91.89% (9,189/10,000) | 342.26 | 1.52 / 1.66 | not recorded | not recorded | transformers==5.20.1 | 60a61259c95e | [d59181f45a441ab9-1a33071f](../v0.1.0/raw/d59181f45a441ab9-1a33071f.json) |

### CIFAR100 · P1 · EN · FP32

| Model | Top-1 (correct/total) | Top-5 (correct/total) | Images/s | Peak allocated/reserved (GiB) | Embedding/vector bytes | FP32 image+text payload | Package | Source commit | Raw result |
|---|---:|---:|---:|---:|---|---:|---|---|---|
| DINOv2 ViT-L/14 + dino.txt | 89.14% (8,914/10,000) | 98.13% (9,813/10,000) | 93.22 | 3.55 / 3.71 | 2,048D · 8,192 B/vector | 78.91 MiB | openvlclip==0.6.0 | 28519573b78f | [37ce26e54de28445](raw/37ce26e54de28445.json) |
| EmbeddingGemma 2 | 62.84% (6,284/10,000) | 84.04% (8,404/10,000) | 10.25 | 4.97 / 5.90 | 768D · 3,072 B/vector | 29.59 MiB | openclip-gemma==0.4.0 | a9909fbe1b15 | [820f9a84ba7355e3](../mrl-openclip-gemma-v0.4.0/raw/820f9a84ba7355e3.json) |
| OpenAI CLIP ViT-B/32 | 62.32% (6,232/10,000) | 87.00% (8,700/10,000) | 723.23 | 0.62 / 0.68 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [c08f92332ebe01a4-2815d62c](../v0.1.0/raw/c08f92332ebe01a4-2815d62c.json) |
| OpenCLIP ViT-B/32 LAION-2B | 75.59% (7,559/10,000) | 93.65% (9,365/10,000) | 773.59 | 0.61 / 0.68 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [a54f8b8dfb17227b-8e3ad59c](../v0.1.0/raw/a54f8b8dfb17227b-8e3ad59c.json) |
| OpenCLIP ViT-L/14 LAION-2B | 83.07% (8,307/10,000) | 96.57% (9,657/10,000) | 106.15 | 1.80 / 1.95 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [5cd03bacdf1e5b06-43791242](../v0.1.0/raw/5cd03bacdf1e5b06-43791242.json) |
| OpenCLIP Multilingual ViT-B/32 | 76.46% (7,646/10,000) | 95.02% (9,502/10,000) | 748.05 | 1.64 / 1.90 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [d4721f7c31f9d23c-e083b306](../v0.1.0/raw/d4721f7c31f9d23c-e083b306.json) |
| SigLIP 2 Base/16 224 | 73.25% (7,325/10,000) | 91.71% (9,171/10,000) | 338.32 | 1.52 / 1.67 | not recorded | not recorded | transformers==5.20.1 | 60a61259c95e | [38a8be7d4b997457-8dbcaf34](../v0.1.0/raw/38a8be7d4b997457-8dbcaf34.json) |

### CROSSMODAL3600 · P2 · EN · FP32 · official_first_caption

| Model | T→I R@1 | T→I R@5 | T→I R@10 | I→T R@1 | I→T R@5 | I→T R@10 | Images/s | Texts/s | Peak allocated/reserved (GiB) | Embedding/vector bytes | FP32 image+text payload | Package | Source commit | Raw result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|
| DINOv2 ViT-L/14 + dino.txt | 51.50% (1,854/3,600) | 77.31% (2,783/3,600) | 84.64% (3,047/3,600) | 53.03% (1,909/3,600) | 77.33% (2,784/3,600) | 85.44% (3,076/3,600) | 97.47 | 206.63 | 3.55 / 3.69 | 2,048D · 8,192 B/vector | 56.25 MiB | openvlclip==0.6.0 | 28519573b78f | [d8c51bd44857ee04](raw/d8c51bd44857ee04.json) |
| EmbeddingGemma 2 | 58.61% (2,110/3,600) | 82.06% (2,954/3,600) | 89.19% (3,211/3,600) | 59.50% (2,142/3,600) | 85.17% (3,066/3,600) | 91.28% (3,286/3,600) | 7.52 | 208.51 | 4.29 / 6.31 | 768D · 3,072 B/vector | 21.09 MiB | openclip-gemma==0.4.0 | a9909fbe1b15 | [02b17361f1a3a0ef](../mrl-openclip-gemma-v0.4.0/raw/02b17361f1a3a0ef.json) |
| OpenCLIP Multilingual ViT-B/32 | 47.08% (1,695/3,600) | 74.86% (2,695/3,600) | 83.03% (2,989/3,600) | 48.31% (1,739/3,600) | 76.64% (2,759/3,600) | 85.19% (3,067/3,600) | 324.41 | 148.03 | 1.64 / 1.91 | not recorded | not recorded | open-clip-torch==3.3.0 | 9d170b6071ee | [b5b51e1810e20bc1](../v0.1.0/raw/b5b51e1810e20bc1.json) |
| SigLIP 2 Base/16 224 | 51.47% (1,853/3,600) | 77.03% (2,773/3,600) | 84.72% (3,050/3,600) | 50.67% (1,824/3,600) | 76.67% (2,760/3,600) | 85.00% (3,060/3,600) | 304.09 | 703.03 | 1.52 / 1.65 | not recorded | not recorded | transformers==5.20.1 | 9d170b6071ee | [25b6a90b6a8e0c82](../v0.1.0/raw/25b6a90b6a8e0c82.json) |

### CROSSMODAL3600 · P2 · JA · FP32 · official_first_caption · exploratory

| Model | T→I R@1 | T→I R@5 | T→I R@10 | I→T R@1 | I→T R@5 | I→T R@10 | Images/s | Texts/s | Peak allocated/reserved (GiB) | Embedding/vector bytes | FP32 image+text payload | Package | Source commit | Raw result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|
| DINOv2 ViT-L/14 + dino.txt | 2.03% (73/3,600) | 5.14% (185/3,600) | 7.75% (279/3,600) | 3.17% (114/3,600) | 8.00% (288/3,600) | 11.97% (431/3,600) | 96.74 | 211.50 | 3.55 / 3.69 | 2,048D · 8,192 B/vector | 56.25 MiB | openvlclip==0.6.0 | 28519573b78f | [347bf1161cfb1180](raw/347bf1161cfb1180.json) |
| EmbeddingGemma 2 | 80.14% (2,885/3,600) | 95.03% (3,421/3,600) | 97.50% (3,510/3,600) | 79.61% (2,866/3,600) | 95.28% (3,430/3,600) | 97.61% (3,514/3,600) | 7.54 | 204.05 | 4.29 / 6.31 | 768D · 3,072 B/vector | 21.09 MiB | openclip-gemma==0.4.0 | a9909fbe1b15 | [5e1276cab5ecb4e8](../mrl-openclip-gemma-v0.4.0/raw/5e1276cab5ecb4e8.json) |
| OpenCLIP Multilingual ViT-B/32 | 63.83% (2,298/3,600) | 86.86% (3,127/3,600) | 92.94% (3,346/3,600) | 67.92% (2,445/3,600) | 90.58% (3,261/3,600) | 95.39% (3,434/3,600) | 319.62 | 148.01 | 1.64 / 1.91 | not recorded | not recorded | open-clip-torch==3.3.0 | 9d170b6071ee | [ae4e1863504978e0](../v0.1.0/raw/ae4e1863504978e0.json) |
| rinna Japanese CLIP ViT-B/16 | 54.00% (1,944/3,600) | 79.86% (2,875/3,600) | 87.22% (3,140/3,600) | 53.94% (1,942/3,600) | 80.67% (2,904/3,600) | 87.69% (3,157/3,600) | 308.22 | 525.83 | 0.86 / 0.97 | not recorded | not recorded | japanese-clip==0.2.0 | 9d170b6071ee | [8bcce7f11a428163](../v0.1.0/raw/8bcce7f11a428163.json) |
| SigLIP 2 Base/16 224 | 37.86% (1,363/3,600) | 63.33% (2,280/3,600) | 72.47% (2,609/3,600) | 45.06% (1,622/3,600) | 69.97% (2,519/3,600) | 78.92% (2,841/3,600) | 297.47 | 688.32 | 1.52 / 1.65 | not recorded | not recorded | transformers==5.20.1 | 9d170b6071ee | [f001095d25c36cb3](../v0.1.0/raw/f001095d25c36cb3.json) |

### Interpretation and comparability notes

- Each recall/accuracy percentage includes the exact hit count and query denominator.
- The payload column is derived as `(image count + text/prompt count) × embedding dimension × 4 bytes`; it is an FP32 vector payload calculation, not a measured on-disk file size, and excludes indexes/metadata.
- DINO.txt's 2048D vectors require 8,192 bytes per vector in FP32; the matched EmbeddingGemma 2 768D vectors require 3,072 bytes. Their embedding spaces are different and are not interchangeable.
- Legacy raw results that do not record an embedding dimension are left blank for storage calculations rather than inferred from the model name.
- Throughput and memory are actual recorded values. The historical campaigns used the same GPU and FP32 here, but batch sizes, software revisions, and source commits differ; consult the CSV and raw records for details.
- Japanese captions were evaluated as supplied, without translation. DINO.txt Japanese retrieval is exploratory and is not evidence of Japanese semantic quality.

## Non-successful runs

### Blocked

| Model | Dataset | Language | Protocol | Retrieval variant | Model weight license | Reason | Run ID |
|---|---|---|---|---|---|---|---|
| DINOv2 ViT-L/14 + dino.txt | CIFAR-10 | en | p0 |  | blocked | OpenVLCLIP v0.6.0 does not state a license for the external dino.txt Vision Head and Text Encoder weights. The package and DINOv2 backbone licenses do not establish rights for those separately distributed weights. | [a68a143e2e19b0d5](raw/a68a143e2e19b0d5.json) |
| DINOv2 ViT-L/14 + dino.txt | CIFAR-10 | en | p1 |  | blocked | OpenVLCLIP v0.6.0 does not state a license for the external dino.txt Vision Head and Text Encoder weights. The package and DINOv2 backbone licenses do not establish rights for those separately distributed weights. | [a84a8a06e04ef7ed](raw/a84a8a06e04ef7ed.json) |
| DINOv2 ViT-L/14 + dino.txt | CIFAR-100 | en | p0 |  | blocked | OpenVLCLIP v0.6.0 does not state a license for the external dino.txt Vision Head and Text Encoder weights. The package and DINOv2 backbone licenses do not establish rights for those separately distributed weights. | [c84d9fc9038d5f29](raw/c84d9fc9038d5f29.json) |
| DINOv2 ViT-L/14 + dino.txt | CIFAR-100 | en | p1 |  | blocked | OpenVLCLIP v0.6.0 does not state a license for the external dino.txt Vision Head and Text Encoder weights. The package and DINOv2 backbone licenses do not establish rights for those separately distributed weights. | [984949e7d7414e90](raw/984949e7d7414e90.json) |
| DINOv2 ViT-L/14 + dino.txt | Crossmodal-3600 (XM3600) | en | p2 | official_first_caption | blocked | OpenVLCLIP v0.6.0 does not state a license for the external dino.txt Vision Head and Text Encoder weights. The package and DINOv2 backbone licenses do not establish rights for those separately distributed weights. | [7a9cfe6313eb4e28](raw/7a9cfe6313eb4e28.json) |
| DINOv2 ViT-L/14 + dino.txt | Crossmodal-3600 (XM3600) | ja | p2 | official_first_caption | blocked | OpenVLCLIP v0.6.0 does not state a license for the external dino.txt Vision Head and Text Encoder weights. The package and DINOv2 backbone licenses do not establish rights for those separately distributed weights. | [0cbffc061dfc6663](raw/0cbffc061dfc6663.json) |

## Validation and reproducibility

Each raw result is schema-validated before aggregation. Per-run JSON records retain status-appropriate provenance. Checkpoint fingerprints, dataset checksums, candidate IDs, and metrics are available only when a run reaches those stages; runs blocked earlier intentionally leave them unset.
- Validation artifacts: [checkpoint-integrity.json](validation/checkpoint-integrity.json), [result-lifecycle.json](validation/result-lifecycle.json), [model-comparison-audit.json](validation/model-comparison-audit.json), [dinotxt-cuda-smoke.json](validation/dinotxt-cuda-smoke.json).

## Known limitations

- Official numerical parity applies only to the exact combinations documented in the available validation records. It does not establish parity for other models, languages, datasets, precisions, or retrieval variants.
- Training-data contamination cannot be ruled out unless a model's pretraining corpus is independently verified.
- Dataset-specific access and attribution requirements remain in force; model use remains subject to applicable component terms. See [model-and-dataset-licenses.md](../../docs/model-and-dataset-licenses.md). DINO.txt use in this local benchmark is covered by the specific determination linked above; this is not a commercial-deployment or dataset-rights grant.
- Missing metrics remain missing. Unsupported languages and unavailable datasets are not represented as zero scores.

## Reproduction

### Execute benchmark inference

The following `vlbench run` commands execute model inference and create new raw runs. They do not regenerate this report. Use the recorded clean source revision and review dataset terms before execution.

```powershell
uv sync --locked --group dev --extra cuda --extra dinotxt
uv run --locked vlbench doctor --device cuda
uv run --locked vlbench run --model dinov2_dinotxt --dataset cifar10 --split test --protocol p0 --language en --precision fp32 --device cuda --batch-size 16 --seed 0 --strict-reproducibility --expected-source-commit 28519573b78f0dcb7ab6515ead02bc6b403a92bd --yes --output .cache/benchmark-staging/dinov2-dinotxt-v0.6.0
uv run --locked vlbench run --model dinov2_dinotxt --dataset cifar10 --split test --protocol p1 --language en --precision fp32 --device cuda --batch-size 16 --seed 0 --strict-reproducibility --expected-source-commit 28519573b78f0dcb7ab6515ead02bc6b403a92bd --yes --output .cache/benchmark-staging/dinov2-dinotxt-v0.6.0
uv run --locked vlbench run --model dinov2_dinotxt --dataset cifar100 --split test --protocol p0 --language en --precision fp32 --device cuda --batch-size 16 --seed 0 --strict-reproducibility --expected-source-commit 28519573b78f0dcb7ab6515ead02bc6b403a92bd --yes --output .cache/benchmark-staging/dinov2-dinotxt-v0.6.0
uv run --locked vlbench run --model dinov2_dinotxt --dataset cifar100 --split test --protocol p1 --language en --precision fp32 --device cuda --batch-size 16 --seed 0 --strict-reproducibility --expected-source-commit 28519573b78f0dcb7ab6515ead02bc6b403a92bd --yes --output .cache/benchmark-staging/dinov2-dinotxt-v0.6.0
uv run --locked vlbench run --model dinov2_dinotxt --dataset crossmodal3600 --split 'all 3,600 images' --protocol p2 --language en --precision fp32 --device cuda --batch-size 16 --seed 0 --retrieval-variant official_first_caption --strict-reproducibility --expected-source-commit 28519573b78f0dcb7ab6515ead02bc6b403a92bd --yes --output .cache/benchmark-staging/dinov2-dinotxt-v0.6.0
uv run --locked vlbench run --model dinov2_dinotxt --dataset crossmodal3600 --split 'all 3,600 images' --protocol p2 --language ja --precision fp32 --device cuda --batch-size 16 --seed 0 --retrieval-variant official_first_caption --strict-reproducibility --expected-source-commit 28519573b78f0dcb7ab6515ead02bc6b403a92bd --yes --output .cache/benchmark-staging/dinov2-dinotxt-v0.6.0
```

### Regenerate this report only (no inference)

This command reads existing raw JSON and regenerates summaries, plots, and validation outputs; it does not load model weights or run inference.

```powershell
uv run --locked vlbench report --input results/dinov2-dinotxt-v0.6.0/raw --output results/dinov2-dinotxt-v0.6.0 --publishable --lifecycle results/dinov2-dinotxt-v0.6.0/validation/result-lifecycle.json --expected-source-commit 28519573b78f0dcb7ab6515ead02bc6b403a92bd --comparison-input results/v0.1.0/raw --comparison-lifecycle results/v0.1.0/validation/result-lifecycle.json --comparison-input results/mrl-openclip-gemma-v0.4.0/raw --comparison-lifecycle results/mrl-openclip-gemma-v0.4.0/validation/result-lifecycle.json
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
