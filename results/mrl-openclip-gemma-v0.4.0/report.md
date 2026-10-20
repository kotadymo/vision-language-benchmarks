# Vision-Language Benchmark Report

This report is regenerated from schema-validated raw JSON. Only completed runs contribute numeric metrics; failed, skipped, and blocked runs are listed separately. No values are estimated or fabricated.

## Conditions

- Validated result files: 24
- Completed measurements: 24
- Skipped runs: 0
- Failed runs: 0
- Blocked runs: 0
- Source commit(s): a9909fbe1b15dc66b44bca51a41651f8f3a5d30c
- Runtime profile(s): NVIDIA GeForce RTX 4070 Ti SUPER (17170956288 bytes, driver 610.88, PyTorch 2.14.1+cu130, CUDA 13.0, fp32)
- Protocols and precisions are kept in separate tables and chart files.
- Accuracy and recall values in JSON are fractions; report tables display percentages.
- Checkpoint sizes are pinned weight-file sizes; tokenizer and configuration files may add to the download. Peak memory is reported as allocated / reserved GiB.
- Each completed table row links to its schema-validated raw result JSON.

## Measurement lifecycle

- Frozen benchmark source commit: `a9909fbe1b15dc66b44bca51a41651f8f3a5d30c`
- Active validated measurements: 24
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

| Model | Embedding dim | Parameters | Checkpoint | Dataset | Split | Images | Classes | Requested / image / text batch | Top-1 | Top-5 | Images/s | Median (ms) | Peak allocated / reserved (GiB) | OOM retries | Raw result |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| EmbeddingGemma 2 | 256 | 744.4 M | 1.39 GiB | CIFAR-100 | test | 10,000 | 100 | 16 / 16 / 16 | 74.98% | 92.33% | 10.2 | 1566.04 | 4.97 / 5.94 | 0 | [1466f26677032f18](raw/1466f26677032f18.json) |
| EmbeddingGemma 2 | 512 | 744.4 M | 1.39 GiB | CIFAR-10 | test | 10,000 | 10 | 16 / 16 / 16 | 94.28% | 99.78% | 10.2 | 1566.49 | 4.97 / 5.90 | 0 | [3835a771df7c4084](raw/3835a771df7c4084.json) |
| EmbeddingGemma 2 | 512 | 744.4 M | 1.39 GiB | CIFAR-100 | test | 10,000 | 100 | 16 / 16 / 16 | 75.57% | 92.68% | 10.2 | 1566.45 | 4.97 / 5.94 | 0 | [9ab2047e95be087e](raw/9ab2047e95be087e.json) |
| EmbeddingGemma 2 | 128 | 744.4 M | 1.39 GiB | CIFAR-100 | test | 10,000 | 100 | 16 / 16 / 16 | 66.74% | 87.44% | 10.2 | 1560.57 | 4.97 / 5.94 | 0 | [a7fde2561ffbbfc8](raw/a7fde2561ffbbfc8.json) |
| EmbeddingGemma 2 | 768 | 744.4 M | 1.39 GiB | CIFAR-100 | test | 10,000 | 100 | 16 / 16 / 16 | 76.08% | 92.92% | 10.2 | 1566.77 | 4.97 / 5.94 | 0 | [b08c9f0fbe719961](raw/b08c9f0fbe719961.json) |
| EmbeddingGemma 2 | 768 | 744.4 M | 1.39 GiB | CIFAR-10 | test | 10,000 | 10 | 16 / 16 / 16 | 94.66% | 99.84% | 10.2 | 1565.56 | 4.97 / 5.90 | 0 | [b8214ecb970362c9](raw/b8214ecb970362c9.json) |
| EmbeddingGemma 2 | 128 | 744.4 M | 1.39 GiB | CIFAR-10 | test | 10,000 | 10 | 16 / 16 / 16 | 90.35% | 99.62% | 10.2 | 1568.36 | 4.97 / 5.90 | 0 | [cf413596e88f09f8](raw/cf413596e88f09f8.json) |
| EmbeddingGemma 2 | 256 | 744.4 M | 1.39 GiB | CIFAR-10 | test | 10,000 | 10 | 16 / 16 / 16 | 93.82% | 99.74% | 10.2 | 1569.68 | 4.97 / 5.90 | 0 | [e639f53a1ac6b75e](raw/e639f53a1ac6b75e.json) |

### P1 · FP32

| Model | Embedding dim | Parameters | Checkpoint | Dataset | Split | Images | Classes | Requested / image / text batch | Top-1 | Top-5 | Images/s | Median (ms) | Peak allocated / reserved (GiB) | OOM retries | Raw result |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| EmbeddingGemma 2 | 128 | 744.4 M | 1.39 GiB | CIFAR-100 | test | 10,000 | 100 | 16 / 16 / 16 | 44.76% | 71.75% | 10.2 | 1568.62 | 4.97 / 5.90 | 0 | [22ace06120892265](raw/22ace06120892265.json) |
| EmbeddingGemma 2 | 512 | 744.4 M | 1.39 GiB | CIFAR-10 | test | 10,000 | 10 | 16 / 16 / 16 | 90.90% | 99.68% | 10.2 | 1567.66 | 4.97 / 5.88 | 0 | [4c5567e303a9f771](raw/4c5567e303a9f771.json) |
| EmbeddingGemma 2 | 768 | 744.4 M | 1.39 GiB | CIFAR-10 | test | 10,000 | 10 | 16 / 16 / 16 | 92.58% | 99.64% | 10.2 | 1569.65 | 4.97 / 5.88 | 0 | [6950817bdd99d973](raw/6950817bdd99d973.json) |
| EmbeddingGemma 2 | 768 | 744.4 M | 1.39 GiB | CIFAR-100 | test | 10,000 | 100 | 16 / 16 / 16 | 62.84% | 84.04% | 10.3 | 1559.34 | 4.97 / 5.90 | 0 | [820f9a84ba7355e3](raw/820f9a84ba7355e3.json) |
| EmbeddingGemma 2 | 512 | 744.4 M | 1.39 GiB | CIFAR-100 | test | 10,000 | 100 | 16 / 16 / 16 | 62.17% | 83.17% | 10.2 | 1562.84 | 4.97 / 5.90 | 0 | [8969fe62960b874c](raw/8969fe62960b874c.json) |
| EmbeddingGemma 2 | 256 | 744.4 M | 1.39 GiB | CIFAR-100 | test | 10,000 | 100 | 16 / 16 / 16 | 61.01% | 83.18% | 10.2 | 1561.61 | 4.97 / 5.90 | 0 | [948187d246867b59](raw/948187d246867b59.json) |
| EmbeddingGemma 2 | 256 | 744.4 M | 1.39 GiB | CIFAR-10 | test | 10,000 | 10 | 16 / 16 / 16 | 88.14% | 99.17% | 10.2 | 1567.69 | 4.97 / 5.88 | 0 | [c44690decef67d24](raw/c44690decef67d24.json) |
| EmbeddingGemma 2 | 128 | 744.4 M | 1.39 GiB | CIFAR-10 | test | 10,000 | 10 | 16 / 16 / 16 | 76.93% | 98.39% | 10.2 | 1569.10 | 4.97 / 5.88 | 0 | [e70f0dcf368d9e28](raw/e70f0dcf368d9e28.json) |

## Japanese/English official-compatible XM3600 retrieval (`official_first_caption`)

### EN · P2 · FP32

| Model | Embedding dim | Parameters | Checkpoint | Dataset | Split | Language | Retrieval variant | Images / captions | Requested / image / text batch | T→I R@1 | T→I R@5 | T→I R@10 | I→T R@1 | I→T R@5 | I→T R@10 | Images/s | Texts/s | Median (ms) | Peak allocated / reserved (GiB) | OOM retries | Raw result |
|---|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| EmbeddingGemma 2 | 768 | 744.4 M | 1.39 GiB | Crossmodal-3600 (XM3600) | all 3,600 images | en | official_first_caption | 3,600 / 3,600 | 16 / 11 / 16 | 58.61% | 82.06% | 89.19% | 59.50% | 85.17% | 91.28% | 7.5 | 208.5 | 926.76 | 4.29 / 6.31 | 0 | [02b17361f1a3a0ef](raw/02b17361f1a3a0ef.json) |
| EmbeddingGemma 2 | 512 | 744.4 M | 1.39 GiB | Crossmodal-3600 (XM3600) | all 3,600 images | en | official_first_caption | 3,600 / 3,600 | 16 / 11 / 16 | 57.56% | 81.94% | 88.83% | 59.28% | 84.64% | 90.81% | 7.5 | 205.6 | 926.62 | 4.29 / 6.31 | 0 | [4ace44f93791fe91](raw/4ace44f93791fe91.json) |
| EmbeddingGemma 2 | 256 | 744.4 M | 1.39 GiB | Crossmodal-3600 (XM3600) | all 3,600 images | en | official_first_caption | 3,600 / 3,600 | 16 / 11 / 16 | 55.75% | 80.69% | 87.89% | 56.86% | 83.36% | 89.75% | 7.5 | 209.2 | 930.31 | 4.29 / 6.31 | 0 | [9bcd1c60eb09d6af](raw/9bcd1c60eb09d6af.json) |
| EmbeddingGemma 2 | 128 | 744.4 M | 1.39 GiB | Crossmodal-3600 (XM3600) | all 3,600 images | en | official_first_caption | 3,600 / 3,600 | 16 / 11 / 16 | 44.17% | 69.28% | 77.89% | 45.28% | 71.33% | 79.75% | 7.5 | 207.0 | 926.74 | 4.29 / 6.31 | 0 | [b69251bbf027b008](raw/b69251bbf027b008.json) |

### JA · P2 · FP32

| Model | Embedding dim | Parameters | Checkpoint | Dataset | Split | Language | Retrieval variant | Images / captions | Requested / image / text batch | T→I R@1 | T→I R@5 | T→I R@10 | I→T R@1 | I→T R@5 | I→T R@10 | Images/s | Texts/s | Median (ms) | Peak allocated / reserved (GiB) | OOM retries | Raw result |
|---|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| EmbeddingGemma 2 | 768 | 744.4 M | 1.39 GiB | Crossmodal-3600 (XM3600) | all 3,600 images | ja | official_first_caption | 3,600 / 3,600 | 16 / 11 / 16 | 80.14% | 95.03% | 97.50% | 79.61% | 95.28% | 97.61% | 7.5 | 204.1 | 914.33 | 4.29 / 6.31 | 0 | [5e1276cab5ecb4e8](raw/5e1276cab5ecb4e8.json) |
| EmbeddingGemma 2 | 128 | 744.4 M | 1.39 GiB | Crossmodal-3600 (XM3600) | all 3,600 images | ja | official_first_caption | 3,600 / 3,600 | 16 / 11 / 16 | 61.75% | 83.36% | 89.28% | 61.11% | 84.33% | 89.86% | 7.5 | 201.7 | 923.46 | 4.29 / 6.31 | 0 | [7984b20f43dbd568](raw/7984b20f43dbd568.json) |
| EmbeddingGemma 2 | 512 | 744.4 M | 1.39 GiB | Crossmodal-3600 (XM3600) | all 3,600 images | ja | official_first_caption | 3,600 / 3,600 | 16 / 11 / 16 | 79.69% | 94.97% | 97.33% | 78.89% | 95.14% | 97.28% | 7.5 | 200.0 | 958.31 | 4.29 / 6.31 | 0 | [82000ad2e515beee](raw/82000ad2e515beee.json) |
| EmbeddingGemma 2 | 256 | 744.4 M | 1.39 GiB | Crossmodal-3600 (XM3600) | all 3,600 images | ja | official_first_caption | 3,600 / 3,600 | 16 / 11 / 16 | 77.25% | 93.69% | 96.78% | 76.86% | 94.17% | 96.94% | 7.6 | 208.6 | 925.28 | 4.29 / 6.31 | 0 | [f2db802561a13e29](raw/f2db802561a13e29.json) |

## Japanese zero-shot classification

No completed measured runs are available.

## XM3600 all-caption retrieval (`all_captions`)

No completed measured runs are available.

## Matched English/Japanese retrieval

### Crossmodal-3600 (XM3600) · P2 · FP32 · official_first_caption

| Model | Embedding dim | Split | Images | Captions (EN / JA) | T→I R@1 (EN → JA; Δ pp) | T→I R@5 (EN → JA; Δ pp) | T→I R@10 (EN → JA; Δ pp) | I→T R@1 (EN → JA; Δ pp) | I→T R@5 (EN → JA; Δ pp) | I→T R@10 (EN → JA; Δ pp) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EmbeddingGemma 2 | 128 | all 3,600 images | 3,600 | 3,600 / 3,600 | 44.17% → 61.75% (+17.58 pp) | 69.28% → 83.36% (+14.08 pp) | 77.89% → 89.28% (+11.39 pp) | 45.28% → 61.11% (+15.83 pp) | 71.33% → 84.33% (+13.00 pp) | 79.75% → 89.86% (+10.11 pp) |
| EmbeddingGemma 2 | 256 | all 3,600 images | 3,600 | 3,600 / 3,600 | 55.75% → 77.25% (+21.50 pp) | 80.69% → 93.69% (+13.00 pp) | 87.89% → 96.78% (+8.89 pp) | 56.86% → 76.86% (+20.00 pp) | 83.36% → 94.17% (+10.81 pp) | 89.75% → 96.94% (+7.19 pp) |
| EmbeddingGemma 2 | 512 | all 3,600 images | 3,600 | 3,600 / 3,600 | 57.56% → 79.69% (+22.14 pp) | 81.94% → 94.97% (+13.03 pp) | 88.83% → 97.33% (+8.50 pp) | 59.28% → 78.89% (+19.61 pp) | 84.64% → 95.14% (+10.50 pp) | 90.81% → 97.28% (+6.47 pp) |
| EmbeddingGemma 2 | 768 | all 3,600 images | 3,600 | 3,600 / 3,600 | 58.61% → 80.14% (+21.53 pp) | 82.06% → 95.03% (+12.97 pp) | 89.19% → 97.50% (+8.31 pp) | 59.50% → 79.61% (+20.11 pp) | 85.17% → 95.28% (+10.11 pp) | 91.28% → 97.61% (+6.33 pp) |

## Model throughput and VRAM

Throughput is computed from model inference only; model downloads, initialization, dataset preparation, and report generation are excluded.

| Model | Dataset | Task | Language | Protocol | Retrieval variant | Precision | Images/s | Texts/s | Median (ms) | Peak allocated / reserved (GiB) | Raw result |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---|
| EmbeddingGemma 2 | CIFAR-10 | classification | en | p0 | — | fp32 | 10.2 | 218.5 | 1566.49 | 4.97 / 5.90 | [3835a771df7c4084](raw/3835a771df7c4084.json) |
| EmbeddingGemma 2 | CIFAR-10 | classification | en | p0 | — | fp32 | 10.2 | 233.1 | 1565.56 | 4.97 / 5.90 | [b8214ecb970362c9](raw/b8214ecb970362c9.json) |
| EmbeddingGemma 2 | CIFAR-10 | classification | en | p0 | — | fp32 | 10.2 | 235.2 | 1568.36 | 4.97 / 5.90 | [cf413596e88f09f8](raw/cf413596e88f09f8.json) |
| EmbeddingGemma 2 | CIFAR-10 | classification | en | p0 | — | fp32 | 10.2 | 237.1 | 1569.68 | 4.97 / 5.90 | [e639f53a1ac6b75e](raw/e639f53a1ac6b75e.json) |
| EmbeddingGemma 2 | CIFAR-10 | classification | en | p1 | — | fp32 | 10.2 | 179.9 | 1567.66 | 4.97 / 5.88 | [4c5567e303a9f771](raw/4c5567e303a9f771.json) |
| EmbeddingGemma 2 | CIFAR-10 | classification | en | p1 | — | fp32 | 10.2 | 176.6 | 1569.65 | 4.97 / 5.88 | [6950817bdd99d973](raw/6950817bdd99d973.json) |
| EmbeddingGemma 2 | CIFAR-10 | classification | en | p1 | — | fp32 | 10.2 | 172.5 | 1567.69 | 4.97 / 5.88 | [c44690decef67d24](raw/c44690decef67d24.json) |
| EmbeddingGemma 2 | CIFAR-10 | classification | en | p1 | — | fp32 | 10.2 | 178.9 | 1569.10 | 4.97 / 5.88 | [e70f0dcf368d9e28](raw/e70f0dcf368d9e28.json) |
| EmbeddingGemma 2 | CIFAR-100 | classification | en | p0 | — | fp32 | 10.2 | 209.6 | 1566.04 | 4.97 / 5.94 | [1466f26677032f18](raw/1466f26677032f18.json) |
| EmbeddingGemma 2 | CIFAR-100 | classification | en | p0 | — | fp32 | 10.2 | 213.0 | 1566.45 | 4.97 / 5.94 | [9ab2047e95be087e](raw/9ab2047e95be087e.json) |
| EmbeddingGemma 2 | CIFAR-100 | classification | en | p0 | — | fp32 | 10.2 | 216.8 | 1560.57 | 4.97 / 5.94 | [a7fde2561ffbbfc8](raw/a7fde2561ffbbfc8.json) |
| EmbeddingGemma 2 | CIFAR-100 | classification | en | p0 | — | fp32 | 10.2 | 203.7 | 1566.77 | 4.97 / 5.94 | [b08c9f0fbe719961](raw/b08c9f0fbe719961.json) |
| EmbeddingGemma 2 | CIFAR-100 | classification | en | p1 | — | fp32 | 10.2 | 247.0 | 1568.62 | 4.97 / 5.90 | [22ace06120892265](raw/22ace06120892265.json) |
| EmbeddingGemma 2 | CIFAR-100 | classification | en | p1 | — | fp32 | 10.3 | 250.3 | 1559.34 | 4.97 / 5.90 | [820f9a84ba7355e3](raw/820f9a84ba7355e3.json) |
| EmbeddingGemma 2 | CIFAR-100 | classification | en | p1 | — | fp32 | 10.2 | 230.4 | 1562.84 | 4.97 / 5.90 | [8969fe62960b874c](raw/8969fe62960b874c.json) |
| EmbeddingGemma 2 | CIFAR-100 | classification | en | p1 | — | fp32 | 10.2 | 245.0 | 1561.61 | 4.97 / 5.90 | [948187d246867b59](raw/948187d246867b59.json) |
| EmbeddingGemma 2 | Crossmodal-3600 (XM3600) | retrieval | en | p2 | official_first_caption | fp32 | 7.5 | 208.5 | 926.76 | 4.29 / 6.31 | [02b17361f1a3a0ef](raw/02b17361f1a3a0ef.json) |
| EmbeddingGemma 2 | Crossmodal-3600 (XM3600) | retrieval | en | p2 | official_first_caption | fp32 | 7.5 | 205.6 | 926.62 | 4.29 / 6.31 | [4ace44f93791fe91](raw/4ace44f93791fe91.json) |
| EmbeddingGemma 2 | Crossmodal-3600 (XM3600) | retrieval | en | p2 | official_first_caption | fp32 | 7.5 | 209.2 | 930.31 | 4.29 / 6.31 | [9bcd1c60eb09d6af](raw/9bcd1c60eb09d6af.json) |
| EmbeddingGemma 2 | Crossmodal-3600 (XM3600) | retrieval | en | p2 | official_first_caption | fp32 | 7.5 | 207.0 | 926.74 | 4.29 / 6.31 | [b69251bbf027b008](raw/b69251bbf027b008.json) |
| EmbeddingGemma 2 | Crossmodal-3600 (XM3600) | retrieval | ja | p2 | official_first_caption | fp32 | 7.5 | 204.1 | 914.33 | 4.29 / 6.31 | [5e1276cab5ecb4e8](raw/5e1276cab5ecb4e8.json) |
| EmbeddingGemma 2 | Crossmodal-3600 (XM3600) | retrieval | ja | p2 | official_first_caption | fp32 | 7.5 | 201.7 | 923.46 | 4.29 / 6.31 | [7984b20f43dbd568](raw/7984b20f43dbd568.json) |
| EmbeddingGemma 2 | Crossmodal-3600 (XM3600) | retrieval | ja | p2 | official_first_caption | fp32 | 7.5 | 200.0 | 958.31 | 4.29 / 6.31 | [82000ad2e515beee](raw/82000ad2e515beee.json) |
| EmbeddingGemma 2 | Crossmodal-3600 (XM3600) | retrieval | ja | p2 | official_first_caption | fp32 | 7.6 | 208.6 | 925.28 | 4.29 / 6.31 | [f2db802561a13e29](raw/f2db802561a13e29.json) |

## Matryoshka embedding dimensions

EmbeddingGemma results use independent native `truncate_dim` inference at each dimension. The [dimension-wise analysis](embedding-dimension-analysis.md) reports classification and retrieval metrics, measured throughput, and FP32 vector-size trade-offs relative to 768D.

## Non-successful runs

No unsuccessful-run JSON records are included in the published raw set. See the machine-readable run status for blocked evaluations and access constraints.

## Validation and reproducibility

Each raw result is schema-validated before aggregation. Per-run JSON records retain the source commit, model checkpoint fingerprint, dataset checksum, ordered candidate IDs and hashes, prompt policy, runtime, and measured metrics.
- Validation artifacts: [result-lifecycle.json](validation/result-lifecycle.json), [embedding-dimension-audit.json](validation/embedding-dimension-audit.json).

## Known limitations

- Official numerical parity applies only to the exact combinations documented in the available validation records. It does not establish parity for other models, languages, datasets, precisions, or retrieval variants.
- Training-data contamination cannot be ruled out unless a model's pretraining corpus is independently verified.
- Dataset and model access restrictions remain in force. See [model-and-dataset-licenses.md](../../docs/model-and-dataset-licenses.md).
- Missing metrics remain missing. Unsupported languages and unavailable datasets are not represented as zero scores.

## Reproduction

```powershell
uv sync --locked --group dev
uv run vlbench doctor --device cuda
uv run vlbench report --input results/v0.1.0/raw --output results/v0.1.0
```

To reproduce a measurement, use the exact model revision, dataset checksum, split, protocol, language, precision, batch size, seed, source commit, and lockfile hash stored in its raw JSON.

## Figures

- Classification Accuracy: [PNG](plots/classification_accuracy.png) · [SVG](plots/classification_accuracy.svg)
- Classification Accuracy P1: [PNG](plots/classification_accuracy_p1.png) · [SVG](plots/classification_accuracy_p1.svg)
- Embedding Dimension Cifar100 P0 En Classification: [PNG](plots/embedding_dimension_cifar100_p0_en_classification.png) · [SVG](plots/embedding_dimension_cifar100_p0_en_classification.svg)
- Embedding Dimension Cifar100 P1 En Classification: [PNG](plots/embedding_dimension_cifar100_p1_en_classification.png) · [SVG](plots/embedding_dimension_cifar100_p1_en_classification.svg)
- Embedding Dimension Cifar10 P0 En Classification: [PNG](plots/embedding_dimension_cifar10_p0_en_classification.png) · [SVG](plots/embedding_dimension_cifar10_p0_en_classification.svg)
- Embedding Dimension Cifar10 P1 En Classification: [PNG](plots/embedding_dimension_cifar10_p1_en_classification.png) · [SVG](plots/embedding_dimension_cifar10_p1_en_classification.svg)
- Embedding Dimension Crossmodal3600 En Official-First-Caption: [PNG](plots/embedding_dimension_crossmodal3600_en_official-first-caption.png) · [SVG](plots/embedding_dimension_crossmodal3600_en_official-first-caption.svg)
- Embedding Dimension Crossmodal3600 Ja Official-First-Caption: [PNG](plots/embedding_dimension_crossmodal3600_ja_official-first-caption.png) · [SVG](plots/embedding_dimension_crossmodal3600_ja_official-first-caption.svg)
- Embedding Dimension Storage: [PNG](plots/embedding_dimension_storage.png) · [SVG](plots/embedding_dimension_storage.svg)
- Performance Comparison: [PNG](plots/performance_comparison.png) · [SVG](plots/performance_comparison.svg)
- Performance Comparison P1: [PNG](plots/performance_comparison_p1.png) · [SVG](plots/performance_comparison_p1.svg)
- Xm3600 Language Comparison Official First Caption: [PNG](plots/xm3600_language_comparison_official_first_caption.png) · [SVG](plots/xm3600_language_comparison_official_first_caption.svg)
- Xm3600 Official Retrieval: [PNG](plots/xm3600_official_retrieval.png) · [SVG](plots/xm3600_official_retrieval.svg)
