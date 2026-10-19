# Vision-Language Benchmark Report

This report is regenerated from schema-validated raw JSON. Only completed runs contribute numeric metrics; failed, skipped, and blocked runs are listed separately. No values are estimated or fabricated.

## Conditions

- Validated result files: 31
- Completed measurements: 31
- Skipped runs: 0
- Failed runs: 0
- Blocked runs: 0
- Source commit(s): 2993187c2b960e6e6690c97205b22838d18a36a3, 60a61259c95e7c387e896b23084e3c3d0fc375f5, d72e4e462d137e9b77581a507e42897ff427eaf1
- Runtime profile(s): NVIDIA GeForce RTX 4070 Ti SUPER (17170956288 bytes, driver 610.88, PyTorch 2.14.1+cu130, CUDA 13.0, fp32)
- Protocols and precisions are kept in separate tables and chart files.
- Accuracy and recall values in JSON are fractions; report tables display percentages.
- Checkpoint sizes are pinned weight-file sizes; tokenizer and configuration files may add to the download. Peak memory is reported as allocated / reserved GiB.
- Each completed table row links to its schema-validated raw result JSON.

## English classification

### P0 · FP32

| Model | Parameters | Checkpoint | Dataset | Split | Images | Classes | Requested / image / text batch | Top-1 | Top-5 | Images/s | Median (ms) | Peak allocated / reserved (GiB) | OOM retries | Raw result |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| OpenCLIP ViT-L/14 LAION-2B | 427.6 M | 1.59 GiB | CIFAR-100 | test | 10,000 | 100 | 16 / 16 / 16 | 83.34% | 96.63% | 107.0 | 148.12 | 1.80 / 1.94 | 0 | [1de2d2b7854b5e80-30c3f626](raw/1de2d2b7854b5e80-30c3f626.json) |
| OpenAI CLIP ViT-B/32 | 151.3 M | 0.33 GiB | CIFAR-10 | test | 10,000 | 10 | 16 / 16 / 16 | 89.83% | 99.61% | 745.5 | 20.62 | 0.62 / 0.68 | 0 | [35cef7e32f8afb13](raw/35cef7e32f8afb13.json) |
| OpenCLIP ViT-B/32 LAION-2B | 151.3 M | 0.56 GiB | CIFAR-100 | test | 10,000 | 100 | 16 / 16 / 16 | 75.56% | 93.87% | 701.1 | 21.64 | 0.61 / 0.68 | 0 | [35dfb13b0aa46c57-54e65389](raw/35dfb13b0aa46c57-54e65389.json) |
| SigLIP 2 Base/16 224 | 375.2 M | 1.40 GiB | CIFAR-10 | test | 10,000 | 10 | 16 / 16 / 16 | 93.62% | 99.85% | 338.3 | 48.06 | 1.52 / 1.67 | 0 | [38b1a8be68ae509c](raw/38b1a8be68ae509c.json) |
| EmbeddingGemma 2 | 744.4 M | 1.39 GiB | CIFAR-100 | test | 10,000 | 100 | 16 / 16 / 16 | 76.08% | 92.92% | 10.2 | 1566.77 | 4.97 / 5.94 | 0 | [4bf47c3e534537b0-fb1914c0](raw/4bf47c3e534537b0-fb1914c0.json) |
| OpenAI CLIP ViT-B/32 | 151.3 M | 0.33 GiB | CIFAR-100 | test | 10,000 | 100 | 16 / 16 / 16 | 64.20% | 88.73% | 668.8 | 22.67 | 0.62 / 0.68 | 0 | [5a8886b0fab54729-30d1b401](raw/5a8886b0fab54729-30d1b401.json) |
| OpenCLIP ViT-B/32 LAION-2B | 151.3 M | 0.56 GiB | CIFAR-10 | test | 10,000 | 10 | 16 / 16 / 16 | 93.56% | 99.81% | 749.6 | 20.78 | 0.61 / 0.68 | 0 | [649c7b44763ebc72](raw/649c7b44763ebc72.json) |
| OpenCLIP Multilingual ViT-B/32 | 366.1 M | 1.36 GiB | CIFAR-10 | test | 10,000 | 10 | 16 / 16 / 16 | 93.64% | 99.88% | 702.0 | 22.65 | 1.64 / 1.90 | 0 | [931adb237a72e7b9](raw/931adb237a72e7b9.json) |
| EmbeddingGemma 2 | 744.4 M | 1.39 GiB | CIFAR-10 | test | 10,000 | 10 | 16 / 16 / 16 | 94.66% | 99.84% | 10.2 | 1573.28 | 4.97 / 5.90 | 0 | [ab5d599956168b86](raw/ab5d599956168b86.json) |
| OpenCLIP ViT-L/14 LAION-2B | 427.6 M | 1.59 GiB | CIFAR-10 | test | 10,000 | 10 | 16 / 16 / 16 | 96.61% | 99.85% | 106.6 | 150.49 | 1.80 / 1.95 | 0 | [c8faef3bbd4c2779](raw/c8faef3bbd4c2779.json) |
| SigLIP 2 Base/16 224 | 375.2 M | 1.40 GiB | CIFAR-100 | test | 10,000 | 100 | 16 / 16 / 16 | 73.68% | 91.89% | 342.3 | 44.61 | 1.52 / 1.66 | 0 | [d59181f45a441ab9-1a33071f](raw/d59181f45a441ab9-1a33071f.json) |
| OpenCLIP Multilingual ViT-B/32 | 366.1 M | 1.36 GiB | CIFAR-100 | test | 10,000 | 100 | 16 / 16 / 16 | 76.55% | 94.99% | 750.9 | 21.08 | 1.64 / 1.90 | 0 | [e4c4922a8b419fc2-ac4a43b4](raw/e4c4922a8b419fc2-ac4a43b4.json) |

### P1 · FP32

| Model | Parameters | Checkpoint | Dataset | Split | Images | Classes | Requested / image / text batch | Top-1 | Top-5 | Images/s | Median (ms) | Peak allocated / reserved (GiB) | OOM retries | Raw result |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| OpenCLIP Multilingual ViT-B/32 | 366.1 M | 1.36 GiB | CIFAR-10 | test | 10,000 | 10 | 16 / 16 / 16 | 93.54% | 99.86% | 739.2 | 21.03 | 1.54 / 1.71 | 0 | [013541e887452604](raw/013541e887452604.json) |
| EmbeddingGemma 2 | 744.4 M | 1.39 GiB | CIFAR-100 | test | 10,000 | 100 | 16 / 16 / 16 | 62.84% | 84.04% | 10.2 | 1564.45 | 4.97 / 5.90 | 0 | [0c75f5095220051c-f8285c69](raw/0c75f5095220051c-f8285c69.json) |
| SigLIP 2 Base/16 224 | 375.2 M | 1.40 GiB | CIFAR-100 | test | 10,000 | 100 | 16 / 16 / 16 | 73.25% | 91.71% | 338.3 | 48.13 | 1.52 / 1.67 | 0 | [38a8be7d4b997457-8dbcaf34](raw/38a8be7d4b997457-8dbcaf34.json) |
| OpenCLIP ViT-L/14 LAION-2B | 427.6 M | 1.59 GiB | CIFAR-100 | test | 10,000 | 100 | 16 / 16 / 16 | 83.07% | 96.57% | 106.1 | 150.43 | 1.80 / 1.95 | 0 | [5cd03bacdf1e5b06-43791242](raw/5cd03bacdf1e5b06-43791242.json) |
| OpenCLIP ViT-L/14 LAION-2B | 427.6 M | 1.59 GiB | CIFAR-10 | test | 10,000 | 10 | 16 / 16 / 16 | 96.92% | 99.93% | 106.8 | 149.89 | 1.80 / 1.90 | 0 | [9c075a8286dd1604](raw/9c075a8286dd1604.json) |
| OpenCLIP ViT-B/32 LAION-2B | 151.3 M | 0.56 GiB | CIFAR-100 | test | 10,000 | 100 | 16 / 16 / 16 | 75.59% | 93.65% | 773.6 | 20.11 | 0.61 / 0.68 | 0 | [a54f8b8dfb17227b-8e3ad59c](raw/a54f8b8dfb17227b-8e3ad59c.json) |
| EmbeddingGemma 2 | 744.4 M | 1.39 GiB | CIFAR-10 | test | 10,000 | 10 | 16 / 16 / 16 | 92.58% | 99.64% | 10.2 | 1571.21 | 4.97 / 5.88 | 0 | [beac7b7ac03ee242](raw/beac7b7ac03ee242.json) |
| OpenAI CLIP ViT-B/32 | 151.3 M | 0.33 GiB | CIFAR-100 | test | 10,000 | 100 | 16 / 16 / 16 | 62.32% | 87.00% | 723.2 | 21.77 | 0.62 / 0.68 | 0 | [c08f92332ebe01a4-2815d62c](raw/c08f92332ebe01a4-2815d62c.json) |
| OpenCLIP ViT-B/32 LAION-2B | 151.3 M | 0.56 GiB | CIFAR-10 | test | 10,000 | 10 | 16 / 16 / 16 | 93.68% | 99.84% | 717.7 | 22.66 | 0.61 / 0.66 | 0 | [c3350bf4370a58b8](raw/c3350bf4370a58b8.json) |
| OpenAI CLIP ViT-B/32 | 151.3 M | 0.33 GiB | CIFAR-10 | test | 10,000 | 10 | 16 / 16 / 16 | 88.30% | 99.24% | 688.7 | 22.46 | 0.62 / 0.68 | 0 | [c542f4a0ea8ac9a8](raw/c542f4a0ea8ac9a8.json) |
| OpenCLIP Multilingual ViT-B/32 | 366.1 M | 1.36 GiB | CIFAR-100 | test | 10,000 | 100 | 16 / 16 / 16 | 76.46% | 95.02% | 748.0 | 20.87 | 1.64 / 1.90 | 0 | [d4721f7c31f9d23c-e083b306](raw/d4721f7c31f9d23c-e083b306.json) |
| SigLIP 2 Base/16 224 | 375.2 M | 1.40 GiB | CIFAR-10 | test | 10,000 | 10 | 16 / 16 / 16 | 93.85% | 99.88% | 336.9 | 48.36 | 1.52 / 1.65 | 0 | [ddcae56da141ed7f](raw/ddcae56da141ed7f.json) |

## English retrieval

### P2 · FP32

| Model | Parameters | Checkpoint | Dataset | Split | Language | Images / captions | Requested / image / text batch | T→I R@1 | T→I R@5 | T→I R@10 | I→T R@1 | I→T R@5 | I→T R@10 | Images/s | Texts/s | Median (ms) | Peak allocated / reserved (GiB) | OOM retries | Raw result |
|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| EmbeddingGemma 2 | 744.4 M | 1.39 GiB | Crossmodal-3600 (XM3600) | all 3,600 images | en | 3,600 / 7,200 | 16 / 11 / 16 | 61.79% | 83.76% | 90.08% | 70.36% | 90.81% | 95.69% | 7.4 | 207.4 | 84.24 | 4.29 / 6.31 | 0 | [3284f6a2ba1863cc](raw/3284f6a2ba1863cc.json) |
| SigLIP 2 Base/16 224 | 375.2 M | 1.40 GiB | Crossmodal-3600 (XM3600) | all 3,600 images | en | 3,600 / 7,200 | 16 / 16 / 16 | 52.44% | 76.92% | 84.08% | 60.67% | 83.89% | 89.72% | 315.4 | 1023.8 | 15.70 | 1.52 / 1.65 | 0 | [36baa9edd5e9fa9a-53dd822c](raw/36baa9edd5e9fa9a-53dd822c.json) |
| OpenCLIP Multilingual ViT-B/32 | 366.1 M | 1.36 GiB | Crossmodal-3600 (XM3600) | all 3,600 images | en | 3,600 / 7,200 | 16 / 16 / 16 | 48.72% | 76.18% | 84.35% | 57.75% | 83.50% | 91.06% | 349.5 | 165.8 | 95.16 | 1.64 / 1.91 | 0 | [8f9f79539fb877d6](raw/8f9f79539fb877d6.json) |

## Japanese classification

No completed measured runs are available.

## Japanese retrieval

### P2 · FP32

| Model | Parameters | Checkpoint | Dataset | Split | Language | Images / captions | Requested / image / text batch | T→I R@1 | T→I R@5 | T→I R@10 | I→T R@1 | I→T R@5 | I→T R@10 | Images/s | Texts/s | Median (ms) | Peak allocated / reserved (GiB) | OOM retries | Raw result |
|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| OpenCLIP Multilingual ViT-B/32 | 366.1 M | 1.36 GiB | Crossmodal-3600 (XM3600) | all 3,600 images | ja | 3,600 / 7,185 | 16 / 16 / 16 | 59.19% | 84.05% | 91.20% | 71.56% | 92.83% | 96.56% | 389.3 | 167.7 | 94.53 | 1.64 / 1.91 | 0 | [028043fb5a078d1e-214521ad](raw/028043fb5a078d1e-214521ad.json) |
| rinna Japanese CLIP ViT-B/16 | 196.6 M | 0.73 GiB | Crossmodal-3600 (XM3600) | all 3,600 images | ja | 3,600 / 7,185 | 16 / 16 / 16 | 50.33% | 77.27% | 85.21% | 58.00% | 84.83% | 91.61% | 319.7 | 751.4 | 21.42 | 0.86 / 0.97 | 0 | [221fdf4bab042e89](raw/221fdf4bab042e89.json) |
| SigLIP 2 Base/16 224 | 375.2 M | 1.40 GiB | Crossmodal-3600 (XM3600) | all 3,600 images | ja | 3,600 / 7,185 | 16 / 16 / 16 | 37.51% | 62.85% | 72.12% | 52.58% | 79.11% | 86.83% | 307.9 | 1024.2 | 15.82 | 1.52 / 1.65 | 0 | [405472b594821180-783ab865](raw/405472b594821180-783ab865.json) |
| EmbeddingGemma 2 | 744.4 M | 1.39 GiB | Crossmodal-3600 (XM3600) | all 3,600 images | ja | 3,600 / 7,185 | 16 / 11 / 16 | 76.23% | 92.86% | 96.10% | 84.25% | 96.78% | 98.64% | 7.5 | 205.5 | 80.98 | 4.29 / 6.31 | 0 | [5a7dc3bfc3d618c1](raw/5a7dc3bfc3d618c1.json) |

## Matched English/Japanese retrieval

### Crossmodal-3600 (XM3600) · P2 · FP32

| Model | Split | Images | Captions (EN / JA) | T→I R@1 (EN → JA; Δ pp) | T→I R@5 (EN → JA; Δ pp) | T→I R@10 (EN → JA; Δ pp) | I→T R@1 (EN → JA; Δ pp) | I→T R@5 (EN → JA; Δ pp) | I→T R@10 (EN → JA; Δ pp) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EmbeddingGemma 2 | all 3,600 images | 3,600 | 7,200 / 7,185 | 61.79% → 76.23% (+14.44 pp) | 83.76% → 92.86% (+9.10 pp) | 90.08% → 96.10% (+6.02 pp) | 70.36% → 84.25% (+13.89 pp) | 90.81% → 96.78% (+5.97 pp) | 95.69% → 98.64% (+2.94 pp) |
| OpenCLIP Multilingual ViT-B/32 | all 3,600 images | 3,600 | 7,200 / 7,185 | 48.72% → 59.19% (+10.47 pp) | 76.18% → 84.05% (+7.87 pp) | 84.35% → 91.20% (+6.86 pp) | 57.75% → 71.56% (+13.81 pp) | 83.50% → 92.83% (+9.33 pp) | 91.06% → 96.56% (+5.50 pp) |
| SigLIP 2 Base/16 224 | all 3,600 images | 3,600 | 7,200 / 7,185 | 52.44% → 37.51% (-14.94 pp) | 76.92% → 62.85% (-14.06 pp) | 84.08% → 72.12% (-11.96 pp) | 60.67% → 52.58% (-8.08 pp) | 83.89% → 79.11% (-4.78 pp) | 89.72% → 86.83% (-2.89 pp) |

## Non-successful runs

No unsuccessful-run JSON records are included in the published raw set. See the machine-readable run status for blocked evaluations and access constraints.

## Limitations

- A completed run is not evidence of numerical parity with CLIP Benchmark; exact parity is verified only for OpenAI CLIP ViT-B/32 on CIFAR-10 P0. See [the comparison record](validation/official-cifar10-parity.json).
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
- Multilingual Comparison: [PNG](plots/multilingual_comparison.png) · [SVG](plots/multilingual_comparison.svg)
- Performance Comparison: [PNG](plots/performance_comparison.png) · [SVG](plots/performance_comparison.svg)
- Performance Comparison P1: [PNG](plots/performance_comparison_p1.png) · [SVG](plots/performance_comparison_p1.svg)
- Retrieval Recall: [PNG](plots/retrieval_recall.png) · [SVG](plots/retrieval_recall.svg)
