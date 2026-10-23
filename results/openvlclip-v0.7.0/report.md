# Vision-Language Benchmark Report

This report is regenerated from schema-validated raw JSON. Only completed runs contribute numeric metrics; failed, skipped, and blocked runs are listed separately. No values are estimated or fabricated.

## Conditions

- Validated result files: 18
- Completed measurements: 18
- Skipped runs: 0
- Failed runs: 0
- Blocked runs: 0
- Source commit(s): 23b469da312efc17f64aae7119b74ceeeb10f42d
- Runtime profile(s): NVIDIA GeForce RTX 4070 Ti SUPER (17170956288 bytes, driver 610.88, PyTorch 2.14.1+cu130, CUDA 13.0, fp32)
- Protocols and precisions are kept in separate tables and chart files.
- Accuracy and recall values in JSON are fractions; report tables display percentages.
- Checkpoint sizes are pinned weight-file sizes; tokenizer and configuration files may add to the download. Peak memory is reported as allocated / reserved GiB.
- Each completed table row links to its schema-validated raw result JSON.

## Measurement lifecycle

- Frozen benchmark source commit: `23b469da312efc17f64aae7119b74ceeeb10f42d`
- Active validated measurements: 18
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
| OpenVLCLIP Qwen3-VL-Embedding-2B | 2,048 | 2127.5 M | 3.96 GiB | CIFAR-100 | test | 10,000 | 100 | 1 / 1 / 1 | 57.66% (5,766/10,000) | 80.99% (8,099/10,000) | 8.5 | 109.04 | 7.94 / 8.06 | 0 | [1e814e0b64df8a09](raw/1e814e0b64df8a09.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | 768 | 375.2 M | 1.40 GiB | CIFAR-10 | test | 10,000 | 10 | 16 / 16 / 16 | 55.30% (5,530/10,000) | 94.81% (9,481/10,000) | 245.2 | 64.64 | 1.56 / 1.70 | 0 | [265a5e3112c0d2b4](raw/265a5e3112c0d2b4.json) |
| OpenVLCLIP SigLIP 2 Base/16 FixRes | 768 | 375.2 M | 1.40 GiB | CIFAR-10 | test | 10,000 | 10 | 16 / 16 / 16 | 57.14% (5,714/10,000) | 94.60% (9,460/10,000) | 293.1 | 54.77 | 1.52 / 1.67 | 0 | [3595fa7c8edbf265](raw/3595fa7c8edbf265.json) |
| OpenVLCLIP Qwen3-VL-Embedding-2B | 2,048 | 2127.5 M | 3.96 GiB | CIFAR-10 | test | 10,000 | 10 | 1 / 1 / 1 | 87.58% (8,758/10,000) | 98.68% (9,868/10,000) | 5.3 | 197.03 | 7.94 / 8.06 | 0 | [44ca47e7559432f0](raw/44ca47e7559432f0.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | 768 | 375.2 M | 1.40 GiB | CIFAR-100 | test | 10,000 | 100 | 16 / 16 / 16 | 29.78% (2,978/10,000) | 57.00% (5,700/10,000) | 250.7 | 64.37 | 1.56 / 1.70 | 0 | [d35c9c3f4583635a](raw/d35c9c3f4583635a.json) |
| OpenVLCLIP SigLIP 2 Base/16 FixRes | 768 | 375.2 M | 1.40 GiB | CIFAR-100 | test | 10,000 | 100 | 16 / 16 / 16 | 24.45% (2,445/10,000) | 51.26% (5,126/10,000) | 306.2 | 53.09 | 1.52 / 1.66 | 0 | [d7302c99a73e813f](raw/d7302c99a73e813f.json) |

### P1 · FP32

| Model | Embedding dim | Parameters | Checkpoint | Dataset | Split | Images | Classes | Requested / image / text batch | Top-1 (correct/total) | Top-5 (correct/total) | Images/s | Median (ms) | Peak allocated / reserved (GiB) | OOM retries | Raw result |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| OpenVLCLIP Qwen3-VL-Embedding-2B | 2,048 | 2127.5 M | 3.96 GiB | CIFAR-100 | test | 10,000 | 100 | 1 / 1 / 1 | 57.30% (5,730/10,000) | 79.31% (7,931/10,000) | 4.1 | 113.28 | 7.94 / 8.06 | 0 | [10a1bb028a2f5318](raw/10a1bb028a2f5318.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | 768 | 375.2 M | 1.40 GiB | CIFAR-100 | test | 10,000 | 100 | 16 / 16 / 16 | 25.50% (2,550/10,000) | 59.91% (5,991/10,000) | 241.8 | 65.17 | 1.56 / 1.70 | 0 | [35f85add213561f9](raw/35f85add213561f9.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | 768 | 375.2 M | 1.40 GiB | CIFAR-10 | test | 10,000 | 10 | 16 / 16 / 16 | 78.28% (7,828/10,000) | 98.70% (9,870/10,000) | 240.2 | 65.34 | 1.56 / 1.70 | 0 | [3d21c3b372fb4491](raw/3d21c3b372fb4491.json) |
| OpenVLCLIP SigLIP 2 Base/16 FixRes | 768 | 375.2 M | 1.40 GiB | CIFAR-100 | test | 10,000 | 100 | 16 / 16 / 16 | 13.13% (1,313/10,000) | 40.02% (4,002/10,000) | 304.3 | 52.76 | 1.52 / 1.67 | 0 | [5fb3917ec73989e4](raw/5fb3917ec73989e4.json) |
| OpenVLCLIP Qwen3-VL-Embedding-2B | 2,048 | 2127.5 M | 3.96 GiB | CIFAR-10 | test | 10,000 | 10 | 1 / 1 / 1 | 87.08% (8,708/10,000) | 98.68% (9,868/10,000) | 5.0 | 199.67 | 7.94 / 8.06 | 0 | [b602850fd3d77ccd](raw/b602850fd3d77ccd.json) |
| OpenVLCLIP SigLIP 2 Base/16 FixRes | 768 | 375.2 M | 1.40 GiB | CIFAR-10 | test | 10,000 | 10 | 16 / 16 / 16 | 55.39% (5,539/10,000) | 94.95% (9,495/10,000) | 304.1 | 52.77 | 1.52 / 1.65 | 0 | [e603831c0064f67a](raw/e603831c0064f67a.json) |

## Japanese/English official-compatible XM3600 retrieval (`official_first_caption`)

### EN · P2 · FP32

| Model | Embedding dim | Parameters | Checkpoint | Dataset | Split | Language | Retrieval variant | Images / captions | Requested / image / text batch | T→I R@1 (hits/queries) | T→I R@5 (hits/queries) | T→I R@10 (hits/queries) | I→T R@1 (hits/queries) | I→T R@5 (hits/queries) | I→T R@10 (hits/queries) | Images/s | Texts/s | Median (ms) | Peak allocated / reserved (GiB) | OOM retries | Raw result |
|---|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| OpenVLCLIP Qwen3-VL-Embedding-2B | 2,048 | 2127.5 M | 3.96 GiB | Crossmodal-3600 (XM3600) | all 3,600 images | en | official_first_caption | 3,600 / 3,600 | 1 / 1 / 1 | 55.78% (2,008/3,600) | 79.58% (2,865/3,600) | 86.67% (3,120/3,600) | 57.94% (2,086/3,600) | 82.31% (2,963/3,600) | 89.25% (3,213/3,600) | 4.3 | 17.2 | 61.35 | 8.01 / 8.15 | 0 | [026e74e2db5b6144](raw/026e74e2db5b6144.json) |
| OpenVLCLIP SigLIP 2 Base/16 FixRes | 768 | 375.2 M | 1.40 GiB | Crossmodal-3600 (XM3600) | all 3,600 images | en | official_first_caption | 3,600 / 3,600 | 16 / 16 / 16 | 0.19% (7/3,600) | 0.89% (32/3,600) | 1.75% (63/3,600) | 0.94% (34/3,600) | 2.56% (92/3,600) | 4.03% (145/3,600) | 234.8 | 7.3 | 915.43 | 1.52 / 1.65 | 0 | [4008d7997a0def30](raw/4008d7997a0def30.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | 768 | 375.2 M | 1.40 GiB | Crossmodal-3600 (XM3600) | all 3,600 images | en | official_first_caption | 3,600 / 3,600 | 16 / 16 / 16 | 0.36% (13/3,600) | 1.97% (71/3,600) | 3.44% (124/3,600) | 1.36% (49/3,600) | 4.19% (151/3,600) | 6.94% (250/3,600) | 162.5 | 5.5 | 943.96 | 1.56 / 1.69 | 0 | [deef20de60c0e3a1](raw/deef20de60c0e3a1.json) |

### JA · P2 · FP32

| Model | Embedding dim | Parameters | Checkpoint | Dataset | Split | Language | Retrieval variant | Images / captions | Requested / image / text batch | T→I R@1 (hits/queries) | T→I R@5 (hits/queries) | T→I R@10 (hits/queries) | I→T R@1 (hits/queries) | I→T R@5 (hits/queries) | I→T R@10 (hits/queries) | Images/s | Texts/s | Median (ms) | Peak allocated / reserved (GiB) | OOM retries | Raw result |
|---|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| OpenVLCLIP Qwen3-VL-Embedding-2B | 2,048 | 2127.5 M | 3.96 GiB | Crossmodal-3600 (XM3600) | all 3,600 images | ja | official_first_caption | 3,600 / 3,600 | 1 / 1 / 1 | 77.81% (2,801/3,600) | 93.58% (3,369/3,600) | 96.47% (3,473/3,600) | 72.25% (2,601/3,600) | 91.22% (3,284/3,600) | 95.17% (3,426/3,600) | 4.2 | 15.1 | 69.95 | 8.01 / 8.15 | 0 | [a123010f08059e2f](raw/a123010f08059e2f.json) |
| OpenVLCLIP SigLIP 2 Base/16 FixRes | 768 | 375.2 M | 1.40 GiB | Crossmodal-3600 (XM3600) | all 3,600 images | ja | official_first_caption | 3,600 / 3,600 | 16 / 16 / 16 | 0.39% (14/3,600) | 1.58% (57/3,600) | 2.83% (102/3,600) | 1.39% (50/3,600) | 3.39% (122/3,600) | 5.00% (180/3,600) | 256.8 | 5.2 | 959.18 | 1.52 / 1.65 | 0 | [b0d09dd268d8159e](raw/b0d09dd268d8159e.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | 768 | 375.2 M | 1.40 GiB | Crossmodal-3600 (XM3600) | all 3,600 images | ja | official_first_caption | 3,600 / 3,600 | 16 / 16 / 16 | 0.44% (16/3,600) | 1.22% (44/3,600) | 2.33% (84/3,600) | 0.81% (29/3,600) | 2.33% (84/3,600) | 3.31% (119/3,600) | 181.7 | 5.4 | 935.42 | 1.56 / 1.69 | 0 | [ea34cf1a5133f4c7](raw/ea34cf1a5133f4c7.json) |

## Japanese zero-shot classification

No completed measured runs are available.

## XM3600 all-caption retrieval (`all_captions`)

No completed measured runs are available.

## Matched English/Japanese retrieval

### Crossmodal-3600 (XM3600) · P2 · FP32 · official_first_caption

| Model | Embedding dim | Split | Images | Captions (EN / JA) | T→I R@1 (EN → JA; Δ pp) | T→I R@5 (EN → JA; Δ pp) | T→I R@10 (EN → JA; Δ pp) | I→T R@1 (EN → JA; Δ pp) | I→T R@5 (EN → JA; Δ pp) | I→T R@10 (EN → JA; Δ pp) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenVLCLIP Qwen3-VL-Embedding-2B | 2,048 | all 3,600 images | 3,600 | 3,600 / 3,600 | 55.78% → 77.81% (+22.03 pp) | 79.58% → 93.58% (+14.00 pp) | 86.67% → 96.47% (+9.81 pp) | 57.94% → 72.25% (+14.31 pp) | 82.31% → 91.22% (+8.92 pp) | 89.25% → 95.17% (+5.92 pp) |
| OpenVLCLIP SigLIP 2 Base/16 FixRes | 768 | all 3,600 images | 3,600 | 3,600 / 3,600 | 0.19% → 0.39% (+0.19 pp) | 0.89% → 1.58% (+0.69 pp) | 1.75% → 2.83% (+1.08 pp) | 0.94% → 1.39% (+0.44 pp) | 2.56% → 3.39% (+0.83 pp) | 4.03% → 5.00% (+0.97 pp) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | 768 | all 3,600 images | 3,600 | 3,600 / 3,600 | 0.36% → 0.44% (+0.08 pp) | 1.97% → 1.22% (-0.75 pp) | 3.44% → 2.33% (-1.11 pp) | 1.36% → 0.81% (-0.56 pp) | 4.19% → 2.33% (-1.86 pp) | 6.94% → 3.31% (-3.64 pp) |

## Model throughput and VRAM

Throughput is computed from model inference only; model downloads, initialization, dataset preparation, and report generation are excluded.

| Model | Dataset | Task | Language | Protocol | Retrieval variant | Precision | Images/s | Texts/s | Median (ms) | Peak allocated / reserved (GiB) | Raw result |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---|
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | CIFAR-10 | classification | en | p0 | — | fp32 | 245.2 | 5.0 | 64.64 | 1.56 / 1.70 | [265a5e3112c0d2b4](raw/265a5e3112c0d2b4.json) |
| OpenVLCLIP SigLIP 2 Base/16 FixRes | CIFAR-10 | classification | en | p0 | — | fp32 | 293.1 | 7.6 | 54.77 | 1.52 / 1.67 | [3595fa7c8edbf265](raw/3595fa7c8edbf265.json) |
| OpenVLCLIP Qwen3-VL-Embedding-2B | CIFAR-10 | classification | en | p0 | — | fp32 | 5.3 | 9.2 | 197.03 | 7.94 / 8.06 | [44ca47e7559432f0](raw/44ca47e7559432f0.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | CIFAR-10 | classification | en | p1 | — | fp32 | 240.2 | 3.7 | 65.34 | 1.56 / 1.70 | [3d21c3b372fb4491](raw/3d21c3b372fb4491.json) |
| OpenVLCLIP Qwen3-VL-Embedding-2B | CIFAR-10 | classification | en | p1 | — | fp32 | 5.0 | 19.1 | 199.67 | 7.94 / 8.06 | [b602850fd3d77ccd](raw/b602850fd3d77ccd.json) |
| OpenVLCLIP SigLIP 2 Base/16 FixRes | CIFAR-10 | classification | en | p1 | — | fp32 | 304.1 | 6.0 | 52.77 | 1.52 / 1.65 | [e603831c0064f67a](raw/e603831c0064f67a.json) |
| OpenVLCLIP Qwen3-VL-Embedding-2B | CIFAR-100 | classification | en | p0 | — | fp32 | 8.5 | 8.5 | 109.04 | 7.94 / 8.06 | [1e814e0b64df8a09](raw/1e814e0b64df8a09.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | CIFAR-100 | classification | en | p0 | — | fp32 | 250.7 | 8.0 | 64.37 | 1.56 / 1.70 | [d35c9c3f4583635a](raw/d35c9c3f4583635a.json) |
| OpenVLCLIP SigLIP 2 Base/16 FixRes | CIFAR-100 | classification | en | p0 | — | fp32 | 306.2 | 8.4 | 53.09 | 1.52 / 1.66 | [d7302c99a73e813f](raw/d7302c99a73e813f.json) |
| OpenVLCLIP Qwen3-VL-Embedding-2B | CIFAR-100 | classification | en | p1 | — | fp32 | 4.1 | 17.3 | 113.28 | 7.94 / 8.06 | [10a1bb028a2f5318](raw/10a1bb028a2f5318.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | CIFAR-100 | classification | en | p1 | — | fp32 | 241.8 | 4.8 | 65.17 | 1.56 / 1.70 | [35f85add213561f9](raw/35f85add213561f9.json) |
| OpenVLCLIP SigLIP 2 Base/16 FixRes | CIFAR-100 | classification | en | p1 | — | fp32 | 304.3 | 7.8 | 52.76 | 1.52 / 1.67 | [5fb3917ec73989e4](raw/5fb3917ec73989e4.json) |
| OpenVLCLIP Qwen3-VL-Embedding-2B | Crossmodal-3600 (XM3600) | retrieval | en | p2 | official_first_caption | fp32 | 4.3 | 17.2 | 61.35 | 8.01 / 8.15 | [026e74e2db5b6144](raw/026e74e2db5b6144.json) |
| OpenVLCLIP SigLIP 2 Base/16 FixRes | Crossmodal-3600 (XM3600) | retrieval | en | p2 | official_first_caption | fp32 | 234.8 | 7.3 | 915.43 | 1.52 / 1.65 | [4008d7997a0def30](raw/4008d7997a0def30.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | Crossmodal-3600 (XM3600) | retrieval | en | p2 | official_first_caption | fp32 | 162.5 | 5.5 | 943.96 | 1.56 / 1.69 | [deef20de60c0e3a1](raw/deef20de60c0e3a1.json) |
| OpenVLCLIP Qwen3-VL-Embedding-2B | Crossmodal-3600 (XM3600) | retrieval | ja | p2 | official_first_caption | fp32 | 4.2 | 15.1 | 69.95 | 8.01 / 8.15 | [a123010f08059e2f](raw/a123010f08059e2f.json) |
| OpenVLCLIP SigLIP 2 Base/16 FixRes | Crossmodal-3600 (XM3600) | retrieval | ja | p2 | official_first_caption | fp32 | 256.8 | 5.2 | 959.18 | 1.52 / 1.65 | [b0d09dd268d8159e](raw/b0d09dd268d8159e.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | Crossmodal-3600 (XM3600) | retrieval | ja | p2 | official_first_caption | fp32 | 181.7 | 5.4 | 935.42 | 1.56 / 1.69 | [ea34cf1a5133f4c7](raw/ea34cf1a5133f4c7.json) |

## Matched historical model comparisons

Rows are matched to each new benchmark run by dataset checksum, split, task, protocol, language, precision, sample/candidate counts, and retrieval variant. Only lifecycle-eligible completed historical results are included. Matching protocol labels do not imply identical model-side preprocessing or instructions.
P0 keeps the common 18-template candidate-text protocol; P1 remains model-specific.
The historical direct Transformers SigLIP 2 CIFAR results predate the PAD-derived attention mask used by OpenVLCLIP v0.7.0. On the same first 128 CIFAR-10 P0 images and 180 prompts, OpenVLCLIP features matched the current masked direct adapter exactly, while the legacy unmasked text features differed (maximum absolute difference 6.9498; mean prompt cosine 0.6882). The historical 93.62% Top-1 result is retained unchanged as context, not a controlled like-for-like quality comparison. The subset diagnostic is `validation/siglip2-p0-parity.json`.
Qwen's fixed classification instruction and separate retrieval query/document instructions are recorded in the comparison CSV and raw JSON. Qwen retrieval throughput counts both text-encoding passes.
Machine-readable rows: [model-comparison.csv](model-comparison.csv); selection details: [model-comparison-audit.json](validation/model-comparison-audit.json).

### CIFAR10 · P0 · EN · FP32

| Model | Top-1 (correct/total) | Top-5 (correct/total) | Images/s | Peak allocated/reserved (GiB) | Embedding/vector bytes | FP32 image+text payload | Package | Source commit | Raw result |
|---|---:|---:|---:|---:|---|---:|---|---|---|
| OpenVLCLIP Qwen3-VL-Embedding-2B | 87.58% (8,758/10,000) | 98.68% (9,868/10,000) | 5.26 | 7.94 / 8.06 | 2,048D · 8,192 B/vector | 79.53 MiB | openvlclip==0.7.0 | 23b469da312e | [44ca47e7559432f0](raw/44ca47e7559432f0.json) |
| OpenVLCLIP SigLIP 2 Base/16 FixRes | 57.14% (5,714/10,000) | 94.60% (9,460/10,000) | 293.12 | 1.52 / 1.67 | 768D · 3,072 B/vector | 29.82 MiB | openvlclip==0.7.0 | 23b469da312e | [3595fa7c8edbf265](raw/3595fa7c8edbf265.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | 55.30% (5,530/10,000) | 94.81% (9,481/10,000) | 245.21 | 1.56 / 1.70 | 768D · 3,072 B/vector | 29.82 MiB | openvlclip==0.7.0 | 23b469da312e | [265a5e3112c0d2b4](raw/265a5e3112c0d2b4.json) |
| DINOv2 ViT-L/14 + dino.txt | 98.24% (9,824/10,000) | 99.95% (9,995/10,000) | 80.99 | 3.55 / 3.71 | 2,048D · 8,192 B/vector | 79.53 MiB | openvlclip==0.6.0 | 28519573b78f | [7d2fd70c52399af1](../dinov2-dinotxt-v0.6.0/raw/7d2fd70c52399af1.json) |
| EmbeddingGemma 2 | 94.66% (9,466/10,000) | 99.84% (9,984/10,000) | 10.21 | 4.97 / 5.90 | 768D · 3,072 B/vector | 29.82 MiB | openclip-gemma==0.4.0 | a9909fbe1b15 | [b8214ecb970362c9](../mrl-openclip-gemma-v0.4.0/raw/b8214ecb970362c9.json) |
| OpenAI CLIP ViT-B/32 | 89.83% (8,983/10,000) | 99.61% (9,961/10,000) | 745.47 | 0.62 / 0.68 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [35cef7e32f8afb13](../v0.1.0/raw/35cef7e32f8afb13.json) |
| OpenCLIP ViT-B/32 LAION-2B | 93.56% (9,356/10,000) | 99.81% (9,981/10,000) | 749.63 | 0.61 / 0.68 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [649c7b44763ebc72](../v0.1.0/raw/649c7b44763ebc72.json) |
| OpenCLIP ViT-L/14 LAION-2B | 96.61% (9,661/10,000) | 99.85% (9,985/10,000) | 106.59 | 1.80 / 1.95 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [c8faef3bbd4c2779](../v0.1.0/raw/c8faef3bbd4c2779.json) |
| OpenCLIP Multilingual ViT-B/32 | 93.64% (9,364/10,000) | 99.88% (9,988/10,000) | 701.99 | 1.64 / 1.90 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [931adb237a72e7b9](../v0.1.0/raw/931adb237a72e7b9.json) |
| SigLIP 2 Base/16 224 | 93.62% (9,362/10,000) | 99.85% (9,985/10,000) | 338.26 | 1.52 / 1.67 | not recorded | not recorded | transformers==5.20.1 | 60a61259c95e | [38b1a8be68ae509c](../v0.1.0/raw/38b1a8be68ae509c.json) |

### CIFAR10 · P1 · EN · FP32

| Model | Top-1 (correct/total) | Top-5 (correct/total) | Images/s | Peak allocated/reserved (GiB) | Embedding/vector bytes | FP32 image+text payload | Package | Source commit | Raw result |
|---|---:|---:|---:|---:|---|---:|---|---|---|
| OpenVLCLIP Qwen3-VL-Embedding-2B | 87.08% (8,708/10,000) | 98.68% (9,868/10,000) | 4.95 | 7.94 / 8.06 | 2,048D · 8,192 B/vector | 78.20 MiB | openvlclip==0.7.0 | 23b469da312e | [b602850fd3d77ccd](raw/b602850fd3d77ccd.json) |
| OpenVLCLIP SigLIP 2 Base/16 FixRes | 55.39% (5,539/10,000) | 94.95% (9,495/10,000) | 304.14 | 1.52 / 1.65 | 768D · 3,072 B/vector | 29.33 MiB | openvlclip==0.7.0 | 23b469da312e | [e603831c0064f67a](raw/e603831c0064f67a.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | 78.28% (7,828/10,000) | 98.70% (9,870/10,000) | 240.23 | 1.56 / 1.70 | 768D · 3,072 B/vector | 29.33 MiB | openvlclip==0.7.0 | 23b469da312e | [3d21c3b372fb4491](raw/3d21c3b372fb4491.json) |
| DINOv2 ViT-L/14 + dino.txt | 98.12% (9,812/10,000) | 99.94% (9,994/10,000) | 92.03 | 3.55 / 3.72 | 2,048D · 8,192 B/vector | 78.20 MiB | openvlclip==0.6.0 | 28519573b78f | [0f885e751a90a8ef](../dinov2-dinotxt-v0.6.0/raw/0f885e751a90a8ef.json) |
| EmbeddingGemma 2 | 92.58% (9,258/10,000) | 99.64% (9,964/10,000) | 10.18 | 4.97 / 5.88 | 768D · 3,072 B/vector | 29.33 MiB | openclip-gemma==0.4.0 | a9909fbe1b15 | [6950817bdd99d973](../mrl-openclip-gemma-v0.4.0/raw/6950817bdd99d973.json) |
| OpenAI CLIP ViT-B/32 | 88.30% (8,830/10,000) | 99.24% (9,924/10,000) | 688.74 | 0.62 / 0.68 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [c542f4a0ea8ac9a8](../v0.1.0/raw/c542f4a0ea8ac9a8.json) |
| OpenCLIP ViT-B/32 LAION-2B | 93.68% (9,368/10,000) | 99.84% (9,984/10,000) | 717.66 | 0.61 / 0.66 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [c3350bf4370a58b8](../v0.1.0/raw/c3350bf4370a58b8.json) |
| OpenCLIP ViT-L/14 LAION-2B | 96.92% (9,692/10,000) | 99.93% (9,993/10,000) | 106.76 | 1.80 / 1.90 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [9c075a8286dd1604](../v0.1.0/raw/9c075a8286dd1604.json) |
| OpenCLIP Multilingual ViT-B/32 | 93.54% (9,354/10,000) | 99.86% (9,986/10,000) | 739.21 | 1.54 / 1.71 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [013541e887452604](../v0.1.0/raw/013541e887452604.json) |
| SigLIP 2 Base/16 224 | 93.85% (9,385/10,000) | 99.88% (9,988/10,000) | 336.90 | 1.52 / 1.65 | not recorded | not recorded | transformers==5.20.1 | 60a61259c95e | [ddcae56da141ed7f](../v0.1.0/raw/ddcae56da141ed7f.json) |

### CIFAR100 · P0 · EN · FP32

| Model | Top-1 (correct/total) | Top-5 (correct/total) | Images/s | Peak allocated/reserved (GiB) | Embedding/vector bytes | FP32 image+text payload | Package | Source commit | Raw result |
|---|---:|---:|---:|---:|---|---:|---|---|---|
| OpenVLCLIP Qwen3-VL-Embedding-2B | 57.66% (5,766/10,000) | 80.99% (8,099/10,000) | 8.45 | 7.94 / 8.06 | 2,048D · 8,192 B/vector | 92.19 MiB | openvlclip==0.7.0 | 23b469da312e | [1e814e0b64df8a09](raw/1e814e0b64df8a09.json) |
| OpenVLCLIP SigLIP 2 Base/16 FixRes | 24.45% (2,445/10,000) | 51.26% (5,126/10,000) | 306.16 | 1.52 / 1.66 | 768D · 3,072 B/vector | 34.57 MiB | openvlclip==0.7.0 | 23b469da312e | [d7302c99a73e813f](raw/d7302c99a73e813f.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | 29.78% (2,978/10,000) | 57.00% (5,700/10,000) | 250.65 | 1.56 / 1.70 | 768D · 3,072 B/vector | 34.57 MiB | openvlclip==0.7.0 | 23b469da312e | [d35c9c3f4583635a](raw/d35c9c3f4583635a.json) |
| DINOv2 ViT-L/14 + dino.txt | 89.71% (8,971/10,000) | 98.17% (9,817/10,000) | 91.73 | 3.55 / 3.71 | 2,048D · 8,192 B/vector | 92.19 MiB | openvlclip==0.6.0 | 28519573b78f | [f0b3f208c3f8f151](../dinov2-dinotxt-v0.6.0/raw/f0b3f208c3f8f151.json) |
| EmbeddingGemma 2 | 76.08% (7,608/10,000) | 92.92% (9,292/10,000) | 10.19 | 4.97 / 5.94 | 768D · 3,072 B/vector | 34.57 MiB | openclip-gemma==0.4.0 | a9909fbe1b15 | [b08c9f0fbe719961](../mrl-openclip-gemma-v0.4.0/raw/b08c9f0fbe719961.json) |
| OpenAI CLIP ViT-B/32 | 64.20% (6,420/10,000) | 88.73% (8,873/10,000) | 668.75 | 0.62 / 0.68 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [5a8886b0fab54729-30d1b401](../v0.1.0/raw/5a8886b0fab54729-30d1b401.json) |
| OpenCLIP ViT-B/32 LAION-2B | 75.56% (7,556/10,000) | 93.87% (9,387/10,000) | 701.07 | 0.61 / 0.68 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [35dfb13b0aa46c57-54e65389](../v0.1.0/raw/35dfb13b0aa46c57-54e65389.json) |
| OpenCLIP ViT-L/14 LAION-2B | 83.34% (8,334/10,000) | 96.63% (9,663/10,000) | 106.97 | 1.80 / 1.94 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [1de2d2b7854b5e80-30c3f626](../v0.1.0/raw/1de2d2b7854b5e80-30c3f626.json) |
| OpenCLIP Multilingual ViT-B/32 | 76.55% (7,655/10,000) | 94.99% (9,499/10,000) | 750.93 | 1.64 / 1.90 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [e4c4922a8b419fc2-ac4a43b4](../v0.1.0/raw/e4c4922a8b419fc2-ac4a43b4.json) |
| SigLIP 2 Base/16 224 | 73.68% (7,368/10,000) | 91.89% (9,189/10,000) | 342.26 | 1.52 / 1.66 | not recorded | not recorded | transformers==5.20.1 | 60a61259c95e | [d59181f45a441ab9-1a33071f](../v0.1.0/raw/d59181f45a441ab9-1a33071f.json) |

### CIFAR100 · P1 · EN · FP32

| Model | Top-1 (correct/total) | Top-5 (correct/total) | Images/s | Peak allocated/reserved (GiB) | Embedding/vector bytes | FP32 image+text payload | Package | Source commit | Raw result |
|---|---:|---:|---:|---:|---|---:|---|---|---|
| OpenVLCLIP Qwen3-VL-Embedding-2B | 57.30% (5,730/10,000) | 79.31% (7,931/10,000) | 4.12 | 7.94 / 8.06 | 2,048D · 8,192 B/vector | 78.91 MiB | openvlclip==0.7.0 | 23b469da312e | [10a1bb028a2f5318](raw/10a1bb028a2f5318.json) |
| OpenVLCLIP SigLIP 2 Base/16 FixRes | 13.13% (1,313/10,000) | 40.02% (4,002/10,000) | 304.31 | 1.52 / 1.67 | 768D · 3,072 B/vector | 29.59 MiB | openvlclip==0.7.0 | 23b469da312e | [5fb3917ec73989e4](raw/5fb3917ec73989e4.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | 25.50% (2,550/10,000) | 59.91% (5,991/10,000) | 241.80 | 1.56 / 1.70 | 768D · 3,072 B/vector | 29.59 MiB | openvlclip==0.7.0 | 23b469da312e | [35f85add213561f9](raw/35f85add213561f9.json) |
| DINOv2 ViT-L/14 + dino.txt | 89.14% (8,914/10,000) | 98.13% (9,813/10,000) | 93.22 | 3.55 / 3.71 | 2,048D · 8,192 B/vector | 78.91 MiB | openvlclip==0.6.0 | 28519573b78f | [37ce26e54de28445](../dinov2-dinotxt-v0.6.0/raw/37ce26e54de28445.json) |
| EmbeddingGemma 2 | 62.84% (6,284/10,000) | 84.04% (8,404/10,000) | 10.25 | 4.97 / 5.90 | 768D · 3,072 B/vector | 29.59 MiB | openclip-gemma==0.4.0 | a9909fbe1b15 | [820f9a84ba7355e3](../mrl-openclip-gemma-v0.4.0/raw/820f9a84ba7355e3.json) |
| OpenAI CLIP ViT-B/32 | 62.32% (6,232/10,000) | 87.00% (8,700/10,000) | 723.23 | 0.62 / 0.68 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [c08f92332ebe01a4-2815d62c](../v0.1.0/raw/c08f92332ebe01a4-2815d62c.json) |
| OpenCLIP ViT-B/32 LAION-2B | 75.59% (7,559/10,000) | 93.65% (9,365/10,000) | 773.59 | 0.61 / 0.68 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [a54f8b8dfb17227b-8e3ad59c](../v0.1.0/raw/a54f8b8dfb17227b-8e3ad59c.json) |
| OpenCLIP ViT-L/14 LAION-2B | 83.07% (8,307/10,000) | 96.57% (9,657/10,000) | 106.15 | 1.80 / 1.95 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [5cd03bacdf1e5b06-43791242](../v0.1.0/raw/5cd03bacdf1e5b06-43791242.json) |
| OpenCLIP Multilingual ViT-B/32 | 76.46% (7,646/10,000) | 95.02% (9,502/10,000) | 748.05 | 1.64 / 1.90 | not recorded | not recorded | open-clip-torch==3.3.0 | 60a61259c95e | [d4721f7c31f9d23c-e083b306](../v0.1.0/raw/d4721f7c31f9d23c-e083b306.json) |
| SigLIP 2 Base/16 224 | 73.25% (7,325/10,000) | 91.71% (9,171/10,000) | 338.32 | 1.52 / 1.67 | not recorded | not recorded | transformers==5.20.1 | 60a61259c95e | [38a8be7d4b997457-8dbcaf34](../v0.1.0/raw/38a8be7d4b997457-8dbcaf34.json) |

### CROSSMODAL3600 · P2 · EN · FP32 · official_first_caption

| Model | T→I R@1 | T→I R@5 | T→I R@10 | I→T R@1 | I→T R@5 | I→T R@10 | Images/s | Texts/s | Peak allocated/reserved (GiB) | Embedding/vector bytes | FP32 image+text payload | Package | Source commit | Raw result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|
| OpenVLCLIP Qwen3-VL-Embedding-2B | 55.78% (2,008/3,600) | 79.58% (2,865/3,600) | 86.67% (3,120/3,600) | 57.94% (2,086/3,600) | 82.31% (2,963/3,600) | 89.25% (3,213/3,600) | 4.33 | 17.21 | 8.01 / 8.15 | 2,048D · 8,192 B/vector | 56.25 MiB | openvlclip==0.7.0 | 23b469da312e | [026e74e2db5b6144](raw/026e74e2db5b6144.json) |
| OpenVLCLIP SigLIP 2 Base/16 FixRes | 0.19% (7/3,600) | 0.89% (32/3,600) | 1.75% (63/3,600) | 0.94% (34/3,600) | 2.56% (92/3,600) | 4.03% (145/3,600) | 234.79 | 7.29 | 1.52 / 1.65 | 768D · 3,072 B/vector | 21.09 MiB | openvlclip==0.7.0 | 23b469da312e | [4008d7997a0def30](raw/4008d7997a0def30.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | 0.36% (13/3,600) | 1.97% (71/3,600) | 3.44% (124/3,600) | 1.36% (49/3,600) | 4.19% (151/3,600) | 6.94% (250/3,600) | 162.54 | 5.54 | 1.56 / 1.69 | 768D · 3,072 B/vector | 21.09 MiB | openvlclip==0.7.0 | 23b469da312e | [deef20de60c0e3a1](raw/deef20de60c0e3a1.json) |
| DINOv2 ViT-L/14 + dino.txt | 51.50% (1,854/3,600) | 77.31% (2,783/3,600) | 84.64% (3,047/3,600) | 53.03% (1,909/3,600) | 77.33% (2,784/3,600) | 85.44% (3,076/3,600) | 97.47 | 206.63 | 3.55 / 3.69 | 2,048D · 8,192 B/vector | 56.25 MiB | openvlclip==0.6.0 | 28519573b78f | [d8c51bd44857ee04](../dinov2-dinotxt-v0.6.0/raw/d8c51bd44857ee04.json) |
| EmbeddingGemma 2 | 58.61% (2,110/3,600) | 82.06% (2,954/3,600) | 89.19% (3,211/3,600) | 59.50% (2,142/3,600) | 85.17% (3,066/3,600) | 91.28% (3,286/3,600) | 7.52 | 208.51 | 4.29 / 6.31 | 768D · 3,072 B/vector | 21.09 MiB | openclip-gemma==0.4.0 | a9909fbe1b15 | [02b17361f1a3a0ef](../mrl-openclip-gemma-v0.4.0/raw/02b17361f1a3a0ef.json) |
| OpenCLIP Multilingual ViT-B/32 | 47.08% (1,695/3,600) | 74.86% (2,695/3,600) | 83.03% (2,989/3,600) | 48.31% (1,739/3,600) | 76.64% (2,759/3,600) | 85.19% (3,067/3,600) | 324.41 | 148.03 | 1.64 / 1.91 | not recorded | not recorded | open-clip-torch==3.3.0 | 9d170b6071ee | [b5b51e1810e20bc1](../v0.1.0/raw/b5b51e1810e20bc1.json) |
| SigLIP 2 Base/16 224 | 51.47% (1,853/3,600) | 77.03% (2,773/3,600) | 84.72% (3,050/3,600) | 50.67% (1,824/3,600) | 76.67% (2,760/3,600) | 85.00% (3,060/3,600) | 304.09 | 703.03 | 1.52 / 1.65 | not recorded | not recorded | transformers==5.20.1 | 9d170b6071ee | [25b6a90b6a8e0c82](../v0.1.0/raw/25b6a90b6a8e0c82.json) |

### CROSSMODAL3600 · P2 · JA · FP32 · official_first_caption · exploratory

| Model | T→I R@1 | T→I R@5 | T→I R@10 | I→T R@1 | I→T R@5 | I→T R@10 | Images/s | Texts/s | Peak allocated/reserved (GiB) | Embedding/vector bytes | FP32 image+text payload | Package | Source commit | Raw result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|
| OpenVLCLIP Qwen3-VL-Embedding-2B | 77.81% (2,801/3,600) | 93.58% (3,369/3,600) | 96.47% (3,473/3,600) | 72.25% (2,601/3,600) | 91.22% (3,284/3,600) | 95.17% (3,426/3,600) | 4.21 | 15.13 | 8.01 / 8.15 | 2,048D · 8,192 B/vector | 56.25 MiB | openvlclip==0.7.0 | 23b469da312e | [a123010f08059e2f](raw/a123010f08059e2f.json) |
| OpenVLCLIP SigLIP 2 Base/16 FixRes | 0.39% (14/3,600) | 1.58% (57/3,600) | 2.83% (102/3,600) | 1.39% (50/3,600) | 3.39% (122/3,600) | 5.00% (180/3,600) | 256.78 | 5.24 | 1.52 / 1.65 | 768D · 3,072 B/vector | 21.09 MiB | openvlclip==0.7.0 | 23b469da312e | [b0d09dd268d8159e](raw/b0d09dd268d8159e.json) |
| OpenVLCLIP SigLIP 2 Base/16 NaFlex | 0.44% (16/3,600) | 1.22% (44/3,600) | 2.33% (84/3,600) | 0.81% (29/3,600) | 2.33% (84/3,600) | 3.31% (119/3,600) | 181.66 | 5.43 | 1.56 / 1.69 | 768D · 3,072 B/vector | 21.09 MiB | openvlclip==0.7.0 | 23b469da312e | [ea34cf1a5133f4c7](raw/ea34cf1a5133f4c7.json) |
| DINOv2 ViT-L/14 + dino.txt | 2.03% (73/3,600) | 5.14% (185/3,600) | 7.75% (279/3,600) | 3.17% (114/3,600) | 8.00% (288/3,600) | 11.97% (431/3,600) | 96.74 | 211.50 | 3.55 / 3.69 | 2,048D · 8,192 B/vector | 56.25 MiB | openvlclip==0.6.0 | 28519573b78f | [347bf1161cfb1180](../dinov2-dinotxt-v0.6.0/raw/347bf1161cfb1180.json) |
| EmbeddingGemma 2 | 80.14% (2,885/3,600) | 95.03% (3,421/3,600) | 97.50% (3,510/3,600) | 79.61% (2,866/3,600) | 95.28% (3,430/3,600) | 97.61% (3,514/3,600) | 7.54 | 204.05 | 4.29 / 6.31 | 768D · 3,072 B/vector | 21.09 MiB | openclip-gemma==0.4.0 | a9909fbe1b15 | [5e1276cab5ecb4e8](../mrl-openclip-gemma-v0.4.0/raw/5e1276cab5ecb4e8.json) |
| OpenCLIP Multilingual ViT-B/32 | 63.83% (2,298/3,600) | 86.86% (3,127/3,600) | 92.94% (3,346/3,600) | 67.92% (2,445/3,600) | 90.58% (3,261/3,600) | 95.39% (3,434/3,600) | 319.62 | 148.01 | 1.64 / 1.91 | not recorded | not recorded | open-clip-torch==3.3.0 | 9d170b6071ee | [ae4e1863504978e0](../v0.1.0/raw/ae4e1863504978e0.json) |
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
uv run --locked vlbench run --model openvlclip_siglip2_naflex_base --dataset cifar10 --split test --protocol p0 --language en --precision fp32 --device cuda --batch-size 16 --seed 0 --strict-reproducibility --expected-source-commit 23b469da312efc17f64aae7119b74ceeeb10f42d --yes --output .cache/benchmark-staging/openvlclip-v0.7.0
uv run --locked vlbench run --model openvlclip_siglip2_base --dataset cifar10 --split test --protocol p0 --language en --precision fp32 --device cuda:0 --batch-size 16 --seed 0 --strict-reproducibility --expected-source-commit 23b469da312efc17f64aae7119b74ceeeb10f42d --yes --output .cache/benchmark-staging/openvlclip-v0.7.0
uv run --locked vlbench run --model openvlclip_qwen3_vl_embedding_2b --dataset cifar10 --split test --protocol p0 --language en --precision fp32 --device cuda --batch-size 1 --seed 0 --strict-reproducibility --expected-source-commit 23b469da312efc17f64aae7119b74ceeeb10f42d --yes --output .cache/benchmark-staging/openvlclip-v0.7.0
uv run --locked vlbench run --model openvlclip_siglip2_naflex_base --dataset cifar10 --split test --protocol p1 --language en --precision fp32 --device cuda --batch-size 16 --seed 0 --strict-reproducibility --expected-source-commit 23b469da312efc17f64aae7119b74ceeeb10f42d --yes --output .cache/benchmark-staging/openvlclip-v0.7.0
uv run --locked vlbench run --model openvlclip_qwen3_vl_embedding_2b --dataset cifar10 --split test --protocol p1 --language en --precision fp32 --device cuda --batch-size 1 --seed 0 --strict-reproducibility --expected-source-commit 23b469da312efc17f64aae7119b74ceeeb10f42d --yes --output .cache/benchmark-staging/openvlclip-v0.7.0
uv run --locked vlbench run --model openvlclip_siglip2_base --dataset cifar10 --split test --protocol p1 --language en --precision fp32 --device cuda:0 --batch-size 16 --seed 0 --strict-reproducibility --expected-source-commit 23b469da312efc17f64aae7119b74ceeeb10f42d --yes --output .cache/benchmark-staging/openvlclip-v0.7.0
uv run --locked vlbench run --model openvlclip_qwen3_vl_embedding_2b --dataset cifar100 --split test --protocol p0 --language en --precision fp32 --device cuda --batch-size 1 --seed 0 --strict-reproducibility --expected-source-commit 23b469da312efc17f64aae7119b74ceeeb10f42d --yes --output .cache/benchmark-staging/openvlclip-v0.7.0
uv run --locked vlbench run --model openvlclip_siglip2_naflex_base --dataset cifar100 --split test --protocol p0 --language en --precision fp32 --device cuda --batch-size 16 --seed 0 --strict-reproducibility --expected-source-commit 23b469da312efc17f64aae7119b74ceeeb10f42d --yes --output .cache/benchmark-staging/openvlclip-v0.7.0
uv run --locked vlbench run --model openvlclip_siglip2_base --dataset cifar100 --split test --protocol p0 --language en --precision fp32 --device cuda:0 --batch-size 16 --seed 0 --strict-reproducibility --expected-source-commit 23b469da312efc17f64aae7119b74ceeeb10f42d --yes --output .cache/benchmark-staging/openvlclip-v0.7.0
uv run --locked vlbench run --model openvlclip_qwen3_vl_embedding_2b --dataset cifar100 --split test --protocol p1 --language en --precision fp32 --device cuda --batch-size 1 --seed 0 --strict-reproducibility --expected-source-commit 23b469da312efc17f64aae7119b74ceeeb10f42d --yes --output .cache/benchmark-staging/openvlclip-v0.7.0
uv run --locked vlbench run --model openvlclip_siglip2_naflex_base --dataset cifar100 --split test --protocol p1 --language en --precision fp32 --device cuda --batch-size 16 --seed 0 --strict-reproducibility --expected-source-commit 23b469da312efc17f64aae7119b74ceeeb10f42d --yes --output .cache/benchmark-staging/openvlclip-v0.7.0
uv run --locked vlbench run --model openvlclip_siglip2_base --dataset cifar100 --split test --protocol p1 --language en --precision fp32 --device cuda:0 --batch-size 16 --seed 0 --strict-reproducibility --expected-source-commit 23b469da312efc17f64aae7119b74ceeeb10f42d --yes --output .cache/benchmark-staging/openvlclip-v0.7.0
uv run --locked vlbench run --model openvlclip_qwen3_vl_embedding_2b --dataset crossmodal3600 --split 'all 3,600 images' --protocol p2 --language en --precision fp32 --device cuda --batch-size 1 --seed 0 --retrieval-variant official_first_caption --strict-reproducibility --expected-source-commit 23b469da312efc17f64aae7119b74ceeeb10f42d --yes --output .cache/benchmark-staging/openvlclip-v0.7.0
uv run --locked vlbench run --model openvlclip_siglip2_base --dataset crossmodal3600 --split 'all 3,600 images' --protocol p2 --language en --precision fp32 --device cuda:0 --batch-size 16 --seed 0 --retrieval-variant official_first_caption --strict-reproducibility --expected-source-commit 23b469da312efc17f64aae7119b74ceeeb10f42d --yes --output .cache/benchmark-staging/openvlclip-v0.7.0
uv run --locked vlbench run --model openvlclip_siglip2_naflex_base --dataset crossmodal3600 --split 'all 3,600 images' --protocol p2 --language en --precision fp32 --device cuda --batch-size 16 --seed 0 --retrieval-variant official_first_caption --strict-reproducibility --expected-source-commit 23b469da312efc17f64aae7119b74ceeeb10f42d --yes --output .cache/benchmark-staging/openvlclip-v0.7.0
uv run --locked vlbench run --model openvlclip_qwen3_vl_embedding_2b --dataset crossmodal3600 --split 'all 3,600 images' --protocol p2 --language ja --precision fp32 --device cuda --batch-size 1 --seed 0 --retrieval-variant official_first_caption --strict-reproducibility --expected-source-commit 23b469da312efc17f64aae7119b74ceeeb10f42d --yes --output .cache/benchmark-staging/openvlclip-v0.7.0
uv run --locked vlbench run --model openvlclip_siglip2_base --dataset crossmodal3600 --split 'all 3,600 images' --protocol p2 --language ja --precision fp32 --device cuda:0 --batch-size 16 --seed 0 --retrieval-variant official_first_caption --strict-reproducibility --expected-source-commit 23b469da312efc17f64aae7119b74ceeeb10f42d --yes --output .cache/benchmark-staging/openvlclip-v0.7.0
uv run --locked vlbench run --model openvlclip_siglip2_naflex_base --dataset crossmodal3600 --split 'all 3,600 images' --protocol p2 --language ja --precision fp32 --device cuda --batch-size 16 --seed 0 --retrieval-variant official_first_caption --strict-reproducibility --expected-source-commit 23b469da312efc17f64aae7119b74ceeeb10f42d --yes --output .cache/benchmark-staging/openvlclip-v0.7.0
```

### Regenerate this report only (no inference)

This command reads existing raw JSON and regenerates summaries, plots, and validation outputs; it does not load model weights or run inference.

```powershell
uv run --locked vlbench report --input results/openvlclip-v0.7.0/raw --output results/openvlclip-v0.7.0 --publishable --lifecycle results/openvlclip-v0.7.0/validation/result-lifecycle.json --expected-source-commit 23b469da312efc17f64aae7119b74ceeeb10f42d --comparison-input results/v0.1.0/raw --comparison-lifecycle results/v0.1.0/validation/result-lifecycle.json --comparison-input results/mrl-openclip-gemma-v0.4.0/raw --comparison-lifecycle results/mrl-openclip-gemma-v0.4.0/validation/result-lifecycle.json --comparison-input results/dinov2-dinotxt-v0.6.0/raw --comparison-lifecycle results/dinov2-dinotxt-v0.6.0/validation/result-lifecycle.json
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
