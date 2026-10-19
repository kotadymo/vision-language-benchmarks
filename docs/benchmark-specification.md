# Vision-Language Benchmarks — Benchmark Specification

- **Document version**: 0.1.0 (implementation and evaluation specification)
- **Created**: 2026-10-19
- **Repository**: `kotadymo/vision-language-benchmarks`
- **Repository URL**: https://github.com/kotadymo/vision-language-benchmarks
- **Development environment**: Windows 11 / PowerShell / Python 3.12 / uv / VS Code / GitHub Copilot
- **Primary execution environment**: NVIDIA GeForce RTX 4070 Ti SUPER (16 GB VRAM)
- **Reference framework**: [LAION-AI/CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark)
- **Status**: Implementation specification. No measured benchmark results have been published.

## 1. Purpose and principles

This project compares pretrained vision-language models—including EmbeddingGemma 2 via `openclip-gemma`, CLIP, OpenCLIP, SigLIP 2, and Japanese CLIP—on English and Japanese image classification and image-text retrieval tasks. It publishes reproducible evaluation code, metadata, measurements, and visualizations.

1. **Standards alignment**: Prefer CLIP Benchmark datasets, splits, class names, templates, metrics, and result JSON conventions. Document every deviation.
2. **Fairness**: Within a task, use the same data split, evaluation images, positive pairs, candidate pool, and measurement definition. Preserve each model's official preprocessing and recommended input format.
3. **Transparency**: Record `model_id`, pretrained tag, model revision, dependency versions, split, prompts, language, dtype, GPU, seed, and Git commit SHA.
4. **Separation of concerns**: Do not conflate model accuracy, `openclip-gemma` wrapper API compatibility, numerical parity, or computational performance.
5. **Accuracy**: Never fabricate unmeasured values, report failures as successes, hide skipped runs, or silently aggregate results from different protocols.
6. **Reproducibility**: Generate CSV, Markdown, SVG, and PNG repeatedly from the same authoritative JSON results.
7. **Safety**: Never commit datasets, model weights, Hugging Face tokens, or large embedding files to GitHub.

**Out of scope for the initial release**: model training or fine-tuning, new rankings based on custom scores, generative VLM question answering, video/audio models, all MIEB tasks, and unverified transcription of published benchmark results.

## 2. Baseline and implementation approach

- Follow CLIP Benchmark for baseline result definitions. The reference implementation is pinned to commit `ee14a9a8a9bcde09d51f4a4b1dff5a75182a3596`.
- Prefer the official evaluation path for models supported by CLIP Benchmark. For other models, provide an adapter and verify whether the official dataset and metric code can be reused.
- If CLIP Benchmark cannot be extended directly, do not patch unrelated external repositories. Implement an independent wrapper with equivalent evaluation functions and verify numerical parity against an existing CLIP model using the official implementation.
- Pin `openclip-gemma==0.3.0` for the initial comparison. Use EmbeddingGemma 2 revision `914f7f89142e33e77833254d9c9b90c3cef7303b`; do not silently update it after the initial evaluation.
- Perform inference only. In general, use `model.eval()`, `torch.inference_mode()`, L2-normalized features, and cosine similarity. Do not compare OpenCLIP's trained `logit_scale` or EmbeddingGemma's absolute scores across model families.

### 2.1 Evaluation protocols

- **P0 / CLIP Benchmark-compatible**: Match the official dataset and split, labels/templates, and aggregation definitions. Record any model-specific tokenizer or required task prefix. Do not claim equivalence to official scores when execution conditions differ.
- **P1 / Model-native**: Use prompts or templates recommended by each model card as a reference comparison. Keep P0 and P1 in separate tables, figures, and rankings.
- **P2 / Japanese versus English**: Compare Japanese and English on the same dataset and identical image/ID set. Report translation-assisted experiments separately as `translation_assisted`.
- Use FP32 for primary accuracy results. Report BF16/AMP separately and state which models support it.

## 3. Candidate comparison models

Before evaluation, verify each checkpoint's existence, license, usage conditions, revision, model size, and CUDA inference compatibility, then record the fixed values in results. **The identifiers below are candidates, not claims that the models have been verified or run.**

| Proposed key | Model/checkpoint | Encoder implementation | Initial phase | Main purpose |
| --- | --- | --- | --- | --- |
| `embeddinggemma2` | `google/embeddinggemma-2`, `openclip-gemma==0.3.0` | `openclip_gemma` | Phase 2 | Primary model; English/Japanese |
| `openai_clip_b32` | OpenCLIP `ViT-B-32`, pretrained=`openai` | `open_clip` | Phase 1 | Standard baseline and parity |
| `openclip_b32_laion2b` | OpenCLIP `ViT-B-32`, pretrained=`laion2b_s34b_b79k` | `open_clip` | Phase 2 | Small OpenCLIP |
| `openclip_l14_laion2b` | OpenCLIP `ViT-L-14`, pretrained=`laion2b_s32b_b82k` | `open_clip` | Phase 2 | Large OpenCLIP |
| `rinna_ja_clip_b16` | `rinna/japanese-clip-vit-b-16` | CLIP Benchmark `ja_clip` or official `japanese_clip` | Phase 3 | Japanese-specialized CLIP |
| `openclip_multilingual_b32` | OpenCLIP `xlm-roberta-base-ViT-B-32`, pretrained=`laion5b_s13b_b90k` | `open_clip` | Phase 3 | Multilingual OpenCLIP |
| `siglip2_base_224` | `google/siglip2-base-patch16-224` | Transformers | Phase 2/3 | SigLIP 2 (English/Japanese) |
| `japanese_stable_clip_l16` | `stabilityai/japanese-stable-clip-vit-l-16` | Official code (to be investigated) | Optional candidate | Large Japanese model |

**Requirements**:

- Record `model_id`, pretrained tag, Hugging Face model/tokenizer revision SHA, implementation-library version, input resolution, image preprocessing, and maximum text length.
- Compare the official `rinna` implementation with the CLIP Benchmark implementation and choose the best-supported adapter. If dependencies conflict, use an isolated environment or subprocess.
- Do not apply CLIP `logit_scale` assumptions to SigLIP 2. Compare rankings using its official processor and normalized features.
- Never automatically accept gated-model terms or trust unverified remote code. The user must review and accept applicable terms.
- Show model size and language support in primary tables. Do not present different-sized models as a same-size competition.

## 4. English evaluation datasets

| Key | Intended use/split | Metrics | Initial phase |
| --- | --- | --- | --- |
| `cifar10` | CIFAR-10 test split, 10 classes | Top-1 / Top-5 accuracy | Phase 1 |
| `cifar100` | CIFAR-100 test split, 100 classes | Top-1 / Top-5 accuracy | Phase 1 |
| `flickr30k` | Flickr30k retrieval test (record the selected Karpathy split and count) | T→I and I→T Recall@1/5/10 | Phase 1 |
| `mscoco_captions` | MS COCO Captions / Karpathy 5K test (do not confuse with a separate 1K variant) | T→I and I→T Recall@1/5/10 | Phase 2 |
| `imagenet1k` | ImageNet-1K validation | Top-1 / Top-5 accuracy | Phase 2 |
| `imagenet-a`, `imagenet-r`, `imagenet-sketch` | Robustness evaluation on official splits | Applicable classification accuracy | Phase 4 |
| `sugar_crepe` | Specified compositionality split | Official metric | Phase 4 |

- Record ground-truth labels, captions per image, ID ordering, and split version. Treat broken image/text correspondence as an evaluation failure.
- Check access restrictions for ImageNet, Flickr30k, and other datasets. Use only locally available copies or documented downloads permitted by their terms. **Do not redistribute data with unclear licensing.**
- Never mix small smoke-test subsets into official full-dataset result tables. Label every subset and record its sample count.

## 5. Japanese and multilingual evaluation datasets

| Key | Languages | Content and evaluation | Initial phase |
| --- | --- | --- | --- |
| `xtd10` | `jp`, `en`, etc. (CLIP Benchmark codes) | Multilingual retrieval over approximately 1,000 image/caption pairs; T→I and I→T R@1/5/10 | Phase 3, highest priority |
| `crossmodal3600` (XM3600) | `ja`, `en`, etc. | Compare Japanese and English retrieval over approximately 3,600 images | Phase 3, highest priority |
| `multilingual_mscoco_captions` | `jp`, `en`, etc. | Multilingual MS COCO retrieval as defined by CLIP Benchmark | Phase 3, additional |
| `babel_imagenet` | `ja`, `en` | Classification on the explicitly validated set of classes with Japanese Babel-ImageNet labels; report class count and label coverage | Phase 3, additional |

**Shared and overlapping data**:

- XTD10 and CLIP Benchmark's `multilingual_mscoco_captions` may use the same XTD10 images and captions. Do not average them as independent evaluations or add their sample counts. Compare image IDs, annotation sources, and checksums; if they are identical, report one representative dataset.
- XTD10 uses `jp` for Japanese while XM3600 uses `ja`. Normalize to ISO `ja` internally only if each dataset adapter preserves the required original identifier.
- Do not assume Japanese labels exist for all 1,000 ImageNet classes. Define the Japanese-labeled class set and compare English on the same classes and images. Report this separately from standard 1,000-class ImageNet-1K.
- Japanese Stable CLIP was trained using MS COCO and STAIR; state the training-data overlap risk for COCO-derived evaluations. Check potential training overlap for other models as well.
- Mark models that cannot accept Japanese as `unsupported`; never record them as 0% accuracy.

## 6. Exact metric definitions and aggregation

### 6.1 Zero-shot classification

- `acc1`: Fraction of all evaluated images for which the correct class is the highest-scoring class.
- `acc5`: Fraction for which the correct class is among the five highest-scoring classes. For tasks with fewer than five classes, follow the official convention.
- Reproduce CLIP Benchmark class names and multi-template handling. Verify and fix the order of template embedding averaging and normalization against the official source.
- The primary evaluation selects from all candidate classes using zero-shot inference. Do not train on test labels, tune thresholds, or search prompts using test labels.
- Store metrics in JSON as fractions in `[0, 1]`. Clearly label units when converting to percentages in CSVs and figures.

### 6.2 Bidirectional retrieval

- `image_retrieval_recall@1/5/10`: **Text query → image retrieval (T→I)**; whether a correct image occurs in the top K.
- `text_retrieval_recall@1/5/10`: **Image query → text retrieval (I→T)**; whether a correct caption occurs in the top K.
- Use these CLIP Benchmark metric keys and directions. Figures must explicitly label `T→I` and `I→T`.
- When an image has multiple captions, an image query succeeds if at least one of its positive captions is in the top K. When a text query has multiple positive images, follow the evaluation set's ground-truth relations.
- Compute the macro average over all queries in each direction, giving each query equal weight. Never silently remove queries with missing positives.
- Explicitly handle `K > candidates`, tied scores, invalid inputs (NaN/Inf), and zero-norm features while preserving evaluation completeness.
- Do not allocate the full similarity matrix on the GPU when unnecessary. Use blockwise exact top-K while preserving the complete candidate set. Approximate search is out of scope for the initial specification.

### 6.3 Classification and retrieval prompts

- P0 uses CLIP Benchmark evaluation data, class names, and template sets. Record their IDs or content hashes.
- EmbeddingGemma 2 classification uses the `Classification` task prompt; caption/query retrieval uses `SearchQuery`. **Do not add task prefixes to image inputs.**
- P1 uses model-recommended templates/prompts. Report differences from P0 separately and do not optimize prompts using test labels.
- Use the official tokenizer/processor for rinna and SigLIP 2. Do not accidentally add duplicate text prefixes.
- Clearly distinguish any additional “translate Japanese to English, then use an English model” experiment from a model's native Japanese zero-shot performance.

## 7. RTX 4070 Ti SUPER / CUDA execution requirements

1. Use Windows 11 / Python 3.12 / uv. Verify the CUDA-enabled PyTorch and torchvision combination at runtime and record it in `uv.lock` and `docs/environment.md`. Do not pin an unverified CUDA index URL.
2. Detect the GPU with `torch.cuda.is_available()`. Verify `torch.cuda.get_device_name(0)` is the RTX 4070 Ti SUPER; collect GPU memory and driver version. Record `nvidia-smi` information too.
3. Use FP32 for primary accuracy. Run BF16 separately only when `torch.cuda.is_bf16_supported()` and the model supports it. Explicitly record AMP use.
4. Load one model at a time. On model changes, release references, collect garbage, and clear recoverable CUDA cache. Start with batch size 16; on OOM, retry at 8, 4, 2, then 1. **Record the final batch size and OOM history.**
5. For models that cannot form fixed-size image batches (e.g. variable resolution), use a model-safe collator or batch size 1. Do not apply another model's preprocessing.
6. CPU fallback is allowed, but record it separately as `device=cpu`. Never silently switch to CPU when a requested CUDA device is unavailable; allow fallback only through an explicit option.
7. Use `torch.inference_mode()` and `torch.cuda.synchronize()` around GPU timing. Warm up first, then report medians and measurement variability across repeated measurements.
8. Prefer separate accuracy and speed runs. Distinguish download, model loading, dataset loading, preprocessing, GPU inference, and retrieval aggregation.
9. Distinguish PyTorch peak allocated/reserved memory and, where possible, process memory from NVML. Use consistent MiB/GB units.
10. If hardware or dependency constraints prevent a run, record it as `skipped` with a reason. Never fill in an estimated accuracy.

## 8. Implementation modules and public interfaces

Develop under the existing `src/vision_language_benchmarks/` package. A recommended layout is:

```text
vision-language-benchmarks/
├── configs/
│   ├── models.yaml
│   ├── datasets.yaml
│   └── benchmark.yaml
├── docs/
│   ├── benchmark-specification.md
│   ├── methodology.md
│   ├── environment.md
│   ├── datasets-and-licenses.md
│   └── reproducibility.md
├── src/vision_language_benchmarks/
│   ├── adapters/
│   │   ├── base.py
│   │   ├── openclip.py
│   │   ├── openclip_gemma.py
│   │   ├── japanese_clip.py
│   │   └── siglip2.py
│   ├── datasets/
│   │   ├── registry.py
│   │   └── downloader.py
│   ├── evaluation/
│   │   ├── classification.py
│   │   ├── retrieval.py
│   │   └── protocols.py
│   ├── reporting/
│   │   ├── schema.py
│   │   ├── aggregate.py
│   │   └── plots.py
│   ├── hardware.py
│   └── cli.py
├── tests/
├── scripts/
├── results/
│   ├── README.md
│   └── v0.1.0/
│       ├── raw/               # Only publish numeric result JSON files
│       ├── summary.csv
│       ├── report.md
│       ├── metadata.json
│       └── plots/
├── pyproject.toml
└── uv.lock
```

### 8.1 Model adapters

- Define a common contract equivalent to `load(model_spec, device, dtype)`, `preprocess_image(image)`, `encode_images(batch)`, `encode_texts(texts, task, language)`, and `metadata()`.
- Preserve image-ID and text-ID order. Return finite `[N, D]` tensors; dimensions may differ by model. Do not reuse or normalize embeddings twice.
- Include model-specific preprocessing, class-name templates, Hugging Face revision, maximum token length, and prefix in adapter metadata.
- If dependencies conflict, run the model in an isolated virtual environment or subprocess and aggregate the result JSON. Do not force-install all model packages into one environment by default.

### 8.2 CLI (planned)

The following commands describe the intended completed interface; they are not a claim that the commands currently work:

```powershell
uv run vlbench doctor --device cuda
uv run vlbench datasets prepare --dataset cifar10
uv run vlbench run --model openai_clip_b32 --dataset cifar10 --protocol p0 --device cuda --precision fp32
uv run vlbench run --model embeddinggemma2 --dataset xtd10 --language ja --protocol p0 --device cuda --precision fp32
uv run vlbench suite --name core-en --device cuda --precision fp32
uv run vlbench report --input results/v0.1.0/raw --output results/v0.1.0
```

- `doctor` diagnoses CUDA/PyTorch compatibility, disk capacity, Hugging Face authentication presence (without displaying secrets), and external dataset access requirements.
- `run` options include model, dataset, split, language, protocol, precision, device, batch size, seed, output, resume, and force. `--dry-run` prints the plan and estimated resource requirements.
- Before downloads, display the dataset/model, size, and license notice. Use the official source; a public download does not override any access or license restriction.
- On resume, compare run ID, model revision, dataset checksum, and configuration. Do not reuse a result when they differ.

## 9. Data storage and reporting

### 9.1 Authoritative data

At minimum, CLIP Benchmark-compatible JSON contains `dataset`, `model`, `pretrained`, `task`, `language`, and `metrics`. Preserve the standard metric keys `acc1`, `acc5`, `image_retrieval_recall@K`, and `text_retrieval_recall@K` whenever applicable.

Keep extra reproducibility data under `benchmark_metadata` or in a sidecar with the same `run_id` (choose and test the format for compatibility with aggregation tools). Include:

- `schema_version`, `run_id`, `timestamp_utc`, `status`, `reason_if_not_success`
- `source_commit_sha`, `framework_name`, `framework_ref`, `python_version`, `uv_lock_sha256`
- `model_repo`, `checkpoint_revision`, `pretrained_tag`, `adapter_name`, `model_license`
- `dataset_id`, `dataset_revision`, `dataset_split`, `dataset_checksum`, `image_count`, `text_count`, `class_count`
- `language`, `original_language_code`, `prompt_policy`, `prompt_name`, `template_hash`, `classnames_hash`
- `device`, `gpu_name`, `gpu_vram_bytes`, `driver_version`, `torch_version`, `torch_cuda_version`, `precision`, `amp_enabled`, `batch_size_actual`, `seed`
- `throughput_images_per_sec`, `throughput_texts_per_sec`, `inference_ms_median`, `gpu_peak_allocated_bytes`, `gpu_peak_reserved_bytes` (only when measured)

**Do not attach measured values to an unsuccessful run.** Distinguish JSON numbers, `null`, and meaningful status values. Include only validated successful runs in `summary.csv`; report failed and skipped runs separately.

### 9.2 Published artifacts

- `results/v0.1.0/raw/*.json`: CLIP Benchmark-compatible results with only metadata permitted for publication
- `results/v0.1.0/summary.csv`: one row per model × dataset × language × protocol × precision × run
- `results/v0.1.0/metadata.json`: overall environment, versions, dataset list, and checksums
- `results/v0.1.0/report.md`: conditions, English classification, English retrieval, Japanese retrieval/classification, limitations, and reproduction steps
- `results/v0.1.0/plots/*.png` and `*.svg`: visualizations reproducible from JSON with a script
- `README.md`: latest **measured** summary table and link to the full report. Do not add placeholder numbers before measurement.

Provide at least four chart types:

1. Zero-shot classification Top-1 by dataset, with a 0–100% axis.
2. Retrieval Recall@1/5/10 with T→I and I→T labeled.
3. English versus Japanese results for the same dataset and identical image-ID set.
4. Accuracy versus throughput, or FP32/BF16 peak VRAM, with separate axes and units clearly labeled.

Every figure must identify the dataset, split, protocol, dtype, and model. Show missing values as missing, not as zero.

### 9.3 Excluded from GitHub

Do not commit image/video contents, complete datasets, model weights, Hugging Face credentials, caches, uncompressed data, secrets, large embedding arrays, or redistribution copies of non-public datasets. Add these exclusions to `.gitignore`.

## 10. Tests, CI, and quality gates

- Use `pytest` for adapters, classification, retrieval, metadata, reports, and cache reuse. Normal CI must not require network access, a GPU, or full datasets.
- Use small synthetic features and known ground truth to test Top-1/Top-5, T→I/I→T R@1/5/10, multiple captions, ties, missing values, and NaN/Inf cases.
- Compare the project's JSON metrics with the official CLIP Benchmark implementation on the same CLIP model and data. Numerical parity is the **highest-priority Phase 1 acceptance condition**. Determine and record an appropriate tolerance from the official computation dtype and preprocessing.
- Require `ruff check .`, `uv lock --check`, and `git diff --check`.
- GitHub Actions must run unit tests, sample JSON→CSV/Markdown/chart generation, and schema validation on Windows/Linux and Python 3.12. Run actual model evaluation only locally or on an explicitly configured GPU runner.
- Check that CSV, Markdown, and charts regenerate from result JSON without unintended diffs. Exclude unstable values such as timestamps and absolute paths from deterministic comparisons.
- Never conflate passing automated tests with successful real-model accuracy evaluation.

## 11. Staged milestones and completion criteria

### Phase 0 — Foundation (required)

- Add the specification, `docs/methodology.md`, and model/dataset registries.
- Verify CUDA-enabled PyTorch on Windows PowerShell and report the result through `doctor`.
- Provide model-independent unit tests, a JSON schema, a report generator, and CI.

**Done**: Environment, metrics, and output format are fixed and CI passes.

### Phase 1 — CLIP Benchmark alignment (highest priority)

- Implement OpenAI CLIP ViT-B/32 and CIFAR-10, CIFAR-100, and Flickr30k.
- Run the official CLIP Benchmark implementation and this repository under identical model/conditions; explain or eliminate metric differences.
- Generate real JSON→CSV→Markdown→PNG/SVG results end to end.

**Done**: Numerical evidence confirms alignment with the benchmark standard; only measured results are published.

### Phase 2 — EmbeddingGemma and English comparison

- Add adapters for EmbeddingGemma 2, OpenCLIP ViT-B/32 and ViT-L/14, and SigLIP 2.
- Evaluate MS COCO 5K and ImageNet-1K in addition to the three Phase 1 datasets.
- Measure FP32 accuracy on the RTX 4070 Ti SUPER. Run BF16, speed, and memory as separate conditions.

**Done**: Execution availability or failure reasons for all candidates are explicit, and comparison tables and reproduction conditions are complete.

### Phase 3 — Japanese comparison

- Add rinna Japanese CLIP and multilingual OpenCLIP. Evaluate SigLIP 2 and EmbeddingGemma 2 in Japanese as well.
- Require XTD10 (`jp`) and XM3600 (`ja`); compare Japanese and English on identical image sets.
- Audit multilingual COCO overlap with XTD10 and establish Babel-ImageNet class coverage before adding it.

**Done**: Public results, prompts, label coverage, and overlap risks are documented for every language.

### Phase 4 — Extensions (optional)

Add Japanese Stable CLIP, ImageNet-A/R/Sketch, SugarCrepe, MIEB Lite, detailed VRAM/runtime measurements, and statistical confidence intervals. Do not delay the baseline release for these.

## 12. Delegation rules for Copilot and other contributors

- First inspect the existing repository, this specification, dependencies, and licenses. **Do not create another repository.**
- Use a dedicated `feature/*` branch and small pull requests. Implement Phase 0→1→2→3 in order and record intermediate results and evidence.
- Do not push directly to `main`, merge a pull request without approval, fabricate published results, or automatically accept unknown licenses. Obtain user authorization for downloads and long evaluations unless already explicitly authorized for the current task.
- A download confirmation flag never substitutes for accepting a legal agreement. If terms require acceptance, stop before access; otherwise follow the user's explicit authorization and preserve attribution.
- Always retain evaluation failures. Never hide unfavorable results or weaken thresholds for convenience.

## 13. Acceptance checklist

- [ ] Confirm IDs, tags, revisions, licenses, and loading methods for all adopted models.
- [ ] Pin the CLIP Benchmark reference commit/version.
- [ ] Verify metric equality against the standard implementation on CIFAR-10/100 and Flickr30k.
- [ ] Record dataset split, image count, ID correspondence, and prompts for all published results.
- [ ] Measure FP32 on the RTX 4070 Ti SUPER; state why if unused.
- [ ] Run BF16 separately and only for supported models.
- [ ] Measure Japanese XTD10 and Japanese/English XM3600.
- [ ] Investigate overlap between XTD10 and multilingual COCO; verify Babel label coverage.
- [ ] Regenerate JSON, CSV, Markdown, PNG, and SVG from the same measured JSON.
- [ ] Report accuracy, speed, and VRAM using distinct definitions.
- [ ] Record failed/skipped runs, licenses, and training-data overlap risks.
- [ ] Keep datasets, weights, and secrets out of GitHub.
- [ ] Pass Windows/Linux CI, lint, unit tests, and report regeneration checks.
- [ ] Link published results and reproduction commands from the README.

## 14. Unresolved items (verify and document during implementation)

1. Numerical parity with the pinned CLIP Benchmark commit and compatibility of its dependencies
   with Python 3.12.
2. Adopted OpenCLIP/SigLIP 2/rinna checkpoint revisions and any license or consent requirements.
3. Whether `openclip-gemma==0.3.0` can share a CUDA environment with other adapters.
4. Access, rights, official split, sample count, and checksums for Flickr30k, ImageNet, COCO, XM3600, XTD10, and Babel-ImageNet.
5. Dataset-specific download implementations. CLIP Benchmark invokes `wget`, `unzip`, and `rm` in some paths; verify Windows portability before use.
6. Numerical agreement between official Japanese CLIP and CLIP Benchmark adapters.
7. Score rounding, multi-template aggregation, and SigLIP 2 similarity extraction details.
8. Maximum acceptable runtime, storage, and any additional dataset acquisition approval.

**Never describe an unresolved item as supported or equivalent to a published result without verification.**

## 15. Primary references

1. CLIP Benchmark: https://github.com/LAION-AI/CLIP_benchmark
2. OpenCLIP: https://github.com/mlfoundations/open_clip
3. EmbeddingGemma 2 model card: https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2
4. `openclip-gemma`: https://github.com/kotadymo/openclip-gemma
5. rinna Japanese CLIP: https://huggingface.co/rinna/japanese-clip-vit-b-16
6. SigLIP 2 Base: https://huggingface.co/google/siglip2-base-patch16-224
7. XTD10: https://github.com/adobe-research/Cross-lingual-Test-Dataset-XTD10
8. Crossmodal-3600: https://google.github.io/crossmodal-3600/
9. Babel-ImageNet: https://github.com/gregor-ge/Babel-ImageNet
10. STAIR Captions: https://github.com/STAIR-Lab-CIT/STAIR-captions
11. Japanese Stable CLIP: https://huggingface.co/stabilityai/japanese-stable-clip-vit-l-16
12. uv / PyTorch CUDA: https://docs.astral.sh/uv/guides/integration/pytorch/

---

**Adoption policy**: Verify CLIP Benchmark compatibility in Phase 1 before publishing Phase 2 English comparisons and Phase 3 Japanese comparisons. Always disclose differences for scores that do not use the same conditions as the official CLIP Benchmark.
